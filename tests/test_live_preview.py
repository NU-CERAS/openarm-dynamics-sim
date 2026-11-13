import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from arm_design.analysis import atomic_json
from arm_design.config import DEFAULT_CONFIG, ROOT, load_config
from arm_design.dashboard import create_app
from arm_design.live_preview import LivePublisher, watched_pose
from arm_design.model import build_arm
from arm_design.trial_playback import model_geometry, visual_geoms
from test_dashboard_jobs import RIGHT, values, start_worker
from arm_design.dashboard_jobs import prepare, save_request, job_state


def test_preview_is_opt_in_throttled_and_expires(tmp_path,monkeypatch):
    import arm_design.live_preview as live
    clock=[100.]
    monkeypatch.setattr(live.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(live.time,'time',lambda:clock[0])
    d=SimpleNamespace(qpos=np.zeros(7),time=.5)
    p=LivePublisher(tmp_path);p.begin('campaign',1.)
    p.sample(d,'sweep')
    assert not (tmp_path/'live-pose.json').exists()
    watched_pose(tmp_path,True)
    clock[0]+=.21;p.sample(d,'sweep')
    first=json.loads((tmp_path/'live-pose.json').read_text())
    assert first['worker']==os.getpid() and first['q_rad']==[0]*7 and first['time_s']==.5
    d.qpos[:]=1
    for _ in range(1000):p.sample(d,'hold')
    assert json.loads((tmp_path/'live-pose.json').read_text())==first
    clock[0]+=.21;p.sample(d,'hold')
    second=json.loads((tmp_path/'live-pose.json').read_text());assert second['q_rad']==[1]*7
    clock[0]+=4;p.sample(d,'hold')
    assert json.loads((tmp_path/'live-pose.json').read_text())==second
    assert watched_pose(tmp_path,False)==second
    # Other pool workers cannot compete for this stream.
    owner=multiprocessing.get_context('spawn').Value('i',os.getpid()+1)
    other=LivePublisher(tmp_path,owner);assert not other.selected
    watched_pose(tmp_path,True);clock[0]+=.21;other.sample(d,'sweep')
    assert json.loads((tmp_path/'live-pose.json').read_text())==second
    d.qpos[0]=np.nan;p.sample(d,'sweep')
    assert json.loads((tmp_path/'live-pose.json').read_text())==second


@pytest.mark.parametrize('config',[DEFAULT_CONFIG,RIGHT])
def test_browser_kinematics_matches_mujoco_with_payload(tmp_path,config):
    c=load_config(config)
    c['analysis_payload']=dict(mass_kg=.5,com_offset_m=[0,.15,0],box_size_m=[.05]*3)
    arm=build_arm(c);m=arm.model;d=arm.data()
    binary=tmp_path/'model.mjb';mujoco.mj_saveModel(m,str(binary),None)
    model=model_geometry(str(binary),binary.stat().st_mtime_ns)
    q=[np.zeros(7).tolist(),[.2,-.4,.6,.8,-.3,.5,-.7],[-.7,.5,-.3,.2,.4,-.6,.8]]
    assets=ROOT/'src/arm_design/assets'
    script=tmp_path/'check.mjs'
    script.write_text('''import fs from 'node:fs';import vm from 'node:vm';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const THREE=await import(input.three);
globalThis.window={};globalThis.document={currentScript:{src:input.source},addEventListener:()=>{}};
vm.runInThisContext(fs.readFileSync(new URL(input.source),'utf8'));
const robot=window.armLiveKinematics.build(THREE,input.model);
console.log(JSON.stringify(input.q.map(q=>{robot.apply(q);return robot.objects.map(o=>[...o.matrixWorld.elements]);})));robot.dispose();
''')
    result=subprocess.run(['node',str(script)],input=json.dumps(dict(model=model,q=q,
        three=(assets/'vendor/three.module.js').as_uri(),source=(assets/'live_preview.js').as_uri())),
        text=True,capture_output=True,check=True)
    actual=np.array(json.loads(result.stdout))
    geoms=visual_geoms(m)
    for k,pose in enumerate(q):
        d.qpos[:]=pose;mujoco.mj_kinematics(m,d)
        expected=np.tile(np.eye(4),(len(geoms),1,1))
        expected[:,:3,:3]=d.geom_xmat[geoms].reshape(-1,3,3)
        expected[:,:3,3]=d.geom_xpos[geoms]
        # Three.js serializes matrices in column-major order.
        np.testing.assert_allclose(actual[k].reshape(-1,4,4).transpose(0,2,1),expected,atol=1e-10)


def test_preview_does_not_change_sweep_results(tmp_path):
    from arm_design.campaign import run_sweep
    cfg=load_config();cfg['sweep'].update(settle_s=.1,hold_s=.1)
    a,b=[0]*7,[10,5,0,10,0,0,0]
    reference=run_sweep(build_arm(cfg),a,b,.5)
    watched_pose(tmp_path,True)
    result=run_sweep(build_arm(cfg),a,b,.5,preview=LivePublisher(tmp_path))
    assert result['counted']==reference['counted']
    np.testing.assert_array_equal(result['maximum'],reference['maximum'])
    np.testing.assert_array_equal(result['minimum'],reference['minimum'])
    assert (tmp_path/'live-pose.json').exists()


@pytest.mark.parametrize('mode',['campaign','restrained'])
def test_live_endpoint_streams_a_real_worker_and_stops_cleanly(tmp_path,mode):
    job=tmp_path/'job-test';v=values()
    v.update(count=30,workers=2,duration=.2)
    save_request(job,prepare(RIGHT,mode,v))
    proc=start_worker(job)
    client=create_app(tmp_path).server.test_client()
    received=[]
    try:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and proc.poll() is None:
            state=job_state(job)
            if state.get('run'):
                response=client.post('/runner-live',json={'run':state['run']})
                assert response.status_code==200
                payload=response.get_json()
                if payload['pose']:
                    received.append(payload['pose'])
                    if len({p['published_at'] for p in received})>=2:break
            time.sleep(.1)
        assert received,(job/'worker.log').read_text()
        assert {p['scenario'] for p in received}=={mode}
        assert len({p['worker'] for p in received})==1
        assert all(len(p['q_rad'])==7 for p in received)
        from arm_design.dashboard_jobs import request_stop
        request_stop(job)
        assert proc.wait(timeout=30)==0,(job/'worker.log').read_text()
        before=json.loads((Path(state['run'])/'live-watch.json').read_text())
        response=client.post('/runner-live',json={'run':state['run']}).get_json()
        assert response['status']=='cancelled'
        assert json.loads((Path(state['run'])/'live-watch.json').read_text())==before
        assert client.post('/runner-live',json={'run':str(tmp_path.parent)}).status_code==404
    finally:
        if proc.poll() is None:proc.kill();proc.wait()

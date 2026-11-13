"""Physics and lifecycle checks for payload-specific dashboard runs."""
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from arm_design.config import DEFAULT_CONFIG, ROOT, load_config
from arm_design.dashboard import create_app
from arm_design.dashboard_jobs import prepare, save_request, job_state, request_stop
from arm_design.model import build_arm, box_inertia

RIGHT = ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml'


def values():
    return dict(count=9,workers=1,seed=17,mass=.05,offset=[0,0,0],size=[.02]*3,
        inertia='',name='Payload test',full=True,motion='aggressive',min_duration=.1,max_duration=2.,
        pose_count=1,poses=json.dumps([[0]*7]),duration=.05,ramp=.02)


@pytest.mark.parametrize('source', [DEFAULT_CONFIG, RIGHT])
def test_added_payload_mass_com_gravity_and_inertia(source):
    cfg = load_config(source)
    original = deepcopy(cfg)
    base = build_arm(cfg)
    loaded = deepcopy(cfg)
    p = dict(mass_kg=.5,com_offset_m=[.08,-.03,.02],box_size_m=[.02,.03,.04])
    loaded['analysis_payload'] = p
    arm = build_arm(loaded)
    m, d = arm.model, arm.data()
    bd = base.data()
    q = np.deg2rad([20,30,10,40,-20,30,10])
    d.qpos[:] = bd.qpos[:] = q
    mujoco.mj_forward(m,d)
    mujoco.mj_forward(base.model,bd)
    payload = m.body('analysis_payload')
    assert m.nv == base.model.nv == 7
    assert m.body_mass.sum()-base.model.body_mass.sum() == pytest.approx(.5)
    assert cfg == original
    R = d.site('tip').xmat.reshape(3,3)
    np.testing.assert_allclose(d.xipos[payload.id],d.site('tip').xpos+R@p['com_offset_m'],atol=1e-10)
    # Added gravity compensation is exactly -J_COM.T @ (mass * gravity).
    jac = np.zeros((3,m.nv))
    mujoco.mj_jacBodyCom(m,d,jac,None,payload.id)
    np.testing.assert_allclose(d.qfrc_bias-bd.qfrc_bias,-jac.T@(p['mass_kg']*m.opt.gravity),atol=1e-10)
    np.testing.assert_allclose(sorted(payload.inertia),sorted(np.diag(box_inertia(.5,p['box_size_m']))),atol=1e-12)
    loaded['analysis_payload']['mass_kg'] = 0
    zero = build_arm(loaded)
    assert zero.model.nbody == base.model.nbody
    np.testing.assert_array_equal(zero.model.body_mass,base.model.body_mass)


@pytest.mark.parametrize('change', [dict(mass=-1),dict(mass=float('nan')),dict(count=1.5),
    dict(count=float('inf')),dict(offset=[0,float('nan'),0]),dict(size=[0,.1,.1]),
    dict(inertia='[1,1,3,0,0,0]'),dict(inertia='[1,1,-1,0,0,0]'),dict(inertia='bad json')])
def test_invalid_requests_rejected_before_launch(change):
    v = values(); v.update(change)
    with pytest.raises((ValueError,TypeError)):
        prepare(RIGHT,'campaign',v)


def test_custom_inertia_and_independent_configuration(tmp_path):
    original = RIGHT.read_bytes()
    v = values();v.update(inertia='[0.01,0.012,0.015,0.001,0,0]',count=10000)
    prepared = prepare(RIGHT,'restrained',v)
    assert prepared['planned_trials'] == 2187
    arm = build_arm(prepared['config'])
    np.testing.assert_allclose(sorted(arm.model.body('analysis_payload').inertia),
        np.linalg.eigvalsh([[.01,.001,0],[.001,.012,0],[0,0,.015]]),atol=1e-10)
    job = tmp_path/'job-test'
    save_request(job,prepared)
    assert RIGHT.read_bytes() == original
    saved = load_config(job/'config.yaml')
    assert saved['analysis_payload'] == prepared['payload']
    assert Path(saved['urdf']['path']).is_absolute()
    assert '--poses-file' in json.loads((job/'request.json').read_text())['args']


def start_worker(job):
    with (job/'worker.log').open('w') as log:
        return subprocess.Popen([sys.executable,'-m','arm_design.dashboard_jobs',str(job)],stdout=log,stderr=subprocess.STDOUT)


@pytest.mark.parametrize('mode,count', [('campaign',9),('restrained',2)])
def test_real_worker_exact_limit_saved_payload_and_live_selection(tmp_path,mode,count):
    v = values();v['count']=count
    job = tmp_path/'job-test'
    save_request(job,prepare(RIGHT,mode,v))
    proc = start_worker(job)
    try:
        assert proc.wait(timeout=60) == 0,(job/'worker.log').read_text()
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait()
    state = job_state(job)
    assert state['status'] == 'finished',(job/'worker.log').read_text()
    run = Path(state['run'])
    info = json.loads((run/'run.json').read_text())
    assert info['trials'] == count
    assert info['settings']['analysis_payload']['mass_kg'] == .05
    assert info['settings']['run_name'] == 'Payload test'
    assert (run/'trials/0/trace.npz').exists()
    m = mujoco.MjModel.from_binary_path(str(run/'model.mjb'))
    assert m.body('analysis_payload').mass == pytest.approx(.05)
    app = create_app(tmp_path)
    callback = lambda prefix: next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if k.startswith(prefix))
    options, selected, followed = callback('..run.options')(0,mode,{mode:str(job)},None,[],[])
    assert selected == str(run) and followed == [str(job)]
    assert '+0.05 kg' in options[0]['label']
    progress = callback('..run-progress.value')(0,str(run))
    assert progress[:2] == (count,count)
    assert callback('..launch.disabled')(0,{mode:str(job)},mode,str(run))[:2] == (False,True)
    if mode == 'campaign':
        validation = json.loads((run/'validation-summary.json').read_text())
        assert validation['timestep_validation']['reason'] == 'Not requested'
    else:
        from arm_design.fea_cases import part_cases
        cases = part_cases(Path(info['export']),'base')
        assert cases
        assert max(c['residual_n'] for c in cases) < 1e-8
        assert max(c['moment_residual_nm'] for c in cases) < 1e-8


def test_stop_before_worker_creates_run(tmp_path):
    job = tmp_path/'job-stopped'
    save_request(job,prepare(RIGHT,'campaign',values()))
    request_stop(job)
    proc = start_worker(job)
    assert proc.wait(timeout=15) == 0
    assert job_state(job)['status'] == 'cancelled'
    assert 'run' not in job_state(job)


def test_dashboard_launch_isolated_modes_duplicate_guard_and_recovery(tmp_path,monkeypatch):
    from dash import no_update
    import arm_design.dashboard as dashboard
    spawned = []
    class Process:
        def __init__(self,*args,**kwargs):
            spawned.append(args[0])
        def poll(self):
            return None
    monkeypatch.setattr(dashboard.subprocess,'Popen',Process)
    app = create_app(tmp_path)
    def callback(app,prefix):
        return next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if k.startswith(prefix))
    def launch(mode,jobs):
        return callback(app,'..launch-status.children')(1,mode,str(RIGHT),9,2,17,1,'aggressive',.1,2,
            .05,.02,1,json.dumps([[0]*7]),['full'],'Test',.05,0,0,0,.02,.02,.02,'',jobs)
    _, jobs = launch('campaign',{})
    first = jobs['campaign']
    message, unchanged = launch('campaign',jobs)
    assert 'already' in message and unchanged is no_update and len(spawned)==1
    _, jobs = launch('restrained',jobs)
    assert jobs['campaign'] == first and jobs['restrained'] != first and len(spawned)==2
    assert json.loads((Path(first)/'request.json').read_text())['planned_trials']==9
    assert json.loads((Path(jobs['restrained'])/'request.json').read_text())['planned_trials']==2
    # A browser session or server restart must not lose an active worker's controls.
    reopened = create_app(tmp_path)
    assert callback(reopened,'..launch.disabled')(0,{},'campaign',None)[:2] == (True,False)
    message = callback(reopened,'cancel-status.children')(1,None,{},'campaign')
    assert 'Stop requested' in message and (Path(first)/'CANCEL').exists()
    assert not (Path(jobs['restrained'])/'CANCEL').exists()


def test_stop_live_worker_retains_completed_trials(tmp_path):
    job = tmp_path/'job-stop-live'
    v = values();v.update(count=100,motion='random',min_duration=4,max_duration=5)
    save_request(job,prepare(RIGHT,'campaign',v))
    proc = start_worker(job)
    try:
        deadline = time.monotonic()+30
        while time.monotonic()<deadline and proc.poll() is None:
            state = job_state(job)
            run = Path(state['run']) if state.get('run') else None
            if run and json.loads((run/'run.json').read_text())['trials'] >= 1:
                break
            time.sleep(.05)
        else:
            pytest.fail('Worker never published trials: '+(job/'worker.log').read_text())
        request_stop(job)
        assert proc.wait(timeout=30) == 0,(job/'worker.log').read_text()
        assert job_state(job)['status'] == 'cancelled'
        info = json.loads((run/'run.json').read_text())
        assert info['status'] == 'cancelled' and 1 <= info['trials'] < 100
        assert (run/'trials/0/trace.npz').exists()
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait()

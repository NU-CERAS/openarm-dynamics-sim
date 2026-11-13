import base64
import gzip
import json

import mujoco
import numpy as np
import pytest
from dash import no_update

from arm_design.dashboard import create_app
from arm_design.trial_playback import playback, model_geometry


@pytest.fixture
def recording(tmp_path):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0  .1 0 0  0 .1 0  0 0 .1"/></asset>
      <worldbody><body name="link" pos="1 2 3"><joint type="hinge" axis="0 0 1"/>
        <geom type="mesh" mesh="tetra" group="1" pos=".2 0 0"/>
      </body></worldbody></mujoco>''')
    mujoco.mj_saveModel(model, str(tmp_path/'model.mjb'), None)
    meta = dict(parts=[dict(name='link', bodies=['link'])], interfaces=[
        dict(kind='mate', part='link', label='link <- mount'),
        dict(kind='mate', part='link', label='link <- tool')])
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    (tmp_path/'run.json').write_text(json.dumps(dict(interfaces=meta['interfaces'])))
    folder = tmp_path/'trials'/'42'
    folder.mkdir(parents=True)
    loads = np.zeros((3, 2, 6)); loads[:,1,0] = [1,-2,3]
    np.savez(folder/'trace.npz', time=[0,1,2], q=[[0],[.4],[.8]], loads=loads)
    (folder/'trial.json').write_text(json.dumps(dict(counted=False, reason='tracking error', full_history=False)))
    peak_loads = np.zeros((2,6)); peak_loads[1] = [-20,3,0,4,0,-5]
    case = dict(time_s=.5, q_rad=[.2], loads_local=peak_loads.tolist())
    with gzip.open(folder/'cases.json.gz', 'wt') as stream:
        json.dump([case], stream)
    return tmp_path, model, peak_loads


def test_playback_preserves_exact_peak_pose_and_loads(recording):
    path, model, peak_loads = recording
    replay = playback(str(path),42,'link',1,'Fx_N','minimum')
    peak = replay['peak_index']
    assert replay['time'][peak] == .5
    assert replay['time'][0] == 0 and replay['time'][-1] == 2
    assert not replay['full_history'] and not replay['accepted']
    assert replay['reason'] == 'tracking error'
    np.testing.assert_array_equal(replay['loads'][peak], peak_loads)
    # Forward-kinematic transform must match the stored model's mesh centering.
    data = mujoco.MjData(model); data.qpos[:] = .2
    mujoco.mj_kinematics(model,data)
    transform = np.array(replay['poses'][peak][0])
    np.testing.assert_allclose(transform[:3], data.geom_xpos[0])
    np.testing.assert_allclose(transform[3:], data.geom_xmat[0])
    geometry = model_geometry(str(path/'model.mjb'), (path/'model.mjb').stat().st_mtime_ns)
    vertices = np.frombuffer(base64.b64decode(geometry['meshes']['0']['vertices']), dtype='<f4').reshape(-1,3)
    np.testing.assert_allclose(vertices, model.mesh_vert)
    assert geometry['geoms'][0]['body'] == 'link'
    # Positive selection should choose +3, not the larger negative magnitude.
    maximum = playback(str(path),42,'link',1,'Fx_N','maximum')
    assert maximum['time'][maximum['peak_index']] == 2
    norm = playback(str(path),42,'link',1,'force_N','minimum')
    assert norm['time'][norm['peak_index']] == .5
    assert norm['peak_label'].startswith('maximum')


def test_playback_missing_history_and_invalid_connection(recording):
    path, _, _ = recording
    with pytest.raises(ValueError, match='no saved model or motion history'):
        playback(str(path),999,'link',1,'force_N','absolute')
    with pytest.raises(ValueError, match='selected link'):
        playback(str(path),42,'different-link',1,'force_N','absolute')


def test_dashboard_selection_uses_trial_id_and_survives_live_refresh(recording):
    path, _, _ = recording
    app = create_app(path)
    callback = next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if k.startswith('..replay-selection.data'))
    args = ([0],str(path),'link',1,'Fx_N','minimum',[dict(id=42)])
    selection, disabled = callback(*args,None)
    assert selection['trial'] == 42 and not disabled
    assert callback(*args,selection)[0] is no_update
    assert callback([],str(path),'link',1,'Fx_N','minimum',[],selection) == (None,True)
    client = app.server.test_client()
    assert client.get('/replay-model',query_string={'run':str(path)}).status_code == 200
    assert client.get('/replay-model',query_string={'run':str(path.parent)}).status_code == 404
    assert client.get('/assets/trial_playback.js').status_code == 200
    page = client.get('/').text
    assert 'src="/assets/vendor/' not in page  # ES modules are imported only on play.

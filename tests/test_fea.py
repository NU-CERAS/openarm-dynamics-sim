import csv
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from arm_design.config import load_config, DEFAULT_CONFIG, ROOT
from arm_design.experiment import Experiment
from arm_design.fea import FEARecording
from arm_design.model import build_arm
from arm_design.reporting import summary, save_result
from arm_design.simulation import step, Sweep


def fixed_cantilever():
    # Rotated part with a COM offset: known force AND bending moment at mount.
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="part" pos=".3 .4 1" euler="0 0 90">
        <inertial pos=".2 0 0" mass="2" diaginertia=".01 .02 .02"/>
      </body></worldbody></mujoco>''')
    cfg = dict(joints=[], simulation=dict(contacts=False))
    return SimpleNamespace(model=m, cfg=cfg, xml='<mujoco/>')


def capture_direct(arm, data):
    mujoco.mj_forward(arm.model, data)
    rec = FEARecording(arm, data)
    row = dict(q=data.qpos.copy(), velocity=data.qvel.copy(), target=data.qpos.copy(),
               torque=data.qfrc_actuator.copy(), cap=np.ones(arm.model.nv))
    # Recorder normally works on rotary arms; this fixture is purely welded.
    rec.capture(arm.model, data, row, 'analytical')
    return rec


def assert_body_balance(rec, sample):
    m = rec.model
    for b in rec.body_ids:
        R = sample['body_rotation_world'][b-1]
        origin, com = sample['body_origin_world_m'][b-1], sample['com_world_m'][b-1]
        force, moment = np.zeros(3), np.zeros(3)
        for i, interface in enumerate(rec.interfaces):
            if interface['body_id'] != b:
                continue
            f = R@sample['loads_local'][i,:3]
            p = origin + R@sample['application_points_local_m'][i]
            force += f
            moment += R@sample['loads_local'][i,3:] + np.cross(p-com, f)
        external = sample['external_resultant_at_com_local'][b-1]
        force += R@external[:3] + m.body_mass[b]*m.opt.gravity
        moment += R@external[3:]
        velocity, acceleration = sample['com_velocity_world'][b-1], sample['com_acceleration_world'][b-1]
        inertial_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(inertial_rotation, m.body_iquat[b])
        Ri = R@inertial_rotation.reshape(3,3)
        inertia = Ri@np.diag(m.body_inertia[b])@Ri.T
        np.testing.assert_allclose(force, m.body_mass[b]*acceleration[3:], atol=2e-8)
        np.testing.assert_allclose(moment, inertia@acceleration[:3] +
                                   np.cross(velocity[:3], inertia@velocity[:3]), atol=2e-8)


def test_static_weight_bending_and_frame_transform():
    arm = fixed_cantilever()
    d = mujoco.MjData(arm.model)
    rec = capture_direct(arm, d)
    np.testing.assert_allclose(rec.rows[0]['loads_local'][0],
                               [0,0,19.62,0,-3.924,0], atol=1e-10)
    np.testing.assert_allclose(rec.rows[0]['com_acceleration_world'], 0, atol=1e-10)
    assert_body_balance(rec, rec.rows[0])


@pytest.mark.parametrize('imported', [False, True])
def test_dynamic_balance_every_body_and_interface(imported):
    path = ROOT/'robots/moving_asm/urdf/moving_asm.sim.yaml' if imported else DEFAULT_CONFIG
    arm = build_arm(load_config(path))
    d = arm.data()
    d.qvel[:] = np.linspace(-.8, 1.1, arm.model.nv)
    rec = FEARecording(arm, d)
    for _ in range(4):
        step(arm, d, d.qpos+.1, recorder=rec)
    for sample in rec.rows:
        assert_body_balance(rec, sample)
    assert rec.rows[0]['simulation_time_s'] == 0
    assert rec.rows[-1]['simulation_time_s'] < d.time
    assert np.linalg.norm(rec.rows[0]['com_acceleration_world']) > 1
    if imported:
        assert all(f'part_{i}' in rec.names.values() for i in range(1,8))
        # Fixed motor mounts must not disappear from the structural handoff.
        assert any(x['label'] == 'part_1 <- rs03output_1' for x in rec.interfaces)


def test_external_resultant_included_once():
    arm = build_arm(load_config())
    d = arm.data()
    body = arm.model.jnt_bodyid[-1]
    d.xfrc_applied[body] = [7, -4, 2, .3, -.5, .2]
    rec = FEARecording(arm, d)
    step(arm, d, d.qpos.copy(), recorder=rec)
    assert_body_balance(rec, rec.rows[0])
    R = rec.rows[0]['body_rotation_world'][body-1]
    external = rec.rows[0]['external_resultant_at_com_local'][body-1]
    np.testing.assert_allclose(R@external[:3], [7,-4,2], atol=1e-10)
    np.testing.assert_allclose(R@external[3:], [.3,-.5,.2], atol=1e-10)


def test_recording_lifecycle_exports_and_snapshot(tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(load_config()))
    e = Experiment(path, tmp_path/'results')
    e.set_gains(0, 145, 14)
    e.start_fea()
    with pytest.raises(ValueError, match='controller gains'):
        e.set_gains(0, 120, 12)
    e.set_target(0, 5)
    for _ in range(10):
        e.tick()
    e.paused = True
    for _ in range(5):
        e.tick()
    assert len(e.fea.rows) == 10  # physics steps, independent of display/pause
    e.stop_fea()
    rec = e.fea
    q = e.data.qpos.copy()
    e.export_fea()
    folder = rec.last_export
    np.testing.assert_array_equal(e.data.qpos, q)
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['samples'] == 10
    with (folder/'interface_loads.csv').open() as f:
        assert len(list(csv.DictReader(f))) == len(rec.interfaces)*10
    history = [json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
    cases = json.loads((folder/'load_cases.json').read_text())['samples']
    for case in cases:
        assert case == history[case['sample_index']]  # simultaneous, uncombined data
    model = mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    assert model.actuator_gainprm[0,0] == 145
    rec.selected_samples.add(3)
    folder = rec.export(tmp_path/'selected', 'cases')
    assert [s['sample_index'] for s in json.loads((folder/'load_cases.json').read_text())['samples']] == [3]
    e.start_fea()
    e.paused = False
    e.tick()
    e.load()
    assert not e.fea.active and 'reloaded' in e.fea.reason
    assert len(e.fea.rows) == 1


def test_capacity_stops_without_discarding_and_zero_sample_export(tmp_path):
    arm = build_arm(load_config())
    d = arm.data()
    rec = FEARecording(arm, d)
    rec.stop()
    with pytest.raises(ValueError, match='No FEA samples'):
        rec.export(tmp_path)
    rec.active = True
    rec.max_samples = 2
    for _ in range(4):
        step(arm, d, d.qpos.copy(), recorder=rec)
    assert len(rec.rows) == 2 and not rec.active
    assert 'capacity' in rec.reason


def test_reverse_sweep_uses_saved_b_then_a_and_reports_direction(tmp_path):
    cfg = load_config()
    cfg['sweep'].update(a_deg=[0,-25,0,50,0,0,0], b_deg=[15,-40,10,60,0,0,0],
                        approach_s=1, settle_s=.5, duration_s=1, hold_s=.5)
    arm = build_arm(cfg)
    data = arm.data()
    rec = FEARecording(arm, data)
    run = Sweep(arm, data, 'B -> A')
    assert run.cfg['sweep']['a_deg'] == cfg['sweep']['a_deg']
    while run.tick(arm, data, recorder=rec):
        pass
    assert run.complete and summary(run)['tracking_ok']
    sweep_rows = [r for r in run.rows if r['phase'] == 'sweep']
    np.testing.assert_allclose(sweep_rows[0]['target'], np.deg2rad(cfg['sweep']['b_deg']))
    np.testing.assert_allclose(run.rows[-1]['target'], np.deg2rad(cfg['sweep']['a_deg']))
    assert len(rec.rows) == len(run.rows)
    assert any(r['phase'] == 'B -> A: sweep' for r in rec.rows)
    folder = save_result(arm, run, tmp_path)
    assert json.loads((folder/'summary.json').read_text())['direction'] == 'B -> A'
    assert 'B -&gt; A' in (folder/'report.html').read_text()

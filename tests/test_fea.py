import csv
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from arm_design.bounds import Statics
from arm_design.config import load_config, validate, DEFAULT_CONFIG, ROOT
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
    """Newton-Euler on every reported part from its individual connection loads."""
    m = rec.model
    loads, points = sample['loads_local'], sample['application_points_local_m']
    for part in rec.parts():
        root = part['body_id']
        ids = [b for b, r in rec.part_of.items() if r == root]
        R, origin = sample['body_rotation_world'][root-1], sample['body_origin_world_m'][root-1]
        mass = m.body_mass[ids]
        com = mass@sample['com_world_m'][[b-1 for b in ids]]/mass.sum() if mass.sum() else origin
        rows = [(i, x) for i, x in enumerate(rec.interfaces) if x['body_id'] == root]
        force, moment, rate, spin = (np.zeros(3) for _ in range(4))
        for i, x in rows:
            assert x['kind'] == 'mate'
            f = R@loads[i,:3]
            p = origin + R@points[i]
            force += f
            moment += R@loads[i,3:] + np.cross(p-com, f)
        for b in ids:
            Rb, c = sample['body_rotation_world'][b-1], sample['com_world_m'][b-1]
            external = sample['external_resultant_at_com_local'][b-1]
            f = Rb@external[:3] + m.body_mass[b]*m.opt.gravity
            force += f
            moment += Rb@external[3:] + np.cross(c-com, f)
            velocity, acceleration = sample['com_velocity_world'][b-1], sample['com_acceleration_world'][b-1]
            inertial_rotation = np.zeros(9)
            mujoco.mju_quat2Mat(inertial_rotation, m.body_iquat[b])
            Ri = Rb@inertial_rotation.reshape(3,3)
            inertia = Ri@np.diag(m.body_inertia[b])@Ri.T
            rate += m.body_mass[b]*acceleration[3:]
            spin += (inertia@acceleration[:3] + np.cross(velocity[:3], inertia@velocity[:3]) +
                     np.cross(c-com, m.body_mass[b]*acceleration[3:]))
        np.testing.assert_allclose(force, rate, atol=2e-8)
        np.testing.assert_allclose(moment, spin, atol=2e-8)


def test_static_weight_bending_and_frame_transform():
    arm = fixed_cantilever()
    d = mujoco.MjData(arm.model)
    rec = capture_direct(arm, d)
    np.testing.assert_allclose(rec.rows[0]['loads_local'][0],
                               [0,0,19.62,0,-3.924,0], atol=1e-10)
    np.testing.assert_allclose(rec.rows[0]['com_acceleration_world'], 0, atol=1e-10)
    assert_body_balance(rec, rec.rows[0])


RIGHT = ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml'
@pytest.mark.parametrize('path', [DEFAULT_CONFIG, ROOT/'robots/moving_asm/urdf/moving_asm.sim.yaml', RIGHT])
def test_dynamic_balance_every_part(path):
    imported = 'moving_asm' in str(path)
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


def recording(cfg):
    arm = build_arm(cfg)
    return FEARecording(arm, arm.data())


def test_actuators_get_no_rows_and_jointless_structure_merges():
    rec = recording(load_config(RIGHT))
    labels = [x['label'] for x in rec.interfaces]
    assert not any(x['part'].startswith('rs') for x in rec.interfaces) and len(set(labels)) == len(labels)
    parts = {p['name']: p for p in rec.parts()}
    assert list(parts) == ['body_1', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'base']
    # A link keeps both actuator mates; the tool is one solid with one mate.
    assert [l for l in labels if l.startswith('l2 ')] == ['l2 <- rs03output', 'l2 <- rs06_1']
    assert all(x['kind'] == 'mate' for x in rec.interfaces)
    assert [l for l in labels if l.startswith(('base', 'screwon', 'drumstick'))] == ['base <- rs00_2']
    tool, m = parts['base'], rec.model
    assert tool['bodies'] == ['base', 'screwon', 'drumstick']
    assert tool['mass_kg'] == pytest.approx(sum(m.body(n).mass[0] for n in tool['bodies']))
    assert tool['mass_kg'] == pytest.approx(.259*.45359237, rel=2e-3)
    inertia = np.array(tool['inertia_com_local_kg_m2'])
    np.testing.assert_allclose(inertia, inertia.T, atol=1e-15)
    assert np.trace(inertia) > sum(m.body(n).inertia.sum() for n in tool['bodies'])
    assert json.loads(json.dumps(rec.metadata()))['excluded_bodies'] == [n for n in rec.names.values() if n.startswith('rs')]
    # Without the key every body is a part, so both views of a link-to-link mate are reported.
    plain = [x['label'] for x in recording(load_config()).interfaces]
    assert {'link2 <- link1', 'link1 <- link2', 'link7 <- link6'} <= set(plain)
    assert len(plain) == 15  # Base mount and both sides of each of seven joints.
    for bad in ('rs*', [''], [3]):
        with pytest.raises(ValueError, match='exclude_bodies'):
            validate(load_config() | dict(fea=dict(exclude_bodies=bad)))
    with pytest.raises(ValueError, match='no structural part'):
        recording(load_config() | dict(fea=dict(exclude_bodies=['*'])))


def test_static_connections_balance_the_part_weight():
    s = Statics(build_arm(load_config(RIGHT)))
    assert s.evaluate(np.deg2rad([20, 40, -30, 60, 10, -40, 25])) is not None
    loads, points = s.rec.interface_loads(s.m, s.d)
    R, g = s.d.xmat.reshape(-1, 3, 3), s.m.opt.gravity
    for part in s.rec.parts():
        root = part['body_id']
        ids = [b for b, r in s.rec.part_of.items() if r == root]
        weight = part['mass_kg']*g
        rows = [i for i, x in enumerate(s.rec.interfaces) if x['body_id'] == root]
        assert all(s.rec.interfaces[i]['kind'] == 'mate' for i in rows)
        com = s.m.body_mass[ids]@s.d.xipos[ids]/part['mass_kg'] if part['mass_kg'] else s.d.xpos[root]
        np.testing.assert_allclose(R[root].T@(com-s.d.xpos[root]), part['com_local_m'], atol=1e-12)
        force, moment = np.zeros(3), np.zeros(3)
        for i in rows:
            f = R[root]@loads[i, :3]
            point = s.d.xpos[root] + R[root]@points[i]
            force += f
            moment += R[root]@loads[i, 3:] + np.cross(point-com, f)
        np.testing.assert_allclose(force, -weight, atol=1e-9)
        np.testing.assert_allclose(moment, 0, atol=1e-9)


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

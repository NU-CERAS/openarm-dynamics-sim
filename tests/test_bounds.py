import csv
import json

import mujoco
import numpy as np
import pytest

from arm_design.bounds import Statics, bounds
from arm_design.config import ROOT, load_config
from arm_design.fea import METRICS
from arm_design.model import build_arm

CAD = ROOT/'robots/moving_asm/urdf/moving_asm.sim.yaml'
RIGHT = ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml'


@pytest.mark.parametrize('path', [ROOT/'configs/arm.yaml', CAD])
def test_gravity_and_blocked_statics_balance(path):
    s = Statics(build_arm(load_config(path)), contacts='all')
    q = np.deg2rad(load_config(path)['sweep']['b_deg'])
    # Everything above the world mount hangs on it.
    assert s.evaluate(q)[0, METRICS.index('force_N')] == pytest.approx(s.m.body_mass.sum()*9.81)
    g = s.pose(q)
    for case in s.cases:
        hit = s.blocking_force(g, *case)
        if hit is None:
            continue
        torque = g - s.jacp.T@hit[2]
        # Statics: motors plus obstacle hold gravity; the first motor to saturate sets the force.
        assert np.all(np.abs(torque) <= s.cap+1e-9)
        assert np.max(np.abs(torque)/s.cap) == pytest.approx(1)


def test_contact_force_cap_and_collision_rejection():
    s = Statics(build_arm(load_config(CAD)), max_force_n=50)
    g = s.pose(np.zeros(7))
    assert all(np.linalg.norm(h[2]) <= 50+1e-9 for h in (s.blocking_force(g, *c) for c in s.cases) if h)
    assert s.pose(np.deg2rad([0, 0, 0, 0, 90, -90, -56])) is None
    with pytest.raises(ValueError):
        Statics(build_arm(load_config(CAD)), contacts='elbow')


@pytest.mark.parametrize('path', [CAD, RIGHT])
def test_clamped_statics_respect_caps_and_balance(path):
    s = Statics(build_arm(load_config(path)), max_force_n=100, max_moment_nm=10)
    rng = np.random.default_rng(3)
    checked = 0
    while checked < 40:
        g = s.pose(rng.uniform(s.limits[:, 0], s.limits[:, 1]))
        if g is None:
            continue
        # Random directions, both on the cap spheres and inside the balls.
        p = rng.normal(size=6)*rng.uniform(.05, 2)
        body, point, force, moment = s.clamping_load(g, p)
        torque = g - s.jacp.T@force - s.jacr.T@moment
        assert np.all(np.abs(torque) <= s.cap+1e-9)
        assert np.linalg.norm(force) <= 100+1e-9 and np.linalg.norm(moment) <= 10+1e-9
        # The load lies along p, as large as the first active limit allows.
        pf, pm = p[:3]/max(np.linalg.norm(p[:3]), 1), p[3:]/max(np.linalg.norm(p[3:]), 1)
        np.testing.assert_allclose(np.cross(force, pf), 0, atol=1e-9)
        np.testing.assert_allclose(np.cross(moment, pm), 0, atol=1e-9)
        assert max(np.max(np.abs(torque)/s.cap), np.linalg.norm(force)/100, np.linalg.norm(moment)/10) == pytest.approx(1)
        # The COM wrench given to RNE is the tip load moved there: same joint torques.
        # The world mount reacts the weight plus the clamp force.
        assert s.loads(body, point, force, moment)[0, METRICS.index('force_N')] == pytest.approx(
            np.linalg.norm(force + s.m.body_mass.sum()*s.m.opt.gravity))
        tip, com = np.zeros(s.m.nv), np.zeros(s.m.nv)
        mujoco.mj_applyFT(s.m, s.d, force, moment, point, body, tip)
        mujoco.mj_applyFT(s.m, s.d, s.d.xfrc_applied[body, :3], s.d.xfrc_applied[body, 3:], s.d.xipos[body], body, com)
        np.testing.assert_allclose(g - tip, torque, atol=1e-9)
        np.testing.assert_allclose(com, tip, atol=1e-9)
        checked += 1
    assert s.clamping_load(g, np.zeros(6)) is None
    with pytest.raises(ValueError):
        Statics(build_arm(load_config(CAD)), max_moment_nm=10)
    with pytest.raises(ValueError):
        Statics(build_arm(load_config(CAD)), max_force_n=100, max_moment_nm=0)


def test_bounds_export_reproduces_every_peak(tmp_path):
    cfg = load_config()
    folder, recs = bounds(cfg, tmp_path, poses=40, iterations=5, workers=2, max_force_n=80,
                          log=lambda *a, **k: None)
    weight = build_arm(cfg).model.body_mass.sum()*9.81
    for kind in ('gravity', 'blocked'):
        cases = {c['sample_index']: c for c in json.loads((folder/kind/'load_cases.json').read_text())['samples']}
        labels = [f['label'] for f in recs[kind].interfaces]
        for peak in csv.DictReader((folder/kind/'peaks.csv').open()):
            i, j = labels.index(peak['interface']), METRICS.index(peak['metric'])
            loads = np.array(cases[int(peak['max_sample'])]['loads_local'][i])
            value = np.concatenate((loads, [np.linalg.norm(loads[:3]), np.linalg.norm(loads[3:])]))[j]
            assert value == pytest.approx(float(peak['maximum']), rel=1e-12, abs=1e-12)
    force = METRICS.index('force_N')
    assert recs['gravity'].maximum[0, force] == pytest.approx(weight)
    assert recs['blocked'].maximum[0, force] <= weight + 80 + 1e-6
    assert all(np.linalg.norm(c['contact_force_on_arm_world_N']) > 0
               for c in json.loads((folder/'blocked'/'load_cases.json').read_text())['samples'])
    rows = list(csv.DictReader((folder/'summary.csv').open()))
    assert len(rows) == len(recs['gravity'].interfaces)
    assert json.loads((folder/'bounds.json').read_text())['max_contact_force_n'] == 80
    with pytest.raises(ValueError):
        bounds(cfg, tmp_path, max_force_n=0)


def test_fea_setup_balances_and_names_fixed_and_applied_faces(tmp_path):
    from arm_design.fea_cases import exports, part_cases, write_csv
    folder, _ = bounds(load_config(CAD), tmp_path, poses=30, iterations=3, workers=2, max_force_n=100,
                       log=lambda *a, **k: None)
    assert [f.name for f in exports(folder)] == ['blocked', 'gravity']
    for part, fixed, applied in (('part_3', 'rs06_1', ['rs06']), ('part_7', 'rs00_1', []), ('part_3', 'rs06', ['rs06_1'])):
        cases = part_cases(folder/'blocked', part, None if fixed != 'rs06' else 'rs06')
        assert cases
        for c in cases:
            assert c['fixed'] == fixed and [r['other'] for r in c['applied']] == applied
            assert c['residual_n'] < 1e-9
        # The tip contact acts on part_7 itself, so it becomes an applied load there.
        assert (part == 'part_7') == all(c['contact'] is not None for c in cases)
    write_csv(tmp_path/'setup.csv', cases)
    assert len(list(csv.DictReader((tmp_path/'setup.csv').open()))) == 2*len(cases)
    with pytest.raises(ValueError, match='cannot fix'):
        part_cases(folder/'blocked', 'part_3', 'world')
    # Actuators are not parts, and never receive load rows.
    with pytest.raises(ValueError, match='No structural part'):
        part_cases(folder/'blocked', 'rs06')
    assert not any(row['interface'].startswith('rs') for row in csv.DictReader((folder/'summary.csv').open()))


def test_fea_setup_treats_the_merged_tool_as_one_part(tmp_path):
    from arm_design.fea_cases import part_cases, report
    folder, recs = bounds(load_config(RIGHT), tmp_path, poses=30, iterations=3, workers=2, max_force_n=100,
                          log=lambda *a, **k: None)
    mass = sum(recs['blocked'].model.body(n).mass[0] for n in ('base', 'screwon', 'drumstick'))
    for kind in ('gravity', 'blocked'):
        cases = part_cases(folder/kind, 'base')
        assert cases
        for c in cases:
            assert c['fixed'] == 'rs00_2' and c['applied'] == [] and 'delta' not in c
            assert c['bodies'] == ['base', 'screwon', 'drumstick'] and c['mass_kg'] == pytest.approx(mass)
            # The tip is on the drumstick: its contact loads the merged part.
            assert c['residual_n'] < 1e-9 and (c['contact'] is not None) == (kind == 'blocked')
        assert 'One solid part: base + screwon + drumstick' in report(folder/kind, 'base', cases)
    link = part_cases(folder/'blocked', 'l3')
    assert all('delta' not in c and c['residual_n'] < 1e-9 for c in link)
    assert 'DELTA' not in report(folder/'blocked', 'l3', link)
    # A legacy aggregate peak must not select an extra FEA case.
    metadata_path = folder/'blocked'/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    body_id = next(p['body_id'] for p in metadata['parts'] if p['name'] == 'l3')
    metadata['interfaces'].append(dict(body_id=body_id, kind='delta', label='l3 delta'))
    metadata_path.write_text(json.dumps(metadata))
    peaks_path = folder/'blocked'/'peaks.csv'
    with peaks_path.open() as f:
        reader = csv.DictReader(f)
        fields, peaks = reader.fieldnames, list(reader)
    peaks.append(dict(peaks[0], interface='l3 delta', metric='force_N', max_sample=999999))
    with peaks_path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(peaks)
    assert [c['sample'] for c in part_cases(folder/'blocked', 'l3')] == [c['sample'] for c in link]
    with pytest.raises(ValueError, match='No structural part'):
        part_cases(folder/'blocked', 'drumstick')


def test_clamped_bounds_export_moments_and_fea_setup(tmp_path):
    from arm_design.fea_cases import exports, part_cases, report, write_csv
    cfg, quiet = load_config(RIGHT), dict(workers=2, log=lambda *a, **k: None)
    folder, recs = bounds(cfg, tmp_path, poses=30, iterations=3, max_force_n=100, max_moment_nm=10, **quiet)
    plain, plain_recs = bounds(cfg, tmp_path, poses=30, iterations=3, max_force_n=100, **quiet)
    assert [f.name for f in exports(folder)] == ['blocked', 'clamped', 'gravity']
    assert [f.name for f in exports(plain)] == ['blocked', 'gravity'] and list(plain_recs) == ['gravity', 'blocked']
    # Clamped samples use their own random stream: gravity and blocked are unchanged by enabling it.
    for kind in ('gravity', 'blocked'):
        np.testing.assert_array_equal(recs[kind].maximum, plain_recs[kind].maximum)
    samples = json.loads((folder/'clamped'/'load_cases.json').read_text())['samples']
    force = METRICS.index('force_N')
    for c in samples:
        assert c['contact_body'] == 'drumstick' and c['phase'].startswith('clamped at tip: ')
        assert np.linalg.norm(c['contact_moment_on_arm_world_Nm']) <= 10+1e-9
        assert np.linalg.norm(c['contact_force_on_arm_world_N']) <= 100+1e-9
        assert np.all(np.abs(c['actuator_torque_Nm']) <= np.array(c['torque_cap_Nm'])+1e-9)
    # Seeds from blocked have no moment; the clamped search adds one.
    assert any(np.linalg.norm(c['contact_moment_on_arm_world_Nm']) > 0 for c in samples)
    # Every blocked tip load is a clamped load with no moment, so clamped peaks cover blocked ones.
    assert np.all(recs['clamped'].maximum[:, 6:] >= recs['blocked'].maximum[:, 6:]-1e-9)
    # The mount carries at most the weight plus the force cap.
    weight = recs['clamped'].model.body_mass.sum()*9.81
    assert recs['clamped'].maximum[0, force] <= weight+100+1e-6
    settings = json.loads((folder/'bounds.json').read_text())
    assert settings['max_contact_moment_nm'] == 10 and any(a.startswith('clamped:') for a in settings['assumptions'])
    assert not any(a.startswith('clamped:') for a in json.loads((plain/'bounds.json').read_text())['assumptions'])
    header = next(csv.reader((folder/'summary.csv').open()))
    assert header[-2:] == ['blocked_force_case', 'clamped_force_case'] and 'clamped_moment_Nm' in header
    assert next(csv.reader((plain/'summary.csv').open()))[-1] == 'blocked_force_case'
    cases = part_cases(folder/'clamped', 'base')
    assert cases and all(c['residual_n'] < 1e-9 and 'moment' in c['contact'] for c in cases)
    assert any(np.linalg.norm(c['contact']['moment']) > 0 for c in cases)
    assert ', M (' in report(folder/'clamped', 'base', cases).split('APPLY contact load')[1].split('\n')[0]
    write_csv(tmp_path/'setup.csv', cases)
    rows = [r for r in csv.DictReader((tmp_path/'setup.csv').open()) if r['role'] == 'apply contact']
    assert len(rows) == len(cases) and any(any(float(r[k]) for k in ('Mx_Nm', 'My_Nm', 'Mz_Nm')) for r in rows)
    assert all(c['residual_n'] < 1e-9 for c in part_cases(folder/'clamped', 'l3'))
    with pytest.raises(ValueError, match='contact-force'):
        bounds(cfg, tmp_path, max_moment_nm=10)
    with pytest.raises(ValueError):
        bounds(cfg, tmp_path, max_force_n=100, max_moment_nm=-1)

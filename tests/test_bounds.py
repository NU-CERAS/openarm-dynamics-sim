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

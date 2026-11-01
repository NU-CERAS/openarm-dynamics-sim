import csv
import json
from copy import deepcopy

import numpy as np
import pytest
import yaml

from arm_design.campaign import Peaks, campaign, run_sweep
from arm_design.collision import CollisionChecker
from arm_design.config import ROOT, load_config
from arm_design.experiment import Experiment
from arm_design.fea import METRICS
from arm_design.model import build_arm


def quick_config():
    c = load_config()
    c['sweep'].update(settle_s=.5, hold_s=.2)
    return c


def fake(minimum, maximum, min_index, max_index):
    shape = (1, 2)
    return dict(counted=True, minimum=np.reshape(minimum, shape), maximum=np.reshape(maximum, shape),
                min_index=np.reshape(min_index, shape), max_index=np.reshape(max_index, shape),
                extrema={i: dict(id=i) for i in set(min_index) | set(max_index)})


def test_peaks_keep_only_samples_behind_extrema():
    p = Peaks((1, 2))
    p.add(0, fake([-1, -1], [1, 1], [0, 0], [1, 1]))
    p.add(1, fake([-2, 0], [0, 3], [5, 5], [6, 6]))
    np.testing.assert_array_equal(p.minimum, [[-2, -1]])
    np.testing.assert_array_equal(p.maximum, [[1, 3]])
    assert set(p.samples) == {(1, 5), (0, 0), (0, 1), (1, 6)}
    p.add(2, fake([-3, -3], [4, 4], [0, 0], [0, 0]))
    assert set(p.samples) == {(2, 0)}
    p.add(3, dict(counted=False))
    assert set(p.samples) == {(2, 0)}


def test_sweep_records_only_sweep_and_hold():
    arm = build_arm(quick_config())
    c = arm.cfg
    d = c['sweep']['duration_s']
    r = run_sweep(arm, c['sweep']['a_deg'], c['sweep']['b_deg'], d)
    assert r['counted'] and r['tracking_ok']
    assert {s['phase'].rsplit(': ', 1)[-1] for s in r['extrema'].values()} <= {'sweep', 'hold'}
    assert r['samples'] == round((d+c['sweep']['hold_s'])/c['simulation']['step_s'])


def test_untrackable_sweeps_are_excluded_unless_requested():
    arm = build_arm(quick_config())
    a, b = np.zeros(7), np.array([-150, -150, 0, 120, 0, 0, 0])
    r = run_sweep(arm, a, b, .1)
    assert r['completed'] and not r['tracking_ok'] and not r['counted'] and 'maximum' not in r
    assert run_sweep(arm, a, b, .1, include_failed=True)['counted']


def test_campaign_exports_load_cases_for_every_peak(tmp_path):
    c = quick_config()
    folder, rec = campaign(c, tmp_path, samples=1, iterations=1, duration_range=(1., 1.5),
                           workers=1, seed=3, log=lambda *a, **k: None)
    sweeps = list(csv.DictReader((folder/'sweeps.csv').open()))
    # Two saved sweeps + one random + one refinement per interface force/moment.
    assert len(sweeps) == 3 + 2*sum(f['sign'] == 1 for f in rec.interfaces)
    assert [s['stage'] for s in sweeps[:3]] == ['saved', 'saved', 'random']
    cases = json.loads((folder/'load_cases.json').read_text())['samples']
    labels = [f['label'] for f in rec.interfaces]
    for peak in csv.DictReader((folder/'peaks.csv').open()):
        i, j = labels.index(peak['interface']), METRICS.index(peak['metric'])
        for side in ('min', 'max'):
            case = cases[int(peak[side+'_sample'])]
            assert case['sweep_id'] == int(peak[side+'_sweep'])
            assert sweeps[case['sweep_id']]['counted'] == 'True'
            loads = np.array(case['loads_local'][i])
            value = np.concatenate((loads, [np.linalg.norm(loads[:3]), np.linalg.norm(loads[3:])]))[j]
            assert value == pytest.approx(float(peak['minimum' if side == 'min' else 'maximum']), rel=1e-12, abs=1e-12)
    # The campaign can only raise peaks over the saved A -> B sweep alone.
    arm = build_arm(quick_config())
    saved = run_sweep(arm, c['sweep']['a_deg'], c['sweep']['b_deg'], c['sweep']['duration_s'])
    assert np.all(rec.maximum >= saved['maximum']) and np.all(rec.minimum <= saved['minimum'])
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['campaign']['sweeps'] == len(sweeps) and meta['samples'] == len(cases)
    assert yaml_unchanged(c)


def yaml_unchanged(c):
    # Sweeps rewrite a private copy of the configuration, never the caller's.
    return c['sweep'] == quick_config()['sweep']


def test_viewer_replay_pauses_on_the_exact_recorded_peak(tmp_path):
    config = tmp_path/'arm.yaml'
    config.write_text(yaml.safe_dump(quick_config(), sort_keys=False))
    campaign(load_config(config), tmp_path/'results'/'nested', samples=0, iterations=0,
             workers=1, log=lambda *a, **k: None)
    e = Experiment(config, tmp_path/'results')
    saved = deepcopy(e.cfg['sweep'])
    e.load_campaign()
    assert 'WARNING' not in e.message
    peak = next(p for p in e.campaign['peaks'] if p['interface'] == 'link2 <- link1' and p['metric'] == 'moment_Nm')
    e.replay_sweep(peak['max_sweep'], peak['max_time_s'], 'test')
    assert e.cfg['sweep'] == saved
    while not e.paused:
        e.tick()
    cases = json.loads((e.campaign['folder']/'load_cases.json').read_text())['samples']
    case = cases[int(peak['max_sample'])]
    np.testing.assert_array_equal(e.data.qpos, case['q_rad'])
    np.testing.assert_array_equal(e.data.qvel, case['qvel_rad_s'])
    e.paused = False
    while e.run:
        e.tick()
    assert e.replay is None and e.last_report.exists() and e.cfg['sweep'] == saved
    e.cfg['joints'][0]['kp_nm_rad'] += 1
    e.load_campaign()
    assert 'WARNING' in e.message


CAD = ROOT/'robots/moving_asm/urdf/moving_asm.sim.yaml'
def test_checker_finds_folded_wrist_and_ignores_designed_overlaps():
    cad = CollisionChecker(build_arm(load_config(CAD)))
    assert cad.ignored == set() and cad.contacts(np.zeros(7)) == {}
    hit = cad.contacts(np.deg2rad([0, 0, 0, 0, 90, -90, -56]))
    assert ('part_5', 'part_7') in hit and hit[('part_5', 'part_7')] > 1
    assert ('part_5', 'part_7') not in CollisionChecker(build_arm(load_config(CAD)), 1e3).contacts(
        np.deg2rad([0, 0, 0, 0, 90, -90, -56]))
    assert cad.path_contacts(np.zeros(7), [0, 0, 0, 0, 90, -90, -56])
    # The built-in J1 housing always overlaps link1 by design.
    builtin = CollisionChecker(build_arm(load_config()))
    assert builtin.ignored == {('base', 'link1')}
    assert builtin.contacts(np.deg2rad(load_config()['sweep']['b_deg'])) == {}


def test_colliding_sweeps_stop_and_never_count():
    c = load_config(CAD)
    c['sweep'].update(settle_s=.5, hold_s=.2)
    arm = build_arm(c)
    checker = CollisionChecker(arm)
    r = run_sweep(arm, np.zeros(7), [0, 0, 0, 0, 90, -90, -56], 2., checker=checker)
    assert not r['completed'] and not r['counted'] and 'part_5-part_7' in r['collision']
    assert r['reason'].startswith('Self-collision')
    clear = run_sweep(arm, np.zeros(7), [30, 20, 0, 30, 0, 0, 0], 2., checker=checker)
    assert clear['counted'] and clear['collision'] == ''

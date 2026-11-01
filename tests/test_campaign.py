import csv
import json

import numpy as np
import pytest

from arm_design.campaign import Peaks, campaign, run_sweep
from arm_design.config import load_config
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

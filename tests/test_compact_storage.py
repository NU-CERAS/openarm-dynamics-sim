import gzip
import json

import numpy as np
import pytest

from arm_design.analysis import RunStore, distribution, metrics
from arm_design.config import load_config
from arm_design.fea import FEARecording
from arm_design.model import build_arm
from arm_design.restrained import build_restrained, trial
from arm_design.trial_recording import regenerate, recording_status, trace_path


def synthetic_result(n, scale=1., counted=True):
    rng = np.random.default_rng(42)
    loads = rng.normal(size=(1201, n, 6))*scale
    rows = [dict(time_s=k*.001, loads_local=v, q_rad=[k*.001]*7,
                 actuator_torque_Nm=[1.]*7, torque_cap_Nm=[2.]*7, sample_index=k)
            for k, v in enumerate(loads)]
    v = metrics(loads)
    low, high = v.argmin(axis=0), v.argmax(axis=0)
    keep = set(low.ravel()) | set(high.ravel())
    return dict(counted=counted, completed=True, reason='' if counted else 'Tracking error', samples=len(rows),
                minimum=v.min(axis=0), maximum=v.max(axis=0), min_index=low, max_index=high,
                extrema={int(k): rows[k] for k in keep}, distribution=distribution(rows, True))


def test_compact_retention_preserves_stats_and_bounds_disk_snapshots(tmp_path):
    arm = build_arm(load_config())
    compact = RunStore(tmp_path/'compact', arm, 'campaign', {})
    full = RunStore(tmp_path/'full', arm, 'campaign', dict(full_history=True))
    for i, (scale, accepted) in enumerate([(1, True), (2, True), (100, False), (.5, True)]):
        result = synthetic_result(len(compact.meta['interfaces']), scale, accepted)
        compact.add(i, result, 'random')
        full.add(i, result, 'random')
        a, b = compact.path/'trials'/str(i), full.path/'trials'/str(i)
        with np.load(a/'distribution.npz') as small, np.load(b/'distribution.npz') as large:
            for key in small.files:
                np.testing.assert_array_equal(small[key], large[key])
        assert json.loads((a/'trial.json').read_text())['minimum'] == result['minimum'].tolist()
        assert (a/'trace.npz').exists() == accepted
        assert (b/'trace.npz').exists()
        if accepted:
            with np.load(a/'trace.npz') as trace:
                assert len(trace['time']) == 120
                assert trace['loads'].dtype == np.float32
                assert trace['q'].dtype == np.float64
                assert trace['time'][0] == 0 and trace['time'][-1] == 1.2
    # Scaling replaces all signed extremes but not necessarily the magnitude minima.
    saved = {}
    for file in (compact.path/'trials').glob('*/cases.json.gz'):
        with gzip.open(file, 'rt') as stream:
            for row in json.load(stream):
                saved[(int(file.parent.name), row['sample_index'])] = row
    assert set(saved) == set(compact.peaks.samples)
    assert not any(t == 2 for t, _ in saved)
    assert len(saved) <= len(compact.meta['interfaces'])*8*2
    assert not (compact.path/'trials/3/cases.json.gz').exists() or any(t == 3 for t, _ in saved)
    assert compact.state['counted'] == 3 and compact.state['rejected'] == 1
    # Snapshot values are still the full-precision original arrays.
    for key, row in saved.items():
        np.testing.assert_array_equal(row['loads_local'], compact.peaks.samples[key]['loads_local'])
    compact.finish('cancelled')
    assert compact.state['retained_snapshots'] == len(saved)


def test_preview_decimation_never_changes_physics_statistics():
    rows = [dict(time_s=k, loads_local=[[np.sin(k), 0, 0, 0, 0, 0]], q_rad=[0],
                 actuator_torque_Nm=[0], torque_cap_Nm=[1]) for k in range(1000)]
    compact, full = distribution(rows), distribution(rows, True)
    for key in ('count', 'counts', 'minimum', 'maximum', 'total', 'squares'):
        np.testing.assert_array_equal(compact[key], full[key])
    assert len(compact['trace']['q']) == 120
    assert len(full['trace']['q']) == 1000


def test_export_shares_one_physical_model(tmp_path):
    arm = build_restrained(load_config())
    store = RunStore(tmp_path, arm, 'restrained', {})
    result = trial(arm, np.zeros(7), np.zeros(7), duration=.02, ramp=.01)
    rec = FEARecording(arm, arm.data())
    rec.rows = list(result['extrema'].values())
    rec.minimum, rec.maximum = result['minimum'], result['maximum']
    rec.min_index = rec.max_index = np.zeros_like(result['min_index'])
    rec.selected_samples = {0}
    rec.stop()
    shared = rec.export(store.path, 'cases', shared_model=store.path/'model.mjb')
    assert (shared/'model.mjb').stat().st_ino == (store.path/'model.mjb').stat().st_ino
    standalone = rec.export(tmp_path, 'cases')
    assert (standalone/'model.mjb').stat().st_ino != (store.path/'model.mjb').stat().st_ino


@pytest.mark.parametrize('scenario', ['restrained', 'campaign'])
def test_generate_selected_recording_preserves_original_data(tmp_path, scenario):
    cfg = load_config()
    if scenario == 'restrained':
        arm = build_restrained(cfg)
        settings = dict(duration=.02, ramp=.01, position_tolerance=.005, angle_tolerance=.05)
        store = RunStore(tmp_path, arm, scenario, settings)
        from arm_design.collision import CollisionChecker
        result = trial(arm, np.zeros(7), np.zeros(7), duration=.02, ramp=.01,
                       checker=CollisionChecker(build_arm(cfg)))
    else:
        from arm_design.campaign import run_sweep
        cfg['sweep'].update(settle_s=.1, hold_s=.1)
        arm = build_arm(cfg)
        store = RunStore(tmp_path, arm, scenario, dict(max_overlap_mm=None))
        result = run_sweep(arm, [0]*7, [0]*7, .1)
    store.add(0, result, 'random')
    store.finish()
    originals = {p: p.read_bytes() for p in store.path.rglob('*') if p.is_file() and not p.name.startswith('trials.sqlite')}
    regenerate(store.path, 0)
    status = recording_status(store.path, 0)
    assert status['verified'] and status['status'] == 'complete'
    with np.load(trace_path(store.path, 0)) as trace:
        assert len(trace['time']) == result['samples']
        assert trace['loads'].dtype == np.float64
    for path, content in originals.items():
        assert path.read_bytes() == content
    # Altered inputs must not silently replace the measured record.
    info_file = store.path/'trials/0/trial.json'
    info = json.loads(info_file.read_text()); info['samples'] += 1
    info_file.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='differs from the original'):
        regenerate(store.path, 0)
    assert recording_status(store.path, 0)['status'] == 'error'
    assert trace_path(store.path, 0).parent.name == '0'

import csv
import json
from copy import deepcopy

import numpy as np
import pytest
import yaml

from arm_design.campaign import PROGRESS, PROGRESS_EVERY, Peaks, campaign, run_sweep
from arm_design.collision import CollisionChecker
from arm_design.config import ROOT, load_config
from arm_design.experiment import Experiment
from arm_design.fea import METRICS
from arm_design.model import build_arm
from arm_design.watch import draw, figure, read, rise_pct, series


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
    assert r['completed'] and not r['tracking_ok'] and not r['counted'] and r['distribution']['count'] == r['samples']
    assert run_sweep(arm, a, b, .1, include_failed=True)['counted']


def test_campaign_exports_load_cases_for_every_peak(tmp_path):
    c = quick_config()
    folder, rec = campaign(c, tmp_path, samples=1, iterations=1, duration_range=(4., 5.),
                           workers=1, seed=3, log=lambda *a, **k: None)
    sweeps = list(csv.DictReader((folder/'sweeps.csv').open()))
    # One random sweep + one refinement per reported row force/moment.
    assert len(sweeps) == 1 + 2*len(rec.interfaces)
    assert sweeps[0]['stage'] == 'random'
    assert all(s['stage'] == 'refine' for s in sweeps[1:])
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
    # The campaign can only raise peaks over its initial random sweep.
    arm = build_arm(quick_config())
    first = rec.sweeps[0]
    assert first['a_deg'] != c['sweep']['a_deg'] and first['b_deg'] != c['sweep']['b_deg']
    saved = run_sweep(arm, first['a_deg'], first['b_deg'], first['duration_s'])
    assert np.all(rec.maximum >= saved['maximum']) and np.all(rec.minimum <= saved['minimum'])
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['campaign']['sweeps'] == len(sweeps) and meta['samples'] == len(cases)
    assert yaml_unchanged(c)
    # Live progress: one row per PROGRESS_EVERY counted sweeps plus a final one, ending at the exported peaks.
    head, values, stages = read(folder/'convergence.csv')
    counted = meta['campaign']['counted']
    assert list(values[:, 0]) == [*range(PROGRESS_EVERY, counted+1, PROGRESS_EVERY)][:len(values)-1] + [counted]
    assert stages[-1] == 'refine' and (folder.parent/PROGRESS).read_bytes() == (folder/'convergence.csv').read_bytes()
    assert len(head) == 4 + 14*len(rec.interfaces)
    logged = [dict(zip(head[4:], row[3:])) for row in values]
    for i, f in enumerate(rec.interfaces):
        for metric, part in (('force_N', slice(0, 3)), ('moment_Nm', slice(3, 6))):
            key, j = f['label']+'|'+metric, METRICS.index(metric)
            assert logged[-1][key] == pytest.approx(rec.maximum[i, j])
            # The full wrench of the record-setting sample rides along with every logged peak.
            for row in logged:
                state = np.array([row[key+'|'+c] for c in METRICS[:6]])
                assert np.linalg.norm(state[part]) == pytest.approx(row[key], rel=1e-12)
            np.testing.assert_allclose(state, cases[rec.max_index[i, j]]['loads_local'][i], rtol=1e-12)
    rows = series(head)
    assert [r['label'] for r in rows] == labels and {r['kind'] for r in rows} == {'proximal', 'distal'}
    assert np.all(rise_pct(values[:, 3:][:, [r[m][0] for r in rows for m in METRICS[6:]]]) >= 0)
    # The live figure draws from this file, from an older magnitude-only one, and past a half-written line.
    import matplotlib
    matplotlib.use('Agg')
    fig, axes, tables = figure()
    draw(folder/'convergence.csv', axes, tables)
    assert any(t.get_text() == 'link1 <- link2' for t in tables[0].texts) and len(axes[0].lines) >= len(rows)
    old = tmp_path/'old.csv'
    old.write_text('counted,sweeps,elapsed_s,stage,b <- a|force_N,b <- a|moment_Nm\n10,12,1.0,random,5.0,2.0\n20,25,2.0,ref')
    assert series(read(old)[0])[0]['force_N'] == (0, None) and len(read(old)[1]) == 1
    draw(old, axes, tables)
    assert any('older campaign' in t.get_text() for t in tables[1].texts)


def yaml_unchanged(c):
    # Sweeps rewrite a private copy of the configuration, never the caller's.
    return c['sweep'] == quick_config()['sweep']


def test_viewer_replay_pauses_on_the_exact_recorded_peak(tmp_path):
    config = tmp_path/'arm.yaml'
    config.write_text(yaml.safe_dump(quick_config(), sort_keys=False))
    campaign(load_config(config), tmp_path/'results'/'nested', samples=1, iterations=0, duration_range=(4., 5.), seed=3,
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


def test_aggressive_duration_search_brackets_feasibility():
    from arm_design.campaign import duration_search
    def run(a, b, duration):
        return dict(counted=duration >= 1., duration_s=duration, saturated_samples=10)
    rows = list(duration_search(run, [0], [1], (.1, 5.), 8, pair_id=7))
    assert len(rows) == 8 and any(not r['counted'] for r in rows)
    assert min(r['duration_s'] for r in rows if r['counted']) < 1.05
    assert all(r['pair_id'] == 7 for r in rows)
    assert [r['search_step'] for r in rows] == list(range(8))
    # Saturation alone does not reject a physically valid trial.
    assert all(r['saturated_samples'] and r['counted'] for r in rows if r['duration_s'] >= 1.)


def test_aggressive_campaign_keeps_both_directions_and_exports_components(tmp_path):
    c = load_config(ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml')
    original = deepcopy(c)
    folder, rec = campaign(c, tmp_path, samples=12, iterations=0, duration_range=(.1, 2.),
                           workers=2, seed=17, aggressive=True, search_steps=6, log=lambda *a, **k:None)
    from arm_design.analysis import trials
    from arm_design.fea_cases import part_cases
    rows = trials(folder.parent, accepted='all')
    assert len(rows) == 12 and c == original
    np.testing.assert_allclose(rows[0]['a_deg'], rows[6]['b_deg'])
    np.testing.assert_allclose(rows[0]['b_deg'], rows[6]['a_deg'])
    assert any(r['counted'] for r in rows)
    assert all(r['acceptance_policy'] == 'motor_loads_v2' for r in rows)
    assert rec.cfg['drive']['trajectory_feedforward']
    assert rec.cfg['sweep']['hold_s'] == .25
    for r in rows:
        assert max(r['peak_torque_fraction']) <= 1+1e-9
        if r['counted']:
            assert not r['numerical_failure']
            assert r['completed'] and not r['partial']
            assert r['max_tracking_error_deg'] <= 5
            assert max(np.abs(r['final_speed_deg_s'])) <= 20
            assert r['samples'] > 0
    cases = part_cases(folder, 'l3', all_extremes=True)
    assert max(c['residual_n'] for c in cases) < 1e-8
    assert max(c['moment_residual_nm'] for c in cases) < 1e-8
    assert (folder/'fea_setup_l3.csv').exists()
    reasons = (folder/'fea_setup_l3.csv').read_text()
    assert 'min Fx_N' in reasons and 'max Fx_N' in reasons
    # Replay uses the saved feedforward controller and shortened hold/settle.
    first = next(r for r in rows if r['counted'] and r['completed'])
    replay = run_sweep(build_arm(rec.cfg), first['a_deg'], first['b_deg'], first['duration_s'])
    np.testing.assert_allclose(replay['maximum'], first['maximum'], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('kwargs', [dict(include_failed=True), dict(max_overlap_mm=None), dict(iterations=1), dict(search_steps=1)])
def test_aggressive_search_rejects_unsafe_or_incomplete_settings(tmp_path, kwargs):
    args = dict(samples=16, iterations=0, aggressive=True)
    args.update(kwargs)
    with pytest.raises(ValueError):
        campaign(quick_config(), tmp_path, **args)


def motor_load_config():
    c = load_config()
    c['sweep'].update(settle_s=.1, hold_s=.1, load_acceptance='motor_loads_v1')
    c['drive']['trajectory_feedforward'] = True
    a = np.mean([j['limits_deg'] for j in c['joints']], axis=1)
    return c, a, a + np.array([70, 40, 0, 40, 0, 0, 0])


def test_motor_loads_accept_tracking_lag_and_residual_speed():
    c, a, b = motor_load_config()
    r = run_sweep(build_arm(c), a, b, .1, strict=True)
    assert r['completed'] and r['counted'] and not r['partial']
    assert not r['tracking_ok'] and r['max_tracking_error_deg'] > 10
    assert max(np.abs(r['final_speed_deg_s'])) > 5
    assert max(r['peak_torque_fraction']) <= 1 + 1e-9
    assert r['distribution']['count'] == r['samples'] == 200
    assert len(r['diagnostics']) == 2


def test_motor_loads_allow_initial_sag():
    c, a, b = motor_load_config()
    c['drive'].update(trajectory_feedforward=False, gravity_compensation=False)
    c['sweep']['tracking_tolerance_deg'] = .000001
    r = run_sweep(build_arm(c), a, b, .1, strict=True)
    assert r['start_tracking_error_deg'] > c['sweep']['tracking_tolerance_deg']
    assert r['samples'] > 0 and r['counted']


@pytest.mark.parametrize('policy', ['motor_loads_v1', 'motor_loads_v2'])
@pytest.mark.parametrize('event', ['collision', 'joint_limit', 'overspeed', 'numerical'])
def test_motor_load_prefix_excludes_event_sample(monkeypatch, event, policy):
    # This compares consecutive physics samples, not the decimated playback preview.
    from arm_design import campaign as campaign_module
    monkeypatch.setattr(campaign_module, '_full_history', True)
    from arm_design import simulation
    c, a, b = motor_load_config()
    c['sweep'].update(load_acceptance=policy, tracking_tolerance_deg=5.)
    baseline = run_sweep(build_arm(deepcopy(c)), a, b, .1, strict=True)
    original = simulation.control
    # 0.1 s approach + 0.1 s settle, then retain exactly 10 valid samples.
    stop_time = .21
    def control(arm, data, *args, **kwargs):
        cap = original(arm, data, *args, **kwargs)
        if data.time >= stop_time - 1e-9:
            if event == 'joint_limit':
                data.qpos[0] = np.deg2rad(c['joints'][0]['limits_deg'][1] + 10)
            elif event == 'overspeed':
                motor = c['motors'][c['joints'][0]['motor']]
                data.qvel[0] = motor['no_load_speed_rpm'] * np.pi / 30 * 1.1
            elif event == 'numerical':
                raise RuntimeError('Injected numerical failure')
        return cap
    monkeypatch.setattr(simulation, 'control', control)
    class Checker:
        calls = 0
        def contacts(self, q):
            self.calls += 1
            return {('test_a', 'test_b'): 1.} if event == 'collision' and self.calls == 211 else {}
    r = run_sweep(build_arm(deepcopy(c)), a, b, .1, checker=Checker(), strict=True)
    assert not r['completed'] and r['samples'] == 10
    assert r['distribution']['count'] == 10
    assert r['measured_duration_s'] == pytest.approx(.01)
    assert r['counted'] is (event != 'numerical' and policy == 'motor_loads_v1')
    assert r['partial'] is (event != 'numerical' and policy == 'motor_loads_v1')
    assert r['numerical_failure'] is (event == 'numerical')
    # Exactly the original, uncontaminated pre-event loads feed all extrema.
    trace = baseline['distribution']['trace']
    loads = trace['loads'][:10]
    values = np.concatenate((loads, np.linalg.norm(loads[..., :3], axis=-1, keepdims=True),
                             np.linalg.norm(loads[..., 3:], axis=-1, keepdims=True)), axis=-1)
    np.testing.assert_allclose(r['minimum'], values.min(axis=0))
    np.testing.assert_allclose(r['maximum'], values.max(axis=0))


def test_motor_load_stop_before_measurement_rejects():
    c, a, b = motor_load_config()
    class Checker:
        def contacts(self, q):
            return {('test_a', 'test_b'): 1.}
    r = run_sweep(build_arm(c), a, b, .1, checker=Checker(), strict=True)
    assert not r['counted'] and not r['partial'] and r['samples'] == 0


def test_partial_trial_does_not_establish_duration_search_success():
    from arm_design.campaign import duration_search
    calls = []
    def run(a, b, duration):
        calls.append(duration)
        return dict(counted=True, completed=len(calls) == 1, partial=len(calls) != 1)
    list(duration_search(run, [0], [1], (.1, 1.), 3))
    assert calls == pytest.approx([1., .6, np.sqrt(.6)])


def test_screened_motor_loads_reject_tracking_over_five_degrees():
    c, a, b = motor_load_config()
    c['sweep'].update(load_acceptance='motor_loads_v2', tracking_tolerance_deg=5.)
    r = run_sweep(build_arm(c), a, b, .1, strict=True, include_failed=True)
    assert not r['counted'] and not r['completed'] and not r['partial']
    assert r['reason'] == 'Tracking error exceeded tolerance.'
    assert 5 < r['max_tracking_error_deg'] < 10
    assert r['samples'] > 0 and r['distribution']['count'] == r['samples']


def test_screened_motor_loads_reject_final_speed_over_twenty():
    c, a, b = motor_load_config()
    # Isolate settling from tracking: the trajectory is stable but ends moving.
    c['sweep'].update(load_acceptance='motor_loads_v2', tracking_tolerance_deg=180.,
                      final_speed_tolerance_deg_s=20.)
    r = run_sweep(build_arm(deepcopy(c)), a, b, .1, strict=True)
    assert r['completed'] and not r['counted'] and not r['partial']
    assert r['reason'] == 'Did not brake to within 20 deg/s at the end of the hold.'
    assert 20 < max(np.abs(r['final_speed_deg_s'])) < 100
    c['sweep']['final_speed_tolerance_deg_s'] = 100.
    accepted = run_sweep(build_arm(c), a, b, .1, strict=True)
    assert accepted['counted'] and accepted['reason'] == ''
    np.testing.assert_array_equal(r['maximum'], accepted['maximum'])

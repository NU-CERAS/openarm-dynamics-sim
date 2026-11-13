"""Search A -> B sweeps for the largest part mate loads; MuJoCo still runs every sweep."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
import hashlib
import multiprocessing

import mujoco
from pathlib import Path
import time

import numpy as np
import yaml

from .collision import CollisionChecker, describe
from .fea import FEARecording, METRICS
from .model import build_arm
from .reporting import summary
from .simulation import Sweep
from .analysis import RunStore, distribution

# Every reported connection is refined on both magnitudes.
OBJECTIVES = (METRICS.index('force_N'), METRICS.index('moment_Nm'))
_arm = None
_include_failed = False
_checker = None
_full_history = False
_cancel_path = None
# Physics samples between collision checks of the simulated motion (5 ms at 1 kHz).
CHECK_EVERY = 5
# Perturbations tried per refinement iteration before giving up on a collision-free path.
PATH_TRIES = 50
# Counted sweeps between rows of the live convergence file (arm-sim watch plots it).
PROGRESS_EVERY = 10
PROGRESS = 'campaign-progress.csv'


def _init(cfg, include_failed=False, max_overlap_mm=0., full_history=False, cancel_path=None):
    global _arm, _include_failed, _checker, _full_history, _cancel_path
    # Sweeps rewrite arm.cfg['sweep']; never alias the caller's configuration.
    _full_history, _cancel_path = full_history, cancel_path
    _arm, _include_failed = build_arm(deepcopy(cfg)), include_failed
    _checker = None if max_overlap_mm is None else CollisionChecker(_arm, max_overlap_mm)


def run_sweep(arm, a, b, duration, include_failed=False, checker=None, strict=False):
    """One recorded A -> B sweep starting at rest at A; only sweep+hold loads count.

    By default a sweep counts only if it tracks within tolerance: commands the
    arm cannot follow end in joint-stop impacts, not motor-driven sweep loads.
    With a checker, a sweep whose simulated motion self-collides is stopped."""
    s = arm.cfg['sweep']
    # Starting at A makes the approach trivial; settle still removes gravity sag.
    s.update(a_deg=[float(x) for x in a], b_deg=[float(x) for x in b],
             duration_s=float(duration), approach_s=.1)
    data = arm.data()
    run = Sweep(arm, data)
    rec = FEARecording(arm, data)
    rec.phases = ('sweep', 'hold')
    result = dict(a_deg=s['a_deg'], b_deg=s['b_deg'], duration_s=float(duration))
    collision, validity = {}, ''
    try:
        while run.tick(arm, data, recorder=rec):
            if not rec.active:
                run.reason = rec.reason
                break
            if strict and run.rows[-1]['phase'] in rec.phases:
                row = run.rows[-1]
                if np.any(data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)):
                    validity = 'Joint-limit constraint activated.'
                elif np.any(np.abs(data.qvel)*30/np.pi > np.array([arm.cfg['motors'][j['motor']]['no_load_speed_rpm'] for j in arm.cfg['joints']])):
                    validity = 'Motor no-load speed exceeded.'
                elif np.max(np.abs(np.rad2deg(row['q']-row['target']))) > s['tracking_tolerance_deg']:
                    validity = 'Tracking error exceeded tolerance.'
                if validity:
                    run.reason = validity
                    break
            # Overshoot can collide even when the planned path is clear.
            if checker and (strict or len(run.rows) % CHECK_EVERY == 0) and run.rows[-1]['phase'] in rec.phases:
                collision = checker.contacts(data.qpos)
                if collision:
                    run.reason = 'Self-collision: '+describe(collision)
                    break
    except RuntimeError as exc:
        run.reason = str(exc)
    rec.stop(run.reason)
    s = summary(run)
    if strict and run.complete and run.rows and np.max(np.abs(np.rad2deg(run.rows[-1]['velocity']))) > 5.:
        validity = 'Did not brake to within 5 deg/s at the end of the hold.'
        run.reason = validity
    if strict and not s['tracking_ok'] and not run.reason:
        run.reason = 'Tracking error exceeded tolerance.'
    result.update(completed=bool(run.complete and rec.rows), reason=run.reason, collision=describe(collision),
                  tracking_ok=s['tracking_ok'], samples=len(rec.rows),
                  max_tracking_error_deg=rec.max_tracking_error_deg,
                  saturated_samples=rec.saturated_samples, contact_samples=rec.contact_samples)
    result['counted'] = result['completed'] and (result['tracking_ok'] or include_failed) and not validity
    task = [r for r in run.rows if r['phase'] in rec.phases]
    if task:
        applied, caps = np.array([r['torque'] for r in task]), np.array([r['cap'] for r in task])
        utilization = np.divide(np.abs(applied), caps, out=np.zeros_like(applied), where=caps > 1e-8)
        result.update(peak_torque_fraction=utilization.max(axis=0).tolist(),
                      peak_speed_deg_s=np.rad2deg(np.abs([r['velocity'] for r in task]).max(axis=0)).tolist(),
                      final_speed_deg_s=np.rad2deg(run.rows[-1]['velocity']).tolist())
    result['distribution'] = distribution(rec.rows, _full_history)
    if not rec.rows:
        return result
    keep = sorted(set(rec.min_index.ravel().tolist()) | set(rec.max_index.ravel().tolist()))
    result.update(minimum=rec.minimum, maximum=rec.maximum, min_index=rec.min_index,
                  max_index=rec.max_index, extrema={i: rec.rows[i] for i in keep})
    return result


def _sweep(params):
    return [run_sweep(_arm, *params, _include_failed, _checker)]


def duration_search(run, a, b, bounds, attempts, pair_id=0, direction='A to B'):
    """Bounded geometric search; every attempted duration remains a recorded trial.

    A failure defines a lower duration bound; validity is a local search heuristic,
    not a proof of monotonicity or global time optimality.
    """
    lower, upper = bounds
    duration, failed, passed = upper, None, None
    for step in range(attempts):
        result = run(a, b, duration)
        result.update(pair_id=pair_id, direction=direction, search_step=step,
                      search_role='baseline' if step == 0 else 'duration_search')
        yield result
        if result['counted']:
            passed = duration
            duration = np.sqrt(failed*passed) if failed is not None else max(lower, duration*.6)
        else:
            failed = duration
            duration = np.sqrt(failed*passed) if passed is not None else max(lower, duration*.6)
        # If the baseline fails, explore faster commands without claiming a valid bracket.


def _aggressive(job):
    a, b, bounds, attempts, pair_id, direction = job
    results = []
    def run(a, b, duration):
        return run_sweep(_arm, a, b, duration, checker=_checker, strict=True)
    for result in duration_search(run, a, b, bounds, attempts, pair_id, direction):
        results.append(result)
        if _cancel_path and Path(_cancel_path).exists():
            break
    return results


def _validation_trial(job):
    index, source = job
    r = run_sweep(_arm, source['a_deg'], source['b_deg'], source['duration_s'], checker=_checker, strict=True)
    return dict(trial=index, counted=r['counted'], reason=r['reason'],
                maximum=np.asarray(r.get('maximum', [])).tolist(), minimum=np.asarray(r.get('minimum', [])).tolist())


def validate_extreme_sweeps(cfg, store, peaks, sweeps, workers, log):
    """Rerun record-setting trials at two finer steps; retain screening and validation separately."""
    ids = sorted(set(peaks.min_source[..., 0].ravel()) | set(peaks.max_source[..., 0].ravel()))
    jobs = [(int(i), sweeps[i]) for i in ids]
    levels = []
    context = multiprocessing.get_context('spawn')
    for divisor in (2, 4):
        fine = deepcopy(cfg)
        fine['simulation']['step_s'] /= divisor
        checked = []
        with ProcessPoolExecutor(workers, context, _init,
                (fine, False, store.state['settings']['max_overlap_mm'], False, str(store.path/'CANCEL'))) as pool:
            for offset in range(0, len(jobs), workers):
                if store.cancelled:
                    break
                checked.extend(pool.map(_validation_trial, jobs[offset:offset+workers]))
                store.state['validation_progress'] = dict(timestep_s=fine['simulation']['step_s'],
                    checked=len(checked), total=len(jobs))
                store.publish()
                log(f"Finer-step validation {divisor}x: {len(checked)}/{len(jobs)}", flush=True)
        levels.append(dict(timestep_s=fine['simulation']['step_s'], trials=checked))
        if store.cancelled:
            break
    stable = []
    if len(levels) == 2:
        previous = {r['trial']:r for r in levels[0]['trials']}
        for r in levels[1]['trials']:
            old = previous.get(r['trial'])
            if old and old['counted'] and r['counted']:
                a, b = np.asarray(old['maximum'])[:, 6:], np.asarray(r['maximum'])[:, 6:]
                stable.append(dict(trial=r['trial'], maximum_relative_peak_change=float(
                    np.max(np.abs(b-a)/np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)))))
    result = dict(screening_timestep_s=cfg['simulation']['step_s'], source_trials=[int(i) for i in ids],
        completed=not store.cancelled, levels=levels, convergence=stable,
        note='Screening peaks remain unchanged. Review finer-step loads and acceptance before final FEA sign-off.')
    (store.path/'timestep-validation.json').write_text(json.dumps(result))
    return dict(checked_trials=len(stable), source_trials=len(ids), completed=not store.cancelled,
        maximum_relative_peak_change=max((r['maximum_relative_peak_change'] for r in stable), default=None),
        trials_over_5_percent=sum(r['maximum_relative_peak_change'] > .05 for r in stable),
        invalid_at_finer_steps=sum(not r['counted'] for level in levels for r in level['trials']))


def score(result, interface, metric):
    return result['maximum'][interface, metric] if result['counted'] else -np.inf


def _refine(job):
    """(1+1) evolution strategy on one row/metric, from its best sweep so far."""
    a, b, duration = (np.array(job[k], float) for k in ('a_deg', 'b_deg', 'duration_s'))
    lo, hi = job['limits_deg'][:, 0], job['limits_deg'][:, 1]
    rng = np.random.default_rng(job['seed'])
    best, sigma, results = job['score'], .1, []
    for _ in range(job['iterations']):
        if _cancel_path and Path(_cancel_path).exists():
            break
        for _ in range(PATH_TRIES):
            ta = np.clip(a + rng.normal(0, sigma, a.shape)*(hi-lo), lo, hi)
            tb = np.clip(b + rng.normal(0, sigma, b.shape)*(hi-lo), lo, hi)
            if _checker is None or not _checker.path_contacts(ta, tb):
                break
        else:
            sigma = max(sigma*.85, .005)
            continue
        td = float(np.clip(duration*np.exp(rng.normal(0, sigma*3)), *job['duration_range']))
        result = run_sweep(_arm, ta, tb, td, _include_failed, _checker)
        results.append(result)
        value = score(result, job['interface'], job['metric'])
        if value > best:
            best, a, b, duration, sigma = value, ta, tb, td, min(sigma*1.5, .5)
        else:
            sigma = max(sigma*.85, .005)
    return results


class Peaks:
    """Running per-row extrema across sweeps, keeping only the samples behind them."""
    def __init__(self, shape):
        self.minimum = np.full(shape, np.inf)
        self.maximum = np.full(shape, -np.inf)
        self.min_source = np.full(shape+(2,), -1)
        self.max_source = np.full(shape+(2,), -1)
        self.samples = {}

    def add(self, sweep_id, result):
        if not result['counted']:
            return
        for values, index, current, source, better in (
                (result['minimum'], result['min_index'], self.minimum, self.min_source, np.less),
                (result['maximum'], result['max_index'], self.maximum, self.max_source, np.greater)):
            mask = better(values, current)
            current[mask] = values[mask]
            source[mask, 0] = sweep_id
            source[mask, 1] = index[mask]
        for sample_index, sample in result['extrema'].items():
            self.samples[(sweep_id, sample_index)] = sample
        used = {tuple(k) for k in np.concatenate((self.min_source, self.max_source)).reshape(-1, 2).tolist()}
        self.samples = {k: v for k, v in self.samples.items() if k in used}


class CampaignRecording(FEARecording):
    """Extrema samples from many sweeps, exported in the normal FEA load-case format."""
    def __init__(self, arm, peaks, sweeps, settings):
        super().__init__(arm, arm.data())
        keys = sorted(peaks.samples)
        row_of = {k: i for i, k in enumerate(keys)}
        for i, (sweep_id, sample_index) in enumerate(keys):
            sample = dict(peaks.samples[(sweep_id, sample_index)])
            sample.update(sample_index=i, sweep_id=sweep_id, sweep_sample_index=sample_index,
                          phase=f'sweep {sweep_id}: '+sample['phase'])
            self.rows.append(sample)
        lookup = lambda source: np.array([row_of[tuple(k)] for k in source.reshape(-1, 2).tolist()]).reshape(source.shape[:2])
        self.minimum, self.maximum = peaks.minimum, peaks.maximum
        self.min_index, self.max_index = lookup(peaks.min_source), lookup(peaks.max_source)
        self.sweeps, self.settings = sweeps, settings
        done = [s for s in sweeps if s['counted']]
        self.max_tracking_error_deg = max(s['max_tracking_error_deg'] for s in done)
        self.saturated_samples = sum(s['saturated_samples'] for s in done)
        self.contact_samples = sum(s['contact_samples'] for s in done)
        self.stop(f'Campaign of {len(sweeps)} sweeps complete.')

    def peaks(self):
        result = super().peaks()
        for row in result:
            row['min_sweep'] = self.rows[row['min_sample']]['sweep_id']
            row['max_sweep'] = self.rows[row['max_sample']]['sweep_id']
        return result

    def metadata(self):
        meta = super().metadata()
        meta.update(samples_meaning='This export contains extrema only; per-trial histories and distributions are in the parent run directory.',
                    duration_s=None, campaign=self.settings,
                    recorded_samples=sum(s['samples'] for s in self.sweeps),
                    max_tracking_error_deg_note='Largest error of any sweep; short sweeps often saturate motors.')
        meta['limitations'] = meta['limitations'] + [
            'Sampled search over A/B poses and durations: the largest loads FOUND, not a proven global maximum.',
            'Only A -> B sweeps from rest, plus the final hold, are searched; jogging and Go to A/B are not.',
            'Unless include_failed is set, only sweeps that tracked within tolerance contribute loads.',
            'Self-collision uses convex hulls of the meshes (conservative); pairs touching at the reference pose are ignored. Contacts never act in the physics.']
        return meta


def campaign(cfg, out, samples=200, iterations=20, duration_range=(.1, 5.), workers=None, seed=0,
             include_failed=False, max_overlap_mm=0., log=print, full_history=False, aggressive=False, search_steps=8,
             timestep_validation=True):
    """max_overlap_mm=None disables the self-collision check."""
    lo_d, hi_d = duration_range
    if not .1 <= lo_d <= hi_d <= 120:
        raise ValueError('Durations must satisfy 0.1 <= minimum <= maximum <= 120 seconds.')
    if samples < 1 or iterations < 0:
        raise ValueError('Need at least one random sample and nonnegative iterations.')
    if workers is not None and workers < 1:
        raise ValueError('Worker count must be positive.')
    if max_overlap_mm is not None and not max_overlap_mm >= 0:
        raise ValueError('Allowed overlap must be a nonnegative number of millimetres.')
    if aggressive and (include_failed or max_overlap_mm is None or iterations):
        raise ValueError('Aggressive search requires collision checks, excludes failed trials, and uses its own duration refinement (--iterations 0).')
    if aggressive and search_steps < 2:
        raise ValueError('Aggressive search needs at least two duration steps per complete direction.')
    cfg = deepcopy(cfg)
    if aggressive:
        cfg['sweep'].update(settle_s=.1, hold_s=.25)
        cfg['drive']['trajectory_feedforward'] = True
        cfg['simulation']['step_s'] = min(cfg['simulation']['step_s'], .0005)
    workers = workers or multiprocessing.cpu_count()
    arm = build_arm(deepcopy(cfg))
    limits = np.array([j['limits_deg'] for j in cfg['joints']], float)
    rng = np.random.default_rng(seed)
    checker = None if max_overlap_mm is None else CollisionChecker(arm, max_overlap_mm)
    rejected = 0

    def draw():
        nonlocal rejected
        # Redraw until the planned path is collision-free; the simulated motion is checked again.
        for _ in range(1000):
            a, b = rng.uniform(limits[:, 0], limits[:, 1]), rng.uniform(limits[:, 0], limits[:, 1])
            if checker is None or not checker.path_contacts(a, b):
                return a, b
            rejected += 1
        raise RuntimeError('No collision-free A/B path in 1000 tries; check the joint limits.')

    settings = dict(samples=samples, iterations=iterations, seed=seed,
        duration_range=list(duration_range), full_history=full_history, workers=workers,
        include_failed=include_failed, max_overlap_mm=max_overlap_mm, timestep_validation=timestep_validation,
        campaign_mode='aggressive' if aggressive else 'random')
    if aggressive:
        settings.update(search_steps=search_steps, pose_pairs=(samples+2*search_steps-1)//(2*search_steps),
                        hold_s=.25, settle_s=.1, timestep_s=cfg['simulation']['step_s'], final_speed_tolerance_deg_s=5.,
                        controller='PD + inverse-dynamics trajectory feedforward, within configured torque/speed caps',
                        assumptions=['Adaptive duration search with torque-limited inverse-dynamics feedforward.',
                                     'Both directions start independently at rest; no continuous reversal is claimed.',
                                     'Torque saturation alone is allowed; tracking, joint stops, overspeed and collisions reject trials.',
                                     'Every physics timestep contributes statistics; search frequencies are not a duty cycle.'])
    if cfg.get('model_type') == 'urdf':
        settings['urdf_sha256'] = hashlib.sha256(Path(cfg['urdf']['path']).read_bytes()).hexdigest()
    store = RunStore(out, arm, 'campaign', settings)
    # Publish the run before checking paths, so even preparation is visible.
    try:
        if aggressive:
            jobs = []
            remaining = samples
            for pair_id in range(settings['pose_pairs']):
                if store.cancelled:
                    break
                a, b = draw()
                if pair_id % 50 == 0:
                    store.state['prepared_pairs'] = pair_id+1
                    store.publish()
                    log(f'Prepared {pair_id+1}/{settings["pose_pairs"]} random pose pairs', flush=True)
                for start, end, direction in ((a,b,'A to B'),(b,a,'B to A')):
                    attempts = min(search_steps, remaining)
                    if attempts:
                        jobs.append((start,end,duration_range,attempts,pair_id,direction))
                        remaining -= attempts
            execute = _aggressive
        else:
            jobs = []
            for _ in range(samples):
                if store.cancelled:
                    break
                jobs.append((*draw(), float(np.exp(rng.uniform(np.log(lo_d), np.log(hi_d))))))
            execute = _sweep
    except BaseException as exc:
        store.finish('failed', error=str(exc))
        raise
    recorder = FEARecording(arm, arm.data())
    interfaces = recorder.interfaces
    peaks = Peaks((len(interfaces), len(METRICS)))
    sweeps, best = [], {}
    targets = [(i, j) for i in range(len(interfaces)) for j in OBJECTIVES]
    started = time.monotonic()
    Path(out).mkdir(parents=True, exist_ok=True)
    progress = (store.path/PROGRESS).open('w', newline='')
    progress_csv, counted = csv.writer(progress), 0
    audited_samples, distribution_failures = 0, []
    # Per target: its peak magnitude, then the full local wrench of the sample that set it.
    progress_csv.writerow(['counted', 'sweeps', 'elapsed_s', 'stage', *(f'{interfaces[i]["label"]}|{METRICS[j]}{c}'
                           for i, j in targets for c in ('', *('|'+x for x in METRICS[:6])))])

    def snapshot(stage):
        record = [x for t in targets for x in (peaks.maximum[t], *peaks.samples[tuple(peaks.max_source[t])]['loads_local'][t[0]])]
        progress_csv.writerow([counted, len(sweeps), round(time.monotonic()-started, 1), stage, *record])
        progress.flush()
        # Compatibility feed for arm-sim watch; durable copy belongs to this run.
        (Path(out)/PROGRESS).write_bytes((store.path/PROGRESS).read_bytes())

    def collect(results, stage):
        nonlocal counted, audited_samples
        for result in results:
            sweep_id = len(sweeps)
            if aggressive and result.get('distribution') is not None:
                dist = result['distribution']
                audited_samples += int(dist['count'])
                if not np.all(dist['counts'].sum(axis=-1) == dist['count']):
                    distribution_failures.append(sweep_id)
                    raise RuntimeError(f'Trial {sweep_id} distribution sample count mismatch.')
                if max(result.get('peak_torque_fraction', [0])) > 1+1e-8:
                    raise RuntimeError(f'Trial {sweep_id} exceeded an actuator torque cap.')
            store.add(sweep_id, result, stage)
            peaks.add(sweep_id, result)
            for target in targets:
                value = score(result, *target)
                if value > best.get(target, (-np.inf,))[0]:
                    best[target] = (value, sweep_id)
            sweeps.append({k: v for k, v in result.items()
                           if k not in ('minimum', 'maximum', 'min_index', 'max_index', 'extrema', 'distribution')} | dict(stage=stage))
            counted += result['counted']
            if result['counted'] and counted % PROGRESS_EVERY == 0:
                snapshot(stage)
        log(f'{len(sweeps)} sweeps, {time.monotonic()-started:.0f} s', flush=True)

    def refine_jobs():
        result = []
        for n, ((i, j), (value, sweep_id)) in enumerate(sorted(best.items())):
            source = sweeps[sweep_id]
            result.append(dict(source, interface=i, metric=j, score=value, iterations=iterations,
                               limits_deg=limits, duration_range=(lo_d, hi_d), seed=[seed, n]))
        return result

    try:
        if workers == 1:
            _init(cfg, include_failed, max_overlap_mm, full_history, str(store.path/'CANCEL'))
            for job in jobs:
                if store.cancelled:
                    break
                collect(execute(job), 'random')
            if iterations:
                for job in refine_jobs():
                    if store.cancelled:
                        break
                    collect(_refine(job), 'refine')
        else:
            context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(workers, context, _init,
                    (cfg, include_failed, max_overlap_mm, full_history, str(store.path/'CANCEL'))) as pool:
                for offset in range(0, len(jobs), workers):
                    if store.cancelled:
                        break
                    for results in pool.map(execute, jobs[offset:offset+workers]):
                        collect(results, 'random')
                if iterations and not store.cancelled:
                    refine = refine_jobs()
                    for offset in range(0, len(refine), workers):
                        if store.cancelled:
                            break
                        for results in pool.map(_refine, refine[offset:offset+workers]):
                            collect(results, 'refine')
        if counted % PROGRESS_EVERY:
            snapshot(sweeps[-1]['stage'])
    except BaseException as exc:
        store.finish('failed', error=str(exc))
        raise
    finally:
        progress.close()
    if not any(s['counted'] for s in sweeps):
        store.finish('cancelled' if store.cancelled else 'failed', error='No accepted sweeps.')
        if store.cancelled:
            return store.path, None
        raise RuntimeError('No sweep tracked within tolerance; lengthen --min-duration or pass --include-failed.')
    settings = dict(**store.state['settings'], refine_iterations=iterations, duration_range_s=[lo_d, hi_d],
                    sweeps=len(sweeps), completed=sum(s['completed'] for s in sweeps),
                    tracking_ok=sum(s['tracking_ok'] for s in sweeps), counted=sum(s['counted'] for s in sweeps),
                    tracking_tolerance_deg=cfg['sweep']['tracking_tolerance_deg'],
                    colliding_paths_skipped=rejected,
                    collided=sum(bool(s['collision']) for s in sweeps),
                    ignored_pairs=sorted(map(list, checker.ignored)) if checker else [],
                    measured_phases=['sweep', 'hold'])
    try:
        rec = CampaignRecording(arm, peaks, sweeps, settings)
        folder = rec.export(store.path, 'cases')
        with (folder/'sweeps.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sweep', 'stage', 'duration_s', 'completed', 'tracking_ok', 'counted', 'max_tracking_error_deg',
                             'saturated_samples', 'reason', 'collision', *(f'a_{j["name"]}_deg' for j in cfg['joints']),
                             *(f'b_{j["name"]}_deg' for j in cfg['joints'])])
            for i, s in enumerate(sweeps):
                writer.writerow([i, s['stage'], s['duration_s'], s['completed'], s['tracking_ok'],
                                 s['counted'], s['max_tracking_error_deg'], s['saturated_samples'], s['reason'], s['collision'], *s['a_deg'], *s['b_deg']])
        (folder/'campaign.json').write_text(json.dumps(settings, indent=2))
        (folder/'convergence.csv').write_bytes((store.path/PROGRESS).read_bytes())
        if aggressive:
            from .fea_cases import part_cases, write_csv
            residuals = {}
            for part in rec.parts():
                cases = part_cases(folder, part['name'], all_extremes=True)
                residuals[part['name']] = dict(cases=len(cases),
                    force_N=max(c['residual_n'] for c in cases), moment_Nm=max(c['moment_residual_nm'] for c in cases))
                if max(residuals[part['name']]['force_N'], residuals[part['name']]['moment_Nm']) > 1e-6:
                    raise RuntimeError(f"FEA balance check failed for {part['name']}.")
                write_csv(folder/f"fea_setup_{part['name']}.csv", cases)
            from collections import Counter
            accepted = [r for r in sweeps if r['counted']]
            validation = dict(trials=len(sweeps), accepted=len(accepted), rejected=len(sweeps)-len(accepted),
                physics_samples=audited_samples, distribution_failures=distribution_failures,
                rejected_reasons=dict(Counter(r['reason'] for r in sweeps if not r['counted'])),
                accepted_saturated_trials=sum(r['saturated_samples'] > 0 for r in accepted),
                accepted_peak_torque_fraction=np.max([r['peak_torque_fraction'] for r in accepted], axis=0).tolist(),
                accepted_duration_range_s=[min(r['duration_s'] for r in accepted), max(r['duration_s'] for r in accepted)],
                fea_balance=residuals, urdf_sha256=settings.get('urdf_sha256'))
            store.state['export'] = str(folder)
            store.state['phase'] = 'finer-step validation'
            store.publish()
            if not store.cancelled and timestep_validation:
                validation['timestep_validation'] = validate_extreme_sweeps(cfg, store, peaks, sweeps, workers, log)
            else:
                validation['timestep_validation'] = dict(completed=False, reason='Cancelled' if store.cancelled else 'Not requested')
            (store.path/'validation-summary.json').write_text(json.dumps(validation, indent=2))
            store.state['phase'] = 'finished'
        store.finish('cancelled' if store.cancelled else 'complete', folder)
    except BaseException as exc:
        store.finish('failed', error=str(exc))
        raise
    return folder, rec


def load_results(folder):
    """A campaign export, parsed for replay: its config, every sweep and every peak."""
    folder = Path(folder).expanduser().resolve()
    if not (folder/'campaign.json').exists():
        raise ValueError(f'{folder} is not a campaign export (no campaign.json).')
    cfg = yaml.safe_load((folder/'config.yaml').read_text())
    names = [j['name'] for j in cfg['joints']]
    with (folder/'sweeps.csv').open() as f:
        sweeps = [dict(stage=row['stage'], duration_s=float(row['duration_s']),
                       counted=row['counted'] == 'True', tracking_ok=row['tracking_ok'] == 'True',
                       a_deg=[float(row[f'a_{n}_deg']) for n in names],
                       b_deg=[float(row[f'b_{n}_deg']) for n in names]) for row in csv.DictReader(f)]
    with (folder/'peaks.csv').open() as f:
        peaks = [dict(row, minimum=float(row['minimum']), maximum=float(row['maximum']),
                      min_sweep=int(row['min_sweep']), max_sweep=int(row['max_sweep']),
                      min_time_s=float(row['min_time_s']), max_time_s=float(row['max_time_s']))
                 for row in csv.DictReader(f) if not row['interface'].endswith(' delta')]
    return dict(folder=folder, cfg=cfg, settings=json.loads((folder/'campaign.json').read_text()),
                sweeps=sweeps, peaks=peaks)


def latest_results(root, joints):
    """Newest campaign export anywhere under root that was run on these joints."""
    for path in sorted(Path(root).glob('**/campaign.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        cfg = yaml.safe_load((path.parent/'config.yaml').read_text())
        if [j['name'] for j in cfg['joints']] == list(joints):
            return path.parent
    raise ValueError(f'No campaign for this robot under {root}. Run arm-sim campaign first.')

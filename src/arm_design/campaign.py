"""Search A -> B sweeps for the largest interface loads; MuJoCo still runs every sweep."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import yaml

from .fea import FEARecording, METRICS
from .model import build_arm
from .reporting import summary
from .simulation import Sweep

# Magnitudes are identical on both receiving-body views, so refine one side only.
OBJECTIVES = (METRICS.index('force_N'), METRICS.index('moment_Nm'))
_arm = None
_include_failed = False


def _init(cfg, include_failed=False):
    global _arm, _include_failed
    # Sweeps rewrite arm.cfg['sweep']; never alias the caller's configuration.
    _arm, _include_failed = build_arm(deepcopy(cfg)), include_failed


def run_sweep(arm, a, b, duration, include_failed=False):
    """One recorded A -> B sweep starting at rest at A; only sweep+hold loads count.

    By default a sweep counts only if it tracks within tolerance: commands the
    arm cannot follow end in joint-stop impacts, not motor-driven sweep loads."""
    s = arm.cfg['sweep']
    # Starting at A makes the approach trivial; settle still removes gravity sag.
    s.update(a_deg=[float(x) for x in a], b_deg=[float(x) for x in b],
             duration_s=float(duration), approach_s=.1)
    data = arm.data()
    run = Sweep(arm, data)
    rec = FEARecording(arm, data)
    rec.phases = ('sweep', 'hold')
    result = dict(a_deg=s['a_deg'], b_deg=s['b_deg'], duration_s=float(duration))
    try:
        while run.tick(arm, data, recorder=rec):
            pass
    except RuntimeError as exc:
        run.reason = str(exc)
    rec.stop(run.reason)
    s = summary(run)
    result.update(completed=bool(run.complete and rec.rows), reason=run.reason,
                  tracking_ok=s['tracking_ok'], samples=len(rec.rows),
                  max_tracking_error_deg=rec.max_tracking_error_deg,
                  saturated_samples=rec.saturated_samples, contact_samples=rec.contact_samples)
    result['counted'] = result['completed'] and (result['tracking_ok'] or include_failed)
    if not result['counted']:
        return result
    keep = sorted(set(rec.min_index.ravel().tolist()) | set(rec.max_index.ravel().tolist()))
    result.update(minimum=rec.minimum, maximum=rec.maximum, min_index=rec.min_index,
                  max_index=rec.max_index, extrema={i: rec.rows[i] for i in keep})
    return result


def _sweep(params):
    return [run_sweep(_arm, *params, _include_failed)]


def score(result, interface, metric):
    return result['maximum'][interface, metric] if result['counted'] else -np.inf


def _refine(job):
    """(1+1) evolution strategy on one interface/metric, from its best sweep so far."""
    a, b, duration = (np.array(job[k], float) for k in ('a_deg', 'b_deg', 'duration_s'))
    lo, hi = job['limits_deg'][:, 0], job['limits_deg'][:, 1]
    rng = np.random.default_rng(job['seed'])
    best, sigma, results = job['score'], .1, []
    for _ in range(job['iterations']):
        ta = np.clip(a + rng.normal(0, sigma, a.shape)*(hi-lo), lo, hi)
        tb = np.clip(b + rng.normal(0, sigma, b.shape)*(hi-lo), lo, hi)
        td = float(np.clip(duration*np.exp(rng.normal(0, sigma*3)), *job['duration_range']))
        result = run_sweep(_arm, ta, tb, td, _include_failed)
        results.append(result)
        value = score(result, job['interface'], job['metric'])
        if value > best:
            best, a, b, duration, sigma = value, ta, tb, td, min(sigma*1.5, .5)
        else:
            sigma = max(sigma*.85, .005)
    return results


class Peaks:
    """Running per-interface extrema across sweeps, keeping only the samples behind them."""
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
        meta.update(samples_meaning='Extrema load cases only; full histories are not retained.',
                    duration_s=None, campaign=self.settings,
                    recorded_samples=sum(s['samples'] for s in self.sweeps),
                    max_tracking_error_deg_note='Largest error of any sweep; short sweeps often saturate motors.')
        meta['limitations'] = meta['limitations'] + [
            'Sampled search over A/B poses and durations: the largest loads FOUND, not a proven global maximum.',
            'Only A -> B sweeps from rest, plus the final hold, are searched; jogging and Go to A/B are not.',
            'Unless include_failed is set, only sweeps that tracked within tolerance contribute loads.']
        return meta


def campaign(cfg, out, samples=200, iterations=20, duration_range=(.1, 5.), workers=None, seed=0,
             include_failed=False, log=print):
    lo_d, hi_d = duration_range
    if not .1 <= lo_d <= hi_d <= 120:
        raise ValueError('Durations must satisfy 0.1 <= minimum <= maximum <= 120 seconds.')
    if samples < 0 or iterations < 0:
        raise ValueError('Sample and iteration counts must be nonnegative.')
    workers = workers or multiprocessing.cpu_count()
    arm = build_arm(deepcopy(cfg))
    limits = np.array([j['limits_deg'] for j in cfg['joints']], float)
    rng = np.random.default_rng(seed)
    s = cfg['sweep']
    # Saved A/B in both directions first, then random poses with log-uniform durations.
    jobs = [(s['a_deg'], s['b_deg'], s['duration_s']), (s['b_deg'], s['a_deg'], s['duration_s'])]
    jobs += [(rng.uniform(limits[:, 0], limits[:, 1]), rng.uniform(limits[:, 0], limits[:, 1]),
              float(np.exp(rng.uniform(np.log(lo_d), np.log(hi_d))))) for _ in range(samples)]
    recorder = FEARecording(arm, arm.data())
    interfaces = recorder.interfaces
    peaks = Peaks((len(interfaces), len(METRICS)))
    sweeps, best = [], {}
    targets = [(i, j) for i, f in enumerate(interfaces) if f['sign'] == 1 for j in OBJECTIVES]
    started = time.monotonic()

    def collect(results, stage):
        for result in results:
            sweep_id = len(sweeps)
            peaks.add(sweep_id, result)
            for target in targets:
                value = score(result, *target)
                if value > best.get(target, (-np.inf,))[0]:
                    best[target] = (value, sweep_id)
            sweeps.append({k: v for k, v in result.items()
                           if k not in ('minimum', 'maximum', 'min_index', 'max_index', 'extrema')} | dict(stage=stage))
        log(f'{len(sweeps)} sweeps, {time.monotonic()-started:.0f} s', flush=True)

    def refine_jobs():
        result = []
        for n, ((i, j), (value, sweep_id)) in enumerate(sorted(best.items())):
            source = sweeps[sweep_id]
            result.append(dict(source, interface=i, metric=j, score=value, iterations=iterations,
                               limits_deg=limits, duration_range=(lo_d, hi_d), seed=[seed, n]))
        return result

    if workers == 1:
        _init(cfg, include_failed)
        for k, job in enumerate(jobs):
            collect(_sweep(job), 'saved' if k < 2 else 'random')
        if iterations:
            collect([r for job in refine_jobs() for r in _refine(job)], 'refine')
    else:
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(workers, context, _init, (cfg, include_failed)) as pool:
            for k, results in enumerate(pool.map(_sweep, jobs, chunksize=1)):
                collect(results, 'saved' if k < 2 else 'random')
            if iterations:
                for results in pool.map(_refine, refine_jobs()):
                    collect(results, 'refine')
    if not any(s['counted'] for s in sweeps):
        raise RuntimeError('No sweep tracked within tolerance; lengthen --min-duration or pass --include-failed.')
    settings = dict(samples=samples, refine_iterations=iterations, duration_range_s=[lo_d, hi_d],
                    seed=seed, sweeps=len(sweeps), completed=sum(s['completed'] for s in sweeps),
                    tracking_ok=sum(s['tracking_ok'] for s in sweeps), counted=sum(s['counted'] for s in sweeps),
                    include_failed=include_failed, tracking_tolerance_deg=cfg['sweep']['tracking_tolerance_deg'],
                    measured_phases=['sweep', 'hold'])
    rec = CampaignRecording(arm, peaks, sweeps, settings)
    folder = rec.export(out, 'cases')
    with (folder/'sweeps.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sweep', 'stage', 'duration_s', 'completed', 'tracking_ok', 'counted', 'max_tracking_error_deg',
                         'saturated_samples', 'reason', *(f'a_{j["name"]}_deg' for j in cfg['joints']),
                         *(f'b_{j["name"]}_deg' for j in cfg['joints'])])
        for i, s in enumerate(sweeps):
            writer.writerow([i, s['stage'], s['duration_s'], s['completed'], s['tracking_ok'],
                             s['counted'], s['max_tracking_error_deg'], s['saturated_samples'], s['reason'], *s['a_deg'], *s['b_deg']])
    (folder/'campaign.json').write_text(json.dumps(settings, indent=2))
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
                 for row in csv.DictReader(f)]
    return dict(folder=folder, cfg=cfg, settings=json.loads((folder/'campaign.json').read_text()),
                sweeps=sweeps, peaks=peaks)


def latest_results(root, joints):
    """Newest campaign export anywhere under root that was run on these joints."""
    for path in sorted(Path(root).glob('**/campaign.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        cfg = yaml.safe_load((path.parent/'config.yaml').read_text())
        if [j['name'] for j in cfg['joints']] == list(joints):
            return path.parent
    raise ValueError(f'No campaign for this robot under {root}. Run arm-sim campaign first.')

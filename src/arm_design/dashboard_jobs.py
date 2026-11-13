"""Durable dashboard job requests, isolated configuration and worker status."""
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

from .analysis import atomic_json
from .config import load_config, validate


def integer(value, label, minimum=1, maximum=None):
    if value is None or not math.isfinite(float(value)) or int(value) != float(value):
        raise ValueError(f'{label} must be a whole number.')
    value = int(value)
    if value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{label} must be at least {minimum}' + (f' and at most {maximum}.' if maximum else '.'))
    return value


def prepare(model, scenario, values):
    """Validate before creating folders or starting processes. Source YAML is read-only."""
    if scenario not in ('campaign','restrained'):
        raise ValueError('Choose Free motion or Restrained.')
    cfg = load_config(model)
    count = integer(values['count'], 'Maximum trials')
    workers = integer(values['workers'], 'Workers', maximum=16)
    seed = integer(values['seed'], 'Seed', minimum=0)
    inertia = values.get('inertia')
    payload = dict(mass_kg=float(values['mass']),com_offset_m=[float(v) for v in values['offset']],
                   box_size_m=[float(v) for v in values['size']],
                   inertia_com_kg_m2=json.loads(inertia) if inertia and inertia.strip() else None,
                   convention='Additional mass; COM offset from tip in tip-local XYZ; inertia about payload COM in those axes.')
    cfg['analysis_payload'] = payload
    cfg['analysis_run_name'] = str(values.get('name') or '').strip()[:80]
    validate(cfg)
    args = [scenario,'--workers',str(workers),'--seed',str(seed)]
    if values.get('full'):
        args += ['--full-history']
    poses = None
    if scenario == 'campaign':
        lo, hi = float(values['min_duration']), float(values['max_duration'])
        if not .1 <= lo <= hi <= 120:
            raise ValueError('Sweep durations must satisfy 0.1 ≤ minimum ≤ maximum ≤ 120 s.')
        args += ['--samples',str(count),'--iterations','0','--min-duration',str(lo),'--max-duration',str(hi),
                 '--skip-timestep-validation']
        if values['motion'] == 'aggressive':
            args += ['--aggressive']
        elif values['motion'] != 'random':
            raise ValueError('Choose an aggressive or random sweep campaign.')
        planned = count
    else:
        nposes = integer(values['pose_count'], 'Number of poses')
        duration, ramp = float(values['duration']), float(values['ramp'])
        if not .01 <= duration <= 120 or not 0 < ramp <= duration:
            raise ValueError('Restrained duration must be 0.01–120 s; torque ramp must be positive and no longer than the trial.')
        args += ['--poses',str(nposes),'--max-trials',str(count),'--trial-duration',str(duration),'--torque-ramp',str(ramp)]
        text = values.get('poses')
        if text and text.strip():
            poses = json.loads(text)
            q = np.asarray(poses,float)
            limits = np.array([j['limits_deg'] for j in cfg['joints']])
            if q.ndim != 2 or q.shape[1] != len(limits) or not len(q) or not np.isfinite(q).all() or np.any(q < limits[:,0]) or np.any(q > limits[:,1]):
                raise ValueError('Each supplied pose needs one finite, in-limit angle per joint.')
            nposes = len(q)
        planned = min(count,nposes*3**len(cfg['joints']))
    return dict(config=cfg,args=args,poses=poses,scenario=scenario,planned_trials=planned,
                max_trials=count,run_name=cfg['analysis_run_name'],payload=payload)


def save_request(folder, prepared):
    folder = Path(folder)
    folder.mkdir(parents=True)
    (folder/'config.yaml').write_text(yaml.safe_dump(prepared['config'],sort_keys=False))
    args = [*prepared['args'],'--config',str(folder/'config.yaml'),'--out',str(folder)]
    if prepared['poses'] is not None:
        atomic_json(folder/'poses.json',prepared['poses'])
        args += ['--poses-file',str(folder/'poses.json')]
    atomic_json(folder/'request.json',dict(prepared,args=args))
    atomic_json(folder/'job.json',dict(status='starting',scenario=prepared['scenario'],
        planned_trials=prepared['planned_trials'],run_name=prepared['run_name'],payload=prepared['payload']))


def job_state(folder):
    folder = Path(folder)
    try:
        state = json.loads((folder/'job.json').read_text())
    except (ValueError,OSError):
        return dict(status='unavailable')
    if state.get('status') == 'running' and state.get('pid'):
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            state.update(status='failed',error='Worker is no longer running. Completed trials are retained; see worker.log.')
        except PermissionError:
            pass
    runs = sorted(folder.glob('*/run.json'))
    if runs:
        state['run'] = str(runs[-1].parent)
    state['stop_requested'] = (folder/'CANCEL').exists()
    return state


def request_stop(folder):
    folder = Path(folder)
    (folder/'CANCEL').touch()
    for path in folder.glob('*/run.json'):
        (path.parent/'CANCEL').touch()


def worker(folder):
    from .cli import main
    folder = Path(folder)
    request = json.loads((folder/'request.json').read_text())
    state = job_state(folder)
    state.update(status='running',pid=os.getpid())
    atomic_json(folder/'job.json',state)
    code, error = 0, ''
    try:
        if not (folder/'CANCEL').exists():
            sys.argv = ['arm-sim',*request['args']]
            main()
    except SystemExit as exc:
        code = exc.code or 0
        error = f'Analysis exited with code {code}. See worker.log.' if code else ''
    except BaseException as exc:
        code, error = 1, str(exc)
        traceback.print_exc()
    finally:
        cancelled = (folder/'CANCEL').exists()
        if cancelled:
            code, error = 0, ''
        state.update(status='cancelled' if cancelled else 'failed' if code else 'finished',error=error)
        atomic_json(folder/'job.json',state)
    return code


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1]))

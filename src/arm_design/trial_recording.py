"""On-demand full recordings, verified against the original trial's statistics.

Re-simulation is explicit and never changes a run's measured results. A changed
solver/controller/model cannot silently substitute new loads for an old trial.
"""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

import mujoco
import numpy as np
import yaml

from .analysis import atomic_json
from .fea import jsonable
from .model import Arm

_lock = RLock()
_processes = {}


def detail_folder(path, trial_id):
    return Path(path)/'trials'/str(int(trial_id))/'detail'


def recording_status(path, trial_id):
    folder = detail_folder(path, trial_id)
    try:
        status = json.loads((folder/'recording.json').read_text())
    except FileNotFoundError:
        return {}
    if status.get('status') == 'running' and status.get('pid'):
        try:
            os.kill(status['pid'], 0)
        except ProcessLookupError:
            return dict(status, status='error', error='Recording process exited before completion; retry to regenerate.')
    return status


def trace_path(path, trial_id):
    folder = detail_folder(path, trial_id)
    if recording_status(path, trial_id).get('status') == 'complete':
        return folder/'trace.npz'
    return folder.parent/'trace.npz'


def start_recording(path, trial_id):
    path = Path(path).resolve()
    folder = detail_folder(path, trial_id)
    with _lock:
        # Avoid multiple expensive simulations from repeated clicks.
        if any(p.poll() is None for p in _processes.values()):
            return 'A full recording is already being generated. Please wait.'
        status = recording_status(path, trial_id).get('status')
        if status in ('complete', 'running'):
            return 'Full recording '+status+'.'
        folder.mkdir(exist_ok=True)
        with (folder/'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'arm_design.trial_recording', str(path), str(int(trial_id))],
                                    stdout=log, stderr=subprocess.STDOUT)
        _processes.clear()
        _processes[str(folder)] = proc
    return 'Generating and verifying a full recording…'


def regenerate(path, trial_id):
    path = Path(path).resolve()
    folder = detail_folder(path, trial_id)
    folder.mkdir(exist_ok=True)
    atomic_json(folder/'recording.json', dict(status='running', pid=os.getpid()))
    try:
        state = json.loads((path/'run.json').read_text())
        original = json.loads((folder.parent/'trial.json').read_text())
        if state['mujoco_version'] != mujoco.__version__:
            raise ValueError('Use the original MuJoCo version to regenerate this trial.')
        cfg = yaml.safe_load((path/'config.yaml').read_text())
        arm = Arm(cfg, mujoco.MjModel.from_binary_path(str(path/'model.mjb')),
                  (path/'source_model.xml').read_text(), {})
        settings = state['settings']
        from .collision import CollisionChecker
        if state['scenario'] == 'campaign':
            from . import campaign
            overlap = settings.get('max_overlap_mm', 0.)
            checker = None if overlap is None else CollisionChecker(arm, overlap)
            previous = campaign._full_history
            try:
                campaign._full_history = True
                result = campaign.run_sweep(arm, original['a_deg'], original['b_deg'], original['duration_s'],
                    include_failed=settings.get('include_failed', False), checker=checker,
                    strict=settings.get('campaign_mode') == 'aggressive')
            finally:
                campaign._full_history = previous
        elif state['scenario'] == 'restrained':
            from .restrained import trial
            result = trial(arm, np.deg2rad(original['q_deg']), original['signs'],
                           duration=original['duration_s'], ramp=settings['ramp'],
                           checker=CollisionChecker(arm), full_history=True,
                           position_tolerance=settings['position_tolerance'], angle_tolerance=settings['angle_tolerance'])
        else:
            raise ValueError('This run type does not support regeneration.')
        for key in ('samples', 'counted', 'completed', 'reason'):
            if result.get(key) != original.get(key):
                raise ValueError(f'Regenerated {key} differs from the original; full recording was not published.')
        stats = result.get('distribution')
        if stats is None:
            raise ValueError('This trial stopped before recording any physics samples.')
        with np.load(folder.parent/'distribution.npz', allow_pickle=False) as saved:
            for key in ('count', 'counts', 'minimum', 'maximum', 'total', 'squares'):
                matches = (np.array_equal(saved[key], stats[key]) if key in ('count', 'counts') else
                           np.allclose(saved[key], stats[key], rtol=1e-9, atol=1e-9))
                if not matches:
                    raise ValueError(f'Regenerated {key} differs from the original; full recording was not published.')
        # The status marker is committed last, so readers never use partial output.
        np.savez_compressed(folder/'trace.npz', **stats['trace'])
        with gzip.open(folder/'cases.json.gz', 'wt') as stream:
            json.dump(list(result['extrema'].values()), stream, default=jsonable)
        atomic_json(folder/'recording.json', dict(status='complete', regenerated=True, verified=True,
                    samples=result['samples'], mujoco_version=mujoco.__version__,
                    verification='Acceptance, sample count and histogram counts match; floating statistics agree within 1e-9.'))
        return folder
    except Exception as exc:
        atomic_json(folder/'recording.json', dict(status='error', error=str(exc)))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('trial', type=int)
    args = parser.parse_args()
    regenerate(args.run, args.trial)

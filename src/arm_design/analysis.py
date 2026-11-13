"""Durable per-trial distributions and live run state shared by both experiments."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from threading import RLock

import mujoco
import numpy as np
import yaml

from .fea import FEARecording, METRICS, jsonable


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, default=jsonable, allow_nan=False))
    tmp.replace(path)


def edges():
    # Shared signed bins; underflow/overflow are real bins, never silently dropped.
    positive = np.geomspace(.001, 1e6, 91)
    return np.r_[-np.inf, -positive[::-1], 0., positive, np.inf]


def metrics(loads):
    return np.concatenate((loads, np.linalg.norm(loads[..., :3], axis=-1)[..., None],
                           np.linalg.norm(loads[..., 3:], axis=-1)[..., None]), axis=-1)


def distribution(rows, full_history=False):
    if not rows:
        return None
    loads = np.array([r['loads_local'] for r in rows])
    v, bins = metrics(loads), edges()
    counts = np.empty((*v.shape[1:], len(bins)-1), dtype=np.int64)
    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            counts[i, j] = np.histogram(v[:, i, j], bins)[0]
    take = np.arange(len(rows)) if full_history else np.unique(np.linspace(0, len(rows)-1, min(400, len(rows))).astype(int))
    return dict(counts=counts, count=len(rows), total=v.sum(axis=0), squares=(v*v).sum(axis=0),
                minimum=v.min(axis=0), maximum=v.max(axis=0),
                trace=dict(time=np.array([rows[k]['time_s'] for k in take]), loads=loads[take],
                           q=np.array([rows[k]['q_rad'] for k in take]),
                           torque=np.array([rows[k]['actuator_torque_Nm'] for k in take]),
                           cap=np.array([rows[k]['torque_cap_Nm'] for k in take])),
                full_history=full_history)


def result_from_recording(rec, **info):
    keep = sorted(set(rec.min_index.ravel()) | set(rec.max_index.ravel())) if rec.rows else []
    return dict(completed=True, counted=bool(rec.rows), tracking_ok=True, samples=len(rec.rows),
                max_tracking_error_deg=rec.max_tracking_error_deg, saturated_samples=rec.saturated_samples,
                contact_samples=rec.contact_samples, reason='', minimum=rec.minimum, maximum=rec.maximum,
                min_index=rec.min_index, max_index=rec.max_index,
                extrema={int(i): rec.rows[i] for i in keep}, **info)


class RunStore:
    def __init__(self, out, arm, scenario, settings):
        settings = dict(settings, analysis_payload=arm.cfg.get('analysis_payload', {'mass_kg':0}),
                        run_name=arm.cfg.get('analysis_run_name',''))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
        self.path = Path(out).resolve()/f'{scenario}-{stamp}'
        (self.path/'trials').mkdir(parents=True)
        self.meta = FEARecording(arm, arm.data()).metadata()
        atomic_json(self.path/'metadata.json', self.meta)
        (self.path/'config.yaml').write_text(yaml.safe_dump(arm.cfg, sort_keys=False))
        mujoco.mj_saveModel(arm.model, str(self.path/'model.mjb'), None)
        (self.path/'source_model.xml').write_text(arm.xml)
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = 'unknown'
        self.state = dict(schema_version=1, scenario=scenario, status='running', settings=settings,
                          robot=arm.cfg['name'], revision=revision, mujoco_version=mujoco.__version__,
                          started_at=stamp, trials=0, counted=0, rejected=0, export=None,
                          interfaces=self.meta['interfaces'], history=[])
        self.started = time.monotonic()
        self.maximum = None
        with sqlite3.connect(self.path/'trials.sqlite') as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE trials (id INTEGER PRIMARY KEY, stage TEXT, counted INTEGER, info TEXT)')
        self.publish()
        print(f'Live analysis: {self.path}', flush=True)

    @property
    def cancelled(self):
        return (self.path/'CANCEL').exists() or (self.path.parent/'CANCEL').exists()

    def publish(self):
        self.state['elapsed_s'] = time.monotonic()-self.started
        atomic_json(self.path/'run.json', self.state)

    def add(self, trial_id, result, stage):
        folder = self.path/'trials'/str(trial_id)
        folder.mkdir()
        stats = result.get('distribution')
        if stats is not None:
            np.savez_compressed(folder/'distribution.npz', edges=edges(),
                                **{k: v for k, v in stats.items() if k not in ('trace', 'full_history')})
            np.savez_compressed(folder/'trace.npz', **stats['trace'])
        if result.get('extrema'):
            with gzip.open(folder/'cases.json.gz', 'wt') as f:
                json.dump(list(result['extrema'].values()), f, default=jsonable)
        omitted = {'distribution', 'extrema', 'minimum', 'maximum', 'min_index', 'max_index'}
        info = {k: v for k, v in result.items() if k not in omitted}
        if result.get('maximum') is not None and result.get('extrema'):
            info.update(minimum=result['minimum'], maximum=result['maximum'])
        info.update(id=trial_id, stage=stage, full_history=bool(stats and stats['full_history']))
        atomic_json(folder/'trial.json', info)
        with sqlite3.connect(self.path/'trials.sqlite') as db:
            db.execute('INSERT INTO trials VALUES (?, ?, ?, ?)',
                       (trial_id, stage, int(result['counted']), json.dumps(info, default=jsonable)))
        self.state['trials'] += 1
        self.state['counted'] += int(result['counted'])
        self.state['rejected'] += int(not result['counted'])
        if result['counted']:
            high = result['maximum']
            self.maximum = high.copy() if self.maximum is None else np.maximum(self.maximum, high)
            self.state['maximum'] = self.maximum
            self.state['history'].append(dict(trial=trial_id, elapsed_s=time.monotonic()-self.started,
                                              maximum=self.maximum[:, 6:].tolist()))
        if len(self.state['history']) > 2000:
            self.state['history'] = self.state['history'][::2] + self.state['history'][-1:]
        self.publish()

    def finish(self, status='complete', export=None, error=None):
        self.state.update(status=status, export=str(export) if export else None, error=error)
        self.publish()


_trial_rows = {}
_trial_lock = RLock()


def trials(path, stage='all', accepted='accepted'):
    dbpath = Path(path)/'trials.sqlite'
    if not dbpath.exists():
        return []
    # RunStore only appends committed trials. Parse each large JSON row once,
    # rather than repeatedly decoding a growing 20,000-trial database on polling.
    key = str(dbpath.resolve())
    stat = dbpath.stat()
    identity = (stat.st_dev, stat.st_ino)
    with _trial_lock:
        if key not in _trial_rows or _trial_rows[key][0] != identity:
            if len(_trial_rows) >= 16:
                _trial_rows.clear()
            _trial_rows[key] = (identity, [], -1)
        _, rows, last_id = _trial_rows[key]
        with sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True) as db:
            added = db.execute('SELECT id, info FROM trials WHERE id > ? ORDER BY id', (last_id,)).fetchall()
        rows.extend(json.loads(info) for _, info in added)
        if added:
            _trial_rows[key] = (identity, rows, added[-1][0])
        return [r for r in rows if (stage == 'all' or r['stage'] == stage)
                and (accepted == 'all' or bool(r['counted']) == (accepted == 'accepted'))]

"""Best-effort, wall-clock-throttled poses. Never wait for a renderer or alter physics."""
import json
import os
from pathlib import Path
import time
from threading import Lock

import numpy as np

from .analysis import atomic_json

INTERVAL = .2
WATCH_SECONDS = 3.
_watch_lock = Lock()


class LivePublisher:
    def __init__(self, folder=None, owner=None):
        self.folder = Path(folder) if folder else None
        self.pid = os.getpid()
        self.selected = bool(folder)
        if self.selected and owner is not None:
            with owner.get_lock():
                if not owner.value:
                    owner.value = self.pid
                self.selected = owner.value == self.pid
        self.next_sample = 0.
        self.sequence = 0
        self.context = {}

    def begin(self, scenario, duration):
        self.sequence += 1
        self.context = dict(worker=self.pid,segment=self.sequence,scenario=scenario,duration_s=duration)

    def sample(self, data, phase):
        if not self.selected:
            return
        now = time.monotonic()
        if now < self.next_sample:
            return
        self.next_sample = now+INTERVAL
        try:
            watch = json.loads((self.folder/'live-watch.json').read_text())
            wall = time.time()
            if watch.get('expires_at',0) < wall or not np.isfinite(data.qpos).all():
                return
            atomic_json(self.folder/'live-pose.json',dict(self.context,phase=phase,
                time_s=float(data.time),published_at=wall,q_rad=data.qpos.tolist()))
        except (OSError,ValueError):
            # Preview failures must never reject a trial or stop a campaign.
            pass


def watched_pose(folder, running):
    folder = Path(folder)
    if running:
        # Each open, visible preview renews this lease; closing/hiding all views
        # automatically stops pose writes within three seconds.
        with _watch_lock:
            path = folder/'live-watch.json'
            try:
                expiry = json.loads(path.read_text()).get('expires_at',0)
            except (OSError,ValueError):
                expiry = 0
            if expiry < time.time()+1.5:
                atomic_json(path,dict(expires_at=time.time()+WATCH_SECONDS))
    try:
        return json.loads((folder/'live-pose.json').read_text())
    except (OSError,ValueError):
        return None

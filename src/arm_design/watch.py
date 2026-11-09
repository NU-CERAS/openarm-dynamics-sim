"""Live plot of a campaign's running peak loads, one point per PROGRESS_EVERY counted sweeps."""
import csv
from pathlib import Path

import numpy as np

from .campaign import PROGRESS, PROGRESS_EVERY


def find(path):
    """A progress/convergence CSV: the file itself, or the newest one under a folder."""
    path = Path(path)
    if path.suffix == '.csv':
        return path
    found = [p for name in (PROGRESS, 'convergence.csv') for p in path.glob('**/'+name)]
    if not found:
        raise ValueError(f'No {PROGRESS} or convergence.csv under {path}. Run arm-sim campaign first.')
    return max(found, key=lambda p: p.stat().st_mtime)


def read(path):
    """(columns, numeric rows, stages); a half-written last line is dropped."""
    with Path(path).open(newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], np.zeros((0, 0)), []
    head, values, stages = rows[0], [], []
    for r in rows[1:]:
        try:
            values.append([float(x) for k, x in enumerate(r) if k != 3])
        except ValueError:
            continue
        if len(r) == len(head):
            stages.append(r[3])
        else:
            values.pop()
    return head, np.array(values).reshape(len(values), len(head)-1), stages


def rise_pct(values):
    """Largest percent increase of any series between consecutive rows."""
    prev, now = values[:-1], values[1:]
    rel = np.divide(now-prev, prev, out=np.zeros_like(now), where=prev > 1e-9)
    return 100*rel.max(axis=1, initial=0)


def watch(path, interval_s=2.):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9), gridspec_kw=dict(height_ratios=(3, 3, 2)))
    fig.subplots_adjust(right=.78, hspace=.12)

    def update(_=None):
        try:
            head, v, stages = read(path)
        except OSError:
            head, v, stages = [], np.zeros((0, 0)), []
        for ax in axes:
            ax.cla()
            ax.grid(alpha=.3)
        if not len(v):
            axes[0].set_title(f'Waiting for {path} (first row after {PROGRESS_EVERY} counted sweeps)…', fontsize=10)
            return
        x, total, elapsed = v[:, 0], v[-1, 1], v[-1, 2]
        cols = head[4:]
        loads = v[:, 3:]
        labels = [c.split('|')[0] for c in cols if c.endswith('force_N')]
        colors = plt.cm.turbo(np.linspace(.05, .95, len(labels)))
        for ax, metric, unit in ((axes[0], 'force_N', 'Peak force (N)'), (axes[1], 'moment_Nm', 'Peak moment (Nm)')):
            series = loads[:, [k for k, c in enumerate(cols) if c.endswith('|'+metric)]]
            for k, label in enumerate(labels):
                ax.plot(x, series[:, k], '.-', ms=3, lw=1, color=colors[k], label=label)
            ax.set_ylabel(unit)
            if len(x) > 1:
                axes[2].plot(x[1:], rise_pct(series), '.-', label=metric.split('_')[0])
        axes[2].set_ylabel('Largest rise\nsince last point (%)')
        axes[2].set_ylim(bottom=0)
        axes[2].set_xlabel('Counted sweeps')
        if len(x) > 1:
            axes[2].legend(fontsize=8, loc='upper right')
        refine = next((x[k] for k, s in enumerate(stages) if s == 'refine'), None)
        for ax in axes:
            if refine is not None:
                ax.axvline(refine, color='k', ls=':', lw=1)
        if refine is not None:
            axes[0].text(refine, axes[0].get_ylim()[1], ' refine →', va='top', fontsize=8)
        axes[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1), title='interface', title_fontsize=8)
        axes[0].set_title(f'{Path(path).parent.name}/{Path(path).name}: {int(x[-1])} counted of {int(total)} sweeps, '
                          f'{elapsed:.0f} s, stage {stages[-1]}', fontsize=10)

    update()
    anim = FuncAnimation(fig, update, interval=interval_s*1000, cache_frame_data=False)
    plt.show()
    return anim

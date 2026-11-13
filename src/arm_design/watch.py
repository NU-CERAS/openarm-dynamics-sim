"""Live plot of a campaign's running peak loads and the full wrench behind each, one point per PROGRESS_EVERY counted sweeps."""
import csv
from pathlib import Path

import numpy as np

from .campaign import PROGRESS, PROGRESS_EVERY
from .fea import METRICS

# One hue per part (fixed order); the row kind picks the line style.
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
STYLES = dict(proximal='-', distal='--', delta=':')


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


def series(head):
    """Reported rows in file order: label, part number, kind, and per magnitude its peak column
    and six record-state columns (None for CSVs written before components were logged)."""
    cols = head[4:]
    at = {c: k for k, c in enumerate(cols)}
    result, parts = [], {}
    for label in dict.fromkeys(c.split('|')[0] for c in cols):
        part = label.removesuffix(' delta').split(' <- ')[0]
        kind = 'delta' if label.endswith(' delta') else 'distal' if part in parts else 'proximal'
        row = dict(label=label, part=parts.setdefault(part, len(parts)), kind=kind)
        for metric in METRICS[6:]:
            state = [at.get(f'{label}|{metric}|{c}') for c in METRICS[:6]]
            row[metric] = (at[f'{label}|{metric}'], None if None in state else state)
        result.append(row)
    return result


def table(ax, rows, loads, metric):
    """Each row's peak and the Fx..Mz of the sample that set it; the line samples double as the legend."""
    ax.cla()
    ax.axis('off')
    n = len(rows)
    ax.set_xlim(0, 1)
    ax.set_ylim(n+.6, -.7)
    size = float(np.clip(170/(n+1), 4, 8))
    x = .43+.08*np.arange(7)
    unit = 'N' if metric == 'force_N' else 'Nm'
    ax.set_title(f'Record state at each peak {metric.split("_")[0]}: wrench of the sample that set it (part frame, N / Nm)',
                 fontsize=9, loc='left')
    for k, name in enumerate((f'peak {unit}', *(c.split('_')[0] for c in METRICS[:6]))):
        ax.text(x[k], 0, name, ha='right', va='center', fontsize=size, weight='bold')
    ax.text(.06, 0, 'mate / delta', va='center', fontsize=size, weight='bold')
    for r, row in enumerate(rows, 1):
        if r % 2:
            ax.axhspan(r-.5, r+.5, color='#000000', alpha=.05, lw=0)
        ax.plot((.005, .05), (r, r), color=PALETTE[row['part'] % len(PALETTE)], ls=STYLES[row['kind']], lw=2)
        ax.text(.06, r, row['label'], va='center', fontsize=size)
        peak, state = row[metric]
        ax.text(x[0], r, f'{loads[-1, peak]:.2f}', ha='right', va='center', fontsize=size, weight='bold')
        for k, column in enumerate(state or ()):
            ax.text(x[k+1], r, f'{loads[-1, column]:.2f}', ha='right', va='center', fontsize=size)
    if rows and rows[0][metric][1] is None:
        ax.text(x[1], 1, 'no Fx..Mz columns: CSV from an older campaign', va='center', fontsize=size, style='italic')


def figure():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(17, 9.5))
    fig.legend([Line2D([], [], color='.3', ls=ls) for ls in STYLES.values()],
               ['proximal mate (toward the base)', 'distal mate', 'delta (sum of mates about the proximal one)'],
               loc='lower right', ncols=3, fontsize=8, frameon=False, title='One colour per part; line style:', title_fontsize=8)
    grid = fig.add_gridspec(1, 2, width_ratios=(9, 10), left=.05, right=.99, top=.95, bottom=.06, wspace=.06)
    left, right = grid[0].subgridspec(3, 1, height_ratios=(3, 3, 2), hspace=.12), grid[1].subgridspec(2, 1, hspace=.1)
    axes = [fig.add_subplot(left[0])]
    axes += [fig.add_subplot(left[k], sharex=axes[0]) for k in (1, 2)]
    return fig, axes, [fig.add_subplot(right[k]) for k in (0, 1)]


def draw(path, axes, tables):
    try:
        head, v, stages = read(path)
    except OSError:
        head, v, stages = [], np.zeros((0, 0)), []
    for ax in axes:
        ax.cla()
        ax.grid(alpha=.3)
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    if not len(v):
        for ax in tables:
            ax.cla()
            ax.axis('off')
        axes[0].set_title(f'Waiting for {path} (first row after {PROGRESS_EVERY} counted sweeps)...', fontsize=10)
        return
    x, total, elapsed = v[:, 0], v[-1, 1], v[-1, 2]
    loads, rows = v[:, 3:], series(head)
    for ax, panel, metric, unit in zip(axes, tables, METRICS[6:], ('Peak force (N)', 'Peak moment (Nm)')):
        for row in rows:
            ax.plot(x, loads[:, row[metric][0]], '.', ls=STYLES[row['kind']], ms=3, lw=1.3,
                    color=PALETTE[row['part'] % len(PALETTE)])
        ax.set_ylabel(unit)
        if len(x) > 1:
            axes[2].plot(x[1:], rise_pct(loads[:, [row[metric][0] for row in rows]]), '.-',
                         color='.15' if metric == 'force_N' else '.6', label=metric.split('_')[0])
        table(panel, rows, loads, metric)
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
    axes[0].set_title(f'{Path(path).parent.name}/{Path(path).name}: {int(x[-1])} counted of {int(total)} sweeps, '
                      f'{elapsed:.0f} s, stage {stages[-1]}', fontsize=10)


def watch(path, interval_s=2.):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, axes, tables = figure()
    draw(path, axes, tables)
    anim = FuncAnimation(fig, lambda _: draw(path, axes, tables), interval=interval_s*1000, cache_frame_data=False)
    plt.show()
    return anim

"""Local linked analysis dashboard. Physics jobs run in independent CLI processes."""
from functools import lru_cache
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

import numpy as np
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update
from flask import request, jsonify, abort

from .analysis import trials, edges, metrics
from .config import ROOT
from .fea import METRICS
from .fea_view import mesh_figure
from .load_histograms import trial_extreme, histogram_counts, percentile_text, preview_svg, split_histogram
from .trial_playback import playback, model_geometry
from .trial_recording import trace_path, recording_status, start_recording
from .dashboard_jobs import prepare, save_request, job_state, request_stop
from .payload import default_payload_offset

COLORS = ['#25c9a3', '#ffb45b', '#65a9ff', '#ed81c5', '#b2a0ff', '#ef7176', '#9dc66b', '#79d5e8']

RUN_FIELDS = ['model','run-name','payload-mass','free-count','restrained-count','motion-mode',
    'min-duration','max-duration','pose-count','duration','ramp','payload-x','payload-y','payload-z',
    'payload-size-x','payload-size-y','payload-size-z','payload-inertia','seed','workers','poses','full']


def runner_form(model, scenario, values):
    """Save exactly the submitted form so switching modes restores active settings."""
    form = dict(zip(RUN_FIELDS,[str(model),values['name'],values['mass'],
        no_update,no_update,values['motion'],values['min_duration'],values['max_duration'],values['pose_count'],
        values['duration'],values['ramp'],*values['offset'],*values['size'],values['inertia'],values['seed'],
        values['workers'],values['poses'],values['full']]))
    form['restrained-count' if scenario == 'restrained' else 'free-count'] = values['count']
    return {k:v for k,v in form.items() if v is not no_update}


def runner_figures(path, metric):
    """Compact, exact record changes from committed trials, never chart-decimated maxima."""
    state = run_info(path)
    refs = state.get('interfaces',[])
    stage = 'enumeration' if state['scenario'] == 'restrained' else 'random'
    rows = trials(path,stage,'all')
    accepted = [r for r in rows if r['counted'] and 'minimum' in r and 'maximum' in r]
    latest = rows[-1] if rows else None
    j = METRICS.index(metric)
    figure = blank(f'Running signed extremes · {metric.replace("_"," ")}')
    table = []
    names = list(dict.fromkeys(part_name(r) for _,r in connection_rows(refs)))
    x = np.array([r['id']+1 for r in accepted])
    for n, part in enumerate(names):
        mates = connection_rows(refs,part)
        i, ref = mates[1] if len(mates)>1 else mates[0]
        color = COLORS[n % len(COLORS)]
        record = dict(part=part,connection=ref['label'],color=color,minimum=None,maximum=None,
                      latest_minimum=None,latest_maximum=None)
        if latest and 'minimum' in latest and 'maximum' in latest:
            record.update(latest_minimum=latest['minimum'][i][j],latest_maximum=latest['maximum'][i][j])
        for side, accumulate, dash in [('maximum',np.maximum.accumulate,'solid'),('minimum',np.minimum.accumulate,'dash')]:
            if not accepted:
                continue
            y = accumulate([r[side][i][j] for r in accepted])
            record[side] = float(y[-1])
            keep = np.unique(np.r_[0,np.flatnonzero(np.diff(y))+1,len(y)-1])
            figure.add_scatter(x=x[keep],y=y[keep],mode='lines+markers',marker_size=3,
                line=dict(color=color,dash=dash,shape='hv'),name=part,legendgroup=part,showlegend=side=='maximum',
                hovertemplate=f'{ref["label"]}<br>Trial %{{x}}<br>Running {side}: %{{y:+.5g}}<extra></extra>')
        table.append(record)
    if not accepted:
        figure.add_annotation(text='Waiting for accepted trials…',x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    figure.update_layout(height=340,margin=dict(l=65,r=20,t=95,b=45),
        legend=dict(orientation='h',y=1.04,yanchor='bottom',groupclick='togglegroup'),
        xaxis_title='Attempted trial number',yaxis_title='N' if metric.startswith('F') else 'Nm',
        uirevision=str(path)+metric)
    figure.update_yaxes(zeroline=True,zerolinecolor='#96aec3',zerolinewidth=1)
    return figure,table,latest


def runner_table(records, previous):
    previous = previous or {}
    def cell(record, field):
        value = record[field]
        old = previous.get(record['part'],{}).get(field)
        changed = old is not None and value is not None and (
            field == 'minimum' and value < old or field == 'maximum' and value > old)
        return html.Td('—' if value is None else f'{value:+.5g}',className='new-record' if changed else '')
    return html.Table([
        html.Thead(html.Tr([html.Th(t) for t in ['Link / FEA connection','Latest trial min','Latest trial max','Overall min','Overall max']])),
        html.Tbody([html.Tr([html.Th(r['part'],title=r['connection'],style={'color':r['color']}),
            *[cell(r,f) for f in ['latest_minimum','latest_maximum','minimum','maximum']]]) for r in records]),
    ])


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def discover(root):
    found = []
    for directory, children, files in os.walk(root):
        if 'run.json' in files or ('metadata.json' in files and 'peaks.csv' in files):
            found.append(Path(directory))
            children[:] = []  # Trial files and nested exports belong to this run.
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def run_version(path):
    for name in ('run.json', 'metadata.json'):
        p = Path(path)/name
        if p.exists():
            return [str(path), str(p.stat().st_mtime_ns)]
    return [str(path), None]


def run_info(path):
    return _run_info(*run_version(path))


@lru_cache(maxsize=32)
def _run_info(path, version):
    value = read_json(Path(path)/'run.json')
    if value:
        return value
    meta = read_json(Path(path)/'metadata.json', {})
    return dict(status='legacy', scenario='campaign' if (Path(path)/'campaign.json').exists() else 'bounds',
                robot='Legacy export', interfaces=meta.get('interfaces', []), history=[],
                export=str(path), counted=meta.get('samples', 0), trials=0,
                settings={'note': 'Older export: no load-frequency dataset.'})


def blank(title, message=None):
    f = go.Figure()
    f.update_layout(template='plotly_dark', paper_bgcolor='#101b29', plot_bgcolor='#101b29',
                    title=dict(text=title, font_size=15), margin=dict(l=55, r=20, t=45, b=60),
                    font=dict(family='Arial', color='#cbd8e5'), height=260, autosize=True, uirevision=title)
    if message:
        f.add_annotation(text=message, x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
    return f


@lru_cache(maxsize=16)
def load_npz(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


_aggregates = {}
_aggregate_lock = RLock()


def aggregate(path, stage, accepted):
    # Dash serves polling and interaction callbacks concurrently. A trial must
    # contribute once, even when both callbacks request the same distribution.
    with _aggregate_lock:
        stats, rows = _aggregate(path, stage, accepted)
        return dict(stats), rows


def _aggregate(path, stage, accepted):
    rows = trials(path, stage, accepted)
    key = (str(path), stage, accepted)
    if key not in _aggregates:
        if len(_aggregates) > 32:
            _aggregates.clear()
        _aggregates[key] = dict(seen=set(), counts=None, count=0, total=None, squares=None, maximum=None, minimum=None)
    a = _aggregates[key]
    for r in rows:
        if r['id'] in a['seen']:
            continue
        p = Path(path)/'trials'/str(r['id'])/'distribution.npz'
        a['seen'].add(r['id'])
        if not p.exists():
            continue
        v = load_npz(str(p))
        for field in ('counts', 'total', 'squares'):
            a[field] = v[field].copy() if a[field] is None else a[field]+v[field]
        a['minimum'] = v['minimum'].copy() if a['minimum'] is None else np.minimum(a['minimum'], v['minimum'])
        a['maximum'] = v['maximum'].copy() if a['maximum'] is None else np.maximum(a['maximum'], v['maximum'])
        a['count'] += int(v['count'])
    return a, rows


def bin_labels():
    e = edges()
    return [f'{a:.3g} … {b:.3g}' for a, b in zip(e[:-1], e[1:])]


def connection_rows(refs, part=None):
    # Preserve stored indices when viewing older runs with extra aggregate rows.
    return [(i, r) for i, r in enumerate(refs) if r.get('kind', 'mate') == 'mate' and (part is None or part_name(r) == part)]


def part_name(ref):
    return ref.get('part') or ref['label'].split(' <- ')[0]


def scoped_interface(refs, part, interface):
    indices = [i for i, _ in connection_rows(refs, part)]
    return interface if interface in indices else (indices[0] if indices else None)


def fea_limits(path, part, stage, accepted, weighting="time"):
    stats, rows = aggregate(path, stage, accepted)
    mates = connection_rows(run_info(path).get('interfaces', []), part)
    if not stats['count'] or not mates:
        return html.P('No recorded loads for this selection.', className='note'), [], {}
    # Proximal support assumption: display the load at the other mounting point.
    i, ref = mates[1] if len(mates) > 1 else mates[0]
    support = mates[0][1]['label'].split(' <- ')[-1]
    role = 'Apply at '+ref['label'].split(' <- ')[-1] if len(mates)>1 else 'Support reaction only'
    options, candidates, cells = [], {}, []
    valid = [r for r in rows if 'minimum' in r and 'maximum' in r]
    for j, metric in enumerate(METRICS[:6]):
        low, high = stats['minimum'][i,j], stats['maximum'][i,j]
        values = []
        for side, value in [('minimum',low),('maximum',high)]:
            counts = histogram_counts(stats,rows,i,j,weighting,side)
            weight_label = ('Simulated time · % of physics samples' if weighting == 'time' else f'One {side} per trial · % of trials')
            preview = dict(title=f"{metric.replace('_',' ')} · {ref['label']}",image=preview_svg(counts,value),
                summary=percentile_text(counts),weight=weight_label,
                scope='Test coverage, not real-world occurrence frequency.',
                path=str(path),part=part,interface=i,metric=metric,side=side,value=float(value))
            values.append(html.Td(html.Button(f'{value:+.5g}',className='load-value',type='button',
                **{'data-load-preview':json.dumps(preview),'aria-label':f'{metric} {side} {value:+.5g}; preview distribution or click to open histogram'})))
        cells.append(html.Tr([html.Th(metric.replace('_',' ')),*values]))
        for side, field, select in [('minimum','minimum',min),('maximum','maximum',max)]:
            if not valid:
                continue
            row = select(valid,key=lambda r:r[field][i][j])
            value = row[field][i][j]
            key = f'{part}|{i}|{metric}|{side}'
            options.append(dict(label=f'{metric.replace("_", " ")} · {"minimum" if side=="minimum" else "maximum"} {value:+.5g}',value=key))
            candidates[key] = dict(trial=row['id'], interface=i, metric=j, side=side, target=value)
    note = (f'Assumed support: {support}. Limits at the loaded connection; each minimum/maximum is a separate case. '
            'Independent extrema are not one simultaneous load case. Values act ON this link in its local XYZ frame.')
    if len(mates)==1:
        note = (f'Only one mounting connection: {support}. These limits are support reactions, not loads to apply to a fixed face. '
                'External/contact and body loads are available in the recorded FEA exports.')
    table = html.Div([html.P(role,className='fea-mate'),
        html.Table([html.Thead(html.Tr([html.Th(x) for x in ['Component','Minimum','Maximum']])),html.Tbody(cells)]),
        html.P(note,className='note')],className='link-summary')
    return table, options, candidates


def link_summary(path, part, stage, accepted, weighting="time"):
    return fea_limits(path, part, stage, accepted, weighting)[0]




def signed_figures(path, part, stage, accepted):
    plots = [blank(f'{part} · signed forces (N)'), blank(f'{part} · signed moments (Nm)')]
    stats, _ = aggregate(path, stage, accepted)
    if not stats['count'] or stats.get('minimum') is None:
        return plots
    mates = connection_rows(run_info(path).get('interfaces', []), part)
    for offset, f in zip((0, 3), plots):
        labels = [f"{ref['label']} · {METRICS[j].split('_')[0]}" for _,ref in mates for j in range(offset, offset+3)]
        for field, name, color in [('minimum','Observed minimum',COLORS[1]), ('maximum','Observed maximum',COLORS[0])]:
            values = [stats[field][i,j] for i,_ in mates for j in range(offset, offset+3)]
            f.add_bar(y=labels, x=values, orientation='h', name=name, marker_color=color,
                      text=[f'{v:+.3g}' for v in values], textposition='auto',
                      hovertemplate='%{y}<br>%{x:+.5g}<extra>%{fullData.name}</extra>')
        bound = max((abs(v) for trace in f.data for v in trace.x), default=1) or 1
        f.update_layout(barmode='group', margin=dict(l=175, r=25, t=65, b=35),
                        legend=dict(orientation='h', y=1.02, yanchor='bottom', font_size=9),
                        yaxis=dict(autorange='reversed', tickfont_size=10))
        f.update_xaxes(range=[-1.15*bound,1.15*bound], zeroline=True, zerolinewidth=2, zerolinecolor='#e5edf5')
    return plots


def link_snapshots(path, trial_id, part):
    samples = snapshots(path, trial_id)
    indices = [i for i, _ in connection_rows(run_info(path).get('interfaces', []), part)]
    if not samples or not indices:
        return []
    values = metrics(np.asarray([s['loads_local'] for s in samples]))[:, indices, :]
    # Retain the first occurrence of each component/norm extreme for this link.
    chosen = set(np.argmin(values, axis=0).ravel()) | set(np.argmax(values, axis=0).ravel())
    return [sample for i, sample in enumerate(samples) if i in chosen]


def figures(path, other, interface, metric, stage, accepted, weighting, part=None, extreme='absolute', focus=None):
    state = run_info(path)
    refs = state.get('interfaces', [])
    mates = connection_rows(refs, part)
    if not mates:
        return [blank(x, 'No recorded data yet') for x in ('Distribution', 'Running maximum', 'Link comparison')], [], ''
    indices = [i for i, _ in mates]
    i = int(interface or 0)
    if i not in indices:
        i = indices[0]
    j = METRICS.index(metric)
    hist, convergence, links = blank(f"Distribution · {refs[i]['label']}"), blank('Running maxima'), blank(f'{part or 'Link'} · connection peaks')
    stats, rows = aggregate(path, stage, accepted)
    description = 'No distributions in this export. New runs record every physics sample.'
    if stats['count']:
        counts = histogram_counts(stats,rows,i,j,weighting,extreme)
        used = np.flatnonzero(counts)
        if len(used):
            lo, hi = max(0, used[0]-1), min(len(counts), used[-1]+2)
            ids = np.arange(lo, hi)
            hist.add_bar(x=ids, y=100*counts[ids]/max(1, counts.sum()), name='Selected run',
                         customdata=ids, marker_color=COLORS[0])
            hist.update_xaxes(tickmode='array', tickvals=ids[::max(1, len(ids)//8)],
                              ticktext=[bin_labels()[k] for k in ids[::max(1, len(ids)//8)]], title=metric)
        hist.update_yaxes(title='% of trial peaks' if weighting == 'trials' else '% of simulated time')
        mean = stats['total'][i, j]/stats['count']
        rms = np.sqrt(stats['squares'][i, j]/stats['count'])
        cumulative = np.cumsum(stats['counts'][i, j])
        quantile_bin = min(len(edges())-2, np.searchsorted(cumulative, .95*stats['count']))
        description = (f"{refs[i]['label']} · {len(rows)} selected trials · {stats['count']:,} physics samples · mean {mean:.3g} · "
                       f"RMS {rms:.3g} · P95 bin {bin_labels()[quantile_bin]} · part-local axes. "
                       + ('Aggressive duration-search frequency is test coverage, not an operating duty cycle.' if state.get('settings', {}).get('campaign_mode') == 'aggressive' else 'Restrained torque-combination coverage, not a duty cycle. The tip is welded; internal joints can move. Peaks depend on the assumed clamp stiffness.' if state.get('scenario') == 'restrained' else 'Random exploration is not a measured drummer duty cycle. Trial-peak frequency is test coverage.'))
        if j < 6:
            links.add_bar(x=[r['label'] for _, r in mates], y=stats['minimum'][indices, j], marker_color=COLORS[1], name='Minimum')
        links.add_bar(x=[r['label'] for _, r in mates], y=stats['maximum'][indices, j], marker_color=COLORS[0], name='Maximum')
        links.update_yaxes(title=metric, zeroline=True, zerolinecolor='#e5edf5')
        links.update_layout(barmode='group', legend=dict(orientation='h', y=1.02, yanchor='bottom'))
    else:
        hist.add_annotation(text=description, x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
        export = state.get('export')
        if export and (Path(export)/'peaks.csv').exists():
            peaks = list(csv.DictReader((Path(export)/'peaks.csv').open()))
            labels = {r['label'] for _, r in mates}
            peaks = [p for p in peaks if p['metric'] == metric and p['interface'] in labels]
            links.add_bar(x=[p['interface'] for p in peaks], y=[float(p['maximum']) for p in peaks])
    # Both charts follow the selected metric; magnitude history uses its own column.
    h = state.get('history', [])
    if j >= 6:
        name, unit = ('Force', 'N') if j == 6 else ('Moment', 'Nm')
        convergence = blank(f'Running maximum · {name.lower()} magnitude')
        for color_index, ci in enumerate(indices if part is not None else [i]):
            convergence.add_scatter(x=[v['trial']+1 for v in h], y=[v['maximum'][ci][j-6] for v in h],
                                    name=refs[ci]['label'], mode='lines', line=dict(color=COLORS[color_index]))
        convergence.update_layout(yaxis_title=unit, xaxis_title='Trial number · all accepted stages',
                                  legend=dict(orientation='h', x=0, y=1.02, yanchor='bottom', font_size=9),
                                  margin=dict(l=55, r=20, t=90, b=55))
    if j < 6:
        convergence = blank(f'Running signed extremes · {metric}')
        ordered = [r for r in rows if 'minimum' in r and 'maximum' in r]
        for color_index, ci in enumerate(indices):
            for field, reducer, dash in [('minimum', np.minimum.accumulate, 'dash'), ('maximum', np.maximum.accumulate, 'solid')]:
                values = reducer([r[field][ci][j] for r in ordered]) if ordered else []
                convergence.add_scatter(x=[r['id']+1 for r in ordered], y=values, mode='lines',
                    name=f"{refs[ci]['label']} · {field}", line=dict(color=COLORS[color_index], dash=dash))
        convergence.update_layout(yaxis_title=metric, xaxis_title='Trial number · current stage/validity filters',
            legend=dict(orientation='h', y=1.02, yanchor='bottom', font_size=9), margin=dict(l=55,r=20,t=90,b=55))
        convergence.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#e5edf5')
    if j < 6 and weighting == 'trials':
        description += ' Trial statistic: '+{'minimum':'most negative value', 'maximum':'most positive value', 'absolute':'largest absolute value, preserving its sign'}[extreme]+'.'
    if other and other != path:
        other_state = run_info(other)
        names = [r['label'] for r in other_state.get('interfaces', [])]
        if refs[i]['label'] in names and other_state.get('robot') == state.get('robot'):
            oi = names.index(refs[i]['label'])
            os, orows = aggregate(other, stage, accepted)
            if os['count']:
                oc = histogram_counts(os,orows,oi,j,weighting,extreme)
                idx = np.flatnonzero(oc)
                hist.add_bar(x=idx, y=100*oc[idx]/max(1, oc.sum()), customdata=idx,
                             marker_color=COLORS[1], name='Compared run')
                hist.update_layout(barmode='overlay')
        else:
            description += ' Comparison requires the same robot and connection label.'
    if hist.data:
        occupied = sorted({int(x) for series in hist.data for x in series.x})
        ticks = occupied[::max(1, len(occupied)//8)]
        hist.update_xaxes(tickmode='array', tickvals=ticks, ticktext=[bin_labels()[k] for k in ticks])
        hist.update_traces(opacity=.7 if len(hist.data)>1 else 1)
        hist.update_layout(legend=dict(orientation='h', x=0, y=1.12, font_size=10), margin_t=65)
    selected = focus if focus and focus.get('path') == str(path) and focus.get('part') == part and focus.get('metric') == metric and focus.get('interface') == i else None
    if selected and stats['count']:
        selected = dict(selected,value=float(stats[selected['side']][i,j]))
    hist = split_histogram(hist,metric,selected)
    links.update_xaxes(tickangle=-45)
    return [hist, convergence, links], rows, description


def snapshots(path, trial_id=None):
    if trial_id is not None:
        p = trace_path(path, trial_id).parent/'cases.json.gz'
        try:
            with gzip.open(p, 'rt') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    export = run_info(path).get('export')
    return read_json(Path(export)/'load_cases.json', {}).get('samples', []) if export else []


def pose_figure(meta, sample, interface=None, part=None, run_path=None):
    if part is None:
        part = part_name(meta['interfaces'][int(interface or 0)])
    return mesh_figure(meta, sample, part, run_path)


def run_controls(models):
    from .config import load_config
    offset = default_payload_offset(load_config(models[0]))
    return html.Section([
        html.H2('Test settings'),
        html.Fieldset([
        html.Label('Robot'), dcc.Dropdown(id='model',options=[{'label':p.parent.parent.name if 'robots' in p.parts else 'Built-in arm','value':str(p)} for p in models],value=str(models[0]),clearable=False),
        html.Label('Run name (optional)'),dcc.Input(id='run-name',placeholder='e.g. 0.5 kg drumstick test',persistence=True),
        html.Label('Additional payload mass (kg)'),dcc.Input(id='payload-mass',type='number',min=0,value=5,step=.05,persistence='runner-defaults-v2'),
        html.P('Extra mass at the end effector; 0 keeps the existing mass.',title='A fixed payload box supplies mass, inertia and collision geometry.',className='note'),
        html.Div([
            html.Label('Maximum trials · free motion'),dcc.Input(id='free-count',type='number',min=1,step=1,value=1000,persistence='runner-defaults-v2'),
            html.Label('Motion'),dcc.Dropdown(id='motion-mode',options=[{'label':'Aggressive duration search','value':'aggressive'}, {'label':'Random sweep durations','value':'random'}],value='aggressive',clearable=False,persistence='runner-defaults-v2'),
            html.Label('Sweep duration · min / max (s)'),html.Div([dcc.Input(id='min-duration',type='number',min=.1,value=.1,persistence='runner-defaults-v2'),dcc.Input(id='max-duration',type='number',min=.1,value=1,persistence='runner-defaults-v2')],className='runner-input-pair'),
        ],id='free-run-controls'),
        html.Div([
            html.Label('Number of restrained poses'),dcc.Input(id='pose-count',type='number',min=1,step=1,value=3,persistence=True),
            html.Label('Maximum trials · restrained'),dcc.Input(id='restrained-count',type='number',min=1,step=1,value=6561,persistence=True),
            html.Label('Trial duration / torque ramp (s)'),html.Div([dcc.Input(id='duration',type='number',min=.01,value=.5,persistence=True),dcc.Input(id='ramp',type='number',min=.001,value=.05,persistence=True)],className='runner-input-pair'),
        ],id='restrained-run-controls',style={'display':'none'}),
        html.P(id='run-plan',className='note'),
        html.Details([html.Summary('Advanced run and payload settings'),
            html.Label('Payload COM offset from tip · XYZ (m)'),
            html.Div([dcc.Input(id=f'payload-{axis}',type='number',value=value,placeholder=axis.upper(),persistence='payload-position-v2') for axis,value in zip('xyz',offset)],className='triple-input'),
            html.P('Right-arm default: 150 mm from the striking tip toward the grip (+Y). Axes match the tip link. The offset moves the added mass, not the clamp point.',className='note'),
            html.Label('Payload box dimensions · XYZ (m)'),
            html.Div([dcc.Input(id=f'payload-size-{axis}',type='number',min=.0001,value=.05,persistence=True) for axis in 'xyz'],className='triple-input'),
            html.Label('Optional COM inertia (kg·m²)'),dcc.Textarea(id='payload-inertia',placeholder='[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]',persistence=True),
            html.P('Blank uses a uniform box. Custom inertia uses the same local XYZ axes; the box still defines collision geometry.',className='note'),
            html.Label('Seed'),dcc.Input(id='seed',type='number',min=0,step=1,value=0,persistence=True),
            html.Label('Worker processes'),dcc.Input(id='workers',type='number',min=1,max=16,step=1,value=2,persistence=True),
            html.Div([html.Label('Optional restrained poses · degrees'),dcc.Textarea(id='poses',placeholder='[[0,0,0,0,0,0,0], ...]',persistence=True)],id='pose-list-control'),
            html.P('Compact storage: all-sample statistics, up to 120 preview frames per accepted trial, and exact global extrema. Generate individual full recordings in the Viewer.',className='note'),
            dcc.Checklist(id='full',options=[{'label':'Save all full histories (large files)','value':'full'}],value=[],persistence='compact-v1'),
        ]),
        ],id='runner-settings'),
        html.Div([html.Button('Start test',id='launch',className='primary'),
            html.Button('Stop test',id='cancel',disabled=True)],className='runner-actions'),
        html.Div(id='launch-status',className='note'),html.Div(id='job-status',className='note'),
        html.Div(id='cancel-status',className='note'),
        html.P('Stop saves completed trials. Trial limits include rejected attempts.',className='note'),
    ],className='run-setup')


def create_app(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = Dash(__name__, title='OpenArm · Load Lab', assets_folder=str(Path(__file__).with_name('assets')),
               assets_ignore=r'(three\.module|OrbitControls)\.js')
    models = [ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml',
              ROOT/'robots/moving_asm/urdf/moving_asm.sim.yaml', ROOT/'configs/arm.yaml']
    launch_lock, processes, active_jobs = RLock(), {}, {}
    for job_file in root.glob('job-*/job.json'):
        state = job_state(job_file.parent)
        if state['status'] in ('starting','running'):
            active_jobs[state['scenario']] = str(job_file.parent)

    def dashboard_job(scenario, jobs):
        active = active_jobs.get(scenario)
        if active and job_state(active)['status'] in ('starting','running'):
            return active
        return (jobs or {}).get(scenario)
    app.layout = html.Div([
        html.Header([html.Div('OPENARM / ENGINEERING', className='eyebrow'), html.H1('Load Lab'),
                     html.P('Explore motion. Inspect restraint. Trace every load back to a trial.')]),
        dcc.Tabs(id='workspace-tab',value='viewer',className='workspace-tabs',children=[
            dcc.Tab(label='Runner',value='runner',className='workspace-tab',selected_className='workspace-tab-selected'),
            dcc.Tab(label='Viewer',value='viewer',className='workspace-tab',selected_className='workspace-tab-selected'),
        ]),
        dcc.Store(id='runner-run'),dcc.Store(id='runner-observed'),dcc.Store(id='runner-settings-key'),
        dcc.Interval(id='refresh', interval=1500), dcc.Store(id='bin-filter'), dcc.Store(id='chart-version'), dcc.Store(id='table-version'), dcc.Store(id='histogram-focus'),
        dcc.Store(id='launched-jobs',data={},storage_type='session'),dcc.Store(id='followed-jobs',data=[],storage_type='session'),
        # Internal selection state: defaults follow the experiment; FEA-table clicks
        # still select the loaded mate and signed extreme without sidebar controls.
        html.Div([
            dcc.Dropdown(id='interface', clearable=False),
            dcc.Input(id='stage', type='hidden', value='random'),
            dcc.Input(id='accepted', type='hidden', value='accepted'),
            dcc.Input(id='weight', type='hidden', value='time'),
            dcc.Input(id='extreme', type='hidden', value='absolute'),
        ], style={'display':'none'}),
        html.Div([
            html.Aside([
                html.Label('EXPERIMENT'),dcc.RadioItems(id='runner-scenario',options=[
                    {'label':'Free motion','value':'campaign'},{'label':'Restrained','value':'restrained'}],value='campaign'),
                run_controls(models),
            ]),
            html.Main([
                html.Div([html.H2('Live test monitor'),html.Button('View results',id='view-results',disabled=True)],className='runner-heading'),
                html.Div(id='runner-status',className='status'),
                html.Progress(id='runner-progress',value=0,max=1),
                html.Div(id='runner-counters',className='runner-counters'),
                html.P(id='runner-config',className='note'),
                html.Div([dcc.Checklist(id='live-arm-enabled',options=[{'label':'Live arm preview · one worker','value':'live'}],value=['live']),
                html.Div(id='runner-arm-ready',style={'display':'none'}),
                html.Div([html.Label('LOAD COMPONENT'),dcc.Dropdown(id='runner-metric',options=[
                    {'label':f'{m.replace("_", " ")}', 'value':m} for m in METRICS[:6]],value='Fx_N',clearable=False),
                ],className='chart-controls')],className='runner-chart-toolbar'),
                html.Div([html.Div(id='runner-arm',style={'display':'none'}),dcc.Graph(id='runner-extremes',figure=blank('Running signed extremes','Start a test to see live records.'),
                    responsive=True,config={'responsive':True})],id='runner-visuals'),
                html.P('Solid: maximum · dashed: minimum · accepted trials at each link’s FEA connection. Click a legend to hide a link.',className='note runner-chart-note'),
                html.Div([html.H2('Latest completed trial'),html.P(id='runner-latest',className='note')],className='runner-latest-heading'),
                html.Div(id='runner-table',className='link-summary runner-table'),
                html.P('Updates arrive as trials finish, approximately every 1.5 seconds. Latest-trial values may be rejected; only accepted trials update overall records. These are not instantaneous timestep readings.',className='note'),
            ]),
        ],id='runner-panel',className='workspace runner-workspace',style={'display':'none'}),
        html.Div([
            html.Aside([
                html.Label('EXPERIMENT'), dcc.RadioItems(id='scenario', options=[{'label': l, 'value': v} for l,v in
                    [('Free motion','campaign'), ('Restrained','restrained')]], value='campaign'),
                html.Label('RUN'), dcc.Dropdown(id='run', placeholder='Select a recorded or live run'),
            ]),
            html.Main([
                html.Label('SELECT A LINK'), dcc.RadioItems(id='part', className='link-picker'),
                html.H2(id='link-title'),
                html.Div([html.Div(id='link-summary'),
                    html.Div(dcc.Graph(id='pose',figure=blank('Selected link mesh'),style={'height':'340px'},config={'responsive':True}),className='fea-view')],className='fea-overview'),
                html.P('Signed component envelopes: left of zero is negative; right is positive, in this link’s local XYZ frame. Independent extrema are not one simultaneous FEA case.', className='note'),
                html.Div([dcc.Graph(id='signed-force', figure=blank('Signed forces'), style={'height':'260px'}, config={'responsive':True}),
                          dcc.Graph(id='signed-moment', figure=blank('Signed moments'), style={'height':'260px'}, config={'responsive':True})], className='grid'),
                html.Div('Loading saved runs…', id='status', className='status'),
                html.Progress(id='run-progress',value=0,max=1),html.P(id='summary', className='note'),
                html.Div([html.Label('HISTOGRAM & RUNNING EXTREMES · LOAD COMPONENT', htmlFor='metric'),
                    dcc.Dropdown(id='metric', options=[
                        *[{'label':f'Force {axis} · F{axis} (N)', 'value':f'F{axis}_N'} for axis in 'xyz'],
                        *[{'label':f'Moment {axis} · M{axis} (Nm)', 'value':f'M{axis}_Nm'} for axis in 'xyz'],
                    ], value='Fx_N', clearable=False),
                    html.P('Applies to both charts below. XYZ components retain positive and negative directions in the selected link’s local frame.', className='note'),
                ], className='chart-controls',id='histogram-section'),
                dcc.Graph(id='histogram', figure=blank('Loading…'), responsive='auto', style={'height':'320px', 'minHeight':0}, config={'responsive':True}),
                html.P('Signed components are split into negative and positive/zero panels. Percentages use the full distribution across both panels; bin widths are logarithmic. Click a bin to inspect contributing trials; press Esc to clear the filter.',className='note'),
                dcc.Graph(id='convergence', figure=blank('Loading…'), responsive='auto', style={'height':'260px', 'minHeight':0}, config={'responsive':True}),
                html.H2('Trial explorer'), html.P('Click a histogram bin to filter contributing trials. Link peaks are the largest connection magnitudes, not summed loads. Select a row to inspect its recorded history.', className='note'),
                dash_table.DataTable(id='trials', columns=[{'name':k,'id':k} for k in ['id','stage','counted','partial','max_tracking_error_deg','duration_s','samples','peak_force_N','peak_moment_Nm','torque_utilization_pct','reason']],
                    page_size=8, row_selectable='single', selected_rows=[], sort_action='native',
                    style_table={'overflowX':'auto'}, style_header={'backgroundColor':'#1c3045','color':'#dde8f3'},
                    style_cell={'backgroundColor':'#101b29','color':'#cbd8e5','border':'1px solid #26394a','textAlign':'left','maxWidth':'360px','whiteSpace':'normal'}),
                html.Div(id='trial-detail', className='note'),
                html.Button('Play trial',id='replay-load',disabled=True,className='primary'),
                html.Button('Generate full recording',id='detail-generate',disabled=True),
                html.Div(id='detail-status',className='note'),
                dcc.Store(id='replay-selection'), dcc.Store(id='replay-data'),
                dcc.Loading(html.Div(id='replay-message',className='note'),type='dot'),
                html.Div(id='trial-player'), html.Div(id='replay-ready',style={'display':'none'}),
                dcc.Graph(id='trace', figure=blank('Loading…'), responsive='auto', style={'height':'260px', 'minHeight':0}, config={'responsive':True}),
                html.Details([html.Summary('Model and experiment assumptions'), html.Pre(id='assumptions')]),
            ])
        ],id='viewer-panel',className='workspace'),
    ])

    def safe(value):
        p = Path(value).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Run must be inside the dashboard results directory.')
        return p

    @app.server.post('/runner-live')
    def runner_live():
        from .live_preview import watched_pose
        try:
            p = safe((request.get_json(silent=True) or {}).get('run',''))
            if not (p/'run.json').exists() or not (p/'model.mjb').exists():
                abort(404)
            info = run_info(p)
            supported = info.get('live_preview_version') == 1
            pose = watched_pose(p,supported and info['status']=='running')
            response = jsonify(status=info['status'],pose=pose,supported=supported)
            response.headers['Cache-Control'] = 'no-store'
            return response
        except (ValueError,OSError):
            abort(404)

    app.clientside_callback("""function(path, enabled, tab) {
        return window.armLivePreview.show(path, (enabled || []).includes('live') && tab === 'runner');
    }""",Output('runner-arm-ready','children'),Input('runner-run','data'),
        Input('live-arm-enabled','value'),Input('workspace-tab','value'))

    app.clientside_callback("""function(tab) {
        document.documentElement.classList.toggle('runner-active', tab === 'runner');
        window.setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 80);
        return tab === 'runner' ? [{}, {display:'none'}] : [{display:'none'}, {}];
    }""",Output('runner-panel','style'),Output('viewer-panel','style'),Input('workspace-tab','value'))

    @app.callback(Output('runner-run','data'),Output('runner-status','children'),
                  Output('runner-progress','value'),Output('runner-progress','max'),
                  Output('runner-counters','children'),Output('runner-config','children'),Output('view-results','disabled'),
                  Input('refresh','n_intervals'),Input('runner-scenario','value'),Input('launched-jobs','data'))
    def monitor(_, scenario, jobs):
        job = dashboard_job(scenario,jobs)
        state = running_job(job) if job else {}
        path = state.get('run')
        if not job:
            paths = [p for p in discover(root) if run_info(p)['scenario'] == scenario]
            path = next((p for p in paths if run_info(p)['status']=='running'),next(iter(paths),None))
        info = run_info(safe(path)) if path else {}
        settings = info.get('settings',{})
        planned = settings.get('expected_trials',settings.get('samples',state.get('planned_trials',0)))
        planned = min(planned,settings.get('max_trials') or planned)
        attempted, counted = info.get('trials',0),info.get('counted',0)
        status = state.get('status') if state.get('status') in ('failed','cancelled') else info.get('status',state.get('status','idle'))
        if state.get('stop_requested') and status in ('starting','running'):
            status = 'stopping'
        phase = info.get('phase','')
        message = f'{status.upper()}'+(f' · {phase}' if phase else '')
        if state.get('error') or info.get('error'):
            message += ' · '+str(state.get('error') or info.get('error'))
        elapsed = info.get('elapsed_s',0)
        if status in ('running','stopping') and info.get('started_at'):
            from datetime import datetime, timezone
            elapsed = (datetime.now(timezone.utc)-datetime.strptime(info['started_at'],'%Y%m%dT%H%M%S-%fZ').replace(tzinfo=timezone.utc)).total_seconds()
        elapsed = max(0,int(elapsed))
        counters = [html.Div([html.Span(label),html.Strong(value)],className='runner-counter') for label,value in [
            ('Trials',f'{attempted:,} / {planned:,}'),('Accepted',f'{counted:,}'),
            ('Rejected',f'{info.get("rejected",attempted-counted):,}'),('Elapsed',f'{elapsed//3600:02}:{elapsed//60%60:02}:{elapsed%60:02}')]]
        payload = settings.get('analysis_payload',state.get('payload',{}))
        name = settings.get('run_name') or state.get('run_name') or info.get('robot','No test yet')
        description = f'{name} · +{payload.get("mass_kg",0):g} kg payload'
        if path:
            description += ' · '+Path(path).name
        if settings.get('acceptance_policy') == 'motor_loads_v1':
            description += ' · Motor-load screening: tracking is diagnostic; accepted includes valid partial trials.'
        elif settings.get('acceptance_policy') == 'motor_loads_v2':
            description += f' · Tracking ≤{settings.get("tracking_tolerance_deg", 5):g}° · final speed ≤20°/s · complete trials only.'
        return str(path) if path else None,message,attempted,max(1,planned,attempted),counters,description,not bool(path)

    @app.callback(Output('runner-extremes','figure'),Output('runner-table','children'),Output('runner-latest','children'),
                  Output('runner-observed','data'),Input('refresh','n_intervals'),Input('runner-run','data'),
                  Input('runner-metric','value'),State('runner-observed','data'))
    def live_records(_, path, metric, previous):
        version = [run_version(safe(path)) if path else None,metric]
        if previous and previous.get('version') == version:
            raise PreventUpdate
        if not path:
            return blank('Running signed extremes','Start a test to see live records.'),'','No completed trials yet.',dict(version=version)
        figure,records,latest = runner_figures(safe(path),metric)
        old = previous.get('records',{}) if previous and previous.get('path')==path and previous.get('metric')==metric else {}
        table = runner_table(records,old)
        detail = 'No completed trials yet.' if latest is None else f'Trial {latest["id"]+1} · '+('accepted partial' if latest.get('partial') else 'accepted' if latest['counted'] else 'rejected')
        if latest and latest.get('acceptance_policy') in ('motor_loads_v1', 'motor_loads_v2'):
            detail += f' · tracking error {latest["max_tracking_error_deg"]:.1f}°'
            detail += f' · final speed {max(np.abs(latest.get("final_speed_deg_s", [0]))):.1f}°/s'
        if latest and latest.get('reason'):
            detail += ' · '+latest['reason']
        return figure,table,detail,dict(version=version,path=path,metric=metric,
            records={r['part']:{'minimum':r['minimum'],'maximum':r['maximum']} for r in records})

    @app.callback(Output('workspace-tab','value'),Output('scenario','value'),Output('run','value',allow_duplicate=True),
                  Input('view-results','n_clicks'),State('runner-run','data'),prevent_initial_call=True)
    def view_results(_, path):
        if not path:
            raise PreventUpdate
        return 'viewer',run_info(safe(path))['scenario'],str(safe(path))

    @app.callback(Output('runner-settings','disabled'),Output('model','disabled'),Output('motion-mode','disabled'),
                  Output('full','options'),Input('launch','disabled'))
    def lock_settings(active):
        return active,active,active,[{'label':'Save all full histories (large files)','value':'full','disabled':bool(active)}]

    @app.callback(*[Output(field,'value') for field in RUN_FIELDS],Output('runner-settings-key','data'),
                  Input('runner-scenario','value'),Input('launched-jobs','data'),State('runner-settings-key','data'))
    def restore_settings(scenario, jobs, previous):
        job = dashboard_job(scenario,jobs)
        key = [scenario,job]
        if key == previous:
            raise PreventUpdate
        form = read_json(safe(job)/'request.json',{}).get('form',{}) if job and running_job(job)['status'] in ('starting','running') else {}
        return *[form.get(field,no_update) for field in RUN_FIELDS],key

    @app.callback(*[Output(f'payload-{axis}','value',allow_duplicate=True) for axis in 'xyz'],
                  Input('model','value'),State('runner-scenario','value'),State('launched-jobs','data'),prevent_initial_call=True)
    def payload_position_default(model, scenario, jobs):
        if Path(model) not in models:
            raise PreventUpdate
        job = dashboard_job(scenario,jobs)
        if job and running_job(job)['status'] in ('starting','running'):
            form = read_json(safe(job)/'request.json',{}).get('form',{})
            if form.get('model') == model:
                return [form.get(f'payload-{axis}',no_update) for axis in 'xyz']
            raise PreventUpdate
        from .config import load_config
        return default_payload_offset(load_config(model))

    @app.server.get('/replay-model')
    def replay_model():
        try:
            p = safe(request.args.get('run', ''))/'model.mjb'
            return jsonify(model_geometry(str(p), p.stat().st_mtime_ns))
        except (ValueError, OSError):
            abort(404)

    @app.callback(Output('replay-selection','data'), Output('replay-load','disabled'),
                  Input('trials','derived_virtual_selected_rows'), Input('run','value'),
                  Input('part','value'), Input('interface','value'), Input('metric','value'), Input('extreme','value'),
                  State('trials','derived_virtual_data'), State('replay-selection','data'))
    def replay_selection(selected, path, part, interface, metric, extreme, rows, previous):
        if not path or not part or not selected or not rows or selected[0] >= len(rows):
            return None, True
        p = safe(path)
        trial_id = int(rows[selected[0]]['id'])
        interface = scoped_interface(run_info(p).get('interfaces', []), part, interface)
        available = (p/'model.mjb').exists() and (p/'trials'/str(trial_id)/'trial.json').exists() and interface is not None
        selection = dict(path=str(p),trial=trial_id,part=part,interface=interface,metric=metric,extreme=extreme,
                         available=available)
        return no_update if selection == previous else selection, not available

    @app.callback(Output('detail-status','children'), Output('detail-generate','disabled'),
                  Input('detail-generate','n_clicks'), Input('refresh','n_intervals'), Input('replay-selection','data'))
    def full_recording(clicks, refresh, selection):
        if not selection:
            return 'Compact playback is a preview. Select a trial to generate its full recording.', True
        path, trial = safe(selection['path']), int(selection['trial'])
        info = run_info(path)
        if info.get('status') == 'running':
            return 'Full recordings can be generated after this run finishes.', True
        saved = json.loads((path/'trials'/str(trial)/'trial.json').read_text())
        status = recording_status(path, trial)
        if saved.get('full_history') or status.get('status') == 'complete':
            return 'Full recording available. Click Play trial to load it.', True
        if status.get('status') == 'running':
            return 'Generating and verifying a full recording…', True
        if ctx.triggered_id == 'detail-generate':
            return start_recording(path, trial), False
        if status.get('status') == 'error':
            return 'Full recording unavailable: '+status.get('error', 'Unknown error'), False
        return 'Generate a full recording only when needed. This reruns and verifies the selected trial; original results stay unchanged.', False

    @app.callback(Output('replay-data','data'), Output('replay-message','children'),
                  Input('replay-load','n_clicks'), Input('replay-selection','data'))
    def load_replay(clicks, selection):
        if not selection:
            return None, 'Select a trial row to play its recorded motion.'
        if not selection['available']:
            return None, 'Playback unavailable: this trial has no saved model or recorded motion history.'
        if ctx.triggered_id != 'replay-load':
            return None, f'Trial {selection["trial"]} selected. Click Play trial to load the recorded motion.'
        try:
            payload = playback(str(safe(selection['path'])),int(selection['trial']),selection['part'],
                               int(selection['interface']),selection['metric'],selection['extreme'])
            return payload, f'Trial {payload["trial"]} · recorded motion · {payload["part"]} highlighted in teal'
        except (ValueError, OSError, KeyError, IndexError) as exc:
            return None, f'Playback unavailable: {exc}'

    app.clientside_callback(
        "function(data) { return window.armTrialReplay.load(data); }",
        Output('replay-ready','children'), Input('replay-data','data'))

    @app.callback(Output('run','options'), Output('run','value'), Output('followed-jobs','data'),
                  Input('refresh','n_intervals'), Input('scenario','value'), Input('launched-jobs','data'), State('run','value'),
                  State('run','options'), State('followed-jobs','data'))
    def refresh(_, scenario, jobs, current, previous, followed):
        all_paths = discover(root)
        def label(p):
            info = run_info(p)
            settings = info.get('settings',{})
            return f"{settings.get('run_name') or info['robot']} · +{settings.get('analysis_payload',{}).get('mass_kg',0):g} kg · {settings.get('campaign_mode',info['scenario'])} · {p.name} · {info['status']}"
        options = [{'label':label(p),'value':str(p)} for p in all_paths]
        selected = [o for o in options if run_info(o['value'])['scenario'] == scenario]
        value = no_update if current in [o['value'] for o in selected] else (selected[0]['value'] if selected else None)
        job = dashboard_job(scenario, jobs)
        followed = followed or []
        seen = no_update
        if job and job not in followed:
            state = job_state(safe(job))
            if state.get('run') in [o['value'] for o in selected]:
                value = state['run']
                seen = [*followed,job][-100:]
        return (no_update if selected == previous else selected), value, seen

    @app.callback(Output('stage','value'), Output('weight','value'), Input('scenario','value'))
    def defaults(scenario):
        return ('enumeration', 'trials') if scenario == 'restrained' else ('random','time')

    @app.callback(Output('free-run-controls','style'),Output('restrained-run-controls','style'),
                  Output('pose-list-control','style'),Output('run-plan','children'),
                  Input('runner-scenario','value'),Input('model','value'),Input('free-count','value'),
                  Input('restrained-count','value'),Input('pose-count','value'),Input('poses','value'))
    def run_plan(scenario, model, free_count, restrained_count, pose_count, poses):
        restrained_mode = scenario == 'restrained'
        visible, hidden = {}, {'display':'none'}
        try:
            from .config import load_config
            from .dashboard_jobs import integer
            if restrained_mode:
                if Path(model) not in models:
                    raise ValueError('Choose a robot.')
                n = len(load_config(model)['joints'])
                count = len(json.loads(poses)) if poses and poses.strip() else integer(pose_count,'Poses')
                total = count*3**n
                cap = integer(restrained_count,'Maximum trials')
                text = f'{count} poses × {3**n:,} torque combinations = {total:,}. This run attempts up to {min(total,cap):,} trials.'
                if cap < total:
                    text += ' The limit stops enumeration early; it does not cover every combination.'
            else:
                text = f'Up to {integer(free_count,"Maximum trials"):,} attempts from random pose pairs. No extra refinement or finer-step reruns beyond this limit.'
        except (ValueError,TypeError,OSError):
            text = 'Enter valid trial and pose limits.'
        return hidden if restrained_mode else visible, visible if restrained_mode else hidden, visible if restrained_mode else hidden, text

    @app.callback(Output('part','options'), Output('part','value'), Input('run','value'), State('part','value'))
    def parts(path, current):
        refs = run_info(safe(path)).get('interfaces', []) if path else []
        names = list(dict.fromkeys(part_name(r) for _, r in connection_rows(refs)))
        labels = {'base': 'base · terminal assembly', 'body_1': 'body_1 · mount'}
        options = [{'label': labels.get(n, n), 'value': n} for n in names]
        return options, current if current in names else ('l1' if 'l1' in names else next(iter(names), None))

    @app.callback(Output('interface','options'), Output('interface','value'), Input('run','value'), Input('part','value'))
    def interfaces(path, part):
        refs = run_info(safe(path)).get('interfaces', []) if path else []
        mates = connection_rows(refs, part) if part else []
        return [{'label':r['label'], 'value':i} for i,r in mates], mates[0][0] if mates else None

    @app.callback(Output('histogram','figure'), Output('convergence','figure'),
                  Output('status','children'), Output('summary','children'), Output('assumptions','children'), Output('link-title','children'), Output('signed-force','figure'), Output('signed-moment','figure'), Output('chart-version','data'),
                  Input('refresh','n_intervals'), Input('run','value'), Input('interface','value'),
                  Input('metric','value'), Input('stage','value'), Input('accepted','value'), Input('weight','value'), Input('part','value'), Input('extreme','value'), State('chart-version','data'), Input('histogram-focus','data'))
    def charts(_, path, interface, metric, stage, accepted, weight, part, extreme, previous, focus=None):
        version = [run_version(path) if path else None, interface, metric, stage, accepted, weight, part, extreme, focus]
        if version == previous:
            raise PreventUpdate
        if not path or not part:
            return *[blank(t,'Start an analysis or select a run') for t in ('Distribution','Running maxima')], 'Select a run and link', '', '', '', blank('Signed forces'), blank('Signed moments'), version
        path = safe(path)
        plots, _, summary = figures(path, None, interface, metric, stage, accepted, weight, part, extreme, focus)
        s = run_info(path)
        status = f"{s['status'].upper()} · {s['counted']} accepted / {s['trials']} trials · {s['robot']}"
        settings = s.get('settings',{})
        status += f" · +{settings.get('analysis_payload',{}).get('mass_kg',0):g} kg payload"
        if settings.get('run_name'):
            status += ' · '+settings['run_name']
        if s.get('settings', {}).get('campaign_mode') == 'aggressive':
            status += f" · aggressive search · {s['settings']['samples']:,} planned attempts"
        return *plots[:2], status, summary, json.dumps({k:v for k,v in s.get('settings', {}).items() if k not in ('poses_deg', 'poses')}, indent=2), f'{part} · loads and FEA', *signed_figures(path, part, stage, accepted), version

    @app.callback(Output('run-progress','value'),Output('run-progress','max'),Output('run-progress','title'),
                  Input('refresh','n_intervals'),Input('run','value'))
    def selected_progress(_, path):
        s = run_info(safe(path)) if path else {}
        settings = s.get('settings',{})
        expected = settings.get('expected_trials',settings.get('samples',1))
        planned = min(expected,settings.get('max_trials') or expected)
        completed = s.get('trials',0)
        return completed,max(1,planned,completed),f'{completed:,} / {planned:,} attempted trials'

    @app.callback(Output('link-summary','children'), Input('chart-version','data'),
                  State('run','value'), State('part','value'), State('stage','value'), State('accepted','value'), State('weight','value'))
    def fea_selection(version, path, part, stage, accepted, weight='time'):
        return link_summary(safe(path),part,stage,accepted,weight) if path and part else ''

    @app.callback(Output('pose','figure'), Input('run','value'), Input('part','value'))
    def mesh_view(path, part):
        if not path or not part:
            return blank('Select a link')
        return pose_figure(read_json(safe(path)/'metadata.json', {'interfaces':[]}),None,part=part,run_path=safe(path))

    @app.callback(Output('bin-filter','data'), Input('histogram','clickData'),
                  Input('run','value'), Input('interface','value'), Input('metric','value'),
                  Input('stage','value'), Input('accepted','value'), Input('weight','value'), Input('part','value'), Input('extreme','value'))
    def select_bin(click, *unused):
        return int(click['points'][0]['customdata']) if ctx.triggered_id == 'histogram' and click else None

    @app.callback(Output('trials','data'), Output('trials','selected_rows'), Output('table-version','data'),
                  Input('run','value'), Input('stage','value'), Input('accepted','value'), Input('bin-filter','data'),
                  Input('interface','value'), Input('metric','value'), Input('weight','value'),
                  Input('refresh','n_intervals'), Input('part','value'), Input('extreme','value'), State('trials','selected_rows'), State('table-version','data'))
    def table(path, stage, accepted, b, interface, metric, weight, tick, part, extreme, selection, previous):
        version = [run_version(path) if path else None, stage, accepted, b, interface, metric, weight, part, extreme]
        if version == previous:
            raise PreventUpdate
        if not path:
            return [], [], version
        rows = trials(safe(path), stage, accepted)
        interface = scoped_interface(run_info(path).get('interfaces', []), part, interface)
        if b is not None and interface is not None:
            i,j = int(interface or 0), METRICS.index(metric)
            found = []
            for r in rows:
                p = safe(path)/'trials'/str(r['id'])/'distribution.npz'
                if weight == 'trials':
                    hit = 'maximum' in r and np.searchsorted(edges(), trial_extreme(r, i, j, extreme), side='right')-1 == b
                else:
                    hit = p.exists() and load_npz(str(p))['counts'][i,j,b] > 0
                if hit:
                    found.append(r)
            rows = found
        indices = [i for i, _ in connection_rows(run_info(path).get('interfaces', []), part)]
        rows = [dict(r, torque_utilization_pct=round(100*max(r.get('peak_torque_fraction', [0])), 1) if 'peak_torque_fraction' in r else None, **{f'peak_{metric}': round(max(r['maximum'][i][METRICS.index(metric)] for i in indices), 3)
                         for metric in ('force_N', 'moment_Nm')}) if indices and 'maximum' in r else r for r in rows]
        # Large per-connection extrema stay on disk; don't resend them in every table row.
        rows = [{k:v for k,v in r.items() if k in ('id','stage','counted','partial','max_tracking_error_deg','final_speed_deg_s','start_tracking_error_deg','diagnostics','measured_duration_s','acceptance_policy','duration_s','samples','reason','peak_force_N','peak_moment_Nm','torque_utilization_pct','pair_id','direction','search_step','search_role','full_history','max_clamp_position_error_m','max_clamp_angle_error_rad')} for r in rows]
        selected = selection if ctx.triggered_id == 'refresh' and selection and selection[0] < len(rows) else []
        return rows, no_update if ctx.triggered_id == 'refresh' else selected, version

    @app.callback(Output('trace','figure'), Output('trial-detail','children'),
                  Input('trials','derived_virtual_selected_rows'), Input('run','value'), Input('interface','value'), Input('metric','value'), Input('part','value'), Input('extreme','value'),
                  State('trials','derived_virtual_data'))
    def inspect(selected, path, interface, metric, part, extreme, rows):
        f = blank('Selected trial history')
        if not path or not part:
            return f, ''
        r = rows[selected[0]] if selected and rows and selected[0] < len(rows) else None
        interface = scoped_interface(run_info(path).get('interfaces', []), part, interface)
        if interface is None:
            return f, ''
        detail = ''
        if r:
            p = trace_path(safe(path), r['id'])
            detail = json.dumps({k:r[k] for k in ('id','partial','max_tracking_error_deg','start_tracking_error_deg','final_speed_deg_s','diagnostics','measured_duration_s','pair_id','direction','search_step','search_role','reason','max_clamp_position_error_m','max_clamp_angle_error_rad') if k in r})
            if p.exists():
                v = load_npz(str(p))
                y = metrics(v['loads'])[:, int(interface or 0), METRICS.index(metric)]
                f.add_scatter(x=v['time'], y=y, mode='lines', line_color=COLORS[0], name=run_info(path)['interfaces'][interface]['label'])
                f.update_layout(xaxis_title='Time (s)', yaxis_title=metric,
                                title='Full physics history' if r.get('full_history') or p.parent.name == 'detail' else 'Preview history · exact extrema are in the table')
            else:
                f.update_layout(title='No preview saved · generate a full recording to inspect this trial')
        f.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#e5edf5')
        return f, detail

    def running_job(path):
        state = job_state(safe(path))
        proc = processes.get(str(path))
        if proc is not None and proc.poll() is not None and state['status'] in ('starting','running'):
            state.update(status='failed',error='Worker exited unexpectedly. Check worker.log.')
            from .analysis import atomic_json
            atomic_json(safe(path)/'job.json',state)
        return state

    @app.callback(Output('launch-status','children'),Output('launched-jobs','data'), Input('launch','n_clicks'),
                  State('runner-scenario','value'),State('model','value'),State('free-count','value'),State('restrained-count','value'),
                  State('seed','value'),State('workers','value'),State('motion-mode','value'),State('min-duration','value'),State('max-duration','value'),
                  State('duration','value'),State('ramp','value'),State('pose-count','value'),State('poses','value'),State('full','value'),
                  State('run-name','value'),State('payload-mass','value'),
                  *[State(f'payload-{axis}','value') for axis in 'xyz'],
                  *[State(f'payload-size-{axis}','value') for axis in 'xyz'],
                  State('payload-inertia','value'),State('launched-jobs','data'),prevent_initial_call=True)
    def launch(_, scenario, model, free_count, restrained_count, seed, workers, motion, mind, maxd,
               duration, ramp, pose_count, poses, full, name, mass, x, y, z, sx, sy, sz, inertia, jobs):
        try:
            if Path(model) not in models:
                raise ValueError('Choose a supported robot model.')
            prepared = prepare(model,scenario,dict(count=restrained_count if scenario=='restrained' else free_count,
                seed=seed,workers=workers,motion=motion,min_duration=mind,max_duration=maxd,duration=duration,ramp=ramp,
                pose_count=pose_count,poses=poses,full=full,name=name,mass=mass,offset=[x,y,z],size=[sx,sy,sz],inertia=inertia))
            prepared['form'] = runner_form(model,scenario,dict(count=restrained_count if scenario=='restrained' else free_count,
                seed=seed,workers=workers,motion=motion,min_duration=mind,max_duration=maxd,duration=duration,ramp=ramp,
                pose_count=pose_count,poses=poses,full=full,name=name,mass=mass,offset=[x,y,z],size=[sx,sy,sz],inertia=inertia))
            with launch_lock:
                previous = active_jobs.get(scenario) or (jobs or {}).get(scenario)
                if previous and running_job(previous)['status'] in ('starting','running'):
                    return 'This mode already has a dashboard run active. Stop it or wait for it to finish.',no_update
                folder = root/('job-'+__import__('uuid').uuid4().hex[:10])
                save_request(folder,prepared)
                try:
                    with (folder/'worker.log').open('w') as log:
                        proc = subprocess.Popen([sys.executable,'-m','arm_design.dashboard_jobs',str(folder)],
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except OSError as exc:
                    from .analysis import atomic_json
                    atomic_json(folder/'job.json',dict(status='failed',error=str(exc),scenario=scenario))
                    raise
                processes[str(folder)] = proc
                active_jobs[scenario] = str(folder)
            return f'New run queued · +{prepared["payload"]["mass_kg"]:g} kg · up to {prepared["planned_trials"]:,} trials. It will open automatically.',dict(jobs or {},**{scenario:str(folder)})
        except (ValueError,TypeError,OSError,KeyError) as exc:
            return str(exc),no_update

    @app.callback(Output('launch','disabled'),Output('cancel','disabled'),Output('job-status','children'),
                  Input('refresh','n_intervals'),Input('launched-jobs','data'),Input('runner-scenario','value'),Input('runner-run','data'))
    def job_progress(_, jobs, scenario, path):
        job = dashboard_job(scenario, jobs)
        selected = run_info(safe(path)) if path else {}
        if selected.get('scenario') != scenario:
            selected = {}
        if not job:
            return False,selected.get('status') != 'running',''
        state = running_job(job)
        active = state['status'] in ('starting','running')
        message = 'Stopping · finishing active trials and saving results.' if active and state.get('stop_requested') else (
            'Preparing model and poses…' if active and not state.get('run') else state['status'].capitalize())
        if state.get('run'):
            info = run_info(safe(state['run']))
            planned = state.get('planned_trials',info.get('trials',0))
            phase = info.get('phase','')
            message += f" · {info['trials']:,} / {planned:,} trials · {info['counted']:,} accepted"
            if phase:
                message += f' · {phase}'
        if state.get('error'):
            message += ' · '+state['error']
        return active,not (active or selected.get('status') == 'running'),message

    @app.callback(Output('cancel-status','children'), Input('cancel','n_clicks'), State('runner-run','data'),
                  State('launched-jobs','data'),State('runner-scenario','value'),prevent_initial_call=True)
    def cancel(_, path, jobs, scenario):
        job = dashboard_job(scenario, jobs)
        if job and running_job(job)['status'] in ('starting','running'):
            request_stop(safe(job))
            return 'Stop requested for the new run in this mode. Active trials finish; completed results are retained.'
        if path and run_info(safe(path))['status'] == 'running' and run_info(safe(path))['scenario'] == scenario:
            (safe(path)/'CANCEL').touch()
            return 'Stop requested for the selected run. Active trials finish; completed results are retained.'
        return 'Select a running analysis.'

    return app


def serve(root, port=8050):
    create_app(root).run(host='127.0.0.1', port=port, debug=False)

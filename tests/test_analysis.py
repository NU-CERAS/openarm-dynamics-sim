import json
import sqlite3
from itertools import islice
from pathlib import Path

import mujoco
import numpy as np
import pytest

from arm_design.analysis import RunStore, distribution, edges, trials
from arm_design.config import ROOT, load_config
from arm_design.dashboard import create_app, aggregate, figures, snapshots, connection_rows
from arm_design.fea import FEARecording
from arm_design.fea_cases import part_cases
from arm_design.restrained import build_restrained, combinations, restrained, trial
from test_fea import assert_body_balance

RIGHT = ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml'


def test_distribution_counts_every_sample_and_keeps_overflow():
    loads = np.array([[[3,4,0,0,0,2]], [[-3,0,0,0,0,0]], [[2e6,0,0,0,0,-2e6]]], float)
    rows = [dict(loads_local=v, time_s=i*.001, q_rad=[i], actuator_torque_Nm=[0], torque_cap_Nm=[1])
            for i,v in enumerate(loads)]
    d = distribution(rows)
    assert d['count'] == 3
    np.testing.assert_array_equal(d['counts'].sum(axis=-1), 3)
    assert d['counts'][0,0,-1] == 1 and d['counts'][0,5,0] == 1
    assert d['total'][0,6] == 5+3+2e6
    assert d['squares'][0,6] == 25+9+4e12


def test_all_motor_combinations_unique():
    values = list(combinations(7))
    assert len(values) == len(set(values)) == 2187
    assert (0,)*7 in values and (-1,)*7 in values and (1,)*7 in values
    for i in range(7):
        assert [v[i] for v in values].count(0) == 729


def test_restrained_weld_balances_and_zero_is_zero_command(tmp_path):
    cfg = load_config(RIGHT)
    arm = build_restrained(cfg)
    for signs in (np.zeros(7), np.ones(7), -np.ones(7)):
        r = trial(arm, np.zeros(7), signs, duration=.1, full_history=True)
        assert r['counted'], r['reason']
        assert r['max_clamp_position_error_m'] < .005
        assert r['max_clamp_angle_error_rad'] < .05
        rec = FEARecording(arm, arm.data())
        for s in r['extrema'].values():
            assert_body_balance(rec, s)
            assert np.all(np.abs(s['actuator_torque_Nm']) <= s['torque_cap_Nm']+1e-9)
            if not signs.any():
                np.testing.assert_array_equal(s['actuator_torque_Nm'], 0)
        assert r['distribution']['counts'].sum(axis=-1).min() == r['samples']
        # Whole arm is NOT frozen, despite the fixed hand.
        assert np.linalg.norm(r['distribution']['trace']['q'][-1]) > 1e-5
    # Reinitialization moves the world anchor to a different clamp pose without preload.
    q = np.deg2rad([20,40,-30,60,10,-40,25])
    from arm_design.restrained import set_pose
    data,p,R = set_pose(arm,q)
    np.testing.assert_allclose(data.site('tip').xpos, data.site('clamp_target').xpos, atol=1e-12)
    np.testing.assert_allclose(data.efc_pos[:6], 0, atol=1e-12)


def test_restrained_export_and_dashboard(tmp_path):
    path = restrained(load_config(RIGHT), tmp_path, poses_deg=[[0]*7], max_trials=2,
                      duration=.05, ramp=.02, workers=2, log=lambda *a, **k:None)
    state = json.loads((path/'run.json').read_text())
    assert state['status'] == 'partial' and state['trials'] == state['counted'] == 2
    rows = trials(path)
    assert [r['signs'] for r in rows] == [list(s) for s in islice(combinations(7), 2)]
    stats, _ = aggregate(path, 'enumeration','accepted')
    assert stats['count'] == 100
    np.testing.assert_array_equal(stats['counts'].sum(axis=-1), 100)
    cases = part_cases(Path(state['export']), 'base')
    assert cases
    assert max(c['residual_n'] for c in cases) < 1e-8
    assert max(c['moment_residual_nm'] for c in cases) < 1e-8
    assert any(np.linalg.norm(c['contact']['moment']) > 0 for c in cases)
    figs, _, summary = figures(path, None, 0, 'force_N', 'enumeration','accepted','trials')
    assert len(figs[0].data) == 1 and sum(figs[0].data[0].y) == pytest.approx(100)
    assert snapshots(path,0)
    app = create_app(tmp_path)
    # Hover previews must use the table's loaded mate (not the sidebar's proximal mate).
    from arm_design.dashboard import link_summary
    preview_table = link_summary(path,'l3','enumeration','accepted','trials')
    def components(node):
        if hasattr(node,'to_plotly_json'):
            yield node
            children=getattr(node,'children',None)
            for child in children if isinstance(children,(list,tuple)) else [children]:
                yield from components(child)
    previews=[json.loads(c.to_plotly_json()['props']['data-load-preview']) for c in components(preview_table)
              if 'data-load-preview' in c.to_plotly_json()['props']]
    assert len(previews)==12
    loaded = connection_rows(state['interfaces'],'l3')[1][0]
    assert all(p['interface']==loaded and p['part']=='l3' and 'per trial' in p['weight'] for p in previews)
    for preview in previews:
        j = ['Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm'].index(preview['metric'])
        assert preview['value'] == stats[preview['side']][loaded,j]
    # Switching links with the old connection still in flight must never leak its loads.
    from arm_design.dashboard import link_snapshots, pose_figure
    refs = state['interfaces']
    indices = [i for i, r in connection_rows(refs, 'l3')]
    figs, _, _ = figures(path, None, 0, 'force_N', 'enumeration', 'accepted', 'time', 'l3')
    assert list(figs[2].data[0].x) == [refs[i]['label'] for i in indices]
    np.testing.assert_allclose(figs[2].data[0].y, stats['maximum'][indices, 6])
    assert len(figs[1].data) == 2
    assert all(t.name.startswith('l3 <- ') for t in figs[1].data)
    from arm_design.fea import METRICS
    for j, metric in enumerate(METRICS):
        selected, _, _ = figures(path, None, indices[0], metric, 'enumeration', 'accepted', 'time', 'l3')
        assert sum(sum(t.y) for t in selected[0].data) == pytest.approx(100)
        if j < 6:
            assert metric in selected[1].layout.title.text
            np.testing.assert_allclose(selected[1].data[0].y, np.minimum.accumulate([r['minimum'][indices[0]][j] for r in rows]))
        else:
            np.testing.assert_allclose(selected[1].data[0].y, [h['maximum'][indices[0]][j-6] for h in state['history']])
            assert len(selected[1].data) == len(indices)
    sample = link_snapshots(path, 0, 'l3')[0]
    meta = json.loads((path/'metadata.json').read_text())
    pose = pose_figure(meta, sample, part='l3')
    assert all(t.name == 'l3' or t.name.startswith('l3 <- ') for t in pose.data)
    callback = lambda prefix: next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if k.startswith(prefix))
    options, selected = callback('..interface.options')(str(path), 'l3')
    assert [o['value'] for o in options] == indices
    assert selected == indices[0]
    assert callback('..part.options')(str(path), 'l3')[1] == 'l3'
    _, detail = callback('..trace.figure')([0], str(path), 0, 'force_N', 'l3', 'absolute', rows)
    assert 'signs' not in detail
    mesh = callback('pose.figure')(str(path),'l3')
    assert any(t.type == 'mesh3d' for t in mesh.data)
    assert not mesh.layout.updatemenus
    for i in indices:
        marker = next(t for t in mesh.data if t.name == refs[i]['label'])
        np.testing.assert_allclose([marker.x[0],marker.y[0],marker.z[0]],sample['application_points_local_m'][i],atol=1e-8)
    assert all(t.name == 'l3' or t.name.startswith('l3 <- ') for t in mesh.data)
    client = app.server.test_client()
    table_key = next(k for k in app.callback_map if k.startswith('..trials.data'))
    response = client.post('/_dash-update-component', json=dict(
        output=table_key,
        outputs=[dict(id='trials', property='data'), dict(id='trials', property='selected_rows'), dict(id='table-version', property='data')],
        inputs=[dict(id=i, property=p, value=v) for i,p,v in [
            ('run','value',str(path)), ('stage','value','enumeration'), ('accepted','value','accepted'),
            ('bin-filter','data',None), ('interface','value',0), ('metric','value','force_N'),
            ('weight','value','trials'), ('refresh','n_intervals',0), ('part','value','l3'), ('extreme','value','absolute')]],
        state=[dict(id='trials',property='selected_rows',value=[0]), dict(id='table-version',property='data',value=None)],
        changedPropIds=['part.value']))
    assert response.status_code == 200
    table = response.json['response']['trials']
    assert table['selected_rows'] == []
    assert table['data'][0]['peak_force_N'] == round(max(rows[0]['maximum'][i][6] for i in indices), 3)
    assert 'signs' not in table['data'][0] and 'maximum' not in table['data'][0]
    for route in ('/', '/_dash-layout','/_dash-dependencies','/assets/dashboard.css'):
        assert client.get(route).status_code == 200


def test_cancel_retains_committed_trials(tmp_path):
    arm = build_restrained(load_config(RIGHT))
    store = RunStore(tmp_path, arm, 'restrained', {})
    result = trial(arm, np.zeros(7), np.zeros(7), duration=.02, ramp=.01)
    store.add(0, result, 'enumeration')
    (store.path/'CANCEL').touch()
    assert store.cancelled
    store.finish('cancelled')
    assert json.loads((store.path/'run.json').read_text())['status'] == 'cancelled'
    assert len(trials(store.path)) == 1
    assert (store.path/'trials/0/cases.json.gz').exists()


def test_legacy_aggregate_rows_are_hidden_without_shifting_loads(tmp_path):
    refs = [dict(kind='mate', label='link <- base'), dict(kind='delta', label='link delta'),
            dict(kind='mate', label='link <- tool')]
    state = dict(interfaces=refs, robot='test', scenario='campaign', history=[])
    (tmp_path/'run.json').write_text(json.dumps(state))
    folder = tmp_path/'trials'/'0'
    folder.mkdir(parents=True)
    stats = distribution([dict(loads_local=np.array([[1,0,0,0,0,0], [999,0,0,0,0,0], [3,0,0,0,0,0]]),
                              time_s=0, q_rad=[0], actuator_torque_Nm=[0], torque_cap_Nm=[1])])
    np.savez_compressed(folder/'distribution.npz',
                        **{k:v for k,v in stats.items() if k not in ('trace','full_history')})
    with sqlite3.connect(tmp_path/'trials.sqlite') as db:
        db.execute('CREATE TABLE trials (id INTEGER PRIMARY KEY, info TEXT)')
        db.execute('INSERT INTO trials VALUES (0, ?)', (json.dumps(dict(id=0, stage='random', counted=True)),))
    assert [i for i,_ in connection_rows(refs)] == [0,2]
    client = create_app(tmp_path).server.test_client()
    response = client.post('/_dash-update-component', json=dict(
        output='..interface.options...interface.value..',
        outputs=[dict(id='interface', property='options'), dict(id='interface', property='value')],
        inputs=[dict(id='run', property='value', value=str(tmp_path)), dict(id='part', property='value', value='link')], changedPropIds=['run.value']))
    assert response.status_code == 200
    assert [o['value'] for o in response.json['response']['interface']['options']] == [0,2]
    plots, _, summary = figures(tmp_path, None, 2, 'force_N', 'all', 'accepted', 'time')
    assert list(plots[2].data[0].x) == ['link <- base', 'link <- tool']
    assert list(plots[2].data[0].y) == [1,3]

    assert 'mean 3 ' in summary
    # Historical convergence files also keep original CSV column positions.
    from arm_design.watch import series
    header = ['counted','sweeps','elapsed_s','stage'] + [
        f'{r["label"]}|{metric}' for r in refs for metric in ('force_N','moment_Nm')]
    rows = series(header)
    assert [r['label'] for r in rows] == ['link <- base', 'link <- tool']
    assert rows[1]['force_N'] == (4, None)
    # Legacy peak-only exports hide the same rows.
    legacy = tmp_path/'legacy'
    legacy.mkdir()
    (legacy/'metadata.json').write_text(json.dumps(dict(interfaces=refs)))
    (legacy/'peaks.csv').write_text('interface,metric,maximum\nlink <- base,force_N,1\nlink delta,force_N,999\nlink <- tool,force_N,3\n')
    plots, _, _ = figures(legacy, None, 2, 'force_N', 'all', 'accepted', 'time')
    assert list(plots[2].data[0].y) == [1,3]


def test_dashboard_poll_skips_unchanged_runs_and_refreshes_new_results(tmp_path, monkeypatch):
    from dash.exceptions import PreventUpdate
    from arm_design import dashboard
    state = dict(status='running', counted=1, trials=1, robot='test', settings={})
    (tmp_path/'run.json').write_text(json.dumps(state))
    calls = []
    def draw(*args):
        calls.append(args)
        return [dashboard.blank('test') for _ in range(3)], [], 'summary'
    monkeypatch.setattr(dashboard, 'figures', draw)
    app = create_app(tmp_path)
    callback = next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if k.startswith('..histogram.figure'))
    args = [0, str(tmp_path), 0, 'force_N', 'all', 'accepted', 'time', 'link', 'absolute']
    first = callback(*args, None)
    version = json.loads(json.dumps(first[-1]))
    assert isinstance(version[0][1], str)  # Nanosecond timestamps exceed JavaScript's exact integer range.
    with pytest.raises(PreventUpdate):
        callback(*args, version)
    assert len(calls) == 1
    (tmp_path/'run.json').write_text(json.dumps(dict(state, counted=2, trials=2, status='complete')))
    refreshed = callback(*args, version)
    assert '2 accepted / 2 trials' in refreshed[2]
    assert len(calls) == 2


def test_link_snapshots_ignore_other_link_extrema(tmp_path, monkeypatch):
    from arm_design import dashboard
    refs = [dict(part='l1', kind='mate', label='l1 <- motor'), dict(part='l2', kind='mate', label='l2 <- motor')]
    (tmp_path/'run.json').write_text(json.dumps(dict(interfaces=refs)))
    samples = [dict(sample_index=i, loads_local=[[v]*6, [other]*6])
               for i, (v, other) in enumerate([(1, 0), (2, 999), (3, 0)])]
    monkeypatch.setattr(dashboard, 'snapshots', lambda *args: samples)
    assert [s['sample_index'] for s in dashboard.link_snapshots(tmp_path, None, 'l1')] == [0, 2]
    assert dashboard.link_snapshots(tmp_path, None, 'missing') == []


def test_trial_index_cache_reads_new_commits_and_filters(tmp_path):
    from arm_design.analysis import trials
    path = tmp_path/'trials.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE trials (id INTEGER PRIMARY KEY, info TEXT)')
        db.execute('INSERT INTO trials VALUES (?, ?)', (0, json.dumps(dict(id=0,stage='random',counted=True))))
    assert [r['id'] for r in trials(tmp_path)] == [0]
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO trials VALUES (?, ?)', (1, json.dumps(dict(id=1,stage='random',counted=False))))
        db.execute('INSERT INTO trials VALUES (?, ?)', (2, json.dumps(dict(id=2,stage='refine',counted=True))))
    assert [r['id'] for r in trials(tmp_path)] == [0,2]
    assert [r['id'] for r in trials(tmp_path, 'random', 'rejected')] == [1]
    assert [r['id'] for r in trials(tmp_path, accepted='all')] == [0,1,2]


def test_signed_load_envelopes_and_negative_trial_peaks(tmp_path, monkeypatch):
    from arm_design.dashboard import signed_figures, trial_extreme, link_summary
    refs = [dict(part='l1',kind='mate',label='l1 <- motor'), dict(part='l2',kind='mate',label='l2 <- motor')]
    (tmp_path/'run.json').write_text(json.dumps(dict(interfaces=refs, robot='test', scenario='campaign', history=[])))
    rows = []
    with sqlite3.connect(tmp_path/'trials.sqlite') as db:
        db.execute('CREATE TABLE trials (id INTEGER PRIMARY KEY, info TEXT)')
        for i, values in enumerate([[-10, 2], [-3, 8]]):
            loads = [dict(loads_local=np.array([[v,1,-2,v/2,-1,2],[999]*6]),time_s=k,q_rad=[0],actuator_torque_Nm=[0],torque_cap_Nm=[1]) for k,v in enumerate(values)]
            d = distribution(loads)
            folder = tmp_path/'trials'/str(i)
            folder.mkdir(parents=True)
            np.savez_compressed(folder/'distribution.npz', **{k:v for k,v in d.items() if k not in ('trace','full_history')})
            row = dict(id=i, stage='random', counted=True, minimum=d['minimum'].tolist(),maximum=d['maximum'].tolist())
            rows.append(row)
            db.execute('INSERT INTO trials VALUES (?, ?)', (i,json.dumps(row)))
    from arm_design.fea import METRICS
    for metric in METRICS:
        selected, _, _ = figures(tmp_path, None, 0, metric, 'all', 'accepted', 'time', 'l1')
        assert sum(sum(t.y) for t in selected[0].data) == pytest.approx(100)
        assert selected[0].layout.xaxis.title.text.startswith(metric.replace('_',' ') if metric not in ('force_N','moment_Nm') else metric)
    plots = signed_figures(tmp_path, 'l1', 'all', 'accepted')
    assert list(plots[0].data[0].x) == [-10, 1, -2]
    assert list(plots[0].data[1].x) == [8, 1, -2]
    assert all(label.startswith('l1 <-') for label in plots[0].data[0].y)
    assert plots[0].layout.xaxis.range[0] == -plots[0].layout.xaxis.range[1]
    assert '+8' in str(link_summary(tmp_path, 'l1', 'all', 'accepted'))
    assert trial_extreme(rows[0],0,0,'absolute') == -10
    assert trial_extreme(rows[0],0,0,'minimum') == -10
    assert trial_extreme(rows[0],0,0,'maximum') == 2
    assert trial_extreme(rows[0],0,6,'minimum') == rows[0]['maximum'][0][6]
    figs, _, _ = figures(tmp_path,None,0,'Fx_N','all','accepted','trials','l1','minimum')
    occupied = [(x,y) for x,y in zip(figs[0].data[0].x,figs[0].data[0].y) if y]
    assert [x for x,y in occupied] == sorted(np.searchsorted(edges(), [-10,-3], side='right')-1)
    assert sum(y for x,y in occupied) == pytest.approx(100)
    assert list(figs[1].data[0].y) == [-10,-10]
    assert list(figs[1].data[1].y) == [2,8]
    assert list(figs[2].data[0].y) == [-10]
    assert list(figs[2].data[1].y) == [8]
    app = create_app(tmp_path)
    table_key = next(k for k in app.callback_map if k.startswith('..trials.data'))
    response = app.server.test_client().post('/_dash-update-component', json=dict(
        output=table_key,
        outputs=[dict(id='trials',property='data'),dict(id='trials',property='selected_rows'),dict(id='table-version',property='data')],
        inputs=[dict(id=i,property=p,value=v) for i,p,v in [
            ('run','value',str(tmp_path)),('stage','value','all'),('accepted','value','accepted'),
            ('bin-filter','data',int(np.searchsorted(edges(), -10, side='right')-1)),('interface','value',0),
            ('metric','value','Fx_N'),('weight','value','trials'),('refresh','n_intervals',0),('part','value','l1'),('extreme','value','minimum')]],
        state=[dict(id='trials',property='selected_rows',value=[]),dict(id='table-version',property='data',value=None)],
        changedPropIds=['bin-filter.data']))
    assert response.status_code == 200
    assert [r['id'] for r in response.json['response']['trials']['data']] == [0]

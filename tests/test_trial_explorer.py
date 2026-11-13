import json
import sqlite3

import pytest

from arm_design.dashboard import create_app, TRIAL_COMPONENT_COLUMNS
from arm_design.fea import METRICS


@pytest.mark.parametrize('scenario,stage', [('campaign','random'), ('restrained','enumeration')])
def test_trial_components_are_numeric_signed_and_scoped_to_connection(tmp_path, scenario, stage):
    refs = [dict(part='l1',kind='mate',label='l1 <- mount'),
            dict(part='l1',kind='mate',label='l1 <- tool'),
            dict(part='l2',kind='mate',label='l2 <- motor')]
    (tmp_path/'run.json').write_text(json.dumps(dict(scenario=scenario,interfaces=refs)))
    rows = [dict(id=7,stage=stage,counted=True,partial=False,
                 minimum=[[-999]*8,[-100,-20,-3,-4,-5,-6,1,1],[-9999]*8],
                 maximum=[[999]*8,[2,30,400,50,60,70,410,105],[9999]*8]),
            dict(id=8,stage=stage,counted=True,minimum=[[0]*8,[-9]*8,[0]*8],
                 maximum=[[0]*8,[10]*8,[0]*8]),
            dict(id=9,stage=stage,counted=True)]
    with sqlite3.connect(tmp_path/'trials.sqlite') as db:
        db.execute('CREATE TABLE trials (id INTEGER PRIMARY KEY, info TEXT)')
        db.executemany('INSERT INTO trials VALUES (?,?)',[(r['id'],json.dumps(r)) for r in rows])
    app = create_app(tmp_path)
    client = app.server.test_client()
    key = next(k for k in app.callback_map if k.startswith('..trials.data'))
    def table(part, interface):
        response = client.post('/_dash-update-component',json=dict(output=key,
            outputs=[dict(id='trials',property='data'),dict(id='trials',property='selected_rows'),dict(id='table-version',property='data')],
            inputs=[dict(id=i,property=p,value=v) for i,p,v in [
                ('run','value',str(tmp_path)),('stage','value',stage),('accepted','value','accepted'),
                ('bin-filter','data',None),('interface','value',interface),('metric','value','Fx_N'),
                ('weight','value','time'),('refresh','n_intervals',0),('part','value',part),('extreme','value','absolute')]],
            state=[dict(id='trials',property='selected_rows',value=[]),dict(id='table-version',property='data',value=None)],
            changedPropIds=['part.value']))
        assert response.status_code == 200
        return response.json['response']['trials']['data']
    result = table('l1',1)
    for j, metric in enumerate(METRICS[:6]):
        for side in ('minimum','maximum'):
            field = f'{metric}_{side}'
            assert result[0][field] == rows[0][side][1][j]
            assert isinstance(result[0][field],float)
            assert result[2][field] is None  # Missing history must not become a zero load.
    assert 'minimum' not in result[0] and 'maximum' not in result[0]
    assert sorted(result[:2],key=lambda r:r['Fx_N_minimum'])[0]['id'] == 7
    assert sorted(result[:2],key=lambda r:r['Fx_N_maximum'],reverse=True)[0]['id'] == 8
    # A stale connection from another link must not leak its loads into the table.
    assert table('l2',1)[0]['Fx_N_minimum'] == -9999

    def find(node, ident):
        if isinstance(node,dict):
            if node.get('props',{}).get('id') == ident:
                return node['props']
            for value in node.values():
                found = find(value,ident)
                if found is not None:
                    return found
        elif isinstance(node,list):
            for value in node:
                found = find(value,ident)
                if found is not None:
                    return found
    props = find(client.get('/_dash-layout').json,'trials')
    assert props['sort_action'] == 'native'
    columns = {c['id']:c for c in props['columns']}
    assert not {'stage','counted','partial'} & columns.keys()
    assert len(TRIAL_COMPONENT_COLUMNS) == 12
    for column in TRIAL_COMPONENT_COLUMNS:
        assert columns[column['id']]['type'] == 'numeric'

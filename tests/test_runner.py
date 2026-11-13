import json
from pathlib import Path

import numpy as np
from dash.exceptions import PreventUpdate
import pytest

import arm_design.dashboard as dashboard
from arm_design.dashboard_jobs import prepare, save_request
from test_dashboard_jobs import RIGHT, values


def callback(app,prefix):
    return next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if k.startswith(prefix))


def test_running_extremes_use_fea_mate_and_exclude_rejected_trials(monkeypatch):
    refs = [dict(part='l1',label='l1 <- proximal'),dict(part='l1',label='l1 <- distal'),
            dict(part='l2',label='l2 <- proximal')]
    def row(i,lo,hi,counted=True):
        low, high = np.zeros((3,8)),np.zeros((3,8))
        low[0,:],high[0,:] = -999,999  # Proximal mate must never leak into l1's chart.
        low[1,:],high[1,:] = lo,hi
        low[2,:],high[2,:] = 2*lo,2*hi
        return dict(id=i,minimum=low.tolist(),maximum=high.tolist(),counted=counted)
    rows = [row(0,-2,3),row(1,-1,2),row(2,-5,4),row(3,-100,100,False)]
    monkeypatch.setattr(dashboard,'run_info',lambda p:dict(scenario='campaign',interfaces=refs))
    monkeypatch.setattr(dashboard,'trials',lambda p,stage,accepted:rows)
    f, records, latest = dashboard.runner_figures('/test','Fy_N')
    assert len(f.data)==4 and latest['id']==3
    assert records[0]['connection']=='l1 <- distal'
    assert records[0]['minimum']==-5 and records[0]['maximum']==4
    assert records[0]['latest_minimum']==-100 and records[0]['latest_maximum']==100
    assert records[1]['minimum']==-10 and records[1]['maximum']==8
    np.testing.assert_array_equal(f.data[0].x,[1,3])
    np.testing.assert_array_equal(f.data[0].y,[3,4])
    np.testing.assert_array_equal(f.data[1].y,[-2,-5])
    assert f.data[0].line.shape=='hv' and f.data[1].line.dash=='dash'
    assert f.data[0].legendgroup==f.data[1].legendgroup
    table=dashboard.runner_table(records,{'l1':{'minimum':-2,'maximum':3}})
    cells=table.children[1].children[0].children
    assert cells[-2].className==cells[-1].className=='new-record'
    assert not cells[1].className


def test_runner_does_not_follow_viewer_and_restores_submitted_settings(tmp_path):
    job=tmp_path/'job-test'
    v=values();v['count']=17
    prepared=prepare(RIGHT,'campaign',v)
    prepared['form']=dashboard.runner_form(RIGHT,'campaign',v)
    save_request(job,prepared)
    run=job/'campaign-test';run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(scenario='campaign',status='running',trials=3,counted=2,rejected=1,
        elapsed_s=10,robot='Test arm',settings=dict(samples=17,analysis_payload=prepared['payload']),interfaces=[])))
    other=tmp_path/'restrained-newer';other.mkdir()
    (other/'run.json').write_text(json.dumps(dict(scenario='restrained',status='complete',trials=2,counted=2,
        robot='Other arm',settings={},interfaces=[])))
    app=dashboard.create_app(tmp_path)
    jobs={'campaign':str(job)}
    monitor=callback(app,'..runner-run.data')
    result=monitor(0,'campaign',jobs)
    assert result[0]==str(run) and result[2:4]==(3,17) and result[-1] is False
    assert '+0.05 kg' in result[5]
    assert callback(app,'..workspace-tab.value')(1,str(run))==('viewer','campaign',str(run))
    restored=callback(app,'..model.value')('campaign',jobs,None)
    fields=dict(zip(dashboard.RUN_FIELDS,restored[:-1]))
    assert fields['payload-mass']==.05 and fields['free-count']==17 and fields['model']==str(RIGHT)
    locked=callback(app,'..runner-settings.disabled')(True)
    assert locked[:3]==(True,True,True) and locked[3][0]['disabled']
    # Switching Runner modes must never stop a run left over from the previous mode.
    assert callback(app,'..launch.disabled')(0,{},'restrained',str(run))[:2]==(False,True)
    callback(app,'cancel-status.children')(1,str(run),{},'restrained')
    assert not (run/'CANCEL').exists()


def test_runner_empty_state_and_polling_preserves_charts(tmp_path):
    app=dashboard.create_app(tmp_path)
    monitor=callback(app,'..runner-run.data')
    result=monitor(0,'campaign',{})
    assert result[0] is None and result[-1] is True and result[1]=='IDLE'
    records=callback(app,'..runner-extremes.figure')
    result=records(0,None,'Fx_N',None)
    assert result[2]=='No completed trials yet.'
    with pytest.raises(PreventUpdate):
        records(1,None,'Fx_N',result[-1])

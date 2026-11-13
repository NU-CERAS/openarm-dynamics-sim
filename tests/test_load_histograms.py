import base64
import json
from xml.etree import ElementTree as ET

import numpy as np
import plotly.graph_objects as go
import pytest

from arm_design.analysis import edges
from arm_design.load_histograms import histogram_counts, percentile_text, preview_svg, split_histogram


def test_signed_panels_preserve_counts_bin_ids_zero_and_overflow():
    e=edges()
    values=[-2e6,-3,0,2,2e6]
    ids=np.searchsorted(e,values,side='right')-1
    f=go.Figure(go.Bar(x=ids,y=[10,20,30,25,15],customdata=ids,name='Selected run'))
    f.update_layout(title='Distribution',yaxis_title='% of simulated time')
    result=split_histogram(f,'Fx_N',dict(side='minimum',value=-2e6))
    assert list(result.data[0].customdata) == list(ids[:2])
    assert list(result.data[1].customdata) == list(ids[2:])
    assert sum(result.data[0].y) == 30
    assert sum(result.data[1].y) == 70
    assert result.layout.yaxis2.matches == 'y'
    assert any('Selected minimum' in a.text for a in result.layout.annotations)
    assert any(s.x0==0 and s.type=='line' for s in result.layout.shapes)
    # Comparisons retain their own full-distribution denominator on both sides.
    f.add_bar(x=ids,y=[20,20,20,20,20],customdata=ids,name='Compared run')
    result=split_histogram(f,'Fx_N')
    for name in ('Selected run','Compared run'):
        assert sum(sum(t.y) for t in result.data if t.name==name) == 100


def test_trial_previews_use_each_columns_statistic_and_binned_percentiles():
    stats=dict(counts=np.zeros((1,8,len(edges())-1)))
    rows=[dict(minimum=[[-10]*8],maximum=[[2]*8]),dict(minimum=[[-3]*8],maximum=[[8]*8])]
    negative=histogram_counts(stats,rows,0,0,'trials','minimum')
    positive=histogram_counts(stats,rows,0,0,'trials','maximum')
    assert list(np.flatnonzero(negative)) == list(np.searchsorted(edges(),[-10,-3],side='right')-1)
    assert list(np.flatnonzero(positive)) == list(np.searchsorted(edges(),[2,8],side='right')-1)
    assert negative.sum()==positive.sum()==2
    summary=percentile_text(negative)
    assert 'Median ≈ -' in summary and 'P5–P95' in summary and 'binned' in summary
    uri=preview_svg(negative,-10)
    root=ET.fromstring(base64.b64decode(uri.split(',')[1]))
    assert root.tag.endswith('svg')
    assert 'No samples' in percentile_text(np.zeros(len(edges())-1))
    negative[:]=0
    negative[0]=1
    assert '≤' in percentile_text(negative)
    negative[:]=0
    negative[-1]=1
    assert '≥' in percentile_text(negative)


def test_tail_percentiles_condition_on_sign_and_move_outward():
    from arm_design.load_histograms import tail_percentile
    values = [-1]*90 + [-100]*10 + [10]*10000
    counts = np.histogram(values,edges())[0]
    median = tail_percentile(counts,'minimum',.5,-100,10)
    p95 = tail_percentile(counts,'minimum',.95,-100,10)
    assert -100 <= p95['value'] < median['value'] < 0
    assert p95['share'] == pytest.approx(100/10100)
    assert p95['tail_fraction'] == pytest.approx(5/10100)
    assert p95['interval'][0] <= p95['value'] <= p95['interval'][1]
    assert not p95['exact']
    assert tail_percentile(counts,'minimum',1,-100,10)['value'] == -100
    assert tail_percentile(counts,'maximum',1,-100,10)['value'] == 10
    assert tail_percentile(counts,'maximum',1,-100,10)['exact']
    # Changing the opposite side's frequency cannot change the conditional quantile.
    counts2 = np.histogram([-1]*90+[-100]*10+[10],edges())[0]
    assert tail_percentile(counts2,'minimum',.95,-100,10)['value'] == p95['value']


def test_tail_percentiles_handle_zero_missing_signs_and_open_bins():
    from arm_design.load_histograms import tail_percentile
    counts=np.histogram([0]*100,edges())[0]
    assert tail_percentile(counts,'minimum',.95,0,0) is None
    zero=tail_percentile(counts,'maximum',.95,0,0)
    assert zero['value']==0 and zero['share']==1 and zero['tail_fraction']==0
    counts=np.histogram([-2e6,2e6],edges())[0]
    for side in ('minimum','maximum'):
        result=tail_percentile(counts,side,.95,-2e6,2e6)
        assert np.isfinite(result['interval']).all()
        assert -2e6 <= result['value'] <= 2e6
    with pytest.raises(ValueError):
        tail_percentile(counts,'minimum',1.1,-2e6,2e6)


def test_percentile_histogram_marks_threshold_and_shades_only_selected_tail():
    counts=np.histogram([-100,-20,-1,2,20,100],edges())[0]
    ids=np.flatnonzero(counts)
    hist=go.Figure(go.Bar(x=ids,y=counts[ids],name='Selected run'))
    focus=dict(kind='percentile',side='minimum',q=.95,value=-90)
    chart=split_histogram(hist,'Fx_N',focus)
    assert any('P95 negative threshold' in a.text for a in chart.layout.annotations)
    shade=[s for s in chart.layout.shapes if s.type=='rect' and s.opacity==.12]
    assert len(shade)==1 and shade[0].xref=='x'
    root=ET.fromstring(base64.b64decode(preview_svg(counts,-90,'minimum').split(',')[1]))
    assert any(e.attrib.get('opacity')=='.12' for e in root.iter())

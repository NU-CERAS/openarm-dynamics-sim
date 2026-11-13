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

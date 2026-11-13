"""Shared signed histogram counts, binned percentiles, and compact previews."""
import base64
from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import edges

NEGATIVE = '#ffb45b'
POSITIVE = '#25c9a3'


def trial_extreme(row, i, j, extreme='absolute'):
    high = row['maximum'][i][j]
    if j >= 6 or 'minimum' not in row or extreme == 'maximum':
        return high
    low = row['minimum'][i][j]
    return low if extreme == 'minimum' or abs(low) > abs(high) else high


def histogram_counts(stats, rows, i, j, weighting='time', extreme='absolute'):
    counts = np.asarray(stats['counts'][i,j]).copy()
    if weighting == 'trials':
        counts[:] = 0
        for row in rows:
            if 'maximum' in row:
                index = np.searchsorted(edges(),trial_extreme(row,i,j,extreme),side='right')-1
                counts[np.clip(index,0,len(counts)-1)] += 1
    return counts


def percentile_text(counts):
    if not np.sum(counts):
        return 'No samples in this selection.'
    e, cumulative = edges(), np.cumsum(counts)
    def estimate(q):
        index = min(len(counts)-1,int(np.searchsorted(cumulative,q*cumulative[-1],side='left')))
        a,b = e[index:index+2]
        if not np.isfinite(a):
            return f'≤ {b:+.3g}'
        if not np.isfinite(b):
            return f'≥ {a:+.3g}'
        return f'{(a+b)/2:+.3g}'
    return f'Median ≈ {estimate(.5)} · P5–P95 ≈ {estimate(.05)} to {estimate(.95)} (binned estimates)'


def preview_svg(counts, value):
    """SVG uses the same signed bin indices as the full chart, including overflow."""
    counts = np.asarray(counts)
    selected = int(np.clip(np.searchsorted(edges(),value,side='right')-1,0,len(counts)-1))
    zero = int(np.searchsorted(edges(),0))
    used = np.flatnonzero(counts)
    lo,hi = min([selected,zero-1,*used]), max([selected,zero,*used])
    width,height,left,right,top,bottom = 360,155,24,12,20,30
    step = (width-left-right)/(hi-lo+1)
    plot_h = height-top-bottom
    highest = max(float(counts.max()),1)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#101b29"/>']
    for index in range(lo,hi+1):
        bar = plot_h*counts[index]/highest
        color = NEGATIVE if edges()[index+1]<=0 else POSITIVE
        parts.append(f'<rect x="{left+(index-lo)*step:.2f}" y="{height-bottom-bar:.2f}" width="{max(.4,step-.5):.2f}" height="{bar:.2f}" fill="{color}"/>')
    xzero=left+(zero-lo)*step
    parts.append(f'<path d="M{xzero:.2f} {top}V{height-bottom}" stroke="#a7b9ca" stroke-dasharray="3 3"/>')
    x=left+(selected-lo+.5)*step
    parts.append(f'<path d="M{x:.2f} {top}V{height-bottom}" stroke="#fff" stroke-width="2"/>')
    parts.append(f'<text x="{x:.2f}" y="12" fill="#fff" text-anchor="middle" font-size="10">{escape(f"{value:+.4g}")}</text>')
    parts.extend([f'<text x="{left}" y="{height-8}" fill="{NEGATIVE}" font-size="11">Negative</text>',
                  f'<text x="{xzero:.2f}" y="{height-8}" fill="#d9e5ef" text-anchor="middle" font-size="10">0</text>',
                  f'<text x="{width-right}" y="{height-8}" fill="{POSITIVE}" text-anchor="end" font-size="11">Positive / zero</text>','</svg>'])
    return 'data:image/svg+xml;base64,'+base64.b64encode(''.join(parts).encode()).decode()


def split_histogram(hist, metric, focus=None):
    """Partition existing series without renormalizing or changing bin-filter IDs."""
    if metric in ('force_N','moment_Nm'):
        hist.update_layout(height=320)
        return hist
    result = make_subplots(rows=1,cols=2,shared_yaxes=True,horizontal_spacing=.09,
                           subplot_titles=('Negative (< 0)','Positive / zero (≥ 0)'))
    result.update_layout(template='plotly_dark',paper_bgcolor='#101b29',plot_bgcolor='#101b29',
        title=dict(text=hist.layout.title.text,font_size=15),font=dict(family='Arial',color='#cbd8e5'),
        height=320,margin=dict(l=65,r=20,t=90,b=70),barmode=hist.layout.barmode or 'group',
        legend=dict(orientation='h',x=0,y=1.25,font_size=10),uirevision=f'{metric}-signed')
    e=edges()
    for col in (1,2):
        occupied=set()
        for series_index,trace in enumerate(hist.data):
            indices=np.asarray(trace.x,dtype=int)
            selected=e[indices+1]<=0 if col==1 else e[indices]>=0
            ids=indices[selected]
            values=np.asarray(trace.y)[selected]
            occupied.update(int(i) for i,y in zip(ids,values) if y)
            labels=[f'{e[k]:+.4g} … {e[k+1]:+.4g}' for k in ids]
            result.add_trace(go.Bar(x=ids,y=values,customdata=ids,text=labels,textposition='none',name=trace.name,
                legendgroup=trace.name,showlegend=len(hist.data)>1 and col==2,
                marker_color=(NEGATIVE if col==1 else POSITIVE) if series_index==0 else '#70aaff',
                opacity=.65 if len(hist.data)>1 else 1,
                hovertemplate='%{text}<br>%{y:.3g}%<extra>%{fullData.name}</extra>'),row=1,col=col)
        ticks=sorted(occupied)[::max(1,len(occupied)//5)]
        result.update_xaxes(tickmode='array',tickvals=ticks,
            ticktext=[f'{e[k]:+.3g} … {e[k+1]:+.3g}' for k in ticks],tickangle=-25,
            title=metric.replace('_',' ')+' · signed bins',row=1,col=col)
        if occupied:
            result.update_xaxes(range=[min(occupied)-.7,max(occupied)+.7],row=1,col=col)
        else:
            result.add_annotation(text='No samples on this side',x=.5,y=.45,
                xref='x domain' if col==1 else 'x2 domain',yref='y domain' if col==1 else 'y2 domain',showarrow=False)
    result.update_yaxes(title=hist.layout.yaxis.title.text,row=1,col=1)
    result.update_yaxes(rangemode='tozero')
    if focus:
        col=1 if focus['value']<0 else 2
        index=int(np.clip(np.searchsorted(e,focus['value'],side='right')-1,0,len(e)-2))
        result.add_vline(x=index,line_color='#fff',line_dash='dot',row=1,col=col)
        result.add_annotation(text=f"Selected {focus['side']}: {focus['value']:+.5g}",
            x=.5,y=.98,yanchor='top',bgcolor='#101b29',xref='x domain' if col==1 else 'x2 domain',yref='y domain' if col==1 else 'y2 domain',showarrow=False,font_color='#fff')
        domain=result.layout.xaxis.domain if col==1 else result.layout.xaxis2.domain
        result.add_shape(type='rect',x0=domain[0],x1=domain[1],y0=0,y1=1,xref='paper',yref='paper',
                         line=dict(color='#d9e5ef',width=1),fillcolor='rgba(0,0,0,0)')
    return result

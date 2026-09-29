import numpy as np
import plotly.graph_objects as go
import plotly.express as px

LABELS={'duration_ms':'Duration (ms)','mean_blockade_nA':'Mean blockade (nA)','peak_blockade_nA':'Peak blockade (nA)',
'ecd_nA_ms':'Event charge deficit (nA·ms)','waveform_rmse_nA':'Waveform disagreement, RMSE (nA)',
'relative_rmse':'RMSE / signal RMS (dimensionless)','peak_error_nA':'Fit peak − measured peak (nA)',
'area_error_pct':'Fit area error (%)','segments':'Saved segment count'}

def trace_figure(e,ref=None,zoom=True):
    x=(e.time-e.bounds[0])*1000;f=go.Figure()
    f.add_trace(go.Scatter(x=x,y=e.baseline-e.current,name='Saved trace',line=dict(color='#35a8c2',width=1.4)))
    if e.fit is not None:f.add_trace(go.Scatter(x=x,y=e.baseline-e.fit,name='Original fit',line=dict(color='#f7a04a',shape='hv')))
    if ref is not None:f.add_trace(go.Scatter(x=x,y=e.baseline-ref['fit'],name='Candidate: '+ref['method'],line=dict(color='#a775e8',shape='linear' if ref['method']=='Rounded pulse (Gaussian)' else 'hv')))
    f.add_vline(x=0,line_dash='dot');dur=float(np.diff(e.bounds)[0])*1000;f.add_vline(x=dur,line_dash='dot')
    if zoom:f.update_xaxes(range=[-max(.02,dur*.1),dur+max(.02,dur*.1)])
    f.update_layout(height=370,xaxis_title='Time from detected start (ms)',yaxis_title='Blockade (nA)',legend=dict(orientation='h'),margin=dict(t=20,b=25))
    return f

def distribution_figures(df,x,y,logx=False,logy=False,bins=45,color=None):
    good=np.isfinite(df[x])&np.isfinite(df[y])
    if logx:good&=df[x]>0
    if logy:good&=df[y]>0
    d=df.loc[good].copy();dropped=len(df)-len(d)
    if len(d)==0:raise ValueError('No finite values compatible with the selected scales.')
    if color and color in d:d[color]=d[color].astype(str)
    hover=[c for c in ['event_index','dataset_row','start_s'] if c in d and c not in [x,y]]
    scatter=px.scatter(d,x=x,y=y,color=color,log_x=logx,log_y=logy,hover_data=hover,labels=LABELS,opacity=.65,render_mode='webgl')
    def edges(v,log):
        low,high=float(v.min()),float(v.max())
        if low==high:
            low=low*.9 if log else low-.5;high=high*1.1 if log else high+.5
        return np.geomspace(low,high,bins+1) if log else np.linspace(low,high,bins+1)
    xe=edges(d[x],logx);ye=edges(d[y],logy);h,_,_=np.histogram2d(d[x],d[y],bins=[xe,ye])
    xc=np.sqrt(xe[:-1]*xe[1:]) if logx else (xe[:-1]+xe[1:])/2
    yc=np.sqrt(ye[:-1]*ye[1:]) if logy else (ye[:-1]+ye[1:])/2
    heat=go.Figure(go.Heatmap(x=xc,y=yc,z=h.T,colorscale='Viridis',colorbar=dict(title='Events/bin'),hovertemplate='x=%{x:.4g}<br>y=%{y:.4g}<br>Count=%{z}<extra></extra>'))
    heat.update_layout(xaxis_title=LABELS.get(x,x),yaxis_title=LABELS.get(y,y))
    heat.update_xaxes(type='log' if logx else 'linear');heat.update_yaxes(type='log' if logy else 'linear')
    return scatter,heat,dropped,h,xe,ye

def profile_figure(profiles,labels,centers,units='nA'):
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1];f=go.Figure()
    palette=px.colors.qualitative.Plotly
    for j,c in enumerate(centers):
        color=palette[j%len(palette)];p=profiles[labels==j];lo,hi=np.percentile(p,[10,90],axis=0)
        f.add_trace(go.Scatter(x=phase,y=lo,line=dict(width=0),showlegend=False,legendgroup=str(j),hoverinfo='skip'))
        f.add_trace(go.Scatter(x=phase,y=hi,fill='tonexty',line=dict(width=0),opacity=.12,fillcolor=color,showlegend=False,legendgroup=str(j),hoverinfo='skip'))
        f.add_trace(go.Scatter(x=phase,y=c,line=dict(color=color,width=2),name=f'Group {j} (n={len(p)})',legendgroup=str(j)))
    f.update_layout(xaxis_title='Fraction of event duration',yaxis_title=f'Profile ({units})',height=420)
    return f

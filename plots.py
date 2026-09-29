import numpy as np
import plotly.graph_objects as go
import plotly.express as px

LABELS={'duration_ms':'Duration (ms)','mean_blockade_nA':'Mean blockade (nA)','peak_blockade_nA':'Peak blockade (nA)',
'ecd_nA_ms':'Event charge deficit (nA·ms)','waveform_rmse_nA':'Waveform disagreement, RMSE (nA)',
'relative_rmse':'RMSE / signal RMS (dimensionless)','peak_error_nA':'Fit peak − measured peak (nA)',
'area_error_pct':'Fit area error (%)','segments':'Saved segment count'}

def trace_figure(e,ref=None,zoom=True,current=False):
    x=(e.time-e.bounds[0])*1000;f=go.Figure()
    f.add_trace(go.Scatter(x=x,y=e.current if current else e.baseline-e.current,name='Measured trace',line=dict(color='#35a8c2',width=1.4)))
    if e.fit is not None:f.add_trace(go.Scatter(x=x,y=e.fit if current else e.baseline-e.fit,name='Uploaded fit',line=dict(color='#f7a04a',shape='hv')))
    if ref is not None:f.add_trace(go.Scatter(x=x,y=ref['fit'] if current else e.baseline-ref['fit'],name='Refined: '+ref['method'],line=dict(color='#a775e8',shape='linear' if ref['method']=='Rounded pulse (Gaussian)' else 'hv')))
    f.add_vline(x=0,line_dash='dot');dur=float(np.diff(e.bounds)[0])*1000;f.add_vline(x=dur,line_dash='dot')
    if zoom:f.update_xaxes(range=[-max(.02,dur*.1),dur+max(.02,dur*.1)])
    f.update_layout(height=370,xaxis_title='Time from detected start (ms)',yaxis_title='Current I (nA)' if current else 'Current blockade ΔI (nA)',legend=dict(orientation='h'),margin=dict(t=20,b=25))
    return f

def distribution_figures(df,x,y,logx=False,logy=False,bins=45,color=None):
    good=np.isfinite(df[x])&np.isfinite(df[y])
    if logx:good&=df[x]>0
    if logy:good&=df[y]>0
    d=df.loc[good].copy();dropped=len(df)-len(d)
    if len(d)==0:raise ValueError('No finite values compatible with the selected scales.')
    if color and color in d:d[color]=d[color].astype(str)
    hover=[c for c in ['event_index','dataset_row','start_s'] if c in d and c not in [x,y]]
    scatter=px.scatter(d,x=x,y=y,color=color,log_x=logx,log_y=logy,hover_data=hover,labels=LABELS,color_discrete_sequence=PALETTE,opacity=.65,render_mode='svg')
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
    palette=PALETTE
    for j,c in enumerate(centers):
        color=palette[j%len(palette)];p=profiles[labels==j];lo,hi=np.percentile(p,[10,90],axis=0)
        f.add_trace(go.Scatter(x=phase,y=lo,line=dict(width=0),showlegend=False,legendgroup=str(j),hoverinfo='skip'))
        f.add_trace(go.Scatter(x=phase,y=hi,fill='tonexty',line=dict(width=0),opacity=.12,fillcolor=color,showlegend=False,legendgroup=str(j),hoverinfo='skip'))
        f.add_trace(go.Scatter(x=phase,y=c,line=dict(color=color,width=2),name=f'Group {j} (n={len(p)})',legendgroup=str(j)))
    f.update_layout(xaxis_title='Fraction of event duration',yaxis_title=f'Profile ({units})',height=420)
    return f


PALETTE=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#332288','#882255','#44AA99','#999933']
def scientific(fig):
    fig.update_layout(template='plotly_white',font=dict(family='Arial, sans-serif',size=15,color='black'),
        paper_bgcolor='white',plot_bgcolor='white',colorway=PALETTE,margin=dict(l=75,r=35,t=45,b=70),
        legend=dict(title_text='',font=dict(color='#18232b'),bgcolor='rgba(255,255,255,0.95)'),
        title_font_color='#18232b',hoverlabel=dict(bgcolor='white',font_color='#18232b'))
    fig.update_xaxes(showline=True,linewidth=1,linecolor='black',ticks='outside',showgrid=False,zeroline=False,mirror=True,title_font_color='#18232b',tickfont_color='#18232b')
    fig.update_yaxes(showline=True,linewidth=1,linecolor='black',ticks='outside',showgrid=False,zeroline=False,mirror=True,title_font_color='#18232b',tickfont_color='#18232b')
    fig.update_annotations(font_color='#18232b')
    for trace in fig.data:
        if hasattr(trace,'colorbar'):
            trace.update(colorbar=dict(tickfont=dict(color='#18232b'),title=dict(font=dict(color='#18232b'))))
    fig.update_coloraxes(colorbar_tickfont_color='#18232b',colorbar_title_font_color='#18232b')
    return fig

def figure_archive(table,x,y,logx=False,color=None):
    import io,zipfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good=np.isfinite(table[x])&np.isfinite(table[y])
    if logx:good&=table[x]>0
    d=table.loc[good]
    if not len(d):raise ValueError('No valid points to export.')
    out=io.BytesIO()
    with plt.rc_context({'font.family':'sans-serif','font.size':10,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42,'xtick.direction':'out','ytick.direction':'out'}):
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for kind in ['scatter','counts']:
                fig,ax=plt.subplots(figsize=(5.2,4.2),layout='constrained')
                if kind=='scatter':
                    if color:
                        for j,(label,g) in enumerate(d.groupby(color,sort=True)):
                            ax.scatter(g[x],g[y],s=10,alpha=.65,edgecolors='none',color=PALETTE[j%len(PALETTE)],label=f'Cluster {label}')
                        ax.legend(frameon=False,fontsize=8)
                    else:ax.scatter(d[x],d[y],s=10,alpha=.6,edgecolors='none',color=PALETTE[0])
                else:
                    lo,hi=d[x].min(),d[x].max()
                    if lo==hi:lo,hi=(lo*.9,hi*1.1) if logx else (lo-.5,hi+.5)
                    xe=np.geomspace(lo,hi,46) if logx else np.linspace(lo,hi,46)
                    lo,hi=d[y].min(),d[y].max()
                    if lo==hi:lo,hi=lo-.5,hi+.5
                    ye=np.linspace(lo,hi,46);h,_,_=np.histogram2d(d[x],d[y],bins=[xe,ye])
                    mesh=ax.pcolormesh(xe,ye,h.T,cmap='viridis',shading='flat');fig.colorbar(mesh,ax=ax,label='Events per bin')
                if logx:ax.set_xscale('log')
                ax.set_xlabel(LABELS.get(x,x));ax.set_ylabel(LABELS.get(y,y))
                for ext in ['pdf','svg','png']:
                    b=io.BytesIO();fig.savefig(b,format=ext,dpi=600);z.writestr(kind+'.'+ext,b.getvalue())
                plt.close(fig)
            z.writestr('plotted_events.csv',d.to_csv(index=False))
            z.writestr('README.txt',f'Scatter and event-count heatmap. {len(d)} plotted, {len(table)-len(d)} omitted. x={x}, y={y}, log x={logx}. SVG/PDF vector exports; PNG 600 dpi. Heatmap uses 45 bins per axis. No smoothing or spectral density. CSV contains the plotted rows.\n')
    return out.getvalue()

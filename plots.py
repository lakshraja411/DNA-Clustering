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


def profile_archive(profiles,labels,centers):
    """Export member bands and representative profiles without scatter plots."""
    import io,zipfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    profiles=np.asarray(profiles);labels=np.asarray(labels);centers=np.asarray(centers)
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1]
    out=io.BytesIO();rows=[]
    with plt.rc_context({'font.family':'sans-serif','font.size':10,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42,'xtick.direction':'out','ytick.direction':'out'}):
        fig,ax=plt.subplots(figsize=(5.6,4.2),layout='constrained')
        for j,center in enumerate(centers):
            members=profiles[labels==j];lo,hi=np.percentile(members,[10,90],axis=0);color=PALETTE[j%len(PALETTE)]
            ax.fill_between(phase,lo,hi,color=color,alpha=.12)
            ax.plot(phase,center,color=color,lw=1.5,label=f'Cluster {j} (n={len(members)})')
            for pos,t in enumerate(phase):rows.append({'cluster':j,'phase':t,'representative_blockade_nA':center[pos],'p10_nA':lo[pos],'p90_nA':hi[pos],'events':len(members)})
        ax.set_xlabel('Fraction of event duration');ax.set_ylabel('Current blockade (nA)');ax.legend(frameon=False,fontsize=8)
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for ext in ['pdf','svg','png']:
                b=io.BytesIO();fig.savefig(b,format=ext,dpi=600);z.writestr('cluster_profiles.'+ext,b.getvalue())
            z.writestr('profile_summary.csv',pd.DataFrame(rows).to_csv(index=False))
            z.writestr('README.txt','Profiles from the selected clustering signal. Bands are 10th–90th member percentiles, not confidence intervals. For DTW the representative curves are aligned barycentres while percentile bands use unwarped profiles. Absolute duration is removed from these profiles; amplitude is retained.\n')
        plt.close(fig)
    return out.getvalue()


def hull_vertices(points):
    """Convex outline in the displayed 2D projection; skip degenerate groups."""
    from scipy.spatial import ConvexHull,QhullError
    points=np.unique(np.asarray(points,float),axis=0)
    if len(points)<3 or np.linalg.matrix_rank(points-points.mean(axis=0),tol=1e-10)<2:return None
    try:return points[ConvexHull(points).vertices]
    except QhullError:return None


def cluster_pca_figure(projection,axis_labels,outlines=True):
    f=go.Figure()
    for j,(label,g) in enumerate(projection.groupby('Cluster',sort=True)):
        color=PALETTE[j%len(PALETTE)];points=g[['PC1','PC2']].to_numpy();poly=hull_vertices(points)
        if outlines and poly is not None:
            closed=np.vstack([poly,poly[0]])
            rgb=tuple(int(color[h:h+2],16) for h in (1,3,5))
            f.add_trace(go.Scatter(x=closed[:,0],y=closed[:,1],mode='lines',line=dict(color=color,width=1),fill='toself',fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)',showlegend=False,hoverinfo='skip'))
        f.add_trace(go.Scatter(x=points[:,0],y=points[:,1],mode='markers',name=f'Cluster {label} (n={len(g)})',marker=dict(color=color,size=5,opacity=.7),
            customdata=g[['Event ID','Duration (ms)','Measured mean blockade (nA)']].to_numpy(),hovertemplate='Event %{customdata[0]:.0f}<br>Duration %{customdata[1]:.4g} ms<br>Measured mean blockade %{customdata[2]:.4g} nA<br>PC1 %{x:.4g}<br>PC2 %{y:.4g}<extra>%{fullData.name}</extra>'))
        mean=points.mean(axis=0)
        f.add_trace(go.Scatter(x=[mean[0]],y=[mean[1]],mode='markers',marker=dict(symbol='x',size=11,color='black'),name=f'Cluster {label} mean position',showlegend=False,hovertemplate='Mean position in displayed PCA coordinates<extra></extra>'))
    f.update_layout(xaxis_title=axis_labels['PC1'],yaxis_title=axis_labels['PC2'],height=460)
    return f


def member_profile_figure(profiles,labels,centers,group,limit=40):
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1]
    members=np.flatnonzero(np.asarray(labels)==group);rng=np.random.default_rng(42)
    chosen=np.sort(rng.choice(members,min(limit,len(members)),replace=False))
    f=go.Figure()
    for pos in chosen:
        f.add_trace(go.Scatter(x=phase,y=profiles[pos],mode='lines',line=dict(color='rgba(0,114,178,0.16)',width=.8),showlegend=False,hoverinfo='skip'))
    f.add_trace(go.Scatter(x=phase,y=centers[group],mode='lines',line=dict(color='#D62728',width=2.5),name='Representative profile'))
    f.update_layout(title=f'Cluster {group} · n={len(members)}',xaxis_title='Fraction of event duration',yaxis_title='Blockade (nA)',height=300)
    low=min(profiles.min(),centers.min());high=max(profiles.max(),centers.max());pad=max(.05,.06*(high-low))
    f.update_yaxes(range=[low-pad,high+pad])
    return f


def publication_archive(info,event_ids,source,method,outlines=True,time_examples=None):
    """Paper-style composite: PCA, cluster members and representative comparison."""
    import io,json,zipfile,math
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    embedding=np.asarray(info['embedding']);profiles=np.asarray(info['profiles']);labels=np.asarray(info['labels']);centers=np.asarray(info['centers'])
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1];k=len(centers);rows=math.ceil((k+1+(time_examples is not None))/3)
    out=io.BytesIO();rng=np.random.default_rng(42);selected={};summary=[]
    with plt.rc_context({'font.family':'sans-serif','font.size':9,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42,'xtick.direction':'out','ytick.direction':'out','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'}):
        fig=plt.figure(figsize=(9,3.4+2.5*rows),layout='constrained');grid=fig.add_gridspec(rows+1,3,height_ratios=[1.3]+[1]*rows)
        ax=fig.add_subplot(grid[0,:]);variance=info.get('pca_variance',[])
        for j in range(k):
            members=np.flatnonzero(labels==j);points=embedding[members];color=PALETTE[j%len(PALETTE)];poly=hull_vertices(points)
            if outlines and poly is not None:ax.fill(poly[:,0],poly[:,1],color=color,alpha=.12);closed=np.vstack([poly,poly[0]]);ax.plot(closed[:,0],closed[:,1],color=color,lw=.8)
            ax.scatter(points[:,0],points[:,1],color=color,s=10,alpha=.7,edgecolors='none',label=f'Cluster {j} (n={len(members)})')
            mean=points.mean(axis=0);ax.scatter(*mean,marker='x',color='black',s=40,lw=1.4)
        ax.set_xlabel(f'PC1 ({100*variance[0]:.1f}% variance)' if variance else 'PC1');ax.set_ylabel(f'PC2 ({100*variance[1]:.1f}% variance)' if len(variance)>1 else 'PC2 (zero if only one PC retained)');ax.legend(frameon=False,fontsize=8,ncol=min(k,5),loc='upper center',bbox_to_anchor=(.5,1.17));ax.set_title('a) PCA projection',loc='left',pad=25)
        # Common profile y scale permits honest comparison across cluster panels.
        low=min(profiles.min(),centers.min());high=max(profiles.max(),centers.max());pad=max(.05,.06*(high-low));axes=[]
        for j in range(k):
            a=fig.add_subplot(grid[1+j//3,j%3]);members=np.flatnonzero(labels==j);chosen=np.sort(rng.choice(members,min(40,len(members)),replace=False));selected[str(j)]=np.asarray(event_ids)[chosen].tolist()
            for pos in chosen:a.plot(phase,profiles[pos],color='#0072B2',alpha=.14,lw=.6)
            a.plot(phase,centers[j],color='#D62728',lw=1.7)
            a.set_title(f'{chr(98+j)}) Cluster {j} (n={len(members)})',loc='left');a.set_xlabel('Fraction of event duration');a.set_ylabel('Blockade (nA)');a.set_ylim(low-pad,high+pad);axes.append(a)
            for i,t in enumerate(phase):summary.append({'cluster':j,'phase':t,'representative_blockade_nA':centers[j,i]})
        index=k;a=fig.add_subplot(grid[1+index//3,index%3])
        for j,c in enumerate(centers):a.plot(phase,c,color=PALETTE[j%len(PALETTE)],lw=1.4,label=f'Cluster {j}')
        a.set_title(f'{chr(98+k)}) Representative comparison',loc='left');a.set_xlabel('Fraction of event duration');a.set_ylabel('Blockade (nA)');a.set_ylim(low-pad,high+pad);a.legend(frameon=False,fontsize=7)
        if time_examples is not None:
            index=k+1;a=fig.add_subplot(grid[1+index//3,index%3])
            for j,(group,sub) in enumerate(time_examples.groupby('cluster',sort=True)):
                a.plot(sub.time_ms,sub.blockade_nA,color=PALETTE[j%len(PALETTE)],lw=1.2,label=f'Cluster {group}, event {sub.event_id.iloc[0]}')
            a.set_title(f'{chr(99+k)}) Example events in actual time',loc='left');a.set_xlabel('Time from event start (ms)');a.set_ylabel('Blockade (nA)');a.legend(frameon=False,fontsize=7)
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for ext in ['pdf','svg','png']:
                b=io.BytesIO();fig.savefig(b,format=ext,dpi=600,facecolor='white');z.writestr('cluster_figure.'+ext,b.getvalue())
            z.writestr('pca_coordinates.csv',pd.DataFrame({'event_id':event_ids,'cluster':labels,'PC1':embedding[:,0],'PC2':embedding[:,1]}).to_csv(index=False))
            if time_examples is not None:z.writestr('actual_time_examples.csv',time_examples.to_csv(index=False))
            z.writestr('representative_profiles.csv',pd.DataFrame(summary).to_csv(index=False))
            z.writestr('figure_settings.json',json.dumps({'source':source,'method':method,'outlines':outlines,'displayed_member_ids':selected,'member_sample_seed':42,'members_per_panel_limit':40,'x_axis':'fraction of event duration','centres':'DTW barycentres' if 'DTW' in method else 'mean member profiles'},indent=2))
            z.writestr('CAPTION.txt',f'PCA view of {len(labels)} events grouped using {method}, signal source {source}. Shaded regions are convex hulls in the displayed 2D coordinates, not confidence regions or decision boundaries; x markers are mean projected group positions, not physical identities. Cluster panels show up to 40 uniformly sampled member profiles (seed 42) behind representative curves computed from all members. Representatives are mean profiles except for DTW, which uses aligned barycentres; DTW members are shown unwarped. Time is normalised separately for each event, so these curves do not represent absolute dwell time. Amplitude remains in nA. Feature clustering can use more PCs than shown; waveform clustering uses PCA only for display. The actual-time panel shows one real event per group nearest to its representative by pointwise profile distance; these are example events, not averaged centroids. Their original time sampling and detected durations are preserved, aligned at detected start. No physical/topological identity is inferred.\n')
        plt.close(fig)
    return out.getvalue()


def representative_time_examples(events,info,source):
    """One real event per group, ranked by pointwise profile distance."""
    import pandas as pd
    rows=[];profiles=np.asarray(info['profiles']);labels=np.asarray(info['labels'])
    for group,center in enumerate(info['centers']):
        positions=np.flatnonzero(labels==group)
        pos=positions[np.argmin(np.linalg.norm(profiles[positions]-center,axis=1))]
        e=events[pos];signal=e.current if source=='Measured trace' else e.fit
        m=(e.time>=e.bounds[0])&(e.time<e.bounds[1])
        for t,y in zip((e.time[m]-e.bounds[0])*1000,(e.baseline-signal)[m]):
            rows.append({'cluster':group,'event_id':e.index,'time_ms':float(t),'blockade_nA':float(y)})
    return pd.DataFrame(rows)


def time_example_figure(table):
    f=go.Figure()
    for j,(group,sub) in enumerate(table.groupby('cluster',sort=True)):
        f.add_trace(go.Scatter(x=sub.time_ms,y=sub.blockade_nA,mode='lines',line=dict(color=PALETTE[j%len(PALETTE)],width=1.7),name=f'Cluster {group} · event {sub.event_id.iloc[0]}'))
    f.update_layout(xaxis_title='Time from detected event start (ms)',yaxis_title='Current blockade (nA)',height=420)
    return f

RELEASE_VERSION='0.7.0'
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
    """Overlay only the representative median profile from each feature-space cluster."""
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1];f=go.Figure()
    for j,c in enumerate(centers):
        n=int(np.sum(np.asarray(labels)==j))
        f.add_trace(go.Scatter(x=phase,y=c,line=dict(color=PALETTE[j%len(PALETTE)],width=2.2),
                               name=f'Cluster {j} (n={n})'))
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

def figure_archive(table,x,y,logx=False,color=None,bins=45):
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
                    xe=np.geomspace(lo,hi,bins+1) if logx else np.linspace(lo,hi,bins+1)
                    lo,hi=d[y].min(),d[y].max()
                    if lo==hi:lo,hi=lo-.5,hi+.5
                    ye=np.linspace(lo,hi,bins+1);h,_,_=np.histogram2d(d[x],d[y],bins=[xe,ye])
                    mesh=ax.pcolormesh(xe,ye,h.T,cmap='viridis',shading='flat');fig.colorbar(mesh,ax=ax,label='Events per bin')
                if logx:ax.set_xscale('log')
                ax.set_xlabel(LABELS.get(x,x));ax.set_ylabel(LABELS.get(y,y))
                for ext in ['pdf','svg','png']:
                    b=io.BytesIO();fig.savefig(b,format=ext,dpi=600);z.writestr(kind+'.'+ext,b.getvalue())
                plt.close(fig)
            z.writestr('plotted_events.csv',d.to_csv(index=False))
            z.writestr('README.txt',f'Scatter and event-count heatmap. {len(d)} plotted, {len(table)-len(d)} omitted. x={x}, y={y}, log x={logx}. SVG/PDF vector exports; PNG 600 dpi. Heatmap uses {bins} bins per axis. No smoothing or spectral density. CSV contains the plotted rows.\n')
    return out.getvalue()


def profile_archive(profiles,labels,centers):
    """Export pointwise median representative profiles without percentile envelopes."""
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
            members=profiles[labels==j];color=PALETTE[j%len(PALETTE)]
            ax.plot(phase,center,color=color,lw=1.8,label=f'Cluster {j} (n={len(members)})')
            for pos,t in enumerate(phase):
                rows.append({'cluster':j,'phase':t,'median_representative_blockade_nA':center[pos],'events':len(members)})
        ax.set_xlabel('Fraction of event duration');ax.set_ylabel('Current blockade (nA)');ax.legend(frameon=False,fontsize=8)
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for ext in ['pdf','svg','png']:
                b=io.BytesIO();fig.savefig(b,format=ext,dpi=600);z.writestr('cluster_profiles.'+ext,b.getvalue())
            z.writestr('profile_summary.csv',pd.DataFrame(rows).to_csv(index=False))
            z.writestr('README.txt','Representative curves are pointwise medians of all duration-normalised member profiles. No percentile envelope is used. Absolute duration is removed from these profiles; blockade amplitude remains in nA.\n')
        plt.close(fig)
    return out.getvalue()


def hull_vertices(points):
    """Convex outline in the displayed 2D projection; skip degenerate groups."""
    from scipy.spatial import ConvexHull,QhullError
    points=np.unique(np.asarray(points,float),axis=0)
    if len(points)<3 or np.linalg.matrix_rank(points-points.mean(axis=0),tol=1e-10)<2:return None
    try:return points[ConvexHull(points).vertices]
    except QhullError:return None


def cluster_pca_figure(projection,axis_labels,outlines=False):
    f=go.Figure()
    for j,(label,g) in enumerate(projection.groupby('Cluster',sort=True)):
        color=PALETTE[j%len(PALETTE)];points=g[['PC1','PC2']].to_numpy();poly=hull_vertices(points)
        if outlines and poly is not None:
            closed=np.vstack([poly,poly[0]])
            rgb=tuple(int(color[h:h+2],16) for h in (1,3,5))
            f.add_trace(go.Scatter(x=closed[:,0],y=closed[:,1],mode='lines',line=dict(color=color,width=1),fill='toself',fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)',showlegend=False,hoverinfo='skip'))
        f.add_trace(go.Scatter(x=points[:,0],y=points[:,1],mode='markers',name=f'Cluster {label} (n={len(g)})',marker=dict(color=color,size=5,opacity=.7),
            customdata=g[['Event ID','Duration (ms)','Measured mean blockade (nA)']].to_numpy(),hovertemplate='Event %{customdata[0]:.0f}<br>Duration %{customdata[1]:.4g} ms<br>Measured mean blockade %{customdata[2]:.4g} nA<br>PC1 %{x:.4g}<br>PC2 %{y:.4g}<extra>%{fullData.name}</extra>'))
        center=np.median(points,axis=0)
        f.add_trace(go.Scatter(x=[center[0]],y=[center[1]],mode='markers',marker=dict(symbol='x',size=11,color='black'),name=f'Cluster {label} median position',showlegend=False,hovertemplate='Median position in displayed PCA coordinates<extra></extra>'))
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
            center=np.median(points,axis=0);ax.scatter(*center,marker='x',color='black',s=40,lw=1.4)
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

# -----------------------------------------------------------------------------
# Hart-style DNA clustering diagnostics (v0.6)
# -----------------------------------------------------------------------------
def pca_scree_figure(scree):
    variance=100*np.asarray(scree,float);pcs=np.arange(1,len(variance)+1);cum=np.cumsum(variance)
    f=go.Figure()
    f.add_trace(go.Bar(x=pcs,y=variance,name='Explained variance'))
    f.add_trace(go.Scatter(x=pcs,y=cum,name='Cumulative variance',mode='lines+markers',yaxis='y2'))
    f.update_layout(xaxis_title='Principal component',yaxis_title='Explained variance (%)',
                    yaxis2=dict(title='Cumulative variance (%)',overlaying='y',side='right',range=[0,105]),height=420,
                    legend=dict(orientation='h'))
    return f


def k_diagnostics_figure(table,elbow_k=None,silhouette_k=None):
    import pandas as pd
    d=pd.DataFrame(table)
    f=go.Figure()
    f.add_trace(go.Scatter(x=d.k,y=d.dispersion,mode='lines+markers',name='Within-cluster dispersion'))
    f.add_trace(go.Scatter(x=d.k,y=d.silhouette,mode='lines+markers',name='Silhouette',yaxis='y2'))
    if elbow_k is not None:f.add_vline(x=elbow_k,line_dash='dot',annotation_text=f'Elbow k={elbow_k}',annotation_position='top left')
    if silhouette_k is not None:f.add_vline(x=silhouette_k,line_dash='dash',annotation_text=f'Silhouette k={silhouette_k}',annotation_position='top right')
    f.update_layout(xaxis_title='Number of clusters, k',yaxis_title='Within-cluster dispersion',
                    yaxis2=dict(title='Silhouette score',overlaying='y',side='right'),height=430,legend=dict(orientation='h'))
    return f


def feature_correlation_figure(corr,names):
    c=np.asarray(corr,float)
    f=go.Figure(go.Heatmap(z=c,x=names,y=names,zmin=-1,zmax=1,colorscale='RdBu',reversescale=True,
                           colorbar=dict(title='Pearson r'),hovertemplate='%{x}<br>%{y}<br>r=%{z:.3f}<extra></extra>'))
    f.update_layout(height=max(430,28*len(names)+160),xaxis_title='Feature',yaxis_title='Feature')
    return f


def cluster_pca_3d_figure(embedding3,labels,variance,event_ids=None):
    import pandas as pd
    e=np.asarray(embedding3,float);lab=np.asarray(labels,int)
    data=pd.DataFrame({'PC1':e[:,0],'PC2':e[:,1],'PC3':e[:,2],'Cluster':lab.astype(str)})
    if event_ids is not None:data['Event ID']=np.asarray(event_ids)
    hover=['Event ID'] if 'Event ID' in data else None
    f=px.scatter_3d(data,x='PC1',y='PC2',z='PC3',color='Cluster',hover_data=hover,opacity=.65,color_discrete_sequence=PALETTE)
    v=list(variance)+[np.nan]*3
    f.update_layout(scene=dict(xaxis_title=f'PC1 ({100*v[0]:.1f}%)',yaxis_title=f'PC2 ({100*v[1]:.1f}%)',zaxis_title=f'PC3 ({100*v[2]:.1f}%)'),height=620)
    return f


def dendrogram_figure(linkage_matrix,k=None,p=50):
    """Truncated Ward dendrogram; useful for visualising the hierarchy without thousands of leaves."""
    from scipy.cluster.hierarchy import dendrogram
    z=np.asarray(linkage_matrix,float)
    threshold=None
    if k is not None and len(z)>=k:
        # k clusters exist immediately before the k-1 largest inter-group merges.
        lo=z[-int(k),2] if int(k)<=len(z) else z[0,2]
        hi=z[-int(k)+1,2] if int(k)>1 else z[-1,2]
        threshold=float((lo+hi)/2)
    d=dendrogram(z,truncate_mode='lastp',p=min(int(p),len(z)+1),show_leaf_counts=True,no_plot=True,
                 color_threshold=threshold,above_threshold_color='#666666')
    f=go.Figure()
    # scipy coordinates use leaf locations 5, 15, 25, ...
    for xs,ys,color in zip(d['icoord'],d['dcoord'],d['color_list']):
        f.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(width=1.2),showlegend=False,hoverinfo='skip'))
    f.update_layout(xaxis_title='Truncated leaves / merged groups',yaxis_title='Ward linkage distance',height=440)
    f.update_xaxes(showticklabels=False)
    return f


def member_profile_figure_hart(profiles,labels,centers,group,limit=60):
    """DNA event-family panel: reproducible member traces + bold median representative."""
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1]
    members=np.flatnonzero(np.asarray(labels)==group);rng=np.random.default_rng(42)
    chosen=np.sort(rng.choice(members,min(limit,len(members)),replace=False))
    f=go.Figure()
    for pos in chosen:
        f.add_trace(go.Scatter(x=phase,y=profiles[pos],mode='lines',
                               line=dict(color='rgba(90,90,90,0.12)',width=.7),
                               showlegend=False,hoverinfo='skip'))
    f.add_trace(go.Scatter(x=phase,y=centers[group],mode='lines',
                           line=dict(color='#D62728',width=2.6),name='Median representative'))
    f.update_layout(title=f'Cluster {group} · n={len(members)}',
                    xaxis_title='Fraction of event duration',yaxis_title='Blockade (nA)',height=310)
    return f



def blockade_dwell_figure(table,blockade_col='clustering_measured_mean_blockade_nA'):
    """Direct physical view: detected dwell time versus measured mean blockade."""
    d=table[np.isfinite(table['duration_ms'])&np.isfinite(table[blockade_col])&(table['duration_ms']>0)].copy()
    d['Cluster']=d['cluster'].astype(str)
    f=px.scatter(d,x='duration_ms',y=blockade_col,color='Cluster',log_x=True,opacity=.55,
                 hover_data=[c for c in ['event_index','clustering_deepest_plateau_nA','clustering_resolved_levels'] if c in d],
                 labels={'duration_ms':'Dwell time (ms)',blockade_col:'Measured mean blockade (nA)'},
                 color_discrete_sequence=PALETTE)
    f.update_layout(height=450)
    return f


def fold_state_figure(table,ratio_col='clustering_deep_to_shallow_ratio',occupancy_col='clustering_deepest_plateau_fraction'):
    """Physical fold-state map: relative deep-state amplitude versus occupancy."""
    need=['cluster',ratio_col,occupancy_col]
    if any(c not in table for c in need):return go.Figure()
    d=table[need+([c for c in ['event_index','duration_ms','clustering_resolved_levels'] if c in table])].dropna().copy()
    d=d[np.isfinite(d[ratio_col])&np.isfinite(d[occupancy_col])&(d[ratio_col]>=1)]
    d['Cluster']=d['cluster'].astype(str)
    f=px.scatter(d,x=occupancy_col,y=ratio_col,color='Cluster',opacity=.55,
                 hover_data=[c for c in ['event_index','duration_ms','clustering_resolved_levels'] if c in d],
                 labels={occupancy_col:'Fraction of analysed duration in deepest state',ratio_col:'Deep / shallow resolved blockade ratio'},
                 color_discrete_sequence=PALETTE)
    f.add_hline(y=1,line_dash='dot',line_width=1)
    f.update_layout(height=450)
    return f


def population_fraction_figure(table):
    counts=table.groupby('cluster').size().sort_index()
    pct=100*counts/counts.sum()
    f=go.Figure(go.Bar(x=[f'Cluster {int(i)}' for i in counts.index],y=pct.to_numpy(),
                       text=[f'{v:.1f}%' for v in pct],textposition='auto'))
    f.update_layout(xaxis_title='Signal family',yaxis_title='Population (%)',height=380)
    return f


def level_composition_figure(table,level_col='clustering_resolved_levels'):
    """Within-cluster percentages of one-, two-, and >=3-level resolved events."""
    d=table[['cluster',level_col]].dropna().copy()
    d['level_class']=np.where(d[level_col]<=1,'1 level',np.where(d[level_col]==2,'2 levels','≥3 levels'))
    order=['1 level','2 levels','≥3 levels'];clusters=sorted(d.cluster.unique())
    f=go.Figure()
    for cls in order:
        vals=[]
        for c in clusters:
            sub=d[d.cluster==c]
            vals.append(100*float((sub.level_class==cls).mean()) if len(sub) else 0.)
        f.add_trace(go.Bar(x=[f'Cluster {int(c)}' for c in clusters],y=vals,name=cls))
    f.update_layout(barmode='stack',xaxis_title='Signal family',yaxis_title='Within-cluster events (%)',
                    yaxis_range=[0,100],height=400)
    return f


def occupancy_figure(table,occupancy_col='clustering_deepest_plateau_fraction'):
    d=table[['cluster',occupancy_col]].dropna().copy()
    f=go.Figure()
    for j,c in enumerate(sorted(d.cluster.unique())):
        vals=d.loc[d.cluster==c,occupancy_col]
        f.add_trace(go.Box(y=vals,name=f'Cluster {int(c)}',boxpoints='outliers',
                           marker=dict(size=3),line=dict(width=1.2)))
    f.update_layout(yaxis_title='Fraction of analysed duration in deepest state',
                    xaxis_title='Signal family',height=400)
    return f


def physical_feature_distributions_figure(table):
    """Four direct physical distributions used for interpretation, not all for clustering."""
    from plotly.subplots import make_subplots
    specs=[
        ('duration_ms','Dwell time (ms)'),
        ('clustering_measured_mean_blockade_nA','Measured mean blockade (nA)'),
        ('clustering_deep_to_shallow_ratio','Deep / shallow blockade ratio'),
        ('clustering_ecd_nA_ms','ECD (nA·ms)'),
    ]
    f=make_subplots(rows=2,cols=2,subplot_titles=[b for _,b in specs])
    clusters=sorted(table.cluster.unique())
    for idx,(field,label) in enumerate(specs):
        r=idx//2+1;c=idx%2+1
        if field not in table:continue
        for j,cl in enumerate(clusters):
            vals=table.loc[table.cluster==cl,field].dropna()
            f.add_trace(go.Box(y=vals,name=f'Cluster {int(cl)}',legendgroup=str(cl),
                               showlegend=(idx==0),boxpoints=False,line=dict(width=1.1)),
                        row=r,col=c)
        f.update_yaxes(title_text=label,row=r,col=c)
    f.update_layout(height=650,boxmode='group')
    return f

def hart_style_archive(info,event_ids,feature_table,cluster_summary,k_scan=None,source='Selected fits',method='PCA + agglomerative',dendrogram_p=50):
    """Export the main and supplementary-style DNA clustering figures as separate files.

    The layouts are inspired by the analysis sequence in Hart et al. (PCA clustering,
    event-family profiles, centroid overlay, scree, elbow/silhouette and dendrogram),
    but are generated directly from the user's data and this app's feature model.
    """
    import io,json,zipfile,math
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    emb=np.asarray(info['embedding']);emb3=info.get('embedding3');profiles=np.asarray(info['profiles']);labels=np.asarray(info['labels']);centers=np.asarray(info['centers'])
    phase=(np.arange(profiles.shape[1])+.5)/profiles.shape[1];k=len(centers);variance=np.asarray(info.get('pca_variance',[]),float)
    out=io.BytesIO()
    rc={'font.family':'sans-serif','font.size':9,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42,'xtick.direction':'out','ytick.direction':'out'}
    with plt.rc_context(rc),zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        def save(fig,stem):
            for ext in ['pdf','svg','png']:
                b=io.BytesIO();fig.savefig(b,format=ext,dpi=600,facecolor='white',bbox_inches='tight');z.writestr(f'{stem}.{ext}',b.getvalue())
            plt.close(fig)

        # Main-style PCA cluster map.
        fig,ax=plt.subplots(figsize=(6.2,4.6),layout='constrained')
        for j in range(k):
            pts=emb[labels==j];ax.scatter(pts[:,0],pts[:,1],s=9,alpha=.55,edgecolors='none',color=PALETTE[j%len(PALETTE)],label=f'Cluster {j} (n={len(pts)})')
            ax.scatter(*np.median(pts,axis=0),marker='x',color='black',s=35,lw=1.2)
        ax.set_xlabel(f'PC1 ({100*variance[0]:.1f}% variance)' if len(variance)>0 else 'PC1');ax.set_ylabel(f'PC2 ({100*variance[1]:.1f}% variance)' if len(variance)>1 else 'PC2')
        ax.legend(frameon=False,fontsize=8);save(fig,'main_pca_clusters')

        # Main-style cluster waveform panels.
        cols=3;rows=math.ceil(k/cols);fig,axes=plt.subplots(rows,cols,figsize=(10,3.1*rows),squeeze=False,layout='constrained')
        low,high=float(profiles.min()),float(profiles.max());pad=max(.05,.06*(high-low));rng=np.random.default_rng(42)
        for j in range(rows*cols):
            ax=axes.flat[j]
            if j>=k:ax.axis('off');continue
            ids=np.flatnonzero(labels==j);sample=np.sort(rng.choice(ids,min(60,len(ids)),replace=False))
            for pos in sample:ax.plot(phase,profiles[pos],color='0.45',alpha=.10,lw=.5)
            ax.plot(phase,centers[j],color='#D62728',lw=1.8);ax.set_title(f'Cluster {j} (n={len(ids)})');ax.set_xlabel('Fraction of event duration');ax.set_ylabel('Blockade (nA)');ax.set_ylim(low-pad,high+pad)
        save(fig,'main_cluster_profiles')

        fig,ax=plt.subplots(figsize=(6.2,4.3),layout='constrained')
        for j,c in enumerate(centers):ax.plot(phase,c,lw=1.7,color=PALETTE[j%len(PALETTE)],label=f'Cluster {j}')
        ax.set_xlabel('Fraction of event duration');ax.set_ylabel('Blockade (nA)');ax.legend(frameon=False);save(fig,'main_representative_overlay')


        # Main physical interpretation: blockade versus dwell.
        if 'clustering_measured_mean_blockade_nA' in feature_table:
            fig,ax=plt.subplots(figsize=(6.2,4.5),layout='constrained')
            for j in range(k):
                sub=feature_table[feature_table.cluster==j]
                ax.scatter(sub.duration_ms,sub.clustering_measured_mean_blockade_nA,s=8,alpha=.45,
                           edgecolors='none',color=PALETTE[j%len(PALETTE)],label=f'Cluster {j}')
            ax.set_xscale('log');ax.set_xlabel('Dwell time (ms)');ax.set_ylabel('Measured mean blockade (nA)')
            ax.legend(frameon=False,fontsize=8);save(fig,'main_blockade_vs_dwell')

        # Main physical feature distributions.
        dist_fields=[('duration_ms','Dwell time (ms)'),
                     ('clustering_measured_mean_blockade_nA','Measured mean blockade (nA)'),
                     ('clustering_deep_to_shallow_ratio','Deep / shallow blockade ratio'),
                     ('clustering_ecd_nA_ms','ECD (nA·ms)')]
        fig,axes=plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
        for ax,(field,label) in zip(axes.flat,dist_fields):
            if field not in feature_table:
                ax.axis('off');continue
            data=[feature_table.loc[feature_table.cluster==j,field].dropna().to_numpy() for j in range(k)]
            ax.boxplot(data,labels=[str(j) for j in range(k)],showfliers=False)
            ax.set_xlabel('Cluster');ax.set_ylabel(label)
        save(fig,'main_physical_distributions')

        # Population fractions.
        counts=feature_table.groupby('cluster').size().reindex(range(k),fill_value=0)
        fig,ax=plt.subplots(figsize=(5.5,4.0),layout='constrained')
        ax.bar(np.arange(k),100*counts.to_numpy()/counts.sum());ax.set_xticks(np.arange(k));ax.set_xlabel('Cluster');ax.set_ylabel('Population (%)')
        save(fig,'main_population_fraction')

        # Resolved-level composition.
        if 'clustering_resolved_levels' in feature_table:
            fig,ax=plt.subplots(figsize=(5.8,4.1),layout='constrained');bottom=np.zeros(k)
            vals=feature_table.clustering_resolved_levels
            classes=[('1 level',vals<=1),('2 levels',vals==2),('≥3 levels',vals>=3)]
            for label,mask in classes:
                pct=np.array([100*np.mean(mask[feature_table.cluster==j]) if np.any(feature_table.cluster==j) else 0. for j in range(k)])
                ax.bar(np.arange(k),pct,bottom=bottom,label=label);bottom+=pct
            ax.set_xticks(np.arange(k));ax.set_xlabel('Cluster');ax.set_ylabel('Within-cluster events (%)');ax.set_ylim(0,100);ax.legend(frameon=False,fontsize=8)
            save(fig,'main_level_composition')

        # Fraction of event spent in its deepest resolved state.
        if 'clustering_deepest_plateau_fraction' in feature_table:
            fig,ax=plt.subplots(figsize=(5.8,4.1),layout='constrained')
            data=[feature_table.loc[feature_table.cluster==j,'clustering_deepest_plateau_fraction'].dropna().to_numpy() for j in range(k)]
            ax.boxplot(data,labels=[str(j) for j in range(k)],showfliers=False);ax.set_xlabel('Cluster');ax.set_ylabel('Deepest-state occupancy fraction')
            save(fig,'main_deepest_state_occupancy')

        # Supplementary scree plot.
        scree=100*np.asarray(info.get('scree',[]));cum=np.cumsum(scree);pcs=np.arange(1,len(scree)+1)
        fig,ax=plt.subplots(figsize=(5.5,4.1),layout='constrained');ax.plot(pcs,scree,'o-',lw=1.2,label='Explained variance');ax.set_xlabel('Principal component');ax.set_ylabel('Explained variance (%)')
        ax2=ax.twinx();ax2.plot(pcs,cum,'s--',lw=1.0,label='Cumulative');ax2.set_ylabel('Cumulative variance (%)');ax2.set_ylim(0,105);save(fig,'supp_scree')

        # Supplementary elbow + silhouette.
        if k_scan:
            d=pd.DataFrame(k_scan['table']);fig,ax=plt.subplots(figsize=(5.7,4.1),layout='constrained');ax.plot(d.k,d.dispersion,'o-',lw=1.2);ax.set_xlabel('Number of clusters, k');ax.set_ylabel('Within-cluster dispersion')
            ax2=ax.twinx();ax2.plot(d.k,d.silhouette,'s--',lw=1.1);ax2.set_ylabel('Silhouette score');ax.axvline(k_scan['elbow_k'],ls=':',lw=1);ax2.axvline(k_scan['silhouette_k'],ls='--',lw=1);save(fig,'supp_elbow_silhouette')
            z.writestr('k_diagnostics.csv',d.to_csv(index=False))

        # Supplementary 3-PC view if possible.
        if emb3 is not None:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(111,projection='3d')
            for j in range(k):
                pts=np.asarray(emb3)[labels==j];ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=6,alpha=.45,color=PALETTE[j%len(PALETTE)],label=f'Cluster {j}')
            ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_zlabel('PC3');ax.legend(frameon=False,fontsize=7);save(fig,'supp_pca_3d')

        # Supplementary agglomerative dendrogram.
        if info.get('linkage_matrix') is not None:
            fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained');dendrogram(np.asarray(info['linkage_matrix']),truncate_mode='lastp',p=min(int(dendrogram_p),len(labels)),show_leaf_counts=True,ax=ax)
            ax.set_xlabel('Truncated leaves / merged groups');ax.set_ylabel('Ward linkage distance');save(fig,'supp_dendrogram')

        z.writestr('cluster_features.csv',feature_table.to_csv(index=False));z.writestr('cluster_summary.csv',cluster_summary.to_csv(index=False))
        z.writestr('pca_coordinates.csv',pd.DataFrame({'event_id':event_ids,'cluster':labels,'PC1':emb[:,0],'PC2':emb[:,1]}).to_csv(index=False))
        z.writestr('representative_profiles.csv',pd.DataFrame([{'cluster':j,'phase':float(t),'blockade_nA':float(v)} for j,c in enumerate(centers) for t,v in zip(phase,c)]).to_csv(index=False))
        z.writestr('README.txt',('Hart-style DNA clustering export generated from the current recording. Main-style outputs: PCA cluster map without convex-hull fills, per-cluster member traces with median representatives, and representative overlay. Supplementary-style outputs: scree plot, elbow-silhouette scan, optional 3-PC view, and Ward dendrogram for agglomerative clustering. These are analysis analogues, not reproductions of published artwork. Cluster labels denote signal families only; physical/topological assignments require independent interpretation.'))
        z.writestr('settings.json',json.dumps({'source':source,'method':method,'n_components':info.get('n_components'),'silhouette':info.get('silhouette'),'calinski_harabasz':info.get('calinski_harabasz'),'davies_bouldin':info.get('davies_bouldin')},indent=2))
    return out.getvalue()

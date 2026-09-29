import io,hashlib,json,zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analysis import (VERSION,load_events,load_dataset,inventory,link_dataset,describe,profile,refine,
    fit_metrics,cluster_profiles,cluster_features,cluster_dtw,feature_table,FEATURE_DESCRIPTIONS,cluster_bundle,npz_bytes,synthetic_demo,safe_settings)
from plots import trace_figure,distribution_figures,profile_figure,LABELS

st.set_page_config(page_title='Nanopore Shape Lab',page_icon='🧬',layout='wide')
st.title('Nanopore Shape Lab')
st.caption(f'v{VERSION} · Inspect signals → compare fits → explore groups → export events')
st.info('Shape groups are exploratory. They do not automatically identify unfolded, folded or knotted DNA. Use one recording condition at a time.')
source=st.sidebar.radio('Data source',['Upload files','Synthetic demonstration'])
wave_upload=data_upload=None;events=[];settings={};rejected=[];dataset=None;dsettings={};mapping={};join_table=None
if source=='Upload files':
    wave_upload=st.sidebar.file_uploader('Waveforms: event_fitting.npz or eventdata.npz',type='npz',key='wave')
    data_upload=st.sidebar.file_uploader('Optional summary: dataset.npz',type='npz',key='data')
    if wave_upload is None and data_upload is None:
        st.markdown('Upload your full waveform export to inspect, refine and cluster events. Add its dataset file to preserve source summary rows in cluster downloads. A dataset alone supports distribution plots. The synthetic demonstration is available in the sidebar.')
        st.stop()
    blobs={name:u.getvalue() for name,u in [('wave',wave_upload),('data',data_upload)] if u is not None}
    fingerprint=hashlib.sha256(b''.join(name.encode()+hashlib.sha256(blob).digest() for name,blob in sorted(blobs.items()))).hexdigest()
    if st.session_state.get('loaded_key')!=fingerprint:
        payload={'errors':[]}
        for name,fn in [('wave',load_events),('data',load_dataset)]:
            if name in blobs:
                try:payload[name]=fn(blobs[name])
                except Exception as ex:payload['errors'].append(f'{name}: {ex}')
        st.session_state.loaded=payload;st.session_state.loaded_key=fingerprint
    loaded=st.session_state.loaded
    if 'wave' in loaded:events,settings,rejected=loaded['wave']
    if 'data' in loaded:dataset,dsettings=loaded['data']
    for msg in loaded['errors']:st.error(msg)
    with st.expander('File structure inspector'):
        for name,blob in blobs.items():
            try:rows,n=inventory(blob);st.write(f'{name}: {n} arrays (first 60 below)');st.dataframe(pd.DataFrame(rows),hide_index=True)
            except Exception as ex:st.warning(str(ex))
    if not events and dataset is None:st.stop()
else:
    fingerprint='synthetic-v2';events,settings,rejected=synthetic_demo();blobs={}
if st.session_state.get('active_source')!=fingerprint:
    for key in ['refinements','group_result','prepared_download','previous_groups','fit_errors']:st.session_state.pop(key,None)
    st.session_state.active_source=fingerprint
st.session_state.setdefault('refinements',{})
refs=st.session_state.refinements
bins=st.sidebar.select_slider('Profile positions',[16,32,64,128],value=32)
if wave_upload and 'SHORT' in wave_upload.name.upper():st.warning('Selected/subset export: population fractions describe only uploaded events.')
if rejected:st.warning(f'{len(rejected)} malformed events omitted; details in the audit tab.')

start_col=8;duration_col=4;height_col=0;tolerance=1e-7
if dataset is not None:
    with st.expander('Dataset column mapping and timestamp matching',expanded=not bool(events)):
        st.caption('Defaults match the supplied NanoSense X table. Column numbers are zero-based. Verify units and mappings before interpreting summary-only plots. No matching by row order.')
        c1,c2,c3,c4=st.columns(4)
        start_col=c1.number_input('Start time column (seconds)',0,dataset.shape[1]-1,min(8,dataset.shape[1]-1))
        duration_col=c2.number_input('Duration column (seconds)',0,dataset.shape[1]-1,min(4,dataset.shape[1]-1))
        height_col=c3.number_input('Height column (nA)',0,dataset.shape[1]-1,0)
        tolerance=c4.number_input('Timestamp tolerance (µs)',min_value=.001,max_value=1000.,value=.1,format='%.3f')*1e-6
        if events:
            mapping,join_table=link_dataset(events,dataset,int(start_col),tolerance)
            st.write(f'{len(mapping)}/{len(events)} waveform events uniquely matched; {len(dataset)-len(mapping)} summary rows not linked.')
            st.dataframe(join_table,hide_index=True)
            if len(mapping)!=len(events):st.warning('Unmatched waveform events remain in the analysis. Their downloads will not contain an invented summary row.')
            if settings.get('file_name') and dsettings.get('file_name') and settings['file_name']!=dsettings['file_name']:
                st.warning('The two files name different acquisition sources. Confirm they belong together; timestamp matches alone are insufficient.')
        else:st.warning('Summary-only mode: no waveforms, fit refinement, shape clustering  are available.')
if events:df,raw_profiles=describe(events,bins,refs)
else:
    df=pd.DataFrame({'dataset_row':np.arange(len(dataset)),'start_s':dataset[:,int(start_col)],'duration_ms':dataset[:,int(duration_col)]*1000,'export_height_nA':dataset[:,int(height_col)]})
if dataset is not None and events:
    df['dataset_row']=[mapping.get(int(i),np.nan) for i in df.event_index]
    for col in range(dataset.shape[1]):df[f'export_col_{col}']=[dataset[mapping[int(i)],col] if int(i) in mapping else np.nan for i in df.event_index]
c1,c2,c3=st.columns(3);c1.metric('Waveform events',len(events));c2.metric('Summary rows',len(dataset) if dataset is not None else '—');c3.metric('Candidate fits',len(refs))
with st.expander('Recording metadata and assumptions'):
    st.json(safe_settings(settings or dsettings));st.caption('Time is assumed seconds; current is assumed nA. Original arrays are retained. A missing baseline is estimated from padding and marked in the results. ')
    if events:st.caption(f'Median saved sample spacing: {df.time_step_us.median():.6g} µs. Saved metadata sampling rate: {settings.get("sampling_rate","not provided")} Hz.')

inspect,audit,fit_tab,group_tab,dist_tab,export_tab=st.tabs(['1 · Inspect','2 · Waveform disagreement','3 · Refine fits','4 · Cluster','5 · Distributions','6 · Download'])
with inspect:
    if events:
        selected=st.selectbox('Event index',[e.index for e in events]);event=next(e for e in events if e.index==selected)
        zoom=st.checkbox('Zoom to event',True)
        st.caption(f'Start {event.bounds[0]:.9f} s · baseline: {event.baseline_source}. IDs are local to the source export; subset exports may renumber them.')
        st.plotly_chart(trace_figure(event,refs.get(event.index),zoom),width='stretch',key='inspect_trace')
        if event.levels is not None and event.widths is not None and len(event.levels)==len(event.widths):
            st.dataframe(pd.DataFrame({'saved_level_nA':event.levels,'saved_duration_ms':event.widths*1000}),hide_index=True)
        st.caption('Saved segment count is not the number of DNA sections in the pore.')
    else:st.info('Upload a waveform file to inspect individual events.')
with audit:
    st.markdown('A rectangular fit can capture peak height and duration while disagreeing with a rounded waveform. **RMSE measures pointwise disagreement inside the event—not event validity or physical fit quality.** No RMSE exclusion is enabled by default.')
    if events:
        y=st.selectbox('Diagnostic metric',['waveform_rmse_nA','relative_rmse','peak_error_nA','area_error_pct'],format_func=lambda v:LABELS[v])
        st.plotly_chart(px.scatter(df,x='duration_ms',y=y,color='segments',log_x=True,hover_data=['event_index','start_s'],labels=LABELS),width='stretch')
        st.caption('Relative RMSE divides by the measured signal RMS. Peak error and area error are signed. More step parameters can reduce RMSE without improving the physical model.')
        st.dataframe(df,hide_index=True)
        if rejected:st.dataframe(pd.DataFrame(rejected),hide_index=True)
    else:st.info('Waveforms and saved fits are needed for this diagnostic.')
with fit_tab:
    st.markdown('Compare candidate representations with the original. Refinement does not recover details lost to bandwidth, and more fitted levels do not establish more DNA folds.')
    if events:
        method=st.selectbox('Candidate method',['Segment means','New levels (PELT)','Rounded pulse (Gaussian)'])
        explanations={'Segment means':'Keep the existing step boundaries and recalculate each height as the mean observed blockade. This optimises squared error for those boundaries; it may lower peak estimates.',
            'New levels (PELT)':'Find new step boundaries with a penalty for extra segments and a minimum segment duration. Correlated noise and rounded edges can still create spurious steps.',
            'Rounded pulse (Gaussian)':'Fit one smooth bell-shaped pulse. Useful as a descriptive comparison for single rounded events; it is not an instrument-response correction or a model of multiple occupancy.'}
        st.write(explanations[method]);c1,c2=st.columns(2)
        minimum=c1.number_input('Minimum step duration (µs; PELT)',5.,10000.,25.,step=5.)
        penalty=c2.number_input('Extra-segment penalty multiplier (PELT)',.1,100.,8.,step=.5)
        scope=st.radio('Apply candidate to',['Selected event','All loaded events'],horizontal=True)
        st.caption(f'Selected event: {selected}. Original baselines and event boundaries are held fixed. Up to 6,000 event samples per event are supported for refinement.')
        if st.button('Calculate candidate fit',type='primary'):
            targets=events if scope=='All loaded events' else [event];errors=[];bar=st.progress(0.)
            for j,e in enumerate(targets):
                try:refs[e.index]=refine(e,method,minimum,penalty)
                except Exception as ex:errors.append({'event_index':e.index,'reason':str(ex)})
                bar.progress((j+1)/len(targets))
            st.session_state.refinements=refs;st.session_state.pop('group_result',None);st.session_state.pop('prepared_download',None)
            st.session_state.fit_errors=errors;st.rerun()
        if st.button('Clear all candidate fits'):
            st.session_state.refinements={};st.session_state.pop('group_result',None);st.session_state.pop('prepared_download',None);st.rerun()
        if st.session_state.get('fit_errors'):st.dataframe(pd.DataFrame(st.session_state.fit_errors),hide_index=True)
        if event.index in refs:
            ref=refs[event.index];st.plotly_chart(trace_figure(event,ref),width='stretch',key='candidate_trace')
            compare=pd.DataFrame([fit_metrics(event,event.fit),fit_metrics(event,ref['fit'])],index=['Original','Candidate']);st.dataframe(compare)
            st.write('Candidate parameters:',ref['parameters'])
            if len(ref['levels']):st.dataframe(pd.DataFrame({'candidate_level_nA':ref['levels'],'candidate_duration_ms':ref['widths']*1000}),hide_index=True)
    else:st.info('Upload waveforms to calculate candidate fits.')

result=df.copy();valid_groups=False;group_info=None;signature=None
with group_tab:
    if events:
        st.markdown('Compare unsupervised methods inspired by NanoBoost (2026). These are adaptations for your DNA exports, not an exact reproduction of its preprocessing or feature set. No method is assumed superior.')
        algorithm=st.selectbox('Clustering method',['PCA + agglomerative (DNA concept)','PCA + k-means (nanorod concept)','Waveform k-means (original baseline)','Time-series k-means (DTW)'])
        feature_mode=algorithm.startswith('PCA')
        features=feature_table(events)
        default_features=['duration_ms','mean_blockade_nA','peak_blockade_nA','blockade_std_nA','peak_position','early_late_difference_nA']
        selected_features=st.multiselect('Features for PCA methods',list(FEATURE_DESCRIPTIONS),default=default_features,disabled=not feature_mode)
        npc=st.slider('Retained PCA components',1,6,2,disabled=not feature_mode)
        radius=st.slider('DTW alignment radius (profile positions)',1,16,4,disabled=algorithm!='Time-series k-means (DTW)')
        with st.expander('What each method compares'):
            st.write('PCA methods: min–max scaled event features → PCA → clustering. Ward linkage is used for the agglomerative option; this choice still has geometric assumptions. Optional Haar coefficient features are available, with no denoising. These are not the paper’s complete 25-feature pipeline.')
            st.write('Waveform baseline: point-by-point Euclidean distance after duration normalisation. DTW: locally align waveform features, with a limited warping window. This may reduce timing sensitivity but can hide physically meaningful differences in fold duration. DTW is limited to 2,000 events and 64 positions.')
            st.dataframe(pd.DataFrame({'feature':list(FEATURE_DESCRIPTIONS),'definition':list(FEATURE_DESCRIPTIONS.values())}),hide_index=True)
        c1,c2=st.columns(2);representation=c1.selectbox('Signal for grouping',['Saved trace','Original fit','Candidate fit'],disabled=feature_mode)
        shape_only=c2.checkbox('Shape only: normalise each event amplitude',False,disabled=feature_mode)
        if shape_only and not feature_mode:st.warning('Amplitude normalisation removes absolute blockade differences. Constant single- and double-occupancy signals may then look identical. Compare with depth-preserving mode.')
        c1,c2,c3=st.columns(3);k=c1.slider('Number of groups',2,10,4);min_samples=c2.number_input('Minimum event samples',3,1000,3)
        gate=c3.checkbox('Exclude by original waveform RMSE',False)
        cutoff=st.number_input('Original waveform RMSE limit (nA)',.01,100.,10.,step=.1,disabled=not gate)
        effective_representation='Saved trace' if feature_mode else representation
        all_profiles=raw_profiles.copy();available=np.ones(len(events),bool)
        for pos,e in enumerate(events):
            if effective_representation=='Original fit':
                available[pos]=e.fit is not None
                if available[pos]:all_profiles[pos]=profile(e,bins,e.fit)
            elif effective_representation=='Candidate fit':
                available[pos]=e.index in refs
                if available[pos]:all_profiles[pos]=profile(e,bins,refs[e.index]['fit'])
        eligible=(df.samples.to_numpy()>=min_samples)&available
        if gate:eligible&=(df.waveform_rmse_nA.to_numpy()<=cutoff)
        ref_signature=hashlib.sha256(b''.join(str(i).encode()+r['fit'].tobytes() for i,r in sorted(refs.items()))).hexdigest()
        signature=(fingerprint,bins,k,min_samples,gate,cutoff,representation,shape_only,ref_signature,start_col,tolerance,algorithm,tuple(selected_features),npc,radius)
        st.write(f'{eligible.sum()} eligible / {len(events)} loaded; {(~eligible).sum()} excluded by availability or selected settings.')
        st.caption('PCA uses the selected features, including duration if selected. Waveform methods remove absolute duration. Choosing k groups does not establish k physical configurations.')
        if st.button('Compare group counts 2–8',disabled=algorithm=='Time-series k-means (DTW)'):
            comparison=[]
            with st.spinner('Exploring k values in the selected space…'):
                for trial_k in range(2,min(8,int(eligible.sum())-1)+1):
                    try:
                        if feature_mode:
                            if len(selected_features)<2:raise ValueError('Choose at least two features.')
                            trial=cluster_features(features.loc[eligible,selected_features].to_numpy(),all_profiles[eligible],trial_k,npc,algorithm)
                        else:trial=cluster_profiles(all_profiles[eligible],trial_k,shape_only)
                        comparison.append({'k':trial_k,'silhouette':trial['silhouette'],'within_group_sum_squares':trial.get('dispersion',np.nan)})
                    except ValueError:continue
            if comparison:
                comp=pd.DataFrame(comparison);st.dataframe(comp);st.plotly_chart(px.line(comp,x='k',y='silhouette',markers=True),width='stretch')
                if feature_mode:st.plotly_chart(px.line(comp,x='k',y='within_group_sum_squares',markers=True),width='stretch')
                st.caption('Use separation and diminishing within-group error as exploration aids; no automatic topology count is selected.')
                st.download_button('Download k comparison',comp.to_csv(index=False),'k_comparison.csv','text/csv')
            else:st.warning('Not enough valid distinct events/features for this comparison.')
        if st.button('Run grouping',type='primary'):
            try:
                with st.spinner('Comparing events; first DTW use may compile numerical routines…'):
                    if feature_mode:
                        if len(selected_features)<2:raise ValueError('Choose at least two features.')
                        info=cluster_features(features.loc[eligible,selected_features].to_numpy(),all_profiles[eligible],k,npc,algorithm)
                    elif algorithm=='Time-series k-means (DTW)':info=cluster_dtw(all_profiles[eligible],k,shape_only,radius)
                    else:info=cluster_profiles(all_profiles[eligible],k,shape_only)
                previous=st.session_state.get('group_result')
                if previous and previous.get('source')==fingerprint:
                    st.session_state.previous_groups=previous['table'][['event_index','group']].copy()
                assigned=df.copy();assigned['group']=-1;assigned.loc[eligible,'group']=info['labels'];assigned['exclusion_reason']=''
                assigned.loc[~available,'exclusion_reason']='selected signal unavailable'
                assigned.loc[df.samples<min_samples,'exclusion_reason']+='; below minimum sample count'
                if gate:assigned.loc[~(df.waveform_rmse_nA<=cutoff),'exclusion_reason']+='; original RMSE missing/above threshold'
                st.session_state.group_result={'signature':signature,'table':assigned,'info':info,'eligible':eligible,'source':fingerprint}
                st.session_state.pop('prepared_download',None)
            except Exception as ex:st.error(str(ex))
        stored=st.session_state.get('group_result');valid_groups=stored is not None and stored['signature']==signature
        if valid_groups:
            result=stored['table'];group_info=stored['info'];eligible=stored['eligible'];info=group_info
            st.write(f"Silhouette: {info['silhouette']:.3f}")
            if np.isfinite(info['ari']):st.write(f"Two-initialisation ARI: {info['ari']:.3f}")
            st.caption(info.get('metric','Euclidean profile distance') + '. Silhouettes in different spaces are not directly comparable evidence of physical superiority.')
            if 'scree' in info:
                st.plotly_chart(px.bar(x=np.arange(1,len(info['scree'])+1),y=info['scree'],labels={'x':'Principal component','y':'Explained variance fraction'}),width='stretch')
                st.dataframe(pd.DataFrame(info['loadings'],columns=selected_features,index=[f'PC{i+1}' for i in range(len(info['loadings']))]))
            if 'center_note' in info:st.caption(info['center_note'])
            previous=st.session_state.get('previous_groups')
            if previous is not None:
                joined=result[['event_index','group']].merge(previous,on='event_index',suffixes=('_current','_previous'))
                joined=joined[(joined.group_current>=0)&(joined.group_previous>=0)]
                if len(joined)>1:
                    from sklearn.metrics import adjusted_rand_score
                    st.write(f'Agreement with previous run on {len(joined)} shared clustered events: ARI {adjusted_rand_score(joined.group_previous,joined.group_current):.3f}')
                    st.dataframe(pd.crosstab(joined.group_previous,joined.group_current))
            st.caption('Geometric separation and limited numerical stability; neither is physical validation. Bands below show the 10th–90th percentile of individual profiles, not confidence intervals.')
            st.plotly_chart(profile_figure(info['profiles'],info['labels'],info['centers'],'relative amplitude' if shape_only and not feature_mode else 'nA'),width='stretch')
            counts=result.group.value_counts().sort_index().rename('events').to_frame();counts['fraction_all_loaded']=counts.events/len(result)
            counts['fraction_clustered']=np.where(counts.index>=0,counts.events/max(1,eligible.sum()),np.nan);st.dataframe(counts)
            emb=pd.DataFrame(info['embedding'],columns=['PC1','PC2']);emb['group']=info['labels'].astype(str);emb['event_index']=df.loc[eligible,'event_index'].to_numpy()
            st.plotly_chart(px.scatter(emb,x='PC1',y='PC2',color='group',hover_data=['event_index'],opacity=.7),width='stretch')
            st.caption(f"First two PCA coordinates shown; additional retained components may also drive feature clustering. For waveform methods PCA is only a visual projection. First-two-component variance: {100*sum(info['pca_variance']):.1f}%.")
            order=np.argsort(info['labels'],kind='stable');heat=go.Figure(go.Heatmap(z=info['profiles'][order],x=(np.arange(bins)+.5)/bins,y=np.arange(len(order)),colorscale='Viridis',colorbar=dict(title='Relative' if shape_only and not feature_mode else 'nA')))
            heat.update_layout(xaxis_title='Fraction of event duration',yaxis_title='Event row, ordered by group');st.plotly_chart(heat,width='stretch')
            gid=st.selectbox('Inspect group',range(k));local=np.flatnonzero(info['labels']==gid);global_positions=np.flatnonzero(eligible)
            nearest=local[np.argsort(np.linalg.norm(info['profiles'][local]-info['centers'][gid],axis=1))[:3]]
            random=np.random.default_rng(42).choice(local,min(3,len(local)),replace=False)
            st.caption('Up to three waveforms nearest to the displayed profile by pointwise distance, plus three random members; duplicates shown once. This example ranking is separate from the clustering distance.')
            for pos in dict.fromkeys(np.r_[nearest,random].tolist()):
                e=events[global_positions[pos]];st.write(f'Event {e.index}');st.plotly_chart(trace_figure(e,refs.get(e.index)),width='stretch',key=f'group_event_{e.index}')
        elif stored:st.info('Analysis settings changed. Run grouping again before using or exporting group assignments.')
    else:st.info('Shape clustering needs the waveform export; summary-only data can be plotted in Distributions.')

with dist_tab:
    st.markdown('These plots show event measurements in physical units. Histogram colours count events, while cluster colours show assignments when current results exist.')
    choices=[c for c in ['duration_ms','mean_blockade_nA','peak_blockade_nA','ecd_nA_ms','waveform_rmse_nA','segments','export_height_nA'] if c in result]
    c1,c2=st.columns(2);x=c1.selectbox('Horizontal variable',choices,format_func=lambda v:LABELS.get(v,v));y=c2.selectbox('Vertical variable',choices,index=min(1,len(choices)-1),format_func=lambda v:LABELS.get(v,v))
    c1,c2,c3=st.columns(3);logx=c1.checkbox('Log horizontal axis',True);logy=c2.checkbox('Log vertical axis',False);nbins=c3.slider('Histogram bins per axis',15,100,45)
    selection='All loaded events'
    if valid_groups:selection=st.selectbox('Population',['All loaded events','Clustered events','Excluded events']+[f'Group {i}' for i in range(k)])
    view=result
    if valid_groups:
        if selection=='Clustered events':view=result[result.group>=0]
        elif selection=='Excluded events':view=result[result.group<0]
        elif selection.startswith('Group '):view=result[result.group==int(selection.split()[-1])]
    try:
        scatter,heat,dropped,h,xe,ye=distribution_figures(view,x,y,logx,logy,nbins,'group' if valid_groups else None)
        st.plotly_chart(scatter,width='stretch');st.plotly_chart(heat,width='stretch')
        st.caption(f'{len(view)-dropped} displayed; {dropped} nonfinite/nonpositive-on-log-axis points omitted. Heatmap shows counts per bin, not probability density. Log axes use logarithmically spaced bins. Group -1 denotes excluded events.')
        hist=go.Figure()
        hist.add_trace(go.Bar(x=np.sqrt(xe[:-1]*xe[1:]) if logx else (xe[:-1]+xe[1:])/2,y=h.sum(axis=1),width=np.diff(xe)))
        hist.update_layout(xaxis_title=LABELS.get(x,x),yaxis_title='Event count');hist.update_xaxes(type='log' if logx else 'linear');st.plotly_chart(hist,width='stretch')
        st.download_button('Download plotted event table',view.to_csv(index=False),'distribution_events.csv','text/csv')
        st.download_button('Download heatmap counts and bin edges',npz_bytes({'counts':h,'x_edges':xe,'y_edges':ye,'x_variable':np.array(x),'y_variable':np.array(y)}),'density_histogram.npz')
    except ValueError as ex:st.info(str(ex))

with export_tab:
    provenance={'version':VERSION,'source_hash':fingerprint,'files':{name:{'filename':u.name,'sha256':hashlib.sha256(u.getvalue()).hexdigest()} for name,u in [('wave',wave_upload),('data',data_upload)] if u},
        'recording_settings':safe_settings(settings),'dataset_settings':safe_settings(dsettings),'profile_positions':bins,'grouping_current':valid_groups,
        'method':algorithm if events else 'summary-only','interpretation':'Exploratory signal groups, not molecular topology labels',
        'dataset_matching':{'start_column':int(start_col),'tolerance_seconds':tolerance,'matched_events':len(mapping)},
        'dataset_plot_columns':{'duration_seconds':int(duration_col),'height_nA':int(height_col)},'rejected':rejected,
        'candidate_fits':{str(i):{'method':r['method'],'parameters':r['parameters']} for i,r in refs.items()}}
    if events:provenance['grouping_settings']={'groups':k,'min_samples':min_samples,'rmse_exclusion_enabled':gate,'rmse_limit_nA':cutoff if gate else None,'representation':effective_representation,'shape_only':shape_only and not feature_mode,'selected_features':selected_features if feature_mode else [],'pca_components':npc if feature_mode else None,'dtw_radius':radius if algorithm=='Time-series k-means (DTW)' else None}
    if valid_groups:provenance['group_diagnostics']={'silhouette':group_info['silhouette'],'two_seed_ari':group_info['ari']}
    st.download_button('Download complete results CSV',result.to_csv(index=False),'event_results.csv','text/csv')
    st.download_button('Download provenance JSON',json.dumps(provenance,indent=2),'analysis_settings.json','application/json')
    if events:
        options=['All loaded events']+([f'Group {j}' for j in range(k)]+['Excluded events','All groups as separate ZIPs'] if valid_groups else [])
        scope=st.selectbox('Event files to download',options)
        export_key=hashlib.sha256((json.dumps(provenance,sort_keys=True)+scope).encode()).hexdigest()
        if st.button('Prepare event download'):
            with st.spinner('Packing original waveforms, candidate fits and matched summary rows…'):
                def bundle(label,subset):
                    ids=set(subset.event_index.astype(int));ev=[e for e in events if e.index in ids];meta={**provenance,'export_population':label,'exported_event_count':len(ev),'loaded_event_count':len(events)}
                    return cluster_bundle(ev,subset,settings,meta,refs,dataset,mapping,dsettings)
                if scope=='All groups as separate ZIPs':
                    buf=io.BytesIO()
                    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                        for j in sorted(result.group.unique()):
                            name=f'group_{j}' if j>=0 else 'excluded';z.writestr(name+'.zip',bundle(name,result[result.group==j]))
                    data=buf.getvalue();name='all_cluster_files.zip'
                else:
                    subset=result
                    if scope.startswith('Group '):subset=result[result.group==int(scope.split()[-1])]
                    elif scope=='Excluded events':subset=result[result.group<0]
                    data=bundle(scope,subset);name=scope.lower().replace(' ','_')+'.zip'
                st.session_state.prepared_download=(export_key,data,name)
        prepared=st.session_state.get('prepared_download')
        if prepared and prepared[0]==export_key:st.download_button('Download event files ZIP',prepared[1],prepared[2],'application/zip')
        st.caption('ZIP includes events.npz, results CSV and provenance. If a dataset is linked, dataset_subset.npz preserves its matched X rows. Original fits are unchanged; candidates use separate REFINED_* arrays. IDs are preserved. These NPZs reload in this app; external NanoSense import compatibility is not established.')

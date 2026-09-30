import hashlib,io,json,zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from analysis import load_events,load_dataset,link_dataset,describe,refine,fit_metrics,feature_table,cluster_features,auto_cluster_features,cluster_profiles,cluster_dtw,FEATURE_DESCRIPTIONS,safe_settings
from physical import level_features,PHYSICAL_DESCRIPTIONS
from workflow import active_events,signal_events,fitting_bytes,match_raw,bundle
from plots import trace_figure,distribution_figures,profile_figure,LABELS,scientific,figure_archive,profile_archive,cluster_pca_figure,member_profile_figure,publication_archive,representative_time_examples,time_example_figure

st.set_page_config(page_title='DNA Event Lab',page_icon='🧬',layout='wide')
st.title('DNA Event Lab')
st.caption('Load → inspect → refine → plot → cluster → save')
st.sidebar.title('Your analysis')
S=st.session_state
STEPS=['1 · Load files','2 · Inspect events','3 · Refine and save fits','4 · Current–duration plots','5 · Cluster events','6 · Save clusters']
step=st.sidebar.radio('Step',STEPS,key='workflow_step')

def move_step(delta):
    pos=STEPS.index(S.get('workflow_step',STEPS[0]))
    S.workflow_step=STEPS[max(0,min(len(STEPS)-1,pos+delta))]

def navigation(location):
    pos=STEPS.index(step);left,middle,right=st.columns([1,2,1])
    left.button('← Previous',on_click=move_step,args=(-1,),disabled=pos==0,key='previous_'+location)
    middle.caption(f'Step {pos+1} of {len(STEPS)} · {step.split(" · ",1)[1]}')
    ready=('project' in S and S.get('recording_confirmed',False))
    if pos==4:ready=ready and S.get('group') is not None
    right.button('Next →',on_click=move_step,args=(1,),disabled=pos==len(STEPS)-1 or not ready,key='next_'+location)

navigation('top')
st.sidebar.caption('Work with one recording at a time. Groups describe signal similarities; topology labels need physical validation.')
S=st.session_state
if st.sidebar.button('Clear files and start over'):
    for key in list(S.keys()):
        del S[key]
    st.rerun()
# Render upload widgets on every step so Streamlit does not discard their state.
with st.sidebar.expander('Files retained for this session',expanded=step.startswith('1')):
    uploads={}
    for key,label in [('dataset','Summary · .dataset.npz'),('raw','Recorded events · .eventdata.npz / .event_data.npz'),('fits','Fitted events · .eventfitting.npz / .event_fitting.npz')]:
        uploads[key]=st.file_uploader(label,type='npz',key='upload_'+key)
    st.caption('Files stay here while you move between steps. Replacing a file takes effect after Load and check files. A new browser session requires uploading again.')
if 'project' in S:
    with st.sidebar.expander('Currently loaded recording'):
        for item in S.project['files'].values():st.write(item['name'])

def show(fig,key):
    st.plotly_chart(scientific(fig),width='stretch',key=key,theme=None,config={'displaylogo':False,'toImageButtonOptions':{'format':'svg','filename':key,'width':900,'height':600}})

def next_step(label):navigation('bottom')

if step.startswith('1'):
    st.header('1 · Load the three files')
    st.write('Choose the dataset, recorded events and fitted events from the same recording. Both eventdata and event_data naming styles are accepted.')
    st.info('Upload or replace the three files in the left-hand Files panel. They remain available throughout this run.')
    with st.expander('Advanced: event matching'):
        col=st.number_input('Dataset start-time column (zero-based)',0,100,8)
        tol=st.number_input('Start-time tolerance (µs)',.001,1000.,.1,format='%.3f')*1e-6
        st.caption('Events are linked by unique start times, never by row position. Current units must be nA and time units seconds.')
    if st.button('Load and check files',type='primary',disabled=not all(u is not None for u in uploads.values())):
        try:
            blobs={k:u.getvalue() for k,u in uploads.items()}
            events,settings,rejected=load_events(blobs['fits']);raw,rsettings,rreject=load_events(blobs['raw']);dataset,dsettings=load_dataset(blobs['dataset'])
            mapping,status=link_dataset(events,dataset,col,tol);rawmap,rstatus=match_raw(events,raw,tol)
            fingerprint=hashlib.sha256(b''.join(hashlib.sha256(blobs[k]).digest() for k in sorted(blobs))).hexdigest()
            same_project=('project' in S and S.project['fingerprint']==fingerprint and S.project['matching']=={'start_column':col,'tolerance_s':tol})
            if not same_project:
                for key in ['refs','group','prepared','fitfile','fit_errors','recording_confirmed','confirmation_widget']:S.pop(key,None)
            S.project=dict(events=events,raw=raw,settings=settings,dsettings=dsettings,dataset=dataset,mapping=mapping,rawmap=rawmap,status=status,rstatus=rstatus,rejected=rejected+rreject,
                fingerprint=fingerprint,files={k:{'name':u.name,'sha256':hashlib.sha256(blobs[k]).hexdigest()} for k,u in uploads.items()},matching={'start_column':col,'tolerance_s':tol})
            if not same_project:S.refs={}
            st.success('Files checked. Existing analysis retained.' if same_project else 'Files read successfully. Review the matches below.')
        except Exception as ex:st.error(str(ex))
    if 'project' in S:
        p=S.project;c=st.columns(3);c[0].metric('Fitted events',len(p['events']));c[1].metric('Matched recorded events',len(p['rawmap']));c[2].metric('Matched dataset rows',len(p['mapping']))
        st.caption('Loaded: '+' · '.join(v['name'] for v in p['files'].values()))
        if len(p['rawmap'])!=len(p['events']) or len(p['mapping'])!=len(p['events']):st.warning('Some events could not be uniquely matched. They remain visible; cluster downloads include only their available linked data.')
        with st.expander('Matching details and rejected events'):
            st.dataframe(p['status'],hide_index=True);st.dataframe(p['rstatus'],hide_index=True);st.write(p['rejected'])
        S.recording_confirmed=st.checkbox('I checked that these three files belong to the same recording.',value=S.get('recording_confirmed',False),key='confirmation_widget')
        next_step('2 · Inspect events')
    st.stop()
if 'project' not in S:st.info('Start at 1 · Load files.');st.stop()
if not S.get('recording_confirmed'):st.info('Confirm the recording match in 1 · Load files before continuing.');st.stop()
p=S.project;events=p['events'];refs=S.setdefault('refs',{});active=active_events(events,refs)
measured,profiles=describe(events,32,refs)
measured['raw_event_index']=[p['rawmap'][e.index].index if e.index in p['rawmap'] else np.nan for e in events]
measured['dataset_row']=[p['mapping'].get(e.index,np.nan) for e in events]
measured['fit_source']=['refined' if e.index in refs else 'uploaded' for e in events]
refhash=hashlib.sha256(b''.join(str(i).encode()+r['fit'].tobytes() for i,r in sorted(refs.items()))).hexdigest()
meta={'version':'0.5.1','source_hash':p['fingerprint'],'files':p['files'],'matching':p['matching'],'settings':safe_settings(p['settings']),
      'refinements':{str(i):{'method':r['method'],'parameters':r['parameters']} for i,r in refs.items()},'fit_hash':refhash}
st.sidebar.metric('Loaded events',len(events));st.sidebar.metric('Refined fits',len(refs))
st.sidebar.caption('Unrefined events retain their uploaded fits.')

def choose_event():
    idx=st.selectbox('Event ID',[e.index for e in events],key='event_choice');return next(e for e in events if e.index==idx)

if step.startswith('2'):
    st.header('2 · Inspect the recorded events')
    e=choose_event();units=st.radio('Vertical axis',['Current I (nA)','Blockade ΔI (nA)'],horizontal=True)
    show(trace_figure(e,refs.get(e.index),current=units.startswith('Current')),'inspection')
    st.caption('Blue: measured trace saved with the fitting file. Orange: uploaded fit. Purple: refined fit, when available. Dashed lines mark the detected event boundaries.')
    if e.index in p['rawmap']:
        with st.expander('Compare the independently saved eventdata trace'):
            show(trace_figure(p['rawmap'][e.index],current=units.startswith('Current')),'recorded_event')
            st.caption('This is the matched event from eventdata; it is not silently substituted for the trace used by the original fitter.')
    st.dataframe(measured[measured.event_index==e.index],hide_index=True)
    with st.expander('Fit disagreement across the recording'):
        st.write('RMSE measures pointwise disagreement, not whether an event is physically valid. No events are excluded by RMSE.')
        show(px.scatter(measured,x='duration_ms',y='waveform_rmse_nA',labels=LABELS,hover_data=['event_index']),'fit_disagreement')
    next_step('3 · Refine and save fits')
elif step.startswith('3'):
    st.header('3 · Refine and save fits')
    st.write('Try one event first. Review its fit before applying the method to all events. Uploaded files are never overwritten.')
    e=choose_event();method=st.selectbox('Refinement method',['Segment means','New levels (PELT)','Rounded pulse (Gaussian)'])
    st.caption({'Segment means':'Keep the existing step boundaries and recalculate the level heights.','New levels (PELT)':'Find new step boundaries; the penalty controls how readily another step is added.','Rounded pulse (Gaussian)':'Fit one smooth pulse. Use for single rounded events, not multilevel events.'}[method])
    with st.expander('Advanced refinement settings',expanded=method=='New levels (PELT)'):
        minimum=st.number_input('Minimum step duration (µs)',5.,10000.,25.,step=5.,disabled=method!='New levels (PELT)')
        penalty=st.number_input('Extra-step penalty',.1,100.,8.,step=.5,disabled=method!='New levels (PELT)')
    scope=st.radio('Refine',['Selected event','All events'],horizontal=True)
    if st.button('Calculate refined fits',type='primary'):
        targets=events if scope=='All events' else [e];errors=[];bar=st.progress(0.)
        for j,item in enumerate(targets):
            try:refs[item.index]=refine(item,method,minimum,penalty)
            except Exception as ex:errors.append({'event_id':item.index,'reason':str(ex)})
            bar.progress((j+1)/len(targets))
        S.refs=refs;S.fit_errors=errors
        for key in ['group','prepared','fitfile']:S.pop(key,None)
        st.rerun()
    show(trace_figure(e,refs.get(e.index)),'refinement')
    if e.index in refs:st.dataframe(pd.DataFrame([fit_metrics(e,e.fit),fit_metrics(e,refs[e.index]['fit'])],index=['Uploaded fit','Refined fit']))
    st.caption('A lower RMSE is not proof of a more accurate physical model. Baseline and event boundaries remain fixed.')
    if S.get('fit_errors'):st.warning('Some refinements failed; their previous fits remain selected.');st.dataframe(pd.DataFrame(S.fit_errors),hide_index=True)
    if st.button('Discard all refinements',disabled=not refs):
        S.refs={}
        for key in ['group','prepared','fitfile']:S.pop(key,None)
        st.rerun()
    if st.button('Prepare new eventfitting file',disabled=not refs):S.fitfile=(refhash,fitting_bytes(events,refs,p['settings'],meta))
    if S.get('fitfile') and S.fitfile[0]==refhash:st.download_button('Save refined.eventfitting.npz',S.fitfile[1],'refined.eventfitting.npz','application/octet-stream')
    st.info(f'{len(refs)} refined fits selected; {len(events)-len(refs)} uploaded fits retained. These selected fits are available in the next steps immediately. Saving does not require re-uploading.')
    next_step('4 · Current–duration plots')
elif step.startswith('4'):
    st.header('4 · Current blockade versus event duration')
    st.write('Each point is one event: duration Δt on the horizontal axis, blockade depth ΔI on the vertical axis. This is a distribution of events, not a frequency spectrum.')
    source=st.radio('Measurements from',['Measured trace','Selected fits'],horizontal=True)
    try:plot_events=signal_events(active,source);table,_=describe(plot_events)
    except ValueError as ex:st.warning(str(ex));st.stop()
    height=st.selectbox('Blockade measurement',['mean_blockade_nA','peak_blockade_nA'],format_func=lambda x:LABELS[x])
    logx=st.checkbox('Logarithmic duration axis',True)
    scatter,heat,dropped,*_=distribution_figures(table,'duration_ms',height,logx,False)
    show(scatter,'current_duration');show(heat,'event_counts')
    st.caption(f'{len(table)-dropped} events plotted; {dropped} omitted because values are incompatible with the axes. Colour in the heatmap is events per bin. Data source: {source}.')
    st.download_button('Save plotted measurements CSV',table.to_csv(index=False),'current_duration.csv')
    if st.button('Prepare publication figures'):S.figures=(p['fingerprint'],refhash,source,height,logx,figure_archive(table,'duration_ms',height,logx))
    if S.get('figures') and S.figures[:5]==(p['fingerprint'],refhash,source,height,logx):st.download_button('Save PDF, SVG and 600 dpi PNG figures',S.figures[5],'current_duration_figures.zip','application/zip')
    next_step('5 · Cluster events')
elif step.startswith('5'):
    st.header('5 · Cluster the events')
    st.write('The default feature set describes sustained levels, their changes and optional duration. Automatic mode standardises those features, retains PCA components and suggests a cluster count from several diagnostics. Cluster numbers are signal groups, not DNA topology labels.')
    cfg=S.get('cluster_config',{})
    source_options=['Selected fits','Measured trace'];source=st.radio('Signal used for clustering',source_options,index=source_options.index(cfg.get('source','Selected fits')) if cfg.get('source','Selected fits') in source_options else 0,horizontal=True)
    methods=['Automatic PCA + agglomerative (exploratory)','PCA + agglomerative (manual)','PCA + k-means (manual)','Waveform k-means (manual)','Time-series k-means (DTW, manual)']
    previous=cfg.get('algorithm',methods[0]);algorithm=st.selectbox('Method',methods,index=methods.index(previous) if previous in methods else 0)
    is_feature=algorithm.startswith('Automatic') or algorithm.startswith('PCA')
    is_auto=algorithm.startswith('Automatic')
    bins=cfg.get('bins',32);radius=cfg.get('radius',4);k=cfg.get('k',4);npc=cfg.get('npc',2);kmax=cfg.get('kmax',8);variance_pct=cfg.get('variance_pct',95);repeats=cfg.get('repeats',6)
    fields=[];feature_scope='Waveforms';physical_params={}
    if is_feature:
        scopes=['Physical level features','Legacy features (comparison)']
        feature_scope=st.radio('Feature set',scopes,index=scopes.index(cfg.get('feature_scope',scopes[0])) if cfg.get('feature_scope') in scopes else 0,horizontal=True)
        if feature_scope==scopes[0]:
            previous_params=cfg.get('physical_params',{})
            st.caption('Selected step fits define plateau boundaries in both signal modes. Heights come from the selected fit or the measured samples within each plateau. These are signal descriptors, not identified DNA topologies.')
            with st.expander('Physical feature settings',expanded=True):
                c1,c2,c3=st.columns(3)
                min_us=c1.number_input('Minimum plateau duration (µs)',min_value=0.,value=float(previous_params.get('min_duration_us',25.)))
                min_height=c2.number_input('Minimum level difference (nA)',min_value=0.,value=float(previous_params.get('min_height_nA',.1)),format='%.3f')
                noise_mult=c3.number_input('Noise multiplier for level merging',min_value=0.,value=float(previous_params.get('noise_multiplier',3.)))
                st.caption('Adjacent levels merge when their difference is ≤ max(minimum level difference, noise multiplier × robust noise scale). Minimum duration is also at least three sample intervals. Set these analysis thresholds for your sampling and instrument bandwidth; the defaults are provisional.')
                omit_edges=st.checkbox('Omit short boundary plateaus from physical features',value=previous_params.get('omit_short_boundaries',True))
                st.caption('Only the first and last merged plateaus can be omitted. This flags unresolved boundaries, not proven artefacts. Internal short levels still exclude the event. Original traces, fits and total event duration are preserved; deeper-level fractions use retained resolved duration.')
                use_ref=st.checkbox('Use a calibrated single-file blockade reference',value=previous_params.get('reference_nA') is not None)
                reference=st.number_input('Single-file reference blockade (nA)',min_value=.000001,value=float(previous_params.get('reference_nA') or 1.),format='%.4f') if use_ref else None
                use_deep=st.checkbox('Include time fraction above a deeper-blockade threshold',value=previous_params.get('deep_threshold_nA') is not None)
                threshold=st.number_input('Deeper-blockade threshold (nA)',min_value=.000001,value=float(previous_params.get('deep_threshold_nA') or 1.),format='%.4f') if use_deep else None
                if use_ref or use_deep:st.caption('Supply a reference/threshold established for this recording condition. A threshold crossing does not establish a fold or strand count. Events near the threshold are flagged.')
                include_duration=st.checkbox('Include event duration',value=cfg.get('include_duration',True))
            physical_params=dict(min_duration_us=min_us,min_height_nA=min_height,noise_multiplier=noise_mult,reference_nA=reference,deep_threshold_nA=threshold,omit_short_boundaries=omit_edges)
            fields=['deepest_plateau_ratio' if use_ref else 'deepest_plateau_nA','resolved_transitions','transition_direction']
            if use_deep:fields.append('deep_time_fraction')
            if include_duration:fields.append('duration_ms')
        else:
            include_duration=True
            fields=st.multiselect('Features used for clustering',list(FEATURE_DESCRIPTIONS),default=[f for f in cfg.get('fields',list(FEATURE_DESCRIPTIONS)) if f in FEATURE_DESCRIPTIONS] or list(FEATURE_DESCRIPTIONS))
        if is_auto:
            c1,c2,c3=st.columns(3)
            kmax=c1.slider('Maximum clusters to test',3,10,int(kmax))
            variance_pct=c2.slider('PCA variance retained (%)',80,99,int(variance_pct))
            repeats=c3.select_slider('Stability repeats',options=[4,6,8,10,12],value=int(repeats) if int(repeats) in [4,6,8,10,12] else 6)
            st.caption(f'Automatic search tests k = 2…{kmax}; it cannot establish that separate populations exist. It ranks silhouette ↑, Calinski–Harabasz ↑, Davies–Bouldin ↓ and subsampling stability ↑, with extra weight on silhouette and stability. PCA keeps the smallest number of components explaining at least {variance_pct}% of scaled-feature variance.')
        else:
            c1,c2=st.columns(2);k=c1.slider('Number of clusters',2,10,int(k));npc=c2.slider('PCA components',1,max(1,min(10,len(fields))),min(int(npc),max(1,min(10,len(fields)))))
    else:
        k=st.slider('Number of clusters',2,10,int(k))
    with st.expander('Advanced clustering settings'):
        bins=st.select_slider('Waveform positions',[16,32,64],value=bins if bins in [16,32,64] else 32)
        if 'DTW' in algorithm:radius=st.slider('DTW alignment radius',1,16,int(radius))
        if is_feature:
            st.caption('Feature values are z-score standardised before PCA. Constant features are removed; automatic mode also removes near-duplicates (|r| ≥ 0.98). Plateau count and direction depend on resolved fit boundaries and instrument resolution. Legacy wavelet summaries are exploratory.')
            st.json({f:({**FEATURE_DESCRIPTIONS,**PHYSICAL_DESCRIPTIONS})[f] for f in fields})
        else:st.caption('Waveform methods compare duration-normalised profiles while retaining blockade depth. DTW permits local time alignment.')
    S.cluster_config=dict(source=source,algorithm=algorithm,k=k,fields=fields,npc=npc,bins=bins,radius=radius,kmax=kmax,variance_pct=variance_pct,repeats=repeats,feature_scope=feature_scope,physical_params=physical_params,include_duration=include_duration if is_feature else True)
    signature=(p['fingerprint'],refhash,source,algorithm,k,tuple(fields),npc,bins,radius,kmax,variance_pct,repeats,feature_scope,json.dumps(physical_params,sort_keys=True))
    physical_audit=None;physical_sequences=None
    if is_feature and feature_scope=='Physical level features':
        physical_audit,physical_sequences=level_features(active,source,**physical_params)
        usable=int(physical_audit.physical_eligible.sum())
        st.info(f'{usable} of {len(active)} events have resolved plateaus under the current settings.')
        with st.expander('Check physical-feature eligibility before clustering'):
            st.dataframe(physical_audit,hide_index=True)
            st.download_button('Save physical feature audit CSV',physical_audit.to_csv(index=False),'physical_feature_audit.csv','text/csv')
        edge_count=int(physical_audit.get('boundary_omission_applied',pd.Series(dtype=bool)).fillna(False).sum())
        if edge_count:st.caption(f'{edge_count} events have brief boundary plateaus omitted from physical descriptors. Their omitted time is recorded in the audit.')
        flagged_ids=physical_audit.loc[(~physical_audit.physical_eligible) | physical_audit.get('boundary_omission_applied',False).fillna(False),'event_index'].tolist() if 'boundary_omission_applied' in physical_audit else physical_audit.loc[~physical_audit.physical_eligible,'event_index'].tolist()
        if flagged_ids:
            with st.expander('Inspect flagged or boundary-adjusted events'):
                audit_id=st.selectbox('Flagged event ID',flagged_ids)
                audit_event=next(e for e in events if e.index==audit_id)
                audit_fig=trace_figure(audit_event,refs.get(audit_id))
                audit_levels=physical_sequences.loc[physical_sequences.event_index==audit_id]
                for level in audit_levels.itertuples():
                    if level.level_status!='resolved':audit_fig.add_vrect(x0=level.start_from_event_ms,x1=level.start_from_event_ms+level.duration_ms,fillcolor='#e89b35',opacity=.25,line_width=0)
                show(audit_fig,'physical_flagged_trace')
                st.caption('Amber shading marks omitted or unresolved levels. Their status and exact duration are listed below. Shading does not establish that a segment is noise.')
                st.dataframe(audit_levels,hide_index=True)
        st.caption('Unresolved events are excluded from this feature model and retained for download. Refine their step fits or review the resolution settings before comparing results.')
    if st.button('Run clustering',type='primary'):
        try:
            with st.spinner('Grouping events… automatic stability testing can take a little longer.'):
                excluded=pd.DataFrame();sequences=pd.DataFrame();eligible_active=active
                if is_feature and feature_scope=='Physical level features':
                    audit,sequences=physical_audit,physical_sequences
                    excluded=audit.loc[~audit.physical_eligible].copy()
                    positions=np.flatnonzero(audit.physical_eligible.to_numpy())
                    eligible_active=[active[int(i)] for i in positions]
                    if len(eligible_active)<3:raise ValueError(f'Only {len(eligible_active)} events have resolved step plateaus. Inspect/refine fits or review resolution settings; at least three are needed.')
                    feat=audit.iloc[positions].reset_index(drop=True)
                    table=measured.iloc[positions].reset_index(drop=True)
                else:
                    table=measured.copy()
                sig=signal_events(eligible_active,source)
                if not (is_feature and feature_scope=='Physical level features'):feat=feature_table(sig)
                _,prof=describe(sig,bins)
                if is_feature:
                    if len(fields)<2:raise ValueError('Select at least two event features.')
                    matrix=feat[fields].to_numpy()
                    base_method='PCA + agglomerative (DNA concept)' if 'agglomerative' in algorithm.lower() or is_auto else 'PCA + k-means (nanorod concept)'
                    if is_auto:
                        info=auto_cluster_features(matrix,prof,base_method,2,kmax,variance_pct/100,repeats,.8)
                        selected_k=int(info['selected_k'])
                    else:
                        info=cluster_features(matrix,prof,k,npc,base_method);selected_k=int(k)
                elif 'DTW' in algorithm:
                    info=cluster_dtw(prof,k,False,radius);selected_k=int(k)
                else:
                    info=cluster_profiles(prof,k,False);selected_k=int(k)
                table['cluster']=info['labels']
                for col in feat.columns:
                    if col!='event_index':table['clustering_'+col]=feat[col].to_numpy()
                clustering_meta={'source':source,'method':algorithm,'k':selected_k,'selection':'automatic' if is_auto else 'manual','features':fields if is_feature else [],
                    'pca_components':info.get('n_components',npc if is_feature else None),'pca_variance_target':variance_pct/100 if is_auto else None,
                    'positions':bins,'dtw_radius':radius,'automatic_k_range':[2,kmax] if is_auto else None,'stability_repeats':repeats if is_auto else None,
                    'feature_set':feature_scope,'physical_settings':physical_params,'excluded_events':len(excluded),
                    'retained_features':[f for f,keep in zip(fields,info.get('feature_keep_mask',[True]*len(fields))) if keep] if is_feature else [],
                    'feature_mean':info.get('feature_mean'),'feature_scale':info.get('feature_scale'),'pca_loadings':info.get('loadings'),
                    'diagnostics':info.get('selection_table'),'selection_method':info.get('selection_method'),'selection_warning':info.get('selection_warning'),
                    'stability_sample_size':info.get('stability_sample_size'),'effective_stability_fraction':info.get('effective_stability_fraction')}
                S.group={'signature':signature,'table':table,'info':info,'excluded':excluded,'sequences':sequences,'meta':{**meta,'clustering':clustering_meta}}
                S.pop('prepared',None)
        except Exception as ex:st.error(str(ex))
    g=S.get('group')
    if g and g['signature']!=signature:
        S.pop('group',None);S.pop('prepared',None);st.info('Settings changed. Run clustering again to update the results.');st.stop()
    if g:
        info=g['info'];table=g['table'];selected_k=len(np.unique(info['labels']))
        if info.get('selected_k') is not None:
            st.success(f'Automatic search suggests {selected_k} clusters for {len(table)} events using {source.lower()}.')
            st.caption(info.get('selection_method',''))
            if info.get('selection_warning'):st.warning(info['selection_warning'])
        else:st.success(f'{len(table)} events grouped into {selected_k} clusters using {source.lower()}.')
        excluded=g.get('excluded',pd.DataFrame());sequences=g.get('sequences',pd.DataFrame())
        if len(excluded):
            st.warning(f'{len(excluded)} of {len(events)} events were excluded from physical-feature clustering because their plateaus could not be resolved under these settings.')
            with st.expander('Excluded events and reasons'):
                st.dataframe(excluded,hide_index=True)
                st.download_button('Save exclusion audit CSV',excluded.to_csv(index=False),'physical_exclusions.csv','text/csv')
        if len(sequences):
            with st.expander('Resolved plateau measurements'):
                st.dataframe(sequences,hide_index=True)
                st.download_button('Save resolved levels CSV',sequences.to_csv(index=False),'resolved_levels.csv','text/csv')
            if 'clustering_threshold_sensitive' in table and table.clustering_threshold_sensitive.any():st.warning(f'{int(table.clustering_threshold_sensitive.sum())} included events have levels near the deeper-blockade threshold. Compare results with alternative thresholds.')
        summary=table.groupby('cluster').agg(Events=('event_index','size'),Median_duration_ms=('duration_ms','median'),Median_mean_blockade_nA=('mean_blockade_nA','median'),Median_peak_blockade_nA=('peak_blockade_nA','median'))
        summary['Fraction']=summary.Events/len(table)
        st.dataframe(summary)
        if g['meta']['clustering'].get('feature_set')=='Physical level features':
            st.write('Median physical features by cluster')
            st.dataframe(table.groupby('cluster')[["clustering_"+f for f in g['meta']['clustering']['features']]].median().rename(columns=lambda c:c.removeprefix('clustering_')))
        st.caption('Summary measurements above come from the measured traces. Compare these with the selected-signal profiles and original event traces below.')
        st.subheader('Members and representative profile of each cluster')
        columns=st.columns(3)
        for group_id in range(selected_k):
            with columns[group_id%3]:
                show(member_profile_figure(info['profiles'],info['labels'],info['centers'],group_id),f'cluster_members_{group_id}')
        st.caption('Faint curves: up to 40 uniformly sampled profiles per cluster (seed 42). Red curve: representative from all members—mean profile for feature/k-means methods, aligned barycentre for DTW. These profiles use the selected clustering signal.')
        st.subheader('Comparison of the representative profiles')
        show(profile_figure(info['profiles'],info['labels'],info['centers']),'cluster_profiles')
        st.caption('Profile bands are member 10th–90th percentiles, not uncertainty in the mean. Group IDs are ordered by increasing mean profile blockade for feature clustering.')
        time_examples=representative_time_examples([e for e in active if e.index in set(table.event_index)],info,g['meta']['clustering']['source'])
        with st.expander('Representative example events in actual time (ms)',expanded=True):
            show(time_example_figure(time_examples),'cluster_actual_time')
            st.caption('One real event per cluster nearest to the representative by pointwise profile distance, aligned at detected start. Original sample times and event duration are retained. These are example events, not averaged centroids; the source matches the clustering signal.')
            st.download_button('Save actual-time example curves',time_examples.to_csv(index=False),'actual_time_examples.csv','text/csv')
        st.subheader('PC1–PC2 view of the event groups')
        embedding=np.asarray(info['embedding'])
        variance=info.get('pca_variance',[])
        axis_labels={f'PC{j+1}':f'PC{j+1} ({100*variance[j]:.1f}% variance)' if j<len(variance) else f'PC{j+1}' for j in range(2)}
        projection=pd.DataFrame(embedding,columns=['PC1','PC2'])
        projection['Cluster']=table['cluster'].astype(str).to_numpy()
        projection['Event ID']=table['event_index'].to_numpy()
        projection['Duration (ms)']=table['duration_ms'].to_numpy()
        projection['Measured mean blockade (nA)']=table['mean_blockade_nA'].to_numpy()
        outlines=st.checkbox('Show shaded cluster outlines',True)
        fig=cluster_pca_figure(projection,axis_labels,outlines)
        show(fig,'cluster_projection')
        st.caption('Shading outlines each group’s convex hull in this 2D view. Outlines may overlap and are not confidence regions or clustering boundaries. Black crosses mark mean projected group positions.')
        if g['meta']['clustering']['features']:
            retained=info.get('n_components',2)
            if retained==1:
                st.caption('Only one PCA component was retained. PC2 is shown as zero; clustering uses PC1 only.')
            else:
                st.caption(f'This plot displays the first two PCA coordinates. Clustering used {retained} retained components from the selected signal features; differences in other components may be hidden here.')
        else:
            st.caption('PCA is used only to display the waveform groups here. Waveform k-means and DTW form their groups from the waveform profiles, not from these two plotted coordinates.')
        st.caption('Each point is one event. Colours indicate its assigned cluster. Hover to find its event ID, then inspect the original trace below. The axes combine signal measurements and have no direct current/time units.')
        st.download_button('Save PC1–PC2 coordinates CSV',projection.to_csv(index=False),'pca_coordinates.csv','text/csv')
        if g['meta']['clustering']['features'] and 'loadings' in info:
            kept=[f for f,keep in zip(g['meta']['clustering']['features'],info.get('feature_keep_mask',[True]*len(g['meta']['clustering']['features']))) if keep]
            loads=np.asarray(info['loadings']);pcs=min(2,len(loads));rows=[]
            for j in range(pcs):
                var=100*info['scree'][j] if j<len(info.get('scree',[])) else np.nan
                for name,value in zip(kept,loads[j]):rows.append({'Feature':name.replace('_',' ').replace(' nA','').title(),'Loading':value,'PC':f'PC{j+1} ({var:.1f}%)'})
            if rows:
                loadfig=px.bar(pd.DataFrame(rows),x='Loading',y='Feature',color='PC',barmode='group',orientation='h',title='What drives the PCA separation?')
                with st.expander('Which features contribute to the PCA coordinates?'):
                    show(loadfig,'pca_loadings')
                    st.caption('Loading size describes contribution to a coordinate, not proof that the feature separates physical populations. The overall sign of a component is arbitrary; relative signs describe relationships between features.')
        if info.get('selection_table'):
            with st.expander('Why was this number of clusters selected?',expanded=False):
                diag=pd.DataFrame(info['selection_table']);display=diag[['k','silhouette','calinski_harabasz','davies_bouldin','stability','stability_sd','min_cluster_size','consensus_rank']].copy()
                display.columns=['k','Silhouette ↑','Calinski–Harabasz ↑','Davies–Bouldin ↓','Stability ARI ↑','Stability SD','Smallest cluster','Consensus rank ↓']
                st.dataframe(display.round(4),hide_index=True)
                dplot=diag.melt(id_vars='k',value_vars=['silhouette','stability'],var_name='Diagnostic',value_name='Score')
                show(px.line(dplot,x='k',y='Score',color='Diagnostic',markers=True),'automatic_cluster_diagnostics')
                st.caption(f'Stability uses {info.get("stability_sample_size","—")} events per repeat ({100*info.get("effective_stability_fraction",.8):.1f}% of this recording), capped at 600. Repeats refit preprocessing and clustering.')
                st.caption('The suggested k has the best weighted rank consensus across four diagnostics, with silhouette and stability weighted twice. This is a reproducible, predefined signal-structure criterion, not proof that the groups are distinct DNA topologies.')
        with st.expander('Numerical diagnostics'):
            st.write('Silhouette:',info.get('silhouette'))
            if 'calinski_harabasz' in info:st.write('Calinski–Harabasz:',info['calinski_harabasz'])
            if 'davies_bouldin' in info:st.write('Davies–Bouldin:',info['davies_bouldin'])
            if 'n_components' in info:
                st.write('PCA components retained:',info['n_components']);st.write('Cumulative PCA variance retained:',float(np.sum(info.get('scree',[])[:info['n_components']])))
            st.caption('These scores measure geometric separation and stability. They do not establish physical identity or topology accuracy.')
        cluster=st.selectbox('View traces from cluster',sorted(table.cluster.unique()));ids=table.loc[table.cluster==cluster,'event_index'].tolist()
        eid=st.selectbox('Event in this cluster',ids);e=next(e for e in events if e.index==eid);show(trace_figure(e,refs.get(e.index)),'cluster_event')
        if st.button('Prepare cluster profile figures'):S.cluster_figures=(signature,profile_archive(info['profiles'],info['labels'],info['centers']))
        if S.get('cluster_figures') and S.cluster_figures[0]==signature:st.download_button('Save profile PDF, SVG and PNG figures',S.cluster_figures[1],'cluster_profiles.zip')
        if st.button('Prepare combined publication figure'):
            S.publication_figure=((signature,outlines),publication_archive(info,table['event_index'].to_numpy(),g['meta']['clustering']['source'],g['meta']['clustering']['method'],outlines,time_examples))
        if S.get('publication_figure') and S.publication_figure[0]==(signature,outlines):
            st.download_button('Save combined PDF, SVG and 600 dpi PNG',S.publication_figure[1],'cluster_publication_figure.zip','application/zip')
        next_step('6 · Save clusters')
elif step.startswith('6'):
    st.header('6 · Save cluster data')
    g=S.get('group')
    if not g:st.info('Run clustering in step 5 first.');st.stop()
    st.caption('Saved grouping: '+g['meta']['clustering']['method']+' · '+g['meta']['clustering']['source'])
    table=g['table'];st.download_button('Save all event assignments CSV',table.to_csv(index=False),'cluster_assignments.csv')
    choice=st.selectbox('Download',['All clusters separately']+[f'Cluster {i}' for i in sorted(table.cluster.unique())])
    key=(str(g['signature']),choice)
    if st.button('Prepare cluster files',type='primary'):
        groups=sorted(table.cluster.unique()) if choice=='All clusters separately' else [int(choice.split()[-1])]
        outer=io.BytesIO()
        with zipfile.ZipFile(outer,'w',zipfile.ZIP_DEFLATED) as z:
            for i in groups:
                sub=table[table.cluster==i];ids=set(sub.event_index);ev=[e for e in events if e.index in ids]
                z.writestr(f'cluster_{i}.zip',bundle(ev,p['rawmap'],refs,p['dataset'],p['mapping'],p['settings'],p['dsettings'],sub,{**g['meta'],'cluster':int(i)},g.get('sequences')))
            excluded=g.get('excluded',pd.DataFrame())
            if choice=='All clusters separately' and len(excluded):
                z.writestr('physical_exclusions.csv',excluded.to_csv(index=False))
                ids=set(excluded.event_index);ev=[e for e in events if e.index in ids]
                z.writestr('unresolved_events.zip',bundle(ev,p['rawmap'],refs,p['dataset'],p['mapping'],p['settings'],p['dsettings'],excluded,{**g['meta'],'assignment':'excluded'},g.get('sequences')))
        S.prepared=(key,outer.getvalue())
    if S.get('prepared') and S.prepared[0]==key:st.download_button('Save cluster files ZIP',S.prepared[1],'cluster_files.zip','application/zip')
    st.write('Each cluster contains its selected eventfitting file, matched eventdata and dataset files, event table and analysis settings. Physical-feature runs also include resolved_levels.csv; the all-clusters download retains excluded events in unresolved_events.zip. Uploaded fits are also retained inside the new fitting file.')
    st.caption('Dataset rows are preserved as uploaded; they are not recalculated after refinement. Updated clustering measurements are in the CSV. Files reload in this app; compatibility with NanoSense re-import is not established.')

    navigation("bottom")

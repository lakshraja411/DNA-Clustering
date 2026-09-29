import hashlib,io,json,zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from analysis import load_events,load_dataset,link_dataset,describe,refine,fit_metrics,feature_table,cluster_features,cluster_profiles,cluster_dtw,FEATURE_DESCRIPTIONS,safe_settings
from workflow import active_events,signal_events,fitting_bytes,match_raw,bundle
from plots import trace_figure,distribution_figures,profile_figure,LABELS,scientific,figure_archive

st.set_page_config(page_title='DNA Event Lab',page_icon='🧬',layout='wide')
st.title('DNA Event Lab')
st.caption('Load → inspect → refine → plot → cluster → save')
st.sidebar.title('Your analysis')
step=st.sidebar.radio('Step',['1 · Load files','2 · Inspect events','3 · Refine and save fits','4 · Current–duration plots','5 · Cluster events','6 · Save clusters'])
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

def next_step(label):st.info('Next: select '+label+' in the left-hand menu.')

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
meta={'version':'0.3.0','source_hash':p['fingerprint'],'files':p['files'],'matching':p['matching'],'settings':safe_settings(p['settings']),
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
    st.write('Use the selected fits from step 3, or compare with measured traces. Cluster numbers are not DNA topology labels.')
    cfg=S.get('cluster_config',{})
    source=st.radio('Signal used for clustering',['Selected fits','Measured trace'],index=cfg.get('source',0),horizontal=True)
    algorithm=st.selectbox('Method',['PCA + agglomerative (DNA concept)','PCA + k-means (nanorod concept)','Waveform k-means','Time-series k-means (DTW)'],index=cfg.get('algorithm',0))
    k=st.slider('Number of clusters',2,10,cfg.get('k',4))
    defaults=['duration_ms','mean_blockade_nA','peak_blockade_nA','blockade_std_nA','peak_position','early_late_difference_nA']
    with st.expander('Advanced clustering settings'):
        fields=st.multiselect('Measured properties for PCA',list(FEATURE_DESCRIPTIONS),default=cfg.get('fields',defaults))
        npc=st.slider('PCA components',1,6,cfg.get('npc',2));bins=st.select_slider('Waveform positions',[16,32,64],value=cfg.get('bins',32));radius=st.slider('DTW alignment radius',1,16,cfg.get('radius',4))
        st.caption('PCA methods compare scaled properties. Waveform methods compare duration-normalised shapes while retaining blockade depth. These are paper-inspired adaptations, not the complete NanoBoost pipeline.')
        st.json(FEATURE_DESCRIPTIONS)
    S.cluster_config=dict(source=['Selected fits','Measured trace'].index(source),algorithm=['PCA + agglomerative (DNA concept)','PCA + k-means (nanorod concept)','Waveform k-means','Time-series k-means (DTW)'].index(algorithm),k=k,fields=fields,npc=npc,bins=bins,radius=radius)
    signature=(p['fingerprint'],refhash,source,algorithm,k,tuple(fields),npc,bins,radius)
    if st.button('Run clustering',type='primary'):
        try:
            with st.spinner('Grouping events… DTW can take longer on its first run.'):
                sig=signal_events(active,source);feat=feature_table(sig);_,prof=describe(sig,bins)
                if algorithm.startswith('PCA'):
                    if len(fields)<2:raise ValueError('Select at least two properties.')
                    info=cluster_features(feat[fields].to_numpy(),prof,k,npc,algorithm)
                elif 'DTW' in algorithm:info=cluster_dtw(prof,k,False,radius)
                else:info=cluster_profiles(prof,k,False)
                table=measured.copy();table['cluster']=info['labels']
                for col in feat.columns:
                    if col!='event_index':table['clustering_'+col]=feat[col].to_numpy()
                S.group={'signature':signature,'table':table,'info':info,'meta':{**meta,'clustering':{'source':source,'method':algorithm,'k':k,'features':fields if algorithm.startswith('PCA') else [],'pca_components':npc,'positions':bins,'dtw_radius':radius}}}
                S.pop('prepared',None)
        except Exception as ex:st.error(str(ex))
    g=S.get('group')
    if g and g['signature']!=signature:
        S.pop('group',None);S.pop('prepared',None);st.info('Settings changed. Run clustering again to update the results.');st.stop()
    if g:
        info=g['info'];table=g['table'];st.success(f'{len(table)} events grouped into {len(np.unique(info["labels"]))} clusters using {source.lower()}.')
        st.dataframe(table.groupby('cluster').size().rename('Events').to_frame())
        scatter,_,dropped,*_=distribution_figures(table,'duration_ms','mean_blockade_nA',True,False,color='cluster');show(scatter,'clusters_physical')
        st.caption('This scatter uses measured duration and mean blockade, coloured by cluster assignment, even when clustering used fits.')
        show(profile_figure(info['profiles'],info['labels'],info['centers']),'cluster_profiles')
        st.caption('Profile bands show the 10th–90th percentile of members, not uncertainty in the mean. DTW centres are aligned representative profiles.')
        emb=pd.DataFrame(info['embedding'],columns=['PC1','PC2']);emb['Cluster']=info['labels'].astype(str);show(px.scatter(emb,x='PC1',y='PC2',color='Cluster'),'cluster_projection')
        with st.expander('Numerical diagnostics'):
            st.write('Silhouette:',info['silhouette']);st.caption('Geometric separation does not establish physical accuracy. Scores from different feature spaces cannot establish which method is better.')
        cluster=st.selectbox('View traces from cluster',sorted(table.cluster.unique()));ids=table.loc[table.cluster==cluster,'event_index'].tolist()
        eid=st.selectbox('Event in this cluster',ids);e=next(e for e in events if e.index==eid);show(trace_figure(e,refs.get(e.index)),'cluster_event')
        if st.button('Prepare cluster publication figures'):S.cluster_figures=(signature,figure_archive(table,'duration_ms','mean_blockade_nA',True,'cluster'))
        if S.get('cluster_figures') and S.cluster_figures[0]==signature:st.download_button('Save cluster PDF, SVG and PNG figures',S.cluster_figures[1],'cluster_figures.zip')
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
                z.writestr(f'cluster_{i}.zip',bundle(ev,p['rawmap'],refs,p['dataset'],p['mapping'],p['settings'],p['dsettings'],sub,{**g['meta'],'cluster':int(i)}))
        S.prepared=(key,outer.getvalue())
    if S.get('prepared') and S.prepared[0]==key:st.download_button('Save cluster files ZIP',S.prepared[1],'cluster_files.zip','application/zip')
    st.write('Each cluster contains its selected eventfitting file, matched eventdata and dataset files, event table and analysis settings. Uploaded fits are also retained inside the new fitting file.')
    st.caption('Dataset rows are preserved as uploaded; they are not recalculated after refinement. Updated clustering measurements are in the CSV. Files reload in this app; compatibility with NanoSense re-import is not established.')

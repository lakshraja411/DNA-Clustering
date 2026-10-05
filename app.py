import hashlib,io,json,zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
# Keep the application modules in lock-step. A stale helper file should give a clear
# Streamlit message instead of failing during a long `from ... import ...` statement.
import analysis as _analysis
import physical as _physical
import workflow as _workflow
import plots as _plots

_REQUIRED = {
    'analysis.py': (_analysis, ['load_events','load_dataset','link_dataset','describe','refine','fit_metrics','cluster_features','feature_space_diagnostics','cluster_count_diagnostics','safe_settings']),
    'physical.py': (_physical, ['level_features','PHYSICAL_DESCRIPTIONS']),
    'workflow.py': (_workflow, ['active_events','signal_events','fitting_bytes','match_raw','bundle']),
    'plots.py': (_plots, ['trace_figure','distribution_figures','profile_figure','LABELS','scientific','figure_archive','profile_archive','cluster_pca_figure','member_profile_figure_hart','representative_time_examples','time_example_figure','pca_scree_figure','k_diagnostics_figure','feature_correlation_figure','cluster_pca_3d_figure','dendrogram_figure','hart_style_archive','blockade_dwell_figure','population_fraction_figure','level_composition_figure','occupancy_figure','physical_feature_distributions_figure']),
}
_missing = {filename:[name for name in names if not hasattr(module,name)] for filename,(module,names) in _REQUIRED.items()}
_missing = {filename:names for filename,names in _missing.items() if names}
if _missing:
    st.error('DNA Event Lab file-version mismatch: one or more helper files are older than this app.py.')
    for filename,names in _missing.items():
        st.code(f'{filename}: missing ' + ', '.join(names))
    st.info('Replace app.py, analysis.py, physical.py, plots.py, workflow.py and requirements.txt together from the same release, then reboot the Streamlit app. Do not keep version-suffixed filenames in the repository; the deployed files must be named exactly app.py, analysis.py, physical.py, plots.py and workflow.py.')
    st.stop()

from analysis import load_events,load_dataset,link_dataset,describe,refine,fit_metrics,cluster_features,feature_space_diagnostics,cluster_count_diagnostics,safe_settings
from physical import level_features,PHYSICAL_DESCRIPTIONS
from workflow import active_events,signal_events,fitting_bytes,match_raw,bundle
from plots import trace_figure,distribution_figures,profile_figure,LABELS,scientific,figure_archive,profile_archive,cluster_pca_figure,member_profile_figure_hart,representative_time_examples,time_example_figure,pca_scree_figure,k_diagnostics_figure,feature_correlation_figure,cluster_pca_3d_figure,dendrogram_figure,hart_style_archive,blockade_dwell_figure,population_fraction_figure,level_composition_figure,occupancy_figure,physical_feature_distributions_figure

st.set_page_config(page_title='DNA Event Lab',page_icon='🧬',layout='wide')
st.title('DNA Event Lab · v0.6.3')
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
                for key in ['refs','group','prepared','fitfile','fit_errors','refinement_history','recording_confirmed','confirmation_widget','cluster_scan','hart_export','cluster_figures']:S.pop(key,None)
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
meta={'version':'0.6.2','source_hash':p['fingerprint'],'files':p['files'],'matching':p['matching'],'settings':safe_settings(p['settings']),
      'refinements':{str(i):{'method':r['method'],'parameters':r['parameters']} for i,r in refs.items()},
      'refinement_history':S.get('refinement_history',[]),'fit_hash':refhash}
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

    # Enrich the Step 2 table with dimensionless/relative fit-QC quantities.
    # These are diagnostics only; they are not automatic event-rejection criteria.
    event_view=measured.copy()
    with np.errstate(divide='ignore',invalid='ignore'):
        event_view['rmse_over_noise']=np.where(
            event_view['padding_std_nA']>0,
            event_view['waveform_rmse_nA']/event_view['padding_std_nA'],
            np.nan)
        event_view['relative_abs_bias_pct']=np.where(
            np.abs(event_view['mean_blockade_nA'])>1e-12,
            100*np.abs(event_view['fit_bias_nA'])/np.abs(event_view['mean_blockade_nA']),
            np.nan)
        event_view['relative_abs_peak_error_pct']=np.where(
            np.abs(event_view['peak_blockade_nA'])>1e-12,
            100*np.abs(event_view['peak_error_nA'])/np.abs(event_view['peak_blockade_nA']),
            np.nan)
        event_view['abs_area_error_pct']=np.abs(event_view['area_error_pct'])
    st.dataframe(event_view[event_view.event_index==e.index],hide_index=True)

    with st.expander('Fit diagnostics across the recording'):
        st.write('Use these plots to find unusual fit behaviour and relationships across the recording. They assess fit–signal agreement; they do not determine whether an event is physically valid, and no event is excluded by these diagnostics.')

        fit_source=st.radio(
            'Fit used for diagnostics',
            ['Uploaded fit','Selected fit (refined where available)'],
            horizontal=True,
            key='fit_diagnostic_source')

        diagnostic=measured.copy()
        metric_names=['waveform_rmse_nA','relative_rmse','fit_bias_nA','peak_error_nA','area_error_pct']
        if fit_source.startswith('Selected'):
            for name in metric_names:
                refined='refined_'+name
                if refined in diagnostic:
                    diagnostic[name]=diagnostic[refined].combine_first(diagnostic[name])

        with np.errstate(divide='ignore',invalid='ignore'):
            diagnostic['rmse_over_noise']=np.where(
                diagnostic['padding_std_nA']>0,
                diagnostic['waveform_rmse_nA']/diagnostic['padding_std_nA'],
                np.nan)
            diagnostic['relative_abs_bias_pct']=np.where(
                np.abs(diagnostic['mean_blockade_nA'])>1e-12,
                100*np.abs(diagnostic['fit_bias_nA'])/np.abs(diagnostic['mean_blockade_nA']),
                np.nan)
            diagnostic['relative_abs_peak_error_pct']=np.where(
                np.abs(diagnostic['peak_blockade_nA'])>1e-12,
                100*np.abs(diagnostic['peak_error_nA'])/np.abs(diagnostic['peak_blockade_nA']),
                np.nan)
            diagnostic['abs_area_error_pct']=np.abs(diagnostic['area_error_pct'])

        diagnostic_labels={
            **LABELS,
            'padding_std_nA':'Padding noise SD (nA)',
            'segments':'Saved segment count',
            'relative_rmse':'Relative RMSE (dimensionless)',
            'rmse_over_noise':'RMSE / padding noise (dimensionless)',
            'fit_bias_nA':'Fit bias (nA)',
            'relative_abs_bias_pct':'Absolute bias / mean blockade (%)',
            'peak_error_nA':'Fit peak − measured peak (nA)',
            'relative_abs_peak_error_pct':'Absolute peak error / measured peak (%)',
            'area_error_pct':'Fit area error (%)',
            'abs_area_error_pct':'Absolute fit area error (%)'
        }
        x_options=['duration_ms','mean_blockade_nA','peak_blockade_nA','ecd_nA_ms','padding_std_nA','segments']
        y_options=['waveform_rmse_nA','relative_rmse','rmse_over_noise','fit_bias_nA','relative_abs_bias_pct',
                   'peak_error_nA','relative_abs_peak_error_pct','area_error_pct','abs_area_error_pct']

        c1,c2,c3=st.columns(3)
        x_metric=c1.selectbox('X axis',x_options,index=0,format_func=lambda x:diagnostic_labels.get(x,x),key='fit_diag_x')
        y_metric=c2.selectbox('Y axis',y_options,index=0,format_func=lambda x:diagnostic_labels.get(x,x),key='fit_diag_y')
        colour_options=['None','segments','padding_std_nA','mean_blockade_nA']
        colour_metric=c3.selectbox(
            'Colour by',
            colour_options,
            format_func=lambda x:'None' if x=='None' else diagnostic_labels.get(x,x),
            key='fit_diag_colour')

        signed_y={'fit_bias_nA','peak_error_nA','area_error_pct'}
        c4,c5=st.columns(2)
        logx=c4.checkbox('Logarithmic X axis',False,key='fit_diag_logx')
        logy=c5.checkbox('Logarithmic Y axis',False,key='fit_diag_logy',disabled=y_metric in signed_y)
        if y_metric in signed_y and S.get('fit_diag_logy',False):
            S.fit_diag_logy=False
            logy=False

        good=np.isfinite(diagnostic[x_metric])&np.isfinite(diagnostic[y_metric])
        if logx:good&=diagnostic[x_metric]>0
        if logy:good&=diagnostic[y_metric]>0
        if colour_metric!='None':good&=np.isfinite(diagnostic[colour_metric])
        d=diagnostic.loc[good].copy()

        if len(d):
            hover_cols=[c for c in [
                'event_index','duration_ms','mean_blockade_nA','peak_blockade_nA',
                'padding_std_nA','waveform_rmse_nA','relative_rmse','rmse_over_noise',
                'fit_bias_nA','area_error_pct'
            ] if c in d and c not in [x_metric,y_metric,colour_metric]]
            fig=px.scatter(
                d,
                x=x_metric,
                y=y_metric,
                color=None if colour_metric=='None' else colour_metric,
                log_x=logx,
                log_y=logy,
                hover_data=hover_cols,
                labels=diagnostic_labels,
                opacity=.65,
                render_mode='svg')
            if y_metric in signed_y:
                fig.add_hline(y=0,line_dash='dot',line_width=1)
            show(fig,'fit_diagnostics')
            dropped=len(diagnostic)-len(d)
            st.caption(f'{len(d)} events plotted; {dropped} omitted because values are missing/non-finite or incompatible with the selected logarithmic axes. Data source for fit metrics: {fit_source}. Zero is shown as a reference for signed-error metrics.')
        else:
            st.warning('No finite values are compatible with the selected axes.')

        st.download_button(
            'Save fit diagnostics CSV',
            diagnostic.to_csv(index=False),
            'fit_diagnostics.csv',
            'text/csv')
        st.caption('Useful views include RMSE/noise versus duration, relative RMSE versus mean blockade, and RMSE versus padding noise. Treat unusual points as candidates for inspection rather than automatic rejection.')

    next_step('3 · Refine and save fits')
elif step.startswith('3'):
    st.header('3 · Refine and save fits')
    st.write('Try one event first. Review its fit before applying the method to all events. Uploaded files are never overwritten.')
    e=choose_event();method=st.selectbox('Refinement method',['Segment means','New levels (PELT)'])
    st.caption({'Segment means':'Keep the existing step boundaries and recalculate the level heights.','New levels (PELT)':'Find new step boundaries; the penalty controls how readily another step is added.'}[method])

    with st.expander('Advanced refinement settings',expanded=method=='New levels (PELT)'):
        minimum=st.number_input('Minimum step duration (µs)',5.,10000.,25.,step=5.,disabled=method!='New levels (PELT)')
        penalty=st.number_input('Extra-step penalty',.1,100.,8.,step=.5,disabled=method!='New levels (PELT)')

    scope=st.radio('Refine',['Selected event','All events'],horizontal=True)

    if st.button('Calculate refined fits',type='primary'):
        targets=events if scope=='All events' else [e]
        errors=[]
        history=S.setdefault('refinement_history',[])
        batch_id=max([int(h.get('batch_id',0)) for h in history],default=0)+1
        bar=st.progress(0.)

        for j,item in enumerate(targets):
            attempted_parameters={'minimum_step_duration_us':float(minimum),'extra_step_penalty':float(penalty)} if method=='New levels (PELT)' else {}
            try:
                new_ref=refine(item,method,minimum,penalty)
                refs[item.index]=new_ref
                history.append({
                    'batch_id':batch_id,
                    'event_id':int(item.index),
                    'scope':scope,
                    'method':method,
                    'status':'success',
                    'parameters':new_ref.get('parameters',attempted_parameters),
                    'reason':''
                })
            except Exception as ex:
                reason=str(ex)
                errors.append({'event_id':item.index,'reason':reason})
                history.append({
                    'batch_id':batch_id,
                    'event_id':int(item.index),
                    'scope':scope,
                    'method':method,
                    'status':'failed',
                    'parameters':attempted_parameters,
                    'reason':reason
                })
            bar.progress((j+1)/len(targets))

        S.refs=refs
        S.fit_errors=errors
        S.refinement_history=history
        for key in ['group','prepared','fitfile','cluster_scan','hart_export','cluster_figures']:S.pop(key,None)
        st.rerun()

    show(trace_figure(e,refs.get(e.index)),'refinement')

    if e.index in refs:
        st.dataframe(pd.DataFrame(
            [fit_metrics(e,e.fit),fit_metrics(e,refs[e.index]['fit'])],
            index=['Uploaded fit','Refined fit']
        ))

    st.caption('A lower RMSE is not proof of a more accurate physical model. Baseline and event boundaries remain fixed.')

    if S.get('fit_errors'):
        st.warning('Some refinements failed; their previous fits remain selected.')
        st.dataframe(pd.DataFrame(S.fit_errors),hide_index=True)

    with st.expander('Refinement audit and history',expanded=bool(refs or S.get('refinement_history'))):
        st.write('The current audit lists the refinement presently selected for each event. The attempt history also records failed and superseded refinement attempts during this analysis session.')

        current_rows=[]
        event_lookup={ev.index:ev for ev in events}
        for event_id,ref in sorted(refs.items()):
            ev=event_lookup[event_id]
            old_metrics=fit_metrics(ev,ev.fit)
            new_metrics=fit_metrics(ev,ref['fit'])

            old_rmse=old_metrics.get('waveform_rmse_nA',np.nan)
            new_rmse=new_metrics.get('waveform_rmse_nA',np.nan)
            rmse_change=new_rmse-old_rmse if np.isfinite(old_rmse) and np.isfinite(new_rmse) else np.nan
            rmse_change_pct=(100*(new_rmse-old_rmse)/old_rmse
                             if np.isfinite(old_rmse) and np.isfinite(new_rmse) and abs(old_rmse)>1e-12
                             else np.nan)

            current_rows.append({
                'event_id':int(event_id),
                'method':ref.get('method',''),
                'parameters':json.dumps(ref.get('parameters',{}),sort_keys=True),
                'uploaded_rmse_nA':old_rmse,
                'refined_rmse_nA':new_rmse,
                'rmse_change_nA':rmse_change,
                'rmse_change_pct':rmse_change_pct,
                'uploaded_relative_rmse':old_metrics.get('relative_rmse',np.nan),
                'refined_relative_rmse':new_metrics.get('relative_rmse',np.nan),
                'uploaded_bias_nA':old_metrics.get('fit_bias_nA',np.nan),
                'refined_bias_nA':new_metrics.get('fit_bias_nA',np.nan),
                'uploaded_peak_error_nA':old_metrics.get('peak_error_nA',np.nan),
                'refined_peak_error_nA':new_metrics.get('peak_error_nA',np.nan),
                'uploaded_area_error_pct':old_metrics.get('area_error_pct',np.nan),
                'refined_area_error_pct':new_metrics.get('area_error_pct',np.nan),
                'refined_segments':len(ref.get('levels',[]))
            })

        current_audit=pd.DataFrame(current_rows)
        if len(current_audit):
            st.subheader('Currently selected refinements')
            st.caption(f'{len(current_audit)} events currently use refined fits; all other events retain their uploaded fits.')
            display_cols=[
                'event_id','method','parameters',
                'uploaded_rmse_nA','refined_rmse_nA','rmse_change_pct',
                'uploaded_relative_rmse','refined_relative_rmse',
                'uploaded_bias_nA','refined_bias_nA',
                'uploaded_area_error_pct','refined_area_error_pct',
                'refined_segments'
            ]
            st.dataframe(current_audit[display_cols].round(4),hide_index=True)
            st.download_button(
                'Save current refinement audit CSV',
                current_audit.to_csv(index=False),
                'refinement_audit.csv',
                'text/csv'
            )
        else:
            st.info('No refined fits are currently selected.')

        history=S.get('refinement_history',[])
        if history:
            history_rows=[]
            for h in history:
                history_rows.append({
                    'batch_id':h.get('batch_id'),
                    'event_id':h.get('event_id'),
                    'scope':h.get('scope'),
                    'method':h.get('method'),
                    'status':h.get('status'),
                    'parameters':json.dumps(h.get('parameters',{}),sort_keys=True),
                    'reason':h.get('reason','')
                })
            history_df=pd.DataFrame(history_rows)
            st.subheader('Refinement attempt history')
            st.caption('A later successful refinement of the same event replaces the active fit, but earlier attempts remain listed here for provenance during this session.')
            st.dataframe(history_df,hide_index=True)
            st.download_button(
                'Save refinement attempt history CSV',
                history_df.to_csv(index=False),
                'refinement_history.csv',
                'text/csv'
            )

    if st.button('Discard all refinements',disabled=not refs):
        S.refs={}
        for key in ['group','prepared','fitfile','cluster_scan','hart_export','cluster_figures']:S.pop(key,None)
        st.rerun()

    if st.button('Prepare new eventfitting file',disabled=not refs):
        S.fitfile=(refhash,fitting_bytes(events,refs,p['settings'],meta))

    if S.get('fitfile') and S.fitfile[0]==refhash:
        st.download_button(
            'Save refined.eventfitting.npz',
            S.fitfile[1],
            'refined.eventfitting.npz',
            'application/octet-stream'
        )

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
    bins=st.slider('Heatmap bins per axis',10,100,45,5,
                   help='Controls the visual resolution of the 2D event-count histogram only. It does not alter the underlying event measurements.')
    scatter,heat,dropped,*_=distribution_figures(table,'duration_ms',height,logx,False,bins=bins)
    show(scatter,'current_duration');show(heat,'event_counts')
    st.caption(f'{len(table)-dropped} events plotted; {dropped} omitted because values are incompatible with the axes. Heatmap resolution: {bins} × {bins} bins. Colour is the number of events per bin. Data source: {source}.')
    st.download_button('Save plotted measurements CSV',table.to_csv(index=False),'current_duration.csv')
    if st.button('Prepare publication figures'):S.figures=(p['fingerprint'],refhash,source,height,logx,bins,figure_archive(table,'duration_ms',height,logx,bins=bins))
    if S.get('figures') and S.figures[:6]==(p['fingerprint'],refhash,source,height,logx,bins):st.download_button('Save PDF, SVG and 600 dpi PNG figures',S.figures[6],'current_duration_figures.zip','application/zip')
    next_step('5 · Cluster events')
elif step.startswith('5'):
    st.header('5 · Cluster the DNA events')
    st.write('This version deliberately keeps the clustering model small and physics-led. Five complementary continuous descriptors form the PCA space; richer resolved-level quantities are calculated separately to explain the clusters afterwards. This avoids asking PCA to encode every piece of DNA physics at once.')
    cfg=S.get('cluster_config',{})

    source_options=['Selected fits','Measured trace']
    source=st.radio('Signal used for resolved-level amplitudes',source_options,index=source_options.index(cfg.get('source','Selected fits')) if cfg.get('source','Selected fits') in source_options else 0,horizontal=True)
    methods=['PCA + agglomerative (Ward)','PCA + k-means']
    previous=cfg.get('algorithm',methods[0]);algorithm=st.selectbox('Clustering algorithm',methods,index=methods.index(previous) if previous in methods else 0)
    st.caption('Both choices use exactly the same DNA features, robust median/IQR scaling and PCA coordinates. Only the final grouping rule changes.')

    previous_params=cfg.get('physical_params',{})
    with st.expander('1 · Resolve the DNA plateau structure',expanded=True):
        c1,c2,c3=st.columns(3)
        min_us=c1.number_input('Minimum plateau duration (µs)',min_value=0.,value=float(previous_params.get('min_duration_us',25.)))
        min_height=c2.number_input('Minimum level difference (nA)',min_value=0.,value=float(previous_params.get('min_height_nA',.1)),format='%.3f')
        noise_mult=c3.number_input('Noise multiplier for level merging',min_value=0.,value=float(previous_params.get('noise_multiplier',3.)))
        st.caption('Adjacent fitted levels merge when their difference is ≤ max(minimum level difference, noise multiplier × robust noise scale). A retained plateau must also last at least max(the selected minimum duration, 3 sampling intervals).')
        omit_edges=st.checkbox('Omit short boundary plateaus from physical features',value=previous_params.get('omit_short_boundaries',True))
        st.caption('Only the first and last short merged plateaus may be omitted. Short internal plateaus keep the event out of the physical feature model rather than silently changing its structure.')
        use_ref=st.checkbox('Use a calibrated single-file blockade reference for interpretation',value=previous_params.get('reference_nA') is not None)
        reference=st.number_input('Single-file reference blockade ΔI₀ (nA)',min_value=.000001,value=float(previous_params.get('reference_nA') or 1.),format='%.4f') if use_ref else None
        use_deep=st.checkbox('Use a deeper-blockade threshold for interpretation',value=previous_params.get('deep_threshold_nA') is not None)
        threshold=st.number_input('Deeper-blockade threshold (nA)',min_value=.000001,value=float(previous_params.get('deep_threshold_nA') or 1.5),format='%.4f') if use_deep else None
        if use_ref:st.caption('The reference creates single-file-equivalent blockade ratios for interpretation. It does not by itself prove strand number or topology.')
        if use_deep:st.caption('The threshold creates a deep-state occupancy descriptor. Events close to the resolution threshold are flagged as threshold-sensitive.')
    physical_params=dict(min_duration_us=min_us,min_height_nA=min_height,noise_multiplier=noise_mult,reference_nA=reference,deep_threshold_nA=threshold,omit_short_boundaries=omit_edges)

    physical_audit,physical_sequences=level_features(active,source,**physical_params)
    usable=int(physical_audit.physical_eligible.sum())
    st.info(f'{usable} of {len(active)} events have resolved plateau structure under the current settings.')
    with st.expander('Physical-feature eligibility and level audit'):
        st.dataframe(physical_audit,hide_index=True)
        st.download_button('Save physical feature audit CSV',physical_audit.to_csv(index=False),'physical_feature_audit.csv','text/csv')
    flagged_ids=physical_audit.loc[(~physical_audit.physical_eligible) | physical_audit.get('boundary_omission_applied',False).fillna(False),'event_index'].tolist()
    if flagged_ids:
        with st.expander('Inspect unresolved or boundary-adjusted events'):
            audit_id=st.selectbox('Flagged event ID',flagged_ids)
            audit_event=next(e for e in events if e.index==audit_id);audit_fig=trace_figure(audit_event,refs.get(audit_id))
            audit_levels=physical_sequences.loc[physical_sequences.event_index==audit_id]
            for level in audit_levels.itertuples():
                if level.level_status!='resolved':audit_fig.add_vrect(x0=level.start_from_event_ms,x1=level.start_from_event_ms+level.duration_ms,fillcolor='#e89b35',opacity=.25,line_width=0)
            show(audit_fig,'physical_flagged_trace');st.dataframe(audit_levels,hide_index=True)
            st.caption('Amber regions failed the declared resolution policy. They remain in the audit and are not automatically called artefacts.')

    # Primary clustering coordinates: deliberately small, continuous and complementary.
    core_fields=['log10_duration_ms','resolved_weighted_mean_nA','deepest_plateau_nA','resolved_weighted_std_nA','blockade_temporal_centroid']
    with st.expander('2 · Primary DNA clustering feature set',expanded=True):
        st.markdown('**These five features form the PCA/clustering space:**')
        for f in core_fields:st.write('• '+PHYSICAL_DESCRIPTIONS[f])
        st.caption('Why these five? They represent kinetics, typical sustained blockade, maximum sustained occupancy, multilevel heterogeneity and temporal asymmetry. ECD, level count, occupancy and transition descriptors are retained for interpretation rather than being allowed to repeatedly weight the PCA.')
        st.caption('Dwell time enters PCA as log₁₀(duration/ms). Amplitude quantities remain in nA. All retained features are then scaled by median and interquartile range (RobustScaler), not mean and standard deviation.')

    positions=st.select_slider('Waveform positions used only for event-family profile plots',[64,128,256],value=cfg.get('positions',128) if cfg.get('positions',128) in [64,128,256] else 128)
    positions_idx=np.flatnonzero(physical_audit.physical_eligible.to_numpy())
    eligible_active=[active[int(i)] for i in positions_idx]
    feat=physical_audit.iloc[positions_idx].reset_index(drop=True)
    table_preview=measured.iloc[positions_idx].reset_index(drop=True)
    if len(eligible_active)<3:st.error('At least three resolved events are required for clustering.');st.stop()
    matrix=feat[core_fields].to_numpy(float)
    try:space_diag=feature_space_diagnostics(matrix,core_fields,.98)
    except Exception as ex:st.error(str(ex));st.stop()

    with st.expander('3 · Feature QC and PCA dimensionality',expanded=True):
        show(feature_correlation_figure(space_diag['correlation'],[f.replace('_',' ') for f in core_fields]),'cluster_feature_correlation')
        if space_diag['dropped_constant']:st.warning('Constant features removed: '+', '.join(space_diag['dropped_constant']))
        if space_diag['dropped_correlated']:st.info('Near-duplicate features removed before PCA: '+', '.join(space_diag['dropped_correlated']))
        st.write('Retained for PCA:',', '.join(space_diag['retained_features']))
        show(pca_scree_figure(space_diag['scree']),'cluster_scree_pre')
        scree_table=pd.DataFrame({'PC':np.arange(1,len(space_diag['scree'])+1),'Explained variance':space_diag['scree'],'Cumulative variance':space_diag['cumulative']})
        st.dataframe(scree_table.round(4),hide_index=True)
        st.caption('The scree plot tells you how many PCA directions are needed to represent these five physical descriptors. Explained variance is not clustering accuracy.')

    max_pc=max(1,min(5,int(space_diag['max_components'])))
    if max_pc>=2:
        npc=st.slider('Principal components used for clustering',2,max_pc,min(max(2,int(cfg.get('npc',2))),max_pc))
    else:
        npc=1;st.metric('Principal components used for clustering',1)
    kmax=st.slider('Largest k to include in elbow–silhouette scan',3,10,int(cfg.get('kmax',8)))

    diag_signature=(p['fingerprint'],refhash,source,algorithm,tuple(core_fields),npc,kmax,positions,json.dumps(physical_params,sort_keys=True))
    if st.button('Run elbow + silhouette scan'):
        try:
            with st.spinner('Testing candidate cluster counts…'):
                base_method='PCA + agglomerative' if 'agglomerative' in algorithm.lower() else 'PCA + k-means'
                S.cluster_scan=(diag_signature,cluster_count_diagnostics(matrix,base_method,npc,2,kmax,core_fields,.98))
        except Exception as ex:st.error(str(ex))
    scan=S.get('cluster_scan')
    if scan and scan[0]!=diag_signature:
        S.pop('cluster_scan',None);scan=None
    if scan:
        scan=scan[1]
        with st.expander('4 · Choose k from elbow + silhouette',expanded=True):
            show(k_diagnostics_figure(scan['table'],scan['elbow_k'],scan['silhouette_k']),'cluster_k_diagnostics_pre')
            st.dataframe(pd.DataFrame(scan['table']).round(4),hide_index=True)
            if scan['agreement']:st.success(f'Elbow and silhouette agree on k = {scan["suggested_k"]}.')
            else:st.warning(f'Elbow suggests k = {scan["elbow_k"]}, while silhouette peaks at k = {scan["silhouette_k"]}. Inspect the physical plots and event-family traces before deciding.')
            st.caption('The scan is a diagnostic, not an automatic statement about the number of DNA conformations.')
    else:
        st.info('Run the elbow + silhouette scan before finalising k. If you skip it, the app will calculate the scan when clustering is run.')

    suggested=int(scan['suggested_k']) if scan else int(cfg.get('k',3));suggested=max(2,min(suggested,kmax))
    previous_k=int(cfg.get('k',suggested));previous_k=max(2,min(previous_k,kmax))
    k=st.slider('Number of clusters used for the final grouping',2,kmax,previous_k)
    S.cluster_config=dict(source=source,algorithm=algorithm,k=k,fields=core_fields,npc=npc,positions=positions,kmax=kmax,physical_params=physical_params)
    signature=(p['fingerprint'],refhash,source,algorithm,k,tuple(core_fields),npc,positions,kmax,json.dumps(physical_params,sort_keys=True))

    if st.button('Run clustering',type='primary'):
        try:
            with st.spinner('Clustering DNA events and preparing physical interpretation…'):
                current_scan=scan
                base_method='PCA + agglomerative' if 'agglomerative' in algorithm.lower() else 'PCA + k-means'
                if current_scan is None:
                    current_scan=cluster_count_diagnostics(matrix,base_method,npc,2,kmax,core_fields,.98)
                    S.cluster_scan=(diag_signature,current_scan)
                excluded=physical_audit.loc[~physical_audit.physical_eligible].copy();sequences=physical_sequences.copy()
                sig=signal_events(eligible_active,source);_,prof=describe(sig,positions)
                info=cluster_features(matrix,prof,k,npc,base_method,.98)
                table=table_preview.copy();table['cluster']=info['labels']
                for col in feat.columns:
                    if col!='event_index':table['clustering_'+col]=feat[col].to_numpy()
                retained=[f for f,keep in zip(core_fields,info.get('feature_keep_mask',[True]*len(core_fields))) if keep]
                clustering_meta={'source':source,'method':algorithm,'k':int(k),'selection':'manual after elbow/silhouette diagnostics',
                    'features':core_fields,'pca_components':int(info.get('n_components',npc)),'profile_positions':positions,'k_scan_range':[2,kmax],
                    'feature_set':'five-feature DNA physical core','scaling':'RobustScaler median/IQR','physical_settings':physical_params,
                    'excluded_events':len(excluded),'retained_features':retained,'feature_center':info.get('feature_center'),
                    'feature_scale':info.get('feature_scale'),'pca_loadings':info.get('loadings'),'k_diagnostics':current_scan,
                    'correlation_threshold':.98}
                S.group={'signature':signature,'table':table,'info':info,'excluded':excluded,'sequences':sequences,'scan':current_scan,'meta':{**meta,'clustering':clustering_meta}}
                S.pop('prepared',None);S.pop('hart_export',None)
        except Exception as ex:st.error(str(ex))

    g=S.get('group')
    if g and g['signature']!=signature:
        S.pop('group',None);S.pop('prepared',None);S.pop('hart_export',None)
        st.info('Clustering settings changed. Run clustering again to update the results.');st.stop()

    if g:
        info=g['info'];table=g['table'];scan=g.get('scan');selected_k=len(np.unique(info['labels']))
        st.success(f'{len(table)} resolved events grouped into {selected_k} signal families using {algorithm}.')
        excluded=g.get('excluded',pd.DataFrame());sequences=g.get('sequences',pd.DataFrame())
        if len(excluded):
            st.warning(f'{len(excluded)} of {len(events)} events were excluded because their plateau structure was unresolved under the declared settings.')
            with st.expander('Excluded events and reasons'):
                st.dataframe(excluded,hide_index=True);st.download_button('Save exclusion audit CSV',excluded.to_csv(index=False),'physical_exclusions.csv','text/csv')
        if len(sequences):
            with st.expander('Resolved plateau measurements'):
                st.dataframe(sequences,hide_index=True);st.download_button('Save resolved levels CSV',sequences.to_csv(index=False),'resolved_levels.csv','text/csv')

        st.subheader('A · Statistical clustering result')
        embedding=np.asarray(info['embedding']);variance=info.get('pca_variance',[])
        projection=pd.DataFrame(embedding,columns=['PC1','PC2']);projection['Cluster']=table['cluster'].astype(str).to_numpy()
        projection['Event ID']=table['event_index'].to_numpy();projection['Duration (ms)']=table['duration_ms'].to_numpy()
        projection['Measured mean blockade (nA)']=table['mean_blockade_nA'].to_numpy()
        axis_labels={f'PC{j+1}':f'PC{j+1} ({100*variance[j]:.1f}% variance)' if j<len(variance) else f'PC{j+1}' for j in range(2)}
        show(cluster_pca_figure(projection,axis_labels,False),'cluster_projection')
        st.caption(f'No convex-hull shading is used. Each point is one event; the black × is the median position of that group in this displayed plane. Final clustering used {info.get("n_components",npc)} PCA component(s), so PC1–PC2 is only a projection.')
        st.download_button('Save PC coordinates CSV',projection.to_csv(index=False),'pca_coordinates.csv','text/csv')

        st.subheader('B · Physical interpretation of the signal families')
        show(blockade_dwell_figure(table),'cluster_blockade_dwell')
        st.caption('This is the direct DNA-physics view: actual dwell time versus duration-weighted sustained blockade. The x-axis is logarithmic only for display.')

        show(physical_feature_distributions_figure(table),'cluster_physical_distributions')
        st.caption('These four distributions are shown in their physical units. ECD and several other quantities are interpretation/QC variables; they did not all create the clusters.')

        p1,p2,p3=st.columns(3)
        with p1:
            show(level_composition_figure(table),'cluster_level_composition')
            st.caption('Level count is supporting evidence, not a PCA coordinate: the plot shows what fraction of each cluster contains 1, 2 or ≥3 resolved sustained levels.')
        with p2:
            show(occupancy_figure(table),'cluster_deepest_occupancy')
            st.caption('Deepest-state occupancy asks how much of the analysed event duration is spent in its deepest resolved state.')
        with p3:
            show(population_fraction_figure(table),'cluster_population')
            st.caption('Cluster population is the fraction of all physically eligible events assigned to each signal family.')

        interpretation=[
            ('duration_ms','Dwell (ms)'),
            ('resolved_weighted_mean_nA','Weighted blockade (nA)'),
            ('deepest_plateau_nA','Deepest blockade (nA)'),
            ('ecd_nA_ms','ECD (nA·ms)'),
            ('resolved_levels','Resolved levels'),
            ('resolved_blockade_range_nA','Level range (nA)'),
            ('deepest_plateau_fraction','Deepest-state fraction'),
            ('deepest_plateau_position','Deepest-state position'),
            ('blockade_temporal_centroid','Temporal centroid'),
            ('transition_direction','Transition direction'),
        ]
        if use_deep:interpretation.append(('deep_time_fraction','Deep-threshold fraction'))
        rows=[]
        for cid,sub in table.groupby('cluster',sort=True):
            row={'Cluster':int(cid),'Events':len(sub),'Population (%)':100*len(sub)/len(table)}
            for f,label in interpretation:
                col='clustering_'+f
                if col in sub:row[label]=float(sub[col].median())
            rows.append(row)
        hart_summary=pd.DataFrame(rows)
        st.dataframe(hart_summary.round(4),hide_index=True)
        st.caption('These are medians in physical units. They are used to explain the signal families after clustering rather than to force every descriptor into the PCA.')
        st.download_button('Save physical cluster summary CSV',hart_summary.to_csv(index=False),'cluster_physical_summary.csv','text/csv')

        st.subheader('C · What do the event families actually look like?')
        columns=st.columns(3)
        for group_id in range(selected_k):
            with columns[group_id%3]:
                show(member_profile_figure_hart(info['profiles'],info['labels'],info['centers'],group_id),f'dna_members_{group_id}')
        st.caption('Grey curves are a deterministic sample of real cluster-member profiles. The red curve is the pointwise median across every member of that cluster. No percentile envelope is drawn, and blockade amplitude is not normalised away.')

        st.write('**Median representative profiles overlaid**')
        show(profile_figure(info['profiles'],info['labels'],info['centers']),'cluster_profiles')

        time_examples=representative_time_examples([e for e in active if e.index in set(table.event_index)],info,g['meta']['clustering']['source'])
        with st.expander('Representative real events in actual time (ms)',expanded=True):
            show(time_example_figure(time_examples),'cluster_actual_time')
            st.download_button('Save actual-time example curves',time_examples.to_csv(index=False),'actual_time_examples.csv','text/csv')
            st.caption('One genuine recorded event per group is chosen as the event nearest the median representative profile. Its original time axis and detected duration are retained.')

        st.subheader('D · Supplementary clustering diagnostics')
        c1,c2=st.columns(2)
        with c1:
            show(feature_correlation_figure(space_diag['correlation'],[f.replace('_',' ') for f in core_fields]),'cluster_feature_correlation_result')
        with c2:
            show(pca_scree_figure(info['scree']),'cluster_scree_result')
        if scan:
            show(k_diagnostics_figure(scan['table'],scan['elbow_k'],scan['silhouette_k']),'cluster_k_diagnostics_result')

        if info.get('embedding3') is not None:
            with st.expander('3-PC view'):
                show(cluster_pca_3d_figure(info['embedding3'],info['labels'],info.get('pca_variance',[]),table.event_index.to_numpy()),'cluster_pca_3d')
                st.caption('The third component is an exploratory check of structure hidden from the PC1–PC2 projection.')

        if info.get('linkage_matrix') is not None:
            with st.expander('Ward dendrogram',expanded=False):
                show(dendrogram_figure(info['linkage_matrix'],selected_k,50),'cluster_dendrogram')
                st.caption('The dendrogram shows the hierarchical Ward merge structure and is truncated so a large recording remains readable.')

        kept=[f for f,keep in zip(core_fields,info.get('feature_keep_mask',[True]*len(core_fields))) if keep]
        loads=np.asarray(info.get('loadings',[]));pcs=min(3,len(loads));load_rows=[]
        for j in range(pcs):
            var=100*info['scree'][j] if j<len(info.get('scree',[])) else np.nan
            for name,value in zip(kept,loads[j]):load_rows.append({'Feature':name.replace('_',' ').title(),'Loading':value,'PC':f'PC{j+1} ({var:.1f}%)'})
        if load_rows:
            with st.expander('PCA loadings: what constructs each PCA direction?'):
                show(px.bar(pd.DataFrame(load_rows),x='Loading',y='Feature',color='PC',barmode='group',orientation='h'),'pca_loadings')
                st.caption('Loadings describe the construction of PCA coordinates from the robust-scaled core features. The overall sign of a principal component can flip without changing its meaning.')

        with st.expander('Numerical diagnostics'):
            st.write('Silhouette:',info.get('silhouette'));st.write('Calinski–Harabasz:',info.get('calinski_harabasz'));st.write('Davies–Bouldin:',info.get('davies_bouldin'))
            if np.isfinite(info.get('ari',np.nan)):st.write('K-means repeat-seed ARI:',info.get('ari'))
            st.write('Scaling: median / interquartile range (RobustScaler)')
            st.write('PCA components used:',info.get('n_components'));st.write('Cumulative variance in used PCs:',float(np.sum(info.get('scree',[])[:info.get('n_components',0)])))
            if scan:st.write('Elbow suggestion:',scan['elbow_k']);st.write('Silhouette peak:',scan['silhouette_k'])
            st.caption('These diagnose signal geometry only. Physical/topological interpretation still comes from the waveform and resolved-level behaviour.')

        cluster=st.selectbox('Inspect every event in cluster',sorted(table.cluster.unique()));ids=table.loc[table.cluster==cluster,'event_index'].tolist();eid=st.selectbox('Event in this cluster',ids)
        e=next(e for e in events if e.index==eid);show(trace_figure(e,refs.get(e.index)),'cluster_event')

        if st.button('Prepare representative profile figures'):
            S.cluster_figures=(signature,profile_archive(info['profiles'],info['labels'],info['centers']))
        if S.get('cluster_figures') and S.cluster_figures[0]==signature:
            st.download_button('Save representative profile PDF, SVG and PNG figures',S.cluster_figures[1],'cluster_profiles.zip')

        if st.button('Prepare main + supplementary DNA clustering figure pack'):
            S.hart_export=(signature,hart_style_archive(info,table['event_index'].to_numpy(),table,hart_summary,scan,source,algorithm))
        if S.get('hart_export') and S.hart_export[0]==signature:
            st.download_button('Save DNA clustering figure pack',S.hart_export[1],'dna_clustering_figures.zip','application/zip')
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

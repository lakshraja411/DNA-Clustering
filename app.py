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
    'analysis.py': (_analysis, ['load_events','load_dataset','link_dataset','describe','refine','fit_metrics','cluster_features','feature_space_diagnostics','cluster_count_diagnostics','safe_settings','DATASET_FEATURE_NAMES','DATASET_FEATURE_DESCRIPTIONS','dataset_feature_name','dataset_feature_label','aligned_event_profiles','cluster_median_profiles']),
    'physical.py': (_physical, ['level_features','PHYSICAL_DESCRIPTIONS']),
    'workflow.py': (_workflow, ['active_events','signal_events','fitting_bytes','match_raw','bundle','comparison_package']),
    'plots.py': (_plots, ['trace_figure','distribution_figures','profile_figure','LABELS','scientific','figure_archive','profile_archive','cluster_pca_figure','member_profile_figure_hart','representative_time_examples','time_example_figure','pca_scree_figure','k_diagnostics_figure','feature_correlation_figure','cluster_pca_3d_figure','dendrogram_figure','hart_style_archive','blockade_dwell_figure','population_fraction_figure','level_composition_figure','occupancy_figure','physical_feature_distributions_figure','fold_state_figure','comparison_condition_preview_figure','cross_salt_family_profiles_figure','cross_salt_population_figure','cross_salt_metric_figure','cross_salt_heatmap_figure','cross_salt_comparison_archive']),
}
_missing = {filename:[name for name in names if not hasattr(module,name)] for filename,(module,names) in _REQUIRED.items()}
_missing = {filename:names for filename,names in _missing.items() if names}
_EXPECTED_RELEASE='0.9.1'
_version_mismatch={filename:getattr(module,'RELEASE_VERSION',None) for filename,(module,_) in _REQUIRED.items() if getattr(module,'RELEASE_VERSION',None)!=_EXPECTED_RELEASE}
if _missing or _version_mismatch:
    st.error('DNA Event Lab file-version mismatch: helper files are not all from release '+_EXPECTED_RELEASE+'.')
    for filename,names in _missing.items():
        st.code(f'{filename}: missing ' + ', '.join(names))
    for filename,version in _version_mismatch.items():
        st.code(f'{filename}: release {version!r}, expected {_EXPECTED_RELEASE}')
    st.info('Replace app.py, analysis.py, physical.py, plots.py, workflow.py and requirements.txt together from the same release, then reboot the Streamlit app. Do not keep version-suffixed filenames in the repository; the deployed files must be named exactly app.py, analysis.py, physical.py, plots.py and workflow.py.')
    st.stop()

from analysis import load_events,load_dataset,link_dataset,describe,refine,fit_metrics,cluster_features,feature_space_diagnostics,cluster_count_diagnostics,safe_settings,DATASET_FEATURE_NAMES,DATASET_FEATURE_DESCRIPTIONS,dataset_feature_name,dataset_feature_label,aligned_event_profiles,cluster_median_profiles
from physical import level_features,PHYSICAL_DESCRIPTIONS
from workflow import active_events,signal_events,fitting_bytes,match_raw,bundle,comparison_package
from plots import trace_figure,distribution_figures,profile_figure,LABELS,scientific,figure_archive,profile_archive,cluster_pca_figure,member_profile_figure_hart,representative_time_examples,time_example_figure,pca_scree_figure,k_diagnostics_figure,feature_correlation_figure,cluster_pca_3d_figure,dendrogram_figure,hart_style_archive,blockade_dwell_figure,population_fraction_figure,level_composition_figure,occupancy_figure,physical_feature_distributions_figure,fold_state_figure,comparison_condition_preview_figure,cross_salt_family_profiles_figure,cross_salt_population_figure,cross_salt_metric_figure,cross_salt_heatmap_figure,cross_salt_comparison_archive

st.set_page_config(page_title='DNA Event Lab',page_icon='🧬',layout='wide')
st.title('DNA Event Lab · v0.9.1')
st.caption('Load → inspect → refine → plot → cluster → save → compare salts')
st.sidebar.title('Your analysis')
S=st.session_state
STEPS=['1 · Load files','2 · Inspect events','3 · Refine and save fits','4 · Current–duration plots','5 · Cluster events','6 · Save clusters','7 · Compare salts']
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

def _auto_axis_limits(values,pad=.05,positive=False):
    a=np.asarray(values,float).ravel();a=a[np.isfinite(a)]
    if positive:a=a[a>0]
    if not len(a):return [0.,1.]
    lo=float(np.min(a));hi=float(np.max(a))
    if lo==hi:
        span=max(abs(lo)*.1,1e-6);lo-=span;hi+=span
    elif positive:
        lo=max(np.nextafter(0.,1.),lo/(1.+pad));hi=hi*(1.+pad)
    else:
        span=hi-lo;lo-=pad*span;hi+=pad*span
    return [float(lo),float(hi)]

def _axis_pair(label,key,default):
    c1,c2=st.columns(2)
    lo=c1.number_input(label+' min',value=float(default[0]),format='%.6f',key=key+'_min')
    hi=c2.number_input(label+' max',value=float(default[1]),format='%.6f',key=key+'_max')
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        st.warning(label+': max must be greater than min; current automatic limits are being used.')
        return list(default)
    return [float(lo),float(hi)]



def _read_comparison_package(uploaded):
    """Read one compact package exported from Step 6."""
    blob=uploaded.getvalue() if hasattr(uploaded,'getvalue') else uploaded
    with zipfile.ZipFile(io.BytesIO(blob),'r') as z:
        names=set(z.namelist());need={'cluster_assignments.csv','cluster_profiles.csv','cluster_summary.csv','comparison_meta.json'}
        missing=sorted(need-names)
        if missing:raise ValueError('Not a DNA Event Lab comparison package; missing: '+', '.join(missing))
        assignments=pd.read_csv(z.open('cluster_assignments.csv'))
        profiles=pd.read_csv(z.open('cluster_profiles.csv'))
        summary=pd.read_csv(z.open('cluster_summary.csv'))
        meta=json.loads(z.read('comparison_meta.json').decode('utf-8'))
    for frame,name in [(assignments,'assignments'),(profiles,'profiles'),(summary,'summary')]:
        if 'cluster' not in frame:raise ValueError(f'{name} table has no cluster column.')
        frame['cluster']=frame['cluster'].astype(int)
    return {'assignments':assignments,'profiles':profiles,'summary':summary,'meta':meta,'name':getattr(uploaded,'name','package.zip')}


def _cross_salt_stats(packages,mapping,reference_family=None):
    rows=[]
    blockade_candidates=['clustering_measured_mean_blockade_nA','mean_blockade_nA']
    ecd_candidates=['clustering_ecd_nA_ms','ecd_nA_ms']
    for salt,pkg in packages.items():
        table=pkg['assignments'];total=max(1,len(table));bcol=next((c for c in blockade_candidates if c in table),None);ecol=next((c for c in ecd_candidates if c in table),None)
        for cluster in sorted(table.cluster.astype(int).unique()):
            fam=mapping.get(salt,{}).get(int(cluster),'Unmapped')
            sub=table[table.cluster.astype(int)==int(cluster)]
            d=sub['duration_ms'].dropna().to_numpy(float) if 'duration_ms' in sub else np.array([])
            b=sub[bcol].dropna().to_numpy(float) if bcol else np.array([])
            ecd=sub[ecol].dropna().to_numpy(float) if ecol else np.array([])
            row={'salt':salt,'cluster':int(cluster),'family':fam,'n':int(len(sub)),'population_pct':100.*len(sub)/total}
            row.update(median_dwell_ms=float(np.median(d)) if len(d) else np.nan,
                       q1_dwell_ms=float(np.quantile(d,.25)) if len(d) else np.nan,
                       q3_dwell_ms=float(np.quantile(d,.75)) if len(d) else np.nan,
                       median_blockade_nA=float(np.median(b)) if len(b) else np.nan,
                       q1_blockade_nA=float(np.quantile(b,.25)) if len(b) else np.nan,
                       q3_blockade_nA=float(np.quantile(b,.75)) if len(b) else np.nan,
                       median_ecd_nA_ms=float(np.median(ecd)) if len(ecd) else np.nan,
                       q1_ecd_nA_ms=float(np.quantile(ecd,.25)) if len(ecd) else np.nan,
                       q3_ecd_nA_ms=float(np.quantile(ecd,.75)) if len(ecd) else np.nan)
            rows.append(row)
    stats=pd.DataFrame(rows)
    if reference_family and len(stats):
        refs=stats[stats.family==reference_family].set_index('salt')['median_blockade_nA'].to_dict()
        stats['relative_blockade']=[(r.median_blockade_nA/refs.get(r.salt,np.nan)) if np.isfinite(refs.get(r.salt,np.nan)) and refs.get(r.salt,np.nan)!=0 else np.nan for r in stats.itertuples()]
    return stats

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

if step.startswith('7'):
    st.header('7 · Compare salts')
    st.write('Compare already-clustered LiCl, NaCl, KCl, RbCl and CsCl recordings without pooling their PCAs. Upload one compact comparison package from Step 6 for each condition, map each recording-specific Cluster ID to a common Family A/B/C… label, then compare family morphology, population and kinetics.')
    st.info('Important: Cluster 0 in one salt is not assumed to equal Cluster 0 in another. The family mapping below is an explicit physical correspondence that you confirm from the profiles and summary statistics.')
    salts=['LiCl','NaCl','KCl','RbCl','CsCl']
    packages={};errors=[]
    cols=st.columns(5)
    for i,salt in enumerate(salts):
        up=cols[i].file_uploader(salt+' comparison package',type='zip',key='salt_compare_'+salt)
        if up is not None:
            try:packages[salt]=_read_comparison_package(up)
            except Exception as ex:errors.append(f'{salt}: {ex}')
    for err in errors:st.error(err)
    if not packages:
        st.caption('To make one: cluster a recording in Step 5, go to Step 6, and use “Prepare salt-comparison package”. Repeat once for each salt.')
        navigation('bottom');st.stop()
    if len(packages)<5:st.warning(f'{len(packages)} of 5 salts loaded. You can preview a partial comparison now; add the remaining packages for the final figure.')

    pkg_rows=[]
    for salt,pkg in packages.items():
        m=pkg['meta'];pkg_rows.append({'Salt':salt,'Events':len(pkg['assignments']),'Clusters':len(pkg['summary']),'Sampling rate (kHz)':float(m.get('sampling_rate_hz',np.nan))/1000.,'Profile source':m.get('profile_source','')})
    st.dataframe(pd.DataFrame(pkg_rows).round(3),hide_index=True,width='stretch')

    st.subheader('1 · Match recording-specific clusters to common DNA event families')
    st.caption('The default suggestion follows the app ordering (shallowest median profile → Cluster 0 → Family A, then B, C…). Change any assignment that does not look physically homologous across salts. Use “Unmapped” rather than forcing a doubtful correspondence.')
    family_options=['Unmapped']+[chr(65+i) for i in range(10)]
    mapping={};duplicate_problem=False
    for salt in salts:
        if salt not in packages:continue
        pkg=packages[salt];clusters=sorted(pkg['summary'].cluster.astype(int).unique().tolist())
        with st.expander(f'{salt} · inspect and map {len(clusters)} clusters',expanded=(salt==next(iter(packages)))):
            show(comparison_condition_preview_figure(pkg['profiles'],salt,'time_ms'),'compare_preview_'+salt)
            qcols=st.columns(min(4,max(1,len(clusters))))
            mapping[salt]={}
            for j,cl in enumerate(clusters):
                default=chr(65+j) if j<10 else 'Unmapped'
                key=f'family_map_{salt}_{cl}'
                current=S.get(key,default)
                if current not in family_options:current=default
                fam=qcols[j%len(qcols)].selectbox(f'Cluster {cl}',family_options,index=family_options.index(current),key=key)
                mapping[salt][int(cl)]=fam
            chosen=[v for v in mapping[salt].values() if v!='Unmapped']
            if len(chosen)!=len(set(chosen)):
                duplicate_problem=True;st.error('Two clusters in this recording are mapped to the same family. Use unique family labels, or leave one Unmapped.')
    if duplicate_problem:
        st.warning('Resolve duplicate family assignments before generating the cross-salt figures.')
        navigation('bottom');st.stop()

    mapped_families=sorted({v for m in mapping.values() for v in m.values() if v!='Unmapped'})
    if not mapped_families:
        st.info('Map at least one cluster to a family label.');navigation('bottom');st.stop()

    st.subheader('2 · Cross-salt comparison settings')
    c1,c2,c3=st.columns(3)
    compare_mode=c1.radio('Family-profile horizontal axis',['Time relative to event midpoint (ms)','Data index'],index=0,key='salt_compare_axis')
    x_mode='time_ms' if compare_mode.startswith('Time') else 'data_index'
    reference_family=c2.selectbox('Reference family for optional relative blockade',['None']+mapped_families,index=0,key='salt_reference_family')
    manual_compare=c3.checkbox('Use fixed shared profile axes',value=True,key='salt_fixed_axes')
    all_x=[];all_y=[]
    for pkg in packages.values():
        all_x.append(pkg['profiles'][x_mode].to_numpy(float));all_y.append(pkg['profiles']['median_blockade_nA'].to_numpy(float))
    x_auto=_auto_axis_limits(np.concatenate(all_x));y_auto=_auto_axis_limits(np.concatenate(all_y))
    if manual_compare:
        a,b=st.columns(2);x_range=_axis_pair('Family profile '+('time (ms)' if x_mode=='time_ms' else 'data index'),'salt_profile_x',x_auto);y_range=_axis_pair('Family profile blockade (nA)','salt_profile_y',y_auto)
    else:x_range=x_auto;y_range=y_auto

    stats=_cross_salt_stats(packages,mapping,None if reference_family=='None' else reference_family)
    st.subheader('3 · Matched family summary')
    summary_cols=['salt','cluster','family','n','population_pct','median_dwell_ms','q1_dwell_ms','q3_dwell_ms','median_blockade_nA','q1_blockade_nA','q3_blockade_nA']
    if 'median_ecd_nA_ms' in stats and np.isfinite(pd.to_numeric(stats['median_ecd_nA_ms'],errors='coerce')).any():summary_cols+=['median_ecd_nA_ms','q1_ecd_nA_ms','q3_ecd_nA_ms']
    if 'relative_blockade' in stats:summary_cols.append('relative_blockade')
    st.dataframe(stats[summary_cols].round(4),hide_index=True,width='stretch')
    st.download_button('Download cross-salt family summary CSV',stats.to_csv(index=False),'cross_salt_family_summary.csv','text/csv')
    st.download_button('Download family mapping JSON',json.dumps({s:{str(k):v for k,v in m.items()} for s,m in mapping.items()},indent=2),'family_mapping.json','application/json')

    st.subheader('4 · Family morphology across electrolytes')
    st.caption('Each panel follows one user-matched family across salts. Lines are the centered pointwise median waveforms from the independently clustered recordings; the same numerical x/y limits are used in every family panel.')
    show(cross_salt_family_profiles_figure(packages,mapping,x_mode,x_range,y_range),'cross_salt_profiles')

    st.subheader('5 · Does salt change the probability of each event family?')
    show(cross_salt_population_figure(stats),'cross_salt_populations')
    if (stats.family=='Unmapped').any():st.caption('Grey “Unmapped” fraction is retained so the stacked bars still represent all clustered events rather than silently renormalising the selected families.')

    st.subheader('6 · Does salt change the kinetics of the same family?')
    st.caption('Points are event-level medians within each mapped family; error bars are the event-level interquartile range (Q1–Q3). These are descriptive within-recording spreads, not replicate-level confidence intervals.')
    show(cross_salt_metric_figure(stats,'median_dwell_ms','q1_dwell_ms','q3_dwell_ms','Family-resolved dwell time across salts','Median dwell time (ms)',False),'cross_salt_dwell')
    show(cross_salt_metric_figure(stats,'median_blockade_nA','q1_blockade_nA','q3_blockade_nA','Family-resolved blockade across salts','Median mean blockade (nA)',False),'cross_salt_blockade')
    if reference_family!='None' and 'relative_blockade' in stats:
        rel=stats[stats.family!='Unmapped'].copy();rel['q1_relative']=np.nan;rel['q3_relative']=np.nan
        # Reference-normalised medians are shown without event-level error bars because
        # the denominator is itself estimated from the reference family in each salt.
        f=px.line(rel,x='salt',y='relative_blockade',color='family',markers=True,category_orders={'salt':salts},labels={'salt':'Electrolyte','relative_blockade':f'Median blockade / Family {reference_family} median','family':'Family'})
        f.add_hline(y=1,line_dash='dot',line_color='gray');f.update_layout(height=400,title=f'Blockade relative to Family {reference_family} within each salt')
        show(f,'cross_salt_relative_blockade')

    st.subheader('7 · Heatmap summary across salts')
    st.caption('Heatmaps condense the family × electrolyte trends into one view. Raw mode preserves physical units. Row z-score is useful for seeing how each family changes across salts; column z-score is useful for comparing families within each salt. Z-scoring here changes only the display colours, never the clustering or stored values.')
    heatmap_options=['Population fraction','Median dwell time','Median blockade']
    if 'median_ecd_nA_ms' in stats and np.isfinite(pd.to_numeric(stats['median_ecd_nA_ms'],errors='coerce')).any():heatmap_options.append('Median ECD')
    if reference_family!='None' and 'relative_blockade' in stats:heatmap_options.append('Relative blockade')
    default_heatmaps=[x for x in ['Population fraction','Median dwell time','Median blockade'] if x in heatmap_options]
    heatmap_metrics=st.multiselect('Heatmaps to show',heatmap_options,default=default_heatmaps,key='salt_heatmap_metrics')
    heatmap_normalization=st.radio('Heatmap colour scaling',['Raw values','Row z-score (compare salts within each family)','Column z-score (compare families within each salt)'],index=0,key='salt_heatmap_normalization',horizontal=True)
    heatmap_specs={
        'Population fraction':('population_pct','Event-family population across salts','Population (%)','%',1),
        'Median dwell time':('median_dwell_ms','Family-resolved dwell time across salts','Median dwell time (ms)',' ms',2),
        'Median blockade':('median_blockade_nA','Family-resolved blockade across salts','Median mean blockade (nA)',' nA',2),
        'Median ECD':('median_ecd_nA_ms','Family-resolved ECD across salts','Median ECD (nA·ms)',' nA·ms',2),
        'Relative blockade':('relative_blockade',f'Blockade relative to Family {reference_family} across salts',f'Blockade / Family {reference_family}','',2),
    }
    for hm in heatmap_metrics:
        metric,title,cbar,suffix,decimals=heatmap_specs[hm]
        show(cross_salt_heatmap_figure(stats,metric,title,cbar,heatmap_normalization,suffix,decimals),'cross_salt_heatmap_'+metric)

    st.subheader('8 · Publication export')
    st.caption('The export contains the matched-family profile grid, 100% population bars, family-resolved dwell/blockade trends, the selected heatmaps, the family-summary CSV and the mapping JSON. Profile panels use the shared axes chosen above.')
    export_signature=(tuple(sorted((s,pkg['name']) for s,pkg in packages.items())),json.dumps(mapping,sort_keys=True),x_mode,tuple(x_range),tuple(y_range),reference_family,tuple(heatmap_metrics),heatmap_normalization)
    if st.button('Prepare cross-salt publication figure pack',type='primary'):
        S.cross_salt_export=(export_signature,cross_salt_comparison_archive(packages,mapping,stats,x_mode,x_range,y_range,None if reference_family=='None' else reference_family,heatmap_metrics,heatmap_normalization))
    if S.get('cross_salt_export') and S.cross_salt_export[0]==export_signature:
        st.download_button('Save cross-salt figure pack',S.cross_salt_export[1],'cross_salt_DNA_families.zip','application/zip')
    navigation('bottom');st.stop()

if 'project' not in S:st.info('Start at 1 · Load files.');st.stop()
if not S.get('recording_confirmed'):st.info('Confirm the recording match in 1 · Load files before continuing.');st.stop()
p=S.project;events=p['events'];refs=S.setdefault('refs',{});active=active_events(events,refs)
measured,profiles=describe(events,32,refs)
measured['raw_event_index']=[p['rawmap'][e.index].index if e.index in p['rawmap'] else np.nan for e in events]
measured['dataset_row']=[p['mapping'].get(e.index,np.nan) for e in events]
measured['fit_source']=['refined' if e.index in refs else 'uploaded' for e in events]
refhash=hashlib.sha256(b''.join(str(i).encode()+r['fit'].tobytes() for i,r in sorted(refs.items()))).hexdigest()
meta={'version':'0.9.1','source_hash':p['fingerprint'],'files':p['files'],'matching':p['matching'],'settings':safe_settings(p['settings']),
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
    st.write('This version returns to the original NanoSense dataset representation: the clustering coordinates come directly from selected columns of dataset.npz → X. No newly engineered plateau feature is required to form a cluster. Resolved-level quantities are calculated only afterwards to help interpret the signal families.')
    cfg=S.get('cluster_config',{})

    dataset=np.asarray(p['dataset'],float);n_dataset_cols=dataset.shape[1]
    start_col=int(p['matching']['start_column'])
    methods=['PCA + agglomerative (Ward)','PCA + k-means']
    previous=cfg.get('algorithm',methods[0]);algorithm=st.selectbox('Clustering algorithm',methods,index=methods.index(previous) if previous in methods else 0)
    st.caption('Both algorithms receive exactly the same selected dataset columns, the same min–max scaling, and the same PCA coordinates. Only the final grouping rule changes.')

    def dataset_col_display(j):
        name=dataset_feature_name(j)
        suffix=''
        if j==start_col:suffix='  ⚠ matching timestamp'
        elif j in (8,9):suffix='  ⚠ time/bookkeeping in the known NanoSense layout'
        elif j==7:suffix='  · baseline/QC in the known NanoSense layout'
        return f'X[:, {j}] · {name}{suffix}'

    known_default=[j for j in range(min(7,n_dataset_cols)) if j!=start_col]
    if len(known_default)<2:
        known_default=[j for j in range(n_dataset_cols) if j!=start_col][:max(2,min(7,n_dataset_cols-1))]
    previous_cols=[int(j) for j in cfg.get('dataset_columns',known_default) if 0<=int(j)<n_dataset_cols and int(j)!=start_col]
    if len(previous_cols)<2:previous_cols=known_default

    with st.expander('1 · Choose the original dataset features',expanded=True):
        st.caption('For the NanoSense layout used in your earlier analysis, columns 0–6 correspond to height, FWHM, height-at-FWHM, area, width, skewness and kurtosis. Column 7 is baseline/QC and columns 8–9 are event-time-like bookkeeping fields. Because NanoSense layouts can vary, the app shows the raw X[:, j] index beside every name and lets you change the selection.')
        options=[j for j in range(n_dataset_cols) if j!=start_col]
        dataset_cols=st.multiselect('Dataset columns used to form the PCA/clusters',options=options,default=previous_cols,format_func=dataset_col_display)
        if len(dataset_cols)<2:
            st.error('Select at least two dataset feature columns.');st.stop()
        prune_corr=st.checkbox('Prune near-duplicate selected columns (|r| ≥ 0.98)',value=bool(cfg.get('prune_corr',False)),help='OFF reproduces the original all-selected-features approach most closely. Turn it on only as a sensitivity check if two dataset descriptors are nearly identical.')
        corr_threshold=.98 if prune_corr else None
        st.caption('Scaling is the original-style MinMaxScaler to [-1, 1] before PCA. Units therefore do not determine feature weight. No log transforms, fold ratios, temporal centroids or hand-built plateau features are inserted into the clustering matrix.')

        rows=[]
        for j in range(n_dataset_cols):
            col=dataset[:,j];finite=col[np.isfinite(col)]
            rows.append({'Column':f'X[:, {j}]','Expected name':dataset_feature_name(j),'Selected':j in dataset_cols,
                         'Finite (%)':100*len(finite)/len(col) if len(col) else 0.,
                         'Median':float(np.median(finite)) if len(finite) else np.nan,
                         'Min':float(np.min(finite)) if len(finite) else np.nan,
                         'Max':float(np.max(finite)) if len(finite) else np.nan,
                         'Role':DATASET_FEATURE_DESCRIPTIONS.get(j,'Unknown/raw dataset column; verify before using.')})
        st.dataframe(pd.DataFrame(rows).round(6),hide_index=True)

    feature_names=[f'{dataset_feature_name(j)} [X{j}]' for j in dataset_cols]
    event_by_id={e.index:e for e in active}
    valid_ids=[];valid_rows=[];matrix_rows=[];excluded_rows=[]
    for e in active:
        row=p['mapping'].get(e.index)
        if row is None:
            excluded_rows.append({'event_index':e.index,'dataset_row':np.nan,'reason':'no unique matched dataset row'})
            continue
        values=dataset[int(row),dataset_cols]
        if not np.isfinite(values).all():
            bad=[dataset_col_display(dataset_cols[q]) for q in np.flatnonzero(~np.isfinite(values))]
            excluded_rows.append({'event_index':e.index,'dataset_row':int(row),'reason':'non-finite selected dataset feature(s): '+', '.join(bad)})
            continue
        valid_ids.append(e.index);valid_rows.append(int(row));matrix_rows.append(values)
    if len(valid_ids)<3:st.error('Fewer than three events have a matched dataset row with finite values in all selected columns.');st.stop()
    matrix=np.asarray(matrix_rows,float);eligible_active=[event_by_id[i] for i in valid_ids]
    measured_by_id=measured.set_index('event_index',drop=False)
    table_preview=measured_by_id.loc[valid_ids].reset_index(drop=True).copy()
    table_preview['dataset_row']=valid_rows
    for q,j in enumerate(dataset_cols):table_preview[f'dataset_X{j}_{dataset_feature_name(j)}']=matrix[:,q]
    dataset_excluded=pd.DataFrame(excluded_rows,columns=['event_index','dataset_row','reason'])
    st.info(f'{len(valid_ids)} of {len(active)} events have a unique matched dataset row and finite values in the {len(dataset_cols)} selected feature columns.')
    if len(dataset_excluded):
        with st.expander('Dataset-feature exclusions'):
            st.dataframe(dataset_excluded,hide_index=True)
            st.download_button('Save dataset-feature exclusion CSV',dataset_excluded.to_csv(index=False),'dataset_feature_exclusions.csv','text/csv')

    try:space_diag=feature_space_diagnostics(matrix,feature_names,corr_threshold)
    except Exception as ex:st.error(str(ex));st.stop()
    with st.expander('2 · Inspect the dataset feature space before clustering',expanded=True):
        show(feature_correlation_figure(space_diag['correlation'],feature_names),'dataset_feature_correlation')
        if space_diag['dropped_constant']:st.warning('Constant selected columns removed: '+', '.join(space_diag['dropped_constant']))
        if space_diag['dropped_correlated']:st.info('Near-duplicate selected columns removed: '+', '.join(space_diag['dropped_correlated']))
        st.write('Retained for PCA:',', '.join(space_diag['retained_features']))
        st.dataframe(pd.DataFrame(space_diag['spread_table']).round(6),hide_index=True)
        show(pca_scree_figure(space_diag['scree']),'dataset_scree_pre')
        scree_table=pd.DataFrame({'PC':np.arange(1,len(space_diag['scree'])+1),'Explained variance':space_diag['scree'],'Cumulative variance':space_diag['cumulative']})
        st.dataframe(scree_table.round(4),hide_index=True)
        st.caption('The PCA is built only from the selected original X columns after min–max scaling to [-1,1]. The correlation matrix is shown so you can see whether several NanoSense descriptors are encoding the same thing.')

    max_pc=max(1,int(space_diag['max_components']))
    if max_pc>=2:
        npc=st.slider('Principal components used for clustering',2,max_pc,min(max(2,int(cfg.get('npc',2))),max_pc),help='Two PCs reproduces the Hart-style DNA clustering view most closely. You can increase this as a sensitivity check; the PC1–PC2 graph remains only a display.')
    else:
        npc=1;st.metric('Principal components used for clustering',1)
    positions=st.select_slider('Internal normalized-profile resolution',[64,128,256],value=cfg.get('positions',128) if cfg.get('positions',128) in [64,128,256] else 128,help='Used internally for representative matching and for the optional normalized-event-position view. Data-index and real-time family plots use the original recorded samples instead.')
    kmax=st.slider('Largest k to include in elbow–silhouette scan',3,10,int(cfg.get('kmax',8)))
    base_method='PCA + agglomerative' if 'agglomerative' in algorithm.lower() else 'PCA + k-means'

    diag_signature=(p['fingerprint'],tuple(dataset_cols),prune_corr,algorithm,npc,kmax,positions)
    if st.button('Run elbow + silhouette scan'):
        try:
            with st.spinner('Testing candidate cluster counts in the original dataset feature space…'):
                S.cluster_scan=(diag_signature,cluster_count_diagnostics(matrix,base_method,npc,2,kmax,feature_names,corr_threshold))
        except Exception as ex:st.error(str(ex))
    scan=S.get('cluster_scan')
    if scan and scan[0]!=diag_signature:
        S.pop('cluster_scan',None);scan=None
    if scan:
        scan=scan[1]
        with st.expander('3 · Choose k from elbow + silhouette',expanded=True):
            show(k_diagnostics_figure(scan['table'],scan['elbow_k'],scan['silhouette_k']),'dataset_k_diagnostics_pre')
            st.dataframe(pd.DataFrame(scan['table']).round(4),hide_index=True)
            if scan['agreement']:st.success(f'Elbow and silhouette agree on k = {scan["suggested_k"]}.')
            else:st.warning(f'Elbow suggests k = {scan["elbow_k"]}, while silhouette peaks at k = {scan["silhouette_k"]}. Inspect both the PCA geometry and the actual waveform families before choosing.')
            st.caption('This scan cannot prove that DNA conformations are truly discrete; it only tests compactness/separation for k ≥ 2.')
    else:st.info('Run the elbow + silhouette scan before finalising k. The app can still cluster if you choose k manually.')

    suggested=int(scan['suggested_k']) if scan else int(cfg.get('k',3));suggested=max(2,min(suggested,kmax))
    previous_k=max(2,min(int(cfg.get('k',suggested)),kmax))
    k=st.slider('Number of clusters used for the final grouping',2,kmax,previous_k)

    with st.expander('4 · Physical interpretation settings (do NOT form the clusters)',expanded=False):
        profile_options=['Measured trace','Selected fits'];profile_source=st.radio('Signal shown in the member-profile panels',profile_options,index=profile_options.index(cfg.get('profile_source','Measured trace')) if cfg.get('profile_source','Measured trace') in profile_options else 0,horizontal=True)
        previous_params=cfg.get('physical_params',{})
        c1,c2,c3=st.columns(3)
        min_us=c1.number_input('Minimum plateau duration (µs)',min_value=0.,value=float(previous_params.get('min_duration_us',25.)))
        min_height=c2.number_input('Minimum level difference (nA)',min_value=0.,value=float(previous_params.get('min_height_nA',.1)),format='%.3f')
        noise_mult=c3.number_input('Noise multiplier for level merging',min_value=0.,value=float(previous_params.get('noise_multiplier',3.)))
        omit_edges=st.checkbox('Omit short boundary plateaus from interpretation',value=previous_params.get('omit_short_boundaries',True))
        use_ref=st.checkbox('Use calibrated single-file blockade reference for interpretation',value=previous_params.get('reference_nA') is not None)
        reference=st.number_input('Single-file reference blockade ΔI₀ (nA)',min_value=.000001,value=float(previous_params.get('reference_nA') or 1.),format='%.4f') if use_ref else None
        use_deep=st.checkbox('Use deeper-blockade threshold for interpretation',value=previous_params.get('deep_threshold_nA') is not None)
        threshold=st.number_input('Deeper-blockade threshold (nA)',min_value=.000001,value=float(previous_params.get('deep_threshold_nA') or 1.5),format='%.4f') if use_deep else None
        st.caption('These resolved-level settings affect only the post-cluster interpretation plots/tables. They never change the PCA coordinates or cluster assignments in this dataset-feature mode.')
    physical_params=dict(min_duration_us=min_us,min_height_nA=min_height,noise_multiplier=noise_mult,reference_nA=reference,deep_threshold_nA=threshold,omit_short_boundaries=omit_edges)
    physical_audit,physical_sequences=level_features(active,profile_source,**physical_params)

    S.cluster_config=dict(algorithm=algorithm,k=k,npc=npc,positions=positions,kmax=kmax,dataset_columns=dataset_cols,prune_corr=prune_corr,profile_source=profile_source,physical_params=physical_params)
    signature=(p['fingerprint'],algorithm,k,tuple(dataset_cols),prune_corr,npc,positions,kmax,profile_source,json.dumps(physical_params,sort_keys=True))

    if st.button('Run clustering',type='primary'):
        try:
            with st.spinner('Clustering the original NanoSense dataset features and preparing physical interpretation…'):
                current_scan=scan
                if current_scan is None:
                    current_scan=cluster_count_diagnostics(matrix,base_method,npc,2,kmax,feature_names,corr_threshold)
                    S.cluster_scan=(diag_signature,current_scan)
                profile_events=signal_events(eligible_active,profile_source);_,prof=describe(profile_events,positions)
                info=cluster_features(matrix,prof,k,npc,base_method,corr_threshold)
                table=table_preview.copy();table['cluster']=info['labels']

                # Merge resolved-level descriptors for interpretation only. Missing/failed
                # physical extraction leaves NaN values but does not remove the event.
                phys=physical_audit.set_index('event_index').reindex(table.event_index).reset_index(drop=True)
                for col in phys.columns:
                    if col!='event_index':table['clustering_'+col]=phys[col].to_numpy()
                # Direct measured quantities are always available even when plateau QC fails.
                table['clustering_measured_mean_blockade_nA']=table['mean_blockade_nA'].to_numpy()
                table['clustering_ecd_nA_ms']=table['ecd_nA_ms'].to_numpy()

                retained=[f for f,keep in zip(feature_names,info.get('feature_keep_mask',[True]*len(feature_names))) if keep]
                clustering_meta={'source':'dataset.npz X','method':algorithm,'k':int(k),'selection':'manual after elbow/silhouette diagnostics',
                    'dataset_columns':[int(j) for j in dataset_cols],'features':feature_names,'pca_components':int(info.get('n_components',npc)),
                    'profile_positions':positions,'profile_source':profile_source,'k_scan_range':[2,kmax],
                    'feature_set':'original NanoSense dataset features','scaling':'MinMaxScaler [-1,1]','correlation_pruning':prune_corr,
                    'correlation_threshold':corr_threshold,'excluded_events':len(dataset_excluded),'retained_features':retained,
                    'feature_min':info.get('feature_min'),'feature_max':info.get('feature_max'),'pca_loadings':info.get('loadings'),
                    'k_diagnostics':current_scan,'physical_interpretation_settings':physical_params}
                S.group={'signature':signature,'table':table,'info':info,'excluded':dataset_excluded,'sequences':physical_sequences,'physical_audit':physical_audit,'scan':current_scan,'meta':{**meta,'clustering':clustering_meta}}
                S.pop('prepared',None);S.pop('hart_export',None);S.pop('cluster_figures',None)
        except Exception as ex:st.error(str(ex))

    g=S.get('group')
    if g and g['signature']!=signature:
        S.pop('group',None);S.pop('prepared',None);S.pop('hart_export',None);S.pop('cluster_figures',None)
        st.info('Clustering settings changed. Run clustering again to update the results.');st.stop()

    if g:
        info=g['info'];table=g['table'];scan=g.get('scan');selected_k=len(np.unique(info['labels']))
        st.success(f'{len(table)} events grouped into {selected_k} signal families from the original dataset.npz feature columns using {algorithm}.')
        excluded=g.get('excluded',pd.DataFrame());sequences=g.get('sequences',pd.DataFrame());phys_audit=g.get('physical_audit',pd.DataFrame())
        if len(excluded):
            st.warning(f'{len(excluded)} events were not clustered because they lacked a unique matched dataset row or had non-finite values in one of the selected X columns.')
            with st.expander('Dataset-feature exclusions'):
                st.dataframe(excluded,hide_index=True);st.download_button('Save dataset-feature exclusions CSV',excluded.to_csv(index=False),'dataset_feature_exclusions.csv','text/csv')
        if len(phys_audit):
            phys_ok=int(phys_audit.set_index('event_index').reindex(table.event_index).physical_eligible.fillna(False).sum()) if 'physical_eligible' in phys_audit else 0
            st.caption(f'Resolved-level interpretation is available for {phys_ok} of the {len(table)} clustered events. Events without resolved plateau features remain valid cluster members.')

        # Cluster-family display controls are intentionally prominent rather than
        # hidden in an expander.  The default Hart-style view centres each detected
        # event on its midpoint in a fixed sample window, preserving real duration
        # while showing baseline before and after the event.
        profile_events_display=signal_events(eligible_active,profile_source)
        st.subheader('Cluster-family display controls')
        family_modes=['Data index (sample number)','Time relative to event midpoint (ms)','Normalized event position (0 = start, 1 = end)']
        previous_mode=S.get('family_x_mode',family_modes[0])
        if previous_mode not in family_modes:previous_mode=family_modes[0]
        family_mode=st.radio('Horizontal axis for cluster-family traces',family_modes,index=family_modes.index(previous_mode),key='family_x_mode',horizontal=True)
        if family_mode==family_modes[2]:
            family_profiles=np.asarray(info['profiles'],float);family_centers=np.asarray(info['centers'],float)
            family_x=(np.arange(family_profiles.shape[1])+.5)/family_profiles.shape[1]
            family_x_label='Normalized event position (0 = start, 1 = end)';family_x_name='normalized_event_position';family_event_start=None
            st.caption('Shape-only view: every event is stretched/compressed to 0–1, so absolute dwell-time differences are intentionally removed.')
        else:
            lengths=np.asarray([int(np.sum((e.time>=e.bounds[0])&(e.time<=e.bounds[1]))) for e in profile_events_display],int)
            suggested=max(300,int(np.ceil(np.percentile(lengths,95)/50.)*50)+200) if len(lengths) else 500
            suggested=min(max(suggested,300),5000)
            window_samples=int(st.number_input('Centered display window length (samples)',min_value=100,max_value=10000,value=int(S.get('family_window_samples',suggested)),step=50,key='family_window_samples',help='Each detected event midpoint is placed at the centre of this fixed window. Events are not stretched. Very long events can be visually clipped if the window is too short; clustering is unaffected.'))
            aligned=aligned_event_profiles(profile_events_display,alignment='midpoint',window_samples=window_samples)
            family_profiles=np.asarray(aligned['profiles'],float)
            family_centers,_=cluster_median_profiles(family_profiles,info['labels'],.5)
            if family_mode==family_modes[0]:
                family_x=np.asarray(aligned['data_index'],float)
                family_x_label=f'Data index (sample number; event midpoint = {aligned["event_midpoint_index"]})';family_x_name='data_index'
            else:
                family_x=np.asarray(aligned['time_ms'],float)
                family_x_label='Time relative to event midpoint (ms)';family_x_name='time_from_event_midpoint_ms'
            family_event_start=None
            fs=aligned.get('sampling_rate_hz',np.nan);dt_us=1e6*aligned.get('time_step_s',np.nan)
            n_long=int(np.sum(aligned['event_lengths_samples']>window_samples))
            st.caption(f'Each event midpoint is centred in a {window_samples}-sample window; event durations are preserved and baseline can appear on both sides. Median sampling interval = {dt_us:.3g} µs'+(f' ({fs/1000:.3g} kHz).' if np.isfinite(fs) else '.'))
            if n_long:st.warning(f'{n_long} events are longer than the selected display window and are visually clipped in these family plots. Increase the window length if you want to see their full start-to-end trace. Cluster assignments are unchanged.')
            if aligned.get('relative_dt_spread',0.)>.01:st.warning('Sampling intervals vary by more than 1% across these events. Data index remains valid as sample number; the shared millisecond axis should be interpreted cautiously.')
            st.caption('The red representative is the pointwise cluster median, drawn only where at least 50% of that cluster has recorded samples. No vertical alignment marker is drawn.')

        family_view={'profiles':family_profiles,'centers':family_centers,'x':np.asarray(family_x,float),'x_label':family_x_label,'x_name':family_x_name,'event_start_x':family_event_start,'mode':family_mode}

        # Plot ranges never affect PCA or cluster membership. Auto mode already forces
        # every cluster-family panel onto one shared blockade axis; manual mode lets
        # the same numerical ranges be reused across salts/voltages/recordings.
        embedding=np.asarray(info['embedding']);variance=info.get('pca_variance',[])
        time_examples=representative_time_examples(eligible_active,info,profile_source)
        profile_auto=_auto_axis_limits(np.r_[np.asarray(family_profiles).ravel(),np.asarray(family_centers).ravel()])
        profile_x_auto=[float(np.nanmin(family_x)),float(np.nanmax(family_x))]
        pca_x_auto=_auto_axis_limits(embedding[:,0]);pca_y_auto=_auto_axis_limits(embedding[:,1])
        dwell_x_auto=_auto_axis_limits(table['duration_ms'].to_numpy(),positive=True)
        blockade_y_auto=_auto_axis_limits(table['clustering_measured_mean_blockade_nA'].to_numpy())
        actual_x_auto=_auto_axis_limits(time_examples['time_ms'].to_numpy(),positive=False) if len(time_examples) else [0.,1.]
        actual_y_auto=_auto_axis_limits(time_examples['blockade_nA'].to_numpy()) if len(time_examples) else profile_auto
        dist_auto={
            'duration_ms':_auto_axis_limits(table['duration_ms'].to_numpy(),positive=True),
            'clustering_measured_mean_blockade_nA':blockade_y_auto,
            'clustering_fold_contrast':_auto_axis_limits(table['clustering_fold_contrast'].dropna().to_numpy()) if 'clustering_fold_contrast' in table else [0.,1.],
            'clustering_ecd_nA_ms':_auto_axis_limits(table['clustering_ecd_nA_ms'].dropna().to_numpy()) if 'clustering_ecd_nA_ms' in table else [0.,1.],
        }
        with st.expander('Axis controls for cluster families, PCA and physical plots',expanded=True):
            st.caption('Auto mode already gives every cluster-family panel the same blockade scale. Turn on manual/shared limits to set exact x/y ranges for the family plots and the other Step 5 figures. These controls change only display/export, never clustering.')
            manual_axes=st.checkbox('Use manual/shared numeric limits',value=bool(S.get('cluster_manual_axes',False)),key='cluster_manual_axes')
            dwell_log=st.checkbox('Use logarithmic dwell-time axis in blockade–dwell plots',value=bool(S.get('cluster_dwell_log',True)),key='cluster_dwell_log')
            if manual_axes:
                st.write('**Cluster-family profiles (shared by every cluster panel and the overlay)**')
                profile_x=_axis_pair(family_x_label,'axis_profile_x',profile_x_auto)
                profile_y=_axis_pair('Blockade (nA)','axis_profile_y',profile_auto)
                st.write('**PCA display**')
                pc1_range=_axis_pair('PC1','axis_pca_x',pca_x_auto);pc2_range=_axis_pair('PC2','axis_pca_y',pca_y_auto)
                st.caption('Matching PC limits are useful for layout consistency within one PCA model. Separately fitted PCAs from different salts/recordings are not physically comparable just because the displayed limits match.')
                st.write('**Measured blockade versus dwell time**')
                dwell_range=_axis_pair('Dwell time (ms)','axis_dwell_x',dwell_x_auto);blockade_range=_axis_pair('Mean blockade (nA)','axis_blockade_y',blockade_y_auto)
                if dwell_log and dwell_range[0]<=0:
                    st.warning('A logarithmic dwell axis needs a positive minimum; the automatic positive minimum is being used.')
                    dwell_range=[dwell_x_auto[0],max(dwell_range[1],dwell_x_auto[1])]
                st.write('**Representative real events in actual time**')
                actual_x=_axis_pair('Time from event start (ms)','axis_actual_x',actual_x_auto);actual_y=_axis_pair('Blockade (nA)','axis_actual_y',actual_y_auto)
                st.write('**Physical-distribution panels**')
                dist_ranges={
                    'duration_ms':_axis_pair('Distribution · dwell time (ms)','axis_dist_duration',dist_auto['duration_ms']),
                    'clustering_measured_mean_blockade_nA':_axis_pair('Distribution · mean blockade (nA)','axis_dist_blockade',dist_auto['clustering_measured_mean_blockade_nA']),
                    'clustering_fold_contrast':_axis_pair('Distribution · fold contrast','axis_dist_fold',dist_auto['clustering_fold_contrast']),
                    'clustering_ecd_nA_ms':_axis_pair('Distribution · ECD (nA·ms)','axis_dist_ecd',dist_auto['clustering_ecd_nA_ms']),
                }
                fold_x=_axis_pair('Fold-state occupancy','axis_fold_x',[0.,1.]);fold_y=_axis_pair('Fold-state contrast','axis_fold_y',[0.,1.])
                occupancy_y=_axis_pair('Deepest-state occupancy','axis_occupancy_y',[0.,1.])
            else:
                profile_x=profile_x_auto;profile_y=profile_auto;pc1_range=pc2_range=dwell_range=blockade_range=actual_x=actual_y=None
                dist_ranges={};fold_x=[0.,1.];fold_y=[0.,1.];occupancy_y=[0.,1.]
                st.info(f'Auto shared cluster-family axes: {family_x_label} = {profile_x[0]:.4g} to {profile_x[1]:.4g}; blockade = {profile_y[0]:.4g} to {profile_y[1]:.4g} nA.')
        axis_settings={'manual':bool(manual_axes),'profile_x':profile_x,'profile_y':profile_y,'pca_x':pc1_range,'pca_y':pc2_range,
                       'dwell_x':dwell_range,'blockade_y':blockade_range,'dwell_log_x':bool(dwell_log),
                       'actual_x':actual_x,'actual_y':actual_y,'fold_x':fold_x,'fold_y':fold_y,
                       'occupancy_y':occupancy_y,'distribution_ranges':dist_ranges,
                       'family_mode':family_mode,'family_x_label':family_x_label,'family_event_start_x':family_event_start,
                       'family_window_samples':int(S.get('family_window_samples',500)) if family_mode!=family_modes[2] else None}
        axis_key=json.dumps(axis_settings,sort_keys=True)

        st.subheader('A · Statistical clustering result from the original dataset features')
        projection=pd.DataFrame(embedding,columns=['PC1','PC2']);projection['Cluster']=table['cluster'].astype(str).to_numpy()
        projection['Event ID']=table['event_index'].to_numpy();projection['Dataset row']=table['dataset_row'].to_numpy()
        projection['Duration (ms)']=table['duration_ms'].to_numpy();projection['Measured mean blockade (nA)']=table['mean_blockade_nA'].to_numpy()
        axis_labels={f'PC{j+1}':f'PC{j+1} ({100*variance[j]:.1f}% variance)' if j<len(variance) else f'PC{j+1}' for j in range(2)}
        show(cluster_pca_figure(projection,axis_labels,False,axis_settings['pca_x'],axis_settings['pca_y']),'cluster_projection')
        st.caption(f'Each point is one matched event. No convex-hull fill is used. The black × is the median displayed PC position of that cluster. The grouping used {info.get("n_components",npc)} PC(s) derived only from the selected original dataset columns.')
        st.download_button('Save PC coordinates CSV',projection.to_csv(index=False),'pca_coordinates.csv','text/csv')

        st.subheader('B · Do the dataset-derived groups make physical sense?')
        show(blockade_dwell_figure(table,log_x=axis_settings['dwell_log_x'],x_range=axis_settings['dwell_x'],y_range=axis_settings['blockade_y']),'cluster_blockade_dwell')
        st.caption('Direct measured dwell-time versus blockade view, coloured only after clustering. This plot did not create the groups.')

        if 'clustering_fold_contrast' in table and table['clustering_fold_contrast'].notna().any():
            show(fold_state_figure(table,x_range=axis_settings['fold_x'],y_range=axis_settings['fold_y']),'cluster_fold_state')
            st.caption('Resolved-level fold-state map for interpretation only. Missing plateau descriptors do not alter cluster membership.')

        show(physical_feature_distributions_figure(table,axis_settings['distribution_ranges']),'cluster_physical_distributions')
        p1,p2,p3=st.columns(3)
        with p1:
            if 'clustering_resolved_levels' in table and table['clustering_resolved_levels'].notna().any():
                show(level_composition_figure(table),'cluster_level_composition')
                st.caption('Resolved level composition is independent supporting evidence because level count was not used to form the clusters.')
        with p2:
            if 'clustering_deepest_plateau_fraction' in table and table['clustering_deepest_plateau_fraction'].notna().any():
                show(occupancy_figure(table,y_range=axis_settings['occupancy_y']),'cluster_deepest_occupancy')
                st.caption('Deepest-state occupancy is also interpretation only.')
        with p3:
            show(population_fraction_figure(table),'cluster_population')

        st.write('**Cluster medians of the original selected dataset features**')
        rows=[]
        for cid,sub in table.groupby('cluster',sort=True):
            row={'Cluster':int(cid),'Events':len(sub),'Population (%)':100*len(sub)/len(table)}
            for j in dataset_cols:
                col=f'dataset_X{j}_{dataset_feature_name(j)}';row[dataset_feature_label(j)]=float(sub[col].median())
            row['Measured dwell (ms)']=float(sub.duration_ms.median());row['Measured mean blockade (nA)']=float(sub.mean_blockade_nA.median())
            rows.append(row)
        cluster_summary=pd.DataFrame(rows)
        st.dataframe(cluster_summary.round(5),hide_index=True)
        st.download_button('Save cluster summary CSV',cluster_summary.to_csv(index=False),'cluster_summary.csv','text/csv')

        st.subheader('C · What do the actual event families look like?')
        columns=st.columns(3)
        for group_id in range(selected_k):
            with columns[group_id%3]:
                show(member_profile_figure_hart(family_view['profiles'],info['labels'],family_view['centers'],group_id,
                    x=family_view['x'],x_label=family_view['x_label'],x_range=axis_settings['profile_x'],y_range=axis_settings['profile_y'],event_start_x=family_view['event_start_x']),f'dataset_members_{group_id}')
        if family_mode==family_modes[2]:
            st.caption('Grey curves are duration-normalised member profiles. The red curve is the pointwise median across all members. The 0–1 horizontal coordinate compares waveform shape but intentionally removes absolute dwell-time differences.')
        else:
            st.caption('Grey curves are centred real recorded traces. The red curve is the pointwise median across cluster members. Individual event durations are not stretched, so starts and ends remain physically meaningful within the centred window.')
        st.write('**Median representative profiles overlaid**')
        show(profile_figure(family_view['profiles'],info['labels'],family_view['centers'],x=family_view['x'],x_label=family_view['x_label'],x_range=axis_settings['profile_x'],y_range=axis_settings['profile_y'],event_start_x=family_view['event_start_x']),'cluster_profiles')

        with st.expander('Representative real events in actual time (ms)',expanded=True):
            show(time_example_figure(time_examples,axis_settings['actual_x'],axis_settings['actual_y']),'cluster_actual_time')
            st.download_button('Save actual-time example curves',time_examples.to_csv(index=False),'actual_time_examples.csv','text/csv')

        st.subheader('D · Supplementary clustering diagnostics')
        c1,c2=st.columns(2)
        with c1:show(feature_correlation_figure(space_diag['correlation'],feature_names),'cluster_feature_correlation_result')
        with c2:show(pca_scree_figure(info['scree']),'cluster_scree_result')
        if scan:show(k_diagnostics_figure(scan['table'],scan['elbow_k'],scan['silhouette_k']),'cluster_k_diagnostics_result')
        if info.get('embedding3') is not None:
            with st.expander('3-PC view'):
                show(cluster_pca_3d_figure(info['embedding3'],info['labels'],info.get('pca_variance',[]),table.event_index.to_numpy()),'cluster_pca_3d')
        if info.get('linkage_matrix') is not None:
            with st.expander('Ward dendrogram',expanded=False):show(dendrogram_figure(info['linkage_matrix'],selected_k,50),'cluster_dendrogram')

        kept=[f for f,keep in zip(feature_names,info.get('feature_keep_mask',[True]*len(feature_names))) if keep]
        loads=np.asarray(info.get('loadings',[]));pcs=min(3,len(loads));load_rows=[]
        for j in range(pcs):
            var=100*info['scree'][j] if j<len(info.get('scree',[])) else np.nan
            for name,value in zip(kept,loads[j]):load_rows.append({'Feature':name,'Loading':value,'PC':f'PC{j+1} ({var:.1f}%)'})
        if load_rows:
            with st.expander('PCA loadings: which original dataset features construct each PC?',expanded=True):
                show(px.bar(pd.DataFrame(load_rows),x='Loading',y='Feature',color='PC',barmode='group',orientation='h'),'pca_loadings')
                st.caption('These loadings refer directly to the selected dataset.npz columns after MinMax scaling to [-1,1].')

        with st.expander('Numerical diagnostics'):
            st.write('Selected dataset columns:',[dataset_col_display(j) for j in dataset_cols])
            st.write('Scaling: MinMaxScaler [-1,1]')
            st.write('PCA components used:',info.get('n_components'))
            st.write('Silhouette:',info.get('silhouette'));st.write('Calinski–Harabasz:',info.get('calinski_harabasz'));st.write('Davies–Bouldin:',info.get('davies_bouldin'))
            if np.isfinite(info.get('ari',np.nan)):st.write('K-means repeat-seed ARI:',info.get('ari'))
            if scan:st.write('Elbow suggestion:',scan['elbow_k']);st.write('Silhouette peak:',scan['silhouette_k'])
            st.caption('These metrics diagnose signal geometry. They do not by themselves identify unfolded/folded DNA states.')

        cluster=st.selectbox('Inspect every event in cluster',sorted(table.cluster.unique()));ids=table.loc[table.cluster==cluster,'event_index'].tolist();eid=st.selectbox('Event in this cluster',ids)
        e=next(e for e in events if e.index==eid);show(trace_figure(e,refs.get(e.index)),'cluster_event')

        if st.button('Prepare representative profile figures'):
            S.cluster_figures=((signature,axis_key),profile_archive(family_view['profiles'],info['labels'],family_view['centers'],axis_settings['profile_y'],
                x=family_view['x'],x_label=family_view['x_label'],x_range=axis_settings['profile_x'],event_start_x=family_view['event_start_x'],x_name=family_view['x_name']))
        if S.get('cluster_figures') and S.cluster_figures[0]==(signature,axis_key):st.download_button('Save representative profile PDF, SVG and PNG figures',S.cluster_figures[1],'cluster_profiles.zip')
        if st.button('Prepare main + supplementary DNA clustering figure pack'):
            S.hart_export=((signature,axis_key),hart_style_archive(info,table['event_index'].to_numpy(),table,cluster_summary,scan,'dataset.npz X',algorithm,axis_settings=axis_settings,family_view=family_view))
        if S.get('hart_export') and S.hart_export[0]==(signature,axis_key):st.download_button('Save DNA clustering figure pack',S.hart_export[1],'dna_clustering_figures.zip','application/zip')
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
                z.writestr('dataset_feature_exclusions.csv',excluded.to_csv(index=False))
                ids=set(excluded.event_index);ev=[e for e in events if e.index in ids]
                z.writestr('excluded_events.zip',bundle(ev,p['rawmap'],refs,p['dataset'],p['mapping'],p['settings'],p['dsettings'],excluded,{**g['meta'],'assignment':'excluded from dataset-feature clustering'},g.get('sequences')))
        S.prepared=(key,outer.getvalue())
    if S.get('prepared') and S.prepared[0]==key:st.download_button('Save cluster files ZIP',S.prepared[1],'cluster_files.zip','application/zip')
    st.write('Each cluster contains its selected eventfitting file, matched eventdata and original dataset subset, event table and analysis settings. The all-clusters download also records dataset-feature exclusions and resolved-level interpretation data when available. Uploaded fits remain preserved inside the fitting export.')
    st.caption('Dataset rows are preserved as uploaded; they are not recalculated after refinement. Updated clustering measurements are in the CSV. Files reload in this app; compatibility with NanoSense re-import is not established.')

    st.divider()
    st.subheader('Prepare this recording for the five-salt comparison')
    st.write('This compact package keeps the cluster assignments, family-level centered median waveforms and summary statistics needed by Step 7. It does not pool or re-run PCA.')
    c1,c2=st.columns(2)
    condition_label=c1.text_input('Condition label (optional)',value=S.get('comparison_condition_label',''),key='comparison_condition_label',placeholder='e.g. LiCl · λ-DNA · 400 mV')
    compare_window=int(c2.number_input('Centered profile window saved in package (samples)',min_value=100,max_value=10000,value=int(S.get('family_window_samples',850)),step=50,key='comparison_package_window'))
    comparison_key=(str(g['signature']),condition_label,compare_window)
    if st.button('Prepare salt-comparison package'):
        profile_source=g['meta']['clustering'].get('profile_source','Measured trace')
        S.comparison_package=(comparison_key,comparison_package(active,table,profile_source,g['meta'],condition_label,compare_window))
    if S.get('comparison_package') and S.comparison_package[0]==comparison_key:
        st.download_button('Save comparison package ZIP',S.comparison_package[1],'DNA_Event_Lab_comparison_package.zip','application/zip')
    st.caption('Repeat this export after clustering LiCl, NaCl, KCl, RbCl and CsCl. Then upload the five packages in Step 7 · Compare salts.')

    navigation("bottom")

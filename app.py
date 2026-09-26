import hashlib, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from analysis import load_export, describe, cluster_profiles, synthetic_demo

st.set_page_config(page_title='Nanopore Shape Lab',page_icon='🧬',layout='wide')
st.title('Nanopore Shape Lab')
st.caption('Inspect the signal. Check the fit. Discover recurring shapes.')
st.info('Exploratory analysis: shape groups are not confirmed DNA topologies. Analyse one pore, salt and voltage at a time.')
source=st.sidebar.radio('Data source',['Upload export','Synthetic demonstration'])
upload=None;digest='synthetic-demo-v1'
if source=='Upload export':
    upload=st.sidebar.file_uploader('NanoSense event_fitting.npz',type=['npz'])
    if upload is None:
        st.markdown('### Start with your full event-fitting export\nUpload the file ending in **`.event_fitting.npz`**. It includes the waveforms and fitted segments. Use the synthetic demonstration to explore the interface.\n\n1. **Inspect** an event and its saved fit.\n2. **Audit** fit disagreement and short segments.\n3. **Explore groups** of time-ordered signal shapes.\n4. **Export** results and settings for review.')
        st.stop()
    blob=upload.getvalue();digest=hashlib.sha256(blob).hexdigest()
    try:
        # No global data cache: each session holds its own uploaded traces.
        if st.session_state.get('loaded_digest')!=digest:
            st.session_state.loaded=load_export(blob);st.session_state.loaded_digest=digest
        events,settings,rejected=st.session_state.loaded
    except Exception as ex:
        st.error(f'Could not load this export: {ex}');st.stop()
else:events,settings,rejected=synthetic_demo()
if st.session_state.get('data_digest')!=digest:
    st.session_state.pop('group_result',None);st.session_state.data_digest=digest
bins=st.sidebar.select_slider('Shape resolution (time bins)',[16,32,64],value=32)
df,profiles=describe(events,bins)
by_id={e.index:e for e in events}
c1,c2,c3=st.columns(3)
c1.metric('Events loaded',len(events));c2.metric('Median duration',f'{df.duration_ms.median():.3f} ms');c3.metric('Median fit RMSE',f'{df.fit_rmse_nA.median():.3f} nA')
if rejected:st.warning(f'{len(rejected)} malformed events excluded. See audit details.')
if upload and 'SHORT' in upload.name.upper():st.warning('This filename suggests a selected subset. Group fractions describe only this file, not the original population.')
with st.expander('Recording settings and assumptions'):
    st.json({k:v for k,v in settings.items() if k!='file_name'})
    st.caption('Uses saved time/current/fit/baseline arrays as exported; current units assumed nA and time seconds. No extra filtering or automatic fit correction. Padding noise is a diagnostic scale and may contain correlated signal. The app cannot recover information lost in reduction.')
    if upload and '4.5' in upload.name and settings.get('threshold_value_multiplier')!=4.5:
        st.warning(f"Filename contains 4.5; saved detection multiplier is {settings.get('threshold_value_multiplier')}. Use recorded metadata after checking the reduction history.")

def draw(e,key):
    t=(e.time-e.bounds[0])*1000;f=go.Figure()
    f.add_trace(go.Scatter(x=t,y=e.baseline-e.current,name='Saved trace',line=dict(color='#287d9b',width=1.5)))
    f.add_trace(go.Scatter(x=t,y=e.baseline-e.fit,name='Saved fit',line=dict(color='#ed9140',shape='hv')))
    f.add_vline(x=0,line_dash='dot');f.add_vline(x=(e.bounds[1]-e.bounds[0])*1000,line_dash='dot')
    f.update_layout(height=360,xaxis_title='Time from detected start (ms)',yaxis_title='Blockade (nA)',margin=dict(t=25,b=35))
    st.plotly_chart(f,width='stretch',key=key)

inspect,audit,groups,export=st.tabs(['1 · Inspect events','2 · Audit fits','3 · Explore groups','4 · Export'])
with inspect:
    eid=st.selectbox('Original event index',[e.index for e in events]);e=by_id[eid]
    st.caption(f'Recording start: {e.bounds[0]:.6f} s. Indices are zero-based within the uploaded file; subset exports may renumber them.')
    draw(e,'inspect')
    st.dataframe(pd.DataFrame({'segment':np.arange(1,len(e.levels)+1),'saved_blockade_nA':e.levels,'duration_ms':e.widths*1000,'duration_fraction':e.widths/e.widths.sum()}),hide_index=True)
    st.caption('Segment count is not DNA occupancy. Short edges can be fitted as separate segments. Depth ratios alone do not prove folding.')
with audit:
    st.markdown('Compare fit disagreement with the current scale and inspect flagged events before grouping. These are adjustable review flags, not validated quality gates.')
    st.plotly_chart(px.scatter(df,x='duration_ms',y='fit_rmse_nA',color='segments',hover_data=['event_index','fit_bias_nA'],log_x=True),width='stretch')
    st.dataframe(df.sort_values('fit_rmse_nA',ascending=False),hide_index=True)
    if rejected:st.dataframe(pd.DataFrame(rejected))
with groups:
    st.markdown('Each trace is represented by its blockade at equally spaced fractions of event duration. This preserves depth and temporal order while removing absolute duration from clustering. Fits are audited, but grouping uses the **saved trace**, so it does not depend on trusting every saved plateau.')
    st.caption('This first version uses k-means on ordered profiles. It is not an occupancy-state model. Rounded pulses and broad transitions may group together; group count is chosen for exploration, not discovered topology.')
    a,b,c=st.columns(3)
    k=a.slider('Exploratory groups',2,8,4)
    min_samples=b.number_input('Minimum event samples',3,1000,10)
    max_rmse=c.number_input('Maximum fit RMSE (nA)',0.01,100.0,10.0,step=.05)
    eligible=(df.samples>=min_samples)&(df.fit_rmse_nA<=max_rmse)
    signature=(digest,bins,k,min_samples,max_rmse)
    st.write(f'{eligible.sum()} eligible; {(~eligible).sum()} excluded by current settings. Fractions below use eligible events only.')
    st.caption('Sample counts do not establish independent temporal resolution. Adjusting the RMSE gate can select particular shapes; retain excluded events for review.')
    if st.button('Run exploratory grouping',type='primary'):
        try:
            labs,centers,sil,ari=cluster_profiles(profiles[eligible],k)
            result=df.copy();result['group']=-1;result.loc[eligible,'group']=labs
            st.session_state.group_result=(signature,result,centers,sil,ari)
        except ValueError as ex:st.error(str(ex))
    saved=st.session_state.get('group_result')
    if saved and saved[0]==signature:
        _,result,centers,sil,ari=saved
        st.write(f'Silhouette: {sil:.3f} · Agreement across two initialisations (ARI): {ari:.3f}')
        st.caption('These measure geometric separation and initialisation sensitivity; they are not physical validation or a full robustness assessment.')
        f=go.Figure()
        for j,row in enumerate(centers):f.add_trace(go.Scatter(x=(np.arange(bins)+.5)/bins,y=row,name=f'Group {j}'))
        f.update_layout(xaxis_title='Fraction of event duration',yaxis_title='Mean blockade (nA)')
        st.plotly_chart(f,width='stretch')
        counts=result[result.group>=0].group.value_counts().sort_index().rename('events').to_frame();counts['fraction_eligible']=counts.events/counts.events.sum();st.dataframe(counts)
        gid=st.selectbox('Inspect group',range(k));members=result[result.group==gid]
        st.write('Three nearest-to-centre events and up to three random members:')
        positions=np.flatnonzero(result.group.to_numpy()==gid)
        nearest=positions[np.argsort(np.linalg.norm(profiles[positions]-centers[gid],axis=1))[:3]]
        random=np.random.default_rng(42).choice(positions,min(3,len(positions)),replace=False)
        for pos in dict.fromkeys(np.r_[nearest,random].tolist()):
            ev=events[pos];st.write(f'Event {ev.index}');draw(ev,f'group_{ev.index}')
    elif saved:st.info('Settings changed. Run grouping again to refresh results.')
with export:
    result=df.copy();saved=st.session_state.get('group_result')
    valid=saved is not None and saved[0]==signature
    if valid:result=saved[1]
    metadata={'app_version':'0.1.0','source_name':upload.name if upload else 'synthetic','sha256':digest,'bins':bins,'recording_settings':{k:v for k,v in settings.items() if k!='file_name'},'grouping_current':valid,'group_count':k if valid else None,'min_samples':min_samples,'max_fit_rmse_nA':max_rmse,'rejected_events':rejected,'method':'k-means on time-normalized measured blockade; common amplitude scaling','interpretation':'Exploratory shape groups, not topology labels','units':{'time':'seconds in source; milliseconds in tables','current':'nA assumed'}}
    st.download_button('Download event results CSV',result.to_csv(index=False),'event_results.csv','text/csv')
    st.download_button('Download analysis settings JSON',json.dumps(metadata,indent=2),'analysis_settings.json','application/json')
    st.caption('Download both files to preserve the source identity, parameters and exclusion rules. Uploaded scientific data are not written to the repository by this app.')

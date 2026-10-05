"""Nanopore Shape Lab 0.4: explicit signal processing and lossless event subsets."""
RELEASE_VERSION='0.8.3'
from dataclasses import dataclass
import io, json, re, zipfile
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

VERSION='0.8.3'
@dataclass
class Event:
    index:int
    time:np.ndarray
    current:np.ndarray
    bounds:np.ndarray
    baseline:np.ndarray
    fit:object=None
    levels:object=None
    widths:object=None
    analysis:object=None
    baseline_source:str='saved'

def read_npz(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as a:
        if sum(v.file_size for v in a.infolist())>512*1024**2:raise ValueError('Expanded archive exceeds 512 MB.')
    return np.load(io.BytesIO(blob),allow_pickle=False)

def inventory(blob):
    rows=[]
    with read_npz(blob) as z:
        for key in z.files[:60]:
            try:a=z[key];rows.append({'key':key,'shape':str(a.shape),'dtype':str(a.dtype)})
            except ValueError:rows.append({'key':key,'shape':'object array','dtype':'unsupported: pickle disabled'})
        return rows,len(z.files)

def metadata(z):
    if 'settings' not in z:return {}
    try:return json.loads(str(z['settings'].item()))
    except (ValueError,TypeError,json.JSONDecodeError):return {}

def mask(e):return (e.time>=e.bounds[0])&(e.time<e.bounds[1])
def validate(e):
    for a in [e.time,e.current,e.bounds,e.baseline]:
        if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Expected finite numeric vectors')
    if len(e.time)<4 or len(e.time)!=len(e.current) or len(e.time)!=len(e.baseline):raise ValueError('Waveform lengths do not match')
    if np.any(np.diff(e.time)<=0) or len(e.bounds)!=2 or e.bounds[1]<=e.bounds[0]:raise ValueError('Invalid times or bounds')
    if e.bounds[0]<e.time[0] or e.bounds[1]>e.time[-1]+np.median(np.diff(e.time))*1.1 or mask(e).sum()<3:raise ValueError('Bounds outside saved trace or fewer than three samples')
    if e.fit is not None and (e.fit.shape!=e.time.shape or not np.isfinite(e.fit).all()):raise ValueError('Invalid fitted waveform')

def baseline_from_padding(t,y,bounds):
    pad=y[(t<bounds[0])|(t>=bounds[1])]
    if len(pad)<4:raise ValueError('No saved baseline and fewer than four padding samples')
    return np.full_like(y,np.median(pad))

def load_events(blob):
    events=[];rejected=[]
    with read_npz(blob) as z:
        if 'events' in z.files and 'sampling_rate' in z.files:
            return load_nanosense_data(blob)
        settings=metadata(z)
        ids=sorted({int(m.group(1)) for k in z.files if (m:=re.fullmatch(r'EVENT_DATA_(\d+)_part_0',k))})
        dense=not ids and all(k in z for k in ['time','current','bounds'])
        if dense:
            current=np.asarray(z['current'],float);time=np.asarray(z['time'],float);bounds=np.asarray(z['bounds'],float)
            if current.ndim!=2 or bounds.shape!=(len(current),2):raise ValueError('Dense layout needs current[event,sample] and bounds[event,2].')
            if time.shape!=current.shape and time.shape!=(current.shape[1],):raise ValueError('Dense time must be shared 1D or match current shape.')
            ids=list(range(len(current)))
        if not ids:raise ValueError('Unrecognised event layout. Supported: EVENT_DATA_<id>_part_0/1/2 (time/current/bounds), or numeric time/current/bounds arrays. Use the schema inspector; share an eventdata.npz sample if your layout differs.')
        for i in ids:
            try:
                if dense:
                    t=time.copy() if time.ndim==1 else time[i].copy();y=current[i].copy();b=bounds[i].copy()
                    fit=np.asarray(z['fit'][i],float) if 'fit' in z else None
                    base=np.asarray(z['baseline'],float) if 'baseline' in z else None
                    if base is not None:
                        if base.ndim==2:base=base[i]
                        elif base.ndim==0:base=np.full_like(y,base.item())
                        elif base.shape!=y.shape:raise ValueError('Dense baseline must be scalar, shared sample vector or match current.')
                    levels=widths=summary=None
                else:
                    t,y,b=[np.asarray(z[f'EVENT_DATA_{i}_part_{j}'],float) for j in range(3)]
                    def opt(k):return np.asarray(z[k],float) if k in z else None
                    fit=opt(f'EVENT_DATA_{i}_part_3');base=opt(f'EVENT_DATA_{i}_part_4')
                    levels=opt(f'SEGMENT_INFO_{i}_segment_mean_diffs');widths=opt(f'SEGMENT_INFO_{i}_segment_widths_time');summary=opt(f'EVENT_ANALYSIS_{i}')
                origin='saved' if base is not None else 'estimated from padding median'
                if base is None:base=baseline_from_padding(t,y,b)
                e=Event(i,t,y,b,base,fit,levels,widths,summary,origin);validate(e);events.append(e)
            except (KeyError,ValueError,IndexError) as ex:rejected.append({'event_index':i,'reason':str(ex)})
    if not events:raise ValueError(f'No valid events; first errors: {rejected[:3]}')
    return events,settings,rejected

DATASET_FEATURE_NAMES={
    0:'height',
    1:'fwhm',
    2:'height_at_fwhm',
    3:'area',
    4:'width',
    5:'skew',
    6:'kurtosis',
    7:'event_baseline_mean',
    8:'event_time',
    9:'event_time_alt',
}

DATASET_FEATURE_DESCRIPTIONS={
    0:'NanoSense event height / blockade feature.',
    1:'NanoSense FWHM event duration feature.',
    2:'NanoSense height-at-FWHM feature.',
    3:'NanoSense integrated event area feature.',
    4:'NanoSense event width / dwell-time feature.',
    5:'NanoSense event skewness feature.',
    6:'NanoSense event kurtosis feature.',
    7:'Mean event baseline; useful for QC but not recommended as a default clustering coordinate.',
    8:'Event-time / start-time bookkeeping column in the known NanoSense layout; excluded from clustering by default.',
    9:'Alternate/duplicate event-time-like bookkeeping column in some NanoSense exports; excluded from clustering by default.',
}

def dataset_feature_name(column):
    column=int(column)
    return DATASET_FEATURE_NAMES.get(column,f'X{column}')

def dataset_feature_label(column):
    column=int(column);name=dataset_feature_name(column)
    return f'X[:, {column}] · {name}'

def load_dataset(blob):
    with read_npz(blob) as z:
        if 'X' not in z:raise ValueError('Expected numeric X[event, feature] in dataset.npz.')
        x=np.asarray(z['X'],float)
        if x.ndim!=2 or not len(x):raise ValueError('X must be a nonempty 2D numeric table.')
        return x,metadata(z)

def link_dataset(events,x,start_column=8,tolerance=1e-7):
    if not 0<=start_column<x.shape[1]:raise ValueError('Invalid timestamp column')
    times=x[:,start_column];order=np.argsort(times);ordered=times[order];mapping={};status=[];used=set()
    for e in events:
        lo=np.searchsorted(ordered,e.bounds[0]-tolerance,'left');hi=np.searchsorted(ordered,e.bounds[0]+tolerance,'right')
        candidates=order[lo:hi]
        if len(candidates)==1 and int(candidates[0]) not in used:
            row=int(candidates[0]);mapping[e.index]=row;used.add(row);note='matched'
        else:row=None;note='no matching timestamp' if len(candidates)==0 else 'ambiguous/reused timestamp'
        status.append({'event_index':e.index,'start_s':e.bounds[0],'dataset_row':row,'status':note})
    return mapping,pd.DataFrame(status)

def profile(e,bins,fit=None):
    y=e.baseline-(e.current if fit is None else fit)
    return np.interp((np.arange(bins)+.5)/bins,(e.time-e.bounds[0])/np.diff(e.bounds)[0],y)

def aligned_event_profiles(events,pre_samples=100,post_samples=100,alignment='start',window_samples=None):
    """Return real-sample blockade traces on a common horizontal coordinate.

    ``alignment='start'`` reproduces the older view: the detected event start is
    placed after ``pre_samples`` and the window extends through the longest event
    plus ``post_samples``.

    ``alignment='midpoint'`` is the Hart-style display used by default in v0.8.3.
    Every detected event midpoint is placed at the centre of a fixed-length sample
    window (``window_samples``).  No event is stretched or compressed, so dwell-time
    differences remain visible while baseline is shown before and after the event.
    Samples falling outside the stored trace are left as NaN rather than invented.
    """
    events=list(events)
    pre_samples=int(pre_samples);post_samples=int(post_samples)
    if not events:raise ValueError('Need at least one event for aligned profiles.')
    if pre_samples<0 or post_samples<0:raise ValueError('Pre/post sample counts must be nonnegative.')
    starts=[];ends=[];lengths=[];dts=[]
    for e in events:
        m=mask(e);idx=np.flatnonzero(m)
        if not len(idx):raise ValueError(f'Event {e.index} has no samples inside its detected bounds.')
        starts.append(int(idx[0]));ends.append(int(idx[-1]));lengths.append(int(len(idx)));dts.append(float(np.median(np.diff(e.time))))
    dt_s=float(np.median(dts));sampling_hz=float(1./dt_s) if dt_s>0 else np.nan

    alignment=str(alignment).lower()
    if alignment=='midpoint':
        if window_samples is None:window_samples=max(200,int(np.ceil(np.percentile(lengths,95)))+200)
        total=max(20,int(window_samples));center=int(total//2)
        profiles=np.full((len(events),total),np.nan,float)
        aligned_starts=[];aligned_ends=[]
        for i,(e,start,end) in enumerate(zip(events,starts,ends)):
            blockade=np.asarray(e.baseline-e.current,float)
            midpoint=int(round((start+end)/2))
            source_left=midpoint-center;source_right=source_left+total
            src0=max(0,source_left);src1=min(len(blockade),source_right)
            if src1>src0:
                dst0=src0-source_left;dst1=dst0+(src1-src0)
                profiles[i,dst0:dst1]=blockade[src0:src1]
            aligned_starts.append(start-source_left);aligned_ends.append(end-source_left)
        return dict(
            profiles=profiles,
            data_index=np.arange(total,dtype=float),
            relative_index=np.arange(total,dtype=float)-center,
            time_ms=(np.arange(total,dtype=float)-center)*dt_s*1000.,
            event_midpoint_index=int(center),
            event_start_indices=np.asarray(aligned_starts,float),
            event_end_indices=np.asarray(aligned_ends,float),
            event_lengths_samples=np.asarray(lengths,int),
            coverage=np.sum(np.isfinite(profiles),axis=0),
            time_step_s=dt_s,
            sampling_rate_hz=sampling_hz,
            alignment='midpoint',
            window_samples=int(total),
            relative_dt_spread=float(np.max(np.abs(np.asarray(dts,float)-dt_s))/dt_s) if dt_s>0 else np.nan,
        )

    if alignment!='start':raise ValueError("alignment must be 'start' or 'midpoint'.")
    event_span=max(lengths);total=pre_samples+event_span+post_samples
    profiles=np.full((len(events),total),np.nan,float)
    for i,(e,start) in enumerate(zip(events,starts)):
        blockade=np.asarray(e.baseline-e.current,float)
        source_left=start-pre_samples;source_right=start+event_span+post_samples
        src0=max(0,source_left);src1=min(len(blockade),source_right)
        if src1<=src0:continue
        dst0=src0-source_left;dst1=dst0+(src1-src0)
        profiles[i,dst0:dst1]=blockade[src0:src1]
    return dict(
        profiles=profiles,
        data_index=np.arange(total,dtype=float),
        relative_index=np.arange(total,dtype=float)-pre_samples,
        time_ms=(np.arange(total,dtype=float)-pre_samples)*dt_s*1000.,
        event_start_index=int(pre_samples),
        event_lengths_samples=np.asarray(lengths,int),
        coverage=np.sum(np.isfinite(profiles),axis=0),
        time_step_s=dt_s,
        sampling_rate_hz=sampling_hz,
        alignment='start',
        relative_dt_spread=float(np.max(np.abs(np.asarray(dts,float)-dt_s))/dt_s) if dt_s>0 else np.nan,
    )

def cluster_median_profiles(profiles,labels,min_coverage_fraction=.5):
    """Pointwise cluster medians, suppressing regions with sparse trace coverage."""
    import warnings
    x=np.asarray(profiles,float);labels=np.asarray(labels)
    if x.ndim!=2 or len(x)!=len(labels):raise ValueError('Profile matrix and labels are misaligned.')
    groups=sorted(np.unique(labels).tolist());centers=[];coverage=[]
    for g in groups:
        members=x[labels==g]
        finite=np.sum(np.isfinite(members),axis=0);minimum=max(1,int(np.ceil(len(members)*float(min_coverage_fraction))))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',category=RuntimeWarning)
            center=np.nanmedian(members,axis=0)
        center[finite<minimum]=np.nan
        centers.append(center);coverage.append(finite)
    return np.asarray(centers,float),np.asarray(coverage,int)

def fit_metrics(e,fit):
    if fit is None:return dict(waveform_rmse_nA=np.nan,relative_rmse=np.nan,fit_bias_nA=np.nan,peak_error_nA=np.nan,area_error_pct=np.nan)
    m=mask(e);b=(e.baseline-e.current)[m];f=(e.baseline-fit)[m];r=np.sqrt(np.mean((b-f)**2));den=np.sqrt(np.mean(b*b))
    return dict(waveform_rmse_nA=float(r),relative_rmse=float(r/den) if den>1e-12 else np.nan,
                fit_bias_nA=float(np.mean(f-b)),peak_error_nA=float(f.max()-b.max()),
                area_error_pct=float(100*(f.sum()-b.sum())/abs(b.sum())) if abs(b.sum())>1e-12 else np.nan)

def describe(events,bins=32,refinements=None):
    rows=[];profiles=[];refinements=refinements or {}
    for e in events:
        m=mask(e);b=e.baseline-e.current;dt=float(np.median(np.diff(e.time)));duration=float(np.diff(e.bounds)[0]);noise=float(np.std(b[~m])) if (~m).sum()>3 else np.nan
        row=dict(event_index=e.index,start_s=e.bounds[0],duration_ms=duration*1000,samples=int(m.sum()),
                 mean_blockade_nA=float(b[m].mean()),peak_blockade_nA=float(b[m].max()),
                 ecd_nA_ms=float(np.trapezoid(np.interp(np.r_[e.bounds[0],e.time[m],e.bounds[1]],e.time,b),np.r_[e.bounds[0],e.time[m],e.bounds[1]])*1000),
                 padding_std_nA=noise,segments=len(e.levels) if e.levels is not None else np.nan,
                 time_step_us=dt*1e6,baseline_source=e.baseline_source,**fit_metrics(e,e.fit))
        if e.index in refinements:
            row.update({f'refined_{k}':v for k,v in fit_metrics(e,refinements[e.index]['fit']).items()})
            row['refined_segments']=len(refinements[e.index]['levels'])
        rows.append(row);profiles.append(profile(e,bins))
    return pd.DataFrame(rows),np.array(profiles)

def refine(e,method='Segment means',min_duration_us=25.,penalty=8.):
    m=mask(e);idx=np.flatnonzero(m);y=(e.baseline-e.current)[m];n=len(y);dt=float(np.median(np.diff(e.time)))
    if n>6000:raise ValueError('Refinement is limited to 6,000 event samples per event for responsiveness.')
    fitted=e.baseline.copy();params={}
    if method=='Segment means':
        if e.fit is None:raise ValueError('Segment-means refinement needs an existing fit. Choose PELT for an unfitted trace.')
        existing=(e.baseline-e.fit)[m];ends=np.r_[np.flatnonzero(np.abs(np.diff(existing))>1e-9)+1,n]
    elif method=='New levels (PELT)':
        import ruptures as rpt
        ms=max(2,int(np.ceil(min_duration_us*1e-6/dt)))
        pad=(e.baseline-e.current)[~m]
        noise=float(np.std(pad)) if len(pad)>3 else float(np.std(np.diff(y))/np.sqrt(2))
        noise=max(noise,float(np.std(y))*1e-3,1e-6)
        ends=[n] if n<2*ms else rpt.Pelt(model='l2',min_size=ms,jump=1).fit(y/noise).predict(pen=float(penalty)*np.log(max(n,2)))
        params=dict(min_duration_us=min_duration_us,penalty_multiplier=penalty,noise_scale_nA=noise)
    else:raise ValueError('Unknown refinement method')

    yf=np.empty(n);levels=[];widths=[];start=0
    for end in ends:
        level=float(y[start:end].mean());yf[start:end]=level;levels.append(level)
        left=e.bounds[0] if start==0 else e.time[idx[start]]
        right=e.bounds[1] if end==n else e.time[idx[end]]
        widths.append(float(right-left));start=end
    levels=np.array(levels);widths=np.array(widths)
    fitted[m]=e.baseline[m]-yf
    return dict(fit=fitted,levels=levels,widths=widths,method=method,parameters=params)

def cluster_profiles(profiles,k,shape_only=False):
    x=np.asarray(profiles,float).copy()
    if shape_only:x/=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-9)
    scale=max(float(np.median(np.max(np.abs(x),axis=1))),1e-9);scaled=x/scale
    if len(x)<=k or np.unique(x,axis=0).shape[0]<k:raise ValueError('Need more distinct eligible profiles than requested groups.')
    model=KMeans(n_clusters=k,n_init=20,random_state=42).fit(scaled)
    order=np.argsort(model.cluster_centers_.mean(axis=1));remap=np.empty(k,int);remap[order]=np.arange(k);labels=remap[model.labels_]
    if len(np.unique(labels))<2:raise ValueError('Fewer than two distinct groups found.')
    # Full metric capped by deterministic, cluster-stratified sample; avoid singleton-label sample failures.
    rng=np.random.default_rng(42);sel=[]
    for j in range(k):
        ix=np.flatnonzero(labels==j);sel.extend(rng.choice(ix,min(len(ix),max(2,1500//k)),replace=False))
    score=float(silhouette_score(scaled[sel],labels[sel])) if len(sel)>k else np.nan
    other=KMeans(n_clusters=k,n_init=20,random_state=43).fit_predict(scaled)
    pca=PCA(n_components=2).fit(scaled);embedding=pca.transform(scaled)
    return dict(labels=labels,centers=model.cluster_centers_[order]*scale,profiles=x,silhouette=score,
                ari=float(adjusted_rand_score(labels,other)),embedding=embedding,pca_variance=pca.explained_variance_ratio_.tolist())

def safe_settings(settings):return {k:v for k,v in settings.items() if k!='file_name'}
def npz_bytes(arrays):
    f=io.BytesIO();np.savez_compressed(f,**arrays);return f.getvalue()

def pack_events(events,settings,provenance,refinements=None):
    arrays={'settings':np.array(json.dumps(safe_settings(settings))), 'shape_lab_provenance':np.array(json.dumps(provenance)), 'original_event_indices':np.array([e.index for e in events],int)}
    refinements=refinements or {}
    for e in events:
        for j,a in [(0,e.time),(1,e.current),(2,e.bounds),(4,e.baseline)]:arrays[f'EVENT_DATA_{e.index}_part_{j}']=a
        if e.fit is not None:arrays[f'EVENT_DATA_{e.index}_part_3']=e.fit
        if e.analysis is not None:arrays[f'EVENT_ANALYSIS_{e.index}']=e.analysis
        if e.levels is not None:arrays[f'SEGMENT_INFO_{e.index}_segment_mean_diffs']=e.levels;arrays[f'SEGMENT_INFO_{e.index}_number_of_segments']=np.array([len(e.levels)])
        if e.widths is not None:arrays[f'SEGMENT_INFO_{e.index}_segment_widths_time']=e.widths
        arrays[f'SEGMENT_INFO_{e.index}_event_width']=np.diff(e.bounds)
        if e.index in refinements:
            ref=refinements[e.index]
            arrays[f'REFINED_{e.index}_current']=ref['fit'];arrays[f'REFINED_{e.index}_levels']=ref['levels'];arrays[f'REFINED_{e.index}_widths']=ref['widths']
            arrays[f'REFINED_{e.index}_metadata']=np.array(json.dumps({'method':ref['method'],'parameters':ref['parameters']}))
    return npz_bytes(arrays)

def cluster_bundle(events,table,settings,provenance,refinements=None,dataset=None,mapping=None,dataset_settings=None):
    mapping=mapping or {};f=io.BytesIO()
    with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('events.npz',pack_events(events,settings,provenance,refinements));z.writestr('event_results.csv',table.to_csv(index=False))
        z.writestr('analysis_settings.json',json.dumps(provenance,indent=2));z.writestr('README.txt','Original waveforms, IDs and fits are preserved in events.npz. REFINED_* arrays are optional candidate fits and do not replace originals. This numeric NPZ reloads in Shape Lab; external NanoSense import compatibility has not been established. dataset_subset.npz contains only uniquely timestamp-matched source X rows, unchanged. See CSV and metadata for population and settings.\n')
        if dataset is not None:
            ids=[e.index for e in events if e.index in mapping];rows=[mapping[i] for i in ids]
            z.writestr('dataset_subset.npz',npz_bytes({'X':dataset[rows],'original_dataset_rows':np.array(rows,int),'original_event_indices':np.array(ids,int),'settings':np.array(json.dumps(safe_settings(dataset_settings or {})))}))
    return f.getvalue()

def synthetic_demo():
    rng=np.random.default_rng(17);events=[]
    for i in range(80):
        t=np.arange(-.0005,.0015,5e-6);m=(t>=0)&(t<.001);p=t[m]/.001
        patterns=[np.ones(len(p)),np.where(p<.4,2.,1.),np.where((p>.3)&(p<.6),3.,1.),np.ones(len(p))*2]
        y0=np.zeros(len(t));y0[m]=patterns[i%4]*rng.uniform(.9,1.1)
        y=17-y0+rng.normal(0,.08,len(t));fit=17-y0
        cp=np.r_[0,np.flatnonzero(np.diff(y0[m]))+1,len(p)];levels=np.array([y0[m][a:b].mean() for a,b in zip(cp[:-1],cp[1:])]);widths=np.diff(cp)*5e-6
        events.append(Event(i,t+i*.01,y,np.array([i*.01,i*.01+.001]),np.full(len(t),17.),fit,levels,widths))
    return events,{'sampling_rate':200000,'source':'Synthetic demonstration'},[]

# NanoSense event_data exports contain NumPy object arrays. Only NumPy's array
# reconstruction primitives are accepted; arbitrary pickle globals stay blocked.
def read_nanosense_records(blob):
    import pickle
    class NumericUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module=='numpy' and name in ('ndarray','dtype'):return getattr(np,name)
            if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
                return getattr(np._core.multiarray,name)
            raise ValueError(f'Unsupported serialized type: {module}.{name}')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if sum(a.file_size for a in archive.infolist())>512*1024**2:raise ValueError('Expanded archive exceeds 512 MB.')
        f=io.BytesIO(archive.read('events.npy'));version=np.lib.format.read_magic(f)
        if version==(1,0):header=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):header=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Unsupported NumPy object-array header version')
        if not header[2].hasobject:raise ValueError('Expected NanoSense records object array')
        records=NumericUnpickler(f).load()
    if not isinstance(records,np.ndarray) or records.ndim!=1:raise ValueError('Expected one-dimensional event records')
    return records

def load_nanosense_data(blob):
    with read_npz(blob) as z:fs=float(z['sampling_rate'].item())
    if not np.isfinite(fs) or fs<=0:raise ValueError('Invalid sampling rate')
    records=read_nanosense_records(blob);events=[];rejected=[];used=set()
    for pos,r in enumerate(records):
        try:
            if not isinstance(r,dict):raise ValueError('Expected dictionary record')
            i=int(r['event_id']);start=float(r['start_time']);end=float(r['end_time']);delta=np.asarray(r['event_data'],float);base=float(r['baseline_value'])
            if i in used:raise ValueError('Duplicate event_id')
            if delta.ndim!=1:raise ValueError('event_data must be a vector')
            # In the inspected export, event_data is current minus baseline.
            # Event boundaries define a half-open interval; n/fs agrees with duration.
            if abs(len(delta)/fs-(end-start))>1.1/fs:raise ValueError('Length, duration and sampling rate disagree')
            t=start+np.arange(len(delta))/fs;e=Event(i,t,base+delta,np.array([start,end]),np.full(len(delta),base),baseline_source='saved baseline; cropped current deviation restored')
            validate(e);events.append(e);used.add(i)
        except (KeyError,ValueError,TypeError) as ex:rejected.append({'event_index':pos,'reason':str(ex)})
    if not events:raise ValueError(f'No valid NanoSense event_data records: {rejected[:3]}')
    return events,{'sampling_rate':fs,'layout':'NanoSense event_data records; current=baseline_value+event_data'},rejected

FEATURE_DESCRIPTIONS={
'duration_ms':'Detected end minus start, milliseconds; not FWHM.',
'mean_blockade_nA':'Mean of saved blockade samples within event.',
'peak_blockade_nA':'Maximum saved blockade within event.',
'blockade_std_nA':'Standard deviation within event (shape plus noise).',
'peak_position':'Location of highest sample as fraction of event duration.',
'fwhm_ms':'Width of contiguous above-half-maximum region around largest peak; linearly interpolated crossings.',
'early_late_difference_nA':'Mean blockade in first half minus second half.',
'wavelet_approx_mean':'Mean Haar approximation coefficient, level up to two.',
'wavelet_approx_std':'Std of Haar approximation coefficients, level up to two.',
'wavelet_detail_energy':'Sum of squared detail coefficients divided by event sample count.'}

def feature_table(events):
    import pywt
    rows=[]
    for e in events:
        m=mask(e);y=(e.baseline-e.current)[m];t=e.time[m];dur=float(np.diff(e.bounds)[0]);peak=int(np.argmax(y));threshold=y[peak]/2
        above=y>=threshold;left=peak;right=peak
        while left>0 and above[left-1]:left-=1
        while right<len(y)-1 and above[right+1]:right+=1
        ltime=float(e.bounds[0]) if left==0 else float(np.interp(threshold,[y[left-1],y[left]],[t[left-1],t[left]]))
        rtime=float(e.bounds[1]) if right==len(y)-1 else float(np.interp(threshold,[y[right+1],y[right]],[t[right+1],t[right]]))
        level=min(2,pywt.dwt_max_level(len(y),pywt.Wavelet('haar').dec_len));coefs=pywt.wavedec(y,'haar',mode='symmetric',level=level);a=coefs[0]
        rows.append(dict(event_index=e.index,duration_ms=dur*1000,mean_blockade_nA=float(y.mean()),peak_blockade_nA=float(y.max()),
             blockade_std_nA=float(y.std()),peak_position=float((t[peak]-e.bounds[0])/dur),fwhm_ms=max(0,rtime-ltime)*1000,
             early_late_difference_nA=float(y[:max(1,len(y)//2)].mean()-y[len(y)//2:].mean()),
             wavelet_approx_mean=float(a.mean()),wavelet_approx_std=float(a.std()),wavelet_detail_energy=float(sum(np.sum(c*c) for c in coefs[1:])/len(y))))
    return pd.DataFrame(rows)

def _feature_space(features,n_components=2,correlation_threshold=None):
    """Build the PCA space from the original dataset feature matrix.

    Version 0.8 intentionally returns to the early analysis strategy: selected
    NanoSense ``dataset.npz`` columns are scaled independently to [-1, 1] with
    ``MinMaxScaler`` and then projected by PCA. Constant columns are removed.
    Near-duplicate correlation pruning is optional and is OFF by default so the
    original dataset descriptors are preserved unless the user explicitly asks
    to prune them.
    """
    from sklearn.preprocessing import MinMaxScaler
    x=np.asarray(features,float)
    if x.ndim!=2 or len(x)<3:raise ValueError('Feature matrix must contain at least three events.')
    if not np.isfinite(x).all():raise ValueError('All selected clustering features must be finite.')

    minimum=np.min(x,axis=0);maximum=np.max(x,axis=0);span=maximum-minimum;std=np.std(x,axis=0)
    keep=span>1e-12
    dropped_constant=np.flatnonzero(~keep).tolist();dropped_correlated=[]
    if correlation_threshold is not None and keep.sum()>1:
        idx=np.flatnonzero(keep);corr=np.corrcoef(x[:,idx],rowvar=False);chosen=[]
        for j in range(len(idx)):
            if not chosen or all(np.isfinite(corr[j,h]) and abs(corr[j,h])<float(correlation_threshold) for h in chosen):
                chosen.append(j)
            else:
                dropped_correlated.append(int(idx[j]))
        reduced=np.zeros_like(keep);reduced[idx[chosen]]=True;keep=reduced
    if keep.sum()<1:raise ValueError('No non-constant dataset feature remains for PCA.')

    retained=np.flatnonzero(keep)
    scaler=MinMaxScaler(feature_range=(-1.,1.));scaled=scaler.fit_transform(x[:,retained])
    full=PCA().fit(scaled);max_nc=min(scaled.shape[1],len(scaled)-1)
    nc=max(1,min(int(n_components or 2),max_nc));transformed=full.transform(scaled);coords=transformed[:,:nc]

    stats=[]
    for j in range(x.shape[1]):
        if j in dropped_constant:reason='constant'
        elif j in dropped_correlated:reason=f'near-duplicate (|r| >= {float(correlation_threshold):.2f})'
        else:reason='retained'
        stats.append(dict(feature_index=int(j),minimum=float(minimum[j]),maximum=float(maximum[j]),range=float(span[j]),
                          standard_deviation=float(std[j]),status=reason))
    return dict(raw=x,keep=keep,scaled=scaled,scaler=scaler,pca=full,n_components=nc,coords=coords,full_coords=transformed,
                dropped_constant=dropped_constant,dropped_low_spread=[],dropped_correlated=dropped_correlated,
                correlation_threshold=correlation_threshold,feature_stats=stats,scaling_method='MinMaxScaler [-1, 1]')


def feature_space_diagnostics(features,feature_names=None,correlation_threshold=None):
    """Return dataset-feature correlation and PCA diagnostics without assigning clusters."""
    x=np.asarray(features,float)
    names=list(feature_names or [f'feature_{i}' for i in range(x.shape[1])])
    space=_feature_space(x,n_components=max(1,x.shape[1]),correlation_threshold=correlation_threshold)
    retained=[n for n,k in zip(names,space['keep']) if k]
    dropped_constant=[names[i] for i in space['dropped_constant']]
    dropped_correlated=[names[i] for i in space['dropped_correlated']]
    raw_corr=np.corrcoef(x,rowvar=False) if x.shape[1]>1 else np.array([[1.]])
    spread_table=[]
    for name,row in zip(names,space['feature_stats']):
        spread_table.append({'Feature':name,'Minimum':row['minimum'],'Maximum':row['maximum'],'Range':row['range'],
                             'Std':row['standard_deviation'],'Status':row['status']})
    return dict(feature_names=names,retained_features=retained,dropped_constant=dropped_constant,dropped_low_spread=[],
        dropped_correlated=dropped_correlated,keep_mask=space['keep'].tolist(),correlation=raw_corr.tolist(),
        scree=space['pca'].explained_variance_ratio_.tolist(),cumulative=np.cumsum(space['pca'].explained_variance_ratio_).tolist(),
        max_components=int(space['full_coords'].shape[1]),spread_table=spread_table,scaling_method=space['scaling_method'])


def _ward_linkage(coords):
    """Build a SciPy-compatible Ward linkage matrix from sklearn's full tree."""
    from sklearn.cluster import AgglomerativeClustering
    model=AgglomerativeClustering(distance_threshold=0,n_clusters=None,linkage='ward',compute_distances=True).fit(coords)
    counts=np.zeros(model.children_.shape[0],float);n=len(model.labels_)
    for i,merge in enumerate(model.children_):
        c=0.
        for child in merge:
            c+=1. if child<n else counts[int(child)-n]
        counts[i]=c
    return np.column_stack([model.children_,model.distances_,counts]).astype(float)


def _cut_ward(linkage_matrix,k):
    from scipy.cluster.hierarchy import cut_tree
    return cut_tree(linkage_matrix,n_clusters=[int(k)]).reshape(-1).astype(int)


def _cluster_coords(coords,k,method,random_state=42,linkage_matrix=None):
    if len(coords)<=k or np.unique(coords,axis=0).shape[0]<k:raise ValueError('Too few distinct PCA coordinates for this k.')
    if 'agglomerative' in method.lower():
        linkage_matrix=_ward_linkage(coords) if linkage_matrix is None else linkage_matrix
        return _cut_ward(linkage_matrix,k),linkage_matrix,None
    model=KMeans(n_clusters=k,n_init=20,random_state=random_state).fit(coords)
    return model.labels_.astype(int),None,model.cluster_centers_.copy()


def _dispersion(coords,labels):
    return float(sum(np.sum((coords[labels==j]-coords[labels==j].mean(axis=0))**2) for j in np.unique(labels)))


def _elbow_from_dispersion(ks,values):
    """Simple scale-free knee estimate: largest departure below endpoint chord."""
    ks=np.asarray(ks,float);y=np.asarray(values,float)
    if len(ks)<3 or np.ptp(y)<=1e-15:return int(ks[np.argmin(y)]),0.
    x=(ks-ks.min())/max(np.ptp(ks),1e-12);yn=(y-y.min())/np.ptp(y)
    chord=1.-x;distance=chord-yn
    j=int(np.argmax(distance));return int(ks[j]),float(max(distance[j],0.))


def cluster_count_diagnostics(features,method='PCA + agglomerative',n_components=2,k_min=2,k_max=8,feature_names=None,correlation_threshold=None,random_state=42):
    """Hart-style k scan: within-cluster dispersion ('elbow') plus silhouette.

    Calinski-Harabasz and Davies-Bouldin are retained as secondary diagnostics, but
    the main visual decision mirrors the paper: inspect the elbow and silhouette together.
    """
    x=np.asarray(features,float);k_min=max(2,int(k_min));k_max=min(int(k_max),len(x)-1)
    if k_max<k_min:raise ValueError('Not enough events for the requested cluster-count range.')
    space=_feature_space(x,n_components=n_components,correlation_threshold=correlation_threshold);coords=space['coords']
    linkage_matrix=_ward_linkage(coords) if 'agglomerative' in method.lower() else None
    rows=[]
    for k in range(k_min,k_max+1):
        try:
            labels,_,_= _cluster_coords(coords,k,method,random_state,linkage_matrix)
            counts=np.bincount(labels,minlength=k)
            rows.append(dict(k=k,dispersion=_dispersion(coords,labels),
                silhouette=float(silhouette_score(coords,labels,sample_size=min(2000,len(coords)),random_state=random_state)),
                calinski_harabasz=float(calinski_harabasz_score(coords,labels)),
                davies_bouldin=float(davies_bouldin_score(coords,labels)),
                min_cluster_size=int(counts.min()),min_cluster_fraction=float(counts.min()/len(coords))))
        except ValueError:pass
    if not rows:raise ValueError('No valid cluster counts could be evaluated.')
    d=pd.DataFrame(rows).sort_values('k').reset_index(drop=True)
    elbow_k,elbow_strength=_elbow_from_dispersion(d.k,d.dispersion)
    silhouette_k=int(d.loc[d.silhouette.idxmax(),'k'])
    agreement=elbow_k==silhouette_k
    suggested_k=elbow_k if agreement else silhouette_k
    names=list(feature_names or [f'feature_{i}' for i in range(x.shape[1])])
    return dict(table=d.to_dict('records'),elbow_k=int(elbow_k),silhouette_k=int(silhouette_k),suggested_k=int(suggested_k),
                elbow_strength=float(elbow_strength),agreement=bool(agreement),scree=space['pca'].explained_variance_ratio_.tolist(),
                cumulative=np.cumsum(space['pca'].explained_variance_ratio_).tolist(),feature_keep_mask=space['keep'].tolist(),
                retained_features=[n for n,k in zip(names,space['keep']) if k],dropped_constant=[names[i] for i in space['dropped_constant']],
                dropped_correlated=[names[i] for i in space['dropped_correlated']],n_components=space['n_components'])


def _ordered_feature_result(space,profiles,k,method,random_state=42):
    coords=space['coords'];labels,linkage_matrix,pca_centers=_cluster_coords(coords,k,method,random_state)
    # Representative waveform = pointwise median of every duration-normalised member profile.
    raw_centers=np.array([np.median(profiles[labels==j],axis=0) for j in range(k)])
    # Stable display IDs: shallowest median waveform becomes Cluster 0.
    order=np.argsort(raw_centers.mean(axis=1));remap=np.empty(k,int);remap[order]=np.arange(k)
    labels=remap[labels];centers=raw_centers[order]
    if pca_centers is not None:pca_centers=pca_centers[order]
    sil=float(silhouette_score(coords,labels,sample_size=min(2000,len(coords)),random_state=42))
    ch=float(calinski_harabasz_score(coords,labels));db=float(davies_bouldin_score(coords,labels));dispersion=_dispersion(coords,labels)
    ari=np.nan
    if 'k-means' in method.lower():
        second,_,_=_cluster_coords(coords,k,method,random_state+1);ari=float(adjusted_rand_score(labels,second))
    full=space['pca'];scaler=space['scaler'];full_coords=space['full_coords']
    embedding=full_coords[:,:2] if full_coords.shape[1]>=2 else np.c_[full_coords[:,0],np.zeros(len(full_coords))]
    embedding3=full_coords[:,:3] if full_coords.shape[1]>=3 else None
    data_min=np.asarray(scaler.data_min_,float);data_max=np.asarray(scaler.data_max_,float)
    center=(data_min+data_max)/2.;scale=np.maximum((data_max-data_min)/2.,1e-12)
    return dict(labels=labels,centers=centers,profiles=np.asarray(profiles,float),silhouette=sil,calinski_harabasz=ch,davies_bouldin=db,ari=ari,
       embedding=embedding,embedding3=embedding3,pca_variance=full.explained_variance_ratio_[:min(3,len(full.explained_variance_ratio_))].tolist(),
       scree=full.explained_variance_ratio_.tolist(),cumulative=np.cumsum(full.explained_variance_ratio_).tolist(),
       loadings=full.components_[:space['n_components']].tolist(),feature_keep_mask=space['keep'].tolist(),
       feature_center=center.tolist(),feature_mean=center.tolist(),feature_scale=scale.tolist(),
       feature_min=data_min.tolist(),feature_max=data_max.tolist(),n_components=space['n_components'],dispersion=dispersion,
       linkage_matrix=linkage_matrix,pca_centers=pca_centers,dropped_constant=space['dropped_constant'],dropped_correlated=space['dropped_correlated'],
       center_note='Pointwise median duration-normalised waveforms of dataset-feature cluster members; not reconstructed PCA centroids.',
       metric='Euclidean in MinMax[-1,1]-scaled retained PCA space')


def cluster_features(features,profiles,k,n_components=2,method='PCA + agglomerative',correlation_threshold=None):
    """Cluster selected dataset features after min-max scaling and PCA."""
    space=_feature_space(features,n_components=n_components,correlation_threshold=correlation_threshold)
    return _ordered_feature_result(space,profiles,k,method)


def auto_cluster_features(features,profiles,method='PCA + agglomerative',k_min=2,k_max=8,variance_threshold=.95,stability_repeats=6,stability_fraction=.8,random_state=42):
    """Backward-compatible wrapper retained for older tests/scripts.

    Version 0.6 uses the Hart-style elbow + silhouette scan in the UI.  This wrapper
    chooses the scan's suggestion and then returns the corresponding clustering.
    ``variance_threshold`` is translated to the smallest PCA dimension reaching the
    requested cumulative variance.
    """
    x=np.asarray(features,float)
    diag0=feature_space_diagnostics(x,correlation_threshold=.98)
    cum=np.asarray(diag0['cumulative']);npc=int(np.searchsorted(cum,float(variance_threshold))+1);npc=max(1,npc)
    scan=cluster_count_diagnostics(x,method,npc,k_min,k_max,correlation_threshold=.98,random_state=random_state)
    result=cluster_features(x,profiles,scan['suggested_k'],npc,method,.98)
    result.update(selected_k=int(scan['suggested_k']),selection_table=scan['table'],selection_method='Hart-style elbow + silhouette scan',
                  elbow_k=scan['elbow_k'],silhouette_k=scan['silhouette_k'],selection_warning='' if scan['agreement'] else f'Elbow suggests k={scan["elbow_k"]} while silhouette peaks at k={scan["silhouette_k"]}; inspect both solutions.')
    return result

def cluster_dtw(profiles,k,shape_only=False,radius=4):
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_dtw
    if len(profiles)>2000 or profiles.shape[1]>64:raise ValueError('DTW mode supports up to 2,000 eligible events and 64 positions; use feature methods for larger files.')
    x=np.asarray(profiles,float).copy()
    if shape_only:x/=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-9)
    if len(x)<=k or np.unique(x,axis=0).shape[0]<k:raise ValueError('Need more distinct profiles than groups.')
    model=TimeSeriesKMeans(n_clusters=k,metric='dtw',metric_params={'global_constraint':'sakoe_chiba','sakoe_chiba_radius':radius},n_init=1,max_iter=15,max_iter_barycenter=10,random_state=42,n_jobs=1).fit(x[:,:,None])
    labels=model.labels_;rng=np.random.default_rng(42);sel=[]
    for j in np.unique(labels):
        ids=np.flatnonzero(labels==j);sel.extend(rng.choice(ids,min(len(ids),max(2,200//k)),replace=False))
    distances=cdist_dtw(x[sel,:,None],global_constraint='sakoe_chiba',sakoe_chiba_radius=radius,n_jobs=1);np.fill_diagonal(distances,0)
    sil=float(silhouette_score(distances,labels[sel],metric='precomputed')) if len(np.unique(labels[sel]))>1 and len(sel)>len(np.unique(labels[sel])) else np.nan
    pca=PCA(n_components=2).fit(x)
    return dict(labels=labels,centers=model.cluster_centers_[:,:,0],profiles=x,silhouette=sil,ari=np.nan,embedding=pca.transform(x),pca_variance=pca.explained_variance_ratio_.tolist(),
        center_note='DTW barycentres; percentile bands describe original unwarped profiles and are not uncertainty on these centres.',metric='Constrained DTW; silhouette on at most about 200 stratified events',dtw_radius=radius)

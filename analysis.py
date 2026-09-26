"""Exploratory shape analysis. No automatic molecular topology assignments."""
import io, json, re, zipfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

@dataclass
class Event:
    index: int
    time: np.ndarray
    current: np.ndarray
    bounds: np.ndarray
    fit: np.ndarray
    baseline: np.ndarray
    levels: np.ndarray
    widths: np.ndarray


def load_export(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as arc:
        if sum(x.file_size for x in arc.infolist()) > 512 * 1024**2:
            raise ValueError('Expanded archive exceeds 512 MB. Split the export first.')
    events, rejected = [], []
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        settings = json.loads(str(z['settings'].item())) if 'settings' in z else {}
        ids = sorted(int(m.group(1)) for k in z.files if (m := re.fullmatch(r'EVENT_ANALYSIS_(\d+)', k)))
        if not ids:
            raise ValueError('Upload an event_fitting.npz file. The dataset.npz summary has no waveforms.')
        for i in ids:
            try:
                parts = [np.asarray(z[f'EVENT_DATA_{i}_part_{j}'], dtype=float) for j in range(5)]
                levels = np.asarray(z[f'SEGMENT_INFO_{i}_segment_mean_diffs'], dtype=float)
                widths = np.asarray(z[f'SEGMENT_INFO_{i}_segment_widths_time'], dtype=float)
                t, y, bounds, fit, base = parts
                if any(a.ndim != 1 or not np.isfinite(a).all() for a in parts + [levels, widths]):
                    raise ValueError('Nonfinite or non-vector data')
                if not (len(t) == len(y) == len(fit) == len(base)) or len(bounds) != 2:
                    raise ValueError('Array lengths do not match')
                if len(t)<4 or np.any(np.diff(t)<=0) or bounds[1]<=bounds[0]:
                    raise ValueError('Invalid time axis')
                if bounds[0]<t[0] or bounds[1]>t[-1] or np.sum((t>=bounds[0])&(t<bounds[1]))<3:
                    raise ValueError('Event bounds outside trace or too few event samples')
                if len(levels)!=len(widths) or not len(levels) or np.any(widths<=0):
                    raise ValueError('Invalid segment data')
                events.append(Event(i,t,y,bounds,fit,base,levels,widths))
            except (KeyError,ValueError) as ex:
                rejected.append({'event_index':i,'reason':str(ex)})
    if not events: raise ValueError('No valid events found.')
    return events, settings, rejected


def describe(events, bins=32):
    rows, profiles = [], []
    for e in events:
        m=(e.time>=e.bounds[0])&(e.time<e.bounds[1]); outside=~m
        b=e.baseline-e.current; f=e.baseline-e.fit
        # Padding may be correlated/filtered; this is a diagnostic scale, not independent noise.
        noise=float(np.std(b[outside])) if outside.sum()>2 else np.nan
        rmse=float(np.sqrt(np.mean((b[m]-f[m])**2)))
        phase=(e.time-e.bounds[0])/(e.bounds[1]-e.bounds[0])
        grid=(np.arange(bins)+.5)/bins
        profiles.append(np.interp(grid,phase,b))
        dt=np.median(np.diff(e.time)); duration=e.bounds[1]-e.bounds[0]
        rows.append(dict(event_index=e.index,start_s=e.bounds[0],duration_ms=duration*1000,
            mean_blockade_nA=b[m].mean(),peak_blockade_nA=b[m].max(),
            fit_rmse_nA=rmse,padding_std_nA=noise,
            fit_rmse_over_padding_std=rmse/max(noise,1e-9),
            fit_bias_nA=np.mean(f[m]-b[m]),segments=len(e.levels),
            min_segment_us=e.widths.min()*1e6,
            segment_duration_error_us=(e.widths.sum()-duration)*1e6,
            samples=int(m.sum()),time_step_us=dt*1e6,
            ecd_nA_ms=float(b[m].sum()*dt*1000)))
    return pd.DataFrame(rows),np.array(profiles)


def cluster_profiles(profiles,k,seed=42):
    # A single shared scale preserves absolute depth differences between events.
    scale=max(float(np.median(np.max(np.abs(profiles),axis=1))),1e-9)
    x=profiles/scale
    if k>=len(x) or np.unique(x,axis=0).shape[0]<k:
        raise ValueError('Need more distinct events than groups.')
    model=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(x)
    # Relabel by centroid mean for readability; labels have no topology meaning.
    order=np.argsort(model.cluster_centers_.mean(axis=1)); remap=np.empty(k,int);remap[order]=np.arange(k)
    labels=remap[model.labels_]
    score=silhouette_score(x,labels,sample_size=min(1500,len(x)),random_state=seed)
    other=KMeans(n_clusters=k,n_init=20,random_state=seed+1).fit_predict(x)
    return labels,model.cluster_centers_[order]*scale,float(score),float(adjusted_rand_score(labels,other))


def synthetic_demo():
    """Deliberately idealised signals for learning the interface, not validation data."""
    rng=np.random.default_rng(17);events=[]
    for i in range(80):
        t=np.arange(-.0002,.0012,5e-6);m=(t>=0)&(t<.001);p=t[m]/.001
        patterns=[np.ones(len(p)),np.where(p<.4,2.,1.),np.where((p>.3)&(p<.6),3.,1.),np.ones(len(p))*2]
        truth=np.zeros(len(t));truth[m]=patterns[i%4]*rng.uniform(.9,1.1)
        y=17-truth+rng.normal(0,.08,len(t));fit=17-truth
        cp=np.r_[0,np.flatnonzero(np.diff(truth[m]))+1,len(p)]
        levels=np.array([truth[m][a:b].mean() for a,b in zip(cp[:-1],cp[1:])]);widths=np.diff(cp)*5e-6
        events.append(Event(i,t,y,np.array([0.,.001]),fit,np.full(len(t),17.),levels,widths))
    return events,{'sampling_rate':200000,'source':'Synthetic demonstration; idealised occupancy patterns'},[]

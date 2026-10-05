"""Explicit waveform representations and repeated-subsample checks.

These are exploratory alternatives, not a reproduction of NanoBoost's DWT features.
"""
RELEASE_VERSION='0.8.0'
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from analysis import _feature_space,_cluster_coords


def phase_average(e,source='Selected fits',bins=32):
    """Bin means from piecewise-linear integration, preserving full event area.

    Fixed relative-time bins avoid interpolation at isolated sample positions.
    No temporal warping, event alignment shifts or per-event smoothing is fitted.
    """
    current=e.fit if source=='Selected fits' else e.current
    if current is None:raise ValueError('Selected fit is missing; use measured trace or refine.')
    start,end=map(float,e.bounds)
    if end<=start:raise ValueError('Event duration is not positive.')
    t=np.asarray(e.time,float);b=np.asarray(e.baseline-current,float)
    inside=(t>start)&(t<end);knots=np.r_[start,t[inside],end]
    values=np.interp(knots,t,b)
    if not np.isfinite(values).all():raise ValueError('Non-finite waveform values.')
    widths=np.diff(knots);slope=np.diff(values)/widths
    integrals=np.r_[0.,np.cumsum(widths*(values[:-1]+values[1:])*.5)]
    edges=np.linspace(start,end,bins+1)
    j=np.clip(np.searchsorted(knots,edges,side='right')-1,0,len(knots)-2)
    offset=edges-knots[j]
    areas=integrals[j]+values[j]*offset+.5*slope[j]*offset**2
    return np.diff(areas)/np.diff(edges)


def waveform_features(events,source='Selected fits',shape_only=False):
    rows=[];vectors=[];positions=[]
    for pos,e in enumerate(events):
        row={'event_index':e.index,'representation_eligible':False,'representation_exclusion_reason':''}
        try:
            profile=phase_average(e,source,32);mean=float(profile.mean());duration=float(np.diff(e.bounds)[0])*1000
            if mean<=1e-12:raise ValueError('Non-positive mean blockade; cannot normalise waveform.')
            vector=profile/mean
            if not shape_only:vector=np.r_[vector,np.log10(mean),np.log10(duration)]
            if not np.isfinite(vector).all():raise ValueError('Non-finite waveform features.')
            vectors.append(vector);positions.append(pos)
            row.update(representation_eligible=True,source_mean_blockade_nA=mean,duration_ms=duration)
        except ValueError as ex:row['representation_exclusion_reason']=str(ex)
        rows.append(row)
    names=[f'shape_phase_{j+1:02d}' for j in range(32)]
    if not shape_only:names+=['log10_source_mean_blockade_nA','log10_duration_ms']
    return pd.DataFrame(rows),np.asarray(positions,int),np.asarray(vectors,float).reshape(-1,len(names)),names


def subsample_stability(features,labels,k,n_components,method,correlation_threshold=None,scaling='standard',repeats=6,fraction=.8,random_state=42):
    """Refit preprocessing and clustering on seeded subsamples, compare common IDs.

    ARI here measures agreement with the full-data assignment on the sampled
    events. It is not held-out prediction accuracy or evidence of topology.
    """
    x=np.asarray(features,float);labels=np.asarray(labels);rng=np.random.default_rng(random_state)
    if k==1:return {'mean_ari':None,'sd_ari':None,'sample_size':0,'repeats':[],'warning':'One-group agreement is trivial; no stability score reported.'}
    size=min(1000,max(k+2,int(np.floor(len(x)*fraction))))
    if size>=len(x):return {'mean_ari':None,'sd_ari':None,'sample_size':size,'repeats':[],'warning':'Too few events for a smaller subsample with this k.'}
    rows=[]
    for repeat in range(repeats):
        ids=np.sort(rng.choice(len(x),size,replace=False))
        try:
            space=_feature_space(x[ids],n_components,correlation_threshold,scaling)
            prediction,_,_=_cluster_coords(space['coords'],k,method,random_state+repeat+1)
            score=float(adjusted_rand_score(labels[ids],prediction))
            rows.append({'repeat':repeat+1,'ARI':score,'events':size,'status':'ok'})
        except ValueError as ex:rows.append({'repeat':repeat+1,'ARI':np.nan,'events':size,'status':str(ex)})
    scores=np.array([r['ARI'] for r in rows]);scores=scores[np.isfinite(scores)]
    return {'mean_ari':float(scores.mean()) if len(scores) else None,'sd_ari':float(scores.std()) if len(scores) else None,
            'sample_size':size,'effective_fraction':size/len(x),'repeats':rows,
            'warning':'Some repeats failed; inspect the table.' if len(scores)<repeats else ''}

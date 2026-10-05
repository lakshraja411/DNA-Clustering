"""Explicit fit selection and roundtrip exports for the guided workflow."""
RELEASE_VERSION='0.9.0'
from dataclasses import replace
import io,json,zipfile
import numpy as np
from analysis import pack_events,npz_bytes,link_dataset,safe_settings

def active_events(events,refs):
    return [replace(e,fit=refs[e.index]['fit'].copy(),levels=refs[e.index]['levels'].copy(),widths=refs[e.index]['widths'].copy(),analysis=None) if e.index in refs else e for e in events]

def signal_events(events,source):
    if source=='Measured trace':return events
    missing=[e.index for e in events if e.fit is None]
    if missing:raise ValueError(f'{len(missing)} events have no selected fit. Refine them or use measured traces.')
    return [replace(e,current=e.fit.copy()) for e in events]

def fitting_bytes(events,refs,settings,meta):
    active=active_events(events,refs)
    with np.load(io.BytesIO(pack_events(active,settings,meta)),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    for e in events:
        if e.fit is not None:arrays[f'INPUT_FIT_{e.index}']=e.fit
        if e.levels is not None:arrays[f'INPUT_LEVELS_{e.index}']=e.levels
        if e.widths is not None:arrays[f'INPUT_WIDTHS_{e.index}']=e.widths
    arrays['refined_event_indices']=np.array(sorted(set(refs)&{e.index for e in events}),int)
    return npz_bytes(arrays)

def match_raw(events,raw,tolerance):
    x=np.array([[e.bounds[0]] for e in raw]);mapping,status=link_dataset(events,x,0,tolerance)
    return {i:raw[row] for i,row in mapping.items()},status

def bundle(events,rawmap,refs,dataset,mapping,settings,dsettings,table,meta,sequences=None):
    ids={e.index for e in events};rows=[mapping[e.index] for e in events if e.index in mapping];matched=[e.index for e in events if e.index in mapping]
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('selected.eventfitting.npz',fitting_bytes(events,refs,settings,meta))
        raw=[rawmap[e.index] for e in events if e.index in rawmap]
        z.writestr('selected.eventdata.npz',pack_events(raw,{},meta))
        z.writestr('selected.dataset.npz',npz_bytes({'X':dataset[rows],'original_dataset_rows':np.array(rows,int),'fitting_event_ids':np.array(matched,int),'settings':np.array(json.dumps(safe_settings(dsettings)))}))
        z.writestr('event_results.csv',table.to_csv(index=False));z.writestr('provenance.json',json.dumps(meta,indent=2))
        if sequences is not None and len(sequences):z.writestr('resolved_levels.csv',sequences[sequences.event_index.isin([e.index for e in events])].to_csv(index=False))
        z.writestr('README.txt','Selected fits are the active fits in selected.eventfitting.npz; INPUT_FIT_* preserves uploaded fits. Measured traces are unchanged. selected.eventdata.npz uses the numeric Shape Lab layout and preserves source raw-event IDs. Dataset X rows are original and are NOT recalculated from refined fits. CSV contains current measured results and selected-fit features. Mapping appears in CSV. Reload supported in Shape Lab; NanoSense re-import compatibility is not established.\n')
    return out.getvalue()

def comparison_package(events,table,profile_source,meta,condition_label='',window_samples=None):
    """Create a compact cross-condition comparison package from one clustered recording.

    The package intentionally stores cluster assignments plus centered real-sample
    median profiles.  It does not re-run PCA or clustering.  A later comparison page
    can therefore map algorithmic cluster IDs onto cross-salt Family A/B/C... labels.
    """
    import pandas as pd
    from analysis import aligned_event_profiles,cluster_median_profiles
    if table is None or not len(table):raise ValueError('Need a non-empty clustered event table.')
    if 'cluster' not in table or 'event_index' not in table:raise ValueError('Cluster table needs cluster and event_index columns.')
    ids=[int(x) for x in table['event_index'].to_numpy()]
    by_id={int(e.index):e for e in events};missing=[i for i in ids if i not in by_id]
    if missing:raise ValueError(f'{len(missing)} clustered events are not available in the active recording.')
    selected=[by_id[i] for i in ids]
    selected=signal_events(selected,profile_source)
    lengths=np.asarray([int(np.sum((e.time>=e.bounds[0])&(e.time<=e.bounds[1]))) for e in selected],int)
    if window_samples is None:
        window_samples=max(300,int(np.ceil(np.percentile(lengths,95)/50.)*50)+200) if len(lengths) else 500
        window_samples=min(max(int(window_samples),300),5000)
    aligned=aligned_event_profiles(selected,alignment='midpoint',window_samples=int(window_samples))
    labels=table['cluster'].astype(int).to_numpy()
    centers,coverage=cluster_median_profiles(aligned['profiles'],labels,.5)
    groups=sorted(np.unique(labels).tolist())
    profile_rows=[]
    for pos,g in enumerate(groups):
        n=int(np.sum(labels==g))
        for j in range(len(aligned['data_index'])):
            profile_rows.append(dict(cluster=int(g),data_index=float(aligned['data_index'][j]),time_ms=float(aligned['time_ms'][j]),
                median_blockade_nA=float(centers[pos,j]) if np.isfinite(centers[pos,j]) else np.nan,
                coverage_count=int(coverage[pos,j]),cluster_n=n))
    profiles=pd.DataFrame(profile_rows)
    blockade_col='clustering_measured_mean_blockade_nA' if 'clustering_measured_mean_blockade_nA' in table else ('mean_blockade_nA' if 'mean_blockade_nA' in table else None)
    summary=[]
    total=len(table)
    for g in groups:
        sub=table[table.cluster.astype(int)==int(g)]
        d=sub['duration_ms'].dropna().to_numpy(float) if 'duration_ms' in sub else np.array([])
        b=sub[blockade_col].dropna().to_numpy(float) if blockade_col else np.array([])
        row=dict(cluster=int(g),n=int(len(sub)),population_fraction=float(len(sub)/total))
        if len(d):row.update(median_dwell_ms=float(np.median(d)),q1_dwell_ms=float(np.quantile(d,.25)),q3_dwell_ms=float(np.quantile(d,.75)))
        else:row.update(median_dwell_ms=np.nan,q1_dwell_ms=np.nan,q3_dwell_ms=np.nan)
        if len(b):row.update(median_blockade_nA=float(np.median(b)),q1_blockade_nA=float(np.quantile(b,.25)),q3_blockade_nA=float(np.quantile(b,.75)))
        else:row.update(median_blockade_nA=np.nan,q1_blockade_nA=np.nan,q3_blockade_nA=np.nan)
        summary.append(row)
    summary=pd.DataFrame(summary)
    package_meta={
        'format':'DNA Event Lab salt-comparison package','format_version':1,'release_version':RELEASE_VERSION,
        'condition_label':str(condition_label or ''),'profile_source':str(profile_source),'window_samples':int(window_samples),
        'event_midpoint_index':int(aligned['event_midpoint_index']),'sampling_rate_hz':float(aligned['sampling_rate_hz']),
        'time_step_s':float(aligned['time_step_s']),'n_events':int(len(table)),'clusters':groups,'clustering':meta.get('clustering',meta) if isinstance(meta,dict) else {},
    }
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('cluster_assignments.csv',table.to_csv(index=False))
        z.writestr('cluster_profiles.csv',profiles.to_csv(index=False))
        z.writestr('cluster_summary.csv',summary.to_csv(index=False))
        z.writestr('comparison_meta.json',json.dumps(package_meta,indent=2))
        z.writestr('README.txt','Cross-salt comparison package. Cluster IDs are algorithmic labels within this recording only. Map them to Family A/B/C... on the Compare salts page before interpreting topology. cluster_profiles.csv contains centered real-sample pointwise medians; events are not stretched.\n')
    return out.getvalue()

"""Reproducible comparison on the supplied recording; no topology ground truth."""
from pathlib import Path
import json,importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis import load_events,describe,feature_space_diagnostics,cluster_features
from physical import level_features
from representations import waveform_features,subsample_stability
from workflow import signal_events
root=Path(__file__).resolve().parent
out=root/'validation';out.mkdir(exist_ok=True)
path=next((root.parent/'upload').glob('*12_18_31.event_fitting.npz'))
events,_,_=load_events(path.read_bytes())
audit,_=level_features(events)
spec=importlib.util.spec_from_file_location('legacy_physical',root.parent/'upload'/'physical(4).py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
legacy_audit,_=legacy.level_features(events)
fields=['log10_duration_ms','measured_mean_blockade_nA','log10_deep_to_shallow_ratio','deepest_plateau_fraction','measured_temporal_centroid','resolved_shape_complexity']
physical_idx=np.flatnonzero(audit.clustering_eligible.to_numpy())
_,wave_idx,wave_x,wave_names=waveform_features(events)
# Compare on exactly the same events to separate model differences from exclusions.
common=np.intersect1d(physical_idx,wave_idx)
phys=audit.iloc[common][fields].to_numpy(float)
legacy_phys=legacy_audit.iloc[common][fields].to_numpy(float)
wave=wave_x[np.searchsorted(wave_idx,common)]
_,profiles=describe(signal_events([events[i] for i in common],'Selected fits'),128)
results=[];panels=[]
for name,x,names,scale,corr,pc in [
 ('Previous six features / robust / 2 PCs',legacy_phys,fields,'legacy',.95,2),
 ('Six features / standard / 95% variance',phys,fields,'standard',.95,None),
 ('Balanced waveform / 95% variance',wave,wave_names,'balanced',None,None),
 ('Shape-only waveform / 95% variance',wave[:,:32],wave_names[:32],'shape',None,None)]:
 d=feature_space_diagnostics(x,names,corr,scale);npc=pc or int(np.searchsorted(d['cumulative'],.95)+1)
 for method in ['PCA + agglomerative','PCA + k-means']:
  r=cluster_features(x,profiles,3,npc,method,corr,scale)
  stab=subsample_stability(x,r['labels'],3,npc,method,corr,scale)
  row=dict(model=name,method=method,n_events=len(common),k=3,PCs=r['n_components'],retained_variance=float(sum(r['scree'][:r['n_components']])),silhouette=r['silhouette'],mean_subsample_ARI=stab['mean_ari'],min_subsample_ARI=min(v['ARI'] for v in stab['repeats']),cluster_counts=np.bincount(r['labels']).tolist())
  results.append(row);panels.append((name,method,r))
pd.DataFrame(results).to_csv(out/'same_events_comparison.csv',index=False)
(out/'comparison_details.json').write_text(json.dumps({'source':path.name,'loaded':len(events),'physical_eligible':len(physical_idx),'waveform_eligible':len(wave_idx),'common_events':len(common),'note':'k=3 is fixed for comparison, not an inferred biological count. Silhouettes across representations use different distance definitions and should not be ranked as accuracy. All groups are unsupervised.','results':results},indent=2))
fig,axes=plt.subplots(4,2,figsize=(10,14),layout='constrained')
colors=['#0072B2','#D55E00','#009E73']
for ax,(name,method,r),row in zip(axes.flat,panels,results):
 for j in range(3):
  pts=r['embedding'][r['labels']==j];ax.scatter(pts[:,0],pts[:,1],s=4,alpha=.45,color=colors[j],label=f'{j}: n={len(pts)}')
 ax.set_title(name+'\n'+method+f"; subsample ARI {row['mean_subsample_ARI']:.2f}, min {row['min_subsample_ARI']:.2f}",fontsize=9)
 ax.set_xlabel(f"PC1 ({100*r['pca_variance'][0]:.1f}%)");ax.set_ylabel(f"PC2 ({100*r['pca_variance'][1]:.1f}%)")
 ax.legend(frameon=False,fontsize=7)
fig.savefig(out/'same_events_comparison.png',dpi=150);plt.close(fig)
print(pd.DataFrame(results).to_string(index=False))

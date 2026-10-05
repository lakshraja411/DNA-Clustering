"""Known-signal checks, independent of biological topology assignments."""
import numpy as np
from dataclasses import replace
from sklearn.metrics import adjusted_rand_score
from analysis import Event,_feature_space,cluster_features,cluster_count_diagnostics
from physical import level_features
from representations import phase_average,waveform_features,subsample_stability

def event(i,shape,amplitude=1.,duration=1.):
    time=np.linspace(0,duration,len(shape));base=np.full(len(time),20.)
    fit=base-np.asarray(shape)*amplitude
    return Event(i,time,fit.copy(),np.array([0.,duration]),base,fit.copy())

shape=np.r_[np.ones(50)*2,np.ones(51)]
a=event(0,shape);b=event(1,shape,amplitude=7,duration=3)
audit,idx,x,names=waveform_features([a,b],shape_only=True)
np.testing.assert_allclose(x[0],x[1],atol=1e-12)
np.testing.assert_allclose(phase_average(a).mean(),np.trapezoid(a.baseline-a.fit,a.time)/(a.bounds[1]-a.bounds[0]))
_,_,xb,_=waveform_features([a,b],shape_only=False)
np.testing.assert_allclose(xb[1,-2:]-xb[0,-2:],[np.log10(7),np.log10(3)])
# Measured-waveform mode supports an event lacking a fit.
c=replace(a,index=2,fit=None)
assert waveform_features([c],'Measured trace')[0].representation_eligible.iloc[0]
assert not waveform_features([c],'Selected fits')[0].representation_eligible.iloc[0]
# A rare but finite feature is retained under standard scaling, unlike legacy QC.
rng=np.random.default_rng(42);rare=np.zeros(100);rare[:3]=2
features=np.c_[rng.normal(size=100),rare]
assert _feature_space(features,2,None,'standard')['keep'].tolist()==[True,True]
assert _feature_space(features,2,None,'legacy')['keep'].tolist()==[True,False]
# Known opposing shapes recover groups despite amplitude/duration variations.
events=[];truth=[]
for j in range(80):
    cls=j%2;sh=shape if cls==0 else shape[::-1]
    events.append(event(j,sh+rng.normal(0,.02,len(sh)),amplitude=rng.uniform(.7,2),duration=rng.uniform(.5,2)));truth.append(cls)
_,_,features,_=waveform_features(events,shape_only=True)
result=cluster_features(features,features,2,2,'PCA + k-means',None,'shape')
assert adjusted_rand_score(truth,result['labels'])>.95
stability=subsample_stability(features,result['labels'],2,2,'PCA + k-means',None,'shape')
assert stability['mean_ari']>.95 and len(stability['repeats'])==6
_,_,features,_=waveform_features(events,shape_only=False)
space=_feature_space(features,4,None,'balanced')
used=np.flatnonzero(space['keep']);scaled=space['scaled']
np.testing.assert_allclose(np.var(scaled[:,used<32],axis=0).sum(),1.)
np.testing.assert_allclose(np.var(scaled[:,used>=32],axis=0),1.)
# One-group output is available without reporting meaningless silhouette/stability.
one=cluster_features(features,features,1,4,'PCA + k-means',None,'balanced')
assert len(np.unique(one['labels']))==1 and np.isnan(one['silhouette'])
assert subsample_stability(features,one['labels'],1,4,'PCA + k-means',None,'balanced')['mean_ari'] is None
scan=cluster_count_diagnostics(features,'PCA + k-means',4,1,4,correlation_threshold=None,scaling='balanced')
assert scan['table'][0]['k']==1 and np.isnan(scan['table'][0]['silhouette'])
print('PASS: area-preserving bin means, amplitude/time invariance, missing-fit handling, rare-feature preservation, known-shape recovery, block weights, subsample stability and one-group behaviour.')

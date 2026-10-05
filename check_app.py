"""Lightweight regression checks for DNA Event Lab v0.8.0 dataset-feature clustering."""
import numpy as np
from sklearn.metrics import adjusted_rand_score

import analysis, physical, plots, workflow
from analysis import feature_space_diagnostics, cluster_count_diagnostics, cluster_features

EXPECTED='0.8.0'
for module in [analysis,physical,plots,workflow]:
    assert getattr(module,'RELEASE_VERSION',None)==EXPECTED, (module.__name__,getattr(module,'RELEASE_VERSION',None))

# Synthetic NanoSense-like feature matrix: height, fwhm, height_at_fwhm,
# area, width, skew and kurtosis.  This checks the actual v0.8 pipeline:
# MinMax[-1,1] -> PCA -> Ward/k-means.
rng=np.random.default_rng(17)
n_each=160
truth=np.repeat(np.arange(3),n_each)
n=len(truth)
x=np.zeros((n,7),float)
x[:,0]=rng.normal(np.choose(truth,[1.0,1.55,2.15]),.10)       # height
x[:,1]=rng.normal(np.choose(truth,[.78,.52,.34]),.055)       # fwhm
x[:,2]=x[:,0]*rng.normal(.76,.025,n)                         # height_at_fwhm
x[:,3]=x[:,0]*x[:,1]*rng.normal(1.0,.04,n)                   # area
x[:,4]=x[:,1]*rng.normal(1.14,.025,n)                        # width
x[:,5]=rng.normal(np.choose(truth,[0.0,.55,1.1]),.20)        # skew
x[:,6]=rng.normal(np.choose(truth,[2.8,3.35,4.0]),.28)       # kurtosis
profiles=np.tile(x[:,0,None],(1,128))
names=['height','fwhm','height_at_fwhm','area','width','skew','kurtosis']

diag=feature_space_diagnostics(x,names,None)
assert diag['retained_features']==names
assert diag['scaling_method']=='MinMaxScaler [-1, 1]'
assert np.isclose(sum(diag['scree']),1.0)

scan=cluster_count_diagnostics(x,'PCA + agglomerative',2,2,6,names,None)
assert scan['suggested_k']==3

ward=cluster_features(x,profiles,3,2,'PCA + agglomerative',None)
kmeans=cluster_features(x,profiles,3,2,'PCA + k-means',None)
assert adjusted_rand_score(truth,ward['labels'])>.90
assert adjusted_rand_score(truth,kmeans['labels'])>.90
assert max(ward['pca_variance'])<.95
assert np.bincount(ward['labels']).min()>100

print('PASS: v0.8.0 modules agree; original dataset-feature MinMax/PCA pipeline recovers a known three-family fixture with Ward and k-means.')

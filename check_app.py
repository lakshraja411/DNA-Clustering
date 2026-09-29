"""Meaningful checks for numerical methods, export round trips and UI changes."""
import io,zipfile, numpy as np
from analysis import *
from streamlit.testing.v1 import AppTest

def button(at,label):return next(b for b in at.button if b.label==label)
def select(at,label):return next(b for b in at.selectbox if b.label==label)

e,s,r=synthetic_demo();df,x=describe(e)
ref=refine(e[0],'Segment means')
assert fit_metrics(e[0],ref['fit'])['waveform_rmse_nA']<=fit_metrics(e[0],e[0].fit)['waveform_rmse_nA']+1e-12
flat=Event(999,np.arange(200)*5e-6,17-np.r_[np.ones(100),np.ones(100)*2],np.array([0.,.001]),np.full(200,17.))
pelt=refine(flat,'New levels (PELT)',25,8)
assert len(pelt['levels'])==2 and np.allclose(pelt['levels'],[1,2])
# Never mutate the original arrays when refining.
assert e[0].fit is not ref['fit']
xsource=np.zeros((len(e),10));xsource[:,8]=[q.bounds[0] for q in e]
# Source summary ordering deliberately differs from event order.
xsource=xsource[::-1].copy();mapping,status=link_dataset(e,xsource)
assert len(mapping)==len(e) and mapping[0]==79
# Ambiguous timestamp must not be silently matched.
duplicate=np.r_[xsource,xsource[-1:]];dm,_=link_dataset(e,duplicate);assert 0 not in dm
sub=e[::3];table=df[df.event_index.isin([q.index for q in sub])]
bundle=cluster_bundle(sub,table,s,{'check':True},{0:ref},xsource,mapping)
with zipfile.ZipFile(io.BytesIO(bundle)) as z:
 restored,_,_=load_events(z.read('events.npz'));assert [q.index for q in restored]==[q.index for q in sub]
 for a,b in zip(sub,restored):assert np.array_equal(a.current,b.current) and np.array_equal(a.fit,b.fit)
 zsub=np.load(io.BytesIO(z.read('dataset_subset.npz')),allow_pickle=False)
 assert np.array_equal(zsub['X'],xsource[[mapping[q.index] for q in sub]])
ft=feature_table(e);a=cluster_features(ft[['duration_ms','mean_blockade_nA','blockade_std_nA','early_late_difference_nA']].to_numpy(),x,4)
assert len(a['labels'])==80
# Reject arbitrary serialized globals in event_data.
import pickle
class Bad:
 def __reduce__(self):return (eval,('1+1',))
buf=io.BytesIO();np.savez(buf,events=np.array([Bad()],object),sampling_rate=200000.)
try:load_events(buf.getvalue())
except ValueError as ex:assert 'Unsupported serialized type' in str(ex)
else:raise AssertionError('Unsafe global accepted')
at=AppTest.from_file('app.py',default_timeout=90).run();assert not at.exception
at.sidebar.radio[0].set_value('Synthetic demonstration').run();assert not at.exception
button(at,'Run grouping').click().run();assert not at.exception
button(at,'Prepare event download').click().run();assert not at.exception
select(at,'Clustering method').set_value('PCA + k-means (nanorod concept)').run();assert not at.exception
assert any('settings changed' in msg.value.lower() for msg in at.info)
button(at,'Run grouping').click().run();assert not at.exception
select(at,'Candidate method').set_value('New levels (PELT)').run();button(at,'Calculate candidate fit').click().run();assert not at.exception
select(at,'Clustering method').set_value('Waveform k-means (original baseline)').run();button(at,'Run grouping').click().run();assert not at.exception
print('PASS: fit comparison, level recovery, timestamp joins, ambiguous joins, export roundtrip, numeric-loader guard, UI grouping/refinement/download and stale-result handling.')

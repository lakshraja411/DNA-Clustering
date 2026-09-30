import io,zipfile
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
from analysis import load_events,load_dataset,link_dataset,describe,refine,feature_table,auto_cluster_features
from workflow import active_events,signal_events,fitting_bytes,match_raw,bundle
from plots import figure_archive
# Clearly separated but highly redundant features must remain clusterable after pruning.
from sklearn.metrics import adjusted_rand_score
rng=np.random.default_rng(17);truth=np.repeat([0,1],60)
x=rng.normal(0,.35,(120,4))+np.where(truth[:,None]==0,-4,4)
known=auto_cluster_features(x,np.tile(x[:,0,None],(1,32)),k_max=5,stability_repeats=4)
assert known['selected_k']==2 and adjusted_rand_score(truth,known['labels'])>.95
root=Path(__file__).resolve().parent.parent/'upload'
fitpath=next(root.glob('*12_18_31.event_fitting.npz'));rawpath=next(root.glob('*12_18_29.event_data.npz'));datapath=next(root.glob('*12_18_31.dataset.npz'))
events,settings,rejected=load_events(fitpath.read_bytes());raw,_,rr=load_events(rawpath.read_bytes());dataset,ds=load_dataset(datapath.read_bytes())
assert len(events)==1400 and not rejected and not rr
mapping,status=link_dataset(events,dataset);rawmap,rstatus=match_raw(events,raw,1e-7)
assert len(mapping)==len(rawmap)==1400
small=events[:32];e=small[0];old=e.fit.copy();refs={e.index:refine(e,'Segment means')}
blob=fitting_bytes(small,refs,settings,{})
loaded,_,bad=load_events(blob);assert not bad
np.testing.assert_array_equal(loaded[0].fit,refs[e.index]['fit']);np.testing.assert_array_equal(loaded[0].current,e.current);np.testing.assert_array_equal(e.fit,old)
with np.load(io.BytesIO(blob)) as z:np.testing.assert_array_equal(z[f'INPUT_FIT_{e.index}'],old)
a=active_events(small,refs);feat=feature_table(signal_events(a,'Selected fits'));meas=feature_table(small)
assert not np.allclose(feat['blockade_std_nA'],meas['blockade_std_nA'])
table,_=describe(small);data=bundle(small,rawmap,refs,dataset,mapping,settings,ds,table,{})
with zipfile.ZipFile(io.BytesIO(data)) as z:
 ev,_,_=load_events(z.read('selected.eventfitting.npz'));np.testing.assert_array_equal(ev[0].fit,refs[e.index]['fit'])
 x,_=load_dataset(z.read('selected.dataset.npz'));np.testing.assert_array_equal(x,dataset[[mapping[e.index] for e in small]])
 rawsub,_,_=load_events(z.read('selected.eventdata.npz'));assert len(rawsub)==32
assert len(zipfile.ZipFile(io.BytesIO(figure_archive(table,'duration_ms','mean_blockade_nA',True))).namelist())==8
p=dict(events=small,raw=raw,settings=settings,dsettings=ds,dataset=dataset,mapping=mapping,rawmap=rawmap,status=status,rstatus=rstatus,rejected=[],fingerprint='test',files={},matching={})
at=AppTest.from_file('app.py',default_timeout=30).run();assert not at.exception
assert len(at.get('file_uploader'))==3
at.session_state['project']=p;at.session_state['refs']={};at.session_state['recording_confirmed']=True

def step(n):
 at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[n-1]).run();assert not at.exception,at.exception
 assert len(at.get('file_uploader'))==3

def button(label):return next(b for b in at.button if b.label==label)
step(2)
next(b for b in at.button if b.key=='next_bottom').click().run();assert not at.exception
assert at.sidebar.radio[0].value.startswith('3')
button('Calculate refined fits').click().run();assert not at.exception
button('Prepare new eventfitting file').click().run();assert not at.exception
next(b for b in at.button if b.key=='previous_bottom').click().run();assert not at.exception
assert at.sidebar.radio[0].value.startswith('2')
assert at.session_state['refs']
step(4);button('Prepare publication figures').click().run();assert not at.exception
step(5);button('Run clustering').click().run();assert not at.exception,at.exception
assert at.session_state['group']['meta']['clustering']['source']=='Selected fits'
assert at.session_state['group']['meta']['clustering']['selection']=='automatic'
assert 2<=at.session_state['group']['meta']['clustering']['k']<=8
assert at.session_state['group']['info']['selection_table']
button('Prepare cluster profile figures').click().run();assert not at.exception
assert len(zipfile.ZipFile(io.BytesIO(at.session_state['cluster_figures'][1])).namelist())==5
next(b for b in at.button if b.key=='next_bottom').click().run();assert not at.exception
assert at.sidebar.radio[0].value.startswith('6')
button('Prepare cluster files').click().run();assert not at.exception
assert at.session_state['prepared'][1]
step(5);assert at.session_state['group']
next(w for w in at.slider if w.label=='Maximum clusters to test').set_value(7).run();assert not at.exception
assert 'group' not in at.session_state
step(6);assert not at.exception
step(1);assert at.session_state['recording_confirmed'];assert at.session_state['refs']
step(2);assert at.session_state['refs']
button('Clear files and start over').click().run();assert not at.exception
assert 'project' not in at.session_state and 'refs' not in at.session_state
print('PASS: 1,400 three-file matches; refined-fit reload and preservation; fitted-feature input; automatic k selection diagnostics; cluster subset alignment; scientific exports; guided UI and stale-group invalidation.')

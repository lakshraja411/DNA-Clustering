"""Known-level checks for the physical descriptors (not topology validation)."""
import numpy as np
from analysis import Event
from physical import level_features

def event(i,heights,widths=None):
    widths=widths or [100]*len(heights)
    blockade=np.repeat(heights,widths)
    t=np.arange(-20,len(blockade)+20)*5e-6
    full=np.r_[np.zeros(20),blockade,np.zeros(20)]
    base=np.full(len(t),20.)
    return Event(i,t,base-full,np.array([0.,len(blockade)*5e-6]),base,base-full)

events=[event(0,[1]),event(1,[2]),event(2,[2,1]),event(3,[1,2]),event(4,[1,2,1]),event(5,[1,1.05]),event(6,[1,2],[100,2])]
original=events[2].fit.copy()
a,seq=level_features(events,reference_nA=1.,deep_threshold_nA=1.5,omit_short_boundaries=False)
assert a.physical_eligible.tolist()==[True]*6+[False]
np.testing.assert_allclose(a.loc[:5,'resolved_transitions'],[0,0,1,1,2,0])
np.testing.assert_allclose(a.loc[:5,'transition_direction'],[0,0,-1,1,0,0])
np.testing.assert_allclose(a.loc[:4,'deep_time_fraction'],[0,1,.5,.5,1/3])
assert 'shorter' in a.loc[6,'physical_exclusion_reason']
for e in events[:6]:np.testing.assert_allclose(seq.loc[seq.event_index==e.index,'duration_ms'].sum(),(e.bounds[1]-e.bounds[0])*1000)
np.testing.assert_array_equal(events[2].fit,original)
e=event(7,[1,2]);e.current=e.current-.2
f,_=level_features([e],'Measured trace');np.testing.assert_allclose(f.deepest_plateau_nA,[2.2])
e=event(8,[1]);e.fit=None
f,_=level_features([e]);assert not f.physical_eligible.iloc[0]
e=event(9,[1]);e.fit=e.baseline-np.exp(-(e.time-.00025)**2/(2*.0001**2))
f,_=level_features([e]);assert not f.physical_eligible.iloc[0]
print('PASS: known plateau depth, occupancy, transition count/direction, merging, duration sum, short/continuous/missing-fit exclusion, signal source and input preservation.')

# Boundary omission retains sustained levels, preserving original time/fit arrays.
boundaries=[event(10,[1,2],[100,2]),event(11,[2,1],[2,100]),event(12,[2,1,2],[2,100,2]),event(13,[1,2,1],[100,2,100]),event(14,[2],[2])]
a,seq=level_features(boundaries,deep_threshold_nA=1.5)
assert a.physical_eligible.tolist()==[True,True,True,False,False]
np.testing.assert_allclose(a.loc[:2,'deepest_plateau_nA'],[1,1,1])
np.testing.assert_allclose(a.loc[:2,'resolved_transitions'],[0,0,0])
np.testing.assert_allclose(a.loc[:2,'deep_time_fraction'],[0,0,0])
np.testing.assert_allclose(a.loc[:2,'omitted_boundary_duration_us'],[10,10,20])
assert 'internal' in a.loc[3,'physical_exclusion_reason']
assert 'no sustained' in a.loc[4,'physical_exclusion_reason']
for e in boundaries:
 levels=seq.loc[seq.event_index==e.index]
 np.testing.assert_allclose(levels.duration_ms.sum(),(e.bounds[1]-e.bounds[0])*1000)
assert len(seq.loc[(seq.event_index==12)&(seq.level_status=='omitted_boundary')])==2
# Occupancy uses retained duration, while event duration retains its detected bounds.
e=event(15,[.5,2,1],[2,100,100]);saved=e.fit.copy()
a,seq=level_features([e],deep_threshold_nA=1.5)
np.testing.assert_allclose(a.deep_time_fraction,[.5]);np.testing.assert_allclose(a.duration_ms,[1.01]);np.testing.assert_allclose(a.analysed_duration_ms,[1.])
np.testing.assert_array_equal(e.fit,saved)
print('PASS: edge omission on/off, both edges, internal short-level exclusion, no sustained core, audit coverage, occupancy denominator and preserved total duration.')

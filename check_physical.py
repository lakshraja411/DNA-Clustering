"""Known-level checks for the physical descriptors (not topology validation)."""
import numpy as np
from analysis import Event
from physical import level_features,level_stability

def event(i,heights,widths=None):
    widths=widths or [100]*len(heights)
    blockade=np.repeat(heights,widths)
    t=np.arange(-20,len(blockade)+20)*5e-6
    full=np.r_[np.zeros(20),blockade,np.zeros(20)]
    base=np.full(len(t),20.)
    return Event(i,t,base-full,np.array([0.,len(blockade)*5e-6]),base,base-full)

events=[event(0,[1]),event(1,[2]),event(2,[2,1]),event(3,[1,2]),event(4,[1,2,1]),event(5,[1,1.05]),event(6,[1,2],[100,2])]
original=events[2].fit.copy()
a,seq=level_features(events,reference_nA=1.,deep_threshold_nA=1.5)
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

# Version 0.6 QC: transition margins are exported and resolution stability is measurable.
f,seq=level_features(events[:6],reference_nA=1.,deep_threshold_nA=1.5)
assert 'transition_snr' in seq and 'transition_from_previous_nA' in seq
assert np.isfinite(seq.loc[seq.level_order>0,'transition_snr']).all()
stable,detail,overall=level_stability(events[:6],reference_nA=1.,deep_threshold_nA=1.5,
    baseline_min_duration_us=25.,baseline_noise_multiplier=3.,duration_values_us=[25.,50.],noise_multipliers=[2.,3.,4.])
assert overall['baseline_eligible_events']==6 and overall['strict_stable_events']==6
assert stable.strict_transition_stable.all() and len(detail)==6*6
print('PASS: transition SNR export and resolved-level parameter-stability scan.')


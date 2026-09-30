"""Physical level descriptors from selected step boundaries, with explicit QC."""
import numpy as np
import pandas as pd
from analysis import mask

PHYSICAL_DESCRIPTIONS={
 'deepest_plateau_nA':'Deepest sustained resolved plateau blockade (nA), excluding isolated sample maxima.',
 'deepest_plateau_ratio':'Deepest plateau divided by a user-supplied, condition-specific single-file reference.',
 'deep_time_fraction':'Fraction of detected event duration in resolved plateaus above the user-defined deeper-blockade threshold.',
 'resolved_transitions':'Number of changes between adjacent resolved plateaus after height merging; not strand count.',
 'transition_direction':'(Last plateau − first plateau) / (highest − lowest resolved plateau); zero for one level. Positive = deeper at the end.',
 'duration_ms':'Detected end minus start in ms; optionally omitted to compare configurations without kinetics.'}

def level_features(events,source='Selected fits',min_duration_us=25.,min_height_nA=.1,noise_multiplier=3.,reference_nA=None,deep_threshold_nA=None):
    if source not in ['Selected fits','Measured trace']:raise ValueError('Unknown physical-feature signal source.')
    if not all(np.isfinite(v) for v in [min_duration_us,min_height_nA,noise_multiplier]):raise ValueError('Resolution settings must be finite.')
    if min_duration_us<0 or min_height_nA<0 or noise_multiplier<0:raise ValueError('Resolution settings must be nonnegative.')
    if reference_nA is not None and (not np.isfinite(reference_nA) or reference_nA<=0):raise ValueError('Single-file reference must be positive and finite.')
    if deep_threshold_nA is not None and (not np.isfinite(deep_threshold_nA) or deep_threshold_nA<=0):raise ValueError('Deeper-blockade threshold must be positive and finite.')
    rows=[];sequences=[]
    for e in events:
        row={'event_index':e.index,'physical_eligible':False,'physical_exclusion_reason':''}
        try:
            if e.fit is None:raise ValueError('missing step fit; refine using PELT or segment means')
            m=mask(e);t=e.time[m];fit=(e.baseline-e.fit)[m];observed=(e.baseline-e.current)[m];dt=float(np.median(np.diff(e.time)));duration=float(e.bounds[1]-e.bounds[0])
            raw_ends=np.r_[np.flatnonzero(np.abs(np.diff(fit))>1e-8)+1,len(fit)]
            if len(raw_ends)>max(8,int(len(fit)*.25)):raise ValueError('fit has too many sample-to-sample changes to use as resolved step plateaus; use a step fit')
            padding=(e.baseline-e.current)[~m]
            if len(padding)>=4:
                noise=float(np.median(np.abs(padding-np.median(padding)))/.67448975);noise_source='padding MAD'
            else:
                dif=np.diff(observed);noise=float(np.median(np.abs(dif-np.median(dif)))/(.67448975*np.sqrt(2)));noise_source='within-event first-difference MAD'
            height_tol=max(float(min_height_nA),noise_multiplier*noise)
            minimum=max(min_duration_us*1e-6,3*dt)
            chosen=fit if source=='Selected fits' else observed
            segments=[];start=0
            for end in raw_ends:
                left=float(e.bounds[0]) if start==0 else float(t[start]);right=float(e.bounds[1]) if end==len(t) else float(t[end])
                value=float(chosen[start:end].mean());segments.append({'left':left,'right':right,'start':start,'end':int(end),'value':value});start=int(end)
            # Greedily merge the closest adjacent heights; recompute over all samples.
            while len(segments)>1:
                differences=np.abs(np.diff([a['value'] for a in segments]));j=int(np.argmin(differences))
                if differences[j]>height_tol:break
                a,b=segments[j:j+2];joined={'left':a['left'],'right':b['right'],'start':a['start'],'end':b['end'],'value':float(chosen[a['start']:b['end']].mean())}
                segments[j:j+2]=[joined]
            row.update(noise_scale_nA=noise,noise_estimator=noise_source,merge_height_nA=height_tol,effective_min_duration_us=minimum*1e6,resolved_levels=len(segments))
            if any(a['right']-a['left']<minimum-1e-12 for a in segments):raise ValueError('a plateau is shorter than the minimum resolved duration; inspect/refine the event')
            heights=np.array([a['value'] for a in segments]);widths=np.array([a['right']-a['left'] for a in segments]);span=float(np.ptp(heights))
            direction=float((heights[-1]-heights[0])/span) if len(segments)>1 and span>height_tol else 0.
            row.update(deepest_plateau_nA=float(heights.max()),resolved_transitions=len(segments)-1,transition_direction=direction,duration_ms=duration*1000,physical_eligible=True)
            if reference_nA is not None:row['deepest_plateau_ratio']=float(heights.max()/reference_nA)
            if deep_threshold_nA is not None:
                row['deep_time_fraction']=float(widths[heights>deep_threshold_nA].sum()/duration)
                row['threshold_sensitive']=bool(np.any(np.abs(heights-deep_threshold_nA)<=height_tol))
            for j,a in enumerate(segments):
                item={'event_index':e.index,'level_order':j,'start_from_event_ms':(a['left']-e.bounds[0])*1000,'duration_ms':(a['right']-a['left'])*1000,'blockade_nA':a['value'],'source':source}
                if reference_nA is not None:item['blockade_ratio']=a['value']/reference_nA
                if deep_threshold_nA is not None:item['above_deeper_threshold']=a['value']>deep_threshold_nA
                sequences.append(item)
        except ValueError as ex:row['physical_exclusion_reason']=str(ex)
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(sequences,columns=['event_index','level_order','start_from_event_ms','duration_ms','blockade_nA','source']+(['blockade_ratio'] if reference_nA is not None else [])+(['above_deeper_threshold'] if deep_threshold_nA is not None else []))

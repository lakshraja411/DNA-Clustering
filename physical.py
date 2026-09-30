"""Physical level descriptors from selected step boundaries, with explicit QC.

Version 0.6 adds transition SNR reporting and a resolution-parameter stability scan.
The plateau boundaries still come from the selected step fit; the measured-trace
mode only re-estimates plateau heights from measured samples inside those bounds.
"""
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


def _robust_sigma(values):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    if len(values)<2:return np.nan
    med=np.median(values);mad=np.median(np.abs(values-med))
    return float(mad/.67448975)


def _noise_scale(e,m,observed):
    padding=(e.baseline-e.current)[~m]
    if len(padding)>=4:
        noise=_robust_sigma(padding);source='padding MAD'
    else:
        dif=np.diff(observed)
        noise=_robust_sigma(dif)/np.sqrt(2) if len(dif)>=2 else np.nan
        source='within-event first-difference MAD'
    # A completely flat numerical trace can give zero MAD. Keep the threshold
    # finite without allowing one floating-point bit to become a "transition".
    fallback=max(float(np.std(observed))*1e-3,1e-9)
    if not np.isfinite(noise) or noise<fallback:noise=fallback
    return float(noise),source


def _resolve_event(e,source,min_duration_us,min_height_nA,noise_multiplier,reference_nA,deep_threshold_nA):
    if e.fit is None:raise ValueError('missing step fit; refine using PELT or segment means')
    m=mask(e);t=e.time[m]
    fit=(e.baseline-e.fit)[m]
    observed=(e.baseline-e.current)[m]
    dt=float(np.median(np.diff(e.time)));duration=float(e.bounds[1]-e.bounds[0])
    raw_ends=np.r_[np.flatnonzero(np.abs(np.diff(fit))>1e-8)+1,len(fit)]
    if len(raw_ends)>max(8,int(len(fit)*.25)):
        raise ValueError('fit has too many sample-to-sample changes to use as resolved step plateaus; use a step fit')
    noise,noise_source=_noise_scale(e,m,observed)
    height_tol=max(float(min_height_nA),float(noise_multiplier)*noise)
    minimum=max(float(min_duration_us)*1e-6,3*dt)
    chosen=fit if source=='Selected fits' else observed

    segments=[];start=0
    for end in raw_ends:
        left=float(e.bounds[0]) if start==0 else float(t[start])
        right=float(e.bounds[1]) if end==len(t) else float(t[end])
        value=float(chosen[start:end].mean())
        segments.append({'left':left,'right':right,'start':start,'end':int(end),'value':value})
        start=int(end)

    # Merge only adjacent levels whose height separation is below the effective
    # experimental resolution. Recompute the merged level from all samples.
    while len(segments)>1:
        differences=np.abs(np.diff([a['value'] for a in segments]));j=int(np.argmin(differences))
        if differences[j]>height_tol:break
        a,b=segments[j:j+2]
        joined={'left':a['left'],'right':b['right'],'start':a['start'],'end':b['end'],
                'value':float(chosen[a['start']:b['end']].mean())}
        segments[j:j+2]=[joined]

    if any(a['right']-a['left']<minimum-1e-12 for a in segments):
        raise ValueError('a plateau is shorter than the minimum resolved duration; inspect/refine the event')

    heights=np.array([a['value'] for a in segments],float)
    widths=np.array([a['right']-a['left'] for a in segments],float)
    span=float(np.ptp(heights))
    direction=float((heights[-1]-heights[0])/span) if len(segments)>1 and span>height_tol else 0.
    row=dict(event_index=e.index,physical_eligible=True,physical_exclusion_reason='',
             noise_scale_nA=noise,noise_estimator=noise_source,merge_height_nA=height_tol,
             effective_min_duration_us=minimum*1e6,resolved_levels=len(segments),
             deepest_plateau_nA=float(heights.max()),resolved_transitions=len(segments)-1,
             transition_direction=direction,duration_ms=duration*1000)
    if reference_nA is not None:row['deepest_plateau_ratio']=float(heights.max()/reference_nA)
    if deep_threshold_nA is not None:
        row['deep_time_fraction']=float(widths[heights>deep_threshold_nA].sum()/duration)
        row['threshold_sensitive']=bool(np.any(np.abs(heights-deep_threshold_nA)<=height_tol))

    sequence=[]
    for j,a in enumerate(segments):
        delta=np.nan if j==0 else float(a['value']-segments[j-1]['value'])
        snr=np.nan if j==0 else float(abs(delta)/noise) if noise>0 else np.inf
        item={'event_index':e.index,'level_order':j,
              'start_from_event_ms':(a['left']-e.bounds[0])*1000,
              'end_from_event_ms':(a['right']-e.bounds[0])*1000,
              'duration_ms':(a['right']-a['left'])*1000,
              'blockade_nA':a['value'],'source':source,
              'transition_from_previous_nA':delta,'transition_snr':snr,
              'noise_scale_nA':noise,'merge_height_nA':height_tol,
              'effective_min_duration_us':minimum*1e6}
        if reference_nA is not None:item['blockade_ratio']=a['value']/reference_nA
        if deep_threshold_nA is not None:item['above_deeper_threshold']=a['value']>deep_threshold_nA
        sequence.append(item)
    return row,sequence


def level_features(events,source='Selected fits',min_duration_us=25.,min_height_nA=.1,noise_multiplier=3.,reference_nA=None,deep_threshold_nA=None):
    if source not in ['Selected fits','Measured trace']:raise ValueError('Unknown physical-feature signal source.')
    if not all(np.isfinite(v) for v in [min_duration_us,min_height_nA,noise_multiplier]):raise ValueError('Resolution settings must be finite.')
    if min_duration_us<0 or min_height_nA<0 or noise_multiplier<0:raise ValueError('Resolution settings must be nonnegative.')
    if reference_nA is not None and (not np.isfinite(reference_nA) or reference_nA<=0):raise ValueError('Single-file reference must be positive and finite.')
    if deep_threshold_nA is not None and (not np.isfinite(deep_threshold_nA) or deep_threshold_nA<=0):raise ValueError('Deeper-blockade threshold must be positive and finite.')
    rows=[];sequences=[]
    for e in events:
        try:
            row,seq=_resolve_event(e,source,min_duration_us,min_height_nA,noise_multiplier,reference_nA,deep_threshold_nA)
            sequences.extend(seq)
        except ValueError as ex:
            row={'event_index':e.index,'physical_eligible':False,'physical_exclusion_reason':str(ex)}
        rows.append(row)
    columns=['event_index','level_order','start_from_event_ms','end_from_event_ms','duration_ms','blockade_nA','source',
             'transition_from_previous_nA','transition_snr','noise_scale_nA','merge_height_nA','effective_min_duration_us']
    if reference_nA is not None:columns.append('blockade_ratio')
    if deep_threshold_nA is not None:columns.append('above_deeper_threshold')
    return pd.DataFrame(rows),pd.DataFrame(sequences,columns=columns)


def level_stability(events,source='Selected fits',min_height_nA=.1,reference_nA=None,deep_threshold_nA=None,
                    baseline_min_duration_us=25.,baseline_noise_multiplier=3.,
                    duration_values_us=(25.,50.,75.,100.),noise_multipliers=(2.,3.,4.)):
    """Sensitivity of resolved transition counts to reasonable resolution settings.

    This probes post-fit plateau resolution (duration and height/noise merging). It
    does not re-estimate change-point boundaries or test the PELT penalty itself.
    """
    durations=sorted({float(baseline_min_duration_us),*[float(v) for v in duration_values_us if float(v)>=0]})
    multipliers=sorted({float(baseline_noise_multiplier),*[float(v) for v in noise_multipliers if float(v)>=0]})
    configs=[(d,n) for d in durations for n in multipliers]
    details=[]
    for d,n in configs:
        audit,_=level_features(events,source,min_duration_us=d,min_height_nA=min_height_nA,noise_multiplier=n,
                               reference_nA=reference_nA,deep_threshold_nA=deep_threshold_nA)
        for _,r in audit.iterrows():
            details.append({'event_index':int(r.event_index),'min_duration_us':d,'noise_multiplier':n,
                            'physical_eligible':bool(r.physical_eligible),
                            'resolved_transitions':float(r.resolved_transitions) if bool(r.physical_eligible) else np.nan,
                            'exclusion_reason':r.get('physical_exclusion_reason','')})
    detail=pd.DataFrame(details)
    base=detail[(np.isclose(detail.min_duration_us,baseline_min_duration_us))&(np.isclose(detail.noise_multiplier,baseline_noise_multiplier))]
    base=base.set_index('event_index')
    rows=[]
    for eid,g in detail.groupby('event_index',sort=False):
        if eid not in base.index:continue
        b=base.loc[eid]
        if isinstance(b,pd.DataFrame):b=b.iloc[0]
        base_ok=bool(b.physical_eligible);base_count=b.resolved_transitions if base_ok else np.nan
        same=(g.physical_eligible.to_numpy() & base_ok & np.isclose(g.resolved_transitions.to_numpy(float),base_count,equal_nan=False))
        eligible=int(g.physical_eligible.sum());total=len(g)
        rows.append({'event_index':int(eid),'baseline_eligible':base_ok,'baseline_transitions':base_count,
                     'tested_configurations':total,'eligible_configurations':eligible,
                     'eligibility_fraction':eligible/total if total else np.nan,
                     'same_as_baseline_fraction':float(same.mean()) if total and base_ok else np.nan,
                     'strict_transition_stable':bool(base_ok and same.all())})
    summary=pd.DataFrame(rows)
    denom=int(summary.baseline_eligible.sum()) if len(summary) else 0
    stable=int(summary.loc[summary.baseline_eligible,'strict_transition_stable'].sum()) if denom else 0
    overall={'baseline_eligible_events':denom,'strict_stable_events':stable,
             'strict_stable_fraction':stable/denom if denom else np.nan,
             'tested_configurations':len(configs),'duration_values_us':durations,'noise_multipliers':multipliers}
    return summary,detail,overall

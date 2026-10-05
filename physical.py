"""DNA-specific resolved-level descriptors with explicit quality control.

The feature engine intentionally separates event detection from event interpretation.
Selected step fits define candidate plateau boundaries; adjacent levels are merged when
they cannot be resolved under the declared amplitude/noise criterion, and plateaus must
persist for the declared minimum time.  The original trace, fit and event boundaries are
never modified.
"""
RELEASE_VERSION='0.6.3'
import numpy as np
import pandas as pd
from analysis import mask

PHYSICAL_DESCRIPTIONS={
 'duration_ms':'Detected event end minus start (ms). Kinetic descriptor; total detected duration is preserved even when short edge plateaus are omitted.',
 'log10_duration_ms':'Base-10 logarithm of detected event duration in ms. Used for clustering so long dwell-time tails do not dominate the PCA scale.',
 'resolved_weighted_mean_nA':'Duration-weighted mean blockade of the retained resolved plateaus (nA). This is the typical sustained blockade state after level QC.',
 'resolved_weighted_mean_ratio':'Duration-weighted resolved blockade divided by the user-supplied condition-specific single-file reference.',
 'deepest_plateau_nA':'Deepest sustained resolved plateau blockade (nA), excluding isolated sample maxima.',
 'deepest_plateau_ratio':'Deepest sustained plateau divided by a user-supplied, condition-specific single-file reference.',
 'resolved_blockade_range_nA':'Highest minus lowest retained resolved plateau blockade (nA). Zero for a one-level event.',
 'resolved_blockade_range_ratio':'Resolved blockade range divided by the single-file reference.',
 'resolved_weighted_std_nA':'Duration-weighted standard deviation of retained plateau heights (nA); a noise-reduced measure of within-event level variation.',
 'resolved_weighted_std_ratio':'Duration-weighted plateau-height SD divided by the single-file reference.',
 'ecd_nA_ms':'Measured event charge deficit: integral of measured blockade over the detected event (nA·ms).',
 'resolved_levels':'Number of retained sustained blockade levels after amplitude merging and duration QC.',
 'resolved_transitions':'Number of changes between retained resolved plateaus; equal to resolved_levels − 1 and retained mainly for audit/backward compatibility.',
 'transition_direction':'(Last plateau − first plateau)/(highest − lowest plateau); zero for one level. Positive means the event ends deeper than it begins.',
 'deepest_plateau_fraction':'Fraction of retained analysed duration spent in the deepest resolved plateau state.',
 'deepest_plateau_position':'Centre of the deepest resolved plateau expressed as a fraction of total detected event duration (0=start, 1=end).',
 'blockade_temporal_centroid':'Blockade-area centre in normalised event time. <0.5 means blockade is weighted earlier; >0.5 means later.',
 'blockade_skewness':'Duration-weighted skewness of the resolved plateau-height distribution. Zero when plateau-height variance is negligible.',
 'deep_time_fraction':'Fraction of retained analysed duration in plateaus above the user-defined deeper-blockade threshold.',
}


def _measured_ecd_ms(e,m):
    """Integral of measured blockade over the detected event in nA·ms."""
    b=e.baseline-e.current
    x=np.r_[e.bounds[0],e.time[m],e.bounds[1]]
    y=np.interp(x,e.time,b)
    return float(np.trapezoid(y,x)*1000.)


def level_features(events,source='Selected fits',min_duration_us=25.,min_height_nA=.1,noise_multiplier=3.,reference_nA=None,deep_threshold_nA=None,omit_short_boundaries=True):
    """Extract physically interpretable DNA event descriptors from resolved plateaus.

    Selected step fits always define the plateau boundaries.  ``source`` chooses whether
    the plateau height itself comes from the selected fit or from the measured samples
    lying inside those same boundaries.  Measured ECD is always calculated from the
    experimental current trace.
    """
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
            m=mask(e);t=e.time[m];fit=(e.baseline-e.fit)[m];observed=(e.baseline-e.current)[m]
            dt=float(np.median(np.diff(e.time)));duration=float(e.bounds[1]-e.bounds[0])
            raw_ends=np.r_[np.flatnonzero(np.abs(np.diff(fit))>1e-8)+1,len(fit)]
            if len(raw_ends)>max(8,int(len(fit)*.25)):raise ValueError('fit has too many sample-to-sample changes to use as resolved step plateaus; use a step fit')

            padding=(e.baseline-e.current)[~m]
            if len(padding)>=4:
                noise=float(np.median(np.abs(padding-np.median(padding)))/.67448975);noise_source='padding MAD'
            else:
                dif=np.diff(observed);noise=float(np.median(np.abs(dif-np.median(dif)))/(.67448975*np.sqrt(2)));noise_source='within-event first-difference MAD'
            noise=max(noise,1e-12)
            height_tol=max(float(min_height_nA),noise_multiplier*noise)
            minimum=max(min_duration_us*1e-6,3*dt)
            chosen=fit if source=='Selected fits' else observed

            segments=[];start=0
            for end in raw_ends:
                left=float(e.bounds[0]) if start==0 else float(t[start]);right=float(e.bounds[1]) if end==len(t) else float(t[end])
                value=float(chosen[start:end].mean())
                segments.append({'left':left,'right':right,'start':start,'end':int(end),'value':value});start=int(end)

            # Merge the closest adjacent levels until every surviving neighbour pair is
            # separated by more than the declared effective amplitude resolution.
            while len(segments)>1:
                differences=np.abs(np.diff([a['value'] for a in segments]));j=int(np.argmin(differences))
                if differences[j]>height_tol:break
                a,b=segments[j:j+2]
                joined={'left':a['left'],'right':b['right'],'start':a['start'],'end':b['end'],
                        'value':float(chosen[a['start']:b['end']].mean())}
                segments[j:j+2]=[joined]

            row.update(noise_scale_nA=noise,noise_estimator=noise_source,merge_height_nA=height_tol,effective_min_duration_us=minimum*1e6)
            short=[a['right']-a['left']<minimum-1e-12 for a in segments]
            statuses=[]
            for j,is_short in enumerate(short):
                edge=j==0 or j==len(segments)-1
                statuses.append('omitted_boundary' if is_short and edge and omit_short_boundaries else 'unresolved_boundary' if is_short and edge else 'unresolved_internal' if is_short else 'resolved')
            resolved=[a for a,status in zip(segments,statuses) if status=='resolved']
            omitted=sum(a['right']-a['left'] for a,status in zip(segments,statuses) if status=='omitted_boundary')
            analysed=sum(a['right']-a['left'] for a in resolved)
            row.update(resolved_levels=len(resolved),boundary_omission_applied=omitted>0,omitted_boundary_duration_us=omitted*1e6,
                       analysed_duration_ms=analysed*1000,analysed_fraction=analysed/duration,
                       short_boundary_levels=sum(status in ['omitted_boundary','unresolved_boundary'] for status in statuses),
                       short_internal_levels=statuses.count('unresolved_internal'))

            reason=''
            if any(status.startswith('unresolved') for status in statuses):
                reason='an internal plateau is shorter than the minimum resolved duration; inspect/refine the event' if 'unresolved_internal' in statuses else 'a boundary plateau is shorter than the minimum resolved duration; enable boundary omission or inspect/refine the event'
            elif not resolved:reason='no sustained plateau remains after boundary omission; inspect/refine the event'

            # Preserve every merged level for auditing, even if it is omitted/unresolved.
            for j,(a,status) in enumerate(zip(segments,statuses)):
                item={'event_index':e.index,'level_order':j,'start_from_event_ms':(a['left']-e.bounds[0])*1000,
                      'duration_ms':(a['right']-a['left'])*1000,'blockade_nA':a['value'],'source':source,
                      'level_status':status,'used_for_features':status=='resolved' and not reason,'event_eligible':not bool(reason)}
                if reference_nA is not None:item['blockade_ratio']=a['value']/reference_nA
                if deep_threshold_nA is not None:item['above_deeper_threshold']=a['value']>deep_threshold_nA
                sequences.append(item)
            if reason:raise ValueError(reason)

            heights=np.array([a['value'] for a in resolved],float)
            widths=np.array([a['right']-a['left'] for a in resolved],float)
            mids=np.array([.5*(a['left']+a['right']) for a in resolved],float)
            total=float(widths.sum())
            weighted_mean=float(np.sum(heights*widths)/total)
            weighted_var=float(np.sum(widths*(heights-weighted_mean)**2)/total)
            weighted_std=float(np.sqrt(max(weighted_var,0.)))
            span=float(np.ptp(heights))
            direction=float((heights[-1]-heights[0])/span) if len(resolved)>1 and span>height_tol else 0.

            deepest=float(heights.max());deep_idx=np.flatnonzero(np.isclose(heights,deepest,rtol=1e-10,atol=1e-12))
            deepest_fraction=float(widths[deep_idx].sum()/total)
            deepest_position=float(np.sum(widths[deep_idx]*mids[deep_idx])/widths[deep_idx].sum())
            deepest_position=float((deepest_position-e.bounds[0])/duration)

            # The temporal centroid treats blockade as a non-negative 'mass' distributed
            # through time.  If unusual signed levels make total blockade non-positive,
            # fall back to a duration-only centre of the retained region.
            area_weights=heights*widths
            if np.sum(area_weights)>1e-15:
                temporal_centroid=float(np.sum(area_weights*((mids-e.bounds[0])/duration))/np.sum(area_weights))
            else:
                temporal_centroid=float(np.sum(widths*((mids-e.bounds[0])/duration))/total)

            if weighted_std>1e-12:
                skew=float(np.sum(widths*(heights-weighted_mean)**3)/total/(weighted_std**3))
            else:skew=0.

            row.update(
                resolved_weighted_mean_nA=weighted_mean,
                deepest_plateau_nA=deepest,
                resolved_blockade_range_nA=span,
                resolved_weighted_std_nA=weighted_std,
                ecd_nA_ms=_measured_ecd_ms(e,m),
                resolved_transitions=len(resolved)-1,
                transition_direction=direction,
                deepest_plateau_fraction=deepest_fraction,
                deepest_plateau_position=deepest_position,
                blockade_temporal_centroid=temporal_centroid,
                blockade_skewness=skew,
                duration_ms=duration*1000,
                log10_duration_ms=float(np.log10(duration*1000.)),
                physical_eligible=True)

            if reference_nA is not None:
                row['resolved_weighted_mean_ratio']=weighted_mean/reference_nA
                row['deepest_plateau_ratio']=deepest/reference_nA
                row['resolved_blockade_range_ratio']=span/reference_nA
                row['resolved_weighted_std_ratio']=weighted_std/reference_nA
            if deep_threshold_nA is not None:
                row['deep_time_fraction']=float(widths[heights>deep_threshold_nA].sum()/total)
                row['threshold_sensitive']=bool(np.any(np.abs(heights-deep_threshold_nA)<=height_tol))
        except ValueError as ex:
            row['physical_exclusion_reason']=str(ex)
        rows.append(row)

    seq_columns=['event_index','level_order','start_from_event_ms','duration_ms','blockade_nA','source','level_status','used_for_features','event_eligible']
    if reference_nA is not None:seq_columns.append('blockade_ratio')
    if deep_threshold_nA is not None:seq_columns.append('above_deeper_threshold')
    return pd.DataFrame(rows),pd.DataFrame(sequences,columns=seq_columns)

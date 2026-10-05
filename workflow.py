"""Explicit fit selection and roundtrip exports for the guided workflow."""
RELEASE_VERSION='0.6.4'
from dataclasses import replace
import io,json,zipfile
import numpy as np
from analysis import pack_events,npz_bytes,link_dataset,safe_settings

def active_events(events,refs):
    return [replace(e,fit=refs[e.index]['fit'].copy(),levels=refs[e.index]['levels'].copy(),widths=refs[e.index]['widths'].copy(),analysis=None) if e.index in refs else e for e in events]

def signal_events(events,source):
    if source=='Measured trace':return events
    missing=[e.index for e in events if e.fit is None]
    if missing:raise ValueError(f'{len(missing)} events have no selected fit. Refine them or use measured traces.')
    return [replace(e,current=e.fit.copy()) for e in events]

def fitting_bytes(events,refs,settings,meta):
    active=active_events(events,refs)
    with np.load(io.BytesIO(pack_events(active,settings,meta)),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    for e in events:
        if e.fit is not None:arrays[f'INPUT_FIT_{e.index}']=e.fit
        if e.levels is not None:arrays[f'INPUT_LEVELS_{e.index}']=e.levels
        if e.widths is not None:arrays[f'INPUT_WIDTHS_{e.index}']=e.widths
    arrays['refined_event_indices']=np.array(sorted(set(refs)&{e.index for e in events}),int)
    return npz_bytes(arrays)

def match_raw(events,raw,tolerance):
    x=np.array([[e.bounds[0]] for e in raw]);mapping,status=link_dataset(events,x,0,tolerance)
    return {i:raw[row] for i,row in mapping.items()},status

def bundle(events,rawmap,refs,dataset,mapping,settings,dsettings,table,meta,sequences=None):
    ids={e.index for e in events};rows=[mapping[e.index] for e in events if e.index in mapping];matched=[e.index for e in events if e.index in mapping]
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('selected.eventfitting.npz',fitting_bytes(events,refs,settings,meta))
        raw=[rawmap[e.index] for e in events if e.index in rawmap]
        z.writestr('selected.eventdata.npz',pack_events(raw,{},meta))
        z.writestr('selected.dataset.npz',npz_bytes({'X':dataset[rows],'original_dataset_rows':np.array(rows,int),'fitting_event_ids':np.array(matched,int),'settings':np.array(json.dumps(safe_settings(dsettings)))}))
        z.writestr('event_results.csv',table.to_csv(index=False));z.writestr('provenance.json',json.dumps(meta,indent=2))
        if sequences is not None and len(sequences):z.writestr('resolved_levels.csv',sequences[sequences.event_index.isin([e.index for e in events])].to_csv(index=False))
        z.writestr('README.txt','Selected fits are the active fits in selected.eventfitting.npz; INPUT_FIT_* preserves uploaded fits. Measured traces are unchanged. selected.eventdata.npz uses the numeric Shape Lab layout and preserves source raw-event IDs. Dataset X rows are original and are NOT recalculated from refined fits. CSV contains current measured results and selected-fit features. Mapping appears in CSV. Reload supported in Shape Lab; NanoSense re-import compatibility is not established.\n')
    return out.getvalue()

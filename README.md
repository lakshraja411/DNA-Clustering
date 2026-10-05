# DNA Event Lab 0.8.0

## Deploy

Replace the repository files together: `app.py`, `analysis.py`, `physical.py`, `plots.py`, `workflow.py`, **`representations.py`**, and `requirements.txt`. Keep these exact names at the repository root and use `app.py` as the Streamlit entry point. Commit and reboot Streamlit. Validation used Python 3.12 and the pinned dependencies.

## Suggested starting settings

Step 5 defaults to **Six physical descriptors → Standard deviation scaling → PCA + k-means → Retain at least 95% variance**. Inspect the elbow/silhouette scan, cluster sizes, physical distributions, member traces and subsampling results before selecting k. The initial k of 3 is a starting value, not a conclusion. k=1 is available when splitting is not useful.

The six descriptors are log10 detected duration, measured mean blockade, log10 deep/shallow plateau ratio, deepest-state occupancy, measured temporal centroid and resolved shape complexity. Physical descriptors use the declared plateau QC. Measured mean blockade now equals measured ECD divided by detected duration, using trapezoidal integration; the old code used the sample mean despite describing it as ECD/duration.

## Changes

- Standard scaling removes constants but preserves descriptors varying in a minority of events. The previous safeguarded robust scaler and its central-90%-spread exclusion remain available for comparison. Extreme observations can influence either choice.
- PCA defaults to the smallest number of components reaching 95% variance. Manual selection includes all available components. PC1-PC2 is a projection and may hide separation or overlap in other coordinates.
- Six seeded subsamples refit scaling, PCA and clustering. ARI compares their labels with the full-data solution on sampled event IDs. This is sampling stability, not held-out prediction accuracy or topology validation. Subsamples contain 80% of events, capped at 1,000. Failed repeats and small clusters are visible.
- k=1 has no silhouette or stability score. Dispersion scans include k=1; silhouette compares k>=2. Neither diagnostic establishes the existence of discrete populations.
- Exact input features, event IDs, scaling/PCA parameters and stability checks are exported with data and figure packs.

## Alternative representations

**Balanced waveform + amplitude + duration** uses 32 mean-normalised phase-bin averages, log10 mean blockade and log10 duration. Bin averages integrate the piecewise-linear selected signal over detected bounds and preserve its area. A shared waveform scale gives that entire block unit total variance. Each scalar block also has unit variance. This weighting is an explicit modelling choice.

**Waveform shape only** uses the same normalised shape block without amplitude or duration. It asks whether relative shapes resemble one another. A small normalisation denominator or noise can create unusual profiles; inspect member traces and tiny clusters. Outliers are not automatically removed or hidden.

Waveform modes use selected fits or measured traces and do not require successful plateau/ratio QC. A missing selected fit or non-positive mean blockade produces an explicit exclusion reason. Physical-level plots may show fewer events when descriptors are unavailable. Level composition uses only events with resolved levels; population plots use all events included by the selected representation.

These are exploratory alternatives, not a reproduction of NanoBoost's DWT pipeline. They were less stable than the new default on the supplied recording. Silhouettes from different representations use different distance definitions and cannot be ranked as biological accuracy.

## Preservation and figures

The original trace, event bounds and selected fit are preserved. Brief edge plateaus may be omitted from physical descriptors under the existing explicit option; short internal levels remain flagged. Occupancy uses retained analysed duration, while event duration remains total detected duration. Ordered level exports retain omitted/unresolved levels and reasons.

Plots show actual assignments, measured distributions, sampled members, median profiles and real example events. No DNA topology cartoons or microscopy are inferred. The paper's feature clusters and time-series clusters are separate analyses with different populations.

## Checks

Run `python check_physical.py` and `python check_representations.py` for known-signal checks. `python check_app.py` additionally needs the original NPZ recording in the sibling `upload` folder. `benchmark_recording.py` also needs the original uploaded `physical(4).py` to reproduce the previous extraction.

See `CLUSTERING_REVIEW.md` and `validation/` for the same-event comparison. No labelled topology benchmark or validation on the latest 650-event recording was available.

## Boundary handling correction (0.5.1)

Step 5 → Physical feature settings → **Omit short boundary plateaus from physical features** is enabled by default and can be disabled for comparison. After adjacent height merging, only the first and last merged plateaus can be omitted if shorter than the effective minimum duration. The algorithm does not repeatedly peel further short segments: short internal plateaus still require review. A wholly short event has no sustained core and remains excluded. This is an explicit analysis policy, not proof that the boundary segment is an artefact.

The original trace, event boundaries and saved/refined fit are unchanged. Plateau depth, transitions and direction use the retained resolved levels. Deeper-level occupancy uses retained analysed duration; the duration feature remains total detected duration. Audit columns include analysed duration/fraction, omitted boundary time, short boundary/internal counts and a boundary omission flag. Ordered level exports retain all merged levels and identify omitted, unresolved and used levels. Step 5 offers flagged-event inspection with amber shading. Cluster exports include omitted levels; excluded-event downloads include unresolved levels when extraction was possible.

Compare clustering with boundary omission enabled/disabled before using it in a final analysis. Report exclusions and omitted durations. A brief edge may be real; omission reflects insufficient resolution under the declared thresholds.

# DNA Event Lab 0.5.1

Replace all application files in your GitHub repository with the files in this folder, including the new `physical.py`. Keep `app.py` as your Streamlit entry point. Commit the changes, then reboot your Streamlit app if it has not refreshed.

## Physical features

PCA clustering now defaults to physical level features:

- Deepest sustained plateau blockade, or its ratio to a supplied calibrated single-file reference.
- Number of resolved transitions between adjacent plateaus.
- Signed end-to-start level change divided by the resolved height range (zero for a single level).
- Optional fraction of duration above a user-defined deeper-blockade threshold.
- Optional event duration.

Selected step fits supply the boundaries. Selected-fit mode uses fitted plateau heights; measured-trace mode uses measured sample means within those same boundaries. Adjacent heights merge at the larger of the absolute height threshold and noise multiplier times a robust noise scale (padding MAD, or first-difference MAD when padding is insufficient). The default settings are provisional analysis choices; adjust for instrument bandwidth and sampling. Minimum plateau duration is at least three sampling intervals. Continuous or missing step fits, short internal plateaus and events without a sustained core are excluded from this model rather than assigned fabricated plateau features. Use step refinement to review those events.

Eligibility is visible before clustering. Feature audit CSV includes resolution settings and exclusion reasons. Each physical cluster exports ordered levels with status and whether they were used for features; the combined download also includes excluded traces, fits and available dataset rows. Duration-normalised waveform/DTW methods and legacy features remain available for comparison. Changing settings invalidates stale cluster results.

No descriptor proves a fold, strand count or DNA topology. Calibrate any reference and threshold for the recording conditions. Threshold-sensitive events are flagged; compare alternative thresholds and inspect the raw signals. Automatic k selection remains exploratory and does not test whether a single population would suffice.

Run `python check_physical.py` for known-level extraction checks. `python check_app.py` additionally uses the supplied recording in the sibling upload folder to check matching, UI, preservation and exports.

---

# DNA Event Lab · 0.4.3

A guided Streamlit workflow for inspecting DNA nanopore events, refining fits, plotting physical distributions, suggesting candidate signal groups, and exporting event subsets. No synthetic demonstration or power spectral density is included in the interface.

## Update your existing app

Copy the files in this folder to the root of your GitHub repository, replacing the corresponding files. `workflow.py` and `plots.py` must sit beside `app.py`. Replace `requirements.txt` as well. Keep your existing `.streamlit/config.toml`. Do not upload experimental data to GitHub.

Entry point: `app.py`.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Six-step workflow

1. **Load files:** upload matching dataset, eventdata and eventfitting NPZ files. Events are linked by unique start timestamps rather than row position. Files remain mounted while moving between tabs during the same Streamlit session.
2. **Inspect events:** view measured current/blockade, uploaded fit, refined fit and the independently saved eventdata trace.
3. **Refine and save fits:** recalculate segment means, find new PELT levels, or fit a rounded Gaussian pulse. Original current and uploaded fits are preserved.
4. **Current–duration plots:** scatter and event-count heatmaps using measured traces or selected fits, with CSV and publication figure export.
5. **Cluster events:** automatic feature clustering is now the default. The app can still run manual PCA + agglomerative, PCA + k-means, waveform k-means and DTW time-series k-means.
6. **Save clusters:** export assignments and one or all event groups with their fitting, eventdata and dataset subsets plus provenance.

## Automatic clustering and legacy comparison

The optional **Legacy features (comparison)** mode offers the following signal-derived features:

- duration
- mean blockade
- peak blockade
- within-event blockade standard deviation
- relative peak position
- FWHM
- early-versus-late blockade difference
- Haar wavelet approximation mean
- Haar wavelet approximation standard deviation
- Haar wavelet detail energy

The clustering pipeline is:

`event features → remove constants / near-duplicates → z-score standardise → PCA → test k → agglomerative clustering`

Near-duplicate selected features with absolute Pearson correlation `|r| >= 0.98` are pruned in automatic mode so nearly identical descriptors do not repeatedly weight the same physical property. PCA then retains the **smallest number of components reaching the requested cumulative explained variance** (95% by default). Clustering uses every retained PC. The PC1–PC2 view accompanies profiles, measured summaries and original traces in step 5.

Automatic mode tests `k = 2 ... kmax` (default `kmax = 8`). Each candidate is evaluated using:

- **Silhouette score** — higher is better.
- **Calinski–Harabasz score** — higher is better.
- **Davies–Bouldin score** — lower is better.
- **Subsampling stability** — adjusted Rand index (ARI) comparing the full-data grouping with repeated subsamples (target 80%, capped at 600 events) that re-fit scaling, PCA and clustering. The interface reports the actual fraction.

The selected cluster count uses a weighted rank consensus: silhouette ×2, stability ×2, Calinski–Harabasz ×1 and Davies–Bouldin ×1. The weighting intentionally gives more importance to direct separation and reproducibility than to any single compactness statistic. The app shows the full diagnostic table instead of hiding the selection process.

If the selected solution has a silhouette below 0.25 or resampling stability below 0.6, the interface displays a warning that the apparent cluster structure is weak. These thresholds are diagnostics, not biological decision rules.

## PCA interpretation

An optional expandable PCA loading plot is available for feature-based clustering. Large absolute loadings identify which original event descriptors contribute most strongly to PC1 and PC2. The overall sign of a component is arbitrary; relative loading signs and magnitudes describe how features combine.

Feature clustering uses standardised values. Therefore a large loading is not simply caused by a feature having units with larger numerical values. Constant and automatically pruned redundant columns are omitted from the loading plot.

## Scientific interpretation

The output groups are **unsupervised signal families, not automatic DNA topology labels**. Linear, folded, complex or other physical interpretations still require inspection of representative current traces and consistency across experimental conditions.

Step 4 retains the uncoloured current–duration distribution. Step 5 uses a cluster summary table with measured durations and blockades, member profiles and original event traces. Profile downloads contain PDF, SVG, PNG and numeric summaries.

Cluster IDs are ordered by increasing mean waveform blockade for feature clustering so Group 0 is the shallowest mean profile. IDs are still labels, not biological categories.

Automatic cluster-number selection identifies the best-supported grouping among the tested candidates; it does not prove that nature contains exactly that many discrete states. Continuous manifolds can still be partitioned by clustering algorithms. Use the diagnostic table, PCA loadings and representative traces together.

## Formats and limitations

The supported NanoSense layouts, timestamp matching, refinement/export behaviour and file-preservation rules are unchanged from v0.3. Dataset `X` rows are preserved as uploaded and are not recalculated after refinement. Generated fitting/event subsets reload in this app; external NanoSense re-import compatibility is not established.

Upload data and generated downloads live in the Streamlit session on its server. A browser/session reset loses unsaved refinements. Keep the original three source files separately.

## Verification

`analysis.py` can be syntax-checked without data using:

```bash
python -m py_compile analysis.py app.py workflow.py plots.py
```

`check_app.py` is the integration test for the supplied private NanoSense fixture files. It expects those three sample exports in the same `upload` fixture location used by the previous version; experimental data are not bundled here.

## Review and navigation update

Previous/Next arrow buttons appear above each step and at the bottom of completed steps. Files and analysis remain in the same session. Next from loading requires a confirmed recording; Next from clustering requires a computed grouping. The sidebar remains available for direct navigation.

Automatic mode is labelled exploratory. It suggests the best weighted-rank candidate among k=2…kmax and cannot test the one-population case. Extra warnings flag missing stability repeats, upper-boundary choices and very small groups. Score weights and warning thresholds are explicit heuristics. Scaling/PCA parameters, retained feature names, diagnostic tables and effective subsample fraction are now included in provenance. See CLUSTERING_REVIEW.md for scientific interpretation and limitations.

## PC1–PC2 view restored in 0.4.2
Step 5 includes the PCA scatter with cluster colours, event IDs and measured quantities on hover, variance labels, and coordinate CSV export. Feature clustering may use more than the two shown coordinates; waveform methods use PCA only for display. A one-component feature solution explicitly marks PC2 as zero. The coloured current–duration cluster scatter remains removed.

## Paper-style figures in 0.4.3

Step 5 now shows faint sampled member profiles with a red representative curve for each cluster, a comparison of representatives, PC1–PC2 points with optional convex hulls and black projected-group mean markers, and one real example event per group in actual milliseconds. Profile panels share a common y range. Hulls are visual outlines, not confidence regions, and are omitted for collinear or undersized groups. PCA points remain unmodified and outlines can overlap.

Each cluster panel displays at most 40 uniformly sampled members (seed 42); its representative uses all members. Representatives are means for feature/waveform k-means methods and barycentres for DTW. Absolute durations are removed only in the profile panels. The actual-time panel shows one real event nearest to the representative by pointwise profile distance, aligned at detected start; its original sample times/duration are retained. It is an example event, not an averaged centroid. No DNA topology cartoons or microscopy are generated.

Use **Prepare combined publication figure** to export a labelled multi-panel PDF, SVG and 600 dpi PNG, with PCA coordinates, representative curves, actual-time examples, selected member IDs and a caption. No smoothing is added for presentation. Time-normalised averages can be rounded even when individual fitted events are steps.

For this update, replace **both app.py and plots.py together**. A previous plots.py will lack the newly imported figure helpers. Keep the reviewed analysis.py/workflow.py and requirements. These exports have been rendered for visual inspection and exercised through the Streamlit UI test.

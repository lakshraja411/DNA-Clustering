# DNA Event Lab · 0.4

A guided Streamlit workflow for inspecting DNA nanopore events, refining fits, plotting physical distributions, automatically discovering signal groups, and exporting event subsets. No synthetic demonstration or power spectral density is included in the interface.

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

## Automatic clustering in v0.4

Automatic mode uses **all available continuous signal-derived event features by default**:

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

Near-duplicate selected features with absolute Pearson correlation `|r| >= 0.98` are pruned in automatic mode so nearly identical descriptors do not repeatedly weight the same physical property. PCA then retains the **smallest number of components reaching the requested cumulative explained variance** (95% by default). The PC1/PC2 plot is only a visual projection; clustering uses every retained PC.

Automatic mode tests `k = 2 ... kmax` (default `kmax = 8`). Each candidate is evaluated using:

- **Silhouette score** — higher is better.
- **Calinski–Harabasz score** — higher is better.
- **Davies–Bouldin score** — lower is better.
- **Subsampling stability** — adjusted Rand index (ARI) comparing the full-data grouping with repeated 80% subsamples that re-fit scaling, PCA and clustering.

The selected cluster count uses a weighted rank consensus: silhouette ×2, stability ×2, Calinski–Harabasz ×1 and Davies–Bouldin ×1. The weighting intentionally gives more importance to direct separation and reproducibility than to any single compactness statistic. The app shows the full diagnostic table instead of hiding the selection process.

If the selected solution has a silhouette below 0.25 or resampling stability below 0.6, the interface displays a warning that the apparent cluster structure is weak. These thresholds are diagnostics, not biological decision rules.

## PCA interpretation

The app now displays a PCA loading plot for feature-based clustering. Large absolute loadings identify which original event descriptors contribute most strongly to PC1 and PC2. The sign of a PCA loading is arbitrary; its magnitude and relationships across features are the useful parts.

Feature clustering uses standardised values. Therefore a large loading is not simply caused by a feature having units with larger numerical values. Constant and automatically pruned redundant columns are omitted from the loading plot.

## Scientific interpretation

The output groups are **unsupervised signal families, not automatic DNA topology labels**. Linear, folded, complex or other physical interpretations still require inspection of representative current traces and consistency across experimental conditions.

The physical duration-versus-mean-blockade plot always uses measured event quantities even if selected fits drive clustering. This makes it possible to see what the machine-derived grouping looks like in experimentally familiar coordinates.

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

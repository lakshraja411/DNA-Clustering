# Clustering review — v0.8.0

Version 0.8.0 is a deliberate reset. Primary clustering now uses the original NanoSense `dataset.npz` feature matrix rather than a custom DNA-conformation feature vector.

For the known NanoSense layout, the default clustering descriptors are columns 0–6: height, FWHM, height-at-FWHM, area, width, skewness and kurtosis. Baseline and event-time-like columns are excluded by default. Raw column indices remain visible and selectable because the schema can vary between NanoSense versions.

The default mathematical pipeline is MinMax scaling to `[-1,1]`, PCA, then either Ward agglomerative clustering or k-means. Two PCs are the default, with scree/loadings and a selectable PCA dimension for sensitivity analysis. Optional correlation pruning is disabled by default so the selected original dataset descriptors are not silently changed.

Resolved-level features are now interpretation-only. This is important: PELT/step-fit resolution, fold contrast, level count, occupancy and related quantities can support or challenge the resulting signal families, but they cannot create those families in this mode. Failed physical-feature extraction does not exclude an otherwise valid matched dataset event.

Recommended interpretation sequence: first inspect the raw feature correlation and scree plots; then choose a defensible PCA dimension and cluster count; then inspect PC1–PC2, blockade-versus-dwell, member traces, representative real events, cluster populations and resolved-level composition. A visually attractive PCA partition is not sufficient on its own.

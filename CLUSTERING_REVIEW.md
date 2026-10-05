# 0.7.1 critical clustering correction

Version 0.7.0 allowed a measured temporal-centroid descriptor to be formed from signed blockade samples. Baseline noise can make some blockade samples negative; if positive and negative samples nearly cancel, the centroid denominator becomes very small and the resulting value can lie far outside its physically meaningful 0–1 range. A handful of such events can dominate ordinary PCA variance, produce PC1≈100%, and be isolated as clusters of only one or a few events. Those solutions are numerical artefacts and must not be interpreted physically.

Version 0.7.1 computes temporal centroid from positive measured blockade mass only and constrains it to [0,1]. The primary clustering vector is reduced to five deliberately complementary descriptors: log10 duration, log10 positive measured mean blockade, bounded fold contrast `(deepest-shallowest)/(deepest+shallowest)`, deepest-state occupancy, and the positive-blockade temporal centroid. Raw deep/shallow ratio, resolved shape complexity, ECD, level count and transition descriptors remain available for post-cluster interpretation.

All retained PCA components are used for Ward or k-means clustering. With every retained component included, PCA is only an orthogonal rotation of the safeguarded robust-scaled feature space; no information is discarded according to an arbitrary PC count. PC1–PC2 remains a visual projection only.

Synthetic descriptor tests and a 657-event LiCl fitting export smoke test were run locally. On that real export the five retained PCs explained approximately 47.8%, 31.5%, 10.9%, 7.4%, and 2.3% of scaled-feature variance; no PC1≈100% blow-up occurred. These numbers validate numerical behaviour only, not biological cluster identity.

# 0.7.0 clustering model note

The primary clustering representation is deliberately frozen to six physically interpretable DNA descriptors rather than a broad feature library. A stable deep/shallow ratio is required; events whose shallowest resolved blockade is not above the local robust noise scale for multilevel events remain auditable but are excluded from this model. This is an unsupervised signal-family model, not a validated topology classifier. Cluster number remains a user decision informed by elbow/silhouette diagnostics and waveform/physical interpretation.

# Version 0.7.0 PCA scaling correction

The v0.6.3 median/IQR scaling could become numerically unstable when a plateau-derived feature had an almost-zero but non-zero IQR. This could generate billion-scale PCA coordinates and an artificial PC1≈100% result. v0.7.0 treats a feature with effectively zero Q05-Q95 spread as uninformative for the global PCA and otherwise uses a safeguarded robust denominator `max(IQR, 0.25*(Q95-Q05))`. Raw physical values remain unchanged and remain available for interpretation.

# Version 0.6.2 clustering design

The primary PCA/clustering space is intentionally restricted to five complementary continuous descriptors: `log10_duration_ms`, `resolved_weighted_mean_nA`, `deepest_plateau_nA`, `resolved_weighted_std_nA`, and `blockade_temporal_centroid`. These represent kinetics, typical sustained blockade, maximum sustained blockade, resolved-level heterogeneity, and temporal asymmetry. ECD, discrete level count, deepest-state occupancy/position, range and transition direction are interpretation/QC descriptors rather than default PCA inputs.

Scaling is now median/IQR based (`RobustScaler`) before PCA. This reduces, but does not eliminate, sensitivity to extreme events; PCA itself remains variance based, so outlier inspection is still required. Waveform representatives are pointwise medians of all cluster members. Member panels show a reproducible sample of actual traces with no percentile envelope. The PCA display has no convex-hull fill by default.

# Version 0.5 addendum

The review below describes the previous legacy-feature engine. Version 0.5 defaults to resolved physical level descriptors; its extraction checks validate signal calculations, not topology assignments. Resolution settings and calibrated thresholds require experimental justification. Exclusions alter the analysed population and must be reported. No full-recording physical clustering accuracy benchmark was performed.

# Review of the uploaded clustering logic

The implemented sequence is coherent for exploratory unsupervised grouping: extract event descriptors, remove uninformative/redundant columns, standardise, retain PCA coordinates and apply Ward clustering. Candidate cluster counts are compared using internal geometry and subsampling reproducibility. This is a candidate signal-grouping method; topology accuracy has not been established.

## What is reasonable

- Standardisation gives features with different numerical scales comparable influence.
- Removing constants and strongly correlated columns reduces some duplicate weighting. Correlation pruning is greedy and depends on feature order; retained names should be reported.
- Selecting enough PCA coordinates to retain 95% of variance avoids forcing clustering into a two-dimensional display. Variance retained measures representation of the standardised data; it is not classification accuracy and can discard low-variance distinguishing signals.
- Testing several k values is more informative than silently fixing one.
- Re-fitting preprocessing during subsampling tests sensitivity of the complete pipeline. ARI on shared events compares assignments without depending on cluster names.
- Profiles and individually browsable original traces, alongside measured summaries and across-recording replication, support interpretable follow-up.

## Limits of automatic selection

The rank combination is a predefined heuristic (weights 2,1,1,2), not an independently validated estimate of the true population count. Silhouette, Calinski–Harabasz and Davies–Bouldin are related geometric diagnostics, rather than three independent biological confirmations. A stable partition can split a continuous cloud consistently. The search begins at two, so it always proposes multiple groups when valid candidates exist. It does not compare against a calibrated one-population null. Scores should guide investigation, not decide topology labels.

The stability sample target is 80%, capped at 600 events. For 1,400 events it is 42.9%, not 80%. The updated interface and provenance report the actual sample size and successful repeats. This is resampling within one recording, not validation on independent pores. Six repeats provide a limited estimate; the reported SD is not a confidence interval.

Z-score scaling is sensitive to extreme events. Equal standardised variance gives every retained descriptor influence, including weak/noisy descriptors. Using all features does not establish that they are all informative. Compare a predeclared physically interpretable subset (duration, mean/peak blockade, variation and early/late difference) against the full set and measured traces against fits. Treat this as a sensitivity analysis rather than choosing whichever view looks prettiest.

Haar coefficients depend on sampling interval, event length, wavelet level and representation. These optional summaries are not the paper's validated wavelet preprocessing. Flat step fits can set variation and early/late difference to zero, and give ambiguous first-maximum peak positions. Their grouping can reflect the fitting model itself. Comparing measured-trace and selected-fit assignments helps reveal that dependency.

Waveform k-means remains a useful comparison: it preserves amplitude and profile order but removes absolute duration. DTW additionally permits timing alignment and can conceal meaningful dwell differences. None is universally superior. Features/PCA and waveform approaches answer different similarity questions.

## Interpretation to establish next

Inspect a fixed random selection of events per group plus boundary/outlying members. Ask which measurable property differs consistently. Compare assignments across sensible parameter changes and independent recordings, controlling pore and acquisition differences. Establish an occupancy/blockade reference before describing folded passage. Group count, fitted step count and topology count are different quantities.

## Primary documentation

- scikit-learn clustering and internal evaluation: https://scikit-learn.org/stable/modules/clustering.html
- StandardScaler and outlier sensitivity: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

The review also fixed rejection of a valid one-feature space after redundant-feature pruning. The initial review removed coloured cluster scatter and PCA scatter plots from step 5. Subsequent requested updates restored the PCA scatter and added paper-style profile panels and a composite figure. Step 4 distributions, arrow navigation, selection limitations and saved diagnostics remain available. The underlying weighted-rank rule is retained so results remain comparable to the uploaded version.

## Checks performed for this update

The six-step UI test passed, including Next/Previous callbacks, session retention, automatic clustering, refinement and cluster downloads. A separated two-population numerical fixture was recovered exactly (ARI 1.0), including after correlated-feature pruning left one informative feature. A single Gaussian cloud was still split into five groups; low separation and stability warnings appeared. This directly demonstrates why the auto suggestion is not a test for distinct populations.

On the supplied 1,400-event fitting export with uploaded fits and all descriptors, default automatic settings suggested two groups, retaining six PCs. Silhouette was about 0.312; stability samples used 600/1,400 events (42.9%). These numbers describe this run's signal geometry, not biological identities, and can change after refinement or feature/source changes. There is no independent topology benchmark here.

---

# Version 0.6 Hart-style DNA clustering redesign

Step 5 now uses a DNA-specific resolved-level feature model rather than the old automatic weighted-rank engine. The interface exposes only PCA + Ward agglomerative and PCA + k-means as primary clustering algorithms. The same five primary features, constant/near-duplicate pruning (|r| >= 0.98), median/IQR scaling and PCA preprocessing are used for both algorithms.

The primary clustering set is fixed to log10 dwell time, duration-weighted resolved blockade, deepest sustained blockade, duration-weighted plateau-height SD, and blockade temporal centroid. ECD, resolved level count, range, deepest-state occupancy/position and transition direction are interpretation/QC descriptors. A calibrated single-file reference switches the principal amplitude descriptors to ratios; an optional deeper-blockade threshold adds deep-time fraction.

The cluster-count workflow now mirrors the Hart/NanoBoost presentation more closely: PCA dimensionality is inspected with a scree/cumulative-variance plot, and candidate k values are compared with within-cluster dispersion (elbow) and silhouette. The app deliberately does not force agreement between these diagnostics. The chosen k is explicit and manual. Calinski-Harabasz and Davies-Bouldin remain secondary numerical checks.

The result view adds a physical feature/population table, dwell-versus-weighted-blockade plot, cluster-wise physical distributions, level composition, deepest-state occupancy, cluster-member waveform panels with median representatives, representative-profile overlay, PC1-PC2 map, optional 3-PC view, and a Ward dendrogram for agglomerative clustering. The export ZIP writes main-style and supplementary-style PDF/SVG/600-dpi PNG panels plus the underlying tables. These plots are analytical analogues generated from the user's data; they do not reproduce published artwork.

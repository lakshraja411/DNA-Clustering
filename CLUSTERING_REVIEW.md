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

# Review and measured comparison — 0.8.0

## Scope and paper comparison

Reviewed the latest uploaded 0.7.0 code and Hart et al., *NanoBoost: a wavelet transform-enhanced machine learning algorithm for nanopore sensing*, npj Biosensing (2026), DOI https://doi.org/10.1038/s44328-026-00105-x, particularly Figure 5, Table 5 and Methods.

The paper describes broader event/wavelet features, min-max scaling, PCA plus agglomerative clustering for DNA, and a separate time-series k-means analysis for representative profiles. Figure 5 reports 49/32/19% for the feature clusters and 54/31/15% for the time-series clusters. Those panels do not simply show medians from identical memberships. Our explicit physical descriptors and new waveform representations do not reproduce the paper's DWT pipeline. Ward linkage also favours compact groups; it should not be described as free of cluster-shape preferences.

Matching a paper's algorithm or PC count cannot ensure distinct populations in another experiment. The paper describes the conformation assignments as interpretation-level insight requiring further targeted validation.

## Issues found

- The central-90%-spread filter could remove a descriptor that varies only in a minority of events.
- Safeguarded median/IQR scaling still let isolated extreme observations determine a group in this recording.
- Two PCs were the default and manual selection was capped at five.
- Repeat-seed k-means agreement was shown, without a refitted-pipeline subsampling check.
- Every analysis required successful plateau/ratio QC. Waveform modes now offer a separate analysis when those descriptors are unavailable.
- Measured mean blockade was documented as ECD/duration but calculated as a sample mean. It now matches the stated integral definition.

Occupancy and other level descriptors still depend on fit segmentation and resolution settings. This release does not establish their robustness to experimental bandwidth or threshold changes.

## Same-event experiment

Source: earlier supplied CsCl, 400 mV lambda-DNA recording, 1,400 loaded events. No new raw data accompanied the latest code upload. The comparison therefore does not establish performance on the latest 650-event experiment, other salts, voltages, pores or DNA lengths.

All eight comparisons use the SAME 1,233 events in the intersection of physical and waveform eligibility. Selected uploaded fits were used; no new refinement was applied. k=3 was fixed for comparison, not inferred as a biological count. Previous descriptor values came from the uploaded 0.7 physical module; updated values include the measured-mean correction.

Six seeded repeats each selected 986 events, refit preprocessing and clustering, and compared labels against the full-data solution on sampled IDs. ARI is invariant to arbitrary cluster numbering. These results informed the new starting settings and constitute development evidence, not independent validation.

| Representation / scaling | Algorithm | PCs | Group sizes | Mean ARI | Lowest ARI |
|---|---|---:|---|---:|---:|
| Previous six / robust | Ward | 2 | 1, 542, 690 | 0.559 | 0.374 |
| Previous six / robust | k-means | 2 | 1, 487, 745 | 0.792 | 0.479 |
| Six / standard / 95% variance | Ward | 5 | 331, 604, 298 | 0.598 | 0.464 |
| Six / standard / 95% variance | k-means | 5 | 402, 564, 267 | 0.967 | 0.947 |
| Balanced waveform / 95% variance | Ward | 9 | 83, 269, 881 | 0.636 | 0.191 |
| Balanced waveform / 95% variance | k-means | 9 | 32, 368, 833 | 0.793 | 0.417 |
| Shape only / 95% variance | Ward | 12 | 1, 1, 1231 | 0.503 | 0.005 |
| Shape only / 95% variance | k-means | 12 | 1, 425, 807 | 0.706 | 0.004 |

The new default is better supported by repeatability here. It is not proven more accurate biologically. Its silhouette was approximately 0.293, and groups overlap in PC1-PC2. Previous k-means achieved approximately 0.410 despite isolating one event; shape-only Ward achieved approximately 0.905 while producing TWO single-event groups. Maximising silhouette or seeking a visually clean panel is insufficient.

The new default retained five PCs, explaining about 98.1% variance. PC1-PC2 displays only about 68.3%. Projection overlap can coexist with repeatability in the full clustering space.

## Checks and limits

Checks cover known plateau extraction, edge handling, measured mean, area-preserving waveform bins, shape invariance to amplitude/time scaling, block weights, minority-feature preservation, known-shape recovery, refitted-subsample agreement, missing-fit handling and k=1. Streamlit AppTest checks file matching, raw/refined-fit preservation, all representations, figure/data exports, navigation and stale-result invalidation.

Next experimental checks should compare independent recordings and measured traces versus fits, inspect outliers and threshold sensitivity, and relate groups to controls or independently supported conformations. Cluster IDs alone cannot establish linear, folded or knotted DNA.

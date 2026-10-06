# DNA Event Lab v0.9.1 — cross-salt heatmap summary

This release keeps the v0.9.0 clustering and five-salt comparison workflow unchanged and adds **family × electrolyte heatmaps** to Step 7.

## New in Step 7

After mapping recording-specific clusters to common Family A/B/C… labels, the app can now plot heatmaps for:

- population fraction (%)
- median dwell time (ms)
- median blockade (nA)
- median ECD (nA·ms), when present in the comparison packages
- relative blockade to a selected reference family, when enabled

Heatmap colour scaling can be:

- **Raw values** — keeps the physical units and is the default for reporting.
- **Row z-score** — standardises each family across salts, useful for highlighting salt-dependent changes within a family.
- **Column z-score** — standardises each salt across families, useful for highlighting which families are relatively high/low within one electrolyte.

The z-score modes affect only the heatmap display. They never alter clustering, family matching, or the saved raw summary values. Hover text retains the raw physical value.

## Publication export

The cross-salt publication ZIP now includes individual heatmaps, a compact multi-heatmap summary (up to four selected metrics), and CSV matrices for both raw and displayed heatmap values, in addition to the existing waveform, population, dwell-time and blockade figures.

## Existing pipeline retained

Primary clustering remains:

`matched original dataset rows → selected NanoSense X columns → MinMax scaling to [-1,1] → PCA → Ward agglomerative OR k-means`

Family correspondence across salts remains user-confirmed; Cluster 0 in one electrolyte is not automatically assumed homologous to Cluster 0 in another.

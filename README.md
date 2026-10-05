# DNA Event Lab v0.9.0 — five-salt comparison

This release keeps the v0.8.x dataset-feature clustering unchanged and adds a cross-salt comparison workflow.

## Existing clustering

Step 5 still uses the original NanoSense dataset feature matrix: selected `X` columns → MinMax scaling to [-1,1] → PCA → Ward agglomerative or k-means. Resolved-level descriptors remain interpretation-only.

## New Step 7 · Compare salts

After clustering each recording, Step 6 can export a compact **salt-comparison package** containing:

- event-to-cluster assignments,
- centered real-sample median profiles for every cluster,
- cluster counts/populations,
- median and IQR dwell times,
- median and IQR mean blockade,
- clustering/profile metadata.

Upload the LiCl, NaCl, KCl, RbCl and CsCl packages in Step 7. Cluster IDs are recording-specific, so the interface explicitly asks you to map each Cluster ID to a common Family A/B/C… label. The default suggestion follows the shallow-to-deep cluster ordering, but the user must confirm the physical correspondence.

Step 7 then creates:

1. matched-family median waveforms across salts,
2. 100% stacked family-population bars,
3. family-resolved median dwell time with event-level IQR,
4. family-resolved median blockade with event-level IQR,
5. optional blockade normalised to a user-selected reference family within each salt,
6. a publication export pack with PDF/SVG/600 dpi PNG plus CSV/JSON.

PCA is **not pooled across salts** in this workflow. Separately fitted PCs are treated as within-recording diagnostics only.

## Important statistical caution

The IQRs in Step 7 describe the spread of events within a recording; they are not replicate-level confidence intervals. If multiple independent pores/recordings are available per salt, inferential statistics should use those independent replicates rather than treating every translocation as an independent experimental replicate.

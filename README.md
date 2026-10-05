# DNA Event Lab v0.8.2

## Hart-style event-family x-axis update

Step 5 keeps the v0.8.1 dataset-feature clustering unchanged. This release changes only the event-family waveform display and figure exports.

The cluster-family panels and representative overlay now offer three horizontal coordinates:

1. **Data index (sample number)** — default. Original recorded samples are aligned at the detected event start. Individual events are not stretched, so dwell-time differences remain visible. The detected event start is marked by a dotted vertical line. The window includes a user-selected number of pre-event samples and extends through the longest detected event plus user-selected post-event samples.
2. **Time relative to detected event start (ms)** — the same aligned real-sample traces, but the sample index is converted to milliseconds using the median sampling interval of the recording. Event start is 0 ms.
3. **Normalized event position (0 = start, 1 = end)** — the previous duration-normalised view, retained only as an optional shape comparison. This wording replaces the less clear “fraction of event duration”. Absolute dwell-time differences are removed in this view.

For aligned data-index/time views, missing samples outside the saved event trace are stored as NaN and are never fabricated. Cluster representatives are pointwise medians, shown only where at least 50% of the cluster has recorded samples. No percentile band is drawn.

The shared/manual axis controls from v0.8.1 now also include the cluster-family horizontal axis, and both the representative-profile ZIP and the main + supplementary clustering figure pack inherit the selected horizontal coordinate and axis limits.

Clustering still uses the selected original `dataset.npz` feature columns with MinMax scaling to `[-1, 1]`, PCA, and Ward agglomerative or k-means. Changing the event-family horizontal coordinate does **not** change PCA coordinates or cluster assignments.

# DNA Event Lab v0.8.3

## Centred Hart-style event-family display

This release keeps the v0.8.2 dataset-feature clustering unchanged. The update is display/export only.

Step 5 now puts the cluster-family controls directly above the family plots so they are easy to find. The horizontal axis can be switched between:

1. **Data index (sample number)** — default. Each detected event midpoint is placed at the centre of a fixed sample window. The original samples are not stretched, so dwell-time differences remain visible while baseline can appear before and after the event.
2. **Time relative to event midpoint (ms)** — the same centred real-sample traces converted to milliseconds using the median sampling interval.
3. **Normalized event position (0 = start, 1 = end)** — optional shape-only view; absolute dwell time is intentionally removed.

The centred window length is user-adjustable. If an event is longer than the selected window, the app warns that it is visually clipped; cluster assignments are unaffected.

The grey dashed/vertical alignment marker has been removed from cluster-family panels, representative overlays, and exports.

The axis-control section is now expanded by default. Auto mode keeps a common blockade axis across every cluster-family panel; manual/shared mode lets the user set exact family x/y limits and matching axes for PCA and physical plots.

Cluster representatives remain pointwise medians and are drawn only where at least 50% of cluster members have recorded samples. No percentile band is drawn.

Clustering remains: selected original `dataset.npz` columns → MinMax scaling to `[-1,1]` → PCA → Ward agglomerative or k-means. None of the display controls change PCA or cluster membership.

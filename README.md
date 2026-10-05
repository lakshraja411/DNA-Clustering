# DNA Event Lab v0.8.0 — original dataset-feature clustering

This release deliberately returns Step 5 to the **original NanoSense `dataset.npz` feature matrix** rather than constructing the clustering space from newly engineered plateau descriptors.

The default feature selection for the NanoSense layout used in this project is:

- `X[:,0]` height
- `X[:,1]` FWHM
- `X[:,2]` height-at-FWHM
- `X[:,3]` area
- `X[:,4]` width
- `X[:,5]` skewness
- `X[:,6]` kurtosis

`X[:,7]` is treated as baseline/QC and is not selected by default. `X[:,8]` and, where present, `X[:,9]` are treated as event-time/bookkeeping columns and are excluded by default. The interface always shows the raw column index because NanoSense layouts can vary between versions. The user can change the feature selection before clustering.

## Step 5 pipeline

The primary clustering pipeline is:

`matched original dataset rows → selected X columns → MinMax scaling to [-1,1] → PCA → Ward agglomerative OR k-means`

This intentionally reproduces the earlier feature-space philosophy. No log-transformed dwell time, fold contrast, deep-state fraction, temporal centroid, ECD, level count, or other hand-engineered physical feature is inserted into the primary clustering matrix.

Constant selected columns are removed. Correlation pruning at `|r| >= 0.98` is available as an optional sensitivity check and is **off by default**. The interface shows the raw feature correlation matrix, min/max/range table, PCA scree plot, PCA loadings, elbow/silhouette scan, optional 3-PC view, and Ward dendrogram.

The default PCA dimension is 2 to make comparison with Hart-style DNA PCA clustering straightforward, but the number of PCs remains adjustable. Both Ward and k-means use the exact same scaled dataset features and PCA coordinates.

## Physical interpretation is deliberately separate

Resolved plateau descriptors are still calculated, but they do **not** decide cluster membership. They are merged onto the clustered event table afterwards for interpretation. This allows checks such as:

- measured blockade versus dwell time,
- resolved-level composition,
- deepest-state occupancy,
- fold-contrast/occupancy maps,
- ECD and measured blockade distributions,
- real member traces and median representative waveforms.

An event can therefore belong to a valid dataset-feature cluster even if its step fit is not suitable for resolved-level interpretation. This avoids allowing a PELT/step-fitting assumption to manufacture the clustering structure.

## Waveform panels

Each cluster panel shows a deterministic sample of real member profiles as faint curves and the pointwise median of **all** cluster members as the bold representative. No percentile envelope is used. The overlay figure also contains representative curves only.

## Files

Replace the repository files together and keep the deployment names exactly:

- `app.py`
- `analysis.py`
- `physical.py`
- `plots.py`
- `workflow.py`
- `requirements.txt`

Run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

`python check_app.py` runs a lightweight synthetic regression test for the dataset-feature clustering engine. `python check_physical.py` checks the resolved-level interpretation descriptors.

## Scientific caution

The resulting groups are unsupervised **signal families**, not automatically unfolded/folded DNA labels. Elbow, silhouette, PCA separation and clustering stability describe geometry, not biological truth. Use waveform families and independent physical descriptors to decide whether a cluster has a defensible conformation interpretation.

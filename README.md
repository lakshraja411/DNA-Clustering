# Nanopore Shape Lab 0.2

Streamlit research software for DNA event inspection, candidate fitting, unsupervised grouping and event-level exports. Inspired by the unsupervised nanorod and DNA concepts in Hart et al., **NanoBoost**, npj Biosensing 3, 38 (2026), DOI https://doi.org/10.1038/s44328-026-00105-x.

This is an adaptation, not an exact reproduction of NanoBoost, and has no demonstrated superiority. No power spectral density calculation or plot is included.

## Update the existing GitHub / Streamlit app

Replace `app.py`, `analysis.py`, `requirements.txt`, `check_app.py` and `README.md` in the repository root. **Add `plots.py`** beside app.py. Keep `.streamlit/config.toml` for styling. Upload the files inside this folder, not the ZIP itself. Commit the changes. Streamlit redeploys when dependencies change.

Use **Python 3.12** on Streamlit Community Cloud. Changing Python requires deleting and recreating the Streamlit deployment, not the GitHub repository. Entry point: `app.py`. No secrets required. Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

For a new repository, upload the same files and deploy at https://share.streamlit.io . Do not upload research NPZ data to GitHub; upload through the running app.

## Run locally

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`. The first DTW run can be slower while numerical routines compile.

## Files accepted

- **event_fitting.npz:** observed flat NanoSense layout: `EVENT_DATA_<id>_part_0` time, part 1 current, part 2 event bounds, part 3 saved fit, part 4 baseline. `SEGMENT_INFO_*` and `EVENT_ANALYSIS_*` are preserved if present.
- **event_data.npz:** verified using the supplied CsCl export. Contains `sampling_rate` and a NumPy object array `events`. Each dictionary has `event_id`, `start_time`, `end_time`, `event_data` (current minus baseline), `baseline_value`. The reader restores current by addition and constructs physical times with sampling_rate. These traces are cropped: no surrounding baseline noise or original fitted levels are present. The object-array reader accepts only NumPy array reconstruction primitives; arbitrary pickle globals are blocked. Inputs are limited to 512 MB expanded size.
- **Numeric waveform layout:** `time`, `current` (event × sample) and `bounds` (event × 2); optional baseline and fit. Shared 1D time and scalar/shared baseline supported. Without baseline, padding is required for an explicitly flagged median estimate.
- **dataset.npz:** numeric `X` table. Alone, it supports summary distribution plots. Alongside waveforms, rows are matched by unique start timestamps, never row position. Defaults for the supplied export: start column 8, duration column 4 (seconds), height column 0 (nA). Verify these configurable mappings. Unmatched and ambiguous matches remain visible. All matched source X rows are retained unchanged in exports.

Time units: seconds in source files; current units: nA. The source-specific event_data interpretation has been checked on the supplied sample; other export versions may require adjustment. The file-structure inspector shows unfamiliar layouts. File names and same start times do not guarantee identical acquisition sources.

## Interface

1. **Inspect:** original waveform, baseline, original fit and optional candidate overlay; event index, recording time and segment table.
2. **Waveform disagreement:** raw and relative RMSE, signed peak-height error and area error. Calculated inside the detected event only. A rectangular peak representation can disagree strongly with a rounded pulse without indicating a bad event. Original-fit disagreement filtering is **off by default**.
3. **Refine fits:** selected event or whole file. Candidate fitting holds baseline and event boundaries fixed. Choose segment means at existing boundaries; PELT change-point detection with minimum duration and penalty; or one Gaussian rounded pulse. Originals are never overwritten. A smaller error is not proof of a better physical model. PELT uses L2 cost, jump=1, minimum samples from requested microseconds; its penalty is multiplier × log(n) in signal units normalised by padding std, or first-difference std if padding is absent. The noise scale has a small floor. Correlated noise can cause false steps. Gaussian is descriptive, not a deconvolution. Refinement is limited to 6,000 samples per event.
4. **Cluster:** choose one of four approaches below. View profile means/barycentres, 10–90 percentile bands of unwarped profiles, PCA projection, event-profile heatmap, population table and example traces. Compare to previous run by ARI/crosstab on shared event IDs. Feature and original waveform methods offer k=2..8 exploration; no automatic topology count is chosen.
5. **Distributions:** actual duration, mean/peak blockade, ECD, disagreement and segment count; scatter plots, 2D count heatmaps and histograms. Select full, clustered, excluded or individual-group populations. Log axes use geometric bins. Counts are not probability density; nonfinite/nonpositive points on log axes are reported. Export plotted tables and histogram counts/edges.
6. **Download:** one group, excluded events, all events or all groups in separate ZIPs. Includes original waveforms and fits, source IDs, result CSV, metadata and matched original dataset rows. Candidate fits are separate `REFINED_*` arrays with method/parameters. Reloading events.npz analyses the original trace/fit; candidates remain available as separate arrays for offline use. NanoSense re-import compatibility is not established.

## Clustering methods and relation to NanoBoost

| Option | Implementation here | Relation to paper |
|---|---|---|
| PCA + agglomerative (default) | Min–max features to [-1,1], PCA, Ward-linkage hierarchy | DNA feature-clustering concept; Ward is our explicit choice |
| PCA + k-means | Same feature scaling/PCA, k-means with 20 starts | Nanorod feature-clustering concept |
| Waveform k-means | Ordered blockade values at 16/32/64/128 relative-time positions, Euclidean k-means | Original simple reference method |
| Time-series k-means (DTW) | tslearn, Sakoe–Chiba alignment constraint, DTW barycentres, 1 start, 15 iterations | Paper's waveform-clustering concept, with explicit implementation limits |

**Differences from the paper:** The paper used a larger feature set, including trough features for biphasic nanopipette events and wavelet-derived features. It applied DWT processing and studied different analytes/pore conditions. This app does not reproduce the authors' full preprocessing, optimise a mother wavelet, use XGBoost, or reproduce their exact data preparation or supplementary hyperparameters. We do not claim identical clusters. Feature formulas and retained PCA components are explicit. Optional Haar coefficient features are exploratory and do not implement the paper's tuned DWT denoising. Default features are duration, mean/peak blockade, within-event std, relative peak position and first-half minus second-half mean blockade. FWHM and three Haar coefficient features can be selected. Duration here is detected end-start, while the selectable FWHM is separately defined.

Feature methods retain duration if selected; waveform methods rescale time to [0,1], removing absolute duration. Default waveform comparison preserves depth. Optional per-event amplitude normalisation removes depth information and can make different occupancy levels indistinguishable. Feature methods always use the measured trace for feature extraction and displayed profiles; fitting and waveform normalisation controls do not change those features.

PCA is a clustering input only in the feature methods; in waveform methods it is a 2D visual projection. Feature-cluster displayed profiles are unaligned averages of their members, not inverse-reconstructed feature centroids. DTW barycentres are displayed separately from unwarped percentile bands. Example selection uses pointwise distance to the displayed profile, plus three deterministic random members; it is distinct from model assignment distance.

DTW supports at most 2,000 eligible events and 64 positions to bound runtime. Its silhouette uses constrained DTW distances on a stratified sample of about 200 events. Other methods use Euclidean distances in their respective clustering spaces. Scores from different spaces do not demonstrate method superiority. Two-initialisation ARI is reported for k-means only; agglomerative is deterministic and DTW repeat stability has not been evaluated. Repeated runs are not independent physical validation. Cluster IDs have no meaning across runs.

## Scientific interpretation

Neither good silhouette nor low waveform error establishes a DNA topology. Step detection may fit rounded edges; amplitude normalisation can remove occupancy evidence; DTW may erase meaningful fold-duration differences. Baseline choice, finite bandwidth, event detection and pore differences affect results. DWT coefficient values depend on wavelet, boundary convention and sampling. Establish the single-section blockade reference separately per pore/condition, use simulations with realistic noise/filtering and independent pore recordings, inspect random cluster members, and test parameter sensitivity before biological interpretation. No automatic knot labels, occupancy-state inference or cross-salt causal comparison is included.

## Data and reproducibility

Session memory holds uploaded data; no global cache shares uploads. Hosted data are processed on the server. Scientific arrays are not written to the repository by the app. Exports contain input hashes, software version, settings, feature selection, refinement parameters, matching rules and exclusion information. Original acquisition file paths are omitted from exported settings. Results become stale when relevant controls change and must be rerun before cluster export. Keep provenance JSON with CSV/NPZ files.

## Checks

Run `python check_app.py` from this folder. Checks cover known-level recovery, fixed-boundary least-squares error improvement, timestamp joins with reordered and ambiguous rows, exact original waveform/fit and dataset subset export roundtrips, disallowed serialized globals, and Streamlit grouping/refinement/download interactions. The provided full fitting export (1,400), cropped event_data export (1,400), and fitting subset (419) all load without rejected events. Synthetic checks establish numerical operation, not scientific validity.

# DNA Event Lab · 0.3

A guided Streamlit workflow for inspecting DNA nanopore events, refining fits, plotting physical distributions, clustering and exporting event subsets. No synthetic demonstration or power spectral density is included in the interface.

## Update your existing app

Unzip this package. Upload the files inside the folder to the root of your existing GitHub repository, replacing the corresponding files. **Both `workflow.py` and `plots.py` must sit beside `app.py`.** Replace requirements.txt as well. Keep `.streamlit/config.toml`. Do not upload experimental data to GitHub.

Entry point: `app.py`. Tested locally using Python 3.12. The package has not been deployed to your live app.

Local launch:

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## The six steps

1. **Load files:** three separate upload controls for dataset, eventdata and eventfitting NPZ files. Both underscore and non-underscore filenames work. Click Load and check files, review the matches and confirm that the files are from the same recording. No row-order matching: unique start timestamps link events. Defaults for the supplied NanoSense X table: start column 8, tolerance 0.1 µs. Missing/ambiguous matches are shown; unmatched data are never invented. Reloading clears refinements and clustering results.
2. **Inspect events:** measured trace, uploaded fit and any refined fit. Switch between actual current I and blockade ΔI. The independently saved eventdata trace is available in an expandable comparison. The fitting file's measured trace is the source for refinement and measured-trace clustering, preserving the original fitter's signal context.
3. **Refine and save fits:** try one event before processing the full file. Segment means keeps existing boundaries; PELT finds step boundaries; Gaussian fits a single rounded pulse. Original baseline and event boundaries remain fixed. Failed refinements keep previous fits. Successful refinements immediately become the selected fits for subsequent analysis. Prepare and download `refined.eventfitting.npz`; its standard fit arrays contain the selected fits, so re-upload uses those fits. INPUT_FIT/LEVELS/WIDTHS arrays retain the fits uploaded in this session. Measured current is never overwritten. If only some events are refined, the rest keep their uploaded fits, clearly counted in the interface.
4. **Current–duration plots:** Δt (ms) versus mean or peak ΔI (nA), from measured traces or selected fits. Scatter plots and event-count heatmaps; optional logarithmic duration. Download plotted measurements and publication figure ZIPs (vector PDF/SVG and 600 dpi PNG). Points incompatible with log axes are counted and reported. Heatmap colours represent events per bin, not probability or spectral density.
5. **Cluster events:** defaults to selected fits and PCA + agglomerative clustering. For this setting, feature extraction really uses the selected fitted waveform, not the measured waveform. Measured traces remain an explicit alternative. Other options are PCA + k-means, waveform k-means and DTW time-series k-means. Select cluster count; technical controls are under Advanced. Inspect physical scatter plots, profile bands, PCA coordinates and every event in a selected cluster. Physical scatter uses measured mean blockade and duration even when fits drive clustering. Refinement changes invalidate assignments; clustering control changes invalidate results when the clustering page runs. Export uses the last explicitly computed configuration, shown in step 6.
6. **Save clusters:** download one or all clusters. Each cluster ZIP contains selected.eventfitting.npz, selected.eventdata.npz, selected.dataset.npz, event_results.csv and provenance.json. Unmatched counterparts are absent, with mapping columns in CSV. The dataset's X rows are preserved unchanged, **not recomputed after fitting**; current clustering features are stored under clustering_* columns in CSV. Raw-event IDs may differ from fitting-event IDs: mapping is explicit.

## Scientific interpretation

These are unsupervised signal groups, not automatic topology classifications. The approaches are inspired by NanoBoost (Hart et al., npj Biosensing, 2026, DOI 10.1038/s44328-026-00105-x). Feature scaling + PCA + k-means follows the nanorod concept; PCA + agglomerative follows the DNA concept; DTW follows the waveform-grouping concept. This is not an exact reproduction of the paper's preprocessing, complete feature set or tuned wavelet denoising, and superiority is not demonstrated.

PCA feature methods use min–max scaling to [-1,1]. Default features: duration, mean blockade, peak blockade, within-event standard deviation, relative position of the peak and first-half minus second-half mean blockade. Optional FWHM and Haar coefficient summaries are available. Ward linkage is used for agglomerative clustering. PCA compresses the features before clustering; chosen feature count and retained components affect results. Piecewise-constant fits can have flat maxima: peak position is the first maximum sample, not a uniquely identified physical transition.

Waveform methods compare duration-normalised profiles while retaining amplitude. DTW permits limited local time alignment; it can conceal meaningful differences in fold duration. Runtime limits: 2,000 events and 64 positions for DTW; up to 6,000 event samples for refinement. DTW uses one initialisation, 15 iterations and constrained alignment. K-means uses 20 starts. Feature and waveform spaces differ; their silhouette scores are not directly comparable physical validation. Cluster IDs are arbitrary and can change across runs. Displayed bands are member 10th–90th percentiles, not confidence intervals.

A lower RMSE can reflect extra parameters or a different descriptive fit, not better physical interpretation. Correlated noise and bandwidth-rounded edges can produce false steps. Gaussian fitting is descriptive, not instrument deconvolution. Refined fits should be reviewed before interpreting groups.

## Formats, provenance and limitations

The supplied NanoSense layouts are supported: numeric EVENT_DATA_<id>_part_* fitting arrays; object-array event_data records with baseline_value, event_data (baseline-relative current), start/end times and sampling_rate; numeric dataset X. The object reader restricts reconstruction globals to required NumPy primitives. Expanded NPZ size is limited to 512 MB. Units are assumed to be seconds and nA.

Exports use the numeric fitting/event layout supported by this app. **NanoSense re-import compatibility has not been established.** Keeping the `.eventfitting.npz` suffix does not guarantee another program can import it. Re-upload in this app is tested. Refined-event summary arrays with obsolete fitting metrics are omitted; original summary dataset rows are retained separately. Each download preserves IDs, selected fits, original measured data, input hashes, processing settings and matching rules. Keep your original three files as the source record, particularly across successive refinement/export sessions.

Upload data and generated downloads are held in the Streamlit session on its server; a session reset loses unsaved refinements. Input files are not saved to the GitHub repository. Figure exports use Matplotlib with explicit units, readable type, colourblind-friendly categorical colours, vector formats and high-resolution raster output. All interactive figures also offer SVG download through the plot toolbar. This is styling/export support, not a claim that every generated figure is publication-ready without scientific review.

## Verification

The supplied three full exports were checked for 1,400 unique timestamp matches. Tests verify refined-fit reload, unchanged measured traces/uploaded fits, fitted-feature extraction, dataset subset correspondence, matched raw subset exports, scientific figure generation, and the six-step UI including stale-result invalidation. These establish software behaviour, not topology accuracy.

`check_app.py` expects the three supplied CsCl sample files in an `upload` folder beside this project folder; private experimental files are not bundled. Run it from the project folder when those fixtures are available.

### Session persistence fix
The three file upload controls remain mounted in the sidebar on every step. Returning to Load does not discard the files or the current analysis. Checking the same files and matching settings preserves refinements and assignments. Clear files and start over explicitly clears this session. Browser/session resets still require re-uploading.

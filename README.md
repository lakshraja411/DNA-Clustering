# Nanopore Shape Lab

A Streamlit app for inspecting NanoSense event-fitting exports and exploring recurring DNA translocation signal shapes. Version 0.1.0. Research prototype: no automatic molecular topology assignment.

## Run locally (Python 3.11 or 3.12 recommended)

Open a terminal inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

## GitHub and Streamlit Community Cloud

1. Create a GitHub repository called `nanopore-shape-lab` (choose visibility appropriate to your work).
2. Upload `app.py`, `analysis.py`, `requirements.txt`, this README and the `.streamlit` folder. The project files belong in the repository root, not inside an extra enclosing folder. Include `.gitignore` if using git locally.
3. At https://share.streamlit.io select Create app, connect that repository and branch, and choose `app.py` as the entry point. Select Python 3.11 or 3.12 in advanced settings if offered.
4. Deploy. Upload the experimental NPZ using the running app, not to GitHub.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Workflow

Upload the full `.event_fitting.npz` for one recording condition. The `.dataset.npz` file has only summary values and is intentionally rejected. A SHORT export works but describes a selected population; event indices can be renumbered by the reduction software.

1. Inspect saved traces, baseline and piecewise fits.
2. Audit RMSE, bias, padding variability, segment timing and sample counts. No silent fit correction.
3. Cluster measured, ordered profiles resampled at 16, 32 or 64 relative-time positions. K-means uses a shared amplitude scale, retaining cross-event depth differences. It ignores absolute dwell time. Group count and review gates are adjustable; groups are not forced into named topologies.
4. Review centroid-nearest and deterministic random members. Silhouette and two-seed ARI measure geometric separation and limited numerical stability, not physical truth.
5. Export CSV and JSON together. Group -1 means excluded by the selected review gates, not anomalous DNA.

The synthetic demo has idealised levels and noise; it tests the interface and is not experimental validation. No real research data are included in the software bundle.

## Scientific limits and next development

Saved currents are assumed nA and times seconds. File layout is inferred from the supplied NanoSense exports. Some exports contain visible fit/trace disagreements; investigate reduction code before interpreting saved plateaus. Original raw acquisition data cannot be recovered from a filtered export. Time bins are shape descriptors, not independent samples or occupancy states. Interpolation, filtering, amplitude variation and pore interactions may drive groups. Padding standard deviation is not necessarily acquisition noise. Detection bias remains.

Before topology claims: verify export definitions; validate bandwidth and baseline treatment; infer reference blockade levels per pore/condition; assess sustained transitions with uncertainty; evaluate parameter changes, simulations passed through the acquisition filter, and held-out pores. Salt effects remain confounded with pore differences without appropriate replication. Fold and knot labels require evidence beyond cluster membership.

This app has no built-in cross-recording comparison, occupancy-state inference or validated topology classifier yet.

## Data handling

Numeric arrays load with `allow_pickle=False`. Expanded archives over 512 MB are rejected. Invalid events are listed. Uploads remain in the app server's session memory and are not explicitly written to disk by the app. Hosted uploads are processed on the hosting server; select app access settings accordingly. Export JSON omits the original acquisition filesystem path. Do not put sensitive data or credentials in GitHub.

## Checks

Run `python check_app.py` from this directory. Dependencies are pinned to the versions used for the bundled checks. The supplied full experimental export (1,400 events) and selected export (419 events) both loaded without rejected events during development. Summary-only input was correctly rejected. Four-group exploratory results overlapped (full-export silhouette approximately 0.224); this is not validation of four physical classes.

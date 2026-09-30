# DNA Event Lab 0.6.0 — update notes

This build adds a validation layer between step fitting and PCA/clustering.

## New
- Resolved-level overlay: measured blockade + selected step fit + final merged plateaus actually used for the physical descriptors.
- Transition audit: signed adjacent-level change and transition SNR (`|ΔI| / robust noise scale`).
- Recording-level robustness scan across minimum plateau duration and noise-multiplier settings.
- Event-level PELT sensitivity table across modest penalty/minimum-duration changes.
- Cluster event browser now shows the resolved-level overlay for physical-feature clustering.
- Resolved-level stability summary is written into clustering provenance when the robustness check was run with the current settings.

## Changed
- PELT noise scaling now uses robust MAD estimation, matching the physical-level QC logic.
- Resolved-level CSVs include start/end time, duration, blockade, transition ΔI, transition SNR, noise scale, merge threshold, and effective minimum duration.
- Version metadata updated to 0.6.0.

## Important interpretation
- The robustness scan tests the post-fit plateau-resolution rules. It does not independently re-find change-point boundaries.
- The PELT sensitivity table is a diagnostic, not an automatic parameter optimiser.
- Signal groups and resolved transitions are not DNA topology or strand-count labels.

## Verification performed here
- `python -m py_compile analysis.py app.py physical.py plots.py workflow.py check_physical.py check_app.py` passed.
- `python check_physical.py` passed, including the new transition-SNR and stability-scan checks.
- A full Streamlit UI integration run was not executed in this environment because Streamlit is not installed here.

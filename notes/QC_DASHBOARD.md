# Data quality dashboard (fMRIPrep -> XCP-D -> data.pt)

Interactive Streamlit app in `analysis/qc_dashboard/`. Give it three folders; it shows per-run /
per-subject QC at each stage, group summaries, cross-stage comparisons, and a slice viewer for `data.pt`.

## Hosting model (data never leaves the cluster)

The app runs **on a Great Lakes compute node** next to the data on turbo and listens on `127.0.0.1` only. You see it in
your laptop browser through an SSH tunnel. Nothing is exposed on the network; only rendered plots cross the tunnel.

- Streamlit telemetry is off (`--browser.gatherUsageStats false`).
- Compute nodes are shared, so another user on the same node could reach `127.0.0.1:<port>`. The launcher therefore
  generates a random token (`QC_DASH_TOKEN`); the app shows nothing without it. The token and the log are `umask 077`.
- Don't run it on the login node (shared, no heavy compute) and don't mount turbo on the laptop (slow, copies data over the wire).
- Fallback if SSH to compute nodes is blocked: start the same `streamlit run ...` from Great Lakes Open OnDemand
  (Desktop / Jupyter session) and use its browser.

## One-time setup (login node)

```bash
module load mamba && mamba env create -f analysis/qc_dashboard/environment.yml   # env "qc_dashboard", Python 3.11, CPU torch
```

The launcher activates `qc_dashboard`; set `QC_ENV=<name>` to use a different env.

## Run on a dataset

```bash
NAME=michigan_d1 \
FMRIPREP=/nfs/turbo/.../fmriprep \
XCPD=/nfs/turbo/umms-huanglab/pipeline/postproc/fmriprep-25.2.5_xcp_d-26.1.1/Michigan_Dataset_1/noGSR_highpass_fd-0.8_nosmooth \
DATAPT=/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1 \
sbatch analysis/qc_dashboard/sbatch_qc_dashboard.sh
```

Then read `logs/qc_dashboard_<jobid>.log` in your scratch dir: it prints the `ssh -N -J ... -L ...` command to run on the laptop
and the `http://localhost:<port>/?token=...` URL. `scancel <jobid>` stops it. Another dataset = same command with different
`NAME` and paths; the dataset picker in the sidebar switches between scanned datasets.

Options (env vars): `DEEP=1` also loads the BOLD NIfTIs (tSNR, DVARS, global-signal CV; ~2-4 GB RAM per worker),
`FD_THR=0.5`, `SKIP_SCAN=1`, `FORCE=1`, `DATAPT_SUBJECT_REGEX` (one group, to turn a data.pt folder name into the subject id).
Any of the three paths may be omitted.

## What is scanned

| Stage | Files found (BIDS, `sub-*/[ses-*/]func/`) | Metrics |
|---|---|---|
| fMRIPrep | `*_desc-confounds_timeseries.tsv` (+ `*_desc-preproc_bold.nii.gz` with `DEEP`) | n_vol, mean/median/max FD, % FD > thr, DVARS, RMSD, max translation/rotation, GS/CSF/WM std, non-steady-state and aCompCor counts |
| XCP-D | `*_motion.tsv`, `*_outliers.tsv`, `*qc*.tsv/csv` (+ `*_desc-denoised*_bold.nii.gz` with `DEEP`) | FD, % censored, n retained, longest censored run, every numeric column of the QC file as `qc_*` |
| data.pt | `<root>/img/<subject>/data.pt` | n_frames, shape/scale, foreground fraction, FOV-edge fraction, background constancy, tSNR, mean `|voxel temporal mean|` ("fingerprint"), temporal std, global signal, DVARS-like, % voxels saturated at the int8 limit |

Subjects are joined across stages by id with any `sub-` prefix removed. Cache: `output/qc_dashboard/<NAME>/` (CSVs +
per-subject series; rescans only recompute files whose mtime changed).

## Using it

- Sidebar: task/run filter, subject multiselect, "Group by" (custom group A vs rest, task, run, or any column of a
  `participants.tsv` it finds next to the derivatives), flag thresholds, focus subject.
- Overview: coverage (runs per stage per subject) and flagged subjects. Stage tabs: per-metric box/histogram/bars by group,
  summary table, Mann-Whitney for two groups, CSV download. Cross-stage: scatter + Spearman, correlation matrix.
- Subject detail: FD/DVARS/GS/motion traces (fMRIPrep), FD with censored volumes shaded (XCP-D), data.pt series, and a
  volume viewer (frame / temporal mean / std / tSNR, 3 orthogonal slices, voxel time series, histogram).

## Reports tab (fMRIPrep / XCP-D images)

"Reports (images)" shows, for the focus subject, either the **figures** (`sub-XX/figures/*.svg|png`, filterable by type and by file
name such as `run-1`) or the **full HTML report** (`sub-XX.html` at the derivatives root, with its relative images inlined). Files are
read live from turbo; HTML runs in an iframe with JavaScript enabled, so only point it at your own derivatives. If nothing is
found it lists the exact paths it looked in.

## Caveats

- Built and tested on synthetic trees only (the laptop has no turbo access): file-name patterns for XCP-D 26.1.1's QC file and
  other site-specific layouts may need a small tweak in `qc_lib.py` (`XCPD_PATTERNS`). If a stage shows no runs, `ls` one `func/` folder.
- data.pt metrics use "foreground = voxels with non-zero temporal std", matching how preprocessing fills the background.
- Mann-Whitney p-values are uncorrected and treat runs as independent; use them for screening, not inference.

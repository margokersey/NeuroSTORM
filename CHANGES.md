# Changes

## Add `--skip_existing` to volume preprocessing

- `datasets/preprocessing_volume.py`: new `--skip_existing` flag skips subjects whose `img/<subj>/data.pt` already exists (default off; behaviour otherwise unchanged). `scripts/preprocess_michigan_xcpd_runs.sh` now passes it, so reruns resume instead of overwriting. Delete a subject's `data.pt` to force a rebuild.

## Add Stage A (Task vs. No Task) for two-stage decoding

- `datasets/fmri_datasets.py`: new `MichiganTaskVsRest` (`MICHIGAN_TASK_VS_REST`): 30 imagery-run-1 blocks (Task) vs. 15 + 15 onset-matched mock blocks from rest-1/rest-2 (No Task). Missing rest runs are skipped per subject (sub-29); rest reads are capped at 770 TRs (sub-22 rest-1). Within-subject `--permute_labels` supported. Registered in `datasets/data_module.py` (shares `nas/splits/MICHIGAN/`).
- `models/lightning_model.py`, `callbacks/imagery_predictions_callback.py`, `main.py`: new binary task `taskvsrest_classification` (block-level accuracy / balanced accuracy / recall / AUROC; per-sample prediction JSON with `notask`/`task` classes). The 3-way path is unchanged.
- `run_finetune_scripts/greatlakes/loocv/`: `run_loocv_taskvsrest.sh` + `sbatch_loocv_taskvsrest*.sh`; `run_loocv_xcpd.sh` now accepts `LOOCV_DATASET/TASK/PROJECT_PREFIX/EXTRA_DIRS` env overrides (defaults unchanged).
- `analysis/two_stage_decoding.py`: composes Stage A with the existing 3-way LOOCV predictions (Stage B accuracy given "Task", end-to-end 4-way confusion, Stage A permutation test). `tests/test_taskvsrest.py`, `notes/LOOCV_QUICKSTART.md` updated.

## Add squeeze-neighbour leakage check for left hand-motor parcels

- `analysis/plot_squeeze_hand_leakage.py`: averages Schaefer-400 parcels 50/51/53/57/59/76 (LH SomMot hand + DorsAttn_Post_8) from `data.pt`, and plots (blocks followed by squeeze) minus (blocks followed by other imagery) around block onset, shaded by block type. Prints a pre-squeeze dip statistic. Accepts several `--root` exports to compare high-pass vs. no-high-pass. Output: `output/squeeze_hand_leakage.png`. Also accepts a directory of raw XCP-D `desc-denoised_bold` NIfTIs plus `--timing` (atlas resampled onto each BOLD grid), so `data.pt` is not required. `analysis/sbatch_squeeze_hand_leakage.sh` runs it on Great Lakes (high-pass fd-0.8 vs no-high-pass fd-999 and fd-1, all `_nosmooth`; cache labels include the fd value so older caches are not reused). The figure now overlays all exports as colour-coded lines on one set of axes (3 panels) instead of one column per export; replotting from `output/cache/` needs no NIfTI reads. A second figure (`*_export_diff.png`) plots (export − `--ref` export) per subject with a paired t-test on the pre-B window, which isolates export (e.g. filter) effects because blocks and order are identical.

## LOOCV permutation analysis: full metric suite and publication figure

- `scripts/analyze_loocv_permutation_results.py`: in addition to accuracy, computes per-subject balanced accuracy (primary), macro-F1, Cohen's κ, macro OvR AUROC, log-loss, Brier and ECE. Each metric gets a subject-level bootstrap 95% CI, an aggregate permutation p-value (direction-aware), and a Wilcoxon signed-rank test across subjects. JSON adds `metrics` and `class_names`; a `_per_fold.csv` is written next to it.
- Replaces the bar + per-sample KDE plot with a figure (per-subject, KDE of per-subject accuracy real vs. permuted, null distribution, metric summary, confusion matrix, ROC, calibration), saved as PNG (600 dpi), PDF and SVG with editable text. Only panels with data are drawn; seaborn-"deep" colors without a seaborn dependency.
- Per-sample predictions are also read from `lightning_logs/version_N/test_predictions.pt` (probs/preds/targets), so ROC, calibration and AUROC work for runs without the JSON dumps. The script reports how many runs have labels / probabilities.
- Runs that only have `test_summary_fold_*.json` (no per-sample predictions) are analysed from the saved confusion matrix (accuracy, balanced accuracy, F1, κ, confusion matrix); ROC/calibration/AUROC/log-loss/Brier/ECE need per-sample softmax outputs.
- Default outputs moved to `output/loocv_permutation/` (were written to the repo root).
- Predictions are read from the latest `lightning_logs/version_N` (was hard-coded to `version_0`, which ignored best-checkpoint backfill re-tests).
- `notes/LABEL_PERMUTATION_TESTING.md` documents the new outputs; `notes/LOOCV_KDE_VISUALIZATION_GUIDE.md` is marked superseded.

## Fix LOOCV permutation testing: within-subject shuffle and valid p-values

- `datasets/fmri_datasets.py` (`Michigan._set_data`): `--permute_labels` now shuffles labels *within each subject* (preserving the 10/10/10 per-subject balance) instead of pooling all training blocks. **Existing `_perm*` runs used the pooled shuffle and should be re-run.**
- `scripts/analyze_loocv_permutation_results.py`: p-values use `(1 + count) / (1 + n)` (never 0). New aggregate across-fold test (mean accuracy over folds vs. per-permutation fold-mean null) is the primary result; per-fold p-values are descriptive. JSON output is now `{aggregate, per_fold}`. Warns when `n_permutations < 19` (p < 0.05 unattainable).
- `notes/LABEL_PERMUTATION_TESTING.md` updated accordingly.

## Final test uses the best-validation checkpoint

- `main.py`: the automatic post-fit `trainer.test()` now loads `checkpoint_callback.best_model_path` (best `valid_acc` / `valid_mse` / `valid_reid_top1`) instead of the last-epoch weights. New flag `--final_test_ckpt {best,last}` (default `best`); falls back to last-epoch weights with a warning if no best checkpoint exists. Pretraining is unchanged.
- The `analysis/backfill_*best_ckpt_test.sh` scripts are only needed for runs finished before this change.

## Add local no-GPU testing sandbox

- New `scripts/sandbox/`: cross-platform conda env (`environment-local.yml`), a pure-PyTorch `mamba_ssm` shim, a synthetic Michigan dataset generator, and `run_smoke.sh` which runs the real `main.py` train/val/test on CPU in ~10 s. `make sandbox-env | sandbox | sandbox-test`. Docs: `notes/local_sandbox.md`. Added `tests/test_sandbox_shim.py`.
- `make sandbox-gl` (`scripts/sandbox/run_greatlakes_checks.sh`, `verify_run.py`, `check_real_ckpt_compat.py`): runs the Great Lakes 5-fold/LOOCV/permutation launchers unmodified on synthetic data and verifies fine-tuning, checkpoints and predictions per fold.
- Local vs. cluster is chosen by the `NEUROSTORM_ENV` env var (`utils/runtime_env.py`): `set_env.sh` (sourced by every `sbatch_*`) exports `greatlakes`; the sandbox scripts export `local`. Only `local` skips the CUDA-only NVTX markers (`models/patchembedding.py`) and CUDA event timers (`LightningModel.on_train_epoch_start`); unset/`greatlakes` runs the original unguarded code.
- Michigan datasets never fall back to a random split: `MICHIGAN_FULL_BLOCK` now reads `nas/splits/MICHIGAN/` (`SPLIT_DIR_ALIASES`), a missing split file raises, and split files with empty sets or train/val/test overlap (incl. val == test) are rejected (`datasets/data_module.py`, `tests/test_split_guard.py`).

## Reorganize and clean up `run_finetune_scripts/`

- Grouped the workstation launchers by experiment: `local/5fold/`, `local/loocv/`, `local/reid/`, `local/within/` (moved with `git mv`; run them from the repo root, e.g. `bash run_finetune_scripts/local/5fold/run_finetune_xcpd_cv.sh`). `greatlakes/` is unchanged in layout.
- Added `lib/common.sh` with the preflight helpers that were copy-pasted in every script; the launchers now source it. Launched commands (args, project names, folds) are unchanged, verified by diffing old vs. new calls against a stub `run_experiment.sh`.
- Replaced the duplicated 25-line headers and large commented-out loops with short per-script headers (old variants are in git history).
- Fixes: `sbatch_finetune_xcpd_cv.sh` pointed at a non-existent runner path; `greatlakes/loocv/run_loocv_xcpd.sh --dry_run` crashed on an unset `$GPUS`; the permutation sbatch had a dead `$?` check.
- Untracked `greatlakes/.ipynb_checkpoints/` and added it to `.gitignore`.
- Preflight "MISSING" log lines now share one format.

---

# Changes Made to LOOCV Script for Great Lakes

## Summary
Updated the LOOCV (Leave-One-Subject-Out) fine-tuning scripts to work on Great Lakes HPC cluster, following the same pattern as the working 5-fold CV implementation.

## Files Modified

### 1. `run_finetune_scripts/greatlakes/run_loocv_xcpd.sh`

#### Before (hardcoded paths, local-only):
```bash
GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"
DATASET="michigan_xcpd_nosmooth_fullblock"
TASK="task6"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"
...
GPUS=0
...
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
```

#### After (dynamic paths, Great Lakes compatible):
```bash
GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Where the preprocessed data lives. Overridable from the job file.
DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"

# dataset for the original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/img"

# Number of GPUs this job can see (set by the job file; 1 if run by hand).
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))   # "0" for 1 GPU, "0,1" for 2, ...

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_folds) SKIP_FOLDS="$2"; shift 2 ;;
```

#### Preflight Check Improvements:

Before:
```bash
echo "Preflight: checking LOOCV split files $SPLIT_DIR/split_fixed_${START_INDEX}.txt .. split_fixed_${END_INDEX}.txt ..."
if [[ ! -d "$IMG_ROOT" ]]; then
    echo "Preflight FAILED: $IMG_ROOT does not exist."
    exit 1
fi

missing=0
for i in $(seq "$START_INDEX" "$END_INDEX"); do
```

After:
```bash
echo "Preflight: checking LOOCV split files $SPLIT_DIR/split_fixed_${START_INDEX}.txt .. split_fixed_${END_INDEX}.txt ..."
if [[ ! -d "$IMG_ROOT" ]]; then
    echo "Preflight FAILED: $IMG_ROOT does not exist -- has the data been copied/preprocessed?"
    exit 1
fi
if [[ ! -f "$GENERAL_CKPT" ]]; then
    echo "Preflight FAILED: checkpoint $GENERAL_CKPT not found (run from the repo root?)"
    exit 1
fi
if ! compgen -G "$SPLIT_DIR"/split_fixed_*.txt > /dev/null; then
    echo "Preflight FAILED: no split files found under $SPLIT_DIR"
    exit 1
fi

missing=0
for i in $(seq "$START_INDEX" "$END_INDEX"); do
```

#### Training Loop Update:

Before:
```bash
bash scripts/run_experiment.sh \
    --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
    --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPUS" \
    --load_model_path "$GENERAL_CKPT" \
    -- --dataset_split_num "$i" --tpt_strategy none \
         --project_name "${PROJECT_PREFIX}_fold-${i}"
```

After:
```bash
bash scripts/run_experiment.sh \
    --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
    --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPU_IDS" \
    --load_model_path "$GENERAL_CKPT" \
    -- --dataset_split_num "$i" --tpt_strategy none \
         --project_name "${PROJECT_PREFIX}_fold-${i}"
```

### 2. `run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh` (New File)

This is a new sbatch job wrapper that:
- Defines array job specification: `#SBATCH --array=101-126`
- Loads modules and activates conda environment
- Sets environment variables from SLURM
- Calls `run_loocv_xcpd.sh` with one fold per task

Key features:
```bash
#SBATCH --array=101-126                # one task per fold
#SBATCH --gpus-per-node=1              # 1 GPU per fold
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                # per fold wall time

# Each task runs exactly one fold:
srun bash "$FINETUNE_SCRIPT" --start_index "$SLURM_ARRAY_TASK_ID" --n_folds 1
```

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Data paths | Hardcoded local path | `$DATA_ROOT` environment variable |
| GPU spec | Fixed `--gpus 0` | Dynamic `$GPU_IDS` from SLURM |
| Execution | Sequential (local bash) | Parallel (sbatch array) |
| Preflight | Basic checks | Enhanced (verify checkpoint, splits) |
| Wall time | ~150-200 hours | ~7-10 hours (26 parallel) |
| Job system | None | SLURM sbatch array |
| Flexibility | Limited | Full array control (selective reruns) |

## Testing Checklist

- ✅ Bash syntax validation: `bash -n run_loocv_xcpd.sh`
- ✅ Matches pattern of `sbatch_finetune_xcpd_cv.sh`
- ✅ Data paths use `$DATA_ROOT` variable
- ✅ GPU IDs computed dynamically from `NUM_GPUS`
- ✅ Preflight checks for checkpoint and split files
- ✅ SKIP_FOLDS handles empty strings safely
- ✅ SLURM_ARRAY_TASK_ID used for fold assignment
- ✅ Log files generated per fold
- ✅ Module loading and conda activation included
- ✅ Environment variables properly exported

## Backward Compatibility

- Local version (`run_finetune_scripts/run_loocv_xcpd.sh`) remains unchanged
- Great Lakes version is parallel, not sequential
- Can still run single folds locally with: `bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --start_index 101 --n_folds 1`

## Documentation

- (removed; see git history) - Comprehensive technical guide
- `notes/LOOCV_QUICKSTART.md` - Quick reference for common commands
- `CHANGES.md` - This file, showing exact changes made

---

## Repo cleanup
- Untracked pretrained `.ckpt` files (kept locally); added `*.ckpt`, `*.pt`, `*.pth` to `.gitignore`.
- Removed root-level result PNGs; root `loocv_permutation_*.png` now ignored.
- Moved custom scripts (`compare_conditions*.py`, `backfill_*best_ckpt_test.sh`) to `analysis/`.
- Moved kept workflow docs to `notes/`; removed empty/duplicate/superseded guides (recoverable from git history).

## Michigan XCP-D preprocessing for other runs
- `datasets/preprocessing_volume.py`: `michigan_xcpd` now takes `--task` / `--run` (default imagery run-1, unchanged), matches files by basename so `sub-XX/func/` layouts work.
- New `scripts/preprocess_michigan_xcpd_runs.sh`: Slurm array for imagery 2-4 and rest 1-2 -> `FIXED_xcp_nosmooth_task-<task>_run-<N>/img`.

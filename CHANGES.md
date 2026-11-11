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

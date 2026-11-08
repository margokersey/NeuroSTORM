# NeuroSTORM LOOCV on Great Lakes - Updates

## Summary
Updated the LOOCV (Leave-One-Subject-Out) fine-tuning scripts to work on University of Michigan's Great Lakes HPC cluster, matching the patterns and conventions already established by the working 5-fold CV scripts.

## Files Created/Updated

### 1. `run_finetune_scripts/greatlakes/run_loocv_xcpd.sh` (Updated)
Updated to use Great Lakes conventions:

**Key Changes:**
- **Data paths**: Changed from hardcoded local paths (`/data/margo_datasets/...`) to configurable `DATA_ROOT` environment variable (defaults to `/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM`)
- **GPU specification**: Changed from hardcoded `--gpus 0` to dynamic `GPU_IDS` (computed from `NUM_GPUS` environment variable set by sbatch)
- **Checkpoint detection**: Enhanced preflight checks to verify checkpoint exists before running folds
- **SKIP_FOLDS handling**: Fixed empty string handling with conditional check `[[ -n "$SKIP_FOLDS" && ... ]]`
- **Command-line options**: Replaced `--gpus` argument with `--skip_folds` (more useful for sbatch array jobs)

**Usage:**
```bash
# Local testing with 1 GPU (default)
bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh

# Override data location and skip specific folds
DATA_ROOT=/path/to/data bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --skip_folds 105,110
```

### 2. `run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh` (New)
Created new sbatch job wrapper for running LOOCV as a parallel array job on Great Lakes.

**Key Features:**
- **Array job**: Runs folds 101-126 as separate parallel tasks (each fold on one A40 GPU)
- **Automatic fold assignment**: Each task gets exactly one fold via `SLURM_ARRAY_TASK_ID`
- **Module/environment setup**: Loads modules (GCC 11.2.0, CUDA 12.8.1, Mamba) and activates conda environment
- **Resource allocation**: 1 GPU, 8 CPUs, 64GB RAM per fold, 12-hour wall time
- **Logging**: Separate log files per fold: `logs/xcpd_loocv_<jobid>_<fold>.log`

**Usage:**
```bash
# Submit all 26 folds (101-126) to run in parallel
sbatch run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Rerun just fold 101
sbatch --array=101 run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Rerun folds 101-105
sbatch --array=101-105 run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Run all 26 folds but limit to 4 concurrent tasks
sbatch --array=101-126%4 run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Watch progress
squeue -u mkersey
tail -f /scratch/huangzu_root/huangzu0/mkersey/logs/xcpd_loocv_<jobid>_101.log

# Cancel all folds for a job
scancel <jobid>

# Cancel just fold 101
scancel <jobid>_101
```

## Design Decisions

### Matching Existing Conventions
- Followed the same structure as `sbatch_finetune_xcpd_cv.sh` (5-fold CV)
- Used identical environment setup, module loading, and conda activation
- Maintained the same resource allocations and time limits
- Used `$DATA_ROOT` pattern for flexibility across different storage locations

### Array Job vs. Sequential Loop
- **sbatch_loocv_xcpd.sh**: Runs each fold on its own GPU in parallel (26 GPUs running simultaneously if available)
  - Much faster for the full experiment
  - Better resource utilization
  - Requires more GPU hours but completes quicker
- **run_loocv_xcpd.sh** (when run locally/manually): Runs folds sequentially
  - Useful for testing single folds or debugging
  - Can be called from sbatch with `--start_index` and `--n_folds 1` to process one fold per task

### Preflight Checks
Enhanced preflight validation to catch issues early:
- Verifies `IMG_ROOT` exists
- Verifies checkpoint file exists
- Verifies split directory has files before starting any folds
- Reports all missing subjects before failing, not just the first one

## Configuration Notes

### Great Lakes Setup
- **Account**: `huangzu99` (edit `#SBATCH --account=` if different)
- **Partition**: `spgpu` (A40 GPUs, optimized for sm_86 compute capability)
- **Data location**: `/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM` (fast, shared storage)
- **Working directory**: `/scratch/huangzu_root/huangzu0/mkersey` (fast, local to node)
- **Log directory**: Create with `mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs`

### Prerequisites
1. LOOCV split files must exist: `nas/splits/MICHIGAN/split_fixed_101.txt` through `split_fixed_126.txt`
   - Generate using: `python scripts/make_loocv_splits.py --dataset_name MICHIGAN --source_fold 1 --start_index 101`
2. Preprocessed data must be at `$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/img/`
3. Checkpoint file must exist: `pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt`

## Performance Expectations

Based on the 5-fold CV timing:
- **Per-fold runtime**: ~6-8 hours per fold on A40 (28 subjects training)
- **Sequential (run_loocv_xcpd.sh)**: ~150-200 hours total (26 folds × 6-8 hours)
- **Parallel (sbatch_loocv_xcpd.sh)**: ~6-8 hours total (all 26 folds running at once)
- **Wall-clock time**: If running all 26 in parallel, expect ~7-10 hours total
- **Cost**: Each fold uses 1 A40 GPU, 8 CPUs, 64GB RAM for the duration

## Troubleshooting

### Job submission issues
```bash
# Check if sbatch script has correct line endings (must be LF, not CRLF)
dos2unix run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Verify you're in the repo directory and can access the script
cd ~/code/NeuroSTORM
sbatch run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh
```

### Job hangs or fails
```bash
# Check job status and see the last N lines of output
squeue -u mkersey -j <jobid>
tail -100f logs/xcpd_loocv_<jobid>_101.log

# Test locally first with a single fold
bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --start_index 101 --n_folds 1 --dry_run

# Or with actual run (no GPU)
NUM_GPUS=0 bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --start_index 101 --n_folds 1
```

### Data path issues
```bash
# Verify data exists
ls -la /nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img/ | head

# Run preflight check manually
bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --dry_run
```

## Comparison: Local vs. Great Lakes

| Aspect | Local Machine | Great Lakes |
|--------|---------------|-------------|
| Script | `run_finetune_scripts/run_loocv_xcpd.sh` | `run_finetune_scripts/greatlakes/run_loocv_xcpd.sh` + `sbatch_loocv_xcpd.sh` |
| Data path | Hardcoded local path | Environment variable (`$DATA_ROOT`) |
| GPU specification | `--gpus 0` (single GPU ID) | `$GPU_IDS` (computed, supports multiple GPUs) |
| Execution | Sequential loop (one fold at a time) | Array job (all folds in parallel via sbatch) |
| Total wall time | ~150-200 hours | ~7-10 hours (with 26 concurrent tasks) |
| Submission | Direct bash call | `sbatch` command with array tasks |

## Next Steps

After the LOOCV run completes:

1. **Aggregate results**:
   ```bash
   python compare_conditions_v2.py \
     --condition "LOOCV (best model)|FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*"
   ```

2. **Check per-subject performance**:
   - The LOOCV setup gives you individual held-out subject scores (all 26 subjects)
   - Compare to see which subjects are easy/hard for the model

3. **Compare to 5-fold CV**:
   - Different fold structure, but useful for sanity checking
   - LOOCV trains on more data per fold (22 vs ~16 subjects)
   - Should expect slightly better generalization if the model scales well

## Files Modified

- ✅ `run_finetune_scripts/greatlakes/run_loocv_xcpd.sh` (updated with Great Lakes paths/patterns)
- ✅ `run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh` (new file, array job wrapper)
- ✅ `LOOCV_GREATLAKES_UPDATES.md` (this file, documentation)

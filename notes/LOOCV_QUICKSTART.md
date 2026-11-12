# LOOCV on Great Lakes - Quick Start

## One-Time Setup
```bash
# Create log directory
mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs

# Ensure split files exist (101-126)
# If not, run: python scripts/make_loocv_splits.py --dataset_name MICHIGAN --source_fold 1 --start_index 101
ls -la nas/splits/MICHIGAN/split_fixed_10*.txt | wc -l   # should be 26

# Verify data is copied to Great Lakes
ls -la /nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img/ | head -5
```

## Run All 26 LOOCV Folds in Parallel
```bash
# Submit array job (all 26 folds on 26 A40s)
sbatch run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh

# Monitor
squeue -u mkersey
tail -f /scratch/huangzu_root/huangzu0/mkersey/logs/xcpd_loocv_<jobid>_101.log
```

**Expected time**: ~7-10 hours for all 26 folds to complete (running in parallel)

## Re-run Specific Folds
```bash
# Just fold 101
sbatch --array=101 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh

# Folds 101-110
sbatch --array=101-110 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh

# All 26 but max 4 running at once (limit GPU usage)
sbatch --array=101-126%4 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh
```

## Run Locally (Single Fold for Testing)
```bash
# Test fold 101 with default settings (requires 1 GPU)
bash run_finetune_scripts/greatlakes/loocv/run_loocv_xcpd.sh --start_index 101 --n_folds 1

# Dry run (see what would run, no actual training)
bash run_finetune_scripts/greatlakes/loocv/run_loocv_xcpd.sh --dry_run

# With custom epochs
bash run_finetune_scripts/greatlakes/loocv/run_loocv_xcpd.sh --start_index 101 --n_folds 1 --max_epochs 20
```

## After Completion: Aggregate Results
```bash
# Summarize all 26 held-out subjects
python analysis/compare_conditions_v2.py \
  --condition "LOOCV (best model)|FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*"
```

## Job Management
```bash
# View all your running jobs
squeue -u mkersey

# Cancel all LOOCV folds
scancel <jobid>

# Cancel just fold 105
scancel <jobid>_105

# View job details
scontrol show job <jobid>

# Check completed job logs
tail /scratch/huangzu_root/huangzu0/mkersey/logs/xcpd_loocv_<jobid>_101.log
```

## Troubleshooting

**Q: Job fails with "MISSING split file"**
- Run: `python scripts/make_loocv_splits.py --dataset_name MICHIGAN --source_fold 1 --start_index 101`
- Then resubmit

**Q: Job fails with "No GPU visible"**
- The sbatch script may not have correct line endings
- Run: `dos2unix run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh`
- Then resubmit

**Q: Data path errors**
- Verify: `ls /nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img/ | head`
- If missing, copy data to Great Lakes first

**Q: Jobs running very slowly**
- Check if too many jobs are running: `squeue -u mkersey`
- Resubmit with `%4` suffix to limit concurrent jobs: `sbatch --array=101-126%4 ...`

## Two-stage decoding (Task vs. No Task, then tennis/navigation/squeeze)

Stage A (new, binary) + Stage B (the existing 3-way LOOCV above, **not retrained**), on the same folds 101-126.

- **Stage A data:** Task = the 30 imagery-run-1 blocks (label 1). No Task = 15 mock blocks from rest-1 + 15 from rest-2 (label 0), cut at the onsets of the first 15 imagery blocks (rest runs are only ~770 TRs, so onsets of later imagery blocks don't fit). `MichiganTaskVsRest` in `datasets/fmri_datasets.py`; config `scripts/configs/datasets/michigan_xcpd_nosmooth_taskvsrest.yaml` (task `task7`). Rest roots are derived from the imagery `image_path`.
- **Known data quirks (handled):** sub-29 has no rest-2 (contributes 15 mock blocks, and is excluded as a test subject in the evaluation); sub-22's rest-1 has 1145 TRs because the scan duration was mis-set, but only the first 770 are valid and only TRs < 770 are ever read.
- **Run Stage A** (needs `scripts/preprocess_michigan_xcpd_runs.sh` outputs for rest 1 and 2):
  ```bash
  bash run_finetune_scripts/greatlakes/loocv/run_loocv_taskvsrest.sh --dry_run     # preflight
  sbatch run_finetune_scripts/greatlakes/loocv/sbatch_loocv_taskvsrest.sh          # 26 folds (4 h each)
  sbatch run_finetune_scripts/greatlakes/loocv/sbatch_loocv_taskvsrest_permutation.sh   # label-permutation null
  ```
  Runs are named `FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-<N>[_perm<k>]`.
- **Compose the two stages:** `python analysis/two_stage_decoding.py --stage_a_glob ... --stage_b_glob ... --tsv .../timing_onehot_task1.tsv --exclude_subjects sub-29` (full example in the script's docstring). Reports Stage A accuracy/balanced accuracy/AUROC, Stage B accuracy on blocks Stage A called "Task", and an end-to-end 4-way confusion matrix. Alignment between stages is verified per fold against the TSV.

## Key Files
- **Submission script**: `run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh`
- **Training script**: `run_finetune_scripts/greatlakes/loocv/run_loocv_xcpd.sh`
- **Change history**: see `CHANGES.md`
- **Split files**: `nas/splits/MICHIGAN/split_fixed_101.txt` ... `split_fixed_126.txt`
- **Data**: `/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img/`

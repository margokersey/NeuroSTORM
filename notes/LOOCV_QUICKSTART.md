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
sbatch run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Monitor
squeue -u mkersey
tail -f /scratch/huangzu_root/huangzu0/mkersey/logs/xcpd_loocv_<jobid>_101.log
```

**Expected time**: ~7-10 hours for all 26 folds to complete (running in parallel)

## Re-run Specific Folds
```bash
# Just fold 101
sbatch --array=101 run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# Folds 101-110
sbatch --array=101-110 run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh

# All 26 but max 4 running at once (limit GPU usage)
sbatch --array=101-126%4 run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh
```

## Run Locally (Single Fold for Testing)
```bash
# Test fold 101 with default settings (requires 1 GPU)
bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --start_index 101 --n_folds 1

# Dry run (see what would run, no actual training)
bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --dry_run

# With custom epochs
bash run_finetune_scripts/greatlakes/run_loocv_xcpd.sh --start_index 101 --n_folds 1 --max_epochs 20
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
- Run: `dos2unix run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh`
- Then resubmit

**Q: Data path errors**
- Verify: `ls /nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img/ | head`
- If missing, copy data to Great Lakes first

**Q: Jobs running very slowly**
- Check if too many jobs are running: `squeue -u mkersey`
- Resubmit with `%4` suffix to limit concurrent jobs: `sbatch --array=101-126%4 ...`

## Key Files
- **Submission script**: `run_finetune_scripts/greatlakes/sbatch_loocv_xcpd.sh`
- **Training script**: `run_finetune_scripts/greatlakes/run_loocv_xcpd.sh`
- **Change history**: see `CHANGES.md`
- **Split files**: `nas/splits/MICHIGAN/split_fixed_101.txt` ... `split_fixed_126.txt`
- **Data**: `/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img/`

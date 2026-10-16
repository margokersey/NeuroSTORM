#!/bin/bash
# =============================================================================
# NeuroSTORM 5-fold CV fine-tune (XCP-D nosmooth) — Great Lakes job array
#
# Each array task runs ONE fold on its own A40, in parallel.
#
# One-time setup:   mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit (from anywhere):
#   sbatch finetune_xcpd_cv.sbatch                # all 5 folds
#   sbatch --array=3 finetune_xcpd_cv.sbatch      # rerun just fold 3
#   sbatch --array=1 --time=00:30:00 finetune_xcpd_cv.sbatch   # quick test
# Watch:   squeue -u mkersey      tail -f logs/xcpd_cv_<jobid>_<fold>.log
# Cancel:  scancel <jobid>   (all folds)   scancel <jobid>_3   (one fold)
# =============================================================================

#SBATCH --account=huangzu99
#SBATCH --job-name=xcpd_cv
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=spgpu              # A40s -- your Mamba build is sm_86 only
#SBATCH --array=1-5                    # one task per fold; %2 suffix (1-5%2) = max 2 at once
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # must equal --gpus-per-node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                # per fold; tune after the first one finishes
#SBATCH --output=logs/xcpd_cv_%A_%a.log   # %A = job id, %a = fold
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

# ── User settings (edit these) ───────────────────────────────────────────────
REPO_DIR="$HOME/code/NeuroSTORM_custom"      # your customized clone
FINETUNE_SCRIPT="/run_finetune_scripts/greatlakes/run_finetune_xcpd_cv.sh"    # relative to REPO_DIR
export DATA_ROOT="/scratch/huangzu_root/huangzu0/mkersey/data/Michigan_Dataset_1_for_NeuroSTORM"

# ── Modules ──────────────────────────────────────────────────────────────────
module load gcc/11.2.0 cuda/12.8.1
module load mamba

# ── Conda ────────────────────────────────────────────────────────────────────
set +u   # conda's activate scripts reference unset variables
__conda_setup="$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate neurostorm
cd "$REPO_DIR"
source ./set_env.sh
set -u

# ── Runtime environment ──────────────────────────────────────────────────────
export FOLDS="$SLURM_ARRAY_TASK_ID"          # this task runs exactly one fold
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"

# ── Record what we ran on ────────────────────────────────────────────────────
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} (fold $FOLDS) on $(hostname) at $(date)"
echo "Repo: $REPO_DIR @ $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible!'; \
print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"

# ── Run ──────────────────────────────────────────────────────────────────────
srun bash "$FINETUNE_SCRIPT"

echo "Fold $FOLDS finished at $(date)"

#!/bin/bash
# =============================================================================
# Stage B of the two-stage test: run each fold's EXISTING 3-way LOOCV checkpoint on
# all Stage A test blocks (real task + mock rest) of that fold's held-out subject.
# Inference only, one fold per array task. See analysis/predict_stage_b_on_stage_a_blocks.py.
#
# One-time setup:  mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit:
#   sbatch --array=101 analysis/sbatch_stage_b_predict.sh        # just fold 101 (check this first)
#   sbatch analysis/sbatch_stage_b_predict.sh                    # all 26 folds
#   sbatch --array=101-126%4 analysis/sbatch_stage_b_predict.sh  # max 4 at once
# Watch:   squeue -u mkersey      tail -f logs/stageB_predict_<jobid>_<fold>.log
# =============================================================================
#SBATCH --account=huangzu99
#SBATCH --job-name=stageB_predict
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=spgpu
#SBATCH --array=101-126
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00                # 60 blocks of inference + dataset setup; raise if the data load is slow
#SBATCH --output=logs/stageB_predict_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

REPO_DIR="$HOME/code/NeuroSTORM"
export DATA_ROOT="/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM"

module load gcc/11.2.0 cuda/12.8.1
module load mamba

set +u
__conda_setup="$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate neurostorm
cd "$REPO_DIR"
source ./set_env.sh
export PATH="$HOME/gcc11-wrappers:$PATH"
set -u

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} (fold $SLURM_ARRAY_TASK_ID) on $(hostname) at $(date)"
echo "Repo: $REPO_DIR @ $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
# DEVICE=cuda (default) or cpu. For CPU: sbatch --export=ALL,DEVICE=cpu --partition=standard --gpus-per-node=0 ...
DEVICE="${DEVICE:-cuda}"
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible!'; print('torch', torch.__version__)"
fi

python analysis/predict_stage_b_on_stage_a_blocks.py --fold "$SLURM_ARRAY_TASK_ID" --device "$DEVICE"

echo "Fold $SLURM_ARRAY_TASK_ID finished at $(date)"

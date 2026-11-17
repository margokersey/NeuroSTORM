#!/bin/bash
# =============================================================================
# Test 3 (task decoding during LOR): run each fold's EXISTING 3-way LOOCV checkpoint on
# every block of that fold's held-out subject in imagery run 2 and run 3, tagged by
# LOR segment. Inference only, one fold per array task.
# See analysis/predict_lor_task_decoding.py.
# Needs: preprocessed task-imagery_run-2/3 (img/ + metadata/timing_onehot_task{2,3}.tsv)
# and the LOR/ROR CSV (set LOR_CSV below or export NEUROSTORM_LOR_CSV).
#
# One-time setup:  mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit:
#   sbatch --array=101 analysis/sbatch_lor_task_decoding.sh        # just fold 101 (check this first)
#   sbatch analysis/sbatch_lor_task_decoding.sh                    # all 26 folds
#   sbatch --array=101-126%4 analysis/sbatch_lor_task_decoding.sh  # max 4 at once
#   LOR_RUNS=4 sbatch --array=101 analysis/sbatch_lor_task_decoding.sh   # run 4 (recovery) only; writes
#                                                                        # lor_predictions_run4_fold_<N>.json
#   LOR_TASK=tennisnav sbatch --array=101-126 analysis/sbatch_lor_task_decoding.sh   # 2-class tennis-vs-navigation model
#                                  (add LOR_RUNS=4 for run 4; outputs go to ..._lor_tennisnav_decoding_fold-<N>)
# Watch:   squeue -u mkersey      tail -f logs/lor_decode_<jobid>_<fold>.log
# =============================================================================
#SBATCH --account=huangzu99
#SBATCH --job-name=lor_decode
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=gpu_mig40         # one MIG slice is enough (small model); starts much sooner than spgpu
#SBATCH --array=101-126
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00                # run 4 alone (~30 blocks) fit well inside 30 min; runs 2-3 (~120 blocks) get 1 h. Raise if the data load is slow
#SBATCH --output=logs/lor_decode_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

REPO_DIR="$HOME/code/NeuroSTORM"
export NEUROSTORM_SPLIT_ROOT="/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/splits"
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
python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible!'; print('torch', torch.__version__)"

LOR_CSV="${LOR_CSV:-$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-2/metadata/LOR_ROR_timing.csv}"
[ -f "$LOR_CSV" ] || { echo "LOR/ROR CSV not found: $LOR_CSV (set LOR_CSV)"; exit 1; }

python analysis/predict_lor_task_decoding.py --fold "$SLURM_ARRAY_TASK_ID" --lor_csv "$LOR_CSV" --runs "${LOR_RUNS:-2,3}" --task "${LOR_TASK:-3way}"

echo "Fold $SLURM_ARRAY_TASK_ID finished at $(date)"

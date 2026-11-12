#!/bin/bash
# =============================================================================
# LOOCV Permutation Testing on Great Lakes
#
# Submits parallel job array to train models on N randomized label permutations.
# Each array task trains ONE fold's permutation models in sequence.
#
# Prerequisites:
#   1. Real LOOCV models (folds 101-126) already trained and completed
#   2. No pre-generation needed; labels are permuted at train time via --permute_labels flag
#
# One-time setup:   mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit:
#   sbatch run_finetune_scripts/greatlakes/loocv/sbatch_loocv_permutation_testing.sh
#
# Submit specific fold:
#   sbatch --array=101 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_permutation_testing.sh
#
# Submit range with concurrency limit:
#   sbatch --array=101-126%4 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_permutation_testing.sh
#
# Watch:   squeue -u mkersey
#          tail -f logs/xcpd_loocv_perm_<jobid>_<fold>.log
# =============================================================================

#SBATCH --account=huangzu99
#SBATCH --job-name=xcpd_loocv_perm
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=spgpu              # A40s
#SBATCH --array=101-126                # One task per fold; %2 suffix = max 2 concurrent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    
#SBATCH --time=10:00:00                # 3 permutations × 2 hours each, per fold
#SBATCH --output=logs/xcpd_loocv_perm_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

# ── User settings (edit these) ───────────────────────────────────────────────
REPO_DIR="$HOME/code/NeuroSTORM"
FINETUNE_SCRIPT="${REPO_DIR}/run_finetune_scripts/greatlakes/loocv/run_loocv_xcpd.sh"
export DATA_ROOT="/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM"

N_PERMUTATIONS=5           # Permutations per fold. p < 0.05 needs >= 19; use 100+ to report (scale --time / array accordingly)
MAX_EPOCHS=30              # Epochs for training on random labels (fewer than real may be needed)

# ── Modules ──────────────────────────────────────────────────────────────────
module load gcc/11.2.0 cuda/12.8.1
module load mamba

# ── Conda ────────────────────────────────────────────────────────────────────
set +u
__conda_setup="$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate neurostorm
cd "$REPO_DIR"
source ./set_env.sh
export PATH="$HOME/gcc11-wrappers:$PATH"
set -u

# ── Runtime environment ──────────────────────────────────────────────────────
export FOLDS="$SLURM_ARRAY_TASK_ID"
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
export TQDM_MININTERVAL=120

# ── Job-specific variables ───────────────────────────────────────────────────
FOLD_ID="$SLURM_ARRAY_TASK_ID"
PROJECT_PREFIX="xcpd_nosmooth_general"

echo "============================================================================"
echo "LOOCV Permutation Testing: Fold $FOLD_ID (Job $SLURM_JOB_ID)"
echo "============================================================================"
echo "N_PERMUTATIONS:  $N_PERMUTATIONS"
echo "MAX_EPOCHS:      $MAX_EPOCHS"
echo "GPU_IDS:         ${NUM_GPUS}"
echo ""

# ── Train Each Permutation for This Fold ─────────────────────────────────────
for perm_idx in $(seq 1 "$N_PERMUTATIONS"); do
    echo ""
    echo "─── Fold $FOLD_ID, Permutation $perm_idx / $N_PERMUTATIONS ───"
    
    # Uses the original split file, with --permute_labels; each permutation
    # gets a different seed for label shuffling.
    perm_seed=$((1234 + perm_idx))

    bash "$FINETUNE_SCRIPT" \
        --start_index "$FOLD_ID" \
        --n_folds 1 \
        --max_epochs "$MAX_EPOCHS" \
        --project_name_suffix "_perm${perm_idx}" \
        --permute_labels --permutation_seed "$perm_seed" \
        || { echo "ERROR: Training failed for fold $FOLD_ID perm $perm_idx"; exit 1; }
done

echo ""
echo "✓ Fold $FOLD_ID: All $N_PERMUTATIONS permutations complete"

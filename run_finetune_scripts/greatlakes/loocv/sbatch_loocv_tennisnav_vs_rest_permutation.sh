#!/bin/bash
# =============================================================================
# Stage A (tennis+navigation task vs. no task) LOOCV permutation testing on Great Lakes
#
# One array task per (fold, permutation): 26 folds x 10 permutations = 260 tasks,
# each a single fine-tune on within-subject-shuffled labels, so every fold and
# every permutation can run in parallel. Task id -> fold/perm:
#     fold = 101 + id / N_PERMS        perm = PERM_START + id % N_PERMS
# Seeds are 1234 + 1000*fold + perm, so every (fold, perm) run gets an independent label shuffle
# (the older 3-way / Task-vs-Rest scripts use 1234+perm, which repeats shuffles across folds).
# Output folders:
#     FIXED_mich_xcpd_nosmooth_tennisnav_vs_rest_loocv_fold-<fold>_perm<perm>
#
# Prerequisites: none beyond the splits (real runs are independent of these).
# One-time setup:   mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit (all 26 folds x 10 perms):
#   sbatch run_finetune_scripts/greatlakes/loocv/sbatch_loocv_tennisnav_vs_rest_permutation.sh
# Change the cap:     sbatch --array=0-259%10 <this script>   (default %20)
# Rerun one task:     sbatch --array=<id> <this script>   (id = (fold-101)*10 + perm-1)
# More perms later (e.g. 11-20; array size = 26 * 10 = 260 again):
#   sbatch --export=ALL,PERM_START=11,PERM_END=20 <this script>
# Watch:   squeue -u mkersey      tail -f logs/tennisnav_vs_rest_loocv_perm_<jobid>_<id>.log
# =============================================================================

#SBATCH --account=huangzu99
#SBATCH --job-name=tennisnav_vs_rest_loocv_perm
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=spgpu              # A40s
#SBATCH --array=0-259%20               # 26 folds x 10 perms; one task each, max 20 running at once
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00                # one run (same budget as a real fold)
#SBATCH --output=logs/tennisnav_vs_rest_loocv_perm_%A_%a.log
#SBATCH --mail-type=FAIL               # not END: 260 emails otherwise
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

# ── User settings (edit these) ───────────────────────────────────────────────
REPO_DIR="$HOME/code/NeuroSTORM"
FINETUNE_SCRIPT="${REPO_DIR}/run_finetune_scripts/greatlakes/loocv/run_loocv_tennisnav_vs_rest.sh"
export DATA_ROOT="/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM"

FIRST_FOLD=101
N_FOLDS=26
PERM_START="${PERM_START:-1}"
PERM_END="${PERM_END:-10}"
N_PERMS=$((PERM_END - PERM_START + 1))
MAX_EPOCHS=30

TASK_ID="$SLURM_ARRAY_TASK_ID"
if (( TASK_ID >= N_FOLDS * N_PERMS )); then
    echo "Array id $TASK_ID out of range for $N_FOLDS folds x $N_PERMS perms"; exit 1
fi
FOLD_ID=$((FIRST_FOLD + TASK_ID / N_PERMS))
perm_idx=$((PERM_START + TASK_ID % N_PERMS))
perm_seed=$((1234 + 1000 * FOLD_ID + perm_idx))   # fold-specific: independent shuffle per (fold, perm)

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
export FOLDS="$FOLD_ID"
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
export TQDM_MININTERVAL=120

echo "Tennis/nav task-vs-rest permutation: task $TASK_ID -> fold $FOLD_ID perm $perm_idx (seed $perm_seed), job $SLURM_JOB_ID"

bash "$FINETUNE_SCRIPT" \
    --start_index "$FOLD_ID" \
    --n_folds 1 \
    --max_epochs "$MAX_EPOCHS" \
    --project_name_suffix "_perm${perm_idx}" \
    --permute_labels --permutation_seed "$perm_seed"

echo "✓ Fold $FOLD_ID perm $perm_idx complete"

#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint (MAE-pretrained on 5
# datasets, never saw a downstream label) on the XCP-D-denoised
# task-imagery_run-1 data (no smoothing), full fine-tune (--tpt_strategy none).
# Great Lakes version of local/5fold/run_finetune_xcpd_cv.sh: data root and
# fold selection come from the environment so Slurm array tasks can run one
# fold each.
#
# Prerequisites:
#   1. scripts/configs/datasets/michigan_xcpd*.yaml are in place, and any paths
#      inside them point at the Great Lakes copy of the data.
#   2. The preprocessed data (img/sub-*/data.pt + metadata/) has been copied to
#      DATA_ROOT.
#   3. GENERAL_CKPT exists.
#   4. nas/splits/MICHIGAN/split_fixed_1.txt .. split_fixed_5.txt exist.
#
# Run from the repo root:
#   bash run_finetune_scripts/greatlakes/5fold/run_finetune_xcpd_cv.sh              # all folds, in order
#   FOLDS=3 bash run_finetune_scripts/greatlakes/5fold/run_finetune_xcpd_cv.sh      # just fold 3
#   FOLDS="2 4" bash run_finetune_scripts/greatlakes/5fold/run_finetune_xcpd_cv.sh  # folds 2 and 4
# On Great Lakes, sbatch_finetune_xcpd_cv.sh sets FOLDS per array task.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Where the preprocessed data lives. Overridable from the job file.
DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"

# Original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5
STRATEGY="none"
DS_LABEL="nosmooth"

# Which folds to run: all by default, or whatever FOLDS is set to.
FOLDS="${FOLDS:-$(seq 1 "$N_FOLDS")}"

# Number of GPUs this job can see (set by the job file; 1 if run by hand).
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))   # "0" for 1 GPU, "0,1" for 2, ...

# ── Preflight ────────────────────────────────────────────────────────────────
echo "Preflight: checking split-file subjects exist under $IMG_ROOT..."
preflight_require_dir "$IMG_ROOT" "has the data been copied/preprocessed?"
preflight_require_file "$GENERAL_CKPT" "run from the repo root?"
preflight_require_splits "$SPLIT_DIR" "split_fixed_*.txt"
for split_file in "$SPLIT_DIR"/split_fixed_*.txt; do
    preflight_check_subjects "$split_file" "$IMG_ROOT" nosmooth
done
preflight_finish "all split-file subjects found."

# ── 5-fold CV fine-tune ──────────────────────────────────────────────────────
# Best model with optimally selected window size (28 frames total, 2 TRs trimmed from start).
for i in $FOLDS; do
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus "$GPU_IDS" \
        --load_model_path "$GENERAL_CKPT" \
        -- --dataset_split_num "$i" --tpt_strategy "$STRATEGY" \
            --project_name "FIXED_mich_xcpd_${DS_LABEL}_tpt_strategy-${STRATEGY}_fold-${i}"
done

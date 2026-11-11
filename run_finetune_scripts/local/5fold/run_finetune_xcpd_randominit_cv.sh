#!/bin/bash
# 5-fold CV fine-tune on the XCP-D-denoised (no smoothing) task-imagery_run-1
# data from RANDOM INITIALIZATION: no --load_model_path is passed, so the
# pretrained checkpoint is deliberately NOT used. Full fine-tune
# (--tpt_strategy none).
#
# Prerequisites: same as run_finetune_xcpd_cv.sh (dataset config, preprocessed
# data under IMG_ROOT, nas/splits/MICHIGAN/split_fixed_1..5.txt).
#
# Run from the repo root:
#   bash run_finetune_scripts/local/5fold/run_finetune_xcpd_randominit_cv.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

# Original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5
DS_LABEL="nosmooth"

# ── Preflight ────────────────────────────────────────────────────────────────
echo "Preflight: checking split-file subjects exist under $IMG_ROOT..."
preflight_require_dir "$IMG_ROOT" "has preprocessing_volume.py been run yet?"
preflight_require_splits "$SPLIT_DIR" "split_fixed_*.txt"
for split_file in "$SPLIT_DIR"/split_fixed_*.txt; do
    preflight_check_subjects "$split_file" "$IMG_ROOT" nosmooth
done
preflight_finish "all split-file subjects found under $IMG_ROOT."

# ── 5-fold CV fine-tune from scratch ─────────────────────────────────────────
echo "--- Dataset: $DATASET (label=$DS_LABEL) ---"

for i in $(seq 1 "$N_FOLDS"); do
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus 0 \
        -- --dataset_split_num "$i" --tpt_strategy none \
            --project_name "FIXED_mich_xcpd_${DS_LABEL}_tpt_strategy-none_randominit_fold-${i}"
done

echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_nosmooth_tpt_strategy-none_randominit_fold-*'"

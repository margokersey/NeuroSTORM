#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint (MAE-pretrained on 5
# datasets, never saw a downstream label) on the XCP-D-denoised
# task-imagery_run-1 data (no smoothing), full fine-tune (--tpt_strategy none).
# To compare against the sex-specific checkpoint, swap GENERAL_CKPT for
# pretrained_models/pretraining/neurostorm_hcpya_sex.ckpt and change the
# project_name suffix accordingly.
#
# Prerequisites:
#   1. scripts/configs/datasets/michigan_xcpd.yaml is in place.
#   2. Preprocessing has produced img/sub-*/data.pt (+ metadata) under IMG_ROOT
#      and SMOOTH_IMG_ROOT.
#   3. GENERAL_CKPT exists.
#   4. nas/splits/MICHIGAN/split_fixed_1.txt .. split_fixed_5.txt exist.
#
# Run from the repo root:
#   bash run_finetune_scripts/local/5fold/run_finetune_xcpd_cv.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"

# Smoothed XCP-D outputs (preflight-checked only; not run by this script)
SMOOTH_IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_smooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5
STRATEGY="none"
DS_LABEL="nosmooth"

# ── Preflight ────────────────────────────────────────────────────────────────
echo "Preflight: checking split-file subjects exist under $IMG_ROOT and $SMOOTH_IMG_ROOT ..."
preflight_require_dir "$IMG_ROOT" "has preprocessing_volume.py been run yet?"
preflight_require_dir "$SMOOTH_IMG_ROOT" "has preprocessing_volume.py been run yet?"
preflight_require_splits "$SPLIT_DIR" "split_fixed_*.txt"
for split_file in "$SPLIT_DIR"/split_fixed_*.txt; do
    preflight_check_subjects "$split_file" "$IMG_ROOT" nosmooth
    preflight_check_subjects "$split_file" "$SMOOTH_IMG_ROOT" smooth
done
preflight_finish "all split-file subjects found under $IMG_ROOT."

# ── 5-fold CV fine-tune ──────────────────────────────────────────────────────
# Best model with optimally selected window size (28 frames total, 2 TRs trimmed from start).
# Other configurations tried previously (see git history): tpt_strategy
# none/prompt/prompt_ln crossed with the nosmooth/smooth datasets.
for i in $(seq 1 "$N_FOLDS"); do
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus 0 \
        --load_model_path "$GENERAL_CKPT" \
        -- --dataset_split_num "$i" --tpt_strategy "$STRATEGY" \
            --project_name "FIXED_mich_xcpd_${DS_LABEL}_tpt_strategy-${STRATEGY}_fold-${i}"
done

echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'mich_xcpd_no_smoothing_tpt_strategy-*'"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'mich_xcpd_smoothed_tpt_strategy-*'"

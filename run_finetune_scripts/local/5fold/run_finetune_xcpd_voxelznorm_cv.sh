#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint on the XCP-D-denoised
# task-imagery_run-1 data with voxel-wise z-normalization, for both the
# nosmooth and smooth variants and for --tpt_strategy none (full fine-tune)
# and prompt_ln.
#
# Prerequisites: same as run_finetune_xcpd_cv.sh, but with the *_voxelznorm
# datasets/data directories below.
#
# Run from the repo root:
#   bash run_finetune_scripts/local/5fold/run_finetune_xcpd_voxelznorm_cv.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Original (no-smoothing) XCP-D outputs, voxel z-normalized
DATASET="michigan_xcpd_nosmooth_voxelznorm_fullblock"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_voxelznorm_task-imagery_run-1/img"

# Smoothed XCP-D outputs, voxel z-normalized (run after the nosmooth run)
SMOOTH_DATASET="michigan_xcpd_smooth_voxelznorm_fullblock"
SMOOTH_IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_smooth_voxelznorm_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5
TPT_CHOICES=("none" "prompt_ln")

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

# ── 5-fold CV fine-tune: tpt_strategy x {nosmooth, smooth} x fold ────────────
for strategy in "${TPT_CHOICES[@]}"; do
    echo "=============================================="
    echo " Running tpt_strategy = $strategy"
    echo "=============================================="

    for ds_tag in ns sm; do
        if [[ "$ds_tag" == "ns" ]]; then
            ds_name="$DATASET"
            ds_label="nosmooth_voxelznorm"
        else
            ds_name="$SMOOTH_DATASET"
            ds_label="smooth_voxelznorm"
        fi

        echo "--- Dataset: $ds_name (label=$ds_label) ---"

        for i in $(seq 1 "$N_FOLDS"); do
            echo "=== Fold $i/$N_FOLDS ==="
            bash scripts/run_experiment.sh \
                --model neurostorm --dataset "$ds_name" --task "$TASK" --mode finetune \
                --max_epochs 50 --batch_size 8 --gpus 0 \
                --load_model_path "$GENERAL_CKPT" \
                -- --dataset_split_num "$i" --tpt_strategy "$strategy" \
                   --project_name "FIXED_mich_xcpd_${ds_label}_tpt_strategy-${strategy}_fold-${i}"
        done
    done
done

echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_nosmooth_voxelznorm_tpt_strategy-*'"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_smooth_voxelznorm_tpt_strategy-*'"

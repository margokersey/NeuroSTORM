#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint (MAE-pretrained on 5
# datasets, never saw a downstream label) on the original AFNI-preprocessed
# task-imagery_run-1 data, run for both --tpt_strategy none (full fine-tune)
# and prompt_ln.
#
# Prerequisites:
#   1. The AFNI dataset config (michigan_afni_fullblock) is in place under
#      scripts/configs/datasets/.
#   2. Preprocessing has produced img/sub-*/data.pt under IMG_ROOT.
#   3. GENERAL_CKPT exists.
#   4. nas/splits/MICHIGAN/split_fixed_1.txt .. split_fixed_5.txt exist.
#
# Run from the repo root:
#   bash run_finetune_scripts/local/5fold/run_finetune_afni_cv.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Original AFNI outputs
DATASET="michigan_afni_fullblock"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_afni_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5
TPT_CHOICES=("none" "prompt_ln")
DS_LABEL="afni"

# ── Preflight ────────────────────────────────────────────────────────────────
echo "Preflight: checking split-file subjects exist under $IMG_ROOT..."
preflight_require_dir "$IMG_ROOT" "has preprocessing_volume.py been run yet?"
preflight_require_splits "$SPLIT_DIR" "split_fixed_*.txt"
for split_file in "$SPLIT_DIR"/split_fixed_*.txt; do
    preflight_check_subjects "$split_file" "$IMG_ROOT" afni
done
preflight_finish "all split-file subjects found under $IMG_ROOT."

# ── 5-fold CV fine-tune, once per tpt_strategy ───────────────────────────────
for strategy in "${TPT_CHOICES[@]}"; do
    echo "=============================================="
    echo " Running tpt_strategy = $strategy"
    echo "=============================================="
    echo "--- Dataset: $DATASET (label=$DS_LABEL) ---"

    for i in $(seq 1 "$N_FOLDS"); do
        echo "=== Fold $i/$N_FOLDS ==="
        bash scripts/run_experiment.sh \
            --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
            --max_epochs 50 --batch_size 8 --gpus 0 \
            --load_model_path "$GENERAL_CKPT" \
            -- --dataset_split_num "$i" --tpt_strategy "$strategy" \
                --project_name "FIXED_${DS_LABEL}_smooth_voxelznorm_tpt_strategy-${strategy}_fold-${i}"
    done
done

echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_${DS_LABEL}_smooth_voxelznorm_tpt_strategy-*'"

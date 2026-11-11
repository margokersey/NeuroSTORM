#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint on the XCP-D-denoised
# (no smoothing) task-imagery_run-1 data using the POOLED splits
# (nas/splits/MICHIGAN_POOLED), full fine-tune (--tpt_strategy none).
# No preflight check is run for this script.
#
# Run from the repo root:
#   bash run_finetune_scripts/local/5fold/run_finetune_xcpd_pooled_cv.sh

set -euo pipefail

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"
DATASET="michigan_xcpd_nosmooth_fullblock"

TASK="task6"
N_FOLDS=5
FIRST_FOLD=2   # folds 2..N_FOLDS (fold 1 is not run by this script)
STRATEGY="none"
DS_LABEL="nosmooth"

# Best model with optimally selected window size (28 frames total, 2 TRs trimmed from start).
for i in $(seq "$FIRST_FOLD" "$N_FOLDS"); do
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus 0 \
        --load_model_path "$GENERAL_CKPT" \
        -- --dataset_split_num "$i" --tpt_strategy "$STRATEGY" \
            --project_name "FIXED_mich_xcpd_${DS_LABEL}_tpt_strategy-${STRATEGY}_pooled_fold-${i}"
done

echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_nosmooth_tpt_strategy-none_pooled_fold-*' --metric acc --output_csv mich_xcpd_nosmooth_tpt_strategy-none_pooled.csv"

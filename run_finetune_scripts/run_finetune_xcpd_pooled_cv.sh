#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint (MAE-pretrained on
# 5 datasets, never seen a downstream label -- pt_neurostorm_mae_5ds.ckpt)
# on the new XCP-D-denoised task-imagery_run-1 data, using --tpt_strategy
# none (full fine-tune) to match the general-vs-sex-specific-checkpoint
# comparison already run on the fMRIPrep version of this data.
#
# Prerequisites:
#   1. scripts/configs/datasets/michigan_xcpd.yaml is in place (delivered
#      alongside this script -- copy it into scripts/configs/datasets/).
#   2. patch_add_michigan_xcpd.py has already been run and
#      preprocessing_volume.py has produced
#      /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/task-imagery_run-1/
#      (img/sub-*/data.pt + metadata/timing_onehot_task1.tsv copied over).
#   3. GENERAL_CKPT below exists.
#   4. nas/splits/MICHIGAN/split_fixed_1.txt .. split_fixed_5.txt already
#      exist (reused as-is from the fMRIPrep runs -- same subjects, only
#      the preprocessing changed).
#
# Run from the repo root:
#   bash run_finetune_xcpd_cv.sh
#
# Edit GENERAL_CKPT / --max_epochs / --batch_size / --gpus / --tpt_strategy
# below to taste. To compare against the sex-specific checkpoint instead,
# just swap GENERAL_CKPT for pretrained_models/pretraining/neurostorm_hcpya_sex.ckpt
# and change the project_name suffix accordingly.

set -euo pipefail

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"
# dataset for the original (no-smoothing) XCP-D outputs
DATASET=michigan_xcpd_nosmooth_fullblock
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN_POOLED"
N_FOLDS=5

# ------------------------------------------------------------------
# 5-fold CV finetune. Note the braces around ${i} in --project_name below:
# without them, bash parses a trailing "_something" right after "$i" as
# part of ONE variable name (e.g. "$i_general" means the variable literally
# named "i_general", not "$i" followed by the text "_general") -- since
# that variable is unset it silently expands to empty, and every fold gets
# the same project name. This exact bug hit a real run before.
# ------------------------------------------------------------------

# best model with optimally selected window size (28 frames total, 2 TRs trimmed from start)
for i in $(seq 2 "$N_FOLDS"); do
    strategy="none"
    ds_name="$DATASET"
    ds_label="nosmooth"
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$ds_name" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus 0 \
        --load_model_path "$GENERAL_CKPT" \
        -- --dataset_split_num "$i" --tpt_strategy "$strategy" \
            --project_name "FIXED_mich_xcpd_${ds_label}_tpt_strategy-${strategy}_pooled_fold-${i}"
done

echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_nosmooth_tpt_strategy-none_pooled_fold-*' --metric acc --output_csv mich_xcpd_nosmooth_tpt_strategy-none_pooled.csv"
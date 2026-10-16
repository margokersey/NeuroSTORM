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

# GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt" # DO NOT USE - WE WANT RANDOM INIT
# dataset for the original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5

# ------------------------------------------------------------------
# Preflight: confirm every subject named in each split file actually has a
# preprocessed data.pt under the new XCP-D image_path. A subject that
# failed patch_add_michigan_xcpd.py's mask-shape assertion during
# preprocessing (space/resolution mismatch between xcp_output and
# fmriprep_output) would simply be MISSING here rather than erroring --
# better to catch that now than partway through fold 3 of 5.
# ------------------------------------------------------------------
echo "Preflight: checking split-file subjects exist under $IMG_ROOT..."
if [[ ! -d "$IMG_ROOT" ]]; then
    echo "Preflight FAILED: $IMG_ROOT does not exist -- has preprocessing_volume.py been run yet?"
    exit 1
fi
if ! compgen -G "$SPLIT_DIR"/split_fixed_*.txt > /dev/null; then
    echo "Preflight FAILED: no split files found under $SPLIT_DIR"
    exit 1
fi

missing=0
for split_file in "$SPLIT_DIR"/split_fixed_*.txt; do
    for subj in $(grep -vE '^(train|val|test)_subjects$' "$split_file"); do
        [[ -z "$subj" ]] && continue
        if [[ ! -f "$IMG_ROOT/$subj/data.pt" ]]; then
            echo "  MISSING (nosmooth): $subj (referenced in $split_file, no data.pt at $IMG_ROOT/$subj/data.pt)"
            missing=$((missing + 1))
        fi
    done
done
if [[ "$missing" -gt 0 ]]; then
    echo "Preflight FAILED: $missing missing subject(s) -- fix before running folds (see above)."
    exit 1
fi
echo "Preflight OK: all split-file subjects found under $IMG_ROOT."
echo ""

# ------------------------------------------------------------------
# 5-fold CV finetune. Note the braces around ${i} in --project_name below:
# without them, bash parses a trailing "_something" right after "$i" as
# part of ONE variable name (e.g. "$i_general" means the variable literally
# named "i_general", not "$i" followed by the text "_general") -- since
# that variable is unset it silently expands to empty, and every fold gets
# the same project name. This exact bug hit a real run before.
# ------------------------------------------------------------------


ds_name="$DATASET"
ds_label="nosmooth"

echo "--- Dataset: $ds_name (label=$ds_label) ---"

# Note that I removed the line that loads the pre-trained checkpoint since we want to try fine-tuning from scratch
for i in $(seq 1 "$N_FOLDS"); do
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$ds_name" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus 0 \
        -- --dataset_split_num "$i" --tpt_strategy none \
            --project_name "FIXED_mich_xcpd_${ds_label}_tpt_strategy-none_randominit_fold-${i}"
done


echo ""
echo "All runs done. Aggregate by pattern, for example:"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_no_smoothing_tpt_strategy-*'"
echo "  python scripts/aggregate_cv_results.py --project_name_glob 'FIXED_mich_xcpd_smoothed_tpt_strategy-*'"

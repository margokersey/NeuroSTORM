#!/bin/bash
# Leave-one-subject-out (LOOCV) fine-tune of the WINNING condition identified
# by compare_conditions_v2.py -- FIXED_mich_xcpd_nosmooth_tpt_strategy-none
# (XCP-D-denoised data, NeuroSTORM's general MAE-pretrained checkpoint, full
# fine-tune) -- same model, data and hyperparameters as the 5-fold CV in
# local/5fold/run_finetune_xcpd_cv.sh, just a different subject partition:
#
#   5-fold CV:  ~16 train / ~5 val / ~5 test subjects per fold, 5 folds.
#   LOOCV:      22 train / 3 val / 1 test subject per fold, 26 folds.
#
# Why: each fold trains on ~50% more subjects than the 5-fold CV, so it is
# closer to a model trained on the whole cohort. The 3-subject validation set
# (sampled from the training subjects) selects the checkpoint without touching
# the held-out subject, and 26 single-subject test sets give a per-subject
# distribution of generalization instead of 5 noisy group-level numbers.
#
# Cost: 26 fine-tunes with more data per fold, roughly 7-8x the compute of the
# 5-fold run. If that is too much, lower --max_epochs (e.g. 20) after checking
# the training curves on fold 101.
#
# Prerequisites:
#   1. Generate the LOOCV split files on the machine where
#      nas/splits/MICHIGAN/split_fixed_1..5.txt exist:
#        python scripts/make_loocv_splits.py \
#          --dataset_name MICHIGAN --source_fold 1 --start_index 101
#      This writes nas/splits/MICHIGAN/split_fixed_101.txt .. split_fixed_126.txt
#      (indices 101+ so the 5-fold files 1-5 are left untouched).
#   2. Same GENERAL_CKPT / michigan_xcpd.yaml / XCP-D data as the 5-fold run.
#
# Run from the repo root:
#   bash run_finetune_scripts/local/loocv/run_loocv_xcpd.sh
#
# Options: --start_index / --n_folds (if you generated a different range),
# --max_epochs / --batch_size / --gpus, or --dry_run to only print the
# preflight + planned run without launching anything.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"
DATASET="michigan_xcpd_nosmooth_fullblock"
TASK="task6"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"
SPLIT_DIR="nas/splits/MICHIGAN"
PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv"

START_INDEX=101
N_FOLDS=26
MAX_EPOCHS=50
BATCH_SIZE=8
GPUS=0
DRY_RUN=0
# Comma-separated list of fold numbers to skip (e.g. "120" or "120,123")
SKIP_FOLDS="120,101,102,103"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_index) START_INDEX="$2"; shift 2 ;;
        --n_folds) N_FOLDS="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

END_INDEX=$((START_INDEX + N_FOLDS - 1))

# ── Preflight ────────────────────────────────────────────────────────────────
# Check every LOOCV split file and its subjects up front, rather than finding
# out fold 19 of 26 is broken after 18 folds have finished.
echo "Preflight: checking LOOCV split files $SPLIT_DIR/split_fixed_${START_INDEX}.txt .. split_fixed_${END_INDEX}.txt ..."
preflight_require_dir "$IMG_ROOT"
for i in $(seq "$START_INDEX" "$END_INDEX"); do
    split_file="$SPLIT_DIR/split_fixed_${i}.txt"
    if [[ ! -f "$split_file" ]]; then
        echo "  MISSING split file: $split_file -- run make_loocv_splits.py first (see header comment)."
        MISSING=$((MISSING + 1))
        continue
    fi
    preflight_check_subjects "$split_file" "$IMG_ROOT"
done
preflight_finish "all $N_FOLDS LOOCV split files present, all referenced subjects found."

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry run] would run dataset_split_num $START_INDEX..$END_INDEX with:"
    echo "  --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE --gpus $GPUS"
    echo "  --project_name ${PROJECT_PREFIX}_fold-<split_num>"
    exit 0
fi

# ── LOOCV fine-tune ──────────────────────────────────────────────────────────
for i in $(seq "$START_INDEX" "$END_INDEX"); do
    if [[ ",${SKIP_FOLDS}," == *",${i},"* ]]; then
        echo "Skipping fold $i per SKIP_FOLDS"
        continue
    fi
    echo "=== LOOCV fold $i (held-out subject $((i - START_INDEX + 1))/$N_FOLDS) ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
        --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPUS" \
        --load_model_path "$GENERAL_CKPT" \
        -- --dataset_split_num "$i" --tpt_strategy none \
             --project_name "${PROJECT_PREFIX}_fold-${i}"
done

echo ""
echo "All $N_FOLDS folds done. Summarize with:"
echo "  python analysis/compare_conditions_v2.py \\"
echo "    --condition \"LOOCV (best model)|${PROJECT_PREFIX}_fold-*\""
echo ""
echo "Note: this is a standalone condition (26 folds, fold numbers ${START_INDEX}-${END_INDEX})"
echo "-- it won't be paired-tested against the 5-fold CV conditions (different fold"
echo "identities / different train-set sizes per fold), but its mean/SEM accuracy and"
echo "AUROC across all 26 held-out subjects, plus the per-class breakdown, is the"
echo "cleanest read on this model's ceiling given all the data you have."

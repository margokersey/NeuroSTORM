#!/bin/bash
# run_loocv_xcpd.sh
#
# Leave-one-subject-out (LOOCV) fine-tune of the WINNING condition
# identified by compare_conditions_v2.py -- FIXED_mich_xcpd_nosmooth_tpt_
# strategy-none (XCP-D-denoised data, NeuroSTORM's general MAE-pretrained
# checkpoint, --tpt_strategy none / full fine-tune) -- same model, same
# data, same hyperparameters as run_finetune_xcpd_cv.sh's 5-fold CV, just
# a different subject partition:
#
#   5-fold CV (already run):  ~16 train / ~5 val / ~5 test subjects/fold,
#                              5 folds total.
#   LOOCV (this script):      22 train / 3 val / 1 test subject/fold,
#                              26 folds total (one held-out test subject per fold).
#
# Why this shows the model's "full potential": every fold trains on 22 of
# 26 subjects (~50% more subjects than the 5-fold CV's ~16), so each
# fold's model is closer to what you'd get training on essentially the
# whole cohort. The 3-subject validation set (sampled from the 22 training
# subjects per fold) provides robust checkpoint selection without using the
# held-out test subject. Because the test set is a SINGLE subject per fold,
# aggregating across all 26 folds gives a full per-subject distribution of
# generalization performance (which individuals the model nails vs. struggles
# with), not just 5 noisy group-level numbers.
#
# Cost: this is 26 full fine-tunes with MORE training data per fold than
# the 5-fold CV, i.e. roughly 26/5 x ~22/16 ~= 7-8x the total compute of
# the 5-fold run you already did. If that's too much wall-clock, lower
# --max_epochs below (e.g. 20) -- with 22 subjects/fold there's more data
# per epoch, so fewer epochs may already saturate valid_acc; check the
# training curves on fold 101 before committing to all 26.
#
# Prerequisites:
#   1. On the machine where nas/splits/MICHIGAN/split_fixed_1..5.txt
#      already exist, generate the LOOCV split files:
#
#        python scripts/make_loocv_splits.py \
#          --dataset_name MICHIGAN --source_fold 1 --start_index 101
#
#      This reads the 26-subject roster from split_fixed_1.txt and writes
#      nas/splits/MICHIGAN/split_fixed_101.txt .. split_fixed_126.txt
#      (indices 101+ so they can't collide with/overwrite the existing
#      5-fold files 1-5, which stay untouched and usable in parallel).
#      Each fold has: 22 training subjects, 3 validation subjects (for
#      checkpoint selection), 1 held-out test subject (for final evaluation).
#   2. Same GENERAL_CKPT / michigan_xcpd.yaml / XCP-D data as the 5-fold
#      run -- nothing else to set up if that already worked.
#
# Run from the repo root:
#   bash run_loocv_xcpd.sh
#
# Optional: --start_index / --n_folds if you generated a different range,
# --max_epochs / --batch_size / --gpus to override, or --dry_run to just
# print the preflight + planned commands without launching anything.

set -euo pipefail

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Where the preprocessed data lives. Overridable from the job file.
DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"

# dataset for the original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv"

START_INDEX=101
N_FOLDS=26
MAX_EPOCHS=30 # Reduced to 30 instead of 50 to save compute time (since it's been reaching max performance by 30 anyway in previous tests)
BATCH_SIZE=8

# Number of GPUs this job can see (set by the job file; 1 if run by hand).
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))   # "0" for 1 GPU, "0,1" for 2, ...

DRY_RUN=0
# Comma-separated list of fold numbers to skip (e.g. "120" or "120,123")
SKIP_FOLDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_index) START_INDEX="$2"; shift 2 ;;
        --n_folds) N_FOLDS="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --skip_folds) SKIP_FOLDS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

END_INDEX=$((START_INDEX + N_FOLDS - 1))

# ------------------------------------------------------------------
# Preflight: confirm every LOOCV split file exists and every subject it
# names has a preprocessed data.pt under the XCP-D image_path. Checking
# this up front avoids discovering fold 19 of 26 is broken after 18 folds
# have already finished.
# ------------------------------------------------------------------
echo "Preflight: checking LOOCV split files $SPLIT_DIR/split_fixed_${START_INDEX}.txt .. split_fixed_${END_INDEX}.txt ..."
if [[ ! -d "$IMG_ROOT" ]]; then
    echo "Preflight FAILED: $IMG_ROOT does not exist -- has the data been copied/preprocessed?"
    exit 1
fi
if [[ ! -f "$GENERAL_CKPT" ]]; then
    echo "Preflight FAILED: checkpoint $GENERAL_CKPT not found (run from the repo root?)"
    exit 1
fi
if ! compgen -G "$SPLIT_DIR"/split_fixed_*.txt > /dev/null; then
    echo "Preflight FAILED: no split files found under $SPLIT_DIR"
    exit 1
fi

missing=0
for i in $(seq "$START_INDEX" "$END_INDEX"); do
    split_file="$SPLIT_DIR/split_fixed_${i}.txt"
    if [[ ! -f "$split_file" ]]; then
        echo "  MISSING split file: $split_file -- run make_loocv_splits.py first (see header comment)."
        missing=$((missing + 1))
        continue
    fi
    for subj in $(grep -vE '^(train|val|test)_subjects$' "$split_file"); do
        [[ -z "$subj" ]] && continue
        if [[ ! -f "$IMG_ROOT/$subj/data.pt" ]]; then
            echo "  MISSING data.pt: $subj (referenced in $split_file)"
            missing=$((missing + 1))
        fi
    done
done
if [[ "$missing" -gt 0 ]]; then
    echo "Preflight FAILED: $missing problem(s) -- fix before running folds (see above)."
    exit 1
fi
echo "Preflight OK: all $N_FOLDS LOOCV split files present, all referenced subjects found."
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry run] would run dataset_split_num $START_INDEX..$END_INDEX with:"
    echo "  --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE --gpus $GPUS"
    echo "  --project_name ${PROJECT_PREFIX}_fold-<split_num>"
    exit 0
fi

# ------------------------------------------------------------------
# 26-fold LOOCV finetune. Note the braces around ${i} in --project_name:
# without them, "$i_general" would mean the unset variable "i_general",
# and every fold would silently get the same project name.
# ------------------------------------------------------------------
for i in $(seq "$START_INDEX" "$END_INDEX"); do
        # skip folds listed in SKIP_FOLDS
        if [[ -n "$SKIP_FOLDS" && ",${SKIP_FOLDS}," == *",${i},"* ]]; then
                echo "Skipping fold $i per SKIP_FOLDS"
                continue
        fi
        echo "=== LOOCV fold $i (held-out subject $((i - START_INDEX + 1))/$N_FOLDS) ==="
        bash scripts/run_experiment.sh \
            --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
            --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPU_IDS" \
            --load_model_path "$GENERAL_CKPT" \
            -- --dataset_split_num "$i" --tpt_strategy none \
                 --project_name "${PROJECT_PREFIX}_fold-${i}"
done

echo ""
echo "All $N_FOLDS folds done. Summarize with:"
echo "  python compare_conditions_v2.py \\"
echo "    --condition \"LOOCV (best model)|${PROJECT_PREFIX}_fold-*\""
echo ""
echo "Note: this is a standalone condition (26 folds, fold numbers ${START_INDEX}-${END_INDEX})"
echo "-- it won't be paired-tested against the 5-fold CV conditions (different fold"
echo "identities / different train-set sizes per fold), but its mean/SEM accuracy and"
echo "AUROC across all 26 held-out subjects, plus the per-class breakdown, is the"
echo "cleanest read on this model's ceiling given all the data you have."

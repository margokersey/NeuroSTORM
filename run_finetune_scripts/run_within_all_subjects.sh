#!/bin/bash
# run_within_all_subjects.sh
#
# Loops MICHIGAN_WITHIN fine-tuning over every subject's split_fixed_<k>.txt
# (k = 1..N, one per subject -- see make_within_pooled_splits.py and
# nas/splits/MICHIGAN_WITHIN/fold_to_subject.txt). Each fold trains and
# tests a COMPLETELY SEPARATE model on that one subject's own 21 train /
# 3 val / 6 test blocks -- N independent full fine-tunes, not one CV run.
#
# Usage:
#   bash run_within_all_subjects.sh \
#     --n_subjects 26 --tpt_strategy none \
#     --load_model_path output/neurostorm/pt_neurostorm_mae_5ds.ckpt
#
# Optional: --max_epochs (default 150 -- see note below), --batch_size
# (default 8), --gpus (default 0), plus anything after a literal -- is
# passed through as extra args to every fold's run.
#
# Why --max_epochs defaults higher than the between/pooled runs (30):
# a within-subject fold only has 21 train blocks vs. pooled's 546 or
# between's ~450 -- at batch_size 8 that's ~3 gradient steps/epoch instead
# of ~55-68. 150 epochs (~450 steps) partially compensates but is a guess,
# not a validated number -- watch val_loss/val_acc for this being too many
# (overfitting shows up fast with this little data) or too few (still
# improving at epoch 150).

set -euo pipefail

N_SUBJECTS=26
MAX_EPOCHS=300
BATCH_SIZE=7 # Changed to 7 so the 21 train blocks fit in 3 batches (7*3=21) -- no partial batch at the end that get dropped
GPUS=0
TPT_STRATEGY="none"
LOAD_MODEL_PATH="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_subjects) N_SUBJECTS="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --tpt_strategy) TPT_STRATEGY="$2"; shift 2 ;;
        --load_model_path) LOAD_MODEL_PATH="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TPT_STRATEGY" || -z "$LOAD_MODEL_PATH" ]]; then
    echo "Usage: bash run_within_all_subjects.sh --tpt_strategy <strategy> --load_model_path <ckpt> [--n_subjects N] [--max_epochs N] [--batch_size N] [--gpus N] [-- extra args...]"
    exit 1
fi

FOLD_MAP="nas/splits/MICHIGAN_WITHIN/fold_to_subject.txt"

for k in $(seq 1 "$N_SUBJECTS"); do
    SUBJECT=""
    if [[ -f "$FOLD_MAP" ]]; then
        SUBJECT=$(awk -F'\t' -v k="$k" '$1==k {print $2}' "$FOLD_MAP")
    fi
    echo "=== Fold $k/$N_SUBJECTS (subject: ${SUBJECT:-unknown}) ==="
    bash scripts/run_experiment.sh \
      --model neurostorm --dataset michigan_within --task task6 --mode finetune \
      --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPUS" \
      --load_model_path "$LOAD_MODEL_PATH" \
      -- --dataset_split_num "$k" --tpt_strategy "$TPT_STRATEGY" \
         --project_name "FIXED_mich_within_${SUBJECT}_fold-${k}" \
         "${EXTRA_ARGS[@]}"
done

echo ""
echo "Done. compare_conditions.py can treat michigan_within_fold-1..${N_SUBJECTS} as"
echo "the 'folds' of a single condition -- each one is a different SUBJECT rather"
echo "than a different partition of subjects, but the fold-aggregation (mean +/-"
echo "SEM across folds) works the same way and is the right way to read this,"
echo "since any single subject's 6-block test set is too small to trust alone."

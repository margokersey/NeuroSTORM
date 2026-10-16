#!/bin/bash
# run_michigan_reid_cv.sh
#
# Subject re-identification (fmri_reid) on the SAME XCP-D-denoised
# task-imagery_run-1 data and the SAME nas/splits/MICHIGAN/split_fixed_1..5.txt
# between-subject 5-fold CV splits as your winning imagery-classification
# condition (run_finetune_xcpd_cv.sh) -- same checkpoint, same batch
# size/epochs, only --task changes (task6 -> task7) and --project_name
# gets a "_reid_" tag so it doesn't collide with the task6 runs.
#
# What "reid" means here: instead of classifying WHICH TASK a window shows
# (tennis/navigation/squeeze), the model classifies WHICH SUBJECT it came
# from (26-way), using the exact same task-imagery windows. At eval time,
# main.py checkpoints on valid_reid_top1 (not valid_acc) -- an embedding-
# retrieval metric (see models/lightning_model.py's _evaluate_reid): does
# a held-out window's LEARNED FEATURE VECTOR land closest, by cosine
# similarity, to another window from the SAME subject, among all windows
# in that fold's val/test set?
#
# Why the between-subject splits are fine for this (not a bug): fold-to-
# fold, val/test subjects are completely disjoint from train subjects --
# the model never sees those specific people's data during training. But
# _evaluate_reid works purely on pooled FEATURES, not classification
# logits, so it doesn't need a subject to have been a trained class. That
# makes this an open-set test: does the model learn a fMRI "fingerprint"
# general enough to reidentify people it never classified during training?
# That's a stronger, more interesting claim than closed-set reid (where
# train/test subjects overlap) -- and it's exactly what reusing your
# existing 5-fold splits gives you for free, no new split files needed.
#
# Prerequisites:
#   1. patch_michigan_reid.py has been applied (adds --task_name fmri_reid
#      support to the MICHIGAN/MICHIGAN_FULL_BLOCK dataset classes) --
#      already done on this Mac copy; if you're on a different machine
#      (e.g. your Linux training box), copy datasets/data_module.py and
#      datasets/fmri_datasets.py over, or re-run the patch there.
#   2. scripts/configs/datasets/michigan_xcpd.yaml has the new task7 block
#      (task_name: fmri_reid) -- already added here; copy this yaml over
#      too if running on a different machine.
#   3. Same GENERAL_CKPT / XCP-D data / split files as the 5-fold CV you
#      already ran for task6 -- nothing else to set up.
#
# Run from the repo root:
#   bash run_michigan_reid_cv.sh
#
# Optional: --max_epochs / --batch_size / --gpus / --reid_gallery_size
# (limits how many samples the retrieval metric compares each query
# against per eval batch; default is "all", which is fine at this scale).

set -euo pipefail

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"
DATASET="michigan_xcpd_nosmooth_fullblock"
TASK="task4"
IMG_ROOT="/data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/img"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5
PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_tpt_strategy-none_reid"

MAX_EPOCHS=30
BATCH_SIZE=8
GPUS=0
REID_GALLERY_SIZE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --reid_gallery_size) REID_GALLERY_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=()
if [[ -n "$REID_GALLERY_SIZE" ]]; then
    EXTRA_ARGS+=(--reid_gallery_size "$REID_GALLERY_SIZE")
fi

# ------------------------------------------------------------------
# Preflight: same check as run_finetune_xcpd_cv.sh -- confirm every split
# file exists and every subject it names has a preprocessed data.pt under
# the XCP-D image_path. This task reuses those same 5 files as-is.
# ------------------------------------------------------------------
echo "Preflight: checking split-file subjects exist under $IMG_ROOT ..."
if [[ ! -d "$IMG_ROOT" ]]; then
    echo "Preflight FAILED: $IMG_ROOT does not exist."
    exit 1
fi
if ! compgen -G "$SPLIT_DIR"/split_fixed_[1-5].txt > /dev/null; then
    echo "Preflight FAILED: no split_fixed_1..5.txt files found under $SPLIT_DIR"
    exit 1
fi

missing=0
for i in $(seq 1 "$N_FOLDS"); do
    split_file="$SPLIT_DIR/split_fixed_${i}.txt"
    for subj in $(grep -vE '^(train|val|test)_subjects$' "$split_file"); do
        [[ -z "$subj" ]] && continue
        if [[ ! -f "$IMG_ROOT/$subj/data.pt" ]]; then
            echo "  MISSING: $subj (referenced in $split_file)"
            missing=$((missing + 1))
        fi
    done
done
if [[ "$missing" -gt 0 ]]; then
    echo "Preflight FAILED: $missing missing subject(s) -- fix before running folds."
    exit 1
fi
echo "Preflight OK: all split-file subjects found under $IMG_ROOT."
echo ""

# ------------------------------------------------------------------
# 5-fold reid finetune, same folds as task6's 5-fold CV.
# ------------------------------------------------------------------
for i in $(seq 1 "$N_FOLDS"); do
    echo "=== ReID fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
      --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
      --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPUS" \
      --load_model_path "$GENERAL_CKPT" \
      -- --dataset_split_num "$i" --tpt_strategy none \
         --project_name "${PROJECT_PREFIX}_fold-${i}" \
         "${EXTRA_ARGS[@]}"
done

echo ""
echo "All $N_FOLDS reid folds done. Checkpointing used valid_reid_top1 (see"
echo "nas/output/neurostorm/${PROJECT_PREFIX}_fold-*/ )."
echo ""
echo "Quick look at one fold's numbers (no code needed):"
echo "  tensorboard --logdir nas/output/neurostorm/${PROJECT_PREFIX}_fold-1"
echo "  -> look at valid_reid_top1 / valid_reid_mAP / test_reid_top1 / test_reid_mAP"
echo ""
echo "To aggregate all 5 folds with the existing tooling, val_checkpoint_monitor"
echo "and the metric names both need to point at reid_top1/reid_mAP instead of"
echo "acc/AUROC (this task has no LABEL_MAP classes, so skip the per-class/stats"
echo "sections, which assume 3-class imagery labels):"
echo "  python compare_conditions_v2.py \\"
echo "    --condition \"ReID (best model)|${PROJECT_PREFIX}_fold-*\" \\"
echo "    --val_checkpoint_monitor reid_top1 \\"
echo "    --test_metrics reid_top1,reid_mAP --val_metrics reid_top1,reid_mAP \\"
echo "    --skip_stats"

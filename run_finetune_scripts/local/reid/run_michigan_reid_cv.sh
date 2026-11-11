#!/bin/bash
# Subject re-identification (fmri_reid) on the SAME XCP-D-denoised
# task-imagery_run-1 data and the SAME nas/splits/MICHIGAN/split_fixed_1..5.txt
# between-subject 5-fold splits as the imagery-classification condition
# (local/5fold/run_finetune_xcpd_cv.sh): same checkpoint, batch size and
# epochs; only --task changes (task6 -> task4) and --project_name gets a
# "_reid" tag so it doesn't collide with the task6 runs.
#
# What "reid" means here: instead of classifying WHICH TASK a window shows
# (tennis/navigation/squeeze), the model classifies WHICH SUBJECT it came
# from (26-way) on the same windows. At eval time main.py checkpoints on
# valid_reid_top1 (not valid_acc) -- an embedding-retrieval metric (see
# _evaluate_reid in models/lightning_model.py): does a held-out window's
# learned feature vector land closest, by cosine similarity, to another
# window from the SAME subject among all windows in that fold's val/test set?
#
# Why between-subject splits are fine: val/test subjects are disjoint from
# train subjects, but _evaluate_reid works on pooled FEATURES, not
# classification logits, so a subject doesn't need to be a trained class.
# That makes this an open-set test -- does the model learn an fMRI
# "fingerprint" general enough to re-identify people it never trained on? --
# and the existing 5-fold splits give it for free.
#
# Prerequisites:
#   1. patch_michigan_reid.py has been applied (adds --task_name fmri_reid
#      support to the MICHIGAN/MICHIGAN_FULL_BLOCK dataset classes); on another
#      machine copy datasets/data_module.py and datasets/fmri_datasets.py or
#      re-run the patch.
#   2. scripts/configs/datasets/michigan_xcpd.yaml has the reid task block
#      (task_name: fmri_reid).
#   3. Same GENERAL_CKPT / XCP-D data / split files as the task6 5-fold CV.
#
# Run from the repo root:
#   bash run_finetune_scripts/local/reid/run_michigan_reid_cv.sh
#
# Options: --max_epochs / --batch_size / --gpus / --reid_gallery_size (limits
# how many samples each query is compared against per eval batch; default is
# "all", which is fine at this scale).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

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

# ── Preflight ────────────────────────────────────────────────────────────────
echo "Preflight: checking split-file subjects exist under $IMG_ROOT ..."
preflight_require_dir "$IMG_ROOT"
preflight_require_splits "$SPLIT_DIR" "split_fixed_[1-5].txt"
for i in $(seq 1 "$N_FOLDS"); do
    preflight_check_subjects "$SPLIT_DIR/split_fixed_${i}.txt" "$IMG_ROOT"
done
preflight_finish "all split-file subjects found under $IMG_ROOT."

# ── 5-fold reid fine-tune (same folds as task6's 5-fold CV) ──────────────────
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
echo "  python analysis/compare_conditions_v2.py \\"
echo "    --condition \"ReID (best model)|${PROJECT_PREFIX}_fold-*\" \\"
echo "    --val_checkpoint_monitor reid_top1 \\"
echo "    --test_metrics reid_top1,reid_mAP --val_metrics reid_top1,reid_mAP \\"
echo "    --skip_stats"

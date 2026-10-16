#!/bin/bash
# backfill_best_ckpt_test.sh
#
# Re-tests each fold of an already-completed (or still-training, run this
# after it finishes) run on its BEST-validation checkpoint instead of the
# last-epoch weights that main.py's automatic post-fit trainer.test() used
# to evaluate before patch_test_best_checkpoint.py. No retraining needed --
# this just reloads the checkpoint ModelCheckpoint already saved to disk and
# runs one more --mode test pass.
#
# Checkpoint location: main.py's TensorBoard-logger branch sets
#   dirpath = args.default_root_dir   (= nas/output/neurostorm/<project_name>)
#   logger = TensorBoardLogger(dirpath)
# i.e. ModelCheckpoint's dirpath is the PROJECT ROOT itself, not a
# lightning_logs/version_N subfolder -- TensorBoardLogger creates that
# version_N structure separately, on its own, for event files only. So
# checkpt-*.ckpt / last.ckpt live directly in
# nas/output/neurostorm/<project_name>/, one level above lightning_logs/.
#
# It writes into the SAME --project_name as the original training run, so
# PyTorch Lightning creates a new, later lightning_logs/version_N there for
# the test-only pass's event log (the checkpoint dir itself is untouched --
# testing doesn't write new checkpoints). compare_conditions.py /
# evaluate_cv_thorough.py already pick the LATEST version_N per project, so
# they'll automatically start reading this corrected result with no changes
# needed on their end.
#
# Usage:
#   bash "Claude outputs/backfill_best_ckpt_test.sh" \
#     --project_base mich_xcpd_nosmooth_tpt_strategy-none \
#     --dataset michigan_xcpd --task task6 --tpt_strategy none
#
# (project_base + "_fold-<i>" must exactly match the --project_name your
# training runs actually used -- check nas/output/neurostorm/ if unsure.
# --dataset/--task/--tpt_strategy must also match what that condition was
# trained with; --dataset_split_num is set automatically per fold.)
#
# Optional: --n_folds (default 5), --gpus (default 0), plus anything after
# a literal -- is passed through as extra args to every fold's run.

set -euo pipefail

START_FOLD_NUM=1 # Need to change to 101-126 for the LOOCV runs
END_FOLD_NUM=5
GPUS=0
EXTRA_ARGS=()
PROJECT_BASE=""
DATASET=""
TASK=""
TPT_STRATEGY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project_base) PROJECT_BASE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --tpt_strategy) TPT_STRATEGY="$2"; shift 2 ;;
        --start_fold) START_FOLD_NUM="$2"; shift 2 ;;
        --end_fold) END_FOLD_NUM="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$PROJECT_BASE" || -z "$DATASET" || -z "$TASK" || -z "$TPT_STRATEGY" ]]; then
    echo "Usage: bash backfill_best_ckpt_test.sh --project_base <name> --dataset <name> --task <name> --tpt_strategy <strategy> [--start_fold N] [--end_fold N] [--gpus N] [-- extra args...]"
    exit 1
fi

OUTPUT_ROOT="nas/output/neurostorm"

for i in $(seq "$START_FOLD_NUM" "$END_FOLD_NUM"); do
    PROJECT_NAME="${PROJECT_BASE}_fold-${i}"
    PROJECT_DIR="${OUTPUT_ROOT}/${PROJECT_NAME}"

    if [[ ! -d "$PROJECT_DIR" ]]; then
        echo "=== Fold $i: SKIPPED -- no directory at $PROJECT_DIR ==="
        continue
    fi

    # checkpt-*.ckpt lives directly in PROJECT_DIR (see header comment) --
    # NOT under lightning_logs/version_N/.
    mapfile -t CKPT_MATCHES < <(ls "${PROJECT_DIR}"/checkpt-*.ckpt 2>/dev/null || true)
    if [[ "${#CKPT_MATCHES[@]}" -eq 0 ]]; then
        echo "=== Fold $i: SKIPPED -- no checkpt-*.ckpt found directly in $PROJECT_DIR (only last.ckpt?) ==="
        continue
    fi
    if [[ "${#CKPT_MATCHES[@]}" -gt 1 ]]; then
        echo "  note: ${#CKPT_MATCHES[@]} checkpt-*.ckpt files found in $PROJECT_DIR (expected 1 with save_top_k=1) -- using the first:"
        printf '    %s\n' "${CKPT_MATCHES[@]}"
    fi
    CKPT="${CKPT_MATCHES[0]}"

    echo "=== Fold $i/$END_FOLD_NUM: testing $CKPT ==="
    bash scripts/run_experiment.sh \
      --model neurostorm --dataset "$DATASET" --task "$TASK" --mode test \
      --gpus "$GPUS" \
      -- --dataset_split_num "$i" --tpt_strategy "$TPT_STRATEGY" \
         --test_ckpt_path "$CKPT" \
         --project_name "$PROJECT_NAME" \
         "${EXTRA_ARGS[@]}"
done

echo ""
echo "Done. Re-run compare_conditions.py / evaluate_cv_thorough.py -- they pick the"
echo "LATEST lightning_logs/version_N per project automatically, so they'll now read"
echo "these best-checkpoint results instead of the original last-epoch ones."

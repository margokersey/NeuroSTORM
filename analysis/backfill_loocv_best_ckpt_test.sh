#!/bin/bash
# backfill_loocv_best_ckpt_test.sh
#
# Re-tests all LOOCV folds (real and permutation) using their BEST validation
# checkpoint instead of the last-epoch weights. This ensures our permutation
# test uses the best-performing model for each condition, not an underfitted
# or overfitted final epoch.
#
# NOTE: No retraining needed -- just reloads checkpoints already saved to disk
# and runs one more --mode test pass, which outputs to a new lightning_logs/
# version_N folder (the checkpoint dir itself is untouched).
#
# Usage:
#   bash analysis/backfill_loocv_best_ckpt_test.sh [--start_fold 101] [--end_fold 126] [--n_permutations 3]
#
# Environment:
#   REPO_DIR: defaults to $HOME/code/NeuroSTORM
#   DATA_ROOT: needed if not already in environment; defaults from Great Lakes
#   CONDA_PREFIX: assumes 'neurostorm' conda env is available
#
# The script auto-discovers which fold/permutation directories exist and tests
# them in order (real models first, then perm1, perm2, perm3 if available).

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

REPO_DIR="${REPO_DIR:-$HOME/code/NeuroSTORM}"
DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"

START_FOLD=101
END_FOLD=126
N_PERMUTATIONS=3
MAX_PARALLEL=2       # Limit concurrent GPU jobs (adjust for your partition)
GPUS=0

# Where LOOCV outputs live
OUTPUT_ROOT="nas/output/neurostorm"

# ──────────────────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_fold) START_FOLD="$2"; shift 2 ;;
        --end_fold) END_FOLD="$2"; shift 2 ;;
        --n_permutations) N_PERMUTATIONS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────
# Main loop: for each fold, find best checkpoint and retest
# ──────────────────────────────────────────────────────────────────────────

echo "=========================================================================="
echo "LOOCV Best Checkpoint Backfill Test"
echo "=========================================================================="
echo "Folds: $START_FOLD - $END_FOLD"
echo "Permutations: $N_PERMUTATIONS (will test real + perm1..perm$N_PERMUTATIONS if available)"
echo "Output root: $OUTPUT_ROOT"
echo "Max concurrent jobs: $MAX_PARALLEL"
echo ""

# Track active background jobs
declare -a ACTIVE_PIDS=()

for fold_num in $(seq "$START_FOLD" "$END_FOLD"); do
    # Test real model first (no suffix)
    for perm_suffix in "" "_perm1" "_perm2" "_perm3"; do
        # Skip permutation tests if N_PERMUTATIONS is low
        if [[ -n "$perm_suffix" ]]; then
            perm_idx="${perm_suffix##_perm}"
            if [[ $perm_idx -gt $N_PERMUTATIONS ]]; then
                continue
            fi
        fi

        PROJECT_NAME="FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-${fold_num}${perm_suffix}"
        PROJECT_DIR="${OUTPUT_ROOT}/${PROJECT_NAME}"

        # Check if directory exists
        if [[ ! -d "$PROJECT_DIR" ]]; then
            if [[ -z "$perm_suffix" ]]; then
                echo "WARNING: Fold $fold_num: no directory at $PROJECT_DIR -- skipping"
            fi
            continue
        fi

        # Find the best checkpoint (checkpt-*.ckpt with highest validation accuracy)
        # ModelCheckpoint saves these as checkpt-epoch=N-valid_acc=X.ckpt
        CKPT_FILES=( $(ls "${PROJECT_DIR}"/checkpt-*.ckpt 2>/dev/null || true) )
        if [[ ${#CKPT_FILES[@]} -eq 0 ]]; then
            echo "WARNING: $PROJECT_NAME: no checkpt-*.ckpt found (only last.ckpt?) -- skipping"
            continue
        fi

        # Take the first (and should be only) checkpoint from ModelCheckpoint with save_top_k=1
        BEST_CKPT="${CKPT_FILES[0]}"
        if [[ ${#CKPT_FILES[@]} -gt 1 ]]; then
            echo "INFO: $PROJECT_NAME: ${#CKPT_FILES[@]} checkpoint(s) found, using first: $(basename "$BEST_CKPT")"
        fi

        # Wait if we've hit max parallel jobs
        while [[ ${#ACTIVE_PIDS[@]} -ge $MAX_PARALLEL ]]; do
            # Check which pids are still alive
            NEW_PIDS=()
            for pid in "${ACTIVE_PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            ACTIVE_PIDS=("${NEW_PIDS[@]}")
            
            if [[ ${#ACTIVE_PIDS[@]} -ge $MAX_PARALLEL ]]; then
                sleep 5
            fi
        done

        # Launch test job in background
        echo "Queueing: $PROJECT_NAME"
        (
            cd "$REPO_DIR"
            # Use mamba/conda run to execute in the neurostorm environment
            # This avoids activation issues in subshells
            if command -v mamba &> /dev/null; then
                mamba run -n neurostorm bash scripts/run_experiment.sh \
                    --model neurostorm \
                    --dataset michigan_xcpd_nosmooth_fullblock \
                    --task task6 \
                    --mode test \
                    --gpus "$GPUS" \
                    -- --dataset_split_num "$fold_num" \
                       --tpt_strategy none \
                       --project_name "$PROJECT_NAME" \
                       --test_ckpt_path "$BEST_CKPT" \
                    > /tmp/backfill_${PROJECT_NAME}.log 2>&1
            elif command -v conda &> /dev/null; then
                conda run -n neurostorm bash scripts/run_experiment.sh \
                    --model neurostorm \
                    --dataset michigan_xcpd_nosmooth_fullblock \
                    --task task6 \
                    --mode test \
                    --gpus "$GPUS" \
                    -- --dataset_split_num "$fold_num" \
                       --tpt_strategy none \
                       --project_name "$PROJECT_NAME" \
                       --test_ckpt_path "$BEST_CKPT" \
                    > /tmp/backfill_${PROJECT_NAME}.log 2>&1
            else
                echo "ERROR: Neither mamba nor conda found in PATH"
                exit 1
            fi
        ) &

        ACTIVE_PIDS+=("$!")
    done
done

# Wait for all background jobs to finish
echo ""
echo "Waiting for all test jobs to complete..."
wait_count=0
failed_count=0
for pid in "${ACTIVE_PIDS[@]}"; do
    if wait "$pid"; then
        wait_count=$((wait_count + 1))
    else
        failed_count=$((failed_count + 1))
        echo "WARNING: Job PID $pid exited with non-zero code"
    fi
done

echo ""
echo "=========================================================================="
echo "Backfill complete. Tested $wait_count folds/permutations using best checkpoints."
if [[ $failed_count -gt 0 ]]; then
    echo "WARNING: $failed_count jobs failed -- check logs above"
fi
echo ""
echo "Next: Re-run the analysis script to pick up the new best-checkpoint"
echo "test results from the latest lightning_logs/version_N per project:"
echo ""
echo "  python scripts/analyze_loocv_permutation_results.py --n_permutations $N_PERMUTATIONS"
echo "=========================================================================="


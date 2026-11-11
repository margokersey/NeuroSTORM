#!/bin/bash
# LOOCV Proper Permutation Testing: Retrain Models on Randomized Labels
#
# This script implements proper permutation testing by:
# 1. For each fold (101-126), train N permutations with randomized class labels
# 2. Each permutation uses the SAME fold split but with shuffled training labels
# 3. Test on the original test set (with true labels)
# 4. Compare: real model accuracy vs. permutation model accuracies
#
# This tests the hypothesis: "Does the model learn real task structure,
# or would it perform equally well on randomized (meaningless) labels?"
#
# Cost: 26 folds × N permutations × ~12 hours per fold = substantial compute
# Recommendation: Start with N=2-3 permutations per fold for quick results,
#                 then increase to N=5-10 for publication-quality results.
#
# Usage:
#   bash scripts/loocv_permutation_testing.sh --n_permutations 3 --start_index 101 --n_folds 26

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
N_PERMUTATIONS=3          # Number of randomized label permutations per fold
START_INDEX=101
N_FOLDS=26
MAX_EPOCHS=50             # Same as real models for fair comparison
BATCH_SIZE=16
GPU_IDS="0"
SEED=1234

# Directories
DATA_ROOT="${DATA_ROOT:-.}"

# Data & model
DATASET="MICHIGAN"
TASK="imagery_task_classification"
PROJECT_PREFIX="xcpd_nosmooth_general"
GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_permutations) N_PERMUTATIONS="$2"; shift 2 ;;
        --start_index) START_INDEX="$2"; shift 2 ;;
        --n_folds) N_FOLDS="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

END_INDEX=$((START_INDEX + N_FOLDS - 1))

# ─── Info ─────────────────────────────────────────────────────────────────────
echo "LOOCV Permutation Testing Configuration:"
echo "  Folds: $START_INDEX-$END_INDEX ($N_FOLDS total)"
echo "  Permutations per fold: $N_PERMUTATIONS"
echo "  Total retrainings: $((N_FOLDS * N_PERMUTATIONS))"
echo "  Max epochs per model: $MAX_EPOCHS"
echo "  Seed base: $SEED"
echo ""
echo "This will train $((N_FOLDS * N_PERMUTATIONS)) permutation models total"
echo "Expected wall-clock time: ~$((N_FOLDS * MAX_EPOCHS / 60)) hours per permutation (parallel across folds)"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] Would train permutation models for folds $START_INDEX-$END_INDEX"
    echo "To execute, remove --dry_run flag"
    exit 0
fi

# ─── Train Models on Permuted Labels ──────────────────────────────────────────
echo "Training models with permuted training labels..."
echo ""

for fold_idx in $(seq "$START_INDEX" "$END_INDEX"); do
    for perm_idx in $(seq 1 "$N_PERMUTATIONS"); do
        echo "=== Fold $fold_idx, Permutation $perm_idx / $N_PERMUTATIONS ==="
        
        # Use original split file, but with --permute_labels flag
        # Each permutation gets a different seed for label shuffling
        perm_seed=$((SEED + perm_idx))
        
        bash scripts/run_experiment.sh \
            --model neurostorm --dataset "$DATASET" --task "$TASK" --mode finetune \
            --max_epochs "$MAX_EPOCHS" --batch_size "$BATCH_SIZE" --gpus "$GPU_IDS" \
            --load_model_path "$GENERAL_CKPT" \
            -- --dataset_split_num "$fold_idx" --tpt_strategy none \
                 --permute_labels --permutation_seed "$perm_seed" \
                 --project_name "${PROJECT_PREFIX}_fold-${fold_idx}_perm${perm_idx}"
        
        echo ""
    done
done

echo "✓ All permutation models trained"
echo ""
echo "Next: Run analyze_loocv_permutation_results.py to compare real vs. random"

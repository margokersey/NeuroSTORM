#!/bin/bash
# 5-fold CV fine-tune of NeuroSTORM's GENERAL checkpoint (MAE-pretrained on
# 5 datasets, never seen a downstream label -- pt_neurostorm_mae_5ds.ckpt)
# on the new XCP-D-denoised task-imagery_run-1 data, using --tpt_strategy
# none (full fine-tune) to match the general-vs-sex-specific-checkpoint
# comparison already run on the fMRIPrep version of this data.
#
# Prerequisites:
#   1. scripts/configs/datasets/michigan_xcpd*.yaml are in place, and any
#      paths inside them point at the Great Lakes copy of the data.
#   2. The preprocessed data (img/sub-*/data.pt + metadata/) has been copied
#      to DATA_ROOT below.
#   3. GENERAL_CKPT below exists.
#   4. nas/splits/MICHIGAN/split_fixed_1.txt .. split_fixed_5.txt exist.
#
# Run from the repo root:
#   bash run_finetune_xcpd_cv.sh              # all folds, one after another
#   FOLDS=3 bash run_finetune_xcpd_cv.sh      # just fold 3
#   FOLDS="2 4" bash run_finetune_xcpd_cv.sh  # folds 2 and 4
# On Great Lakes, finetune_xcpd_cv.sbatch sets FOLDS per array task.

set -euo pipefail

GENERAL_CKPT="pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

# Where the preprocessed data lives. Overridable from the job file.
DATA_ROOT="${DATA_ROOT:-/scratch/huangzu_root/huangzu0/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"

# dataset for the original (no-smoothing) XCP-D outputs
DATASET="michigan_xcpd_nosmooth_fullblock"
IMG_ROOT="$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/img"

TASK="task6"
SPLIT_DIR="nas/splits/MICHIGAN"
N_FOLDS=5

# Which folds to run: all by default, or whatever FOLDS is set to.
FOLDS="${FOLDS:-$(seq 1 "$N_FOLDS")}"

# Number of GPUs this job can see (set by the job file; 1 if run by hand).
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))   # "0" for 1 GPU, "0,1" for 2, ...

# ------------------------------------------------------------------
# Preflight: confirm every subject named in each split file actually has a
# preprocessed data.pt under the XCP-D image paths. A subject that failed
# preprocessing would simply be MISSING here rather than erroring --
# better to catch that now than partway through fold 3 of 5.
# ------------------------------------------------------------------
echo "Preflight: checking split-file subjects exist under $IMG_ROOT..."
if [[ ! -d "$IMG_ROOT" ]]; then
    echo "Preflight FAILED: $IMG_ROOT does not exist -- has the data been copied/preprocessed?"
    exit 1
fi
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
echo "Preflight OK: all split-file subjects found."
echo ""

# ------------------------------------------------------------------
# CV finetune. Note the braces around ${i} in --project_name below:
# without them, "$i_general" would mean the unset variable "i_general",
# and every fold would silently get the same project name.
# ------------------------------------------------------------------

# best model with optimally selected window size (28 frames total, 2 TRs trimmed from start)
for i in $FOLDS; do
    strategy="none"
    ds_name="$DATASET"
    ds_label="nosmooth"
    echo "=== Fold $i/$N_FOLDS ==="
    bash scripts/run_experiment.sh \
        --model neurostorm --dataset "$ds_name" --task "$TASK" --mode finetune \
        --max_epochs 50 --batch_size 8 --gpus "$GPU_IDS" \
        --load_model_path "$GENERAL_CKPT" \
        -- --dataset_split_num "$i" --tpt_strategy "$strategy" \
            --project_name "FIXED_mich_xcpd_${ds_label}_tpt_strategy-${strategy}_fold-${i}"
done

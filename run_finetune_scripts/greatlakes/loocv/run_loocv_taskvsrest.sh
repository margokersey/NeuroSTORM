#!/bin/bash
# Stage A of the two-stage decoding test: Task vs. No Task, leave-one-subject-out.
#
#   Task    = 30 imagery-run-1 blocks      (label 1)
#   No Task = 15 mock blocks from rest-1 + 15 from rest-2, at the onsets of the
#             first 15 imagery blocks        (label 0)
# Same 26 folds (nas/splits/MICHIGAN/split_fixed_101..126), model, checkpoint
# and hyperparameters as the 3-way LOOCV in run_loocv_xcpd.sh -- this script
# only swaps the dataset/task/project name and forwards all options (--start_index,
# --n_folds, --max_epochs, --permute_labels, --project_name_suffix, --dry_run, ...).
#
# Stage B (tennis/navigation/squeeze) is NOT retrained: the existing 3-way LOOCV
# fold predictions are reused. Combine with analysis/two_stage_decoding.py.
#
# sub-29 has no rest-2 (and no task-4), so it contributes 15 mock blocks instead of
# 30 when it is in a train/val set; its own test fold (see two_stage_decoding.py
# --exclude_subjects) is left out of the two-stage evaluation.
#
# Run from the repo root:
#   bash run_finetune_scripts/greatlakes/loocv/run_loocv_taskvsrest.sh --dry_run
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"
export DATA_ROOT
export LOOCV_DATASET="michigan_xcpd_nosmooth_taskvsrest"
export LOOCV_TASK="task7"
export LOOCV_PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_taskvsrest_loocv"
export LOOCV_EXTRA_DIRS="$DATA_ROOT/FIXED_xcp_nosmooth_task-rest_run-1/img $DATA_ROOT/FIXED_xcp_nosmooth_task-rest_run-2/img"

exec bash "$SCRIPT_DIR/run_loocv_xcpd.sh" "$@"

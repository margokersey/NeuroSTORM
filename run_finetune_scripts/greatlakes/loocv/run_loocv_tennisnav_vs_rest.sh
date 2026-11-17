#!/bin/bash
# Stage A of the tennis/navigation two-stage test: Task vs. No Task, leave-one-subject-out.
#
#   Task    = 20 imagery-run-1 blocks (10 tennis + 10 navigation; squeeze dropped)  (label 1)
#   No Task = 10 mock blocks from rest-1 + 10 from rest-2, at the onsets of the
#             first 10 tennis/navigation blocks                                      (label 0)
# Same 26 folds, model, checkpoint and hyperparameters as run_loocv_taskvsrest.sh;
# only the dataset/task/project name change. Forwards all options (--start_index,
# --n_folds, --max_epochs, --permute_labels, --project_name_suffix, --dry_run, ...).
#
# Stage B (tennis vs navigation) is the existing run_loocv_tennisnav.sh model; see
# analysis/predict_stage_b_on_stage_a_blocks.py --stage_b tennisnav.
#
#   bash run_finetune_scripts/greatlakes/loocv/run_loocv_tennisnav_vs_rest.sh --dry_run
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"
export DATA_ROOT
export LOOCV_DATASET="michigan_xcpd_nosmooth_tennisnav_vs_rest"
export LOOCV_TASK="task9"
export LOOCV_PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_tennisnav_vs_rest_loocv"
export LOOCV_EXTRA_DIRS="$DATA_ROOT/FIXED_xcp_nosmooth_task-rest_run-1/img $DATA_ROOT/FIXED_xcp_nosmooth_task-rest_run-2/img"

exec bash "$SCRIPT_DIR/run_loocv_xcpd.sh" "$@"

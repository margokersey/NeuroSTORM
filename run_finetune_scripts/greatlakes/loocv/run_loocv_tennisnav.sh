#!/bin/bash
# 2-class (tennis vs navigation) LOOCV: the 3-way run_loocv_xcpd.sh with squeeze
# blocks dropped. Same 26 folds (split_fixed_101..126), model, checkpoint, data and
# hyperparameters; only the dataset/task/project name change. Forwards all options
# (--start_index, --n_folds, --permute_labels, --project_name_suffix, --dry_run, ...).
#
#   bash run_finetune_scripts/greatlakes/loocv/run_loocv_tennisnav.sh --dry_run
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"
export LOOCV_DATASET="michigan_xcpd_nosmooth_tennisnav"
export LOOCV_TASK="task8"
TPT_STRATEGY="${LOOCV_TPT_STRATEGY:-none}"
export LOOCV_PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_tennisnav_tpt_strategy-${TPT_STRATEGY}_loocv"

exec bash "$SCRIPT_DIR/run_loocv_xcpd.sh" "$@"

#!/bin/bash
# 26-fold LOOCV, 3-way (tennis / navigation / squeeze), 28-TR window starting at block onset
# (shift 0). Same as run_loocv_xcpd.sh except the dataset config (sequence_length 28 instead of
# 37) and the project name. Options are passed through (--start_index, --n_folds, --dry_run, ...).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM}"
export LOOCV_DATASET="michigan_xcpd_nosmooth_fullblock_28tr"
export LOOCV_TASK="task6"
TPT_STRATEGY="${LOOCV_TPT_STRATEGY:-none}"
export LOOCV_PROJECT_PREFIX="FIXED_mich_xcpd_nosmooth_win28tr_tpt_strategy-${TPT_STRATEGY}_loocv"

exec bash "$SCRIPT_DIR/run_loocv_xcpd.sh" "$@"

#!/bin/bash
# LOOCV with --tpt_strategy prompt_ln (prompt tokens + LayerNorms tuned, rest of
# the backbone frozen). Identical to run_loocv_xcpd.sh (end-to-end, strategy none)
# in data, splits, checkpoint and hyperparameters, so the two are directly
# comparable fold-by-fold (same held-out subject per fold 101-126).
# Outputs: FIXED_mich_xcpd_nosmooth_tpt_strategy-prompt_ln_loocv_fold-<N>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOCV_TPT_STRATEGY="prompt_ln"
exec bash "$SCRIPT_DIR/run_loocv_xcpd.sh" "$@"

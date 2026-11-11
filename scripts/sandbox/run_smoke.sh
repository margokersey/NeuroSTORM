#!/bin/bash
# Local (no-GPU) smoke test: generate a tiny synthetic Michigan dataset and run
# the REAL main.py fine-tune path (data module -> NeuroSTORM -> Lightning
# train/val/test) on CPU for a couple of epochs.
#
# It catches: import errors, arg-parser / yaml wiring mistakes, data-loading and
# split bugs, shape errors, loss/metric/callback crashes. It does NOT tell you
# anything about accuracy, speed, memory, or CUDA/mamba-kernel behaviour.
#
# Usage (from repo root, with the neurostorm conda env active):
#   bash scripts/sandbox/run_smoke.sh                  # default tiny run
#   bash scripts/sandbox/run_smoke.sh -- --max_epochs 3 --task_name imagery_task_classification
#   SANDBOX_OUT=output/sandbox2 PYTHON=/path/to/python bash scripts/sandbox/run_smoke.sh
#
# Anything after `--` is appended to the main.py command (later flags win), so
# you can try out a new flag/option locally before submitting to Great Lakes.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT="${SANDBOX_OUT:-$REPO/output/sandbox}"
[[ "$OUT" = /* ]] || OUT="$REPO/$OUT"
EXTRA=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA=("$@"); fi

# Refuse to run on the cluster login node: this is the laptop path.
if command -v sbatch >/dev/null 2>&1 && [[ -z "${SANDBOX_ALLOW_CLUSTER:-}" ]]; then
    echo "sbatch found -> looks like the cluster. This sandbox is for the laptop;"
    echo "submit real jobs with sbatch. (Set SANDBOX_ALLOW_CLUSTER=1 to override.)"
    exit 1
fi

# Use the real mamba_ssm if it happens to be installed, else the pure-PyTorch shim.
if "$PYTHON" -c "import mamba_ssm" >/dev/null 2>&1; then
    SHIM=""
    echo "[sandbox] real mamba_ssm found; not using shim"
else
    SHIM="$REPO/scripts/sandbox/shims:"
    echo "[sandbox] mamba_ssm not installed -> using pure-PyTorch shim (slow, smoke-test only)"
fi

"$PYTHON" "$REPO/scripts/sandbox/make_synthetic_michigan.py" --out "$OUT"

# Fall back to CPU for ops MPS lacks; harmless if MPS isn't used.
export NEUROSTORM_ENV=local   # laptop: skip CUDA-only profiling hooks (utils/runtime_env.py)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="${SHIM}${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="ignore"

# Run from $OUT so main.py's relative ./nas/splits and nas/output stay inside the
# (gitignored) sandbox directory instead of touching the repo's real nas/.
cd "$OUT"
set -x
"$PYTHON" "$REPO/main.py" \
    --gpu_ids 0 --num_nodes 1 --loggername tensorboard \
    --dataset_name MICHIGAN_FULL_BLOCK --image_path "$OUT/dataset" \
    --dataset_split_num 1 --limit_training_samples 1.0 \
    --downstream_task_id 6 --downstream_task_type classification \
    --task_name imagery_task_classification --num_classes 3 \
    --model neurostorm --embed_dim 12 --depths 1 1 --num_heads 2 4 --c_multiplier 2 \
    --last_layer_full_MSA false --clf_head_version v1 \
    --sequence_length 6 --img_size 24 24 24 6 \
    --first_window_size 2 2 2 2 --window_size 2 2 2 2 \
    --batch_size 4 --eval_batch_size 3 --num_workers 0 \
    --max_epochs 2 --learning_rate 0.0005 --weight_decay 0.05 --attn_drop_rate 0.1 \
    --stride_between_seq 1 --seed 1 --strategy auto \
    --project_name sandbox_smoke \
    ${EXTRA[@]+"${EXTRA[@]}"}
set +x
echo "[sandbox] OK -- full train/val/test pipeline ran. Outputs: $OUT/nas/output/neurostorm/sandbox_smoke"

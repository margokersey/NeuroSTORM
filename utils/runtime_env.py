"""Which machine are we on? Controlled by the NEUROSTORM_ENV environment variable.

  NEUROSTORM_ENV=local       laptop sandbox (no NVIDIA GPU): skip CUDA-only profiling hooks.
                             Set by scripts/sandbox/run_smoke.sh and run_greatlakes_checks.sh.
  NEUROSTORM_ENV=greatlakes  cluster. Exported by set_env.sh (sourced by every sbatch_* job file).
  unset / anything else      treated like the cluster: original, unguarded behaviour.

Only the exact value "local" changes behaviour, so a missing variable on the cluster can never
switch off anything the GPU runs rely on.
"""
import os


def is_local() -> bool:
    return os.environ.get("NEUROSTORM_ENV", "").strip().lower() == "local"

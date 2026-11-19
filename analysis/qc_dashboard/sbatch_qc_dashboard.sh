#!/bin/bash
# =============================================================================
# Scan (incrementally) + serve the QC dashboard on a Great Lakes compute node.
# The server binds to 127.0.0.1 ONLY and requires a random access token; you
# reach it from your laptop through an SSH tunnel (the command is printed in the log).
#
#   NAME=michigan_d1 \
#   FMRIPREP=/nfs/turbo/.../fmriprep \
#   XCPD=/nfs/turbo/.../xcp_d \
#   DATAPT=/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1 \
#   sbatch analysis/qc_dashboard/sbatch_qc_dashboard.sh
#
# Optional env: DEEP=1 (also load BOLD NIfTIs: tSNR/DVARS; slower, more RAM), FD_THR=0.5,
#   SKIP_SCAN=1 (serve existing cache only), FORCE=1 (rescan everything),
#   DATAPT_SUBJECT_REGEX='...', CACHE_ROOT=..., PORT=..., QC_ENV=qc_dashboard (conda env name)
# Any of FMRIPREP / XCPD / DATAPT may be left unset to skip that stage.
# Stop the dashboard with `scancel <jobid>` (it also ends at the time limit).
# =============================================================================
#SBATCH --account=huangzu99
#SBATCH --job-name=qc_dash
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/qc_dashboard_%j.log

set -euo pipefail
umask 077   # log, token and cache readable by you only

REPO_DIR="${REPO_DIR:-$HOME/code/NeuroSTORM}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_DIR/output/qc_dashboard}"
NAME="${NAME:?set NAME=<dataset label>}"

module load mamba
set +u
eval "$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook')"
conda activate "${QC_ENV:-qc_dashboard}"
set -u

python - <<'PY' || { echo "Missing packages in env ${QC_ENV:-qc_dashboard}. On the login node: module load mamba && mamba env create -f $REPO_DIR/analysis/qc_dashboard/environment.yml"; exit 1; }
import streamlit, plotly, torch, nibabel, scipy
PY

cd "$REPO_DIR"
if [ "${SKIP_SCAN:-0}" != "1" ]; then
  ARGS=(--name "$NAME" --cache_root "$CACHE_ROOT" --workers "${SLURM_CPUS_PER_TASK:-4}" --fd_thr "${FD_THR:-0.5}")
  [ -n "${FMRIPREP:-}" ] && ARGS+=(--fmriprep "$FMRIPREP")
  [ -n "${XCPD:-}" ]     && ARGS+=(--xcpd "$XCPD")
  [ -n "${DATAPT:-}" ]   && ARGS+=(--datapt "$DATAPT")
  [ "${DEEP:-0}" = "1" ] && ARGS+=(--deep)
  [ "${FORCE:-0}" = "1" ] && ARGS+=(--force)
  [ -n "${DATAPT_SUBJECT_REGEX:-}" ] && ARGS+=(--datapt_subject_regex "$DATAPT_SUBJECT_REGEX")
  python analysis/qc_dashboard/scan.py "${ARGS[@]}"
fi

PORT="${PORT:-$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')}"
export QC_DASH_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')"
NODE="$(hostname -s)"
cat <<MSG

================ QC dashboard is up (localhost-only on $NODE:$PORT) ================
1) On your laptop, open a tunnel (leave it running):
   ssh -N -J $USER@greatlakes.arc-ts.umich.edu -L $PORT:127.0.0.1:$PORT $USER@$NODE
2) Open in your laptop browser:
   http://localhost:$PORT/?token=$QC_DASH_TOKEN
Stop with: scancel $SLURM_JOB_ID
====================================================================================

MSG

exec streamlit run analysis/qc_dashboard/app.py \
  --server.address 127.0.0.1 --server.port "$PORT" --server.headless true \
  --server.fileWatcherType none --browser.gatherUsageStats false \
  -- --cache_root "$CACHE_ROOT"

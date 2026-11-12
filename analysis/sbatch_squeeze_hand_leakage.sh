#!/bin/bash
# =============================================================================
# Squeeze-neighbour leakage plot (left hand-motor parcels), high-pass vs no-high-pass.
# Single CPU job: reads each subject's XCP-D denoised NIfTI (read-only) and writes
# output/cache/*.npz + output/squeeze_hand_leakage.png inside the repo.
#
# One-time setup:  mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit:          sbatch analysis/sbatch_squeeze_hand_leakage.sh
# Watch:           squeue -u mkersey   (or)   tail -f /scratch/huangzu_root/huangzu0/mkersey/logs/squeeze_leak_<jobid>.log
# Override roots:  sbatch --export=ALL,NOHP_DIR=<path> analysis/sbatch_squeeze_hand_leakage.sh
# =============================================================================
#SBATCH --account=huangzu99
#SBATCH --job-name=squeeze_leak
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/squeeze_leak_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

REPO_DIR="$HOME/code/NeuroSTORM"
BASE="/nfs/turbo/umms-huanglab/pipeline/postproc/fmriprep-25.2.5_xcp_d-26.1.1/Michigan_Dataset_1"
HP_DIR="${HP_DIR:-$BASE/noGSR_highpass_fd-0.8}"
NOHP_DIR="${NOHP_DIR:-$BASE/noGSR_bp_0-999_fd-999}"
TIMING="${TIMING:-/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv}"

module load mamba
set +u
eval "$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook')"
conda activate general
set -u

cd "$REPO_DIR"
python -u analysis/plot_squeeze_hand_leakage.py \
    --root "highpass=$HP_DIR" \
    --root "nohp=$NOHP_DIR" \
    --timing "$TIMING"

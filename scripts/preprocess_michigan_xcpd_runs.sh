#!/bin/bash
# =============================================================================
# Build data.pt volumes from XCP-D denoised BOLD for Michigan Dataset 1.
# One array task per (task, run); each writes
#   $OUT_ROOT/FIXED_xcp_nosmooth_task-<task>_run-<N>/img/sub-XX/data.pt
#
# Submit all five:  sbatch scripts/preprocess_michigan_xcpd_runs.sh
# Submit one:       sbatch --array=0 scripts/preprocess_michigan_xcpd_runs.sh
# =============================================================================
#SBATCH --account=huangzu99
#SBATCH --job-name=mich_prep
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=standard
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mich_prep_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

REPO_DIR="$HOME/code/NeuroSTORM"
LOAD_ROOT="/nfs/turbo/umms-huanglab/pipeline/postproc/fmriprep-25.2.5_xcp_d-26.1.1/Michigan_Dataset_1/noGSR_highpass_fd-0.8_nosmooth"
OUT_ROOT="/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM"

COMBOS=("imagery 2") # "imagery 3" "imagery 4" "rest 1" "rest 2")
read -r TASK RUN <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
SAVE_ROOT="$OUT_ROOT/FIXED_xcp_nosmooth_task-${TASK}_run-${RUN}"

module load mamba
set +u
eval "$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook')"
conda activate neurostorm
set -u

cd "$REPO_DIR/datasets"
echo "task=$TASK run=$RUN -> $SAVE_ROOT"
python preprocessing_volume.py \
    --dataset_name michigan_xcpd \
    --task "$TASK" --run "$RUN" \
    --load_root "$LOAD_ROOT" \
    --save_root "$SAVE_ROOT" \
    --scaling_method z-norm \
    --skip_existing \
    --num_processes 4 --prefetch_chunk_size 1

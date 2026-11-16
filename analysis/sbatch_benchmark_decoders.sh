#!/bin/bash
# =============================================================================
# Benchmark GLM / logistic regression / SVM / LDA / random forest against NeuroSTORM on the
# 3-way imagery task across propofol states (run 1 awake, pre-LOR, LOR, ROR, recovery).
# CPU only. See analysis/benchmark_decoders.py.
#
# One-time setup:  mkdir -p /scratch/huangzu_root/huangzu0/mkersey/logs
# Submit:          sbatch analysis/sbatch_benchmark_decoders.sh
# Re-evaluate only (features cached):  STAGE=eval sbatch analysis/sbatch_benchmark_decoders.sh
# =============================================================================
#SBATCH --account=huangzu99
#SBATCH --job-name=bench_dec
#SBATCH --chdir=/scratch/huangzu_root/huangzu0/mkersey
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench_dec_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkersey@umich.edu

set -euo pipefail

REPO_DIR="$HOME/code/NeuroSTORM"
export DATA_ROOT="/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM"
OUT_ROOT="/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output"

module load mamba
set +u
__conda_setup="$('/sw/pkgs/arc/mamba/py3.12/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate neurostorm
cd "$REPO_DIR"
source ./set_env.sh
set -u

export OMP_NUM_THREADS=1   # parallelism is across subjects/folds (joblib), not inside BLAS

LOR_CSV="${LOR_CSV:-$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-2/metadata/LOR_ROR_timing.csv}"
[ -f "$LOR_CSV" ] || { echo "LOR/ROR CSV not found: $LOR_CSV (set LOR_CSV)"; exit 1; }

# sub-29 run 3 is skipped automatically (frame count != timing file). Add --exclude_subjects here if needed.
python analysis/benchmark_decoders.py --stage "${STAGE:-all}" \
    --data_root "$DATA_ROOT" --lor_csv "$LOR_CSV" \
    --cache_dir "$REPO_DIR/output/benchmark_decoders/cache" \
    --out_dir "$REPO_DIR/output/benchmark_decoders" \
    --pred_glob "$OUT_ROOT/FIXED_mich_xcpd_nosmooth_lor_task_decoding_fold-*/lor_predictions_fold_*.json" \
    --output_root "$OUT_ROOT"

echo "Finished at $(date)"

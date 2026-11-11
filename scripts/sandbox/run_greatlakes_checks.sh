#!/bin/bash
# Exercise run_finetune_scripts/greatlakes/* end-to-end on a laptop, UNMODIFIED.
#
# What runs for real: run_finetune_xcpd_cv.sh, run_loocv_xcpd.sh, lib/common.sh,
# scripts/run_experiment.sh + the dataset/model YAMLs, main.py, the data module,
# fine-tune loop, checkpointing, test-prediction callback, and the sbatch_* job
# files' own bash (array-id -> FOLDS wiring, env, arg passing).
#
# What is substituted (see $WORK/bin):
#   python main.py   -> same main.py, but a wrapper appends tiny-model/synthetic-data
#                       overrides (tiny arch, 24^3 volumes, 2 epochs, 0 workers) so a
#                       fold takes ~1 min on CPU. Original args are logged to
#                       $WORK/main_invocations.log and are what gets asserted on.
#   srun/module/conda/nvidia-smi and the torch.cuda "No GPU visible" check -> stubs.
#   set_env.sh       -> no-op (it only locates CUDA/GCC and exports NEUROSTORM_ENV=greatlakes);
#                       the harness exports NEUROSTORM_ENV=local instead.
#   pretrained ckpt  -> tiny random ckpt in the same format.
# NOT covered: Slurm itself, partitions/time limits/memory, real CUDA/Mamba kernels,
# set_env.sh, accuracy.
#
# Usage (repo root, neurostorm-sandbox env active):
#   bash scripts/sandbox/run_greatlakes_checks.sh [setup|static|guard|5fold|loocv|sbatch|all]   (default: all)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_REAL="${PYTHON:-$(command -v python)}"
WORK="${SANDBOX_GL_DIR:-$REPO/output/sandbox_gl}"
STAGE="${1:-all}"
EPOCHS="${SANDBOX_MAX_EPOCHS:-2}"
N_SUBJ=26
GL="run_finetune_scripts/greatlakes"
DATA_ROOT="$WORK/data_root"
IMG_DIR="$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1"
TINY=(--embed_dim 12 --depths 1 1 --num_heads 2 4 --c_multiplier 2 --img_size 24 24 24 6
      --window_size 2 2 2 2 --first_window_size 2 2 2 2)

if command -v sbatch >/dev/null 2>&1 && [[ -z "${SANDBOX_ALLOW_CLUSTER:-}" ]]; then
    echo "sbatch found -> cluster. This harness is for the laptop (SANDBOX_ALLOW_CLUSTER=1 to override)."; exit 1
fi
export NEUROSTORM_ENV=local PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
SHIM=""; "$PYTHON_REAL" -c "import mamba_ssm" 2>/dev/null || SHIM="$REPO/scripts/sandbox/shims:"
FAILS=0
ok()   { echo "  PASS: $*"; }
bad()  { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }
check() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$d"; else bad "$d"; fi; }

# ── setup ────────────────────────────────────────────────────────────────────
setup() {
    echo "== setup: $WORK"
    rm -rf "$WORK"; mkdir -p "$WORK/bin" "$WORK/nas/splits/MICHIGAN" "$WORK/pretrained_models/pretraining"
    cd "$WORK"
    ln -s "$REPO/scripts" scripts
    ln -s "$REPO/run_finetune_scripts" run_finetune_scripts
    echo ': # sandbox no-op (real set_env.sh only locates CUDA/GCC)' > set_env.sh

    export PYTHONPATH="${SHIM}${REPO}"

    "$PYTHON_REAL" "$REPO/scripts/sandbox/make_synthetic_michigan.py" --out "$WORK" --dataset_dir "$IMG_DIR" \
        --no_split --n_subjects $N_SUBJ --blocks_per_task 10 | tail -1
    "$PYTHON_REAL" "$REPO/scripts/sandbox/make_tiny_pretrained_ckpt.py" "${TINY[@]}" \
        --out pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt || { echo "setup failed: tiny ckpt"; exit 1; }

    # 5-fold splits (5 held-out test subjects per fold; 5 val; rest train), in the dir the docs/preflight use
    "$PYTHON_REAL" - <<'PY'
import os
subs=[f"sub-{i:02d}" for i in range(1,27)]
for f in range(5):
    test=subs[f*5:(f+1)*5] if f<4 else subs[20:]
    rest=[s for s in subs if s not in test]; val=rest[:5]; train=rest[5:]
    with open(f"nas/splits/MICHIGAN/split_fixed_{f+1}.txt","w") as fh:
        for n,l in (("train_subjects",train),("val_subjects",val),("test_subjects",test)):
            fh.write(n+"\n"+"".join(s+"\n" for s in l))
PY
    "$PYTHON_REAL" "$REPO/scripts/make_loocv_splits.py" --dataset_name MICHIGAN --source_fold 1 --start_index 101 | tail -2

    # python wrapper: only main.py is rewritten
    cat > bin/python <<EOF
#!/bin/bash
if [[ "\${1:-}" == "main.py" ]]; then
    shift
    echo "\$*" >> "$WORK/main_invocations.log"
    exec "$PYTHON_REAL" "$REPO/main.py" "\$@" ${TINY[@]} \\
        --sequence_length 6 --num_workers 0 --max_epochs $EPOCHS --image_path "$IMG_DIR"
fi
if [[ "\${1:-}" == "-c" && "\${2:-}" == *"torch.cuda.is_available()"* ]]; then
    echo "torch (sandbox stub: GPU assertion skipped) | GPUs: 0"; exit 0
fi
exec "$PYTHON_REAL" "\$@"
EOF
    printf '#!/bin/bash\nexec "$@"\n' > bin/srun
    printf '#!/bin/bash\nexit 0\n' > bin/module
    printf '#!/bin/bash\necho "NVIDIA A40 (sandbox stub), 46068 MiB"\n' > bin/nvidia-smi
    printf '#!/bin/bash\n[[ "${1:-}" == "shell.bash" ]] && exit 0\nexit 0\n' > bin/conda
    chmod +x bin/*
    : > main_invocations.log
}

env_run() {  # run a command inside the sandbox env
    ( cd "$WORK"
      export PATH="$WORK/bin:$PATH" DATA_ROOT PYTHONPATH="${SHIM}${REPO}"
      "$@" )
}

# ── verification helper: arg = project_name expected_test_subjects_file_split
verify_run() {  # verify_run <project_name> <split_file> [tag]
    local proj="$1" split="$2"
    env_run "$PYTHON_REAL" "$REPO/scripts/sandbox/verify_run.py" --project "$proj" --split "$split" \
        --ckpt pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt \
        && ok "$proj: fine-tuned, checkpoints saved, predictions valid" \
        || bad "$proj: see verify output above"
}

stage_static() {
    echo "== static checks"
    local ds yaml_name preflight_dir
    for f in $GL/5fold/run_finetune_xcpd_cv.sh $GL/loocv/run_loocv_xcpd.sh; do
        bash -n "$REPO/$f" && ok "bash -n $f" || bad "syntax $f"
    done
    for f in $GL/5fold/sbatch_finetune_xcpd_cv.sh $GL/loocv/sbatch_loocv_xcpd.sh $GL/loocv/sbatch_loocv_permutation_testing.sh; do
        bash -n "$REPO/$f" && ok "bash -n $f" || bad "syntax $f"
    done
    for f in $GL/5fold/run_finetune_xcpd_cv.sh $GL/loocv/run_loocv_xcpd.sh; do
        ds=$(sed -n 's/^DATASET="\(.*\)"/\1/p' "$REPO/$f" | head -1)
        preflight_dir=$(sed -n 's/^SPLIT_DIR="\(.*\)"/\1/p' "$REPO/$f" | head -1)
        yaml_name=$(sed -n 's/^dataset_name: *\([A-Za-z_]*\).*/\1/p' "$REPO/scripts/configs/datasets/$ds.yaml" | head -1)
        read_dir=$(PYTHONPATH="${SHIM}${REPO}" "$PYTHON_REAL" -c "from datasets.data_module import SPLIT_DIR_ALIASES as A; print('nas/splits/'+A.get('$yaml_name','$yaml_name'))" 2>/dev/null)
        if [[ "$preflight_dir" == "$read_dir" ]]; then
            ok "$(basename "$f"): preflight dir == dir main.py reads ($read_dir)"
        else
            bad "$(basename "$f"): preflight checks '$preflight_dir' but main.py reads '$read_dir'"
        fi
    done
}

# Prove a random / invalid split can never be used for the Michigan datasets.
stage_guard() {
    echo "== random-split guard"
    local before after out
    before=$(ls "$WORK/nas/splits/MICHIGAN" | wc -l)
    out=$(env_run bash $GL/loocv/run_loocv_xcpd.sh --start_index 150 --n_folds 1 2>&1)
    [[ $? -ne 0 ]] && echo "$out" | grep -q "MISSING split file" \
        && ok "launcher preflight aborts on a missing split file" || bad "launcher preflight did not abort on missing split"

    # bypass the launcher: call main.py directly for a fold with no split file
    out=$(env_run bash -c "python main.py --dataset_name MICHIGAN_FULL_BLOCK --dataset_split_num 150 --image_path x --max_epochs 1 --model neurostorm --task_name imagery_task_classification --downstream_task_type classification --num_classes 3 --project_name guard_test" 2>&1)
    echo "$out" | grep -q "Refusing to auto-generate a random split" \
        && ok "main.py refuses to run with a missing split file" || { bad "main.py did not refuse a missing split"; echo "$out" | tail -5; }
    after=$(ls "$WORK/nas/splits/MICHIGAN" | wc -l)
    [[ -e "$WORK/nas/splits/MICHIGAN_FULL_BLOCK" ]] && bad "nas/splits/MICHIGAN_FULL_BLOCK was (re)created" \
        || ok "no nas/splits/MICHIGAN_FULL_BLOCK folder needed or created"
    [[ "$before" == "$after" ]] && ok "no split file was auto-generated" || bad "split dir changed: $before -> $after files"

    # a split that looks randomly generated (test == val) must be rejected
    local bad_split="$WORK/nas/splits/MICHIGAN/split_fixed_151.txt"
    printf 'train_subjects\nsub-05\nsub-06\nval_subjects\nsub-02\nsub-03\ntest_subjects\nsub-02\nsub-03\n' > "$bad_split"
    out=$(env_run bash -c "python main.py --dataset_name MICHIGAN_FULL_BLOCK --dataset_split_num 151 --image_path x --max_epochs 1 --model neurostorm --task_name imagery_task_classification --downstream_task_type classification --num_classes 3 --project_name guard_test" 2>&1)
    echo "$out" | grep -q "Invalid split file" && ok "main.py rejects a split with val == test" || { bad "val==test split was not rejected"; echo "$out" | tail -5; }
    rm -f "$bad_split"
}

stage_5fold() {
    echo "== 5-fold runner (all 5 folds)"
    env_run bash $GL/5fold/run_finetune_xcpd_cv.sh > "$WORK/5fold.log" 2>&1 \
        && ok "run_finetune_xcpd_cv.sh exit 0" || { bad "run_finetune_xcpd_cv.sh exit non-zero (tail below)"; tail -25 "$WORK/5fold.log"; }
    grep -q "Preflight OK" "$WORK/5fold.log" && ok "preflight passed" || bad "preflight did not pass"
    for i in 1 2 3 4 5; do
        verify_run "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_fold-$i" "nas/splits/MICHIGAN/split_fixed_$i.txt"
    done
}

stage_loocv() {
    echo "== LOOCV runner: dry-run over all 26 folds, then folds 101-102 for real"
    env_run bash $GL/loocv/run_loocv_xcpd.sh --dry_run > "$WORK/loocv_dry.log" 2>&1 \
        && grep -q "Preflight OK" "$WORK/loocv_dry.log" && ok "26-fold preflight + dry run" \
        || { bad "26-fold dry run"; tail -15 "$WORK/loocv_dry.log"; }
    env_run bash $GL/loocv/run_loocv_xcpd.sh --start_index 101 --n_folds 2 > "$WORK/loocv.log" 2>&1 \
        && ok "run_loocv_xcpd.sh (folds 101-102) exit 0" || { bad "run_loocv_xcpd.sh exit non-zero"; tail -25 "$WORK/loocv.log"; }
    for i in 101 102; do
        verify_run "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-$i" "nas/splits/MICHIGAN/split_fixed_$i.txt"
    done
}

# Run an sbatch_* file's own bash with Slurm env faked. Only edits: the hard-coded
# /sw/.../conda hook path -> stub, and the hard-coded /nfs DATA_ROOT -> sandbox data (neither exists off-cluster).
run_sbatch() {  # run_sbatch <sbatch file> <array id> [extra env...]
    local f="$1" id="$2"; shift 2
    local copy="$WORK/sbatch_copy_$(basename "$f")"
    sed -e "s#/sw/pkgs/arc/mamba/py3.12/bin/conda#$WORK/bin/conda#" \
        -e "s#^export DATA_ROOT=.*#export DATA_ROOT=\"$DATA_ROOT\"#" "$REPO/$f" > "$copy"
    mkdir -p "$WORK/fakehome/code"; ln -sfn "$WORK" "$WORK/fakehome/code/NeuroSTORM"
    env_run env HOME="$WORK/fakehome" SLURM_ARRAY_TASK_ID="$id" SLURM_ARRAY_JOB_ID=9001 SLURM_JOB_ID=9001 \
        SLURM_CPUS_PER_TASK=2 SLURM_GPUS_ON_NODE=1 "$@" bash "$copy"
}

stage_sbatch() {
    echo "== sbatch job files (Slurm env faked)"
    run_sbatch $GL/5fold/sbatch_finetune_xcpd_cv.sh 4 > "$WORK/sb_5fold.log" 2>&1 \
        && ok "sbatch_finetune_xcpd_cv.sh task 4" || { bad "sbatch_finetune_xcpd_cv.sh task 4"; tail -20 "$WORK/sb_5fold.log"; }
    grep -q "=== Fold 4/5 ===" "$WORK/sb_5fold.log" && ! grep -q "=== Fold 1/5" "$WORK/sb_5fold.log" \
        && ok "array task 4 ran exactly fold 4" || bad "array task 4 did not run exactly fold 4"

    run_sbatch $GL/loocv/sbatch_loocv_xcpd.sh 126 > "$WORK/sb_loocv.log" 2>&1 \
        && ok "sbatch_loocv_xcpd.sh task 126" || { bad "sbatch_loocv_xcpd.sh task 126"; tail -20 "$WORK/sb_loocv.log"; }
    grep -q "=== LOOCV fold 126" "$WORK/sb_loocv.log" && ! grep -q "=== LOOCV fold 101" "$WORK/sb_loocv.log" \
        && ok "array task 126 ran exactly fold 126" || bad "array task 126 did not run exactly fold 126"
    verify_run "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-126" "nas/splits/MICHIGAN/split_fixed_126.txt"

    run_sbatch $GL/loocv/sbatch_loocv_permutation_testing.sh 105 > "$WORK/sb_perm.log" 2>&1 \
        && ok "sbatch_loocv_permutation_testing.sh task 105 (3 perms)" || { bad "sbatch_loocv_permutation_testing.sh task 105"; tail -20 "$WORK/sb_perm.log"; }
    for p in 1 2 3; do
        verify_run "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-105_perm$p" "nas/splits/MICHIGAN/split_fixed_105.txt"
    done
    n=$(grep -c -- "--permute_labels" "$WORK/main_invocations.log")
    [[ "$n" -eq 3 ]] && ok "3 permutation runs passed --permute_labels" || bad "expected 3 --permute_labels invocations, saw $n"
    [[ $(grep -o -- "--permutation_seed [0-9]*" "$WORK/main_invocations.log" | sort -u | wc -l) -eq 3 ]] \
        && ok "3 distinct permutation seeds" || bad "permutation seeds not distinct"
}

case "$STAGE" in
    setup)  setup ;;
    static) stage_static ;;
    guard)  stage_guard ;;
    5fold)  stage_5fold ;;
    loocv)  stage_loocv ;;
    sbatch) stage_sbatch ;;
    all)    setup; stage_static; stage_guard; stage_5fold; stage_loocv; stage_sbatch ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo; echo "== done: $FAILS failure(s). Logs/outputs in $WORK"
[[ "$FAILS" -eq 0 ]]

#!/bin/bash
# run_test_only_all.sh
#
# Generic test-only backfill, usable for ANY of your fold-based conditions
# (pooled, within, loocv, reid, win28, randominit, or anything new) as long
# as its project directories follow the "..._fold-<N>" naming convention
# every one of your run scripts already uses (run_pooled_all_folds.sh,
# run_within_all_subjects.sh, run_loocv_xcpd.sh, run_michigan_reid_cv.sh --
# same convention compare_conditions_v2.py's own FOLD_RE regex relies on).
#
# >>> You may not need this at all for most folds. <<<
# Every finetune/train_scratch run in main.py ALREADY auto-evaluates its
# best-validation checkpoint on the test set at the end of trainer.fit()
# (see main.py's "Evaluate the BEST-validation checkpoint" block, right
# after trainer.fit() returns) -- so if a fold finished training without
# crashing, its test_<metric> numbers are already sitting in that fold's
# own TensorBoard event files. No backfill needed -- just read them (e.g.
# via compare_conditions_v2.py, or `tensorboard --logdir <project_dir>`).
#
# Use THIS script only for folds that crashed, were killed, or ran out of
# wall-clock time BEFORE that final auto-test step (so test_* was never
# logged for them), or when you deliberately want to re-test a specific
# saved checkpoint. For each matching project directory, it:
#   1. Finds that fold's checkpoint: exactly one checkpt-*.ckpt file is
#      expected per directory (ModelCheckpoint's save_top_k defaults to 1
#      in this codebase's pinned pytorch-lightning==1.9.4 -- no override
#      anywhere in main.py -- so Lightning deletes the previous "best" each
#      time a better epoch is found, leaving exactly the global-best one).
#   2. Reads the fold number out of the directory's "fold-<N>" substring
#      and uses it directly as --dataset_split_num (matches how every one
#      of your run scripts names its directories).
#   3. Re-runs --mode test against that checkpoint (main.py's
#      trainer.test(model, ckpt_path=args.test_ckpt_path)), logging a NEW
#      version_N under the SAME project directory (grouped with the
#      original run, nothing gets overwritten -- test-only doesn't touch
#      the checkpoint files either, only fit() does).
#
# Usage:
#   bash run_test_only_all.sh \
#     --project_glob "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_fold-*" \
#     --dataset michigan_xcpd --task task6
#
# Optional: --tpt_strategy (default none, matching every run so far),
# --output_root (default nas/output/neurostorm), --dry_run (print the plan,
# run nothing).
#
# Per-condition examples (fill in once you confirm each project_glob):
#   Pooled:     --project_glob "michigan_pooled_fold-*"                         --dataset michigan_pooled --task task6
#   Within:     --project_glob "michigan_within_fold-*"                        --dataset michigan_within --task task6
#   LOOCV:      --project_glob "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*" --dataset michigan_xcpd --task task6
#   ReID:       --project_glob "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_reid_fold-*"  --dataset michigan_xcpd --task task7
#   win28:          <ask -- need the exact project_glob/--dataset/--task you used>
#   randominit:     <ask -- need the exact project_glob/--dataset/--task you used;
#                     if this was run with --mode train_scratch, --dataset/--task
#                     should otherwise match whichever config it was testing>

set -euo pipefail

OUTPUT_ROOT="nas/output/neurostorm"
TPT_STRATEGY="none"
DRY_RUN=0
PROJECT_GLOB=""
DATASET=""
TASK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project_glob) PROJECT_GLOB="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --tpt_strategy) TPT_STRATEGY="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$PROJECT_GLOB" || -z "$DATASET" || -z "$TASK" ]]; then
    echo "Usage: bash run_test_only_all.sh --project_glob '<pattern>_fold-*' --dataset <name> --task <taskN> [--tpt_strategy none] [--dry_run]"
    exit 1
fi

shopt -s nullglob
dirs=("$OUTPUT_ROOT"/$PROJECT_GLOB)
shopt -u nullglob

if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No directories matched $OUTPUT_ROOT/$PROJECT_GLOB"
    exit 1
fi

echo "Found ${#dirs[@]} matching project directories."
echo ""

for dir in "${dirs[@]}"; do
    name=$(basename "$dir")
    if [[ "$name" =~ fold-([0-9]+) ]]; then
        fold_num="${BASH_REMATCH[1]}"
    else
        echo "SKIP $name -- no 'fold-<N>' substring found, can't infer --dataset_split_num"
        continue
    fi

    ckpts=("$dir"/checkpt-*.ckpt)
    if [[ ! -e "${ckpts[0]:-}" ]]; then
        echo "SKIP $name -- no checkpt-*.ckpt found (training may never have reached a validation epoch)"
        continue
    fi
    if [[ ${#ckpts[@]} -gt 1 ]]; then
        echo "WARNING $name -- found ${#ckpts[@]} checkpt-*.ckpt files (expected exactly 1 -- check manually"
        echo "         if save_top_k was overridden for this run); using the first: ${ckpts[0]}"
        echo "         others: ${ckpts[@]:1}"
    fi
    ckpt="${ckpts[0]}"

    echo "=== $name (fold $fold_num) -- ckpt: $(basename "$ckpt") ==="
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry run] would run dataset_split_num=$fold_num test_ckpt_path=$ckpt"
        continue
    fi
    bash scripts/run_experiment.sh \
      --model neurostorm --dataset "$DATASET" --task "$TASK" --mode test \
      -- --dataset_split_num "$fold_num" --tpt_strategy "$TPT_STRATEGY" \
         --test_ckpt_path "$ckpt" --project_name "$name"
done

echo ""
echo "Done. Each fold's fresh test run logged a NEW version_N under its own"
echo "$OUTPUT_ROOT/<project_name>/ directory (same project_name as the"
echo "original run, so it's grouped there, not overwritten) -- check"
echo "test_<metric> there, e.g.:"
echo "  tensorboard --logdir $OUTPUT_ROOT/<project_name>"

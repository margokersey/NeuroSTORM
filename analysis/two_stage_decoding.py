"""Two-stage decoding during wakefulness (Test 2).

Stage A (binary):  Task vs. No Task      -- run_loocv_taskvsrest.sh (new)
Stage B (3-way):   tennis / navigation / squeeze -- existing 3-way LOOCV models

Both stages are leave-one-subject-out on the SAME folds (split_fixed_101..126),
so for every fold the held-out subject was unseen by both models. This script
composes their saved test predictions:

  * Stage A metrics on Task vs. No Task blocks (accuracy, balanced accuracy, AUROC)
  * Stage B accuracy on the blocks Stage A called "Task"   (the two-stage number)
    and on ALL true-task blocks (= the plain 3-way LOOCV number, for reference)
  * End-to-end 4-way outcome: No Task / tennis / navigation / squeeze. A mock
    block that Stage A wrongly calls "Task" is never seen by Stage B (it has no
    3-way prediction), so it is counted as an error in a "task (unclassified)" column.
  * Optional Stage A label-permutation test (aggregate across folds), mirroring
    scripts/analyze_loocv_permutation_results.py.

Alignment: test dataloaders are unshuffled and both datasets walk the 30 imagery
blocks of timing_onehot_task1.tsv in order, so Stage A's true-Task samples and
Stage B's samples line up one-to-one. This is VERIFIED per fold (sample counts,
and Stage B's true-label sequence against the TSV); a fold that fails is skipped
with a warning rather than silently misaligned.

Usage (from the repo root, after both sets of folds finished):
    python analysis/two_stage_decoding.py \\
      --stage_a_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*" \\
      --stage_b_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*" \\
      --tsv "$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv" \\
      --exclude_subjects sub-29 \\
      --output_json output/two_stage_decoding.json

Add --stage_a_perm_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*_perm*"
for the Stage A permutation test. `_perm` directories are never picked up by the
real-run globs above.
"""
import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np

STAGE_B_NAMES = ['tennis', 'navigation', 'squeeze']          # label ids 0, 1, 2
LABEL_MAP = {'tennis': 0, 'navigation': 1, 'squeeze': 2}
E2E_ROWS = ['notask', 'tennis', 'navigation', 'squeeze']      # true classes
E2E_COLS = ['notask', 'tennis', 'navigation', 'squeeze', 'task_unclassified']


# ---------------------------------------------------------------- loading ----
def predictions_dir(output_dir):
    """Latest lightning_logs/version_N/test_predictions (re-tests write new versions)."""
    versions = []
    for d in (Path(output_dir) / "lightning_logs").glob("version_*/test_predictions"):
        try:
            versions.append((int(d.parent.name.split("_")[-1]), d))
        except ValueError:
            pass
    if versions:
        return max(versions)[1]
    return Path(output_dir) / "test_predictions"


def load_predictions(output_dir):
    files = sorted(predictions_dir(output_dir).glob("test_predictions_fold_*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def fold_of(path):
    m = re.search(r'fold-(\d+)', str(path))
    return int(m.group(1)) if m else None


def perm_of(path):
    m = re.search(r'_perm(\d+)\b', str(path))
    return int(m.group(1)) if m else None


def tsv_block_labels(tsv_path):
    """Class id (0/1/2) of each imagery block, in time order (matches Michigan._blocks_from_tsv)."""
    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t')
    df['squeeze'] = df['squeeze_imagery'] | df['squeeze_action']
    labels = df[['rest', 'squeeze', 'navigation', 'tennis']].idxmax(axis=1).tolist()
    out, prev = [], None
    for lab in labels:
        if lab != prev and lab != 'rest':
            out.append(LABEL_MAP[lab])
        prev = lab
    return out


# ---------------------------------------------------------------- metrics ----
def auroc(scores, labels):
    """Mann-Whitney AUROC with tie-averaged ranks; NaN if one class is missing."""
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = scores.argsort().argsort().astype(float) + 1.0   # 1-based
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stage_a_metrics(preds):
    y = np.array([int(p['true_label']) for p in preds])
    yhat = np.array([int(p['predicted_label']) for p in preds])
    p_task = np.array([p['softmax_probs'][1] for p in preds], float)
    rec_task = float((yhat[y == 1] == 1).mean()) if (y == 1).any() else float('nan')
    rec_notask = float((yhat[y == 0] == 0).mean()) if (y == 0).any() else float('nan')
    return {
        'n': int(len(y)),
        'accuracy': float((yhat == y).mean()),
        'balanced_accuracy': float(np.nanmean([rec_task, rec_notask])),
        'recall_task': rec_task,
        'recall_notask': rec_notask,
        'auroc': auroc(p_task, y),
    }


def compose_fold(a_preds, b_preds, block_labels, fold):
    """Return per-fold results or raise ValueError describing an alignment problem."""
    a_task = [p for p in a_preds if int(p['true_label']) == 1]
    a_notask = [p for p in a_preds if int(p['true_label']) == 0]
    if len(a_task) != len(b_preds) or len(b_preds) != len(block_labels):
        raise ValueError(f"fold {fold}: Stage A task blocks={len(a_task)}, Stage B samples="
                         f"{len(b_preds)}, TSV blocks={len(block_labels)} -- cannot align")
    b_true = [int(p['true_label']) for p in b_preds]
    if b_true != list(block_labels):
        raise ValueError(f"fold {fold}: Stage B true-label order does not match the TSV block order")
    subjects = {p['subject'] for p in a_preds} | {p['subject'] for p in b_preds}
    if len(subjects) != 1:
        raise ValueError(f"fold {fold}: expected one held-out subject, got {sorted(subjects)}")

    cm = np.zeros((4, 5), dtype=int)  # rows: E2E_ROWS (true), cols: E2E_COLS (predicted)
    b_correct_all, b_correct_called = 0, 0
    n_called = 0
    for pa, pb in zip(a_task, b_preds):
        t, bp = int(pb['true_label']), int(pb['predicted_label'])
        b_correct_all += int(bp == t)
        if int(pa['predicted_label']) == 1:           # Stage A says Task -> Stage B decides
            n_called += 1
            b_correct_called += int(bp == t)
            cm[1 + t, 1 + bp] += 1
        else:                                          # Stage A says No Task
            cm[1 + t, 0] += 1
    for pa in a_notask:
        if int(pa['predicted_label']) == 0:
            cm[0, 0] += 1
        else:
            cm[0, 4] += 1                              # false alarm: no 3-way prediction exists

    n_total = int(cm.sum())
    e2e_correct = int(cm[0, 0] + sum(cm[1 + k, 1 + k] for k in range(3)))
    return {
        'fold': fold,
        'subject': subjects.pop(),
        'stage_a': stage_a_metrics(a_preds),
        'stage_b_all_task_blocks_acc': b_correct_all / len(b_preds),
        'stage_b_called_task_n': n_called,
        'stage_b_called_task_acc': (b_correct_called / n_called) if n_called else float('nan'),
        'end_to_end_acc': e2e_correct / n_total,
        'end_to_end_confusion': cm.tolist(),
    }


def mean_sem(vals):
    v = np.array([x for x in vals if x == x], float)
    if len(v) == 0:
        return {'mean': float('nan'), 'sem': float('nan'), 'n': 0}
    sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')
    return {'mean': float(v.mean()), 'sem': sem, 'n': int(len(v))}


# ------------------------------------------------------------------- main ----
def collect(glob_pat, exclude_subjects, want_perm):
    """{fold: [(perm_idx|None, preds), ...]} from run dirs matching the glob."""
    found = {}
    for d in sorted(glob(glob_pat)):
        is_perm = '_perm' in Path(d).name
        if is_perm != want_perm:
            continue
        fold = fold_of(d)
        preds = load_predictions(d)
        if fold is None or preds is None:
            print(f"  [skip] no predictions in {d}", file=sys.stderr)
            continue
        if any(p['subject'] in exclude_subjects for p in preds):
            continue
        found.setdefault(fold, []).append((perm_of(d), preds))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage_a_glob', required=True)
    ap.add_argument('--stage_b_glob', required=True)
    ap.add_argument('--tsv', required=True, help='timing_onehot_task1.tsv of the imagery run-1 data')
    ap.add_argument('--exclude_subjects', nargs='*', default=['sub-29'],
                    help='held-out subjects left out of the evaluation (default: sub-29, no rest-2)')
    ap.add_argument('--stage_a_perm_glob', default=None)
    ap.add_argument('--output_json', default=None)
    args = ap.parse_args()

    block_labels = tsv_block_labels(args.tsv)
    excl = set(args.exclude_subjects)
    a_runs = collect(args.stage_a_glob, excl, want_perm=False)
    b_runs = collect(args.stage_b_glob, excl, want_perm=False)

    per_fold = []
    for fold in sorted(set(a_runs) & set(b_runs)):
        try:
            per_fold.append(compose_fold(a_runs[fold][0][1], b_runs[fold][0][1], block_labels, fold))
        except ValueError as e:
            print(f"  [warn] {e}", file=sys.stderr)
    missing = sorted(set(a_runs) ^ set(b_runs))
    if missing:
        print(f"  [warn] folds present in only one stage (skipped): {missing}", file=sys.stderr)
    if not per_fold:
        sys.exit("No folds could be composed. Check the globs and that both stages finished.")

    pooled = np.sum([np.array(r['end_to_end_confusion']) for r in per_fold], axis=0)
    result = {
        'n_folds': len(per_fold),
        'excluded_subjects': sorted(excl),
        'chance': {'stage_a': 0.5, 'stage_b': 1 / 3, 'end_to_end_majority': 0.5},
        'stage_a': {k: mean_sem([r['stage_a'][k] for r in per_fold])
                    for k in ('accuracy', 'balanced_accuracy', 'recall_task', 'recall_notask', 'auroc')},
        'stage_b_all_task_blocks_acc': mean_sem([r['stage_b_all_task_blocks_acc'] for r in per_fold]),
        'stage_b_called_task_acc': mean_sem([r['stage_b_called_task_acc'] for r in per_fold]),
        'end_to_end_acc': mean_sem([r['end_to_end_acc'] for r in per_fold]),
        'end_to_end_confusion_pooled': {'rows_true': E2E_ROWS, 'cols_pred': E2E_COLS, 'counts': pooled.tolist()},
        'per_fold': per_fold,
    }

    if args.stage_a_perm_glob:
        perms = collect(args.stage_a_perm_glob, excl, want_perm=True)
        folds_used = [r['fold'] for r in per_fold]
        by_perm = {}
        for fold in folds_used:
            for k, preds in perms.get(fold, []):
                y = np.array([int(p['true_label']) for p in preds])
                yhat = np.array([int(p['predicted_label']) for p in preds])
                by_perm.setdefault(k, {})[fold] = float((y == yhat).mean())
        complete = {k: v for k, v in by_perm.items() if set(v) == set(folds_used)}
        real_mean = result['stage_a']['accuracy']['mean']
        if complete:
            null = np.array([np.mean(list(v.values())) for v in complete.values()])
            n = len(null)
            result['stage_a_permutation'] = {
                'real_fold_mean_accuracy': real_mean,
                'null_fold_mean_accuracy': {'mean': float(null.mean()), 'max': float(null.max()), 'n_permutations': n},
                'p_value': float((1 + (null >= real_mean).sum()) / (1 + n)),
                'note': 'p = (1+count)/(1+n); p<0.05 is unattainable with n<19.',
            }
        else:
            print("  [warn] no permutation index has results for every fold; permutation test skipped", file=sys.stderr)

    # ----- print summary
    def fmt(d):
        return f"{d['mean']:.3f} ± {d['sem']:.3f} (n={d['n']})"
    print(f"\nTwo-stage decoding, {len(per_fold)} held-out subjects (excluded: {sorted(excl)})")
    print(f"  Stage A accuracy          {fmt(result['stage_a']['accuracy'])}   chance 0.50")
    print(f"  Stage A balanced acc      {fmt(result['stage_a']['balanced_accuracy'])}")
    print(f"  Stage A AUROC             {fmt(result['stage_a']['auroc'])}")
    print(f"  Stage B acc | called Task {fmt(result['stage_b_called_task_acc'])}   chance 0.33")
    print(f"  Stage B acc | all task    {fmt(result['stage_b_all_task_blocks_acc'])}   (plain 3-way LOOCV)")
    print(f"  End-to-end 4-way acc      {fmt(result['end_to_end_acc'])}")
    print("  Pooled end-to-end confusion (rows=true, cols=predicted):")
    print("    " + " ".join(f"{c[:10]:>10}" for c in E2E_COLS))
    for name, row in zip(E2E_ROWS, pooled):
        print(f"  {name[:10]:>10} " + " ".join(f"{v:>10d}" for v in row))
    if 'stage_a_permutation' in result:
        pm = result['stage_a_permutation']
        print(f"  Stage A permutation: real {pm['real_fold_mean_accuracy']:.3f} vs null mean "
              f"{pm['null_fold_mean_accuracy']['mean']:.3f} (n={pm['null_fold_mean_accuracy']['n_permutations']}), "
              f"p = {pm['p_value']:.3f}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == '__main__':
    main()

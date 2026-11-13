"""Two-stage decoding during wakefulness (Test 2).

Stage A (binary):  Task vs. No Task      -- run_loocv_taskvsrest.sh (new)
Stage B (3-way):   tennis / navigation / squeeze -- existing 3-way LOOCV models,
                   run on ALL Stage A test blocks by
                   analysis/predict_stage_b_on_stage_a_blocks.py (--stage_b_pred_glob).
                   The older --stage_b_glob mode only looks up the 3-way run's saved
                   predictions for the true task blocks (no mock-block predictions).

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

With --stage_b_pred_glob the report also gives, per block group (real task / mock
x Stage A right / wrong): Stage B accuracy (real blocks), predicted-class counts
(mock blocks have no true 3-way label), and confidence (logit margin, max logit, max
softmax prob, entropy), plus within-subject paired comparisons (one value per
held-out subject, Wilcoxon signed-rank; no bootstrap).

Usage (from the repo root, after both sets of folds finished):
    python analysis/two_stage_decoding.py \\
      --stage_a_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*" \\
      --stage_b_pred_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_stageB_allblocks_fold-*" \\
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
    # Score = raw Task logit (monotonic in P(Task)); softmax_probs saturate to exactly 1.0 for
    # large logits, which would create artificial ties in the AUROC.
    p_task = np.array([p['logits'][0] if p.get('logits') else p['softmax_probs'][1] for p in preds], float)
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


# ------------------------------------- Stage B run on ALL Stage A blocks ----
# (analysis/predict_stage_b_on_stage_a_blocks.py). Every block of the held-out
# subject, real task AND mock rest, has a 3-way prediction, so Stage A's call
# only decides which predictions count -- false-positive mock blocks get a real
# tennis/navigation/squeeze label instead of "unclassified".
GROUPS = [  # (key, block type, Stage A called Task?, label)
    ('real_called_task', 'task', True, 'Real task, called Task (correct)'),
    ('real_missed', 'task', False, 'Real task, called No Task (missed)'),
    ('mock_rejected', 'mock', False, 'Mock, called No Task (correct)'),
    ('mock_called_task', 'mock', True, 'Mock, called Task (false alarm)'),
]
GROUP_LABELS = {k: lab for k, _, _, lab in GROUPS}
CONF_MEASURES = {   # per-block Stage B confidence measures
    'logit_margin': 'Logit margin (top1 - top2)',
    'max_logit': 'Max logit',
    'max_prob': 'Max softmax prob',
    'entropy': 'Softmax entropy (nats)',
}
E2E4_LABELS = ['notask', 'tennis', 'navigation', 'squeeze']


def load_stage_b_all(output_dir):
    files = sorted(Path(output_dir).glob("stage_b_predictions_fold_*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def collect_stage_b_all(glob_pat, exclude_subjects):
    """{fold: stage-B-all JSON dict}; folds whose subject is excluded are dropped."""
    found = {}
    for d in sorted(glob(glob_pat)):
        fold = fold_of(d)
        data = load_stage_b_all(d)
        if fold is None or data is None:
            print(f"  [skip] no stage_b_predictions in {d}", file=sys.stderr)
            continue
        if any(p['subject'] in exclude_subjects for p in data['predictions']):
            continue
        found[fold] = data
    return found


def block_confidence(logits):
    lg = np.asarray(logits, float)
    srt = np.sort(lg)[::-1]
    e = np.exp(lg - lg.max())
    p = e / e.sum()
    return {'logit_margin': float(srt[0] - srt[1]), 'max_logit': float(srt[0]),
            'max_prob': float(p.max()), 'entropy': float(-(p * np.log(np.clip(p, 1e-12, None))).sum())}


def compose_fold_all(a_preds, b_data, block_labels, fold):
    """Per-block records + per-fold metrics, or ValueError describing a misalignment."""
    b_preds = b_data['predictions']
    if len(a_preds) != len(b_preds):
        raise ValueError(f"fold {fold}: Stage A has {len(a_preds)} blocks, Stage B {len(b_preds)}")
    subjects = {p['subject'] for p in a_preds} | {p['subject'] for p in b_preds}
    if len(subjects) != 1:
        raise ValueError(f"fold {fold}: expected one held-out subject, got {sorted(subjects)}")
    n_task = sum(int(p['true_label']) == 1 for p in a_preds)
    if n_task != len(block_labels):
        raise ValueError(f"fold {fold}: {n_task} task blocks but the TSV has {len(block_labels)}")
    subject = next(iter(subjects))

    records, k = [], 0
    for pa, pb in zip(a_preds, b_preds):
        if int(pa['true_label']) != int(pb['stage_a_label']):
            raise ValueError(f"fold {fold}: block {pa['sample_idx']} real/mock label differs between stages")
        is_task = int(pa['true_label']) == 1
        called = int(pa['predicted_label']) == 1
        b_true = None
        if is_task:
            b_true = int(block_labels[k])
            k += 1
        key = next(g for g, bt, c, _ in GROUPS if (bt == 'task') == is_task and c == called)
        rec = {'fold': fold, 'subject': subject, 'sample_idx': int(pa['sample_idx']),
               'block_type': 'task' if is_task else 'mock', 'group': key,
               'stage_a_called_task': bool(called), 'stage_a_logit': float(pa['logits'][0]),
               'stage_b_pred': int(pb['predicted_label']), 'stage_b_true': b_true,
               'stage_b_correct': (int(pb['predicted_label']) == b_true) if is_task else None,
               'stage_b_logits': [float(x) for x in pb['logits']],
               'stage_b_probs': [float(x) for x in pb['softmax_probs']]}
        rec.update(block_confidence(pb['logits']))
        records.append(rec)

    # 4x4 end-to-end confusion: rows true (notask, tennis, navigation, squeeze),
    # cols final prediction. Stage A "No Task" -> col 0; "Task" -> Stage B's label.
    cm = np.zeros((4, 4), dtype=int)
    for r in records:
        row = 0 if r['block_type'] == 'mock' else 1 + r['stage_b_true']
        col = 1 + r['stage_b_pred'] if r['stage_a_called_task'] else 0
        cm[row, col] += 1
    real = [r for r in records if r['block_type'] == 'task']
    called = [r for r in real if r['stage_a_called_task']]
    return {
        'fold': fold, 'subject': subject, 'records': records,
        'stage_a': stage_a_metrics(a_preds),
        'stage_b_all_task_blocks_acc': float(np.mean([r['stage_b_correct'] for r in real])),
        'stage_b_called_task_n': len(called),
        'stage_b_called_task_acc': float(np.mean([r['stage_b_correct'] for r in called])) if called else float('nan'),
        'end_to_end_acc': float(np.trace(cm) / cm.sum()),
        'end_to_end_confusion': cm.tolist(),
    }


def subject_group_means(per_fold, measure):
    """{subject: {group: mean of `measure` over that subject's blocks in the group}}."""
    out = {}
    for fr in per_fold:
        for g, _, _, _ in GROUPS:
            vals = [(1.0 if r[measure] else 0.0) if isinstance(r[measure], bool) else r[measure]
                    for r in fr['records'] if r['group'] == g and r[measure] is not None]
            if vals:
                out.setdefault(fr['subject'], {})[g] = float(np.mean(vals))
    return out


def within_subject_test(per_fold, measure, group_a, group_b):
    """Paired within-subject comparison group_a - group_b of a per-block measure.
    group_a/group_b are group keys, or 'real' / 'mock' (all blocks of that type).
    Each subject contributes one difference of its own two group means; subjects
    missing either group are skipped. Returns n, mean diff +/- SEM, Wilcoxon p."""
    def per_subj(spec):
        d = {}
        btype = {'real': 'task', 'mock': 'mock'}.get(spec)
        for fr in per_fold:
            sel = [r for r in fr['records'] if r['group'] == spec or r['block_type'] == btype]
            vals = [float(r[measure]) for r in sel if r[measure] is not None]
            if vals:
                d[fr['subject']] = float(np.mean(vals))
        return d
    A, B = per_subj(group_a), per_subj(group_b)
    subs = sorted(set(A) & set(B))
    diffs = np.array([A[s] - B[s] for s in subs])
    res = {'measure': measure, 'a': group_a, 'b': group_b, 'n_subjects': int(len(subs)),
           'mean_a': float(np.mean([A[s] for s in subs])) if subs else None,
           'mean_b': float(np.mean([B[s] for s in subs])) if subs else None,
           'mean_diff': float(diffs.mean()) if len(subs) else None,
           'sem_diff': float(diffs.std(ddof=1) / np.sqrt(len(subs))) if len(subs) > 1 else None,
           'wilcoxon_p': None}
    if len(subs) >= 2 and np.any(diffs != 0):
        try:
            from scipy.stats import wilcoxon
            res['wilcoxon_p'] = float(wilcoxon(diffs).pvalue)
        except Exception:
            pass
    return res


def group_summary(per_fold):
    """Pooled counts, Stage B accuracy (real blocks) and class distribution per group."""
    out = {}
    for g, _, _, _ in GROUPS:
        recs = [r for fr in per_fold for r in fr['records'] if r['group'] == g]
        d = {'n_blocks': len(recs), 'n_subjects': len({r['subject'] for r in recs}),
             'predicted_class_counts': {STAGE_B_NAMES[k]: sum(r['stage_b_pred'] == k for r in recs) for k in range(3)}}
        if recs and recs[0]['block_type'] == 'task':
            d['stage_b_accuracy_pooled'] = float(np.mean([r['stage_b_correct'] for r in recs]))
        for m in CONF_MEASURES:
            d[f'mean_{m}'] = float(np.mean([r[m] for r in recs])) if recs else None
        out[g] = d
    return out


WITHIN_SUBJECT_COMPARISONS = [   # (measure, group a, group b) -> a minus b, per subject
    ('logit_margin', 'mock_called_task', 'mock_rejected'),
    ('max_logit', 'mock_called_task', 'mock_rejected'),
    ('max_prob', 'mock_called_task', 'mock_rejected'),
    ('entropy', 'mock_called_task', 'mock_rejected'),
    ('logit_margin', 'real_called_task', 'real_missed'),
    ('max_logit', 'real_called_task', 'real_missed'),
    ('stage_b_correct', 'real_called_task', 'real_missed'),
    ('logit_margin', 'real', 'mock'),
    ('max_logit', 'real', 'mock'),
    ('max_prob', 'real', 'mock'),
    ('entropy', 'real', 'mock'),
    ('logit_margin', 'real_called_task', 'mock_called_task'),
    ('max_logit', 'real_called_task', 'mock_called_task'),
    ('max_prob', 'real_called_task', 'mock_called_task'),
    ('entropy', 'real_called_task', 'mock_called_task'),
]


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
    ap.add_argument('--stage_b_pred_glob', default=None,
                    help='Stage B run on ALL Stage A blocks (analysis/predict_stage_b_on_stage_a_blocks.py output dirs, '
                         'e.g. nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_stageB_allblocks_fold-*). Preferred.')
    ap.add_argument('--stage_b_glob', default=None,
                    help='OLD lookup mode: the 3-way LOOCV run dirs; Stage B predictions exist only for true task blocks')
    ap.add_argument('--tsv', required=True, help='timing_onehot_task1.tsv of the imagery run-1 data')
    ap.add_argument('--output_csv', default=None, help='(--stage_b_pred_glob) one row per block with Stage A/B outputs')
    ap.add_argument('--exclude_subjects', nargs='*', default=['sub-29'],
                    help='held-out subjects left out of the evaluation (default: sub-29, no rest-2)')
    ap.add_argument('--stage_a_perm_glob', default=None)
    ap.add_argument('--output_json', default=None)
    args = ap.parse_args()

    if bool(args.stage_b_pred_glob) == bool(args.stage_b_glob):
        sys.exit("Pass exactly one of --stage_b_pred_glob (preferred) or --stage_b_glob (old lookup mode).")
    block_labels = tsv_block_labels(args.tsv)
    excl = set(args.exclude_subjects)
    a_runs = collect(args.stage_a_glob, excl, want_perm=False)
    all_blocks = bool(args.stage_b_pred_glob)

    per_fold = []
    if all_blocks:
        b_runs = collect_stage_b_all(args.stage_b_pred_glob, excl)
        for fold in sorted(set(a_runs) & set(b_runs)):
            try:
                per_fold.append(compose_fold_all(a_runs[fold][0][1], b_runs[fold], block_labels, fold))
            except ValueError as e:
                print(f"  [warn] {e}", file=sys.stderr)
    else:
        b_runs = collect(args.stage_b_glob, excl, want_perm=False)
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
        'mode': 'stage_b_all_blocks' if all_blocks else 'lookup_true_task_blocks_only',
        'n_folds': len(per_fold),
        'excluded_subjects': sorted(excl),
        'chance': {'stage_a': 0.5, 'stage_b': 1 / 3, 'end_to_end_majority': 0.5},
        'stage_a': {k: mean_sem([r['stage_a'][k] for r in per_fold])
                    for k in ('accuracy', 'balanced_accuracy', 'recall_task', 'recall_notask', 'auroc')},
        'stage_b_all_task_blocks_acc': mean_sem([r['stage_b_all_task_blocks_acc'] for r in per_fold]),
        'stage_b_called_task_acc': mean_sem([r['stage_b_called_task_acc'] for r in per_fold]),
        'end_to_end_acc': mean_sem([r['end_to_end_acc'] for r in per_fold]),
        'end_to_end_confusion_pooled': {'rows_true': E2E4_LABELS if all_blocks else E2E_ROWS,
                                        'cols_pred': E2E4_LABELS if all_blocks else E2E_COLS,
                                        'counts': pooled.tolist()},
        'per_fold': [{k: v for k, v in r.items() if k != 'records'} for r in per_fold],
    }
    if all_blocks:
        result['groups'] = group_summary(per_fold)
        result['within_subject_tests'] = [within_subject_test(per_fold, m, a, b)
                                          for m, a, b in WITHIN_SUBJECT_COMPARISONS]
        result['within_subject_note'] = ('Each test uses one value per held-out subject (that subject\'s mean in '
                                         'each group; difference = a - b); subjects lacking either group are skipped. '
                                         'Wilcoxon signed-rank on the per-subject differences; p needs >= 6 subjects '
                                         'to be able to fall below 0.05.')
        if args.output_csv:
            import csv
            rows = [r for fr in per_fold for r in fr['records']]
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_csv, 'w', newline='') as f:
                cols = ['fold', 'subject', 'sample_idx', 'block_type', 'group', 'stage_a_called_task', 'stage_a_logit',
                        'stage_b_true', 'stage_b_pred', 'stage_b_correct', 'logit_margin', 'max_logit', 'max_prob',
                        'entropy', 'stage_b_logits', 'stage_b_probs']
                w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
                w.writeheader()
                for r in rows:
                    w.writerow({**r, 'stage_b_logits': ' '.join(f'{x:.4f}' for x in r['stage_b_logits']),
                                'stage_b_probs': ' '.join(f'{x:.6f}' for x in r['stage_b_probs'])})
            print(f"Wrote {args.output_csv}")

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
    rows_lab = E2E4_LABELS if all_blocks else E2E_ROWS
    cols_lab = E2E4_LABELS if all_blocks else E2E_COLS
    print("  Pooled end-to-end confusion (rows=true, cols=predicted):")
    print("    " + " ".join(f"{c[:10]:>10}" for c in cols_lab))
    for name, row in zip(rows_lab, pooled):
        print(f"  {name[:10]:>10} " + " ".join(f"{v:>10d}" for v in row))
    if all_blocks:
        print("\n  Stage B on every Stage A block, by block type and Stage A call (pooled):")
        print(f"    {'group':<38}{'n':>5}{'subj':>5}  {'tennis/nav/squeeze':>19}{'acc':>7}{'margin':>8}{'maxprob':>9}")
        for g, _, _, lab in GROUPS:
            d = result['groups'][g]
            cc = d['predicted_class_counts']
            acc = f"{d['stage_b_accuracy_pooled']:.3f}" if 'stage_b_accuracy_pooled' in d else '   -'
            mm = f"{d['mean_logit_margin']:.2f}" if d['mean_logit_margin'] is not None else '-'
            mp = f"{d['mean_max_prob']:.3f}" if d['mean_max_prob'] is not None else '-'
            print(f"    {lab:<38}{d['n_blocks']:>5}{d['n_subjects']:>5}  "
                  f"{cc['tennis']:>6}/{cc['navigation']:>3}/{cc['squeeze']:>5}{acc:>7}{mm:>8}{mp:>9}")
        print("\n  Within-subject comparisons (a - b; one value per held-out subject, Wilcoxon signed-rank):")
        for t in result['within_subject_tests']:
            if t['n_subjects'] == 0:
                print(f"    {t['measure']:<13} {t['a']} vs {t['b']}: no subject has both groups")
                continue
            sem = f" ± {t['sem_diff']:.3f}" if t['sem_diff'] is not None else ''
            pv = f"p = {t['wilcoxon_p']:.3f}" if t['wilcoxon_p'] is not None else 'p n/a'
            print(f"    {t['measure']:<15} {t['a']:<17} vs {t['b']:<17} n={t['n_subjects']:>2}  "
                  f"{t['mean_a']:.3f} vs {t['mean_b']:.3f}  diff {t['mean_diff']:+.3f}{sem}  {pv}")
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

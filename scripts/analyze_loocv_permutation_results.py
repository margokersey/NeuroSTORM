#!/usr/bin/env python
"""
Analyze LOOCV Permutation Test Results: Real vs. Permuted-Label Models

After training models on both real and within-subject-permuted training labels
(loocv_permutation_testing.sh), this script evaluates every LOOCV fold
(= one held-out subject) and tests the real model against the permutation null.

Evaluation (unit of inference = held-out subject):
  * Per-fold metrics from the saved per-sample predictions:
      accuracy, balanced accuracy, macro-F1, Cohen's kappa, macro one-vs-rest
      AUROC (threshold-free), log-loss, multiclass Brier score, and expected
      calibration error (ECE, 10 bins, top-label).
  * Each metric is averaged over subjects. 95% CIs come from a subject-level
    (cluster) bootstrap over folds.
  * Permutation test per metric: permutation k gives one complete null LOOCV
    run; its subject-mean metric is one null draw.
    p = (1 + #{null at least as good as real}) / (1 + n_perm).
  * Subject-level consistency: one-sided Wilcoxon signed-rank test of
    (real - mean permuted) per subject (supplementary; needs scipy).
  * Pooled (concatenated across folds) confusion matrix, one-vs-rest ROC and
    reliability diagram are reported for visualization.

Test labels are always the true labels; only training labels are permuted.

Usage:
    python scripts/analyze_loocv_permutation_results.py \
        --real_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
        --perm_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*_perm*" \
        --n_permutations 100

Output (defaults under output/loocv_permutation/):
    - loocv_permutation_results.json  {aggregate, metrics, per_fold}
    - loocv_permutation_results_per_fold.csv  (per-subject metric table)
    - loocv_permutation_figure.{png,pdf,svg}  (6-panel publication figure)
    - Console summary with p-values, CIs and effect sizes
"""

import csv
import json
import argparse
import numpy as np
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will not be generated.")

try:
    from sklearn.metrics import (f1_score, cohen_kappa_score,
                                 roc_auc_score, log_loss, roc_curve, confusion_matrix)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Only accuracy will be computed.")

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


DEFAULT_CLASS_NAMES = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}

# name -> (label, higher_is_better, chance level or None)
METRICS = {
    'accuracy':          ('Accuracy',          True,  'uniform'),
    'balanced_accuracy': ('Balanced accuracy', True,  'uniform'),
    'macro_f1':          ('Macro F1',          True,  'uniform'),
    'macro_auroc':       ('Macro AUROC',       True,  0.5),
    'cohen_kappa':       ("Cohen's κ",         True,  0.0),
    'log_loss':          ('Log-loss',          False, 'log_uniform'),
    'brier':             ('Brier score',       False, None),
    'ece':               ('ECE',               False, None),
}
PRIMARY_METRIC = 'balanced_accuracy'


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _predictions_dir(output_dir):
    # Latest lightning_logs/version_N that has predictions (re-tests such as the
    # best-checkpoint backfill write a new version_N), then the old flat layout.
    versions = []
    for d in (Path(output_dir) / "lightning_logs").glob("version_*/test_predictions"):
        try:
            versions.append((int(d.parent.name.split("_")[-1]), d))
        except ValueError:
            pass
    if versions:
        return max(versions)[1]
    return Path(output_dir) / "test_predictions"


def load_fold_summary(output_dir):
    """Load test_summary_fold_X.json from fold output directory."""
    summary_files = list(_predictions_dir(output_dir).glob("test_summary_fold_*.json"))
    if not summary_files:
        return None
    with open(summary_files[0]) as f:
        return json.load(f)


def load_per_sample_predictions(output_dir):
    """Load per-sample predictions from test_predictions_fold_X.json."""
    pred_files = list(_predictions_dir(output_dir).glob("test_predictions_fold_*.json"))
    if not pred_files:
        return None
    with open(pred_files[0]) as f:
        return json.load(f)


def parse_predictions(preds):
    """Per-sample prediction list -> dict of arrays (probs may be None)."""
    if not preds:
        return None
    y = np.array([int(p['true_label']) for p in preds])
    yhat = np.array([int(p['predicted_label']) for p in preds])
    probs = None
    if all('softmax_probs' in p for p in preds):
        probs = np.array([p['softmax_probs'] for p in preds], dtype=float)
    names = {int(p['true_label']): p['true_label_name']
             for p in preds if 'true_label_name' in p}
    return {'y': y, 'yhat': yhat, 'probs': probs, 'names': names}


def extract_fold_id(project_name):
    """Extract fold ID from project name.
    Examples:
    - 'xcpd_nosmooth_general_fold-101' -> (101, None)
    - 'xcpd_nosmooth_general_fold-101_perm1' -> (101, 1)
    """
    try:
        fold_part = project_name.split("fold-")[-1]
        if "_perm" in fold_part:
            fold_id, perm_part = fold_part.split("_perm")
            return int(fold_id), int(perm_part)
        return int(fold_part), None
    except (ValueError, IndexError):
        return None, None


def _load_run(run_dir, label):
    summary = load_fold_summary(run_dir)
    if summary is None:
        print(f"  Warning: No summary found for {label}")
        return None, None
    acc = summary.get('overall_accuracy')
    if acc is None:
        print(f"  Warning: No accuracy in summary for {label}")
        return None, None
    return float(acc), parse_predictions(load_per_sample_predictions(run_dir))


def aggregate_results(real_pattern, perm_pattern, n_permutations):
    """
    Load results for real and permuted models.

    Returns:
        dict fold_id -> {
            'real_accuracy': float,
            'real_preds': parsed predictions or None,
            'perm_acc_by_id': {perm_id: float},
            'perm_preds': {perm_id: parsed predictions or None},
        }
    """
    results = {}

    print(f"Loading real model results from: {real_pattern}")
    for real_dir in sorted(glob(real_pattern)):
        fold_id, perm_id = extract_fold_id(Path(real_dir).name)
        if fold_id is None or perm_id is not None:
            continue
        acc, preds = _load_run(real_dir, Path(real_dir).name)
        if acc is None:
            continue
        results[fold_id] = {'real_accuracy': acc, 'real_preds': preds,
                            'perm_acc_by_id': {}, 'perm_preds': {}}
        n = len(preds['y']) if preds else 0
        print(f"  fold {fold_id}: real accuracy {acc:.3f} ({n} samples)")

    print(f"\nLoading permuted model results from: {perm_pattern}")
    for perm_dir in sorted(glob(perm_pattern)):
        fold_id, perm_id = extract_fold_id(Path(perm_dir).name)
        if fold_id is None or perm_id is None:
            continue
        if fold_id not in results:
            print(f"  Warning: No real model for fold {fold_id}, skipping perm{perm_id}")
            continue
        acc, preds = _load_run(perm_dir, f"fold {fold_id} perm {perm_id}")
        if acc is None:
            continue
        results[fold_id]['perm_acc_by_id'][perm_id] = acc
        results[fold_id]['perm_preds'][perm_id] = preds
        n = len(preds['y']) if preds else 0
        print(f"  fold {fold_id}, perm{perm_id}: {acc:.3f} ({n} samples)")

    valid = {}
    for fold_id, data in results.items():
        n = len(data['perm_acc_by_id'])
        if n != n_permutations:
            print(f"  Warning: fold {fold_id} has {n} permutations "
                  f"(expected {n_permutations}), skipping")
            continue
        valid[fold_id] = data
    return valid


def infer_class_names(results):
    names, n_classes = {}, 0
    for data in results.values():
        runs = [data['real_preds']] + list(data['perm_preds'].values())
        for r in runs:
            if r is None:
                continue
            names.update(r['names'])
            n_classes = max(n_classes, int(r['y'].max()) + 1, int(r['yhat'].max()) + 1)
            if r['probs'] is not None:
                n_classes = max(n_classes, r['probs'].shape[1])
    n_classes = n_classes or len(DEFAULT_CLASS_NAMES)
    return [names.get(c, DEFAULT_CLASS_NAMES.get(c, str(c))) for c in range(n_classes)]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def expected_calibration_error(y, probs, n_bins=10):
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(preds, n_classes, fallback_accuracy=None):
    """All metrics for one run (one fold). Missing ones are NaN."""
    out = {k: np.nan for k in METRICS}
    if preds is None:
        if fallback_accuracy is not None:
            out['accuracy'] = fallback_accuracy
        return out
    y, yhat, probs = preds['y'], preds['yhat'], preds['probs']
    labels = list(range(n_classes))
    out['accuracy'] = float((y == yhat).mean())
    if not HAS_SKLEARN:
        return out
    present = np.unique(y)
    out['balanced_accuracy'] = float(np.mean([(yhat[y == c] == c).mean() for c in present]))
    out['macro_f1'] = float(f1_score(y, yhat, labels=labels, average='macro', zero_division=0))
    if len(present) > 1:
        out['cohen_kappa'] = float(cohen_kappa_score(y, yhat, labels=labels))
    if probs is not None:
        p = np.clip(probs, 1e-12, 1)
        p = p / p.sum(1, keepdims=True)
        if len(present) == n_classes:
            try:
                out['macro_auroc'] = float(roc_auc_score(y, p, multi_class='ovr',
                                                         average='macro', labels=labels))
            except ValueError:
                pass
        out['log_loss'] = float(log_loss(y, p, labels=labels))
        onehot = np.eye(n_classes)[y]
        out['brier'] = float(((p - onehot) ** 2).sum(1).mean())
        out['ece'] = expected_calibration_error(y, p)
    return out


def chance_level(metric, n_classes):
    c = METRICS[metric][2]
    if c == 'uniform':
        return 1.0 / n_classes
    if c == 'log_uniform':
        return float(np.log(n_classes))
    return c


def compute_statistics(results, n_classes):
    """Per-fold metrics for real and every permutation."""
    stats = {}
    for fold_id in sorted(results):
        data = results[fold_id]
        real_m = compute_metrics(data['real_preds'], n_classes, data['real_accuracy'])
        perm_m = {k: compute_metrics(data['perm_preds'][k], n_classes, data['perm_acc_by_id'][k])
                  for k in sorted(data['perm_acc_by_id'])}

        real_acc = data['real_accuracy']
        perm_accs = np.array([data['perm_acc_by_id'][k] for k in sorted(data['perm_acc_by_id'])])
        # Descriptive per-fold p-value (one held-out subject, so coarse; use
        # the aggregate test for inference). Add-one correction: never 0.
        p_value = (1 + (perm_accs >= real_acc).sum()) / (1 + len(perm_accs))

        stats[fold_id] = {
            'real_accuracy': real_acc,
            'perm_mean': float(perm_accs.mean()),
            'perm_std': float(perm_accs.std()),
            'perm_min': float(perm_accs.min()),
            'perm_max': float(perm_accs.max()),
            'effect_size': float(real_acc - perm_accs.mean()),
            'effect_size_std': float((real_acc - perm_accs.mean()) / (perm_accs.std() + 1e-8)),
            'p_value': float(p_value),
            'significant_p05': bool(p_value < 0.05),
            'n_samples': int(len(data['real_preds']['y'])) if data['real_preds'] else None,
            'real_metrics': real_m,
            'perm_metrics': {int(k): v for k, v in perm_m.items()},
        }
    return stats


def aggregate_permutation_test(results):
    """Across-fold permutation test on mean accuracy (kept for continuity).

    Statistic: mean test accuracy over all LOOCV folds. Null: for each
    permutation index k, the mean over folds of the permutation-k accuracy.
    p = (1 + #{null >= real}) / (1 + n_perms).
    """
    fold_ids = sorted(results)
    perm_ids = sorted(set.intersection(*[set(results[f]['perm_acc_by_id']) for f in fold_ids]))
    real = float(np.mean([results[f]['real_accuracy'] for f in fold_ids]))
    null = np.array([np.mean([results[f]['perm_acc_by_id'][k] for f in fold_ids]) for k in perm_ids])
    n = len(null)
    p = (1 + int((null >= real).sum())) / (1 + n)
    return {
        'n_folds': len(fold_ids),
        'n_permutations': n,
        'real_mean_accuracy': real,
        'null_mean_accuracies': [float(x) for x in null],
        'null_mean': float(null.mean()),
        'null_std': float(null.std(ddof=1)) if n > 1 else None,
        'p_value': float(p),
        'min_attainable_p': 1.0 / (1 + n),
        'significant_p05': bool(p < 0.05),
    }


def metric_tests(stats, n_classes, n_bootstrap=10000, seed=0):
    """Subject-level summary, bootstrap CI and permutation test for every metric."""
    rng = np.random.default_rng(seed)
    fold_ids = sorted(stats)
    perm_ids = sorted(set.intersection(*[set(stats[f]['perm_metrics']) for f in fold_ids]))
    boot_idx = rng.integers(0, len(fold_ids), size=(n_bootstrap, len(fold_ids)))
    out = {}
    for m, (label, higher, _) in METRICS.items():
        real_f = np.array([stats[f]['real_metrics'][m] for f in fold_ids])
        if np.all(np.isnan(real_f)):
            continue
        perm_fk = np.array([[stats[f]['perm_metrics'][k][m] for k in perm_ids] for f in fold_ids])
        real = float(np.nanmean(real_f))
        null = np.nanmean(perm_fk, axis=0)            # one subject-mean per permutation
        null = null[~np.isnan(null)]
        boot = np.nanmean(real_f[boot_idx], axis=1)
        ci = np.nanpercentile(boot, [2.5, 97.5])
        n = len(null)
        if n:
            hits = (null >= real) if higher else (null <= real)
            p = (1 + int(hits.sum())) / (1 + n)
        else:
            p = np.nan
        # subject-level paired test: real vs. mean over that subject's permutations
        diff = real_f - np.nanmean(perm_fk, axis=1)
        diff = diff[~np.isnan(diff)]
        w_p = None
        if HAS_SCIPY and len(diff) >= 6 and np.any(diff != 0):
            try:
                w_p = float(wilcoxon(diff, alternative='greater' if higher else 'less').pvalue)
            except ValueError:
                pass
        out[m] = {
            'label': label,
            'higher_is_better': higher,
            'chance': chance_level(m, n_classes),
            'real_mean': real,
            'real_sd_across_subjects': float(np.nanstd(real_f, ddof=1)) if len(fold_ids) > 1 else None,
            'real_ci95': [float(ci[0]), float(ci[1])],
            'null_mean': float(null.mean()) if n else None,
            'null_sd': float(null.std(ddof=1)) if n > 1 else None,
            'null_ci95': [float(x) for x in np.percentile(null, [2.5, 97.5])] if n else None,
            'null_means': [float(x) for x in null],
            'delta_vs_null': float(real - null.mean()) if n else None,
            'perm_p_value': float(p),
            'wilcoxon_p_value': w_p,
            'n_subjects_better_than_null': int((diff > 0).sum() if higher else (diff < 0).sum()),
            'n_subjects': int(len(diff)),
        }
    return out


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def print_aggregate(agg):
    print("\n" + "=" * 80)
    print("AGGREGATE PERMUTATION TEST (mean accuracy over all folds)")
    print("=" * 80)
    print(f"  Folds: {agg['n_folds']}   Permutations: {agg['n_permutations']}")
    print(f"  Real mean accuracy:  {agg['real_mean_accuracy']:.3f}")
    sd = f" ± {agg['null_std']:.3f}" if agg['null_std'] is not None else ""
    print(f"  Null mean accuracy:  {agg['null_mean']:.3f}{sd}")
    print(f"  p = (1 + #null>=real) / (1 + n) = {agg['p_value']:.4f}")
    if agg['min_attainable_p'] >= 0.05:
        print(f"  WARNING: with {agg['n_permutations']} permutations the smallest attainable p is "
              f"{agg['min_attainable_p']:.3f}; p < 0.05 needs >= 19 permutations "
              f"(>= 100 recommended for reporting).")
    print("=" * 80)


def print_metric_table(tests):
    print("\n" + "=" * 96)
    print(f"SUBJECT-LEVEL METRICS (mean over held-out subjects; primary = {METRICS[PRIMARY_METRIC][0]})")
    print("=" * 96)
    print(f"{'Metric':<19s}{'Real':>8s}{'95% CI':>18s}{'Null':>8s}{'Chance':>8s}"
          f"{'Δ':>8s}{'p_perm':>9s}{'p_Wilc':>9s}{'Better':>9s}")
    print("-" * 96)
    for t in tests.values():
        ci =f"[{t['real_ci95'][0]:.3f}, {t['real_ci95'][1]:.3f}]"
        null = f"{t['null_mean']:.3f}" if t['null_mean'] is not None else "–"
        ch = f"{t['chance']:.3f}" if t['chance'] is not None else "–"
        d = f"{t['delta_vs_null']:+.3f}" if t['delta_vs_null'] is not None else "–"
        w = t['wilcoxon_p_value']
        wp = "–" if w is None else (f"{w:.4f}" if w >= 1e-4 else f"{w:.1e}")
        better = f"{t['n_subjects_better_than_null']}/{t['n_subjects']}"
        print(f"{t['label']:<19s}{t['real_mean']:>8.3f}{ci:>18s}{null:>8s}{ch:>8s}"
              f"{d:>8s}{t['perm_p_value']:>9.4f}{wp:>9s}{better:>9s}")
    print("-" * 96)
    print("  p_perm: across-subject permutation test; p_Wilc: one-sided Wilcoxon signed-rank of\n"
          "  per-subject (real - mean permuted); Better: subjects where real beats their null mean.")
    print("=" * 96)


def print_summary(stats):
    fold_ids = sorted(stats)
    print("\nPer-fold breakdown (per-fold p-values are descriptive only):")
    print(f"{'Fold':>6s} {'n':>4s} {'Acc':>7s} {'BalAcc':>7s} {'AUROC':>7s} "
          f"{'PermAcc':>8s} {'Δacc':>7s} {'p':>7s}")
    print("-" * 60)
    for f in fold_ids:
        s = stats[f]
        rm = s['real_metrics']
        n = s['n_samples'] if s['n_samples'] is not None else 0
        print(f"{f:>6d} {n:>4d} {s['real_accuracy']:>7.3f} {rm['balanced_accuracy']:>7.3f} "
              f"{rm['macro_auroc']:>7.3f} {s['perm_mean']:>8.3f} {s['effect_size']:>+7.3f} "
              f"{s['p_value']:>7.3f}")


def _clean(x):
    """Make NaN JSON-safe (null)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def save_results(payload, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(_clean(payload), f, indent=2)
    print(f"Saved results to {output_file}")


def save_per_fold_csv(stats, output_file):
    path = Path(output_file).with_name(Path(output_file).stem + '_per_fold.csv')
    cols = list(METRICS)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'n_samples']
                   + [f'real_{c}' for c in cols] + [f'perm_mean_{c}' for c in cols])
        for fid in sorted(stats):
            s = stats[fid]
            perm = s['perm_metrics'].values()
            w.writerow([fid, s['n_samples']]
                       + [s['real_metrics'][c] for c in cols]
                       + [np.nanmean([p[c] for p in perm]) for c in cols])
    print(f"Saved per-fold table to {path}")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

# Colors: real model = ink, permutation null = neutral gray (a reference, not a
# series), classes = first three slots of a CVD-validated categorical palette
# (blue / orange / aqua), confusion matrix = single-hue sequential blue.
INK = '#1d1d1b'
INK_2 = '#52514e'
NULL = '#b4b3ad'
NULL_DARK = '#7d7c77'
CHANCE = '#8a8984'
CLASS_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
SEQ_BLUE = ['#f7f9fc', '#d3e3f7', '#9cc2ee', '#5c9be2', '#2a78d6', '#1b569f', '#0f3567']


def _style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 7.5,
        'axes.titleweight': 'bold',
        'axes.titlepad': 6,
        'axes.labelsize': 7,
        'axes.labelcolor': INK,
        'axes.edgecolor': INK_2,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'xtick.color': INK_2,
        'ytick.color': INK_2,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'legend.fontsize': 6.5,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'text.color': INK,
        'pdf.fonttype': 42,     # editable text in Illustrator
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.dpi': 600,
        'figure.dpi': 150,
    })


TITLE_PAD = 17   # points; leaves room for a one-row legend between title and axes


def _panel_letter(ax, letter):
    # Same baseline as the panel title, offset left of the y-axis.
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-30, TITLE_PAD),
                textcoords='offset points', fontsize=9, fontweight='bold',
                va='baseline', ha='left', color=INK)


def _top_legend(ax, handles):
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(handles),
              borderaxespad=0.25, borderpad=0, columnspacing=1.0, handletextpad=0.4,
              fontsize=6)


def _fmt_p(p, n_perm):
    if p <= 1.0 / (1 + n_perm) + 1e-12:
        return f"p ≤ {p:.3g}" if p >= 0.001 else "p < 0.001"
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def _pooled(results, which='real'):
    ys, ps, yh = [], [], []
    for f in sorted(results):
        runs = [results[f]['real_preds']] if which == 'real' else list(results[f]['perm_preds'].values())
        for r in runs:
            if r is None:
                continue
            ys.append(r['y']); yh.append(r['yhat'])
            if r['probs'] is not None:
                ps.append(r['probs'])
    if not ys:
        return None, None, None
    y, yhat = np.concatenate(ys), np.concatenate(yh)
    probs = np.concatenate(ps) if len(ps) == len(ys) else None
    return y, yhat, probs


def _unavailable(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=6.5, color=INK_2)
    ax.set_xticks([]); ax.set_yticks([])


def plot_per_subject(ax, stats, metric, chance):
    fold_ids = sorted(stats)
    real = np.array([stats[f]['real_metrics'][metric] for f in fold_ids])
    perm = [np.array([v[metric] for v in stats[f]['perm_metrics'].values()]) for f in fold_ids]
    order = np.argsort(real)
    x = np.arange(len(fold_ids))
    for xi, i in zip(x, order):
        lo, hi = np.nanmin(perm[i]), np.nanmax(perm[i])
        ax.plot([xi, xi], [lo, hi], color=NULL, lw=2.4, solid_capstyle='round', zorder=1)
        ax.plot([xi - 0.32, xi + 0.32], [np.nanmean(perm[i])] * 2, color=NULL_DARK, lw=0.9, zorder=2)
    ax.scatter(x, real[order], s=16, color=INK, edgecolor='white', linewidth=0.6, zorder=3)
    ax.axhline(chance, color=CHANCE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_ylim(0, 1.02)
    if len(fold_ids) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels([str(fold_ids[i]) for i in order], rotation=90, fontsize=5.5)
        ax.tick_params(axis='x', length=0, pad=2)
    else:
        ax.set_xticks([])
    ax.set_xlabel('Held-out subject (sorted)')
    ax.set_ylabel(METRICS[metric][0])
    ax.set_title('Per-subject performance', loc='left', pad=TITLE_PAD)
    _top_legend(ax, [
        Line2D([], [], marker='o', ls='', color=INK, markersize=4, label='Real labels'),
        Line2D([], [], color=NULL, lw=2.4, label='Permuted (range, mean)'),
        Line2D([], [], color=CHANCE, lw=0.7, ls=(0, (3, 2)), label='Chance'),
    ])


def plot_null(ax, test, n_perm, chance):
    null = np.array(test['null_means'])
    real = test['real_mean']
    lo = min(null.min(), real, chance) if len(null) else min(real, chance)
    hi = max(null.max(), real) if len(null) else real
    pad = 0.08 * (hi - lo + 1e-6)
    if len(null) >= 10:
        bins = np.linspace(lo - pad, hi + pad, 30)
        ax.hist(null, bins=bins, color=NULL, edgecolor='white', linewidth=0.5, zorder=1)
        ax.set_ylabel('Permutations')
    else:
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(null))
        ax.scatter(null, jitter, s=18, color=NULL, edgecolor='white', linewidth=0.5, zorder=2)
        ax.set_ylim(-1, 1.6)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    ytop = ax.get_ylim()[1]
    ax.axvline(chance, color=CHANCE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axvline(real, color=INK, lw=1.4, zorder=3)
    right_side = real < hi - 0.3 * (hi - lo)
    ax.annotate(f"Real = {real:.3f}\n{_fmt_p(test['perm_p_value'], n_perm)}",
                xy=(real, ytop), xycoords='data', xytext=(4 if right_side else -4, -2),
                textcoords='offset points', ha='left' if right_side else 'right', va='top',
                fontsize=6.5, color=INK, linespacing=1.3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(f"Mean {METRICS[PRIMARY_METRIC][0].lower()} across subjects")
    ax.set_title(f'Permutation null (n = {n_perm})', loc='left', pad=TITLE_PAD)


def plot_metric_forest(ax, tests, n_perm):
    keys = [k for k in ('balanced_accuracy', 'macro_f1', 'macro_auroc', 'cohen_kappa') if k in tests]
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        t = tests[k]
        if t['null_means']:
            nlo, nhi = (t['null_ci95'] if n_perm >= 20 else
                        (min(t['null_means']), max(t['null_means'])))
            ax.plot([nlo, nhi], [yi, yi], color=NULL, lw=5, solid_capstyle='butt', zorder=1)
            ax.plot([t['null_mean']] * 2, [yi - 0.17, yi + 0.17], color=NULL_DARK, lw=0.9, zorder=2)
        if t['chance'] is not None:
            ax.plot([t['chance']] * 2, [yi - 0.3, yi + 0.3], color=CHANCE, lw=0.7,
                    ls=(0, (2, 1.5)), zorder=0)
        ax.plot(t['real_ci95'], [yi, yi], color=INK, lw=1.0, zorder=3)
        ax.scatter([t['real_mean']], [yi], s=18, color=INK, edgecolor='white', linewidth=0.6, zorder=4)
        ax.text(1.03, yi, _fmt_p(t['perm_p_value'], n_perm), transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=6, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([tests[k]['label'] for k in keys])
    ax.tick_params(axis='y', length=0)
    xmin = min(-0.05, min(tests[k]['real_ci95'][0] for k in keys) - 0.05)
    ax.set_xlim(xmin, 1.0)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel('Score (mean over subjects)')
    null_lbl = 'Permuted (95%)' if n_perm >= 20 else 'Permuted (range)'
    _top_legend(ax, [
        Line2D([], [], marker='o', color=INK, lw=1.0, markersize=4, label='Real (95% CI)'),
        Line2D([], [], color=NULL, lw=5, label=null_lbl),
        Line2D([], [], color=CHANCE, lw=0.7, ls=(0, (2, 1.5)), label='Chance'),
    ])
    ax.set_title('Summary metrics', loc='left', pad=TITLE_PAD)


def plot_confusion(ax, fig, y, yhat, class_names):
    k = len(class_names)
    cm = confusion_matrix(y, yhat, labels=list(range(k))).astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    im = ax.pcolormesh(cmn, cmap=cmap, vmin=0, vmax=1, edgecolors='white', linewidth=1.5)
    for i in range(k):
        for j in range(k):
            v = cmn[i, j]
            ax.text(j + 0.5, i + 0.5, f"{100 * v:.0f}%", ha='center', va='center', fontsize=6.5,
                    color='white' if v > 0.55 else INK, fontweight='bold' if i == j else 'normal')
    ax.set_xlim(0, k); ax.set_ylim(k, 0)
    ax.set_aspect('equal')
    labels = [c.capitalize() for c in class_names]
    ax.set_xticks(np.arange(k) + 0.5); ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(k) + 0.5); ax.set_yticklabels(labels, rotation=90, va='center')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion matrix (pooled)', loc='left', pad=TITLE_PAD)
    ax.set_anchor('N')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['0%', '50%', '100%'])
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label('Row-normalized recall', fontsize=6.5)


def plot_roc(ax, y, probs, class_names):
    ax.plot([0, 1], [0, 1], color=CHANCE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    for c, name in enumerate(class_names):
        yt = (y == c).astype(int)
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, probs[:, c])
        auc = roc_auc_score(yt, probs[:, c])
        ax.plot(fpr, tpr, color=CLASS_COLORS[c % len(CLASS_COLORS)], lw=1.4,
                label=f"{name.capitalize()}  {auc:.2f}")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC, one-vs-rest (pooled)', loc='left', pad=TITLE_PAD)
    ax.set_anchor('N')
    ax.legend(loc='lower right', title='AUROC', title_fontsize=6.5, borderaxespad=0.2)


def _reliability(y, probs, n_bins=10, min_count=5):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= min_count:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


def plot_calibration(ax, real, null, n_classes):
    lo = 1.0 / n_classes
    ax.plot([lo, 1], [lo, 1], color=CHANCE, lw=0.7, ls=(0, (3, 2)), zorder=0, label='Perfect')
    for (y, probs), color, lbl, z in ((null, NULL, 'Permuted', 1), (real, INK, 'Real', 2)):
        if y is None or probs is None:
            continue
        xs, ys = _reliability(y, probs)
        ece = expected_calibration_error(y, probs)
        ax.plot(xs, ys, color=color, lw=1.4, marker='o', markersize=3.2,
                markeredgecolor='white', markeredgewidth=0.5, zorder=z,
                label=f"{lbl} (ECE {ece:.3f})")
    ax.set_xlim(lo - 0.02, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Confidence (max softmax prob.)')
    ax.set_ylabel('Observed accuracy')
    ax.set_title('Calibration (pooled)', loc='left', pad=TITLE_PAD)
    ax.legend(loc='lower right', borderaxespad=0.2)


def plot_results(stats, results, tests, class_names, n_perm, output_file):
    """Six-panel publication figure (183 mm double-column width)."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not installed)")
        return
    _style()
    n_classes = len(class_names)
    metric = PRIMARY_METRIC if PRIMARY_METRIC in tests else 'accuracy'

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.75, wspace=0.6,
                          left=0.07, right=0.95, top=0.92, bottom=0.1)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    plot_per_subject(axes[0], stats, metric, chance_level(metric, n_classes))
    plot_null(axes[1], tests[metric], n_perm, chance_level(metric, n_classes))
    if len(tests) > 1:
        plot_metric_forest(axes[2], tests, n_perm)
    else:
        _unavailable(axes[2], 'Per-sample predictions\nnot available')

    y, yhat, probs = _pooled(results, 'real')
    y0, _, probs0 = _pooled(results, 'perm')
    if y is not None and HAS_SKLEARN:
        plot_confusion(axes[3], fig, y, yhat, class_names)
    else:
        _unavailable(axes[3], 'Per-sample predictions\nnot available')
    if probs is not None and HAS_SKLEARN:
        plot_roc(axes[4], y, probs, class_names)
        plot_calibration(axes[5], (y, probs), (y0, probs0), n_classes)
    else:
        _unavailable(axes[4], 'Softmax probabilities\nnot available')
        _unavailable(axes[5], 'Softmax probabilities\nnot available')

    for ax, letter in zip(axes, 'abcdef'):
        _panel_letter(ax, letter)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved figure to {out.with_suffix('.png')} (+ .pdf, .svg)")


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze LOOCV permutation test results')
    parser.add_argument('--real_dir', type=str, required=True,
                        help='Glob pattern for real model output dirs')
    parser.add_argument('--perm_dir', type=str, required=True,
                        help='Glob pattern for permutation model output dirs')
    parser.add_argument('--n_permutations', type=int, required=True,
                        help='Number of permutations per fold')
    parser.add_argument('--output_file', type=str,
                        default='output/loocv_permutation/loocv_permutation_results.json',
                        help='Output JSON (a _per_fold.csv is written next to it)')
    parser.add_argument('--plot_file', type=str,
                        default='output/loocv_permutation/loocv_permutation_figure.png',
                        help='Figure path; .png, .pdf and .svg are all written')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                        help='Subject-level bootstrap resamples for 95%% CIs')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the bootstrap')
    parser.add_argument('--no_plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    print("Aggregating LOOCV permutation test results...")
    try:
        results = aggregate_results(args.real_dir, args.perm_dir, args.n_permutations)
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1
    if not results:
        print("No results found. Check glob patterns and directory structure.")
        return 1

    class_names = infer_class_names(results)
    n_classes = len(class_names)
    stats = compute_statistics(results, n_classes)
    agg = aggregate_permutation_test(results)
    tests = metric_tests(stats, n_classes, args.n_bootstrap, args.seed)

    print_aggregate(agg)
    print_metric_table(tests)
    print_summary(stats)
    save_results({'aggregate': agg, 'metrics': tests, 'class_names': class_names,
                  'per_fold': stats}, args.output_file)
    save_per_fold_csv(stats, args.output_file)

    if not args.no_plot:
        plot_results(stats, results, tests, class_names, agg['n_permutations'], args.plot_file)
    return 0


if __name__ == '__main__':
    exit(main())

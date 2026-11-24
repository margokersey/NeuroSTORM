#!/usr/bin/env python
"""LOOCV figure comparing two runs of the same folds (default: tpt prompt_ln vs. tpt none).

Same style/metrics as scripts/analyze_loocv_permutation_results.py, but there is no
permutation null: the second condition (tpt none) takes the place of "Permuted".

  a  per-subject balanced accuracy, paired (none vs. prompt_ln), sorted by prompt_ln
  b  distribution of per-subject accuracy, both conditions (+ paired Wilcoxon)
  c  summary metrics, mean over subjects with subject-bootstrap 95% CI, both conditions
  d  pooled confusion matrix (prompt_ln)
  e  pooled one-vs-rest ROC (prompt_ln)
  f  calibration, both conditions

Usage (on the cluster; OUT is the turbo output dir):
  OUT=/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output
  python analysis/plot_tpt_comparison.py \
      --a_dir "$OUT/FIXED_mich_xcpd_nosmooth_tpt_strategy-prompt_ln_loocv_fold-*" \
      --b_dir "$OUT/FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*"
"""
import argparse
import csv
import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_loocv_permutation_results as A  # noqa: E402
from analyze_loocv_permutation_results import (  # noqa: E402
    METRICS, REAL, PERM, PERM_DARK, CHANCE, INK, TITLE_PAD, plt, Line2D)

REAL_DARK = '#34568c'
FOREST_KEYS = ('balanced_accuracy', 'macro_f1', 'macro_auroc', 'cohen_kappa')


def load(pattern):
    """fold_id -> {'acc', 'preds', 'metrics'} for every run matching the glob."""
    out = {}
    for d in sorted(glob(pattern)):
        fid, perm = A.extract_fold_id(Path(d).name)
        if fid is None or perm is not None:
            continue
        acc, preds = A._load_run(d, Path(d).name)
        if acc is not None:
            out[fid] = {'acc': acc, 'preds': preds}
    return out


def pooled(runs, folds):
    ys, yh, ps, gs = [], [], [], []
    for f in folds:
        p = runs[f]['preds']
        if p is None:
            continue
        ys.append(p['y']); yh.append(p['yhat']); gs.append(np.full(len(p['y']), f))
        if p['probs'] is not None:
            ps.append(p['probs'])
    if not ys:
        return None, None, None, None
    probs = np.concatenate(ps) if len(ps) == len(ys) else None
    return np.concatenate(ys), np.concatenate(yh), probs, np.concatenate(gs)


def boot_ci(v, idx):
    return np.nanpercentile(np.nanmean(v[idx], axis=1), [2.5, 97.5])


# ---- panels ---------------------------------------------------------------

def panel_per_subject(ax, folds, ma, mb, subj, la, lb, metric, chance):
    ra = np.array([ma[f][metric] for f in folds])
    rb = np.array([mb[f][metric] for f in folds])
    order = np.argsort(ra)
    x = np.arange(len(folds))
    for xi, i in zip(x, order):
        ax.plot([xi, xi], [ra[i], rb[i]], color=CHANCE, alpha=0.5, lw=1.0, zorder=1)
    ax.scatter(x, rb[order], s=20, color=PERM, edgecolor='white', linewidth=0.6, zorder=3)
    ax.scatter(x, ra[order], s=20, color=REAL, edgecolor='white', linewidth=0.6, zorder=4)
    ax.axhline(chance, color=CHANCE, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([A.participant_label(subj.get(folds[i]), folds[i]) for i in order],
                       rotation=90, fontsize=6)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_xlabel(f'Held-out participant (sorted by {la})')
    ax.set_ylabel(METRICS[metric][0])
    ax.set_title('Per-subject performance', loc='left', pad=TITLE_PAD)
    A._top_legend(ax, [
        Line2D([], [], marker='o', ls='', color=REAL, markersize=4.5, label=la),
        Line2D([], [], marker='o', ls='', color=PERM, markersize=4.5, label=lb),
        Line2D([], [], color=CHANCE, lw=0.8, ls=(0, (3, 2)), label='Chance'),
    ], fontsize=6, ncol=2)


def panel_kde(ax, folds, runs_a, runs_b, la, lb, n_classes, wilcoxon_p):
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 400)
    rows = ((1.0, np.array([runs_b[f]['acc'] for f in folds]), PERM, PERM_DARK, lb),
            (0.0, np.array([runs_a[f]['acc'] for f in folds]), REAL, REAL_DARK, la))
    for y0, vals, color, dark, _ in rows:
        if A.HAS_SCIPY and len(vals) > 2 and np.std(vals) > 0:
            kde = A.gaussian_kde(vals)
            dens = kde(grid) + kde(-grid) + kde(2 - grid)   # reflect at 0/1
            dens = 0.42 * dens / dens.max()
            ax.fill_between(grid, y0 + 0.08, y0 + 0.08 + dens, color=color, alpha=0.4, linewidth=0, zorder=1)
            ax.plot(grid, y0 + 0.08 + dens, color=color, lw=1.2, zorder=2)
        bp = ax.boxplot(vals, positions=[y0], vert=False, widths=0.12, showfliers=False,
                        patch_artist=True, manage_ticks=False, zorder=2)
        for part in ('boxes', 'whiskers', 'caps', 'medians'):
            for item in bp[part]:
                item.set_color(dark); item.set_linewidth(0.9)
        for box in bp['boxes']:
            box.set_facecolor('white')
        ax.scatter(vals, y0 - 0.2 + rng.uniform(-0.06, 0.06, len(vals)), s=8, color=color,
                   edgecolor='white', linewidth=0.3, zorder=3)
        m = vals.mean()
        ax.plot([m, m], [y0 - 0.32, y0 + 0.55], color=dark, lw=1.0, ls=(0, (4, 2)), zorder=3)
        ax.text(m, y0 + 0.58, f'mean = {m:.3f}', color=dark, fontsize=6.5, ha='center', va='bottom')
    ax.axvline(1.0 / n_classes, color=CHANCE, lw=0.8, ls=(0, (1, 2)), zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 1.85)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([f'{lb}\n(n = {len(folds)})', f'{la}\n(n = {len(folds)})'])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Test accuracy per held-out subject')
    ax.set_title('Accuracy distributions', loc='left', pad=TITLE_PAD)
    leg = [Line2D([], [], color=CHANCE, lw=0.8, ls=(0, (1, 2)), label='Chance')]
    if wilcoxon_p is not None:
        leg.append(Line2D([], [], color='none', label=f'paired Wilcoxon p = {wilcoxon_p:.3f}'))
    A._top_legend(ax, leg, fontsize=6)


def panel_forest(ax, folds, ma, mb, la, lb, idx):
    keys = [k for k in FOREST_KEYS
            if not np.all(np.isnan([ma[f][k] for f in folds]))
            and not np.all(np.isnan([mb[f][k] for f in folds]))]
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        for off, m, color, dark in ((0.16, mb, PERM, PERM_DARK), (-0.16, ma, REAL, REAL_DARK)):
            v = np.array([m[f][k] for f in folds])
            lo, hi = boot_ci(v, idx)
            ax.plot([lo, hi], [yi + off] * 2, color=color, lw=1.2, zorder=3)
            ax.scatter([np.nanmean(v)], [yi + off], s=22, color=color, edgecolor='white',
                       linewidth=0.6, zorder=4)
            ax.text(np.nanmean(v), yi + off + 0.1, f'{np.nanmean(v):.2f}', ha='center', va='bottom',
                    fontsize=6.5, color=dark, zorder=5)
        ch = A.chance_level(k, 3) if k != 'cohen_kappa' else 0.0
        if ch is not None:
            ax.plot([ch] * 2, [yi - 0.38, yi + 0.38], color=CHANCE, lw=0.8, ls=(0, (2, 1.5)), zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([METRICS[k][0] for k in keys])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlabel('Score (mean over subjects)')
    ax.set_title('Summary metrics', loc='left', pad=TITLE_PAD)
    A._top_legend(ax, [
        Line2D([], [], marker='o', color=REAL, lw=1.2, markersize=4.5, label=f'{la} (±95% CI)'),
        Line2D([], [], marker='o', color=PERM, lw=1.2, markersize=4.5, label=f'{lb} (±95% CI)'),
        Line2D([], [], color=CHANCE, lw=0.8, ls=(0, (2, 1.5)), label='Chance'),
    ], fontsize=6, ncol=2)


def panel_calibration(ax, ca, cb, la, lb, n_classes):
    lo = 1.0 / n_classes
    ax.plot([lo, 1], [lo, 1], color=CHANCE, lw=0.8, ls=(0, (3, 2)), zorder=0, label='Perfect')
    edges = np.linspace(0, 1, A.CAL_BINS + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    strip = ax.inset_axes([0, -0.36, 1, 0.2], sharex=ax)
    shares = []
    for (y, probs, g), color, lbl, z in ((cb, PERM, lb, 1), (ca, REAL, la, 2)):
        if y is None or probs is None:
            continue
        _, xs, ys, ylo, yhi, tn = A._reliability(y, probs, g)
        ece = A.expected_calibration_error(y, probs)
        ax.fill_between(xs, ylo, yhi, color=color, alpha=0.2, linewidth=0, zorder=z - 0.5)
        ax.plot(xs, ys, color=color, lw=1.6, marker='o', markersize=3.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=z,
                label=f'{lbl} (ECE {ece:.3f})')
        shares.append((color, tn / tn.sum()))
    for k, (color, sh) in enumerate(shares):
        strip.bar(mids + (k - (len(shares) - 1) / 2) * 0.04, sh * 100, width=0.04,
                  color=color, alpha=0.85, linewidth=0)
    strip.set_ylabel('% samples', fontsize=6.5)
    strip.tick_params(labelsize=6)
    for sp in ('top', 'right'):
        strip.spines[sp].set_visible(False)
    ax.set_xlim(lo - 0.02, 1.01)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelbottom=False)
    strip.set_xlabel('Confidence (max softmax prob.)')
    ax.set_ylabel('Observed accuracy')
    ax.set_title('Calibration (95% CI over subjects)', loc='left', pad=TITLE_PAD)
    ax.legend(loc='lower right', borderaxespad=0.2)


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a_dir', required=True, help='glob of run dirs for condition A (blue), e.g. prompt_ln')
    ap.add_argument('--b_dir', required=True, help='glob of run dirs for condition B (orange), e.g. none')
    ap.add_argument('--a_label', default='prompt_ln')
    ap.add_argument('--b_label', default='none')
    ap.add_argument('--out', default='output/tpt_prompt_ln/tpt_prompt_ln_vs_none.png')
    ap.add_argument('--n_bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    la, lb = args.a_label, args.b_label

    runs_a, runs_b = load(args.a_dir), load(args.b_dir)
    if not runs_a or not runs_b:
        print(f'No runs found (A: {len(runs_a)}, B: {len(runs_b)}). Check the globs.')
        return 1
    folds = sorted(set(runs_a) & set(runs_b))
    missing = sorted(set(runs_a) ^ set(runs_b))
    if missing:
        print(f'Folds in only one condition (excluded): {missing}')
    print(f'{len(folds)} folds in both conditions')
    if len(folds) < 2:
        return 1

    ya, _, _, _ = pooled(runs_a, folds)
    n_classes = A.infer_class_names({f: {'real_preds': runs_a[f]['preds'], 'perm_preds': {}}
                                     for f in folds}).__len__()
    class_names = A.infer_class_names({f: {'real_preds': runs_a[f]['preds'], 'perm_preds': {}}
                                       for f in folds})
    ma = {f: A.compute_metrics(runs_a[f]['preds'], n_classes, runs_a[f]['acc']) for f in folds}
    mb = {f: A.compute_metrics(runs_b[f]['preds'], n_classes, runs_b[f]['acc']) for f in folds}
    subj = {f: (runs_a[f]['preds'] or {}).get('subject') for f in folds}
    metric = 'balanced_accuracy' if not np.all(np.isnan([ma[f]['balanced_accuracy'] for f in folds])) else 'accuracy'

    idx = np.random.default_rng(args.seed).integers(0, len(folds), size=(args.n_bootstrap, len(folds)))

    # paired subject-level test + console summary
    diff = np.array([ma[f][metric] - mb[f][metric] for f in folds])
    wp = None
    if A.HAS_SCIPY and np.any(diff != 0):
        wp = float(A.wilcoxon(diff).pvalue)
    d_boot = np.nanpercentile(np.nanmean(diff[idx], axis=1), [2.5, 97.5])
    print(f'\n{METRICS[metric][0]} (mean over subjects): {la} {np.nanmean([ma[f][metric] for f in folds]):.3f}, '
          f'{lb} {np.nanmean([mb[f][metric] for f in folds]):.3f}; '
          f'diff {diff.mean():+.3f} [95% CI {d_boot[0]:+.3f}, {d_boot[1]:+.3f}]; '
          f'{int((diff > 0).sum())}/{len(diff)} subjects better with {la}; '
          f'paired Wilcoxon p = {wp if wp is None else round(wp, 4)}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix('.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['fold', 'subject'] + [f'{k}_{c}' for c in (la, lb) for k in METRICS])
        for f in folds:
            w.writerow([f, subj[f] or ''] + [ma[f][k] for k in METRICS] + [mb[f][k] for k in METRICS])

    ya, yha, pa, ga = pooled(runs_a, folds)
    yb, _, pb, gb = pooled(runs_b, folds)

    A._style()
    panels = [
        lambda ax, fig: panel_per_subject(ax, folds, ma, mb, subj, la, lb, metric,
                                          A.chance_level(metric, n_classes)),
        lambda ax, fig: panel_kde(ax, folds, runs_a, runs_b, la, lb, n_classes, wp),
        lambda ax, fig: panel_forest(ax, folds, ma, mb, la, lb, idx),
    ]
    if ya is not None and A.HAS_SKLEARN:
        panels.append(lambda ax, fig: A.plot_confusion(ax, fig, ya, yha, class_names))
    if pa is not None and A.HAS_SKLEARN:
        panels.append(lambda ax, fig: A.plot_roc(ax, ya, pa, class_names))
        if pb is not None:
            panels.append(lambda ax, fig: panel_calibration(ax, (ya, pa, ga), (yb, pb, gb),
                                                            la, lb, n_classes))
    nrows, ncols = A._grid_shape(len(panels))
    fig = plt.figure(figsize=(7.2, 2.55 * nrows + 0.3))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.85, wspace=0.7,
                          left=0.08, right=0.97, top=0.93, bottom=0.08)
    for i, draw in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        draw(ax, fig)
        A._panel_letter(ax, 'abcdefgh'[i])
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    print(f'Saved {out.with_suffix(".png")} (+ .pdf, .svg, .csv)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

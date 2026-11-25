#!/usr/bin/env python
"""Paired comparison of LOOCV conditions (e.g. 28-TR vs 37-TR window) from saved test predictions.

Each fold holds out one subject, so every condition gives one accuracy per subject and the
conditions are paired by subject. Reads <output_root>/<run>/**/test_predictions_fold_*.json (the
newest file per fold if there are several) and writes, to --out_dir:
  window_compare_per_subject.csv   subject x condition accuracy
  window_compare.png/.pdf          left: paired per-subject accuracy; right: per-subject difference
  (console)                        mean +/- SEM, pooled accuracy / AUROC, paired Wilcoxon, bootstrap CI

  python analysis/compare_window_conditions.py \
      --condition "37 TR|FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-1[0-2][0-9]" \
      --condition "28 TR|FIXED_mich_xcpd_nosmooth_win28tr_tpt_strategy-none_loocv_fold-1[0-2][0-9]"

--condition is "label|glob" (glob relative to --output_root; use fold-1[0-2][0-9] so
_perm<K> permutation runs are not matched). The first condition is the reference; differences are
others minus reference. Folds missing in any condition are dropped from all (pairing).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

CLASSES = ['tennis', 'navigation', 'squeeze']


def load_condition(output_root, pattern):
    """-> DataFrame of block predictions with a 'fold' column (newest prediction file per fold)."""
    best = {}
    for run in sorted(glob.glob(os.path.join(output_root, pattern))):
        m = re.search(r'fold-(\d+)', os.path.basename(run))
        if not m:
            continue
        fold = int(m.group(1))
        files = glob.glob(os.path.join(run, '**', 'test_predictions_fold_*.json'), recursive=True)
        if files:
            best[fold] = max(files, key=os.path.getmtime)
    rows = []
    for fold, f in sorted(best.items()):
        with open(f) as fh:
            for r in json.load(fh):
                rows.append(dict(fold=fold, subject=r['subject'], y=r['true_label'],
                                 pred=r['predicted_label'], probs=r['softmax_probs']))
    return pd.DataFrame(rows)


def per_subject_acc(df):
    return df.assign(ok=(df['y'] == df['pred'])).groupby('subject')['ok'].mean()


def pooled(df):
    from sklearn.metrics import roc_auc_score
    p = np.array(df['probs'].tolist())
    return (df['y'] == df['pred']).mean(), roc_auc_score(df['y'], p, multi_class='ovr', average='macro')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--condition', action='append', required=True, dest='conds', help='"label|glob" (>=2, first = reference)')
    ap.add_argument('--output_root', default='/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output')
    ap.add_argument('--out_dir', default='output/window_sweep')
    ap.add_argument('--n_boot', type=int, default=10000)
    args = ap.parse_args()

    data = {}
    for c in args.conds:
        label, pattern = c.split('|', 1)
        df = load_condition(args.output_root, pattern)
        if df.empty:
            raise SystemExit(f'no predictions found for "{label}" ({pattern})')
        data[label] = df
        print(f'{label}: {df["fold"].nunique()} folds, {len(df)} blocks')

    common = sorted(set.intersection(*[set(d['subject']) for d in data.values()]))
    for k in data:
        data[k] = data[k][data[k]['subject'].isin(common)]
    labels = list(data)
    acc = pd.DataFrame({k: per_subject_acc(data[k]).loc[common] for k in labels})
    print(f'\n{len(common)} subjects in every condition (blocks per subject: '
          f'{sorted(set(data[labels[0]].groupby("subject").size()))})')

    rng = np.random.default_rng(0)

    def boot_ci(v):
        b = [rng.choice(v, len(v)).mean() for _ in range(args.n_boot)]
        return np.percentile(b, [2.5, 97.5])

    print(f'\n{"condition":<14}{"mean subj acc":>16}{"95% CI":>18}{"pooled acc":>12}{"pooled AUROC":>14}')
    for k in labels:
        a = acc[k].values
        lo, hi = boot_ci(a)
        pa, pu = pooled(data[k])
        print(f'{k:<14}{a.mean():>16.3f}{f"[{lo:.3f}, {hi:.3f}]":>18}{pa:>12.3f}{pu:>14.3f}')

    from scipy.stats import wilcoxon
    ref = labels[0]
    stats = {}
    for k in labels[1:]:
        d = (acc[k] - acc[ref]).values
        lo, hi = boot_ci(d)
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float('nan')
        stats[k] = dict(mean=d.mean(), lo=lo, hi=hi, p=p)
        print(f'\n{k} - {ref}: mean diff {d.mean():+.3f} (95% bootstrap CI {lo:+.3f} to {hi:+.3f}), '
              f'Wilcoxon p={p:.3f}; better in {(d > 0).sum()}, worse in {(d < 0).sum()}, tied {(d == 0).sum()} subjects')

    os.makedirs(args.out_dir, exist_ok=True)
    acc.to_csv(os.path.join(args.out_dir, 'window_compare_per_subject.csv'))
    plot(acc, labels, stats, boot_ci, args.out_dir)


def stack_offsets(values, step):
    """Offsets that spread identical (accuracy is on a 1/30 grid) values side by side so no subject is hidden."""
    off = np.zeros(len(values))
    keys = np.round(np.asarray(values), 6)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        off[idx] = (np.arange(len(idx)) - (len(idx) - 1) / 2) * step
    return off


def plot(acc, labels, stats, boot_ci, out_dir):
    """Two panels: (a) every subject's accuracy per window; (b) paired per-subject difference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    INK, MUTED, GRID = '#1f2937', '#6b7280', '#e5e7eb'
    COL = ['#8a94a6', '#1f5fa8']       # reference, comparison
    UP, DOWN = '#1f5fa8', '#c8601a'    # better / worse than reference
    ref, k = labels[0], labels[1]
    st = stats[k]
    d = (acc[k] - acc[ref]).values
    n = len(d)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1, 1.35]))
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.2, top=0.88, wspace=0.32)

    # (a) accuracy per window
    for i, lab in enumerate((ref, k)):
        v = acc[lab].values
        a.scatter(i + stack_offsets(v, 0.045), v, s=16, color=COL[i], alpha=0.85, edgecolor='white', lw=0.4, zorder=2)
        lo, hi = boot_ci(v)
        a.plot([i + 0.34] * 2, [lo, hi], color=INK, lw=1.6, solid_capstyle='butt', zorder=3)
        a.scatter([i + 0.34], [v.mean()], s=34, color='white', edgecolor=INK, lw=1.3, zorder=4)
        a.text(i + 0.43, v.mean(), f'{v.mean():.3f}', va='center', ha='left', fontsize=8, color=INK)
    a.axhline(1 / 3, color=MUTED, ls=(0, (3, 3)), lw=0.8, zorder=1)
    a.text(1.78, 1 / 3 + 0.012, 'chance', color=MUTED, fontsize=7, ha='right', va='bottom')
    a.set_xlim(-0.55, 1.85)
    a.set_ylim(0.25, 1.02)
    a.set_xticks([0, 1])
    a.set_xticklabels([f'{ref} window', f'{k} window'])
    a.set_ylabel('Accuracy per held-out subject')
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)

    # (b) paired difference, one dot per subject
    step = 1 / 30
    ys = np.zeros(n)
    keys = np.round(d, 6)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        ys[idx] = np.arange(len(idx))
    ymax = ys.max() + 1
    b.axvline(0, color=INK, lw=0.8, zorder=1)
    b.scatter(d, ys + 0.5, s=30, color=[UP if v > 0 else (DOWN if v < 0 else MUTED) for v in d],
              edgecolor='white', lw=0.5, zorder=3)
    yb = -1.6
    b.plot([st['lo'], st['hi']], [yb, yb], color=INK, lw=2, solid_capstyle='butt', zorder=3)
    b.scatter([st['mean']], [yb], s=46, color='white', edgecolor=INK, lw=1.5, zorder=4)
    ptxt = f"p = {st['p']:.2f}" if st['p'] >= 0.01 else 'p < 0.01'
    b.text(0.98, 0.97, f"mean {st['mean']:+.3f}\n95% CI [{st['lo']:+.3f}, {st['hi']:+.3f}]\nWilcoxon {ptxt}",
           transform=b.transAxes, ha='right', va='top', fontsize=7.5, color=INK, linespacing=1.4)
    lim = max(abs(d).max(), abs(st['lo']), abs(st['hi'])) + step
    b.set_xlim(-lim, lim)
    b.set_ylim(-3.4, ymax + 3.0)
    b.set_yticks([])
    b.spines['left'].set_visible(False)
    b.set_xlabel(f'Accuracy difference, {k} \u2212 {ref} (per subject)')
    b.text(-lim, -3.0, f'\u2190 {ref} better', color=DOWN, fontsize=7.5, ha='left', va='center')
    b.text(lim, -3.0, f'{k} better \u2192', color=UP, fontsize=7.5, ha='right', va='center')
    b.text(st['hi'] + 0.008, yb, '95% CI', color=MUTED, fontsize=7, va='center', ha='left')

    for ax in (a, b):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=0.8, length=3)
    a.text(-0.2, 1.06, 'a', transform=a.transAxes, fontsize=11, fontweight='bold')
    b.text(-0.07, 1.06, 'b', transform=b.transAxes, fontsize=11, fontweight='bold')
    fig.text(0.5, 0.005, f'Leave-one-subject-out, n = {n} subjects, 30 blocks each.\n'
             'Dots are single subjects (identical values placed side by side). Open circles: mean with 95% bootstrap CI.',
             ha='center', va='bottom', fontsize=6.5, color=MUTED)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(out_dir, f'window_compare.{ext}'), dpi=300)
    print(f'saved {out_dir}/window_compare.png/.pdf/.svg')


if __name__ == '__main__':
    main()

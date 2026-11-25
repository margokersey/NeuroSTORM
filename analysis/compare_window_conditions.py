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
    print(f'\n{"condition":<14}{"mean subj acc":>16}{"pooled acc":>12}{"pooled AUROC":>14}')
    for k in labels:
        a = acc[k]
        pa, pu = pooled(data[k])
        print(f'{k:<14}{a.mean():>9.3f} +/- {a.std(ddof=1) / np.sqrt(len(a)):.3f}{pa:>12.3f}{pu:>14.3f}')

    from scipy.stats import wilcoxon
    ref = labels[0]
    for k in labels[1:]:
        d = (acc[k] - acc[ref]).values
        boots = [rng.choice(d, len(d)).mean() for _ in range(args.n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float('nan')
        print(f'\n{k} - {ref}: mean diff {d.mean():+.3f} (95% bootstrap CI {lo:+.3f} to {hi:+.3f}), '
              f'Wilcoxon p={p:.3f}; better in {(d > 0).sum()}, worse in {(d < 0).sum()}, tied {(d == 0).sum()} subjects')

    os.makedirs(args.out_dir, exist_ok=True)
    acc.to_csv(os.path.join(args.out_dir, 'window_compare_per_subject.csv'))
    plot(acc, labels, args.out_dir)


def plot(acc, labels, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#6b7280', '#08519c', '#c2410c', '#15803d']
    ref = labels[0]
    others = labels[1:]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[1.1, 1]))
    x = np.arange(len(labels))
    for subj, row in acc.iterrows():
        a1.plot(x, row[labels].values, color='#9ca3af', lw=0.8, alpha=0.7, zorder=1)
    for i, k in enumerate(labels):
        a1.scatter(np.full(len(acc), i), acc[k], s=22, color=colors[i % 4], alpha=0.8, zorder=2, edgecolor='white', lw=0.5)
        m, s = acc[k].mean(), acc[k].std(ddof=1) / np.sqrt(len(acc))
        a1.errorbar(i, m, yerr=s, color='black', marker='D', ms=8, capsize=4, lw=2, zorder=3, mfc='white')
        a1.text(i + 0.08, m, f'{m:.3f}', va='center', fontsize=10)
    a1.axhline(1 / 3, color='#6b7280', ls='--', lw=1)
    a1.text(len(labels) - 0.55, 1 / 3 + 0.01, 'chance', color='#6b7280', fontsize=9, ha='right')
    a1.set_xticks(x)
    a1.set_xticklabels(labels)
    a1.set_xlim(-0.4, len(labels) - 0.4)
    a1.set_ylim(0, 1.02)
    a1.set_ylabel('Held-out subject accuracy (30 blocks each)')
    a1.set_title('Per subject (lines pair the same subject); diamond = mean +/- SEM', fontsize=10, loc='left')

    k = others[0]
    d = (acc[k] - acc[ref]).sort_values()
    a2.axvline(0, color='#6b7280', lw=1)
    a2.barh(np.arange(len(d)), d.values, color=[colors[1] if v >= 0 else colors[2] for v in d.values], height=0.7)
    a2.set_yticks(np.arange(len(d)))
    a2.set_yticklabels(d.index, fontsize=7)
    a2.axvline(d.mean(), color='black', ls='--', lw=1.5)
    a2.set_xlabel(f'Accuracy difference: {k} minus {ref}')
    a2.set_title(f'Per-subject difference; dashed = mean {d.mean():+.3f}', fontsize=10, loc='left')
    for ax in (a1, a2):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'window_compare.{ext}'), dpi=200)
    print(f'saved {out_dir}/window_compare.png')


if __name__ == '__main__':
    main()

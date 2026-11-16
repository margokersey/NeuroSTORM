#!/usr/bin/env python
"""Plot 3-way task-decoding accuracy vs head motion, one panel per imagery run (1-4).

Per held-out subject and run: block accuracy of the fold's 3-way model (run 1 = its own held-out
test predictions; runs 2/3 = lor_predictions_fold_<N>.json; run 4 = lor_predictions_run4_fold_<N>.json)
against a run-level motion metric computed from the XCP-D motion.tsv / outliers.tsv of that run.
Each panel shows Spearman rho (+ 95% bootstrap CI) and p. Runs 2/3 mix pre-LOR / LOR / ROR blocks, so
accuracy there is over all blocks of the run (use --drop_boundary to exclude boundary blocks).

Motion metrics (--metric):
  pct_fd_gt_0.3   % volumes with framewise displacement > 0.3 mm        (default; spread out in every run)
  pct_censored    % volumes flagged in outliers.tsv (FD >= 0.8 mm for this pipeline)
  median_fd       median FD (mm)
Mean FD is deliberately not offered: single extreme volumes (max FD up to ~1000 mm in run 3) dominate it.

    python analysis/plot_lor_accuracy_vs_motion.py \\
        --pred_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_lor_task_decoding_fold-*/lor_predictions_fold_*.json" \\
        --output_file output/lor_task_decoding/lor_accuracy_vs_motion

Writes .png (600 dpi), .pdf, .svg and a per-subject/run .csv.
"""
import argparse
import csv
import importlib.util
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_lor_task_decoding as P  # noqa: E402  (style S, plt, load_run1, boot_ci helpers)
S, plt = P.S, P.plt

DERIV = ('/nfs/turbo/umms-huanglab/pipeline/postproc/fmriprep-25.2.5_xcp_d-26.1.1/'
         'Michigan_Dataset_1/noGSR_highpass_fd-0.8_nosmooth')
RUN_COLORS = {1: P.SEG_COLORS['run1'], 2: '#4C72B0', 3: '#C44E52', 4: P.SEG_COLORS['recovery']}
RUN_TITLES = {1: 'Run 1 (awake baseline)', 2: 'Run 2', 3: 'Run 3', 4: 'Run 4 (recovery)'}
METRICS = {'pct_fd_gt_0.3': '% volumes with FD > 0.3 mm', 'pct_censored': '% volumes censored (FD ≥ 0.8 mm)',
           'median_fd': 'Median framewise displacement (mm)'}


def motion_metric(deriv, sub, run, metric, task='imagery'):
    base = Path(deriv) / sub / 'func' / f'{sub}_task-{task}_run-{run:02d}'
    if metric == 'pct_censored':
        f = Path(str(base) + '_outliers.tsv')
        return float(100 * (pd.read_csv(f, sep='\t').iloc[:, 0] > 0).mean()) if f.exists() else np.nan
    f = Path(str(base) + '_motion.tsv')
    if not f.exists():
        return np.nan
    fd = pd.read_csv(f, sep='\t')['framewise_displacement'].dropna()
    return float(100 * (fd > 0.3).mean()) if metric == 'pct_fd_gt_0.3' else float(fd.median())


def acc_of(preds, drop_boundary):
    if drop_boundary:
        preds = [p for p in preds if not str(p.get('segment', '')).startswith('boundary')]
    return (float(np.mean([p['predicted_label'] == p['true_label'] for p in preds])), len(preds)) if preds else (np.nan, 0)


def spearman_boot(x, y, n=5000, seed=0):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x))
    b = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        if np.ptp(x[i]) and np.ptp(y[i]):
            b.append(spearmanr(x[i], y[i])[0])
    return rho, p, np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred_glob', required=True)
    ap.add_argument('--run4_glob', default=None, help='default: derived from --pred_glob')
    ap.add_argument('--output_file', default='output/lor_task_decoding/lor_accuracy_vs_motion')
    ap.add_argument('--output_root', default='/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output')
    ap.add_argument('--stage_b_prefix', default=P.STAGE_B_PREFIX)
    ap.add_argument('--deriv_dir', default=DERIV, help='XCP-D derivatives (sub-*/func/*_motion.tsv)')
    ap.add_argument('--metric', choices=list(METRICS), default='pct_fd_gt_0.3')
    ap.add_argument('--min_blocks', type=int, default=5)
    ap.add_argument('--drop_boundary', action='store_true')
    ap.add_argument('--exclude_subjects', nargs='*', default=[])
    ap.add_argument('--no_labels', action='store_true', help='do not label points with subject ids')
    args = ap.parse_args()

    files = sorted(glob(args.pred_glob))
    if not files:
        sys.exit(f'no files match {args.pred_glob}')
    run4_glob = args.run4_glob or args.pred_glob.replace('lor_predictions_fold_', 'lor_predictions_run4_fold_')
    run4 = {}
    for f in sorted(glob(run4_glob)):
        d4 = json.load(open(f))
        run4[d4['meta']['fold']] = d4

    rows = []
    for f in files:
        d = json.load(open(f))
        sub, fold = d['meta']['subjects'][0], d['meta']['fold']
        if sub in args.exclude_subjects:
            continue
        per_run = {}
        r1, _ = P.load_run1(args.output_root, args.stage_b_prefix, fold)
        if r1 is not None and {p['subject'] for p in r1} == {sub}:
            per_run[1] = r1
        for run in (2, 3):
            per_run[run] = [p for p in d['predictions'] if int(p['run']) == run]
        d4 = run4.get(fold)
        if d4 is not None and d4['meta']['subjects'] == [sub]:
            per_run[4] = d4['predictions']
        for run, ps in per_run.items():
            acc, n = acc_of(ps, args.drop_boundary)
            rows.append({'subject': sub, 'fold': fold, 'run': run, 'n_blocks': n, 'accuracy': acc,
                         args.metric: motion_metric(args.deriv_dir, sub, run, args.metric)})
    df = pd.DataFrame(rows)
    df = df[(df.n_blocks >= args.min_blocks) & df.accuracy.notna() & df[args.metric].notna()]

    S._style()
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9), sharey=True)
    stats = []
    for ax, run in zip(axes, (1, 2, 3, 4)):
        g = df[df.run == run]
        ax.scatter(g[args.metric], g.accuracy, s=16, color=RUN_COLORS[run], alpha=0.8, edgecolor='white',
                   linewidth=0.4, zorder=3, clip_on=False)
        if not args.no_labels:
            for _, r in g.iterrows():
                ax.annotate(r.subject.replace('sub-', ''), (r[args.metric], r.accuracy), fontsize=4.5,
                            color=S.INK_2, xytext=(2, 2), textcoords='offset points')
        ax.axhline(P.CHANCE, color=S.CHANCE, ls='--', lw=0.8, zorder=0)
        ax.set_title(RUN_TITLES[run], fontsize=8)
        ax.set_ylim(0, 1.03)
        if len(g) >= 4 and g[args.metric].nunique() > 1:
            rho, p, (lo, hi) = spearman_boot(g[args.metric].values, g.accuracy.values)
            stats.append((run, len(g), rho, p, lo, hi))
            ax.text(0.03, 0.04, f'ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]\n{P.fmt_p(p)}, n = {len(g)}',
                    transform=ax.transAxes, fontsize=6.5, color=S.INK_2, va='bottom')
        ax.set_xlim(left=-0.02 * max(1.0, g[args.metric].max() if len(g) else 1))
    axes[0].set_ylabel('Block accuracy (3-way task decoding)')
    fig.supxlabel(METRICS[args.metric], fontsize=8)
    fig.suptitle('Task-decoding accuracy vs head motion', fontsize=9, y=1.02)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white', dpi=600)
    plt.close(fig)
    df.to_csv(out.with_suffix('.csv'), index=False)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg, .csv)")
    for run, n, rho, p, lo, hi in stats:
        print(f'  run {run}: n={n} Spearman rho={rho:.2f} [{lo:.2f}, {hi:.2f}] {P.fmt_p(p)}')


if __name__ == '__main__':
    main()

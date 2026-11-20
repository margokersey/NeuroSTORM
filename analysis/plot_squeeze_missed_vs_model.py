#!/usr/bin/env python
"""Are missed / weak squeezes the squeeze blocks the 3-way model got wrong? One panel per imagery run.

Joins analysis/classify_squeeze_force.py's per-block CSV with the 3-way (tennis / navigation / squeeze)
LOOCV model's predictions on the held-out subject's squeeze blocks (true_label == 2):
  run 1     fold's own saved test predictions (<output_root>/<prefix>_fold-N/lightning_logs/version_*/
            test_predictions/test_predictions_fold_*.json). They carry no block onset, so the k-th
            true-squeeze sample is the k-th squeeze block of run 1 (checked: must be 10 per subject).
  runs 2-4  lor_predictions_fold_N.json (runs 2, 3) and lor_predictions_run4_fold_N.json (run 4) from
            analysis/predict_lor_task_decoding.py; matched on run + block onset TR (`start`).

Figure: for each run, % of squeeze blocks classified correctly by response category (squeezed,
weak squeeze, early squeeze, missed) with Wilson 95% CIs and n; dashed line = chance (1/3). The
annotation is a Fisher exact test of missed vs squeezed (weak included in squeezed) pooled over subjects
in that run. Blocks are pooled over subjects (subject-level clustering is NOT modelled; the CSV lets you
do that). Also writes <out>.csv (one row per squeeze block with model correctness) and <out>_summary.csv.

    python analysis/plot_squeeze_missed_vs_model.py \\
      --output_root /nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output \\
      --out output/squeeze_force/squeeze_missed_vs_model
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
STAGE_B_PREFIX = 'FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv'   # 3-way LOOCV runs
LOR_PREFIX = 'FIXED_mich_xcpd_nosmooth_lor_task_decoding'
SQUEEZE = 2
CATS = [('squeezed', 'Squeezed\n(normal)', '#55A868'), ('weak', 'Squeezed\n(weak)', '#9ad0a6'),
        ('early', 'Early\nsqueeze', '#E69F00'), ('missed', 'Missed', '#C44E52')]


def subj_id(s):
    m = re.search(r'(\d+)', str(s))
    return f'P{int(m.group(1)):02d}'


def latest_test_predictions(fold_dir):
    best = None
    for d in glob.glob(os.path.join(fold_dir, 'lightning_logs', 'version_*', 'test_predictions')):
        try:
            v = int(Path(d).parent.name.split('_')[-1])
        except ValueError:
            continue
        files = sorted(glob.glob(os.path.join(d, 'test_predictions_fold_*.json')))
        if files and (best is None or v > best[0]):
            best = (v, files[0])
    return json.load(open(best[1])) if best else None


def collect_model_rows(output_root, prefix, lor_prefix):
    """-> DataFrame subject, run, key (block onset TR, or squeeze-block index for run 1), correct, pred."""
    rows = []
    for fd in sorted(glob.glob(os.path.join(output_root, f'{prefix}_fold-*'))):
        preds = latest_test_predictions(fd)
        if not preds:
            continue
        subs = {subj_id(p['subject']) for p in preds}
        sq = [p for p in preds if int(p['true_label']) == SQUEEZE]
        if len(subs) != 1:
            print(f'WARNING {fd}: {len(subs)} subjects in test predictions; skipped')
            continue
        if len(sq) != 10:
            print(f'WARNING {fd}: {len(sq)} true-squeeze samples (expected 10); run 1 skipped for this fold')
            continue
        for k, p in enumerate(sq, 1):
            rows.append(dict(subject=subs.copy().pop(), run=1, squeeze_block=k, pred=int(p['predicted_label']),
                             correct=int(p['predicted_label']) == SQUEEZE))
    for pat in (f'{lor_prefix}_fold-*/lor_predictions_fold_*.json', f'{lor_prefix}_fold-*/lor_predictions_run4_fold_*.json'):
        for f in sorted(glob.glob(os.path.join(output_root, pat))):
            d = json.load(open(f))
            for p in d['predictions']:
                if int(p['true_label']) != SQUEEZE:
                    continue
                rows.append(dict(subject=subj_id(p['subject']), run=int(p['run']), start=int(p['start']),
                                 pred=int(p['predicted_label']), correct=int(p['predicted_label']) == SQUEEZE))
    return pd.DataFrame(rows)


def join(force, model):
    m1 = model[model.run == 1].drop(columns=['start'], errors='ignore')
    f1 = force[force.run == 1].merge(m1, on=['subject', 'run', 'squeeze_block'], how='left')
    mr = model[model.run != 1].drop(columns=['squeeze_block'], errors='ignore')
    fr = force[force.run != 1].copy()
    fr['start'] = fr.imagery_start_tr
    fr = fr.merge(mr, on=['subject', 'run', 'start'], how='left')
    return pd.concat([f1, fr], ignore_index=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def assign_cat(r):
    if r.category == 'missed':
        return 'missed'
    if r.category == 'early':
        return 'early'
    return 'weak' if r.weak else 'squeezed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(REPO / 'output/squeeze_force/squeeze_block_response.csv'))
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--prefix', default=STAGE_B_PREFIX)
    ap.add_argument('--lor_prefix', default=LOR_PREFIX)
    ap.add_argument('--out', default=str(REPO / 'output/squeeze_force/squeeze_missed_vs_model'))
    a = ap.parse_args()

    force = pd.read_csv(a.csv)
    force = force[force.category != 'no_data']
    model = collect_model_rows(a.output_root, a.prefix, a.lor_prefix)
    if model.empty:
        raise SystemExit('no model predictions found')
    df = join(force, model)
    df['cat'] = df.apply(assign_cat, axis=1)
    lost = df[df.correct.isna()]
    print(f'{len(lost)} of {len(df)} squeeze blocks have no model prediction (subject/run not predicted, '
          f'e.g. P29 run 3/4, or block skipped); dropped')
    print(lost.groupby(['subject', 'run']).size().to_string() if len(lost) else '')
    df = df[df.correct.notna()].copy()
    df['correct'] = df.correct.astype(bool)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=['start'], errors='ignore').to_csv(out.with_suffix('.csv'), index=False)

    try:
        from scipy.stats import fisher_exact
    except ImportError:
        fisher_exact = None
    runs = sorted(df.run.unique())
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    summ = []
    for ax, r in zip(axes, runs):
        d = df[df.run == r]
        for i, (c, label, col) in enumerate(CATS):
            dc = d[d.cat == c]
            n, k = len(dc), int(dc.correct.sum())
            summ.append(dict(run=r, category=c, n=n, n_correct=k, acc=k / n if n else np.nan))
            if n == 0:
                ax.text(i, 0.03, 'n=0', ha='center', fontsize=8, color='#888888')
                continue
            lo, hi = wilson(k, n)
            ax.bar(i, k / n, color=col, width=0.7)
            ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color='k', capsize=3, lw=1)
            ax.text(i, 1.04, f'{k}/{n}', ha='center', fontsize=8)
        ax.axhline(1 / 3, ls='--', color='#555555', lw=1)
        ax.set_xticks(range(len(CATS)))
        ax.set_xticklabels([c[1] for c in CATS], fontsize=8)
        ax.set_ylim(0, 1.12)
        title = f'Run {r}'
        if fisher_exact is not None:
            resp, miss = d[d.cat != 'missed'], d[d.cat == 'missed']
            if len(resp) and len(miss):
                _, p = fisher_exact([[int(resp.correct.sum()), int((~resp.correct).sum())],
                                     [int(miss.correct.sum()), int((~miss.correct).sum())]])
                title += f'\nmissed vs responded: Fisher p={p:.3g}'
        ax.set_title(title, fontsize=10)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('Squeeze blocks classified as squeeze (%)')
    axes[0].set_yticks([0, .25, .5, .75, 1])
    axes[0].set_yticklabels(['0', '25', '50', '75', '100'])
    fig.suptitle('3-way model accuracy on squeeze blocks, by what the participant did (correct/total above bars;\n'
                 'dashed = chance 33%; bars: Wilson 95% CI; blocks pooled over subjects)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), dpi=200 if ext == 'png' else None)
    pd.DataFrame(summ).to_csv(f'{out}_summary.csv', index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print(f'saved {out}.png/.pdf/.csv')


if __name__ == '__main__':
    main()

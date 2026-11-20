#!/usr/bin/env python
"""Are missed / weak squeezes the squeeze blocks the 3-way model got wrong? One panel per imagery run.

Joins analysis/classify_squeeze_force.py's per-block CSV with the 3-way (tennis / navigation / squeeze)
LOOCV model's predictions on the held-out subject's squeeze blocks (true_label == 2):
  run 1     fold's own saved test predictions (<output_root>/<prefix>_fold-N/lightning_logs/version_*/
            test_predictions/test_predictions_fold_*.json). They carry no block onset, so the k-th
            true-squeeze sample is the k-th squeeze block of run 1 (checked: must be 10 per subject).
  runs 2-4  lor_predictions_fold_N.json (runs 2, 3) and lor_predictions_run4_fold_N.json (run 4) from
            analysis/predict_lor_task_decoding.py; matched on run + block onset TR (`start`).

Figure, one row per run. Left: stacked bars of what the participant did (squeezed / weak / early /
missed) for ALL squeeze blocks vs only the blocks the model got WRONG; if the error bar is much redder
than the all-blocks bar, errors coincide with missed squeezes. Right: model error rate within each
response category (Wilson 95% CI), dashed = overall error rate. The annotation is a Fisher exact test of missed vs squeezed (weak included in squeezed) pooled over subjects
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
    """Attach model rows to force rows: run 1 on (subject, squeeze_block), runs 2-4 on (subject, run, start).
    Works when the model table has only run-1 rows (e.g. runs 2-4 not predicted yet)."""
    model = model.copy()
    for c in ('start', 'squeeze_block'):
        if c not in model.columns:
            model[c] = np.nan
        model[c] = model[c].astype('Int64')
    m1 = model[model.run == 1].drop(columns=['start'])
    f1 = force[force.run == 1].copy()
    f1['squeeze_block'] = f1.squeeze_block.astype('Int64')
    f1 = f1.merge(m1, on=['subject', 'run', 'squeeze_block'], how='left')
    mr = model[model.run != 1].drop(columns=['squeeze_block'])
    fr = force[force.run != 1].copy()
    fr['start'] = fr.imagery_start_tr.astype('Int64')
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


def draw_breakdown(df, bad, out, noun_bar, noun_title, rate_title, suptitle):
    """One row per run. Left: participant behaviour (squeezed / weak / early / missed) for ALL squeeze
    blocks vs only the blocks where boolean column `bad` is True. Right: share of each behaviour
    category with bad=True (Wilson 95% CI). df needs columns run, cat, <bad>."""
    out = Path(out)
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        fisher_exact = None
    runs = sorted(df.run.unique())
    order = ['squeezed', 'weak', 'early', 'missed']
    names = {'squeezed': 'Squeezed normally', 'weak': 'Weak squeeze', 'early': 'Squeezed at wrong time\n(early)', 'missed': "Didn't squeeze\n(missed)"}
    col = {c: k for c, _, k in CATS}
    # margins in inches so the title/legend never collide with the rows, whatever the number of runs
    top_in, bot_in, row_in = 1.5, 0.9, 3.2
    H = top_in + bot_in + row_in * len(runs)
    fig, axes = plt.subplots(len(runs), 2, figsize=(15, H), squeeze=False,
                             gridspec_kw={'width_ratios': [1.5, 1], 'wspace': 0.3, 'hspace': 0.85})
    summ = []
    for i, r in enumerate(runs):
        d = df[df.run == r]
        err = d[d[bad]]
        axL, axR = axes[i]
        # left: what was the participant doing, for all squeeze blocks vs the blocks the model got wrong
        for yy, sub_, lab in ((1, d, f'All squeeze blocks\n(n={len(d)})'), (0, err, f'{noun_bar}\n(n={len(err)})')):
            left = 0.0
            for c in order:
                n = int((sub_.cat == c).sum())
                w = n / len(sub_) if len(sub_) else 0
                axL.barh(yy, w, left=left, color=col[c], height=0.62, edgecolor='white')
                if w >= 0.06:
                    axL.text(left + w / 2, yy, f'{n}\n{100 * w:.0f}%', ha='center', va='center', fontsize=8,
                             color='white' if c in ('missed', 'squeezed') else 'k', fontweight='bold')
                left += w
            axL.text(-0.01, yy, lab, ha='right', va='center', fontsize=9)
        axL.set_xlim(0, 1)
        axL.set_ylim(-0.5, 1.5)
        axL.axis('off')
        n_err_missed = int((err.cat == 'missed').sum())
        axL.set_title(f'Run {r}: {n_err_missed} of {len(err)} {noun_title}\n'
                      f'({100 * n_err_missed / max(len(err), 1):.0f}%) were blocks where the squeeze was missed', fontsize=10, loc='left', fontweight='bold')
        # right: how often is the model wrong, per participant behaviour
        for j, c in enumerate(order):
            dc = d[d.cat == c]
            n, k = len(dc), int(dc[bad].sum())
            summ.append(dict(run=r, category=c, n=n, n_bad=k, rate=k / n if n else np.nan))
            if n == 0:
                axR.text(0.01, j, f'{names[c]}: n=0', va='center', fontsize=8, color='#888888')
                continue
            lo, hi = wilson(k, n)
            axR.barh(j, k / n, color=col[c], height=0.6)
            axR.errorbar(k / n, j, xerr=[[k / n - lo], [hi - k / n]], color='k', capsize=2.5, lw=1)
            axR.text(1.02, j, f'{k}/{n}', va='center', fontsize=8)
        overall = d[bad].mean()
        axR.axvline(overall, color='#444444', ls='--', lw=1)
        axR.set_yticks(range(len(order)))
        axR.set_yticklabels([names[c] for c in order], fontsize=8)
        axR.set_ylim(len(order) - 0.4, -0.6)
        axR.set_xlim(0, 1.12)
        axR.set_xticks([0, .25, .5, .75, 1])
        axR.set_xticklabels(['0', '25', '50', '75', '100%'], fontsize=8)
        ttl = f'{rate_title} (dashed = all blocks, {100 * overall:.0f}%)'
        if fisher_exact is not None:
            resp, miss = d[d.cat != 'missed'], d[d.cat == 'missed']
            if len(resp) and len(miss):
                _, p = fisher_exact([[int(resp[bad].sum()), int((~resp[bad]).sum())],
                                     [int(miss[bad].sum()), int((~miss[bad]).sum())]])
                ttl += f'\nmissed vs responded: Fisher p={p:.2g}'
        axR.set_title(ttl, fontsize=8.5, loc='left')
        for sp in ('top', 'right'):
            axR.spines[sp].set_visible(False)
    fig.suptitle(suptitle, fontsize=10.5, y=1 - 0.12 / H, va='top')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col[c], label=names[c].replace('\n', ' ')) for c in order],
               loc='lower center', bbox_to_anchor=(0.5, 0.2 / H), ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.95, top=1 - top_in / H, bottom=bot_in / H)
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), dpi=200 if ext == 'png' else None)
    pd.DataFrame(summ).to_csv(f'{out}_summary.csv', index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print(f'saved {out}.png/.pdf')
    plt.close(fig)


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

    draw_breakdown(df.assign(bad=~df.correct), 'bad', out, 'Model errors', 'model errors',
                   'Model error rate by what the participant did',
                   '3-way model errors on squeeze blocks: do they coincide with missed / weak / early squeezes?\n'
                   'Left: if errors are mostly red, missed squeezes explain them (compare with the "all blocks" bar).\n'
                   'Right: % of each kind of block the model got wrong (bars: Wilson 95% CI).')



if __name__ == '__main__':
    main()

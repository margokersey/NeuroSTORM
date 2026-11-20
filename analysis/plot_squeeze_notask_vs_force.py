#!/usr/bin/env python
"""Squeeze blocks the Stage A model called "No Task": what was the participant doing on them?

Stage A = the binary Task-vs-No-Task LOOCV model (run_loocv_taskvsrest.sh). The 3-way model has no "no task"
class, so this uses Stage A's per-block call on the held-out subject's squeeze blocks; "misclassified as
no task" = predicted_label 0 (1 = task). Joined with analysis/classify_squeeze_force.py's per-block CSV.

Figure (same layout as plot_squeeze_missed_vs_model.py), one row per imagery run:
  left : participant behaviour (squeezed normally / weak / early = squeezed at the wrong time / missed =
         didn't squeeze) for ALL squeeze blocks vs only the blocks Stage A called No Task.
  right: % of each behaviour category that Stage A called No Task (Wilson 95% CI).

Where predictions come from:
  run 1   Stage A's own saved test predictions (<output_root>/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-N/
          lightning_logs/version_*/test_predictions). They have no block onset, so the true-Task samples
          (label 1), in order, are matched to the 30 task blocks of timing_onehot_task1.tsv (the order two_stage_
          decoding.py verifies); the k-th squeeze block of that list is squeeze block k.
  runs 2-4  lor_predictions*_fold_N.json from `predict_lor_task_decoding.py --task taskvsrest`
          (LOR_TASK=taskvsrest sbatch analysis/sbatch_lor_task_decoding.sh; LOR_RUNS=4 for run 4), matched on
          run + block onset TR. Not run yet: until those exist the figure only has run 1.

    python analysis/plot_squeeze_notask_vs_force.py \\
      --output_root /nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plot_squeeze_missed_vs_model import assign_cat, draw_breakdown, join, latest_test_predictions, subj_id

REPO = Path(__file__).resolve().parent.parent
STAGE_A_PREFIX = 'FIXED_mich_xcpd_nosmooth_taskvsrest_loocv'
LOR_STAGE_A_PREFIX = 'FIXED_mich_xcpd_nosmooth_lor_taskvsrest_decoding'
SQUEEZE = 2


def run1_task_block_kinds(tsv):
    """Kinds ('tennis'/'navigation'/'squeeze') of the task blocks of run 1, in time order."""
    d = pd.read_csv(tsv, sep='\t')
    lab = d.values.argmax(1)
    ch = np.r_[0, np.where(np.diff(lab) != 0)[0] + 1]
    kinds = []
    for s in ch:
        name = d.columns[lab[s]]
        if name in ('tennis', 'navigation'):
            kinds.append(name)
        elif name == 'squeeze_imagery':
            kinds.append('squeeze')
    return kinds


def collect(output_root, prefix, lor_prefix, tsv):
    rows = []
    kinds = run1_task_block_kinds(tsv)
    sq_pos = [i for i, k in enumerate(kinds) if k == 'squeeze']
    for fd in sorted(glob.glob(os.path.join(output_root, f'{prefix}_fold-*'))):
        preds = latest_test_predictions(fd)
        if not preds:
            continue
        subs = {subj_id(p['subject']) for p in preds}
        task = [p for p in preds if int(p['true_label']) == 1]
        if len(subs) != 1 or len(task) != len(kinds):
            print(f'WARNING {fd}: {len(subs)} subjects, {len(task)} true-Task samples (expected 1 and {len(kinds)}); skipped')
            continue
        for k, pos in enumerate(sq_pos, 1):
            rows.append(dict(subject=next(iter(subs)), run=1, squeeze_block=k,
                             pred=int(task[pos]['predicted_label']),
                             bad=int(task[pos]['predicted_label']) == 0))
    for pat in (f'{lor_prefix}_fold-*/lor_predictions_fold_*.json', f'{lor_prefix}_fold-*/lor_predictions_run4_fold_*.json'):
        for f in sorted(glob.glob(os.path.join(output_root, pat))):
            for p in json.load(open(f))['predictions']:
                if int(p['true_label']) == SQUEEZE:
                    rows.append(dict(subject=subj_id(p['subject']), run=int(p['run']), start=int(p['start']),
                                     pred=int(p['predicted_label']), bad=int(p['predicted_label']) == 0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(REPO / 'output/squeeze_force/squeeze_block_response.csv'))
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--prefix', default=STAGE_A_PREFIX, help='Stage A LOOCV run name prefix')
    ap.add_argument('--lor_prefix', default=LOR_STAGE_A_PREFIX)
    ap.add_argument('--tsv', default=str(REPO / 'data/task_timing/timing_onehot_task1.tsv'))
    ap.add_argument('--out', default=str(REPO / 'output/squeeze_force/squeeze_notask_vs_force'))
    a = ap.parse_args()

    force = pd.read_csv(a.csv)
    force = force[force.category != 'no_data']
    model = collect(a.output_root, a.prefix, a.lor_prefix, a.tsv)
    if model.empty:
        raise SystemExit('no Stage A predictions found')
    df = join(force, model)
    df['cat'] = df.apply(assign_cat, axis=1)
    lost = df[df.bad.isna()]
    print(f'{len(lost)} of {len(df)} squeeze blocks have no Stage A prediction; dropped. Runs with predictions: '
          f'{sorted(df[df.bad.notna()].run.unique())}')
    df = df[df.bad.notna()].copy()
    df['bad'] = df.bad.astype(bool)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=['start'], errors='ignore').rename(columns={'bad': 'called_no_task'}).to_csv(
        out.with_suffix('.csv'), index=False)
    draw_breakdown(df, 'bad', out, 'Called\n"No Task"', 'squeeze blocks called "No Task"',
                   'Called "No Task" by what the participant did',
                   'Squeeze blocks the Stage A model called "No Task": did the participant actually squeeze?\n'
                   'Left: blocks called No Task vs all squeeze blocks, by what the participant did '
                   '(early = squeezed at the wrong time, missed = did not squeeze).\n'
                   'Right: % of each kind of block called No Task (Wilson 95% CI).')


if __name__ == '__main__':
    main()

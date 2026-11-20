#!/usr/bin/env python
"""Classify every squeeze block of imagery runs 1-4 by what the participant's force trace did.

Input : data/squeeze_force/Pxx/Pxx_task{1-4}_cut.1D (force, one value per TR = 0.8 s; task N = imagery
        run N) and data/task_timing/timing_onehot_task{N}.tsv (block timing, differs between runs).
Output: one CSV row per squeeze block (default output/squeeze_force/squeeze_block_response.csv).

A squeeze block = 13 TR squeeze_imagery (imagine) followed by 7 TR squeeze_action (cue: really squeeze).
Force baseline and squeeze size differ a lot between people, so everything is scaled by the subject's
own typical squeeze: ref_amp = median peak-to-peak force in the action window of their run-1 blocks.

  ratio_cue   = peak-to-peak force in [action start, action end + LAG TR) / ref_amp
  ratio_early = max |force - run median| in the imagery window / ref_amp

  category (mutually exclusive, in this order):
    early    ratio_early >= SQUEEZE_RATIO  (squeezed while still meant to be imagining; the cue
             squeeze, if any, is in cue_squeeze)
    squeezed ratio_cue >= WEAK_RATIO
    missed   otherwise
  weak = squeezed (or early with a cue squeeze) and ratio_cue < SQUEEZE_RATIO: a real but small squeeze.
  no_data  the force file ends before the block (P29 run 3, P29 run 4).

    python analysis/classify_squeeze_force.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNS = [1, 2, 3, 4]
ACTION_LAG = 3        # TR after the end of the action window still counted as the response
SQUEEZE_RATIO = 0.3   # >= this fraction of the subject's run-1 squeeze size = a normal squeeze
WEAK_RATIO = 0.05     # below this = no squeeze (flat line); between WEAK and SQUEEZE = weak squeeze


def blocks(tsv):
    d = pd.read_csv(tsv, sep='\t')
    lab = d.values.argmax(1)
    ch = np.r_[0, np.where(np.diff(lab) != 0)[0] + 1]
    en = np.r_[ch[1:], len(lab)]
    return [(d.columns[lab[s]], int(s), int(e)) for s, e in zip(ch, en)]


def load_force(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    x = np.loadtxt(path)
    return x if x.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force_dir', default=str(REPO / 'data' / 'squeeze_force'))
    ap.add_argument('--timing_dir', default=str(REPO / 'data' / 'task_timing'))
    ap.add_argument('--out', default=str(REPO / 'output' / 'squeeze_force' / 'squeeze_block_response.csv'))
    a = ap.parse_args()

    rows = []
    for pdir in sorted(Path(a.force_dir).glob('P??')):
        sub = pdir.name
        if not (pdir / f'{sub}_task1_cut.1D').exists():
            print(f'skip {sub}: no force files')
            continue
        raw = {}
        for r in RUNS:
            x = load_force(pdir / f'{sub}_task{r}_cut.1D')
            bl = blocks(Path(a.timing_dir) / f'timing_onehot_task{r}.tsv')
            raw[r] = (x, bl)
        # subject's own squeeze size from run 1 (awake, everyone squeezes)
        x1, bl1 = raw[1]
        amps = [np.ptp(x1[e:e + 7 + ACTION_LAG]) for n, s, e in bl1 if n == 'squeeze_imagery'
                if x1 is not None and e < len(x1)]
        ref = float(np.median(amps))
        for r in RUNS:
            x, bl = raw[r]
            base = float(np.median(x)) if x is not None else np.nan
            k = 0
            for name, s, e in bl:
                if name != 'squeeze_imagery':
                    continue
                k += 1
                rec = dict(subject=sub, run=r, squeeze_block=k, imagery_start_tr=s, action_start_tr=e,
                           ref_amp=ref)
                if x is None or e >= len(x):
                    rec.update(category='no_data', weak=False, cue_squeeze=False, ratio_cue=np.nan,
                               ratio_early=np.nan)
                    rows.append(rec)
                    continue
                cue = x[e:min(e + 7 + ACTION_LAG, len(x))]
                img = x[s:e]
                rc = float(np.ptp(cue)) / ref
                re_ = float(np.abs(img - base).max()) / ref
                if re_ >= SQUEEZE_RATIO:
                    cat = 'early'
                elif rc >= WEAK_RATIO:
                    cat = 'squeezed'
                else:
                    cat = 'missed'
                rec.update(category=cat, ratio_cue=rc, ratio_early=re_, cue_squeeze=rc >= WEAK_RATIO,
                           weak=bool(cat != 'missed' and WEAK_RATIO <= rc < SQUEEZE_RATIO))
                rows.append(rec)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'wrote {out} ({len(df)} blocks)')
    print(pd.crosstab(df.run, df.category, margins=True))
    print('weak:', df.groupby('run').weak.sum().to_dict())


if __name__ == '__main__':
    main()

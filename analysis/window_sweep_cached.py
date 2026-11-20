#!/usr/bin/env python
"""Cheap window-length / window-shift screen for the 3-way imagery task (tennis / navigation / squeeze).

Replacement for scripts/window_sweep_linear_probe.py that does the expensive work ONCE:

  stage "cache"  Read each subject's data.pt (mmap), average-pool EVERY frame to a coarse grid
                 (--pool 16 -> 6^3 = 216 features/TR) and save [T, 216] per subject. ~1 pass over
                 the imagery run-1 data (single-threaded, CPU only; submit with sbatch if you can).
  stage "sweep"  From the cached per-TR features, build each block's window mean for every
                 (window length, shift) and run leave-one-subject-out logistic regression
                 (subject-mean-centred, same probe as baseline_linear_probe.py). Seconds, no I/O.

Blocks come from MichiganFullBlock (TRIM_START=0, full task+trailing-rest span, 37-38 TR), so the
window is [onset + shift, onset + shift + W). A (W, shift) pair is only used if it fits inside every
block, so every class always gets the same window length.

Natural anchors: 13 TR = imagery only (all classes); 20 TR = imagery + squeeze action (all classes);
37 TR = current full block.

NOTE: this is a screen. Picking the best window by LOSO accuracy on all 26 subjects is optimistic;
confirm the top 2-3 windows with NeuroSTORM on a few held-out folds before trusting the number.

Usage (neurostorm env, repo root):
  python analysis/window_sweep_cached.py --stage cache --data_root $DATA_ROOT --cache_dir output/window_sweep/cache
  python analysis/window_sweep_cached.py --stage sweep --data_root $DATA_ROOT --cache_dir output/window_sweep/cache \
      --out_dir output/window_sweep --windows 13,20,24,28,32,37 --shifts 0,2,4
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASS_NAMES = ['tennis', 'navigation', 'squeeze']
RUN_DIR = 'FIXED_xcp_nosmooth_task-imagery_run-1'


def subjects_in(img_root):
    return sorted(d for d in os.listdir(img_root)
                  if d.startswith('sub-') and os.path.isfile(os.path.join(img_root, d, 'data.pt')))


def stage_cache(args):
    import torch
    import torch.nn.functional as F
    from datasets.fmri_datasets import BaseDataset
    torch.set_num_threads(args.threads)
    img_root = os.path.join(args.data_root, RUN_DIR, 'img')
    os.makedirs(args.cache_dir, exist_ok=True)
    pf = args.pool
    for subj in subjects_in(img_root):
        out = os.path.join(args.cache_dir, f'{subj}.npy')
        if os.path.isfile(out):
            print(f'{subj}: cached, skipping')
            continue
        blob = BaseDataset._load_blob_file(os.path.join(img_root, subj, 'data.pt'), mmap=True)
        frames, scale, n = blob['frames'], float(blob['scale']), int(blob['num_frames'])
        feats = []
        for t0 in range(0, n, 32):
            x = frames[t0:t0 + 32].to(torch.float32) * scale          # [t, H, W, D]
            H, W, D = x.shape[1:]
            x = x[:, :H - H % pf, :W - W % pf, :D - D % pf]
            x = F.avg_pool3d(x.unsqueeze(1), pf).flatten(1)          # [t, (H/pf)(W/pf)(D/pf)]
            feats.append(x.numpy().astype(np.float32))
        np.save(out, np.concatenate(feats))
        print(f'{subj}: {n} frames -> {out}', flush=True)


def stage_sweep(args):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from datasets.fmri_datasets import MichiganFullBlock

    tsv = os.path.join(args.data_root, RUN_DIR, 'metadata', 'timing_onehot_task1.tsv')
    blocks = MichiganFullBlock._blocks_from_tsv(MichiganFullBlock.__new__(MichiganFullBlock), tsv)
    label_map = {'tennis': 0, 'navigation': 1, 'squeeze': 2}
    min_len = min(e - s for s, e, _ in blocks)
    print(f'{len(blocks)} blocks, shortest block = {min_len} TR')

    cache = {Path(f).stem: np.load(os.path.join(args.cache_dir, f))
             for f in sorted(os.listdir(args.cache_dir)) if f.endswith('.npy')}
    subjects = sorted(cache)
    print(f'{len(subjects)} subjects cached')

    windows = [int(w) for w in args.windows.split(',')]
    shifts = [int(s) for s in args.shifts.split(',')]
    rows = []
    for W in windows:
        for sh in shifts:
            if sh + W > min_len:
                continue
            X, y, g = [], [], []
            for subj in subjects:
                T = len(cache[subj])
                sx = []
                for (s, e, lab) in blocks:
                    a = s + sh
                    if a + W > T:
                        continue
                    sx.append((cache[subj][a:a + W].mean(0), label_map[lab]))
                if not sx:
                    continue
                xs = np.stack([v for v, _ in sx])
                xs = xs - xs.mean(0, keepdims=True)                   # subject-mean centring
                X.append(xs)
                y += [l for _, l in sx]
                g += [subj] * len(sx)
            X, y, g = np.concatenate(X), np.array(y), np.array(g)
            correct, aucs, per_subj = 0, [], []
            P = np.zeros((len(y), 3))
            for held in np.unique(g):
                te = g == held
                sc = StandardScaler().fit(X[~te])
                clf = LogisticRegression(max_iter=5000, C=args.C).fit(sc.transform(X[~te]), y[~te])
                P[te] = clf.predict_proba(sc.transform(X[te]))
                per_subj.append((P[te].argmax(1) == y[te]).mean())
            acc = (P.argmax(1) == y).mean()
            auc = roc_auc_score(y, P, multi_class='ovr', average='macro')
            sem = np.std(per_subj, ddof=1) / np.sqrt(len(per_subj))
            rows.append(dict(window=W, shift=sh, n_blocks=len(y), loso_acc=acc, acc_subj_sem=sem, loso_auroc=auc))
            print(f'W={W:>2} shift={sh}: acc={acc:.3f} (+/-{sem:.3f} subj SEM) AUROC={auc:.3f}  n={len(y)}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'window_sweep_loso.csv'), index=False)
    print(df.pivot(index='shift', columns='window', values='loso_acc').round(3).to_string())
    print(f'Chance = 0.333. Saved {args.out_dir}/window_sweep_loso.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['cache', 'sweep'], required=True)
    ap.add_argument('--data_root', default='/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM')
    ap.add_argument('--cache_dir', default='output/window_sweep/cache')
    ap.add_argument('--out_dir', default='output/window_sweep')
    ap.add_argument('--pool', type=int, default=16)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--C', type=float, default=1.0)
    ap.add_argument('--windows', default='13,20,24,28,32,37')
    ap.add_argument('--shifts', default='0,2,4')
    args = ap.parse_args()
    {'cache': stage_cache, 'sweep': stage_sweep}[args.stage](args)


if __name__ == '__main__':
    main()

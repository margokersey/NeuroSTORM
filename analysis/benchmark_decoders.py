#!/usr/bin/env python
"""Benchmark classical decoders (GLM template matching, LR, SVM, LDA, RF) against NeuroSTORM
on the 3-way imagery task (tennis / navigation / squeeze), across propofol states.

Every model is trained ONLY on imagery run 1 of all-but-one subjects (leave-one-subject-out,
the same held-out subjects as the 26 LOOCV folds) and scored on the held-out subject's
  run1 (awake) | pre_lor (run 2, propofol on, still responsive) | lor | ror | recovery (run 4)
using the SAME block/window/segment definitions as the NeuroSTORM LOR analysis
(datasets.fmri_datasets.Michigan / MichiganLOR; asserted below, not re-derived).

Why: if a simple GLM/linear decoder loses the same amount of accuracy under propofol that
NeuroSTORM does, the signal is genuinely weaker; if it holds up, NeuroSTORM's drop is
distribution shift (calibration), which retraining / normalisation can fix.

Features (per block, on a spatially average-pooled grid, --pool 4 -> 24^3 voxels)
  winmean  time-mean of the same 32-frame window NeuroSTORM sees
  beta     single-trial GLM beta (LSS: target-block regressor + all-other-blocks regressor +
           intercept + linear drift; task epoch boxcar x canonical HRF), fit on the WHOLE run.
           NB: the GLM uses the full run, NeuroSTORM only its 32-frame window, so the GLM is
           the standard-practice, slightly information-rich comparator.
Normalisation (applied per subject-run, label-free, so it is usable on a patient scan)
  raw | run_z (voxelwise z across the run's blocks) | spatial_z (z across voxels, per block)
Classifiers (fixed hyper-parameters, no tuning on held-out data)
  template (GLM-style group template matching: correlate a block with class-mean maps
  from training subjects), logreg, linsvm, lda (PCA->shrinkage LDA), rf

Stages (features are cached, evaluation is cheap):
  python analysis/benchmark_decoders.py --stage features --data_root $DATA_ROOT --lor_csv ... --cache_dir ...
  python analysis/benchmark_decoders.py --stage eval --cache_dir ... --out_dir output/benchmark_decoders
Submit both with analysis/sbatch_benchmark_decoders.sh (CPU only; do not run on the login node).

Outputs in --out_dir: blocks.csv (one row per held-out block x model), summary.csv (per model x state),
summary.txt (printed table). Optional --pred_glob adds NeuroSTORM rows from lor_predictions_fold_*.json
and each fold's run-1 test predictions (same loader as analysis/plot_lor_task_decoding.py).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TR = 0.8
CLASS_NAMES = ['tennis', 'navigation', 'squeeze']
LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}
STATES = ['run1', 'pre_lor', 'lor', 'ror', 'recovery']
RUN_DIR = 'FIXED_xcp_nosmooth_task-imagery_run-{run}'


# ------------------------------------------------------------------ blocks / GLM
def hrf(tr=TR, length=32.0):
    from scipy.stats import gamma
    t = np.arange(0, length, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def parse_blocks(tsv_path, trim):
    """[(task_on, task_off, win_start, block_end, label)] -- mirrors Michigan._blocks_from_tsv
    (task run + absorbed trailing rest, squeeze imagery+action merged) and ASSERTS agreement."""
    df = pd.read_csv(tsv_path, sep='\t')
    df['squeeze'] = df['squeeze_imagery'] | df['squeeze_action']
    labs = df[['rest', 'squeeze', 'navigation', 'tennis']].idxmax(axis=1).tolist()
    runs, prev, st = [], None, 0
    for i, lab in enumerate(labs + [None]):
        if lab != prev:
            if prev is not None:
                runs.append((st, i, prev))
            st, prev = i, lab
    blocks = []
    for k, (s, e, lab) in enumerate(runs):
        if lab == 'rest':
            continue
        end = runs[k + 1][1] if k + 1 < len(runs) and runs[k + 1][2] == 'rest' else e
        blocks.append((s, e, s + trim, end, lab))
    from datasets.fmri_datasets import Michigan
    ref = Michigan._blocks_from_tsv(None, tsv_path, trim)
    mine = [(b[2], b[3], b[4]) for b in blocks]
    if mine != [tuple(r) for r in ref]:
        raise RuntimeError(f"block definition drifted from Michigan._blocks_from_tsv for {tsv_path}")
    return blocks, len(df)


def lss_betas(Y, blocks, T):
    """Single-trial (LSS) betas, one row per block. Y [T, V]."""
    h = hrf()
    box = np.zeros((len(blocks), T))
    for k, (on, off, *_r) in enumerate(blocks):
        box[k, on:off] = 1.0
    conv = np.stack([np.convolve(b, h)[:T] for b in box])          # [B, T]
    total = conv.sum(0)
    drift = np.linspace(-1, 1, T)
    out = np.zeros((len(blocks), Y.shape[1]), np.float32)
    for k in range(len(blocks)):
        X = np.stack([conv[k], total - conv[k], np.ones(T), drift], 1)
        out[k] = (np.linalg.pinv(X)[0] @ Y).astype(np.float32)
    return out


def load_pooled(subject_dir, pool):
    """data.pt -> float32 [T, V] on a pool^3 average-pooled grid (same blob format the model reads)."""
    import torch
    import torch.nn.functional as F
    from datasets.fmri_datasets import BaseDataset
    blob = BaseDataset._load_blob_file(os.path.join(subject_dir, 'data.pt'), mmap=True)
    frames, scale, T = blob['frames'], float(blob['scale']), int(blob['num_frames'])
    out = []
    for i in range(0, T, 64):
        c = frames[i:min(i + 64, T)].to(torch.float32) * scale
        out.append(F.avg_pool3d(c.unsqueeze(1), pool).reshape(c.shape[0], -1).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- feature stage
def segment_tags(run, k, lor_blk, ror_blk):
    if run == 1:
        return 'run1'
    if run == 4:
        return 'recovery'
    if run == 2:
        return 'pre_lor' if k < lor_blk else ('boundary_lor' if k == lor_blk else 'lor')
    if ror_blk is None:
        return 'lor'
    return 'lor' if k < ror_blk else ('boundary_ror' if k == ror_blk else 'ror')


def extract_one(sub, run, args, lor_table):
    cache = Path(args.cache_dir) / f'{sub}_run{run}.npz'
    if cache.exists() and not args.overwrite:
        return str(cache), 'cached'
    rdir = Path(args.data_root) / RUN_DIR.format(run=run)
    sdir = rdir / 'img' / sub
    if not (sdir / 'data.pt').is_file():
        return None, 'no data'
    blocks, n_tsv = parse_blocks(rdir / 'metadata' / f'timing_onehot_task{run}.tsv', args.trim)
    Y = load_pooled(str(sdir), args.pool)
    if Y.shape[0] != n_tsv:
        return None, f'{Y.shape[0]} frames != {n_tsv} timing rows, scan skipped'
    lor_blk = ror_blk = None
    if run in (2, 3):
        from datasets.fmri_datasets import MichiganLOR
        if sub not in lor_table:
            return None, 'not in LOR table'
        lor_tr, ror_tr = lor_table[sub]
        b0 = [(b[2], b[3], b[4]) for b in blocks_untrimmed(rdir, run)]    # untrimmed, as in MichiganLOR
        if run == 2:
            lor_blk = MichiganLOR._block_containing(b0, lor_tr)
        elif ror_tr is not None:
            ror_blk = MichiganLOR._block_containing(b0, ror_tr)
    betas = lss_betas(Y, blocks, Y.shape[0])
    keep, wm, lab, seg, bidx = [], [], [], [], []
    for k, (_on, _off, ws, end, name) in enumerate(blocks):
        if ws + args.seq_len > end or ws + args.seq_len > Y.shape[0]:
            continue
        keep.append(k)
        wm.append(Y[ws:ws + args.seq_len].mean(0))
        lab.append(LABEL[name]); bidx.append(k)
        seg.append(segment_tags(run, k, lor_blk, ror_blk))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, winmean=np.stack(wm).astype(np.float32), beta=betas[keep], label=np.array(lab),
             segment=np.array(seg), block_idx=np.array(bidx))
    return str(cache), 'ok'


def blocks_untrimmed(rdir, run):
    return parse_blocks(rdir / 'metadata' / f'timing_onehot_task{run}.tsv', 0)[0]


def stage_features(args):
    from joblib import Parallel, delayed
    from datasets.fmri_datasets import MichiganLOR
    MichiganLOR.LOR_CSV = args.lor_csv
    lor_table = MichiganLOR.load_lor_table()
    r1 = Path(args.data_root) / RUN_DIR.format(run=1) / 'img'
    subjects = sorted(p.name for p in r1.iterdir() if p.name.startswith('sub-') and p.name not in args.exclude_subjects)
    jobs = [(s, r) for s in subjects for r in (1, 2, 3, 4)]
    print(f"[features] {len(subjects)} subjects x runs 1-4 -> {len(jobs)} scans, n_jobs={args.n_jobs}", flush=True)
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(extract_one)(s, r, args, lor_table) for s, r in jobs)
    for (s, r), (_p, msg) in zip(jobs, res):
        if msg not in ('ok', 'cached'):
            print(f"[features] {s} run-{r}: {msg}", flush=True)


# ------------------------------------------------------------------- eval stage
class TemplateMatcher:
    """GLM-style group template matching: class-mean (minus grand-mean) map from training subjects;
    a block is assigned to the class whose template it correlates with most."""
    def fit(self, X, y):
        self.mu = X.mean(0)
        Xc = X - self.mu
        T = np.stack([Xc[y == c].mean(0) for c in range(3)])
        self.T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)
        return self

    def corr(self, X):
        Xc = X - self.mu
        Xc = (Xc - Xc.mean(1, keepdims=True)) / (Xc.std(1, keepdims=True) + 1e-12)
        return Xc @ self.T.T / X.shape[1]

    def predict_proba(self, X):
        z = self.corr(X) * 10
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def make_model(name):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    if name == 'template':
        return TemplateMatcher()
    if name == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    if name == 'linsvm':
        return make_pipeline(StandardScaler(), LinearSVC(C=1e-4, max_iter=5000))
    if name == 'lda':
        return make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0),
                             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, max_features='sqrt', n_jobs=1, random_state=0)
    raise ValueError(name)


def scores(model, X):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
    else:
        z = model.decision_function(X)
        e = np.exp(z - z.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
    return p


def normalise(X, meta, how):
    if how == 'raw':
        return X
    X = X.copy()
    if how == 'run_z':
        for _, idx in meta.groupby(['subject', 'run']).indices.items():
            if len(idx) < 5:
                continue
            X[idx] = (X[idx] - X[idx].mean(0)) / (X[idx].std(0) + 1e-6)
        return X
    if how == 'spatial_z':
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    raise ValueError(how)


def run_fold(held, X, meta, model_name):
    tr = ((meta['run'] == 1) & (meta['subject'] != held)).to_numpy()
    te = (meta['subject'] == held).to_numpy()
    y = meta['label'].to_numpy()
    m = make_model(model_name).fit(X[tr], y[tr])
    p = scores(m, X[te])
    out = meta[te].copy()
    out['pred'] = p.argmax(1)
    for c in range(3):
        out[f'p_{CLASS_NAMES[c]}'] = p[:, c]
    out['p_true'] = p[np.arange(len(p)), y[te]]
    out['r_true'] = m.corr(X[te])[np.arange(len(p)), y[te]] if model_name == 'template' else np.nan
    return out


def load_cache(cache_dir):
    rows, mats = [], {'winmean': [], 'beta': []}
    for f in sorted(Path(cache_dir).glob('sub-*_run*.npz')):
        sub, run = f.stem.split('_run')
        z = np.load(f)
        for i in range(len(z['label'])):
            rows.append({'subject': sub, 'run': int(run), 'block_idx': int(z['block_idx'][i]),
                         'label': int(z['label'][i]), 'segment': str(z['segment'][i])})
        for k in mats:
            mats[k].append(z[k])
    if not rows:
        raise SystemExit(f"no cached features in {cache_dir}; run --stage features first")
    return pd.DataFrame(rows), {k: np.concatenate(v) for k, v in mats.items()}


def summarise(blocks):
    from sklearn.metrics import roc_auc_score
    recs = []
    keys = ['feature', 'norm', 'model']
    for key, g in blocks.groupby(keys):
        for st in STATES:
            s = g[g['segment'] == st]
            if s.empty:
                continue
            per_sub = (s['pred'] == s['label']).groupby(s['subject']).mean()
            P = s[[f'p_{c}' for c in CLASS_NAMES]].to_numpy()
            try:
                auc = roc_auc_score(s['label'], P, multi_class='ovr', average='macro', labels=[0, 1, 2])
            except ValueError:
                auc = np.nan
            recs.append(dict(zip(keys, key), state=st, n_subjects=len(per_sub), n_blocks=len(s),
                             acc_mean=per_sub.mean(), acc_sem=per_sub.std(ddof=1) / np.sqrt(len(per_sub)) if len(per_sub) > 1 else np.nan,
                             auroc_pooled=auc, mean_p_true=s['p_true'].mean(), mean_r_true=s['r_true'].mean(),
                             max_pred_frac=np.bincount(s['pred'], minlength=3).max() / len(s)))
    return pd.DataFrame(recs)


def neurostorm_rows(args):
    sys.path.insert(0, str(REPO / 'analysis'))
    from plot_lor_task_decoding import load
    rows = load(args.pred_glob, set(args.exclude_subjects), args.output_root, args.stage_b_prefix,
                args.pred_glob.replace('lor_predictions_fold_', 'lor_predictions_run4_fold_'))
    recs = []
    for st in STATES:
        v = np.array([r[f'{st}_acc'] for r in rows], float)
        n = np.array([r[f'{st}_n'] for r in rows])
        ok = ~np.isnan(v)
        if ok.any():
            recs.append(dict(feature='voxel', norm='-', model='NeuroSTORM', state=st, n_subjects=int(ok.sum()),
                             n_blocks=int(n[ok].sum()), acc_mean=v[ok].mean(),
                             acc_sem=v[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan))
    return pd.DataFrame(recs)


def stage_eval(args):
    from joblib import Parallel, delayed
    meta, mats = load_cache(args.cache_dir)
    keep = ~meta['subject'].isin(args.exclude_subjects)
    meta = meta[keep].reset_index(drop=True)
    mats = {k: v[keep.to_numpy()] for k, v in mats.items()}
    subjects = sorted(meta.loc[meta['run'] == 1, 'subject'].unique())
    print(f"[eval] {len(subjects)} held-out subjects; blocks per run: {meta.groupby('run').size().to_dict()}", flush=True)
    all_blocks = []
    for feat in args.features:
        live = (mats[feat] != 0).mean(0) > 0.99                 # in-brain voxels present in (almost) every block
        print(f"[eval] feature {feat}: {int(live.sum())} voxels", flush=True)
        for norm in args.norms:
            X = normalise(mats[feat][:, live], meta, norm)
            for mname in args.models:
                parts = Parallel(n_jobs=args.n_jobs)(delayed(run_fold)(s, X, meta, mname) for s in subjects)
                df = pd.concat(parts)
                df.insert(0, 'model', mname); df.insert(0, 'norm', norm); df.insert(0, 'feature', feat)
                all_blocks.append(df)
                a = df[df['segment'] == 'run1']
                print(f"[eval] {feat:8s} {norm:9s} {mname:9s} run1 acc {np.mean(a['pred'] == a['label']):.3f}", flush=True)
    blocks = pd.concat(all_blocks)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(out / 'blocks.csv', index=False)
    summ = summarise(blocks)
    if args.pred_glob:
        summ = pd.concat([summ, neurostorm_rows(args)], ignore_index=True)
    summ.to_csv(out / 'summary.csv', index=False)
    tab = summ.pivot_table(index=['model', 'feature', 'norm'], columns='state', values='acc_mean')[
        [s for s in STATES if s in summ['state'].unique()]]
    txt = ("Per-subject mean accuracy (chance 0.333)\n" + tab.round(3).to_string() + "\n")
    (out / 'summary.txt').write_text(txt)
    print(txt)
    print(f"[eval] wrote {out}/blocks.csv, summary.csv, summary.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['features', 'eval', 'all'], default='all')
    ap.add_argument('--data_root', default=os.environ.get('DATA_ROOT'),
                    help='dir holding FIXED_xcp_nosmooth_task-imagery_run-N/ (env DATA_ROOT)')
    ap.add_argument('--lor_csv', default=os.environ.get('NEUROSTORM_LOR_CSV'))
    ap.add_argument('--cache_dir', default=str(REPO / 'output' / 'benchmark_decoders' / 'cache'))
    ap.add_argument('--out_dir', default=str(REPO / 'output' / 'benchmark_decoders'))
    ap.add_argument('--pool', type=int, default=4, help='spatial avg-pool factor (96 -> 24^3 voxels)')
    ap.add_argument('--trim', type=int, default=2, help='onset trim in TRs; 2 matches the MICHIGAN 3-way model')
    ap.add_argument('--seq_len', type=int, default=32)
    ap.add_argument('--n_jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--exclude_subjects', nargs='*', default=[])
    ap.add_argument('--features', nargs='+', default=['beta', 'winmean'])
    ap.add_argument('--norms', nargs='+', default=['raw', 'run_z', 'spatial_z'])
    ap.add_argument('--models', nargs='+', default=['template', 'logreg', 'linsvm', 'lda', 'rf'])
    ap.add_argument('--pred_glob', default=None,
                    help='NeuroSTORM lor_predictions_fold_*.json glob; adds NeuroSTORM rows to the summary')
    ap.add_argument('--output_root', default='/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output')
    ap.add_argument('--stage_b_prefix', default='FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv')
    args = ap.parse_args()
    if args.stage in ('features', 'all'):
        if not (args.data_root and args.lor_csv):
            sys.exit('--data_root (or DATA_ROOT) and --lor_csv (or NEUROSTORM_LOR_CSV) are required for --stage features')
        stage_features(args)
    if args.stage in ('eval', 'all'):
        stage_eval(args)


if __name__ == '__main__':
    main()

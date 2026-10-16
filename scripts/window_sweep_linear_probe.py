"""
scripts/window_sweep_linear_probe.py

Sweeps block onset-trim (TRIM_START) and window length (sample_duration) to find
the combination that gives the most linearly-decodable signal for the Michigan
3-way imagery task -- using the same subject-mean-centered, coarse-spatial-pooling
linear probe that first confirmed real task-locked signal exists (see
baseline_linear_probe.py / baseline_linear_probe_subtract_subject_mean.py).

Why this exists: the production pipeline hardcodes TRIM_START=2 and a 32-frame
window (Michigan.TRIM_START, sequence_length_override in michigan*.yaml). Neither
number was derived from a canonical HRF-lag calculation or an architectural
constraint -- checked and ruled out: NeuroSTORM's window-attention config
tolerates non-uniform sequence lengths via a shrink-to-fit fallback
(get_window_size in models/neurostorm.py), and 32 is not a codebase-wide default
(HCPTASK and TransDiag both use sequence_length_override: 40). They were
reasonable choices made in the moment, not empirically tuned. This script checks
whether a different (trim, window) choice gives the linear probe -- and plausibly
the deep model -- more signal to work with.

Method:
  - For each candidate trim_start, re-derive block boundaries directly from
    timing_onehot_task1.tsv via Michigan._blocks_from_tsv(trim_start=...), so
    block definitions can never drift from what the real training pipeline would
    produce for that trim.
  - For each candidate window length that fits inside every block at that trim
    (window <= the shortest available block across ALL subjects), build the same
    subject-mean-centered, 216-dim coarse-spatial-pooling features used in the
    analysis that first confirmed real signal exists (pool_factor=16 by default,
    matching baseline_linear_probe.py's default).
  - Fit on TRAIN. Select the best (trim, window) by VAL performance ONLY, then
    report that combination's TEST performance as the unbiased confirmation
    number. Selecting by test would leak the sweep into the reported result --
    exactly the mistake this script is careful to avoid.

Each subject's data.pt is loaded once (mmap) and reused across every (trim,
window) combination in the sweep -- only the block boundaries and window slicing
change, not the underlying data read.

Usage (CPU only, run from the NeuroSTORM repo root in the neurostorm conda env):

    python scripts/window_sweep_linear_probe.py \
        --image_path /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_sub_voxel_norm \
        --split_root ./nas/splits --fold 1

    # narrower/wider sweep:
    python scripts/window_sweep_linear_probe.py \
        --image_path .../processed_output_sub_voxel_norm \
        --trims 0,1,2,3,4,5,6,8 --windows 8,12,16,20,24,28,32,36

Run it against processed_output_sub_voxel_norm (the corrected, per-voxel-demeaned
data) -- that's where the real signal actually showed up; running it against the
original processed_output_v2 would just re-discover the subject-confound problem
at every window setting.
"""
import argparse
import os
import sys

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# repo root = parent of this script's directory (scripts/), matching main.py's
# import resolution when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.fmri_datasets import BaseDataset, Michigan  # noqa: E402

CLASS_NAMES = ["tennis", "navigation", "squeeze"]  # index order must match Michigan.LABEL_MAP


def load_split(split_path):
    train, val, test = [], [], []
    bucket = {"train_subjects": train, "val_subjects": val, "test_subjects": test}
    section = None
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in bucket:
                section = line
                continue
            bucket[section].append(line)
    return train, val, test


def load_all_blobs(subjects, img_root):
    """Load every subject's data.pt once (mmap-backed); reused across the whole
    sweep so no subject is re-read from disk for each (trim, window) pair."""
    blobs = {}
    for subj in subjects:
        blob_path = os.path.join(img_root, subj, "data.pt")
        blobs[subj] = BaseDataset._load_blob_file(blob_path, mmap=True)
    return blobs


def max_window_for_trim(blocks, blobs):
    """Largest window length that fits inside every block, for every subject,
    given this trim's block boundaries. This is the hard ceiling on candidate
    window lengths at this trim -- exceeding it would silently drop blocks or
    subjects from smaller/shorter scans, biasing the comparison."""
    m = None
    for subj, blob in blobs.items():
        num_frames = int(blob["num_frames"])
        for (start, end, _lab) in blocks:
            avail = min(end, num_frames) - start
            if avail <= 0:
                continue
            m = avail if m is None else min(m, avail)
    return m if m is not None else 0


def build_features_for_window(subjects, blobs, blocks, window, pool_factor):
    """One row per (subject, block) whose [start, start+window) fits within
    that subject's scan and within the block. Coarse spatial pooling only --
    this is the feature set that actually carried signal in the original probe;
    whole-brain mean is omitted here since the sweep is about window geometry,
    not re-litigating which feature set works."""
    X, y, groups = [], [], []
    for subj in subjects:
        blob = blobs[subj]
        frames = blob["frames"]
        scale = float(blob["scale"])
        num_frames = int(blob["num_frames"])

        for (start, end, task_name) in blocks:
            if start + window > end or start + window > num_frames:
                continue
            clip = frames[start:start + window].to(torch.float32) * scale
            vol = clip.mean(dim=0)

            pf = pool_factor
            H, W, D = vol.shape
            H2, W2, D2 = H - H % pf, W - W % pf, D - D % pf
            vpool = vol[:H2, :W2, :D2].reshape(
                H2 // pf, pf, W2 // pf, pf, D2 // pf, pf
            ).mean(dim=(1, 3, 5))

            X.append(vpool.reshape(-1).numpy().astype(np.float32))
            y.append(Michigan.LABEL_MAP[task_name])
            groups.append(subj)

    if not X:
        return None, None, None
    return np.stack(X), np.array(y), groups


def center_by_subject(X, groups):
    X = X.copy()
    groups = np.asarray(groups)
    for g in np.unique(groups):
        mask = groups == g
        X[mask] = X[mask] - X[mask].mean(axis=0, keepdims=True)
    return X


def fit_eval(X_train, y_train, X_eval, y_eval, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xev = scaler.transform(X_eval)
    clf = LogisticRegression(max_iter=5000, C=C)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xev)
    proba = clf.predict_proba(Xev)
    acc = accuracy_score(y_eval, preds)
    try:
        auroc = roc_auc_score(y_eval, proba, multi_class="ovr", average="macro",
                               labels=list(range(len(CLASS_NAMES))))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_path", required=True,
                     help="e.g. /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_sub_voxel_norm")
    ap.add_argument("--split_root", default="./nas/splits")
    ap.add_argument("--dataset_name", default="MICHIGAN")
    ap.add_argument("--fold", type=int, default=1, help="which split_fixed_N.txt to use")
    ap.add_argument("--pool_factor", type=int, default=16, help="spatial average-pool factor, matches baseline_linear_probe.py's default")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 regularization strength for logistic regression")
    ap.add_argument("--trims", default="0,1,2,3,4,5,6",
                     help="comma-separated TRIM_START candidates, in TR (0.8s each)")
    ap.add_argument("--windows", default="12,16,20,24,28,32,36",
                     help="comma-separated window-length candidates, in frames (0.8s each)")
    ap.add_argument("--out_dir", default="Claude outputs/window_sweep",
                     help="directory to save CSV and figure outputs")
    args = ap.parse_args()

    img_root = os.path.join(args.image_path, "img")
    tsv_path = os.path.join(args.image_path, "metadata", "timing_onehot_task1.tsv")
    if not os.path.isfile(tsv_path):
        raise SystemExit(f"Could not find {tsv_path} -- check --image_path.")

    split_path = os.path.join(args.split_root, args.dataset_name, f"split_fixed_{args.fold}.txt")
    train_subj, val_subj, test_subj = load_split(split_path)
    print(f"fold {args.fold} ({split_path}): train={len(train_subj)} val={len(val_subj)} test={len(test_subj)} subjects")

    print("loading each subject's data.pt once (mmap, reused for the whole sweep)...")
    all_subjects = train_subj + val_subj + test_subj
    blobs = load_all_blobs(all_subjects, img_root)

    trims = [int(t) for t in args.trims.split(",")]
    windows_requested = sorted(set(int(w) for w in args.windows.split(",")))

    results = []
    for trim in trims:
        blocks = Michigan._blocks_from_tsv(None, tsv_path, trim_start=trim)
        max_w = max_window_for_trim(blocks, blobs)
        windows = [w for w in windows_requested if w <= max_w]
        skipped = [w for w in windows_requested if w > max_w]
        print(f"\n=== trim={trim} TR ({trim * 0.8:.1f}s)  |  max window that fits every block = "
              f"{max_w} frames ({max_w * 0.8:.1f}s) ===")
        if skipped:
            print(f"  (skipping requested windows {skipped} -- exceed max fitting window at this trim)")
        if not windows:
            continue

        for window in windows:
            Xtr, ytr, gtr = build_features_for_window(train_subj, blobs, blocks, window, args.pool_factor)
            Xva, yva, gva = build_features_for_window(val_subj, blobs, blocks, window, args.pool_factor)
            Xte, yte, gte = build_features_for_window(test_subj, blobs, blocks, window, args.pool_factor)
            if Xtr is None or Xva is None or Xte is None:
                print(f"  window={window:>3}: no samples produced, skipping")
                continue

            Xtr_c = center_by_subject(Xtr, gtr)
            Xva_c = center_by_subject(Xva, gva)
            Xte_c = center_by_subject(Xte, gte)

            val_acc, val_auroc = fit_eval(Xtr_c, ytr, Xva_c, yva, args.C)
            test_acc, test_auroc = fit_eval(Xtr_c, ytr, Xte_c, yte, args.C)

            print(f"  window={window:>3} ({window * 0.8:4.1f}s)  "
                  f"val_acc={val_acc:.3f} val_AUROC={val_auroc:.3f}   "
                  f"test_acc={test_acc:.3f} test_AUROC={test_auroc:.3f}")
            results.append(dict(trim=trim, window=window, val_acc=val_acc, val_auroc=val_auroc,
                                 test_acc=test_acc, test_auroc=test_auroc))

    if not results:
        raise SystemExit("No (trim, window) combination produced results -- check --trims/--windows against block lengths.")

    # Model selection uses VAL only, never test -- selecting by test would leak
    # the sweep itself into the reported number.
    best = max(results, key=lambda r: r["val_auroc"])
    print("\n" + "=" * 78)
    print(f"BEST BY VAL AUROC:  trim={best['trim']} TR ({best['trim']*0.8:.1f}s), "
          f"window={best['window']} frames ({best['window']*0.8:.1f}s)")
    print(f"  val_acc={best['val_acc']:.3f}  val_AUROC={best['val_auroc']:.3f}")
    print(f"  test_acc={best['test_acc']:.3f}  test_AUROC={best['test_auroc']:.3f}   "
          f"<- unbiased confirmation number (config was chosen from val, not test)")
    print("=" * 78)

    current = next((r for r in results if r["trim"] == Michigan.TRIM_START and r["window"] == 32), None)
    if current:
        print(f"\nFor comparison, the current production setting (trim={Michigan.TRIM_START}, window=32):")
        print(f"  val_acc={current['val_acc']:.3f}  val_AUROC={current['val_auroc']:.3f}   "
              f"test_acc={current['test_acc']:.3f}  test_AUROC={current['test_auroc']:.3f}")
    else:
        print(f"\n(current production setting trim={Michigan.TRIM_START}, window=32 was not in the requested "
              f"sweep grid -- add it to --trims/--windows to compare directly)")

    # Save results and generate a compact figure summarizing the sweep
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"window_sweep_fold_{args.fold}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved sweep CSV to {csv_path}")

    # Pivot for heatmaps (rows=trim, cols=window)
    if not df.empty:
        piv_auroc = df.pivot(index="trim", columns="window", values="val_auroc")
        piv_acc = df.pivot(index="trim", columns="window", values="val_acc")

        sns.set_theme(style="whitegrid", context="talk")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        cmap = "viridis"
        ax0 = axes[0]
        im0 = sns.heatmap(piv_auroc, annot=True, fmt=".3f", cmap=cmap, ax=ax0, cbar_kws={"label": "val AUROC"}, vmin=0.0, vmax=1.0)
        ax0.set_title("Validation AUROC by trim / window")
        ax0.set_xlabel("window (frames)")
        ax0.set_ylabel("trim (TR)")

        ax1 = axes[1]
        im1 = sns.heatmap(piv_acc, annot=True, fmt=".3f", cmap=cmap, ax=ax1, cbar_kws={"label": "val acc"}, vmin=0.0, vmax=1.0)
        ax1.set_title("Validation Accuracy by trim / window")
        ax1.set_xlabel("window (frames)")
        ax1.set_ylabel("")

        # Highlight the best selection on both panels
        try:
            best_trim = best["trim"]
            best_window = best["window"]
            # find index positions in the pivot matrices
            x_pos = list(piv_auroc.columns).index(best_window)
            y_pos = list(piv_auroc.index).index(best_trim)
            for ax in (ax0, ax1):
                ax.scatter([x_pos + 0.5], [y_pos + 0.5], s=300, marker="*", color="white", edgecolors="black", linewidths=1.2)
        except Exception:
            pass

        # Mark production setting if present
        try:
            prod_trim = Michigan.TRIM_START
            prod_window = 32
            if prod_window in piv_auroc.columns and prod_trim in piv_auroc.index:
                x_p = list(piv_auroc.columns).index(prod_window)
                y_p = list(piv_auroc.index).index(prod_trim)
                for ax in (ax0, ax1):
                    ax.scatter([x_p + 0.5], [y_p + 0.5], s=200, marker="o", facecolors="none", edgecolors="red", linewidths=1.6)
        except Exception:
            pass

        fig_path = os.path.join(out_dir, f"window_sweep_fold_{args.fold}.png")
        fig.savefig(fig_path, dpi=300)
        fig.savefig(os.path.join(out_dir, f"window_sweep_fold_{args.fold}.pdf"))
        plt.close(fig)
        print(f"Saved sweep figure to {fig_path}")


if __name__ == "__main__":
    main()

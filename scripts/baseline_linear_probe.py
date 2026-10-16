"""
scripts/baseline_linear_probe.py

Deep-model-free sanity check for the Michigan 3-way imagery task (tennis /
navigation / squeeze). Fits a plain multinomial logistic regression on
frame-averaged, spatially-pooled fMRI signal for the EXACT SAME per-block
windows the NeuroSTORM fine-tuning pipeline uses -- same TRIM_START, same
32-frame window per block (via Michigan._blocks_from_tsv, imported directly
from datasets/fmri_datasets.py so this can never drift out of sync with the
real pipeline), and the same subject-level train/val/test split file a given
fold used. Reports accuracy / macro-F1 / macro-AUROC / confusion matrix on
held-out subjects, for two feature sets:

  1. whole-brain mean (1 feature/block) -- the simplest possible check.
  2. coarse spatially-pooled signal (default 6x6x6=216 features/block via
     average-pooling, no atlas needed) -- a slightly richer linear probe.

Why this run exists: NeuroSTORM fine-tuning (prompt, prompt_ln, and full
fine-tune / tpt_strategy=none) all plateaued at chance with the model's
predicted-class distribution collapsing to a fixed default regardless of
input. This script asks a narrower question, independent of the deep model
entirely: does ANY linearly-decodable class signal exist in this exact
block/window definition at all? If this also can't beat chance, the issue is
very likely upstream (block/window/label definition, alignment, or genuinely
weak signal at this window length) rather than model capacity or training
hyperparameters. If it CAN beat chance clearly, that's strong evidence real
signal exists here and the problem is on the NeuroSTORM training side.

Usage (run from the NeuroSTORM repo root, in the `neurostorm` conda env --
scikit-learn is already in requirements.txt, nothing extra to install):

    python scripts/baseline_linear_probe.py \
        --image_path /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_v2 \
        --split_root ./nas/splits --fold 1

Runs on CPU only, no GPU needed. Takes a few minutes (reads every subject's
data.pt blob once).
"""
import argparse
import os
import sys

import numpy as np
import torch

# repo root = parent of this script's directory (scripts/), so `datasets.*`
# resolves the same way it does when main.py is run from the repo root.
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


def build_features(subjects, img_root, blocks, sequence_length, pool_factor):
    """One row per (subject, block) that fits within that subject's scan."""
    X_pooled, X_mean, y, groups = [], [], [], []
    for subj in subjects:
        subject_path = os.path.join(img_root, subj)
        blob_path = os.path.join(subject_path, "data.pt")
        blob = BaseDataset._load_blob_file(blob_path, mmap=True)
        frames = blob["frames"]              # int8 [T, H, W, D], mmap-backed
        scale = float(blob["scale"])
        num_frames = int(blob["num_frames"])

        for (start, end, task_name) in blocks:
            if start + sequence_length > end or start + sequence_length > num_frames:
                continue

            clip = frames[start:start + sequence_length].to(torch.float32) * scale  # [T,H,W,D]
            vol = clip.mean(dim=0)  # time-average -> [H, W, D]

            whole_brain_mean = float(vol.mean())

            pf = pool_factor
            H, W, D = vol.shape
            H2, W2, D2 = H - H % pf, W - W % pf, D - D % pf
            vpool = vol[:H2, :W2, :D2].reshape(
                H2 // pf, pf, W2 // pf, pf, D2 // pf, pf
            ).mean(dim=(1, 3, 5))

            X_pooled.append(vpool.reshape(-1).numpy().astype(np.float32))
            X_mean.append(whole_brain_mean)
            y.append(Michigan.LABEL_MAP[task_name])
            groups.append(subj)

        del blob, frames

    return (
        np.stack(X_pooled),
        np.array(X_mean, dtype=np.float32).reshape(-1, 1),
        np.array(y),
        groups,
    )


def evaluate(name, X_train, y_train, X_eval, y_eval, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xev = scaler.transform(X_eval)

    clf = LogisticRegression(max_iter=5000, C=C)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xev)
    proba = clf.predict_proba(Xev)

    acc = accuracy_score(y_eval, preds)
    f1 = f1_score(y_eval, preds, average="macro")
    try:
        auroc = roc_auc_score(y_eval, proba, multi_class="ovr", average="macro",
                               labels=list(range(len(CLASS_NAMES))))
    except ValueError:
        auroc = float("nan")
    cm = confusion_matrix(y_eval, preds, labels=list(range(len(CLASS_NAMES))))

    print(f"\n=== {name} ===")
    print(f"acc={acc:.4f}  macro_f1={f1:.4f}  macro_AUROC={auroc:.4f}")
    print(f"confusion matrix (rows=true, cols=pred, order={CLASS_NAMES}):")
    print(cm)
    return dict(name=name, acc=acc, macro_f1=f1, macro_auroc=auroc, confusion_matrix=cm.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_path", required=True, help="e.g. /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_v2")
    ap.add_argument("--split_root", default="./nas/splits")
    ap.add_argument("--dataset_name", default="MICHIGAN")
    ap.add_argument("--fold", type=int, default=1, help="which split_fixed_N.txt to use")
    ap.add_argument("--sequence_length", type=int, default=32, help="must match the deep-model run's --sequence_length")
    ap.add_argument("--pool_factor", type=int, default=16,
                     help="spatial average-pool factor (96 -> 96//pool_factor per axis, e.g. 16 -> 6x6x6=216 features). "
                          "Kept well below the ~450-sample training set size by default to avoid the probe itself overfitting; "
                          "try 8 (1728 features) for a finer probe if 16 clears chance and you want to push further.")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 regularization strength for logistic regression")
    args = ap.parse_args()

    img_root = os.path.join(args.image_path, "img")
    tsv_path = os.path.join(args.image_path, "metadata", "timing_onehot_task1.tsv")
    if not os.path.isfile(tsv_path):
        raise SystemExit(f"Could not find {tsv_path} -- check --image_path.")

    # Michigan._blocks_from_tsv never touches `self`, so this is safe to call
    # unbound. Importing it directly (rather than re-deriving block bounds
    # here) guarantees this baseline uses IDENTICAL blocks to the real
    # training pipeline -- same trim, same merge-with-following-rest logic.
    blocks = Michigan._blocks_from_tsv(None, tsv_path)
    print(f"{len(blocks)} blocks parsed from {tsv_path} (should be 30: 10 each of tennis/navigation/squeeze)")

    split_path = os.path.join(args.split_root, args.dataset_name, f"split_fixed_{args.fold}.txt")
    train_subj, val_subj, test_subj = load_split(split_path)
    print(f"fold {args.fold} ({split_path}): train={len(train_subj)} val={len(val_subj)} test={len(test_subj)} subjects")

    print("extracting features (reads every subject's data.pt once)...")
    Xtr_pool, Xtr_mean, ytr, _ = build_features(train_subj, img_root, blocks, args.sequence_length, args.pool_factor)
    Xva_pool, Xva_mean, yva, _ = build_features(val_subj, img_root, blocks, args.sequence_length, args.pool_factor)
    Xte_pool, Xte_mean, yte, _ = build_features(test_subj, img_root, blocks, args.sequence_length, args.pool_factor)
    print(f"train samples={len(ytr)}  val={len(yva)}  test={len(yte)}  pooled_dim={Xtr_pool.shape[1]}")

    results = []
    for split_name, Xp, Xm, y in [("val", Xva_pool, Xva_mean, yva), ("test", Xte_pool, Xte_mean, yte)]:
        results.append(evaluate(f"whole-brain mean -> {split_name}", Xtr_mean, ytr, Xm, y, args.C))
        results.append(evaluate(f"pooled {Xtr_pool.shape[1]}-dim -> {split_name}", Xtr_pool, ytr, Xp, y, args.C))

    chance = 1.0 / len(CLASS_NAMES)
    print(f"\nchance level: acc={chance:.3f}, AUROC=0.5")
    print("If none of the above clear chance by a healthy margin, the problem is very likely in the")
    print("block/window/label definition (or genuinely weak/absent signal at this window), not NeuroSTORM's capacity.")
    print("If one or more DOES clear chance clearly, that's evidence real linear signal exists here and the")
    print("issue is on the NeuroSTORM training side (LR schedule, tpt_strategy, optimization) instead.")


if __name__ == "__main__":
    main()

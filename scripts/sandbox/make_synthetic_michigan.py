#!/usr/bin/env python
"""Generate a tiny synthetic Michigan-style dataset for laptop smoke tests.

Produces exactly the on-disk layout the real MICHIGAN_FULL_BLOCK loader reads:

  <out>/dataset/img/sub-XX/data.pt        {'frames': int8 [T,H,W,D], 'scale': float, 'num_frames': int}
  <out>/dataset/metadata/timing_onehot_task1.tsv
  <out>/nas/splits/MICHIGAN/split_fixed_<N>.txt   (MICHIGAN_FULL_BLOCK reads this dir too)

The class signal is a bright blob whose position depends on the task, so a
working pipeline's loss should visibly drop within a few epochs. The data are
random -- never use results from this for anything but "does the code run".

Usage:
  python scripts/sandbox/make_synthetic_michigan.py --out output/sandbox
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

TASKS = ["tennis", "navigation", "squeeze"]  # label ids 0, 1, 2 (Michigan.LABEL_MAP)


def build_timing(blocks_per_task, task_len, rest_len, lead_rest, seed):
    """Block-design onehot TSV (task then rest, tasks interleaved)."""
    rng = np.random.RandomState(seed)
    order = np.repeat(np.arange(len(TASKS)), blocks_per_task)
    rng.shuffle(order)
    rows, spans = [], []
    t = lead_rest
    rows += [dict(rest=1, squeeze_imagery=0, squeeze_action=0, navigation=0, tennis=0)] * lead_rest
    for task_id in order:
        name = TASKS[task_id]
        row = dict(rest=0, squeeze_imagery=int(name == "squeeze"), squeeze_action=0,
                   navigation=int(name == "navigation"), tennis=int(name == "tennis"))
        rows += [row] * task_len
        rows += [dict(rest=1, squeeze_imagery=0, squeeze_action=0, navigation=0, tennis=0)] * rest_len
        spans.append((t, t + task_len + rest_len, name))
        t += task_len + rest_len
    return pd.DataFrame(rows), spans


def make_subject(spans, n_frames, vol, rng):
    frames = rng.randn(n_frames, vol, vol, vol).astype(np.float32) * 0.3
    c = vol // 4
    centers = {"tennis": (c, c, c), "navigation": (3 * c, c, c), "squeeze": (c, 3 * c, 3 * c)}
    for start, end, name in spans:
        x, y, z = centers[name]
        frames[start:end, x - 2:x + 2, y - 2:y + 2, z - 2:z + 2] += 2.0
    scale = float(np.abs(frames).max() / 127.0)
    q = np.clip(np.round(frames / scale), -127, 127).astype(np.int8)
    return {"frames": torch.from_numpy(q), "scale": scale, "num_frames": n_frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/sandbox", help="sandbox root (gitignored under output/)")
    ap.add_argument("--n_subjects", type=int, default=6)
    ap.add_argument("--blocks_per_task", type=int, default=3)
    ap.add_argument("--task_len", type=int, default=6, help="TRs of task per block (== --sequence_length)")
    ap.add_argument("--rest_len", type=int, default=2)
    ap.add_argument("--vol", type=int, default=24, help="cubic volume edge in voxels")
    ap.add_argument("--split_num", type=int, default=1)
    ap.add_argument("--split_dir_name", default="MICHIGAN", help="folder under nas/splits (MICHIGAN_FULL_BLOCK shares MICHIGAN's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_dir", default=None, help="exact dataset root to write (default: <out>/dataset)")
    ap.add_argument("--no_split", action="store_true", help="only write img/ + metadata/, skip the split file")
    args = ap.parse_args()

    root = args.dataset_dir or os.path.join(args.out, "dataset")
    split_dir = os.path.join(args.out, "nas", "splits", args.split_dir_name)
    os.makedirs(os.path.join(root, "metadata"), exist_ok=True)

    df, spans = build_timing(args.blocks_per_task, args.task_len, args.rest_len, lead_rest=2, seed=args.seed)
    df.to_csv(os.path.join(root, "metadata", "timing_onehot_task1.tsv"), sep="\t", index=False)
    n_frames = len(df) + 2

    rng = np.random.RandomState(args.seed)
    subjects = [f"sub-{i:02d}" for i in range(1, args.n_subjects + 1)]
    for s in subjects:
        d = os.path.join(root, "img", s)
        os.makedirs(d, exist_ok=True)
        torch.save(make_subject(spans, n_frames, args.vol, rng), os.path.join(d, "data.pt"))

    if args.no_split:
        print(f"synthetic dataset (no split): {args.n_subjects} subjects x {n_frames} frames at {root}")
        return
    os.makedirs(split_dir, exist_ok=True)
    # LOOCV-style split: last subject test, one val, rest train
    n_test, n_val = 1, 1
    train, val, test = subjects[:-(n_val + n_test)], subjects[-(n_val + n_test):-n_test], subjects[-n_test:]
    with open(os.path.join(split_dir, f"split_fixed_{args.split_num}.txt"), "w") as f:
        for name, lst in (("train_subjects", train), ("val_subjects", val), ("test_subjects", test)):
            f.write(name + "\n" + "".join(s + "\n" for s in lst))

    print(f"synthetic dataset: {args.n_subjects} subjects x {n_frames} frames, {args.vol}^3 voxels "
          f"({len(spans)} blocks/subject, {args.task_len}-TR clips)")
    print(f"  image_path : {root}")
    print(f"  split file : {split_dir}/split_fixed_{args.split_num}.txt "
          f"(train {len(train)} / val {len(val)} / test {len(test)})")


if __name__ == "__main__":
    main()

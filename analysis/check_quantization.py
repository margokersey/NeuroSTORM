"""Check how badly int8 quantization (preprocessing_volume.py) hurts the temporal signal.

For each subject data.pt: scale, how much of the int8 range is actually used,
and the median per-voxel temporal std measured in quantization levels.
Rule of thumb: median temporal std < ~3 levels = signal badly quantized;
>= ~10 levels = fine.

Usage:
  python analysis/check_quantization.py --root <preproc_root> [--n 5]
  where <preproc_root> contains img/<subject>/data.pt
"""
import argparse, glob, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--n", type=int, default=5, help="subjects to sample (evenly spaced)")
args = ap.parse_args()

paths = sorted(glob.glob(os.path.join(args.root, "img", "*", "data.pt")))
if not paths:
    raise SystemExit(f"no data.pt under {args.root}/img/*/")
idx = np.unique(np.linspace(0, len(paths) - 1, min(args.n, len(paths))).round().astype(int))

print(f"{len(paths)} subjects found; checking {len(idx)}")
print(f"{'subject':<14}{'scale':>9}{'p99.9|q|':>10}{'max|q|':>8}{'med_tstd(levels)':>18}{'p10_tstd':>10}")
for i in idx:
    p = paths[i]
    blob = torch.load(p, mmap=True, weights_only=False)
    fr = blob["frames"]                      # int8 [T, H, W, D]
    scale = float(blob["scale"])
    brain = (fr != 0).any(dim=0)             # background was filled with 0
    x = fr.float()[:, brain]                 # [T, Nvox]
    tstd = x.std(dim=0)                      # per-voxel temporal std, in levels
    absq = x.abs().flatten()
    sub = absq[torch.randperm(absq.numel())[:2_000_000]]
    p999 = torch.quantile(sub, 0.999).item()
    print(f"{os.path.basename(os.path.dirname(p)):<14}{scale:>9.4f}{p999:>10.1f}"
          f"{absq.max().item():>8.0f}{tstd.median().item():>18.2f}{torch.quantile(tstd, 0.10).item():>10.2f}")
    del x, fr, blob

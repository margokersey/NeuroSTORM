"""
scripts/fix_voxelwise_normalization.py

Rewrites the Michigan dataset's preprocessed data.pt blobs to remove each
subject's PER-VOXEL temporal mean (their "spatial fingerprint" -- anatomy,
registration, coil-sensitivity differences that are constant across every
frame of their scan), on top of the per-subject GLOBAL scalar z-norm that
preprocessing_volume.py already applies.

Why: preprocessing_volume.py's z-norm is a single scalar mean/std for each
subject's whole masked scan:

    global_mean = data[~background].mean()
    global_std  = data[~background].std()
    data_temp = (data - global_mean) / global_std

That only removes ONE overall offset/scale per subject -- any spatially
varying pattern (this subject's motor cortex reads relatively brighter than
their visual cortex, say, for anatomical/registration reasons) survives
identically on every single frame and every block. The linear-probe baseline
(scripts/baseline_linear_probe.py) showed this directly: predictions were
driven almost entirely by SUBJECT identity until each subject's own block-
level features were mean-centered, at which point the SAME 216-dim pooled
features jumped from chance (33% acc / 0.50 AUROC) to 59-67% accuracy /
0.75-0.81 AUROC on held-out subjects -- real task signal exists, it was
being masked by this per-subject spatial fingerprint.

This script applies that same fix at the source, once, so NeuroSTORM's
actual training pipeline (not just the diagnostic probe) sees demeaned data:
for each subject, dequantize their existing data.pt, subtract their own
per-voxel mean computed across their WHOLE scan (~1145 frames, not just one
block -- this is a single fixed, block-label-independent baseline per
subject, so it can't leak any block's label into the features used to
classify it), then re-quantize to int8 with a fresh scale. Output goes to a
NEW folder -- the original processed_output_v2 is never modified.

Note this is different from the (currently unused, for Michigan) codebase
--with_voxel_norm mechanism in datasets/fmri_datasets.py, which APPENDS
separately-saved voxel_mean.pt/voxel_std.pt as extra input channels rather
than subtracting anything from the data itself. This script actually
subtracts, so no dataset-loading or model code needs to change at all -- the
output is a drop-in replacement dataset folder with the exact same data.pt
format and directory layout (img/<subject>/data.pt, metadata/*.tsv).

Usage (run from the NeuroSTORM repo root, CPU only, no GPU needed):

    python scripts/fix_voxelwise_normalization.py \
        --src_root /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_v2 \
        --dst_root /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_sub_voxel_norm

Then point a dataset yaml's image_path at --dst_root (e.g. copy
scripts/configs/datasets/michigan.yaml to a new file with image_path
updated) to fine-tune NeuroSTORM on the corrected data.

Takes a few minutes total; needs roughly as much free disk space as
--src_root already uses (a full second copy is written, nothing in
--src_root is modified or deleted).
"""
import argparse
import os
import shutil
import sys

import torch

# repo root = parent of this script's directory (scripts/), so `datasets.*`
# resolves the same way it does when main.py is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.fmri_datasets import BaseDataset  # noqa: E402


def fix_subject(src_subject_dir, dst_subject_dir, save_voxel_mean=True):
    os.makedirs(dst_subject_dir, exist_ok=True)
    blob_path = os.path.join(src_subject_dir, "data.pt")
    # mmap=False: we need a full in-memory copy anyway (dequantize, subtract,
    # requantize), and mmap-backed tensors are best treated as read-only.
    blob = BaseDataset._load_blob_file(blob_path, mmap=False)
    frames = blob["frames"]  # int8 [T, H, W, D]
    scale = float(blob["scale"])
    num_frames = int(blob["num_frames"])

    data = frames.to(torch.float32) * scale          # dequantize -> globally-normalized float [T,H,W,D]
    voxel_mean = data.mean(dim=0)                     # per-voxel mean across this subject's WHOLE scan -> [H,W,D]
    data_demeaned = data - voxel_mean.unsqueeze(0)     # [T,H,W,D]; background stays exactly 0 (0 - 0 = 0)

    abs_max = float(data_demeaned.abs().max().item())
    new_scale = abs_max / 127.0 if abs_max > 0 else 1.0
    data_int = (data_demeaned / new_scale).round().clamp_(-127, 127).to(torch.int8)

    new_blob = {"frames": data_int, "scale": float(new_scale), "num_frames": num_frames}
    torch.save(new_blob, os.path.join(dst_subject_dir, "data.pt"))

    if save_voxel_mean:
        # Informational only -- NOT the codebase's --with_voxel_norm voxel_mean.pt
        # (that mechanism expects a matching voxel_std.pt too, and appends rather
        # than subtracts; it's unused for Michigan). Saved under a different name
        # purely so the removed per-subject baseline can be inspected later.
        torch.save(voxel_mean.to(torch.float32), os.path.join(dst_subject_dir, "voxel_time_mean_removed.pt"))

    return num_frames, abs_max, new_scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_root", required=True, help="existing preprocessed dataset root, e.g. .../processed_output_v2")
    ap.add_argument("--dst_root", required=True, help="new dataset root to write to, e.g. .../processed_output_sub_voxel_norm")
    ap.add_argument("--overwrite", action="store_true",
                     help="reprocess subjects that already exist in --dst_root (default: skip them, so an interrupted run can resume)")
    ap.add_argument("--no_save_voxel_mean", action="store_true",
                     help="skip saving the informational voxel_time_mean_removed.pt per subject")
    args = ap.parse_args()

    src_img_root = os.path.join(args.src_root, "img")
    dst_img_root = os.path.join(args.dst_root, "img")
    if not os.path.isdir(src_img_root):
        raise SystemExit(f"Could not find {src_img_root} -- check --src_root.")

    # metadata/ (timing_onehot_task1.tsv) is dataset-fixed, not subject-specific --
    # copy it once so --dst_root is immediately usable standalone as an image_path.
    src_meta = os.path.join(args.src_root, "metadata")
    dst_meta = os.path.join(args.dst_root, "metadata")
    if os.path.isdir(src_meta):
        os.makedirs(dst_meta, exist_ok=True)
        for fname in os.listdir(src_meta):
            src_f, dst_f = os.path.join(src_meta, fname), os.path.join(dst_meta, fname)
            if args.overwrite or not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)
        print(f"copied metadata/ -> {dst_meta}")
    else:
        print(f"warning: no metadata/ found at {src_meta} -- copy timing_onehot_task1.tsv into {dst_meta} manually.")

    subjects = sorted(os.listdir(src_img_root))
    print(f"found {len(subjects)} subjects in {src_img_root}")

    os.makedirs(dst_img_root, exist_ok=True)
    for i, subj in enumerate(subjects):
        src_subject_dir = os.path.join(src_img_root, subj)
        dst_subject_dir = os.path.join(dst_img_root, subj)
        if not os.path.isfile(os.path.join(src_subject_dir, "data.pt")):
            print(f"[{i + 1}/{len(subjects)}] {subj}: no data.pt found, skipping")
            continue
        if not args.overwrite and os.path.isfile(os.path.join(dst_subject_dir, "data.pt")):
            print(f"[{i + 1}/{len(subjects)}] {subj}: already exists in dst, skipping (use --overwrite to redo)")
            continue
        num_frames, abs_max, new_scale = fix_subject(
            src_subject_dir, dst_subject_dir, save_voxel_mean=not args.no_save_voxel_mean
        )
        print(f"[{i + 1}/{len(subjects)}] {subj}: {num_frames} frames, "
              f"post-demean abs_max={abs_max:.4f}, new_scale={new_scale:.6f} -> {dst_subject_dir}/data.pt")

    print(f"\ndone. New dataset root: {args.dst_root}")
    print("Point a dataset yaml's image_path at this folder (e.g. copy scripts/configs/datasets/michigan.yaml")
    print("to a new file with image_path updated) to fine-tune NeuroSTORM on the per-voxel-demeaned data.")


if __name__ == "__main__":
    main()

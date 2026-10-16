"""
scripts/visualize_normalization_fix.py

Q3: save side-by-side images comparing processed_output_v2 (original, global
per-subject scalar normalization) against processed_output_sub_voxel_norm
(corrected, per-voxel demeaned -- see project doc section 15 / the docstring
of fix_voxelwise_normalization.py) so the difference is visible at a glance,
for dropping into the lab-meeting deck.

Produces three PNGs in --out_dir:

  1. single_subject_slices.png
     One subject, one frame, three orthogonal mid-slices (axial / coronal /
     sagittal). Top row = processed_output_v2, bottom row = sub_voxel_norm.
     This is the most direct "here's what changed for one scan" comparison.

  2. fingerprint_removed.png
     The actual per-voxel temporal-mean map that was SUBTRACTED for that
     subject (mid-axial slice) -- i.e. the "spatial fingerprint" itself, not
     just its effect. Uses the voxel_time_mean_removed.pt that
     fix_voxelwise_normalization.py already saved next to each corrected
     subject's data.pt (falls back to computing it from processed_output_v2
     on the fly if that file isn't present). This is the clearest single
     image for explaining WHAT was removed and why, in a talk.

  3. multi_subject_grid.png
     The same mid-axial slice and frame index, across several subjects,
     side by side. Top row = processed_output_v2, bottom row =
     sub_voxel_norm. Illustrates the between-subject confound directly: in
     v2, overall brightness/contrast patterns vary subject-to-subject in a
     way that isn't task-related; in sub_voxel_norm they look more uniform.

Usage (run on the machine that has the actual data -- CPU only, no GPU
needed, just torch + matplotlib):

    python scripts/visualize_normalization_fix.py \\
        --v2_root /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_v2 \\
        --norm_root /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_sub_voxel_norm \\
        --out_dir ./normalization_fix_figures

Optional flags: --subject (defaults to the first subject common to both
folders), --frame_index (defaults to the middle frame of that subject's
scan), --n_subjects (how many subjects in the grid figure, default 6),
--subjects_for_grid (comma-separated list to pick specific ones instead of
the first N found).
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root = parent of this script's directory (scripts/), so `datasets.*`
# resolves the same way it does when main.py is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.fmri_datasets import BaseDataset  # noqa: E402


def load_frame(root, subject, frame_index=None):
    """Dequantize one subject's data.pt and return one frame [H,W,D] plus
    the frame index actually used and the total frame count."""
    blob_path = os.path.join(root, "img", subject, "data.pt")
    blob = BaseDataset._load_blob_file(blob_path, mmap=True)
    frames = blob["frames"]
    scale = float(blob["scale"])
    num_frames = int(blob["num_frames"])
    if frame_index is None:
        frame_index = num_frames // 2
    frame_index = min(frame_index, num_frames - 1)
    frame = frames[frame_index].to(torch.float32) * scale
    return frame.numpy(), frame_index, num_frames


def load_voxel_mean(norm_root, v2_root, subject):
    """Return the per-voxel temporal-mean map that was subtracted for this
    subject: prefer the file fix_voxelwise_normalization.py already saved,
    else compute it from processed_output_v2 directly."""
    saved_path = os.path.join(norm_root, "img", subject, "voxel_time_mean_removed.pt")
    if os.path.exists(saved_path):
        return torch.load(saved_path, map_location="cpu").numpy()

    blob_path = os.path.join(v2_root, "img", subject, "data.pt")
    blob = BaseDataset._load_blob_file(blob_path, mmap=False)
    data = blob["frames"].to(torch.float32) * float(blob["scale"])
    return data.mean(dim=0).numpy()


def mid_slices(volume):
    """volume: [H,W,D] -> (axial, coronal, sagittal) mid-slices as 2D arrays."""
    h, w, d = volume.shape
    axial = volume[:, :, d // 2]
    coronal = volume[:, w // 2, :]
    sagittal = volume[h // 2, :, :]
    return axial, coronal, sagittal


def common_subjects(v2_root, norm_root):
    v2_subs = set(os.listdir(os.path.join(v2_root, "img")))
    norm_subs = set(os.listdir(os.path.join(norm_root, "img")))
    return sorted(v2_subs & norm_subs)


def make_single_subject_figure(v2_root, norm_root, subject, frame_index, out_path):
    v2_frame, used_idx, num_frames = load_frame(v2_root, subject, frame_index)
    norm_frame, _, _ = load_frame(norm_root, subject, used_idx)

    v2_slices = mid_slices(v2_frame)
    norm_slices = mid_slices(norm_frame)
    titles = ["Axial", "Coronal", "Sagittal"]

    # shared color scale per column so the two rows are visually comparable
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for col in range(3):
        vmax = max(np.abs(v2_slices[col]).max(), np.abs(norm_slices[col]).max())
        vmax = vmax if vmax > 0 else 1.0
        axes[0, col].imshow(np.rot90(v2_slices[col]), cmap="gray", vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(titles[col])
        axes[0, col].axis("off")
        axes[1, col].imshow(np.rot90(norm_slices[col]), cmap="gray", vmin=-vmax, vmax=vmax)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("processed_output_v2", fontsize=11)
    axes[1, 0].set_ylabel("sub_voxel_norm", fontsize=11)
    for row, label in enumerate(["processed_output_v2 (original)", "processed_output_sub_voxel_norm (corrected)"]):
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                           rotation=90, va="center", ha="center", fontsize=11)

    fig.suptitle(f"Subject {subject}, frame {used_idx}/{num_frames - 1}: before vs. after per-voxel demeaning", fontsize=13)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fingerprint_figure(v2_root, norm_root, subject, out_path):
    voxel_mean = load_voxel_mean(norm_root, v2_root, subject)
    axial, coronal, sagittal = mid_slices(voxel_mean)
    vmax = max(np.abs(axial).max(), np.abs(coronal).max(), np.abs(sagittal).max())
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, sl, title in zip(axes, [axial, coronal, sagittal], ["Axial", "Coronal", "Sagittal"]):
        im = ax.imshow(np.rot90(sl), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="per-voxel temporal mean (removed)")
    fig.suptitle(f"Subject {subject}: the per-subject 'spatial fingerprint' subtracted by the fix", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_multi_subject_figure(v2_root, norm_root, subjects, frame_index, out_path):
    n = len(subjects)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, subject in enumerate(subjects):
        v2_frame, used_idx, _ = load_frame(v2_root, subject, frame_index)
        norm_frame, _, _ = load_frame(norm_root, subject, used_idx)
        v2_axial = mid_slices(v2_frame)[0]
        norm_axial = mid_slices(norm_frame)[0]
        vmax = max(np.abs(v2_axial).max(), np.abs(norm_axial).max())
        vmax = vmax if vmax > 0 else 1.0

        axes[0, col].imshow(np.rot90(v2_axial), cmap="gray", vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(subject, fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(np.rot90(norm_axial), cmap="gray", vmin=-vmax, vmax=vmax)
        axes[1, col].axis("off")

    axes[0, 0].text(-0.4, 0.5, "v2", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=10)
    axes[1, 0].text(-0.4, 0.5, "sub_voxel_norm", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle("Same mid-axial slice, same frame, across subjects: before vs. after", fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2_root", required=True, help="Path to processed_output_v2")
    parser.add_argument("--norm_root", required=True, help="Path to processed_output_sub_voxel_norm")
    parser.add_argument("--out_dir", default="./normalization_fix_figures")
    parser.add_argument("--subject", default=None, help="Subject folder name for figures 1 & 2 (default: first common subject)")
    parser.add_argument("--frame_index", type=int, default=None, help="Frame index to visualize (default: middle frame)")
    parser.add_argument("--n_subjects", type=int, default=6, help="How many subjects in the grid figure")
    parser.add_argument("--subjects_for_grid", default=None, help="Comma-separated subject list for the grid figure (overrides --n_subjects)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    subjects = common_subjects(args.v2_root, args.norm_root)
    if not subjects:
        raise SystemExit("No subject folders found in common between --v2_root and --norm_root.")

    subject = args.subject or subjects[0]
    if subject not in subjects:
        raise SystemExit(f"Subject '{subject}' not found in both --v2_root and --norm_root. "
                          f"Available: {subjects}")

    if args.subjects_for_grid:
        grid_subjects = [s.strip() for s in args.subjects_for_grid.split(",") if s.strip()]
    else:
        grid_subjects = subjects[: args.n_subjects]

    make_single_subject_figure(
        args.v2_root, args.norm_root, subject, args.frame_index,
        os.path.join(args.out_dir, "single_subject_slices.png"),
    )
    make_fingerprint_figure(
        args.v2_root, args.norm_root, subject,
        os.path.join(args.out_dir, "fingerprint_removed.png"),
    )
    make_multi_subject_figure(
        args.v2_root, args.norm_root, grid_subjects, args.frame_index,
        os.path.join(args.out_dir, "multi_subject_grid.png"),
    )

    print(f"\nAll figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()

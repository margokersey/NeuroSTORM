"""
check_xcpd_masking.py

Answers one empirical question that decides whether re-applying the
fMRIPrep brain mask to XCP-D-denoised BOLD is necessary, or whether
XCP-D has already zeroed everything outside the brain (in which case
NeuroSTORM's data==0 convention -- the same one already used for
HCP-family/UKB datasets in preprocessing_volume.py's NO_MASK_DATASETS
set -- would work just as well, with no cross-directory mask lookup at
all).

What it does, for ONE subject's raw (pre-NeuroSTORM-preprocessing) files:
  1. Loads the XCP-D desc-denoised_bold.nii.gz and the fMRIPrep
     desc-brain_mask.nii.gz from the SIBLING fmriprep_output/ directory
     (same lookup convention as _mask_path_for() in
     datasets/preprocessing_volume.py).
  2. Reports intensity statistics INSIDE vs OUTSIDE the fMRIPrep mask,
     on the time-averaged BOLD volume.
  3. Prints a plain-language verdict:
       - if outside-mask voxels are already ~0 (mean/std near 0, most
         voxels exactly 0) -> XCP-D already masks its own output;
         dropping the external fMRIPrep mask (data==0 background) is
         safe and removes the cross-directory dependency entirely.
       - otherwise -> XCP-D's output still carries real (nonzero, non-
         random) signal outside the mask; you need SOME brain mask, and
         the fMRIPrep one (XCP-D's own input mask) is the principled
         choice, not an arbitrary substitute.
  4. Also reports a Dice overlap between the fMRIPrep mask footprint
     and a data-derived "nonzero signal" footprint, as a sanity check
     on how tightly the two agree (only meaningful if XCP-D does zero
     non-brain voxels -- see point 3).

Usage:
    python check_xcpd_masking.py \
      --bold /path/to/xcp_output/.../sub-02_..._desc-denoised_bold.nii.gz \
      --mask /path/to/fmriprep_output/.../sub-02_..._desc-brain_mask.nii.gz

If --mask is omitted, it's derived automatically using the same
_desc-denoised_bold.nii.gz -> ../../fmriprep_output/<...>_desc-brain_mask.nii.gz
convention preprocessing_volume.py uses for michigan_xcpd.
"""
import argparse
import os

import nibabel as nib
import numpy as np


def derive_mask_path(bold_path):
    suffix = "_desc-denoised_bold.nii.gz"
    if not bold_path.endswith(suffix):
        return None
    stem = bold_path[: -len(suffix)]
    mask_name = os.path.basename(stem) + "_desc-brain_mask.nii.gz"
    xcp_dir = os.path.dirname(bold_path)
    parent_dir = os.path.dirname(xcp_dir)
    return os.path.join(parent_dir, "fmriprep_output", mask_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bold", required=True, help="XCP-D desc-denoised_bold.nii.gz path")
    ap.add_argument("--mask", help="fMRIPrep desc-brain_mask.nii.gz path (auto-derived if omitted)")
    args = ap.parse_args()

    mask_path = args.mask or derive_mask_path(args.bold)
    if mask_path is None:
        raise SystemExit("Could not auto-derive --mask from --bold's filename; pass --mask explicitly.")
    print(f"BOLD: {args.bold}")
    print(f"Mask: {mask_path}\n")

    bold_img = nib.load(args.bold)
    bold = bold_img.get_fdata()
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()

    print(f"BOLD shape: {bold.shape}   Mask shape: {mask.shape}")
    if mask.shape[:3] != bold.shape[:3]:
        print("*** SHAPE MISMATCH -- this pair would fail the assertion in "
              "_read_task_io() and can't be used together at all. ***")
        return
    print("Shapes match (grid-compatible).\n")

    mask_bool = mask > 0.5
    mean_vol = bold.mean(axis=3)  # time-averaged volume

    inside = mean_vol[mask_bool]
    outside = mean_vol[~mask_bool]

    def report(name, arr):
        n = arr.size
        n_exact_zero = int(np.sum(arr == 0))
        print(f"  {name:8s}  n={n:>9d}   mean={arr.mean():+.4f}   std={arr.std():.4f}   "
              f"max|x|={np.abs(arr).max():.4f}   exact-zero frac={n_exact_zero/n:.3f}")

    print("Time-averaged BOLD intensity, split by fMRIPrep mask:")
    report("INSIDE", inside)
    report("OUTSIDE", outside)

    outside_frac_zero = float(np.mean(outside == 0))
    outside_mean_abs = float(np.abs(outside).mean())
    inside_mean_abs = float(np.abs(inside).mean())

    # data-derived "nonzero signal" footprint, for the Dice check
    signal_bool = mean_vol != 0
    intersection = np.sum(mask_bool & signal_bool)
    dice = 2 * intersection / (mask_bool.sum() + signal_bool.sum() + 1e-8)
    print(f"\nDice(fMRIPrep mask, XCP-D-nonzero footprint) = {dice:.3f}")

    print("\n" + "=" * 70)
    if outside_frac_zero > 0.95 and outside_mean_abs < 0.02 * inside_mean_abs:
        print("VERDICT: XCP-D's own output already zeroes almost everything outside")
        print("the fMRIPrep mask. External re-masking is REDUNDANT here -- switching")
        print("michigan_xcpd to the data==0 background convention (like the")
        print("NO_MASK_DATASETS group) would very likely give the same result,")
        print("without needing the fmriprep_output/ cross-directory lookup at all.")
    else:
        print("VERDICT: XCP-D's output still carries non-trivial signal outside the")
        print("fMRIPrep mask (not just noise-level residue). Dropping the mask and")
        print("using data==0 as background would let that signal into the global")
        print("z-score normalization as 'foreground' -- keep using the fMRIPrep mask.")
    print("=" * 70)


if __name__ == "__main__":
    main()

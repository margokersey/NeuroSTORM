"""Print TR (frame) counts per scan from NIfTI headers only (no image data is read).

Usage (cluster; pass an explicit glob, no recursion is done for you):
  python scripts/count_scan_trs.py \
    "/nfs/turbo/umms-huanglab/pipeline/postproc/.../noGSR_highpass_fd-0.8_nosmooth/sub-*/func/*task-imagery_run-2*_bold.nii.gz"
Prints one line per file plus a summary of the distinct lengths.
"""
import sys
from collections import Counter
from glob import glob

import nibabel as nib

files = sorted(glob(sys.argv[1]))
if not files:
    sys.exit("no files matched")
counts = Counter()
for f in files:
    n = nib.load(f).header.get_data_shape()
    t = n[3] if len(n) > 3 else 1
    tr = nib.load(f).header.get_zooms()[3] if len(n) > 3 else None
    counts[t] += 1
    print(f"{t:5d} TRs  (TR={tr}s)  {f}")
print("\nDistinct lengths:", dict(counts))

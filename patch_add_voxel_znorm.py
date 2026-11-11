"""
Add 'voxel-znorm' as a third --scaling_method option to
datasets/preprocessing_volume.py, alongside the existing 'z-norm' (global
scalar) and 'minmax'.

'voxel-znorm' is the standard per-voxel/temporal fMRI normalization (e.g.
nilearn's default standardize=True): each voxel is centered and scaled by
its OWN mean/std computed across time, instead of sharing one global
mean/std with every other foreground voxel the way 'z-norm' does. See the
inline comment this patch adds for why that matters here.

This is purely additive:
  - Default --scaling_method stays 'z-norm' -- existing behavior and any
    existing invocation without --scaling_method is completely unchanged.
  - The two normalization methods are ALTERNATIVES, not something to stack:
    z-scoring a voxel's own time series removes any prior constant
    shift/scale applied to it (per-voxel z-score is invariant to whatever
    affine transform came before, per voxel), so running 'z-norm' and then
    'voxel-znorm' on top would give the identical result to running
    'voxel-znorm' alone on the raw data -- the global step would be
    completely undone. Pick one via --scaling_method, don't chain them.

Run from the repo root:
    python patch_add_voxel_znorm.py
"""
import io

TARGET = "datasets/preprocessing_volume.py"

OLD_ARGPARSE = (
    "    parser.add_argument('--prefetch_chunk_size', type=int, default=0,\n"
    "                        help='files per worker chunk for IO lookahead. 0 = auto '\n"
    "                             '(~ceil(N / (num_processes*4))). Larger means more prefetch '\n"
    "                             'benefit but worse load balancing. Set to 1 to disable prefetch.')\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    dataset_name = args.dataset_name\n"
    "    load_root = args.load_root\n"
    "    save_root = args.save_root\n"
    "    scaling_method = 'z-norm'"
)

NEW_ARGPARSE = (
    "    parser.add_argument('--prefetch_chunk_size', type=int, default=0,\n"
    "                        help='files per worker chunk for IO lookahead. 0 = auto '\n"
    "                             '(~ceil(N / (num_processes*4))). Larger means more prefetch '\n"
    "                             'benefit but worse load balancing. Set to 1 to disable prefetch.')\n"
    "    parser.add_argument('--scaling_method', type=str, default='z-norm',\n"
    "                        choices=['z-norm', 'minmax', 'voxel-znorm'],\n"
    "                        help=\"foreground intensity normalization. 'z-norm' (default, unchanged \"\n"
    "                             \"behavior): one global mean/std shared by every foreground voxel. \"\n"
    "                             \"'voxel-znorm': each voxel z-scored independently against its own \"\n"
    "                             \"mean/std across time -- the standard fMRI per-voxel/temporal \"\n"
    "                             \"normalization (e.g. nilearn's standardize=True). 'minmax': global \"\n"
    "                             \"min-max scaling. These are alternatives -- pick one, they are not \"\n"
    "                             \"meant to be chained.\")\n"
    "    args = parser.parse_args()\n"
    "\n"
    "    dataset_name = args.dataset_name\n"
    "    load_root = args.load_root\n"
    "    save_root = args.save_root\n"
    "    scaling_method = args.scaling_method"
)

OLD_NORM = (
    "    if scaling_method == 'z-norm':\n"
    "        global_mean = data[~background].mean()\n"
    "        global_std = data[~background].std()\n"
    "        data_temp = (data - global_mean) / global_std\n"
    "    elif scaling_method == 'minmax':\n"
    "        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())"
)

NEW_NORM = (
    "    if scaling_method == 'z-norm':\n"
    "        global_mean = data[~background].mean()\n"
    "        global_std = data[~background].std()\n"
    "        data_temp = (data - global_mean) / global_std\n"
    "    elif scaling_method == 'minmax':\n"
    "        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())\n"
    "    elif scaling_method == 'voxel-znorm':\n"
    "        # Per-voxel (per-timeseries) z-score, a.k.a. standard fMRI \"temporal\"\n"
    "        # or \"voxelwise\" normalization (e.g. nilearn's default\n"
    "        # standardize=True): each voxel is centered/scaled by its OWN\n"
    "        # mean/std across time, instead of the one global scalar mean/std\n"
    "        # every foreground voxel shares in the 'z-norm' branch above. This\n"
    "        # removes each voxel's baseline intensity level and noise scale\n"
    "        # individually -- exactly what a single global scalar can't touch,\n"
    "        # since those differences (tissue type, coil sensitivity, distance\n"
    "        # from coil, susceptibility dropout) vary spatially rather than\n"
    "        # being one brain-wide constant.\n"
    "        #\n"
    "        # Background voxels were already zeroed for every frame above\n"
    "        # (data[background] = 0), so their own time series is a constant\n"
    "        # zero and std() is exactly 0 there. Clamping the denominator avoids\n"
    "        # a divide-by-zero/NaN; those voxels come out as 0 either way and\n"
    "        # get overwritten by the background fill value below regardless.\n"
    "        voxel_mean = data.mean(dim=-1, keepdim=True)\n"
    "        voxel_std = data.std(dim=-1, keepdim=True).clamp(min=1e-6)\n"
    "        data_temp = (data - voxel_mean) / voxel_std\n"
    "    else:\n"
    "        raise ValueError(f\"unknown scaling_method: {scaling_method!r}\")"
)


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    checks = (
        (OLD_ARGPARSE, "argparse + scaling_method assignment block"),
        (OLD_NORM, "z-norm/minmax if/elif normalization block"),
    )
    for old, label in checks:
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_ARGPARSE, NEW_ARGPARSE)
    src = src.replace(OLD_NORM, NEW_NORM)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added --scaling_method {z-norm,minmax,voxel-znorm} CLI flag (default: z-norm, unchanged)")
    print("  - added the voxel-znorm per-voxel/temporal normalization branch")


if __name__ == "__main__":
    main()

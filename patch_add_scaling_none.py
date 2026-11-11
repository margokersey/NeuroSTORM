"""
Add 'none' as a fourth --scaling_method option to
datasets/preprocessing_volume.py, for data that's already been normalized
upstream (e.g. AFNI output that's already voxel-wise z-scored) and should
pass through this script's normalization step unchanged.

This does TWO things, not just one -- adding the 'none' branch alone is not
enough:

  1. Adds `elif scaling_method == 'none': data_temp = data` so no additional
     normalization is applied.

  2. Guards the existing, UNCONDITIONAL `data[data < 0] = 0` line (which runs
     before scaling_method is even considered) to skip when
     scaling_method == 'none'. That line assumes raw BOLD magnitude
     intensities, which are non-negative in native scanner units -- true for
     every other scaling_method here, but NOT true for data that arrives
     already z-scored (roughly half its values are negative by
     construction). Without this guard, feeding already-normalized AFNI
     data through with scaling_method='none' would silently zero out about
     half of every voxel's time series before your 'none' pass-through ever
     ran.

As with voxel-znorm, 'none' is an ALTERNATIVE to 'z-norm'/'voxel-znorm', not
something to run in addition to them -- see the comment this patch adds.

This is purely additive: default --scaling_method stays 'z-norm', and
'z-norm'/'minmax'/'voxel-znorm' behavior (including the data<0 clip that
matters for them) is completely unchanged.

Run from the repo root:
    python patch_add_scaling_none.py
"""
import io

TARGET = "datasets/preprocessing_volume.py"

OLD_CLIP = (
    "    data[background] = 0\n"
    "    data[data < 0] = 0\n"
    "    data = torch.Tensor(data)"
)

NEW_CLIP = (
    "    data[background] = 0\n"
    "    if scaling_method != 'none':\n"
    "        # Only valid for raw (not-yet-normalized) BOLD magnitude intensities,\n"
    "        # which are non-negative in native scanner units. Data that arrives\n"
    "        # already normalized (scaling_method='none', e.g. AFNI output that's\n"
    "        # already been voxel-wise z-scored) is roughly half negative by\n"
    "        # construction -- zeroing every negative value here would silently\n"
    "        # destroy about half of every voxel's time series. Skip it for that\n"
    "        # case.\n"
    "        data[data < 0] = 0\n"
    "    data = torch.Tensor(data)"
)

OLD_ELSE = (
    "        voxel_mean = data.mean(dim=-1, keepdim=True)\n"
    "        voxel_std = data.std(dim=-1, keepdim=True).clamp(min=1e-6)\n"
    "        data_temp = (data - voxel_mean) / voxel_std\n"
    "    else:"
)

NEW_ELSE = (
    "        voxel_mean = data.mean(dim=-1, keepdim=True)\n"
    "        voxel_std = data.std(dim=-1, keepdim=True).clamp(min=1e-6)\n"
    "        data_temp = (data - voxel_mean) / voxel_std\n"
    "    elif scaling_method == 'none':\n"
    "        # Data already normalized upstream (e.g. AFNI voxel-wise z-scoring)\n"
    "        # -- pass it through unchanged. This is a separate ALTERNATIVE to\n"
    "        # 'z-norm'/'voxel-znorm', not an additional step on top of them:\n"
    "        # running 'voxel-znorm' on already-per-voxel-normalized data would\n"
    "        # be close to a no-op (z-scoring is invariant to whatever affine\n"
    "        # transform was already applied to that same voxel's own series),\n"
    "        # and running 'z-norm' on top would NOT be a no-op -- it would\n"
    "        # still apply one extra global scalar shift/scale, which is\n"
    "        # redundant at best and needlessly changes the int8 quantization\n"
    "        # scale computed below.\n"
    "        data_temp = data\n"
    "    else:"
)

OLD_ARGPARSE_CHOICES = (
    "    parser.add_argument('--scaling_method', type=str, default='z-norm',\n"
    "                        choices=['z-norm', 'minmax', 'voxel-znorm'],\n"
    "                        help=\"foreground intensity normalization. 'z-norm' (default, unchanged \"\n"
    "                             \"behavior): one global mean/std shared by every foreground voxel. \"\n"
    "                             \"'voxel-znorm': each voxel z-scored independently against its own \"\n"
    "                             \"mean/std across time -- the standard fMRI per-voxel/temporal \"\n"
    "                             \"normalization (e.g. nilearn's standardize=True). 'minmax': global \"\n"
    "                             \"min-max scaling. These are alternatives -- pick one, they are not \"\n"
    "                             \"meant to be chained.\")"
)

NEW_ARGPARSE_CHOICES = (
    "    parser.add_argument('--scaling_method', type=str, default='z-norm',\n"
    "                        choices=['z-norm', 'minmax', 'voxel-znorm', 'none'],\n"
    "                        help=\"foreground intensity normalization. 'z-norm' (default, unchanged \"\n"
    "                             \"behavior): one global mean/std shared by every foreground voxel. \"\n"
    "                             \"'voxel-znorm': each voxel z-scored independently against its own \"\n"
    "                             \"mean/std across time -- the standard fMRI per-voxel/temporal \"\n"
    "                             \"normalization (e.g. nilearn's standardize=True). 'minmax': global \"\n"
    "                             \"min-max scaling. 'none': data is already normalized upstream (e.g. \"\n"
    "                             \"AFNI voxel-wise z-scoring) and is passed through unchanged -- also \"\n"
    "                             \"skips the raw-intensity data<0 clip, which assumes non-negative \"\n"
    "                             \"native BOLD magnitude values. These are all alternatives -- pick \"\n"
    "                             \"one, they are not meant to be chained.\")"
)


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    checks = (
        (OLD_CLIP, "background-zero + data<0 clip block"),
        (OLD_ELSE, "voxel-znorm branch + trailing else"),
        (OLD_ARGPARSE_CHOICES, "--scaling_method argparse block"),
    )
    for old, label in checks:
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_CLIP, NEW_CLIP)
    src = src.replace(OLD_ELSE, NEW_ELSE)
    src = src.replace(OLD_ARGPARSE_CHOICES, NEW_ARGPARSE_CHOICES)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added scaling_method='none' (pass-through, for pre-normalized data)")
    print("  - guarded the data<0 clip to skip when scaling_method='none'")
    print("  - added 'none' to the --scaling_method CLI choices")


if __name__ == "__main__":
    main()

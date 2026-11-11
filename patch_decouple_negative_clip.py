"""
Decouple the raw-intensity `data[data < 0] = 0` clip in
datasets/preprocessing_volume.py from --scaling_method entirely, and gate it
with its own dedicated flag, --allow_negative_input.

Why this patch is needed (found by inspecting real XCP-D / AFNI output in
AFNI's viewer): that clip's assumption -- "voxel intensities are non-negative
native BOLD magnitude values" -- holds for RAW/undenoised data (e.g.
fMRIPrep's own desc-preproc_bold: motion/distortion-corrected and
normalized, but NOT yet filtered or regressed). It does NOT hold for data
that has already been through temporal filtering, nuisance/confound
regression, or z-scoring upstream. XCP-D's desc-denoised_bold (high-pass
filtered + regressed against motion/WM/CSF/global-signal confounds) and
already-normalized AFNI output are BOTH genuinely, expectedly bipolar --
roughly half of every voxel's time series is negative BY CONSTRUCTION
(deviation below the local mean/trend), not artifact.

The previous patch (patch_add_scaling_none.py) only skipped this clip for
scaling_method == 'none', which does NOT cover the case this project has
actually been running: XCP-D data through the DEFAULT 'z-norm' (and
'voxel-znorm') methods. That means every XCP-D fine-tuning run so far --
including the full 12-condition, 60-run smoothing x normalization x
tpt_strategy sweep -- has had roughly half of every foreground voxel's
signal silently zeroed out before normalization ever ran. This patch fixes
the general case: the clip is now controlled by --allow_negative_input,
completely independent of --scaling_method, so it can be skipped for XCP-D
(or AFNI) data while STILL choosing 'z-norm' or 'voxel-znorm' as the
normalization method.

IMPORTANT: this only fixes the code going forward. Any data.pt files
already generated with the old unconditional clip were built from
half-zeroed signal and need to be REGENERATED (rerun preprocessing_volume.py
with --allow_negative_input) -- rerunning fine-tuning alone on the existing
data.pt files does not fix this.

Run from the repo root:
    python patch_decouple_negative_clip.py
"""
import io

TARGET = "datasets/preprocessing_volume.py"

OLD_UNPACK = (
    "    dataset_name, delete_after_preprocess, filename, load_root, save_root, subj_name, count, scaling_method = task"
)
NEW_UNPACK = (
    "    dataset_name, delete_after_preprocess, filename, load_root, save_root, subj_name, count, scaling_method, allow_negative_input = task"
)

OLD_CLIP = (
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
NEW_CLIP = (
    "    data[background] = 0\n"
    "    if not allow_negative_input:\n"
    "        # Only valid for raw (not-yet-processed-past-detrending) BOLD\n"
    "        # magnitude intensities, which are non-negative in native scanner\n"
    "        # units -- true of e.g. fMRIPrep's own desc-preproc_bold (motion/\n"
    "        # distortion corrected and normalized, but NOT denoised or\n"
    "        # filtered). It is NOT true of data that has been through temporal\n"
    "        # filtering, nuisance/confound regression, or any kind of z-scoring\n"
    "        # upstream -- XCP-D's desc-denoised_bold (high-pass filtered +\n"
    "        # regressed against motion/WM/CSF/global-signal confounds) and\n"
    "        # already-normalized AFNI output both come out genuinely bipolar,\n"
    "        # roughly half negative, and that's real signal (deviation below\n"
    "        # the local mean/trend), not artifact. Zeroing it here would\n"
    "        # silently destroy about half of every voxel's time series. Pass\n"
    "        # --allow_negative_input to skip this clip for that kind of data,\n"
    "        # regardless of which --scaling_method you then apply.\n"
    "        data[data < 0] = 0\n"
    "    data = torch.Tensor(data)"
)

OLD_NONE_BRANCH = (
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
NEW_NONE_BRANCH = (
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
    "        # scale computed below. Almost always paired with\n"
    "        # --allow_negative_input (see above) -- already-normalized data is\n"
    "        # bipolar by construction.\n"
    "        data_temp = data\n"
    "    else:"
)

OLD_ARGPARSE = (
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
    "                             \"one, they are not meant to be chained.\")\n"
    "    args = parser.parse_args()"
)
NEW_ARGPARSE = (
    "    parser.add_argument('--scaling_method', type=str, default='z-norm',\n"
    "                        choices=['z-norm', 'minmax', 'voxel-znorm', 'none'],\n"
    "                        help=\"foreground intensity normalization. 'z-norm' (default, unchanged \"\n"
    "                             \"behavior): one global mean/std shared by every foreground voxel. \"\n"
    "                             \"'voxel-znorm': each voxel z-scored independently against its own \"\n"
    "                             \"mean/std across time -- the standard fMRI per-voxel/temporal \"\n"
    "                             \"normalization (e.g. nilearn's standardize=True). 'minmax': global \"\n"
    "                             \"min-max scaling. 'none': data is already normalized upstream (e.g. \"\n"
    "                             \"AFNI voxel-wise z-scoring) and is passed through unchanged. These \"\n"
    "                             \"are all alternatives -- pick one, they are not meant to be chained. \"\n"
    "                             \"See --allow_negative_input, which is a SEPARATE, independent flag.\")\n"
    "    parser.add_argument('--allow_negative_input', action='store_true',\n"
    "                        help=\"skip the raw-intensity data<0 clip. Default (off) is correct for \"\n"
    "                             \"data that is still raw BOLD magnitude scale (e.g. fMRIPrep's own \"\n"
    "                             \"desc-preproc_bold -- motion/distortion-corrected and normalized \"\n"
    "                             \"but NOT yet denoised or filtered, so still essentially \"\n"
    "                             \"non-negative). Pass this flag for data that has been through \"\n"
    "                             \"temporal filtering, nuisance/confound regression, or any \"\n"
    "                             \"z-scoring upstream -- e.g. XCP-D's desc-denoised_bold, or \"\n"
    "                             \"already-normalized AFNI output -- since that data is genuinely, \"\n"
    "                             \"expectedly bipolar (roughly half negative) and clipping it would \"\n"
    "                             \"destroy real signal, not artifacts. Independent of \"\n"
    "                             \"--scaling_method: set it regardless of whether you then also pick \"\n"
    "                             \"'z-norm', 'voxel-znorm', or 'none'.\")\n"
    "    args = parser.parse_args()"
)

OLD_TASK_APPEND = (
    "        tasks.append((dataset_name, args.delete_after_preprocess, filename, load_root, save_root, subj_name, count, scaling_method))"
)
NEW_TASK_APPEND = (
    "        tasks.append((dataset_name, args.delete_after_preprocess, filename, load_root, save_root, subj_name, count, scaling_method, args.allow_negative_input))"
)


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    checks = (
        (OLD_UNPACK, "task-tuple unpack line"),
        (OLD_CLIP, "background-zero + data<0 clip block"),
        (OLD_NONE_BRANCH, "scaling_method == 'none' branch"),
        (OLD_ARGPARSE, "--scaling_method argparse block"),
        (OLD_TASK_APPEND, "tasks.append(...) line"),
    )
    for old, label in checks:
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_UNPACK, NEW_UNPACK)
    src = src.replace(OLD_CLIP, NEW_CLIP)
    src = src.replace(OLD_NONE_BRANCH, NEW_NONE_BRANCH)
    src = src.replace(OLD_ARGPARSE, NEW_ARGPARSE)
    src = src.replace(OLD_TASK_APPEND, NEW_TASK_APPEND)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added --allow_negative_input, independent of --scaling_method")
    print("  - the data<0 clip is now gated solely by --allow_negative_input")
    print("  - threaded allow_negative_input through the task tuple")


if __name__ == "__main__":
    main()

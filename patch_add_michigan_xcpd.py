"""
Patch datasets/preprocessing_volume.py to add a 'michigan_xcpd' dataset name
for preprocessing the XCP-D denoised task-imagery outputs:

    sub-##_task-imagery_run-1_space-MNI152NLin2009cAsym_res-04_desc-denoised_bold.nii.gz

living in xcp_output/, a SIBLING directory of fmriprep_output/ under the same
parent. Same spatial/temporal resampling, crop, normalization, and int8
quantization as every other dataset here -- the only new wrinkle is that the
brain mask lives in a different folder than the bold file.

Five edits, all additive ('michigan' and 'michigan_resting' behavior unchanged):
  1. New MICHIGAN_XCPD_BOLD_RE regex, matching desc-denoised_bold instead of
     desc-preproc_bold.
  2. _mask_path_for(): new 'michigan_xcpd' branch. Strips '_desc-denoised_bold.nii.gz'
     from the xcp_output path, and looks for '<stem>_desc-brain_mask.nii.gz' in
     the SIBLING fmriprep_output/ directory (i.e. one level up from xcp_output/,
     then into fmriprep_output/) -- since XCP-D's own output folder may not
     contain a mask, and fMRIPrep's mask is valid here as long as space +
     resolution match (both carry space-MNI152NLin2009cAsym_res-04).
  3. determine_subject_name(): same 'sub-02' extraction as 'michigan'/'michigan_resting'.
  4. main()'s per-file allow-list filter: new 'michigan_xcpd' branch using
     MICHIGAN_XCPD_BOLD_RE.
  5. _read_task_io(): a safety check, scoped ONLY to dataset_name=='michigan_xcpd'
     (no effect on any other dataset) -- asserts the raw (pre-resampling) bold
     and mask volumes have matching spatial shape. This is the first time this
     codebase pulls a mask from a different directory than the bold file, so
     it's worth failing loudly and immediately if a given subject's XCP-D and
     fMRIPrep grids ever turn out not to match, rather than silently
     misaligning voxels deep inside the resampling pipeline.

Run from the repo root:
    python patch_add_michigan_xcpd.py
"""
import io

TARGET = "datasets/preprocessing_volume.py"

# --- 1. regex -----------------------------------------------------------
OLD_REGEX = '''MICHIGAN_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-imagery_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)

# Same idea but for the resting-state run (task-rest_run-1) instead of the
# task-imagery run -- used for subject-level phenotype prediction (e.g. sex)
# rather than block-level task classification.
MICHIGAN_RESTING_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-rest_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)'''

NEW_REGEX = '''MICHIGAN_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-imagery_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)

# Same idea but for the resting-state run (task-rest_run-1) instead of the
# task-imagery run -- used for subject-level phenotype prediction (e.g. sex)
# rather than block-level task classification.
MICHIGAN_RESTING_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-rest_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)

# Same task-imagery run, but the XCP-D DENOISED output (desc-denoised_bold)
# living in xcp_output/ instead of fMRIPrep's desc-preproc_bold in
# fmriprep_output/. Same block/label structure as MICHIGAN -- this is a
# preprocessing-source variant, not a new task.
MICHIGAN_XCPD_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-imagery_run-1_space-[^_]+_res-[^_]+_desc-denoised_bold\\.nii\\.gz$'
)'''

# --- 2. mask path ---------------------------------------------------------
OLD_MASK_PATH = '''    if dataset_name in ('michigan', 'michigan_resting'):
        # fMRIPrep convention: *_desc-preproc_bold.nii.gz -> *_desc-brain_mask.nii.gz
        # (identical convention for both the task-imagery and resting-state runs)
        if path.endswith('_desc-preproc_bold.nii.gz'):
            return path[: -len('_desc-preproc_bold.nii.gz')] + '_desc-brain_mask.nii.gz'
        return None'''

NEW_MASK_PATH = '''    if dataset_name in ('michigan', 'michigan_resting'):
        # fMRIPrep convention: *_desc-preproc_bold.nii.gz -> *_desc-brain_mask.nii.gz
        # (identical convention for both the task-imagery and resting-state runs)
        if path.endswith('_desc-preproc_bold.nii.gz'):
            return path[: -len('_desc-preproc_bold.nii.gz')] + '_desc-brain_mask.nii.gz'
        return None
    if dataset_name == 'michigan_xcpd':
        # XCP-D denoised bold lives in xcp_output/; the brain mask is looked up
        # in the SIBLING fmriprep_output/ directory instead (XCP-D's own output
        # folder may not contain one). Valid as long as space + resolution match
        # -- both carry space-<...>_res-<...> tags; see _read_task_io's shape
        # assertion for a runtime check of this assumption.
        if path.endswith('_desc-denoised_bold.nii.gz'):
            stem = path[: -len('_desc-denoised_bold.nii.gz')]
            mask_name = os.path.basename(stem) + '_desc-brain_mask.nii.gz'
            xcp_dir = os.path.dirname(path)
            parent_dir = os.path.dirname(xcp_dir)
            return os.path.join(parent_dir, 'fmriprep_output', mask_name)
        return None'''

# --- 3. subject name --------------------------------------------------------
OLD_SUBJECT_NAME = '''    elif dataset_name in ('michigan', 'michigan_resting'):
        return filename.split('_')[0]  # 'sub-02_task-imagery_run-1_..._bold.nii.gz' -> 'sub-02'
                                        # (same split works for 'sub-02_task-rest_run-1_..._bold.nii.gz')'''

NEW_SUBJECT_NAME = '''    elif dataset_name in ('michigan', 'michigan_resting', 'michigan_xcpd'):
        return filename.split('_')[0]  # 'sub-02_task-imagery_run-1_..._bold.nii.gz' -> 'sub-02'
                                        # (same split works for the resting-state and XCP-D filenames too)'''

# --- 4. main() filter --------------------------------------------------------
OLD_MAIN_FILTER = '''        if dataset_name == 'michigan':
            # Strict allow-list match: only exact task-imagery run-1 bold files pass,
            # regardless of what else (task-rest, run-2/3/4, masks, sidecars) lives
            # in the same folder.
            if not MICHIGAN_BOLD_RE.match(filename):
                continue
        elif dataset_name == 'michigan_resting':
            # Strict allow-list match: only exact resting-state run-1 bold files pass,
            # regardless of what else (task-imagery, run-2/3/4, masks, sidecars) lives
            # in the same folder.
            if not MICHIGAN_RESTING_BOLD_RE.match(filename):
                continue
        elif not (filename.endswith('.nii.gz') or filename.endswith('.nii')) or 'mask' in filename \\
                or 'imagery' in filename or 'task-REST_acq' in filename:
            continue'''

NEW_MAIN_FILTER = '''        if dataset_name == 'michigan':
            # Strict allow-list match: only exact task-imagery run-1 bold files pass,
            # regardless of what else (task-rest, run-2/3/4, masks, sidecars) lives
            # in the same folder.
            if not MICHIGAN_BOLD_RE.match(filename):
                continue
        elif dataset_name == 'michigan_resting':
            # Strict allow-list match: only exact resting-state run-1 bold files pass,
            # regardless of what else (task-imagery, run-2/3/4, masks, sidecars) lives
            # in the same folder.
            if not MICHIGAN_RESTING_BOLD_RE.match(filename):
                continue
        elif dataset_name == 'michigan_xcpd':
            # Strict allow-list match: only exact XCP-D denoised task-imagery run-1
            # bold files pass.
            if not MICHIGAN_XCPD_BOLD_RE.match(filename):
                continue
        elif not (filename.endswith('.nii.gz') or filename.endswith('.nii')) or 'mask' in filename \\
                or 'imagery' in filename or 'task-REST_acq' in filename:
            continue'''

# --- 5. shape safety check in _read_task_io ---------------------------------
OLD_READ_IO = '''    mask_path = _mask_path_for(dataset_name, path)
    if mask_path is not None:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_header = mask_img.header
    else:
        mask_data, mask_header = None, None'''

NEW_READ_IO = '''    mask_path = _mask_path_for(dataset_name, path)
    if mask_path is not None:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_header = mask_img.header
        if dataset_name == 'michigan_xcpd':
            # First time this codebase pulls a mask from a different directory
            # than the bold file (fmriprep_output vs xcp_output) -- fail loudly
            # here if a subject's grids don't actually match, rather than
            # silently misaligning voxels inside the resampling pipeline.
            assert mask_data.shape[:3] == data.shape[:3], (
                f"michigan_xcpd: mask shape {mask_data.shape[:3]} from {mask_path} "
                f"does not match bold shape {data.shape[:3]} from {path} -- "
                f"space/resolution mismatch between xcp_output and fmriprep_output?"
            )
    else:
        mask_data, mask_header = None, None'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    checks = (
        (OLD_REGEX, "MICHIGAN_BOLD_RE / MICHIGAN_RESTING_BOLD_RE block"),
        (OLD_MASK_PATH, "_mask_path_for() michigan/michigan_resting branch"),
        (OLD_SUBJECT_NAME, "determine_subject_name() michigan/michigan_resting branch"),
        (OLD_MAIN_FILTER, "main() per-file filter"),
        (OLD_READ_IO, "_read_task_io() mask-loading block"),
    )
    for old, label in checks:
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_REGEX, NEW_REGEX)
    src = src.replace(OLD_MASK_PATH, NEW_MASK_PATH)
    src = src.replace(OLD_SUBJECT_NAME, NEW_SUBJECT_NAME)
    src = src.replace(OLD_MAIN_FILTER, NEW_MAIN_FILTER)
    src = src.replace(OLD_READ_IO, NEW_READ_IO)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added MICHIGAN_XCPD_BOLD_RE")
    print("  - _mask_path_for() now handles 'michigan_xcpd' (mask from sibling fmriprep_output/)")
    print("  - determine_subject_name() and main()'s filter now handle 'michigan_xcpd'")
    print("  - _read_task_io() now asserts bold/mask shape match for 'michigan_xcpd' only")


if __name__ == "__main__":
    main()

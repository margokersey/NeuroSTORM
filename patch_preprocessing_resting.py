"""
Patch datasets/preprocessing_volume.py to add a 'michigan_resting' dataset
name for preprocessing the resting-state fMRIPrep outputs

    sub-##_task-rest_run-1_space-MNI152NLin2009cAsym_res-04_desc-preproc_bold.nii.gz
    sub-##_task-rest_run-1_space-MNI152NLin2009cAsym_res-04_desc-brain_mask.nii.gz

the exact same way 'michigan' already preprocesses the task-imagery run --
same spatial resampling to 2x2x2mm, temporal resampling to TR=0.8s, center
crop/pad to 96^3, brain-mask-derived foreground, global per-subject z-norm,
symmetric int8 quantization (this is the NeuroSTORM paper's own standard
preprocessing recipe -- the same one used for every other dataset in this
codebase, including its HCP1200/ABCD/UKB subject-level gender/age
benchmarks; no extra per-voxel correction is part of that recipe).

Four purely additive edits (existing 'michigan' behavior is unchanged):
  1. New MICHIGAN_RESTING_BOLD_RE regex (mirrors MICHIGAN_BOLD_RE, matching
     task-rest_run-1 instead of task-imagery_run-1).
  2. _mask_path_for(): 'michigan' -> ('michigan', 'michigan_resting') --
     both use the identical fMRIPrep desc-preproc_bold -> desc-brain_mask
     convention.
  3. determine_subject_name(): same widening -- 'sub-02_..._bold.nii.gz'
     -> 'sub-02' works identically regardless of the task name in the middle.
  4. main()'s per-file allow-list filter: add a 'michigan_resting' branch
     using MICHIGAN_RESTING_BOLD_RE.

Run from the repo root:
    python patch_preprocessing_resting.py
"""
import io

TARGET = "datasets/preprocessing_volume.py"

OLD_REGEX = '''MICHIGAN_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-imagery_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)'''

NEW_REGEX = '''MICHIGAN_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-imagery_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)

# Same idea but for the resting-state run (task-rest_run-1) instead of the
# task-imagery run -- used for subject-level phenotype prediction (e.g. sex)
# rather than block-level task classification.
MICHIGAN_RESTING_BOLD_RE = re.compile(
    r'^sub-[^_]+_task-rest_run-1_space-[^_]+_res-[^_]+_desc-preproc_bold\\.nii\\.gz$'
)'''

OLD_MASK_PATH = '''    if dataset_name == 'michigan':
        # fMRIPrep convention: *_desc-preproc_bold.nii.gz -> *_desc-brain_mask.nii.gz
        if path.endswith('_desc-preproc_bold.nii.gz'):
            return path[: -len('_desc-preproc_bold.nii.gz')] + '_desc-brain_mask.nii.gz'
        return None'''

NEW_MASK_PATH = '''    if dataset_name in ('michigan', 'michigan_resting'):
        # fMRIPrep convention: *_desc-preproc_bold.nii.gz -> *_desc-brain_mask.nii.gz
        # (identical convention for both the task-imagery and resting-state runs)
        if path.endswith('_desc-preproc_bold.nii.gz'):
            return path[: -len('_desc-preproc_bold.nii.gz')] + '_desc-brain_mask.nii.gz'
        return None'''

OLD_SUBJECT_NAME = '''    elif dataset_name == 'michigan':
        return filename.split('_')[0]  # 'sub-02_task-imagery_run-1_..._bold.nii.gz' -> 'sub-02\''''

NEW_SUBJECT_NAME = '''    elif dataset_name in ('michigan', 'michigan_resting'):
        return filename.split('_')[0]  # 'sub-02_task-imagery_run-1_..._bold.nii.gz' -> 'sub-02'
                                        # (same split works for 'sub-02_task-rest_run-1_..._bold.nii.gz')'''

OLD_MAIN_FILTER = '''        if dataset_name == 'michigan':
            # Strict allow-list match: only exact task-imagery run-1 bold files pass,
            # regardless of what else (task-rest, run-2/3/4, masks, sidecars) lives
            # in the same folder.
            if not MICHIGAN_BOLD_RE.match(filename):
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
        elif not (filename.endswith('.nii.gz') or filename.endswith('.nii')) or 'mask' in filename \\
                or 'imagery' in filename or 'task-REST_acq' in filename:
            continue'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    for old, label in ((OLD_REGEX, "MICHIGAN_BOLD_RE block"),
                        (OLD_MASK_PATH, "_mask_path_for() 'michigan' branch"),
                        (OLD_SUBJECT_NAME, "determine_subject_name() 'michigan' branch"),
                        (OLD_MAIN_FILTER, "main() per-file filter")):
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_REGEX, NEW_REGEX)
    src = src.replace(OLD_MASK_PATH, NEW_MASK_PATH)
    src = src.replace(OLD_SUBJECT_NAME, NEW_SUBJECT_NAME)
    src = src.replace(OLD_MAIN_FILTER, NEW_MAIN_FILTER)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added MICHIGAN_RESTING_BOLD_RE")
    print("  - _mask_path_for() and determine_subject_name() now also handle 'michigan_resting'")
    print("  - main()'s file filter now allow-lists task-rest_run-1 bold files for 'michigan_resting'")


if __name__ == "__main__":
    main()

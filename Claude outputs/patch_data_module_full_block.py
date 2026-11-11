"""
Patch datasets/data_module.py to wire up the new MICHIGAN_FULL_BLOCK dataset
name introduced by patch_michigan_full_block.py (which adds the
MichiganFullBlock class to datasets/fmri_datasets.py -- run that patch
first).

Three edits, all additive (no existing MICHIGAN behavior changes):
  1. import MichiganFullBlock alongside Michigan.
  2. get_dataset(): 'MICHIGAN_FULL_BLOCK' -> MichiganFullBlock.
  3. subject-dict builder: also match dataset_name == "MICHIGAN_FULL_BLOCK"
     (same subject-enumeration logic as MICHIGAN; labels are block-level,
     assigned in MichiganFullBlock._set_data, inherited unchanged).

Run from the repo root:
    python patch_data_module_full_block.py
"""
import io

TARGET = "datasets/data_module.py"

OLD_IMPORT = "from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE, Michigan"
NEW_IMPORT = "from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE, Michigan, MichiganFullBlock"

OLD_DISPATCH = """        elif name == 'MICHIGAN':
            return Michigan
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")"""

NEW_DISPATCH = """        elif name == 'MICHIGAN':
            return Michigan
        elif name == 'MICHIGAN_FULL_BLOCK':
            return MichiganFullBlock
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")"""

OLD_SUBJECT_DICT = '''        elif self.hparams.dataset_name == "MICHIGAN":
            # Labels live at the block level (tennis/navigation/squeeze), assigned
            # per-clip inside Michigan._set_data from timing_onehot_task1.tsv, not
            # per-subject -- this dict just needs to enumerate subject folders.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                final_dict[subject] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset MICHIGAN, {} subjects'.format(len(final_dict)))'''

NEW_SUBJECT_DICT = '''        elif self.hparams.dataset_name in ("MICHIGAN", "MICHIGAN_FULL_BLOCK"):
            # Labels live at the block level (tennis/navigation/squeeze), assigned
            # per-clip inside Michigan._set_data (MichiganFullBlock inherits it
            # unchanged) from timing_onehot_task1.tsv, not per-subject -- this
            # dict just needs to enumerate subject folders.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                final_dict[subject] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset {}, {} subjects'.format(self.hparams.dataset_name, len(final_dict)))'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    for old, label in ((OLD_IMPORT, "import line"),
                        (OLD_DISPATCH, "get_dataset() dispatch"),
                        (OLD_SUBJECT_DICT, "subject-dict builder")):
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_IMPORT, NEW_IMPORT)
    src = src.replace(OLD_DISPATCH, NEW_DISPATCH)
    src = src.replace(OLD_SUBJECT_DICT, NEW_SUBJECT_DICT)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - imported MichiganFullBlock")
    print("  - get_dataset('MICHIGAN_FULL_BLOCK') now returns MichiganFullBlock")
    print("  - subject-dict builder now also handles MICHIGAN_FULL_BLOCK")


if __name__ == "__main__":
    main()

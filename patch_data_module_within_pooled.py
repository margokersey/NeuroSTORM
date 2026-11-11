"""
Patch datasets/data_module.py to wire up the new MICHIGAN_WITHIN /
MICHIGAN_POOLED dataset names introduced by patch_michigan_within_pooled.py
(which adds the MichiganSampleSplit class to datasets/fmri_datasets.py --
run that patch first).

Three edits, all additive (no existing MICHIGAN/MICHIGAN_FULL_BLOCK/
MICHIGAN_GENDER behavior changes):
  1. import MichiganSampleSplit alongside the other Michigan classes.
  2. get_dataset(): both 'MICHIGAN_WITHIN' and 'MICHIGAN_POOLED' ->
     MichiganSampleSplit (same class serves both -- it only cares about
     the "<subject>::<part>" keys it's handed, not which of the two
     conditions is asking).
  3. subject-dict builder: new branch for dataset_name in
     ("MICHIGAN_WITHIN", "MICHIGAN_POOLED") that enumerates subject
     folders same as MICHIGAN, but emits THREE virtual keys per real
     subject ("<subject>::train", "<subject>::val", "<subject>::test")
     instead of one -- the split file's train/val/test subject lists then
     pick out which of those keys go where. For MICHIGAN_POOLED every
     subject's 3 keys are split across all three lists (pooled/mixed).
     For MICHIGAN_WITHIN a given split_fixed_<N>.txt only lists ONE real
     subject's 3 keys at all (one subject per fold => one model per
     subject) -- see make_within_pooled_splits.py, which generates both
     kinds of split file.

Run from the repo root:
    python patch_data_module_within_pooled.py
"""
import io

TARGET = "datasets/data_module.py"

OLD_IMPORT = "from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE, Michigan, MichiganFullBlock, MichiganGender"
NEW_IMPORT = "from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE, Michigan, MichiganFullBlock, MichiganGender, MichiganSampleSplit"

OLD_DISPATCH = '''        elif name == 'MICHIGAN':
            return Michigan
        elif name == 'MICHIGAN_FULL_BLOCK':
            return MichiganFullBlock
        elif name == 'MICHIGAN_GENDER':
            return MichiganGender
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")'''

NEW_DISPATCH = '''        elif name == 'MICHIGAN':
            return Michigan
        elif name == 'MICHIGAN_FULL_BLOCK':
            return MichiganFullBlock
        elif name == 'MICHIGAN_GENDER':
            return MichiganGender
        elif name == 'MICHIGAN_WITHIN':
            return MichiganSampleSplit
        elif name == 'MICHIGAN_POOLED':
            return MichiganSampleSplit
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")'''

OLD_SUBJECT_DICT = '''        elif self.hparams.dataset_name in ("MICHIGAN", "MICHIGAN_FULL_BLOCK"):
            # Labels live at the block level (tennis/navigation/squeeze), assigned
            # per-clip inside Michigan._set_data (MichiganFullBlock inherits it
            # unchanged) from timing_onehot_task1.tsv, not per-subject -- this
            # dict just needs to enumerate subject folders.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                final_dict[subject] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset {}, {} subjects'.format(self.hparams.dataset_name, len(final_dict)))'''

NEW_SUBJECT_DICT = '''        elif self.hparams.dataset_name in ("MICHIGAN", "MICHIGAN_FULL_BLOCK"):
            # Labels live at the block level (tennis/navigation/squeeze), assigned
            # per-clip inside Michigan._set_data (MichiganFullBlock inherits it
            # unchanged) from timing_onehot_task1.tsv, not per-subject -- this
            # dict just needs to enumerate subject folders.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                final_dict[subject] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset {}, {} subjects'.format(self.hparams.dataset_name, len(final_dict)))

        elif self.hparams.dataset_name in ("MICHIGAN_WITHIN", "MICHIGAN_POOLED"):
            # Same block-level labeling as MICHIGAN, but MichiganSampleSplit
            # needs virtual "<subject>::<part>" keys (part in train/val/test)
            # so the SAME real subject can land in more than one of
            # train_dict/val_dict/test_dict -- see patch_michigan_within_pooled.py
            # and MichiganSampleSplit's docstring. Emit all three parts for
            # every subject; the split file (make_within_pooled_splits.py)
            # decides which of these keys actually end up in train/val/test
            # for a given run.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                for part in ("train", "val", "test"):
                    final_dict[f"{subject}::{part}"] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset {}, {} subjects ({} virtual train/val/test keys)'.format(
                self.hparams.dataset_name, len(subject_list), len(final_dict)))'''


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
    print("  - imported MichiganSampleSplit")
    print("  - get_dataset('MICHIGAN_WITHIN' / 'MICHIGAN_POOLED') now returns MichiganSampleSplit")
    print("  - subject-dict builder now also handles MICHIGAN_WITHIN / MICHIGAN_POOLED")


if __name__ == "__main__":
    main()

"""
Patch datasets/data_module.py to wire up the new MICHIGAN_GENDER dataset
name introduced by patch_michigan_gender.py (which adds the MichiganGender
class to datasets/fmri_datasets.py -- run that patch first, which itself
requires patch_michigan_full_block.py and patch_data_module_full_block.py
to have already been applied).

Three edits, all additive:
  1. import MichiganGender alongside Michigan / MichiganFullBlock.
  2. get_dataset(): 'MICHIGAN_GENDER' -> MichiganGender.
  3. subject-dict builder: new branch that reads participants.tsv (columns:
     participant_id, age, sex, lor_esc, original_id) from
     <image_path>/metadata/participants.tsv and maps each subject folder to
     a binary sex label (1 = M, 0 = F), skipping any subject folder with no
     matching participants.tsv row rather than guessing.

Run from the repo root:
    python patch_data_module_gender.py
"""
import io

TARGET = "datasets/data_module.py"

OLD_IMPORT = "from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE, Michigan, MichiganFullBlock"
NEW_IMPORT = "from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE, Michigan, MichiganFullBlock, MichiganGender"

OLD_DISPATCH = """        elif name == 'MICHIGAN':
            return Michigan
        elif name == 'MICHIGAN_FULL_BLOCK':
            return MichiganFullBlock
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")"""

NEW_DISPATCH = """        elif name == 'MICHIGAN':
            return Michigan
        elif name == 'MICHIGAN_FULL_BLOCK':
            return MichiganFullBlock
        elif name == 'MICHIGAN_GENDER':
            return MichiganGender
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")"""

OLD_SUBJECT_DICT = '''        elif self.hparams.dataset_name in ("MICHIGAN", "MICHIGAN_FULL_BLOCK"):
            # Labels live at the block level (tennis/navigation/squeeze), assigned
            # per-clip inside Michigan._set_data (MichiganFullBlock inherits it
            # unchanged) from timing_onehot_task1.tsv, not per-subject -- this
            # dict just needs to enumerate subject folders.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                final_dict[subject] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset {}, {} subjects'.format(self.hparams.dataset_name, len(final_dict)))

        elif self.hparams.dataset_name == "ABIDE":'''

NEW_SUBJECT_DICT = '''        elif self.hparams.dataset_name in ("MICHIGAN", "MICHIGAN_FULL_BLOCK"):
            # Labels live at the block level (tennis/navigation/squeeze), assigned
            # per-clip inside Michigan._set_data (MichiganFullBlock inherits it
            # unchanged) from timing_onehot_task1.tsv, not per-subject -- this
            # dict just needs to enumerate subject folders.
            subject_list = [subj for subj in os.listdir(img_root)]
            for subject in subject_list:
                final_dict[subject] = [0, 0]  # sex, target -- both unused/overridden

            print('Load dataset {}, {} subjects'.format(self.hparams.dataset_name, len(final_dict)))

        elif self.hparams.dataset_name == "MICHIGAN_GENDER":
            # Subject-level binary sex classification -- labels come from
            # participants.tsv (columns: participant_id, age, sex, lor_esc,
            # original_id), not from timing_onehot_task1.tsv. final_dict is
            # [sex, target] with target == sex (this task predicts sex itself),
            # matching the existing task_name == 'sex' convention used above
            # for HCP1200/ABCD/UKB.
            participants_path = os.path.join(self.hparams.image_path, 'metadata', 'participants.tsv')
            participants_df = pd.read_csv(participants_path, sep='\\t')
            sex_by_subject = dict(zip(participants_df['participant_id'], participants_df['sex']))

            subject_list = [subj for subj in os.listdir(img_root)]
            skipped = []
            for subject in subject_list:
                if subject not in sex_by_subject:
                    skipped.append(subject)
                    continue
                sex_label = 1 if str(sex_by_subject[subject]).strip().upper().startswith('M') else 0
                final_dict[subject] = [sex_label, sex_label]

            if skipped:
                print('Skipped {} subject(s) with no participants.tsv row: {}'.format(len(skipped), skipped))
            print('Load dataset MICHIGAN_GENDER, {} subjects'.format(len(final_dict)))

        elif self.hparams.dataset_name == "ABIDE":'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    for old, label in ((OLD_IMPORT, "import line"),
                        (OLD_DISPATCH, "get_dataset() dispatch"),
                        (OLD_SUBJECT_DICT, "subject-dict builder")):
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- run "
            "patch_data_module_full_block.py first, or the file has changed "
            "unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_IMPORT, NEW_IMPORT)
    src = src.replace(OLD_DISPATCH, NEW_DISPATCH)
    src = src.replace(OLD_SUBJECT_DICT, NEW_SUBJECT_DICT)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - imported MichiganGender")
    print("  - get_dataset('MICHIGAN_GENDER') now returns MichiganGender")
    print("  - subject-dict builder now handles MICHIGAN_GENDER via participants.tsv")


if __name__ == "__main__":
    main()

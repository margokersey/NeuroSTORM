"""
Patch datasets/fmri_datasets.py to add a MichiganGender dataset class for
subject-level binary sex classification on the Michigan task-imagery_run-1
recordings (participants.tsv: participant_id, age, sex, lor_esc, original_id
-- not the block-level timing_onehot_task1.tsv used by Michigan/MichiganFullBlock).

PREREQUISITE: run patch_michigan_full_block.py first (this patch's anchor
text assumes MichiganFullBlock already exists).

Windowing follows the same convention as HCP1200/ABCD's own subject-level
phenotype-prediction classes: slide a fixed sample_duration window across
each subject's whole scan with stride self.stride, assigning every window
that subject's sex label (no block/TSV parsing at all).

Run from the repo root (after patch_michigan_full_block.py):
    python patch_michigan_gender.py
"""
import io

TARGET = "datasets/fmri_datasets.py"

OLD_CLASS_BOUNDARY = '''    TRIM_START = 0


class TransDiag(BaseDataset):'''

NEW_CLASS_BOUNDARY = '''    TRIM_START = 0


class MichiganGender(BaseDataset):
    """Subject-level binary sex classification on the same task-imagery_run-1
    recordings as Michigan/MichiganFullBlock, using participants.tsv
    (columns: participant_id, age, sex, lor_esc, original_id) instead of
    block-level timing_onehot_task1.tsv labels.

    Slides a fixed sample_duration window across each subject's whole scan
    (same windowing convention as HCP1200/ABCD's subject-level phenotype
    prediction), assigning every window that subject's sex label. No TSV
    block parsing -- every frame of the scan is equally valid for this task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data


class TransDiag(BaseDataset):'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    assert src.count(OLD_CLASS_BOUNDARY) == 1, (
        "Expected exactly one match for the MichiganFullBlock/TransDiag class "
        "boundary in %s -- run patch_michigan_full_block.py first, or the file "
        "has changed unexpectedly." % TARGET
    )

    src = src.replace(OLD_CLASS_BOUNDARY, NEW_CLASS_BOUNDARY)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added class MichiganGender(BaseDataset)")


if __name__ == "__main__":
    main()

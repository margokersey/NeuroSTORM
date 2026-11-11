"""
Patch datasets/fmri_datasets.py to add MichiganSampleSplit, which enables
two new evaluation conditions on top of the existing task-imagery_run-1
block design (tennis / navigation / squeeze):

  - MICHIGAN_WITHIN:  one model PER SUBJECT, trained and tested only on
    that subject's own blocks (pure within-subject decoding).
  - MICHIGAN_POOLED:  one model trained on ALL subjects pooled together,
    where each subject contributes SOME blocks to train and OTHER blocks
    to val/test (subjects can appear in both train and test -- the
    "mixed" condition).

Neither of these is expressible with the existing Michigan class, because
data_module.py's _build_single_dataset() partitions subject_dict into
train_dict/val_dict/test_dict purely by whether each SUBJECT NAME appears
in the split file's train/val/test subject lists -- a whole subject always
lands in exactly one of the three. Both new conditions need the SAME
subject to land in more than one, at the level of individual blocks.

The trick (see MichiganSampleSplit's docstring): subject_dict keys become
virtual "<real_subject>::<part>" strings (part in {train, val, test}).
_build_single_dataset's existing membership-based partitioning logic is
completely unchanged -- it just now partitions these virtual keys instead
of raw subject names, which is enough to put the same subject's data into
more than one dict. MichiganSampleSplit._set_data resolves the real
subject name back out for file I/O and keeps only the blocks
assign_blocks_to_parts assigned to that key's part.

This does NOT touch the Michigan / MichiganFullBlock / MichiganGender
classes or any data.pt files -- purely additive.

Run from the repo root (after patch_michigan_full_block.py and
patch_michigan_gender.py, if not already applied):
    python patch_michigan_within_pooled.py
"""
import io

TARGET = "datasets/fmri_datasets.py"

# --- Anchor 1: insert assign_blocks_to_parts() right before class Michigan ---

OLD_MICHIGAN_HEADER = '''class Michigan(BaseDataset):
    """task-imagery_run-1 block design: tennis / navigation / squeeze.'''

NEW_MICHIGAN_HEADER = '''def assign_blocks_to_parts(blocks, fracs, seed=0):
    """Deterministic, class-stratified train/val/test assignment over a
    list of (start, end, label) blocks -- used by MichiganSampleSplit to
    split an individual subject's OWN blocks (for MICHIGAN_WITHIN /
    MICHIGAN_POOLED) instead of only whole-subject membership.

    Stratifying by label keeps every part class-balanced even with as few
    as 10 instances per class (Michigan's task6 has exactly 10 tennis /
    10 navigation / 10 squeeze blocks per subject). The assignment only
    depends on `blocks`, `fracs`, `seed` -- NOT on which subject is
    asking -- so e.g. "block #14 (a navigation instance)" is the same
    held-out test instance for every subject. That keeps every subject's
    train/val/test class balance identical and the whole pooled dataset
    directly comparable subject to subject.

    Returns `part_of_block`, a list parallel to `blocks`, with values in
    {"train", "val", "test"}.
    """
    import random
    by_label = {}
    for i, (_, _, lab) in enumerate(blocks):
        by_label.setdefault(lab, []).append(i)

    part_of_block = [None] * len(blocks)
    rng = random.Random(seed)
    for lab, idxs in sorted(by_label.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = min(int(round(n * fracs["train"])), n)
        n_val = min(int(round(n * fracs["val"])), n - n_train)
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:n_train + n_val]
        test_idxs = idxs[n_train + n_val:]
        for i in train_idxs:
            part_of_block[i] = "train"
        for i in val_idxs:
            part_of_block[i] = "val"
        for i in test_idxs:
            part_of_block[i] = "test"
    return part_of_block


class Michigan(BaseDataset):
    """task-imagery_run-1 block design: tennis / navigation / squeeze.'''

# --- Anchor 2: insert MichiganSampleSplit between MichiganFullBlock and MichiganGender ---

OLD_CLASS_BOUNDARY = '''    """
    TRIM_START = 0


class MichiganGender(BaseDataset):'''

NEW_CLASS_BOUNDARY = '''    """
    TRIM_START = 0


class MichiganSampleSplit(Michigan):
    """Same task-imagery_run-1 block design as Michigan (10 tennis / 10
    navigation / 10 squeeze blocks per subject, from
    timing_onehot_task1.tsv), but splits an individual subject's OWN
    blocks across train/val/test instead of assigning a whole subject to
    exactly one of the three sets. This is what makes both of the
    following possible without touching any data:

      - MICHIGAN_WITHIN:  one model per subject, trained and tested only
        on that subject's own blocks (pure within-subject decoding).
      - MICHIGAN_POOLED:  one model trained on ALL subjects pooled
        together, where each subject contributes some blocks to train
        and other blocks to val/test (the "mixed" condition -- subjects
        can appear in both train and test, unlike Michigan's
        between-subject 5-fold CV).

    subject_dict keys must be virtual "<real_subject>::<part>" strings
    (part in {train, val, test}), as produced by data_module.py's
    MICHIGAN_WITHIN / MICHIGAN_POOLED subject-dict branch -- this is the
    only way to get the SAME subject to land in more than one of
    train_dict/val_dict/test_dict, since _build_single_dataset partitions
    subject_dict purely by key membership in the split file's
    train/val/test subject lists.

    _set_data resolves each virtual key back to its real subject folder
    for file I/O, computes the same 30-block design Michigan does, and
    keeps only the blocks assign_blocks_to_parts assigned to that key's
    `part`. See BLOCK_SPLIT_FRACS / BLOCK_SPLIT_SEED to change the
    per-subject train/val/test ratio -- the same assignment is used for
    every subject (see assign_blocks_to_parts's docstring), so every part
    stays class-balanced and directly comparable subject to subject.

    Default 0.7/0.1/0.2 split -> per subject: 21 train (7/class),
    3 val (1/class), 6 test (2/class) -- test is kept at 2/class rather
    than 1/class since a single-subject "fold" only has 30 blocks total
    to begin with and a 1-sample-per-class test set is too noisy to read
    (only 4 possible accuracy values). Still very little data for a
    ~5M-parameter full fine-tune per subject (MICHIGAN_WITHIN) -- treat
    any single subject's number as noisy and look at the distribution
    across all 26 subjects (e.g. via compare_conditions.py, treating each
    subject as a "fold") rather than trusting one subject's result alone.
    """
    BLOCK_SPLIT_FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}
    BLOCK_SPLIT_SEED = 0

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        tsv_path = os.path.join(root, 'metadata', 'timing_onehot_task1.tsv')
        all_blocks = self._blocks_from_tsv(tsv_path)
        part_of_block = assign_blocks_to_parts(
            all_blocks, self.BLOCK_SPLIT_FRACS, seed=self.BLOCK_SPLIT_SEED)

        for i, key in enumerate(subject_dict):
            if '::' not in key:
                raise ValueError(
                    "MichiganSampleSplit expects '<subject>::<part>' subject_dict "
                    "keys (see data_module.py's MICHIGAN_WITHIN/MICHIGAN_POOLED "
                    "branch), got %r" % (key,))
            real_subject, part = key.rsplit('::', 1)
            subject_path = os.path.join(img_root, real_subject)
            num_frames = self._count_frames(subject_path)
            for bi, (start, end, task_name) in enumerate(all_blocks):
                if part_of_block[bi] != part:
                    continue
                if start + self.sample_duration > end or start + self.sample_duration > num_frames:
                    continue
                label = self.LABEL_MAP[task_name]
                data_tuple = (i, key, subject_path, start, self.stride, num_frames, label, 0)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([t[6] for t in data]).reshape(-1, 1)
        return data


class MichiganGender(BaseDataset):'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    for old, label in ((OLD_MICHIGAN_HEADER, "class Michigan header"),
                        (OLD_CLASS_BOUNDARY, "MichiganFullBlock/MichiganGender class boundary")):
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_MICHIGAN_HEADER, NEW_MICHIGAN_HEADER)
    src = src.replace(OLD_CLASS_BOUNDARY, NEW_CLASS_BOUNDARY)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added module-level assign_blocks_to_parts()")
    print("  - added class MichiganSampleSplit(Michigan)")


if __name__ == "__main__":
    main()

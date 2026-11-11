"""
make_within_pooled_splits.py

Generates the split files for two new evaluation conditions to go with the
existing between-subject 5-fold CV (nas/splits/MICHIGAN/split_fixed_N.txt):

  - MICHIGAN_POOLED: ONE split file, nas/splits/MICHIGAN_POOLED/split_fixed_1.txt.
    Every subject contributes all three of "<subject>::train",
    "<subject>::val", "<subject>::test" to the pooled train/val/test lists
    -- a single model, trained/tested on data pooled across ALL subjects,
    where each subject appears in both train and test via DIFFERENT blocks
    (see MichiganSampleSplit.assign_blocks_to_parts in fmri_datasets.py for
    exactly which of a subject's 30 blocks land in which part -- it's the
    same assignment for every subject: default 21 train / 3 val / 6 test).

  - MICHIGAN_WITHIN: N split files (one per subject),
    nas/splits/MICHIGAN_WITHIN/split_fixed_<k>.txt for k = 1..N. Each file
    lists ONLY that one subject's "::train"/"::val"/"::test" keys -- run
    with --dataset_split_num <k>, this trains and tests a completely
    separate model on that one subject's own blocks only (pure
    within-subject decoding: 21 train / 3 val / 6 test blocks, all from
    the same person). Also writes
    nas/splits/MICHIGAN_WITHIN/fold_to_subject.txt (fold number -> subject
    name) since there's no other way to tell which fold is which subject.

Requires patch_michigan_within_pooled.py and
patch_data_module_within_pooled.py to already be applied (so
MICHIGAN_WITHIN / MICHIGAN_POOLED are real, loadable dataset names) --
this script only WRITES split files, it never imports or touches any
model/dataset code.

Usage (run from the repo root, on the machine with the real data/splits):
    python make_within_pooled_splits.py \\
      --source_split nas/splits/MICHIGAN/split_fixed_1.txt \\
      --split_root nas/splits

--source_split just supplies the roster of subject names -- any one of the
existing MICHIGAN split_fixed_*.txt files works, since they all cover the
same 26 subjects, just partitioned differently into between-subject folds.
"""
import argparse
import os


def load_split_subjects(path):
    """Read train/val/test subject name lists the same way
    fMRIDataModule.load_split() does (data_module.py ~line 487), and
    return their union (order preserved, duplicates dropped) -- the full
    subject roster used by the existing MICHIGAN between-subject splits."""
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]
    train_i = next(i for i, l in enumerate(lines) if "train" in l)
    val_i = next(i for i, l in enumerate(lines) if "val" in l)
    test_i = next(i for i, l in enumerate(lines) if "test" in l)
    train_names = lines[train_i + 1:val_i]
    val_names = lines[val_i + 1:test_i]
    test_names = lines[test_i + 1:]

    roster = []
    seen = set()
    for name in train_names + val_names + test_names:
        if name and name not in seen:
            roster.append(name)
            seen.add(name)
    return roster


def write_split(path, train_keys, val_keys, test_keys):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("train_subjects\n")
        for k in train_keys:
            f.write(k + "\n")
        f.write("val_subjects\n")
        for k in val_keys:
            f.write(k + "\n")
        f.write("test_subjects\n")
        for k in test_keys:
            f.write(k + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_split", default="nas/splits/MICHIGAN/split_fixed_1.txt",
                     help="Any existing MICHIGAN split_fixed_*.txt -- used only to read the subject roster.")
    ap.add_argument("--split_root", default="nas/splits")
    args = ap.parse_args()

    roster = load_split_subjects(args.source_split)
    print(f"Read {len(roster)} subjects from {args.source_split}:")
    print(" ", roster)

    # --- MICHIGAN_POOLED: one split file, all subjects in all three lists ---
    pooled_dir = os.path.join(args.split_root, "MICHIGAN_POOLED")
    pooled_path = os.path.join(pooled_dir, "split_fixed_1.txt")
    train_keys = [f"{s}::train" for s in roster]
    val_keys = [f"{s}::val" for s in roster]
    test_keys = [f"{s}::test" for s in roster]
    write_split(pooled_path, train_keys, val_keys, test_keys)
    print(f"\nWrote {pooled_path}")
    print(f"  {len(train_keys)} subjects x 21 train blocks, "
          f"{len(val_keys)} subjects x 3 val blocks, "
          f"{len(test_keys)} subjects x 6 test blocks "
          f"(MichiganSampleSplit's default 0.7/0.1/0.2 per-subject block split)")

    # --- MICHIGAN_WITHIN: one split file per subject ---
    within_dir = os.path.join(args.split_root, "MICHIGAN_WITHIN")
    fold_map_lines = ["fold\tsubject"]
    for k, subject in enumerate(roster, start=1):
        path = os.path.join(within_dir, f"split_fixed_{k}.txt")
        write_split(path, [f"{subject}::train"], [f"{subject}::val"], [f"{subject}::test"])
        fold_map_lines.append(f"{k}\t{subject}")
    fold_map_path = os.path.join(within_dir, "fold_to_subject.txt")
    with open(fold_map_path, "w") as f:
        f.write("\n".join(fold_map_lines) + "\n")
    print(f"\nWrote {len(roster)} split files to {within_dir}/split_fixed_1.txt .. split_fixed_{len(roster)}.txt")
    print(f"Wrote {fold_map_path} (fold number -> subject name lookup)")

    print("\nExample commands:")
    print("  # Pooled (one run, all subjects):")
    print("  bash scripts/run_experiment.sh --model neurostorm --dataset michigan_pooled --task task6 \\")
    print("    --mode finetune --max_epochs 30 --batch_size 8 --gpus 0 \\")
    print("    --load_model_path output/neurostorm/pt_neurostorm_mae_5ds.ckpt \\")
    print("    -- --tpt_strategy none --project_name michigan_pooled")
    print()
    print("  # Within-subject, one example fold (repeat for --dataset_split_num 1..N,")
    print("  # or see run_within_all_subjects.sh to loop over all of them):")
    print("  bash scripts/run_experiment.sh --model neurostorm --dataset michigan_within --task task6 \\")
    print("    --mode finetune --max_epochs 150 --batch_size 8 --gpus 0 \\")
    print("    --load_model_path output/neurostorm/pt_neurostorm_mae_5ds.ckpt \\")
    print("    -- --dataset_split_num 1 --tpt_strategy none --project_name michigan_within_fold-1")


if __name__ == "__main__":
    main()

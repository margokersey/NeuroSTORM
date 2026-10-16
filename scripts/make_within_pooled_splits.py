"""
make_within_pooled_splits.py

Generates the split files for two new evaluation conditions to go with the
existing between-subject 5-fold CV (nas/splits/MICHIGAN/split_fixed_N.txt):

  - MICHIGAN_POOLED: N_POOLED_FOLDS (default 5) split files,
    nas/splits/MICHIGAN_POOLED/split_fixed_1.txt .. split_fixed_5.txt. Every
    subject contributes all three of "<subject>::train::<seed>",
    "<subject>::val::<seed>", "<subject>::test::<seed>" to that fold's
    pooled train/val/test lists, with a DIFFERENT seed per fold -- so each
    fold independently reshuffles which of a subject's 30 blocks land in
    train/val/test (see MichiganSampleSplit.assign_blocks_to_parts in
    fmri_datasets.py), giving 5 genuinely different pooled models rather
    than 5 copies of the same split. This is what makes "5-fold metrics for
    pooled" meaningful, and what compare_conditions_v2.py's raincloud plots
    and statistical tests (which need 5 paired folds) actually need.
    Requires patch_michigan_pooled_multifold.py to be applied (adds the
    optional "::<seed>" key segment) -- without it, every fold would
    silently use the same fixed split.

  - MICHIGAN_WITHIN: N split files (one per subject),
    nas/splits/MICHIGAN_WITHIN/split_fixed_<k>.txt for k = 1..N. Each file
    lists ONLY that one subject's "::train"/"::val"/"::test" keys (no seed
    segment -- one subject IS the fold here, there's nothing to reshuffle)
    -- run with --dataset_split_num <k>, this trains and tests a
    completely separate model on that one subject's own blocks only (pure
    within-subject decoding: 21 train / 3 val / 6 test blocks, all from
    the same person). Also writes
    nas/splits/MICHIGAN_WITHIN/fold_to_subject.txt (fold number -> subject
    name) since there's no other way to tell which fold is which subject.

Requires, in this order:
    python patch_michigan_within_pooled.py
    python patch_michigan_pooled_multifold.py
    python patch_data_module_within_pooled.py
    python patch_data_module_pooled_multifold.py
so MICHIGAN_WITHIN / MICHIGAN_POOLED are real, loadable dataset names with
seed-aware block splitting -- this script only WRITES split files, it
never imports or touches any model/dataset code.

Usage (run from the repo root, on the machine with the real data/splits):
    python make_within_pooled_splits.py \\
      --source_split nas/splits/MICHIGAN/split_fixed_1.txt \\
      --split_root nas/splits \\
      --n_pooled_folds 5

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
    ap.add_argument("--n_pooled_folds", type=int, default=5,
                     help="Number of independently-reshuffled MICHIGAN_POOLED folds to generate "
                          "(must be <= N_POOLED_FOLDS in data_module.py's subject-dict branch, "
                          "which provisions seeds 0..4 by default).")
    args = ap.parse_args()

    roster = load_split_subjects(args.source_split)
    print(f"Read {len(roster)} subjects from {args.source_split}:")
    print(" ", roster)

    # --- MICHIGAN_POOLED: N_POOLED_FOLDS split files, one seed each ---
    pooled_dir = os.path.join(args.split_root, "MICHIGAN_POOLED")
    for fold in range(1, args.n_pooled_folds + 1):
        seed = fold - 1
        train_keys = [f"{s}::train::{seed}" for s in roster]
        val_keys = [f"{s}::val::{seed}" for s in roster]
        test_keys = [f"{s}::test::{seed}" for s in roster]
        path = os.path.join(pooled_dir, f"split_fixed_{fold}.txt")
        write_split(path, train_keys, val_keys, test_keys)
    # verify pooled split files were written
    expected_files = [os.path.join(pooled_dir, f"split_fixed_{fold}.txt") for fold in range(1, args.n_pooled_folds + 1)]
    missing = [p for p in expected_files if not os.path.isfile(p)]
    if missing:
        print("ERROR: The following pooled split files are missing:")
        for m in missing:
            print("  -", m)
        raise SystemExit("Pooled split file generation failed; see missing files above.")
    print(f"\nWrote {args.n_pooled_folds} pooled split files to "
          f"{pooled_dir}/split_fixed_1.txt .. split_fixed_{args.n_pooled_folds}.txt")
    print(f"  Each fold: {len(roster)} subjects x 21 train blocks, "
          f"{len(roster)} subjects x 3 val blocks, {len(roster)} subjects x 6 test blocks "
          f"(MichiganSampleSplit's default 0.7/0.1/0.2 per-subject block split), "
          f"reshuffled independently per fold via a different seed (0..{args.n_pooled_folds - 1}).")

    # --- MICHIGAN_WITHIN: one split file per subject (no seed -- one subject IS the fold) ---
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
    print("  # Pooled, all 5 folds:")
    print("  bash run_pooled_all_folds.sh --tpt_strategy none \\")
    print("    --load_model_path output/neurostorm/pt_neurostorm_mae_5ds.ckpt")
    print()
    print("  # Within, all subjects:")
    print("  bash run_within_all_subjects.sh --tpt_strategy none \\")
    print("    --load_model_path output/neurostorm/pt_neurostorm_mae_5ds.ckpt")


if __name__ == "__main__":
    main()

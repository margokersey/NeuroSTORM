"""
scripts/make_loocv_splits.py

Generates leave-one-subject-out (LOOCV) split_fixed_<N>.txt files for a
26-subject Michigan dataset, reusing fMRIDataModule's own split-file format
and loader (datasets/data_module.py: save_split()/load_split()) so no code
changes are needed anywhere else -- these files just slot into the existing
--dataset_split_num mechanism.

Why fold numbers start at 101 by default, not 1: the existing 5-fold CV
splits already occupy split_fixed_1.txt .. split_fixed_5.txt under
nas/splits/<DATASET_NAME>/. Writing LOOCV splits at indices 1..26 in that
same directory would silently overwrite (or be shadowed by) those 5-fold
files. --start_index puts LOOCV splits at a disjoint range (101..126 for 26
subjects) in the SAME directory, so both the 5-fold CV and LOOCV splits
coexist under one --dataset_name with zero collision risk.

Each LOOCV fold i holds out exactly one subject as the test set, and rotates
to the NEXT subject (wrapping around) as a single-subject validation set
(used only for best-checkpoint selection via valid_acc) -- the remaining 24
subjects are train. Using a val subject distinct from the test subject avoids
leaking the held-out subject into checkpoint selection.

Usage:
    python "Claude outputs/make_loocv_splits.py" \
      --dataset_name MICHIGAN_FULL_BLOCK \
      --source_fold 1 \
      --start_index 101

Then run 26 fine-tunes, one per fold, passing --dataset_split_num
101..126 (see the command template this script prints at the end).
"""
import argparse
import os


def load_split(path):
    lines = [x.rstrip("\n") for x in open(path, "r").readlines()]
    train_index = next(i for i, line in enumerate(lines) if "train" in line)
    val_index = next(i for i, line in enumerate(lines) if "val" in line)
    test_index = next(i for i, line in enumerate(lines) if "test" in line)
    train_names = lines[train_index + 1:val_index]
    val_names = lines[val_index + 1:test_index]
    test_names = lines[test_index + 1:]
    return train_names, val_names, test_names


def save_split(sets_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w+") as f:
        for name, subj_list in sets_dict.items():
            f.write(name + "\n")
            for subj_name in subj_list:
                f.write(str(subj_name) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_root", default="./nas/splits")
    ap.add_argument("--dataset_name", required=True,
                     help="Must match an existing nas/splits/<name>/split_fixed_<source_fold>.txt directory "
                          "(e.g. MICHIGAN, MICHIGAN_FULL_BLOCK) -- LOOCV splits are written into the SAME directory.")
    ap.add_argument("--source_fold", type=int, default=1,
                     help="Which existing split_fixed_<N>.txt to read the full 26-subject roster from "
                          "(its train+val+test union). Any of the 5 existing folds works -- they all cover "
                          "the same 26 subjects.")
    ap.add_argument("--start_index", type=int, default=101,
                     help="First --dataset_split_num to write LOOCV folds at. Default 101 stays clear of "
                          "the existing 5-fold CV's indices 1-5.")
    ap.add_argument("--dry_run", action="store_true", help="Print the plan without writing files.")
    args = ap.parse_args()

    source_path = os.path.join(args.split_root, args.dataset_name, f"split_fixed_{args.source_fold}.txt")
    if not os.path.exists(source_path):
        raise SystemExit(f"Source split file not found: {source_path}")

    train_names, val_names, test_names = load_split(source_path)
    subjects = sorted(set(train_names) | set(val_names) | set(test_names))
    n = len(subjects)
    print(f"Read {n} unique subjects from {source_path}:")
    print("  " + ", ".join(subjects))

    print(f"\nWriting {n} LOOCV split files to {os.path.join(args.split_root, args.dataset_name)}/ "
          f"as split_fixed_{args.start_index}.txt .. split_fixed_{args.start_index + n - 1}.txt\n")

    print(f"{'dataset_split_num':>18}  {'held-out (test)':>18}  {'val (3-subj)':>30}  train (n)")
    for idx, held_out in enumerate(subjects):
        fold_num = args.start_index + idx
        # choose the next 3 subjects (wrapping) as validation set
        val_indices = [(idx + 1 + j) % n for j in range(3)]
        val_subjs = [subjects[i] for i in val_indices]
        train_subjs = [s for s in subjects if s != held_out and s not in val_subjs]
        print(f"{fold_num:>18}  {held_out:>18}  {', '.join(val_subjs):>30}  {len(train_subjs)}")

        if not args.dry_run:
            out_path = os.path.join(args.split_root, args.dataset_name, f"split_fixed_{fold_num}.txt")
            save_split(
                {"train_subjects": train_subjs, "val_subjects": val_subjs, "test_subjects": [held_out]},
                out_path,
            )

    if args.dry_run:
        print("\n[dry run] no files written.")
    else:
        print(f"\nDone -- {n} split files written.")

    print("\nRun template (fill in --dataset/--task/--tpt_strategy/--load_model_path for your winning condition):")
    print(f"for i in $(seq {args.start_index} {args.start_index + n - 1}); do")
    print(f"  bash scripts/run_experiment.sh --model neurostorm --dataset <DATASET> --task <TASK> --mode finetune \\")
    print(f"    --max_epochs 50 --batch_size 8 --gpus 0 --load_model_path <CKPT> \\")
    print(f"    -- --dataset_split_num $i --tpt_strategy <TPT_STRATEGY> --project_name <PROJECT_BASE>_loocv_fold-$i")
    print("done")


if __name__ == "__main__":
    main()

"""Generate subject-level k-fold split files for the MICHIGAN dataset.

data_module.py's built-in random splitter (`determine_split_randomly`) hardcodes
`np.random.seed(0)`, so asking it for multiple --dataset_split_num values does NOT
give you different folds -- it silently regenerates the identical split every time.
This script writes the split_fixed_N.txt files directly instead, in the exact
format fMRIDataModule.load_split() expects, with real subject-level rotation.

Usage:
    python scripts/make_michigan_splits.py \
        --image_path ./processed_data/michigan \
        --split_root ./nas/splits \
        --n_folds 5 --seed 0
"""
import argparse
import os


def save_split(path, train, val, test):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("train_subjects\n")
        for s in train:
            f.write(s + "\n")
        f.write("val_subjects\n")
        for s in val:
            f.write(s + "\n")
        f.write("test_subjects\n")
        for s in test:
            f.write(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True, help="dataset root, e.g. ./processed_data/michigan")
    ap.add_argument("--split_root", default="./nas/splits")
    ap.add_argument("--dataset_name", default="MICHIGAN")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_root = os.path.join(args.image_path, "img")
    subjects = sorted(os.listdir(img_root))
    print(f"found {len(subjects)} subjects in {img_root}")

    import random
    rng = random.Random(args.seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    # nearly-equal contiguous groups over the shuffled order
    n = args.n_folds
    groups = [shuffled[i::n] for i in range(n)]
    for i, g in enumerate(groups):
        print(f"  group {i}: {len(g)} subjects -> {g}")

    for fold in range(n):
        test = groups[fold]
        val = groups[(fold + 1) % n]
        train = [s for i, g in enumerate(groups) if i not in (fold, (fold + 1) % n) for s in g]
        assert not (set(train) & set(val) & set(test))
        assert set(train) | set(val) | set(test) == set(subjects)

        split_path = os.path.join(args.split_root, args.dataset_name, f"split_fixed_{fold + 1}.txt")
        save_split(split_path, train, val, test)
        print(f"fold {fold + 1}: train={len(train)} val={len(val)} test={len(test)} -> {split_path}")


if __name__ == "__main__":
    main()

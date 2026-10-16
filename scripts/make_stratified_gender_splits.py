"""
Generate sex-stratified train/val/test splits for MICHIGAN_GENDER, replacing
the copied-from-MICHIGAN split (which was built for the tennis/nav/squeeze
task and has no relationship to sex balance -- with only 26 subjects, an
unstratified split can easily hand the model a test fold that's mostly or
entirely one class, which alone explains a chance-level accuracy regardless
of what the model actually learned).

Produces N_FOLDS stratified-by-sex splits (default 5, for 5-fold CV -- a
single fixed 15/5/6 split is too fragile to trust at n=26) in the exact
text format fMRIDataModule expects, e.g. nas/splits/MICHIGAN_GENDER/split_fixed_1.txt:

    train_subjects
    sub-28
    sub-15
    ...
    val_subjects
    sub-02
    ...
    test_subjects
    sub-19
    ...

Usage (run on the Linux machine, from the repo root -- needs pandas + sklearn,
both already used elsewhere in this codebase):

    python make_stratified_gender_splits.py \\
      --participants_tsv /data/margo_datasets/Michigan_Dataset_1_for_NeuroSTORM/processed_output_resting/metadata/participants.tsv \\
      --out_dir nas/splits/MICHIGAN_GENDER \\
      --n_folds 5 --val_frac 0.2 --seed 1

This OVERWRITES nas/splits/MICHIGAN_GENDER/split_fixed_1.txt..split_fixed_N.txt.
Back up the existing (copied-from-MICHIGAN) files first if you want to keep them.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_sex_labels(participants_tsv):
    df = pd.read_csv(participants_tsv, sep='\t')
    subjects = df['participant_id'].astype(str).tolist()
    sex = df['sex'].astype(str).str.strip().str.upper().tolist()
    # Binary label matching MichiganGender's own convention (1 if starts with 'M', else 0)
    labels = np.array([1 if s.startswith('M') else 0 for s in sex])
    return subjects, labels, sex


def write_split(path, train_subj, val_subj, test_subj):
    with open(path, 'w') as f:
        f.write('train_subjects\n')
        for s in train_subj:
            f.write(s + '\n')
        f.write('val_subjects\n')
        for s in val_subj:
            f.write(s + '\n')
        f.write('test_subjects\n')
        for s in test_subj:
            f.write(s + '\n')


def class_counts(subjects, labels, ids):
    idx = [subjects.index(s) for s in ids]
    lab = labels[idx]
    return int((lab == 0).sum()), int((lab == 1).sum())  # (F count, M count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participants_tsv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--val_frac', type=float, default=0.2,
                     help='fraction of each fold\'s train pool held out as val, stratified by sex')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    subjects, labels, sex_raw = load_sex_labels(args.participants_tsv)
    subjects = np.array(subjects)

    n_f = int((labels == 0).sum())
    n_m = int((labels == 1).sum())
    print('Loaded {} subjects: {} F, {} M'.format(len(subjects), n_f, n_m))
    if min(n_f, n_m) < args.n_folds:
        print('WARNING: minority class ({} subjects) is smaller than n_folds={}. '
              'Some test folds will end up with 0 or 1 minority-class subject -- '
              'consider a smaller --n_folds.'.format(min(n_f, n_m), args.n_folds))

    os.makedirs(args.out_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    subjects_list = subjects.tolist()

    for fold_i, (train_pool_idx, test_idx) in enumerate(skf.split(subjects, labels), start=1):
        train_pool = subjects[train_pool_idx]
        train_pool_labels = labels[train_pool_idx]
        test_subj = subjects[test_idx].tolist()

        # Further stratified split of the train pool into train/val
        train_idx, val_idx = train_test_split(
            np.arange(len(train_pool)),
            test_size=args.val_frac,
            stratify=train_pool_labels,
            random_state=args.seed,
        )
        train_subj = train_pool[train_idx].tolist()
        val_subj = train_pool[val_idx].tolist()

        out_path = os.path.join(args.out_dir, 'split_fixed_{}.txt'.format(fold_i))
        write_split(out_path, train_subj, val_subj, test_subj)

        tr_f, tr_m = class_counts(subjects_list, labels, train_subj)
        va_f, va_m = class_counts(subjects_list, labels, val_subj)
        te_f, te_m = class_counts(subjects_list, labels, test_subj)
        print('Fold {}: train {} (F{}/M{})  val {} (F{}/M{})  test {} (F{}/M{})  -> {}'.format(
            fold_i, len(train_subj), tr_f, tr_m, len(val_subj), va_f, va_m,
            len(test_subj), te_f, te_m, out_path))


if __name__ == '__main__':
    main()

# Class Labels Guide for NeuroSTORM Fine-Tuning (Michigan Dataset)

How block labels, windows and subject splits are built for the Michigan imagery run 1 tasks.

| Task | Config / task id | Dataset class | Classes | Blocks per subject |
|------|------------------|---------------|---------|--------------------|
| 3-way | `michigan_xcpd_nosmooth_fullblock.yaml` / `task6` | `MichiganFullBlock` | tennis 0, navigation 1, squeeze 2 | 30 (10 each) |
| 2-way | `michigan_xcpd_nosmooth_tennisnav.yaml` / `task8` | `MichiganTennisNav` | tennis 0, navigation 1 | 20 (10 each) |

The 2-way task is the 3-way task with every squeeze block dropped. Everything else (timing file, windows, splits, hyperparameters) is the same.

## Where the labels come from

`$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv` (a copy is in `data/task_timing/`).

- One row per TR (0.8 s), 1145 rows, columns `rest, squeeze_imagery, squeeze_action, navigation, tennis`. Exactly one column is 1 per row.
- The same file is used for every subject. The code assumes all subjects share this timing and have a run of the same length (no per-subject timing check).
- The run starts with 20 rest TRs, then 30 task blocks. Each task block is 20 TR (16 s) of task followed by 17-18 TR of rest.
- `squeeze_imagery` and `squeeze_action` are merged into one `squeeze` class.
- The block order is pseudo-random but **identical for every subject**: `t n s n s t s t n t s n s n t n t s t n s n s t s t n t s n` (t = tennis, n = navigation, s = squeeze).

## From timing file to samples (`datasets/fmri_datasets.py`)

1. `Michigan._blocks_from_tsv` run-length encodes the per-TR label, then merges each task run with the rest run that immediately follows it. A block is therefore 37-38 TR: the task plus its trailing rest (participants keep imagining during the rest).
2. `MichiganFullBlock` sets `TRIM_START = 0`, so each block starts exactly at the TSV onset. (Plain `Michigan` trims 2 TR.)
3. `MichiganTennisNav` keeps only blocks labeled tennis or navigation (`LABEL_MAP = {'tennis': 0, 'navigation': 1}`).
4. `_set_data` creates **one sample per block**: a single window of `sequence_length = 37` TR starting at the block onset. There is no sliding window and no overlap between samples. A block is skipped if the 37-TR window does not fit in the block or in the subject's frame count (from `frame_counts.json`, else the `data.pt` header). The last block ends exactly at TR 1145 with no slack, so a subject with a shorter run silently loses it. Check per-subject sample counts in the saved predictions.
5. Every sample from a block gets that block's class id. Input is `[1, 96, 96, 96, 37]`.

Per subject this gives 20 samples for tennis/nav (10 + 10) and 30 for 3-way.

## Model output and metrics

- 3-way: 3 logits, cross-entropy.
- 2-way (`tennis_vs_nav_classification`): **one logit**, binary cross-entropy. P(navigation) = sigmoid(logit); predicted class is navigation if the logit is >= 0.
- Metrics are computed at block level (accuracy, balanced accuracy, per-class recall, AUROC), not averaged per subject first.
- Per-fold predictions are written to `<run>/lightning_logs/version_N/test_predictions/` by `callbacks/imagery_predictions_callback.py`.

## Splits (leave-one-subject-out)

Files: `/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/splits/MICHIGAN/split_fixed_101.txt` ... `split_fixed_126.txt`, written by `scripts/make_loocv_splits.py`. `MICHIGAN_FULL_BLOCK` and `MICHIGAN_TENNIS_NAV` read the same `MICHIGAN` folder.

Format: a `train_subjects` header followed by one subject id per line, then `val_subjects`, then `test_subjects`.

For fold `101 + i` (subjects sorted alphabetically):

- **test**: subject `i` (1 subject)
- **val**: the next 3 subjects, wrapping around (used only to pick the best checkpoint by `valid_acc`)
- **train**: the remaining 22 subjects

So each fold trains on 22 x 20 = 440 tennis/nav samples, validates on 60, and tests on 20. The split is by subject, so all of a subject's blocks stay in one partition. `validate_premade_split` rejects empty sets and any train/val/test overlap.

## Training and testing flow

1. Fold launcher (`run_loocv_tennisnav.sh` -> `run_loocv_xcpd.sh`) sets dataset `michigan_xcpd_nosmooth_tennisnav`, task `task8`, 30 epochs, batch size 8, and the pretrained MAE checkpoint.
2. `fMRIDataModule` loads the split file and builds train/val/test datasets from the subject lists.
3. During fit, each epoch evaluates both the val set and the test set (`val_dataloader` returns both). Only `valid_acc` is used for checkpoint selection; the per-epoch `test_*` values are for monitoring only and must not be used to pick an epoch.
4. After fit, `trainer.test` runs on the best-`valid_acc` checkpoint and the per-sample predictions are saved.

## Label permutation testing

`--permute_labels` shuffles **training** labels only, within each subject (so each subject keeps 10/10). Val and test labels stay real. See `notes/LABEL_PERMUTATION_TESTING.md`.

## Checking labels

```python
import numpy as np
# after building a dataset (train=True), e.g. from fMRIDataModule.setup()
vals, counts = np.unique(ds.target_values, return_counts=True)   # expect 10 x n_subjects per class
```

```bash
# block sequence and lengths straight from the timing file
python3 - <<'EOF'
import csv
rows = list(csv.DictReader(open('data/task_timing/timing_onehot_task1.tsv'), delimiter='\t'))
print(len(rows), 'TRs')
EOF
```

## Related files

| File | Purpose |
|------|---------|
| `datasets/fmri_datasets.py` | `Michigan`, `MichiganFullBlock`, `MichiganTennisNav`: block extraction, labels, permutation |
| `datasets/data_module.py` | split loading, `validate_premade_split`, dataset construction |
| `scripts/make_loocv_splits.py` | writes `split_fixed_101..126.txt` |
| `scripts/configs/datasets/michigan_xcpd_nosmooth_fullblock.yaml`, `..._tennisnav.yaml` | task ids, class counts, 37-TR window |
| `models/lightning_model.py`, `callbacks/imagery_predictions_callback.py` | block-level metrics and saved predictions |
| `data/task_timing/timing_onehot_task1.tsv` | ground-truth block timing |

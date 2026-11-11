# Label Permutation Testing for LOOCV

## Overview

Proper permutation testing for statistical significance requires **retraining models on randomized labels**, not just shuffling subjects or post-hoc analysis. This implementation:

1. Uses the **original LOOCV split files** (same subject groupings as real models)
2. **Randomizes class labels** during training via `--permute_labels` flag
3. Tests on the **original test set with true labels**
4. Compares real-label accuracies vs. random-label accuracies

## Implementation Details

### Code Changes

#### 1. `datasets/fmri_datasets.py` - Michigan dataset class

Added label permutation to the `Michigan` class:

```python
def __init__(self, permute_labels=False, permutation_seed=None, **kwargs):
    self.permute_labels = permute_labels
    self.permutation_seed = permutation_seed
    super().__init__(**kwargs)
```

In `_set_data()`, after loading all training data, labels are shuffled:

```python
if self.permute_labels and self.train and not is_reid:
    seed = self.permutation_seed if self.permutation_seed is not None else 1234
    rng = np.random.RandomState(seed)
    labels = np.array([t[6] for t in data])
    shuffled_labels = labels.copy()
    rng.shuffle(shuffled_labels)
    # Replace labels in data tuples
    data = [data[idx][:6] + (int(shuffled_labels[idx]),) + data[idx][7:]
            for idx in range(len(data))]
```

Key properties:
- **Only affects training** (checked via `self.train`)
- **Does not affect validation/test** (they always use true labels)
- **Deterministic** (seed-based shuffling)
- **Per-fold and per-permutation** (different seed for each permutation)

#### 2. `datasets/data_module.py` - DataModule

Added command-line arguments:

```python
group.add_argument("--permute_labels", action='store_true', 
                   help="Permute/randomize training labels (for permutation testing)")
group.add_argument("--permutation_seed", type=int, default=None, 
                   help="Seed for label permutation (if None, uses --seed)")
```

Modified `_build_single_dataset()` to pass these flags to the dataset:

```python
perm_kwargs = {}
if getattr(self.hparams, 'permute_labels', False):
    perm_kwargs['permute_labels'] = True
    perm_kwargs['permutation_seed'] = getattr(self.hparams, 'permutation_seed', None) or self.hparams.seed

train_ds = Dataset(**params, subject_dict=train_dict, use_augmentations=False, train=True, **perm_kwargs)
```

#### 3. `run_finetune_scripts/greatlakes/loocv/run_loocv_xcpd.sh`

Added argument parsing and passing through to training:

```bash
--permute_labels) PERMUTE_LABELS="--permute_labels"; shift ;;
--permutation_seed) PERMUTATION_SEED="--permutation_seed $2"; shift 2 ;;
```

In the training loop:

```bash
bash scripts/run_experiment.sh \
    ... \
    -- --dataset_split_num "$i" --tpt_strategy none \
         --project_name "${PROJECT_PREFIX}_fold-${i}${PROJECT_NAME_SUFFIX}" \
         $PERMUTE_LABELS $PERMUTATION_SEED
```

#### 4. `scripts/loocv_permutation_testing.sh` - Rewritten

Completely rewritten to use `--permute_labels` flag instead of generating permuted split files:

```bash
for fold_idx in $(seq "$START_INDEX" "$END_INDEX"); do
    for perm_idx in $(seq 1 "$N_PERMUTATIONS"); do
        perm_seed=$((SEED + perm_idx))
        
        bash scripts/run_experiment.sh \
            ... \
            -- --dataset_split_num "$fold_idx" --tpt_strategy none \
                 --permute_labels --permutation_seed "$perm_seed" \
                 --project_name "${PROJECT_PREFIX}_fold-${fold_idx}_perm${perm_idx}"
    done
done
```

#### 5. `run_finetune_scripts/greatlakes/loocv/sbatch_loocv_permutation_testing.sh`

Updated to use the new approach:

- Removed check for permuted split files
- Passes `--permute_labels` and `--permutation_seed` to `run_loocv_xcpd.sh`
- Updated prerequisites in header comment

## Usage

### Generate and train permutation models

```bash
# Sequential training (for testing or single-fold runs)
bash scripts/loocv_permutation_testing.sh \
    --n_permutations 3 \
    --start_index 101 \
    --n_folds 26

# Parallel training on Great Lakes (recommended)
sbatch --array=101-126 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_permutation_testing.sh
```

Or with concurrency limit (e.g., max 2 folds at a time):

```bash
sbatch --array=101-126%2 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_permutation_testing.sh
```

### Configure number of permutations

Edit `sbatch_loocv_permutation_testing.sh`:

```bash
N_PERMUTATIONS=3    # Change to 5, 10, etc. for more permutations
```

### Analyze results

After all models complete:

```bash
python scripts/analyze_loocv_permutation_results.py \
    --real_dir "nas/output/supervised/FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*" \
    --perm_dir "nas/output/supervised/FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv_fold-*_perm*" \
    --n_permutations 3
```

Output:
- `loocv_permutation_results.json` - Numerical statistics
- `loocv_permutation_comparison.png` - Visualization (bar chart + KDE distribution)
- Console summary with p-values and effect sizes

## Expected Results

**For task-imagery_classification with real labels:**
- Real-label model accuracy: ~0.82 (80-82% per fold)
- Randomized-label model accuracy: ~0.33 (1/3 chance for 3 classes)
- P-value: < 0.05 (highly significant)
- Interpretation: Model clearly learned task structure, not random patterns

**For random labels:**
- All models should cluster near 0.33
- High variance (some permutations slightly better/worse by chance)
- Per-sample KDE should show clear separation from real-label distribution

## Key Differences from Subject-Shuffling Approach

| Aspect | Subject Shuffling | Label Permutation |
|--------|-------------------|-------------------|
| What changes | Subject order in split files | Class labels in training data |
| Test set | Original (true labels) | Original (true labels) |
| Hypothesis | Data order doesn't matter | Random labels ≠ real structure |
| Biological relevance | Low (tests artifact, not structure) | High (tests structure learning) |
| Statistical rigor | Weaker (not standard permutation test) | Stronger (proper null distribution) |

## Notes

- **Seed strategy**: Base seed is 1234; each permutation gets `seed + perm_idx`
  - Permutation 1: seed = 1235
  - Permutation 2: seed = 1236
  - Permutation 3: seed = 1237
  - This ensures different shuffles each permutation while remaining reproducible

- **Train-only permutation**: Labels are ONLY shuffled for training; validation and test always use true labels (necessary for model selection and fair evaluation)

- **No subject stratification**: All class labels are pooled and shuffled uniformly (not stratified by class)

- **Compute cost**: Same as real LOOCV × N permutations
  - 26 folds × 3 permutations × ~12 hours = ~36 hours wall-clock (parallel)
  - Increase N_PERMUTATIONS for more robust null distribution (5-10 permutations recommended for publication)

## Troubleshooting

### Models not training
- Check that real LOOCV models (folds 101-126) exist with correct names
- Verify pretrained checkpoint path: `pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt`

### Wrong output directory
- Ensure `nas/output/supervised/` exists
- Check SLURM working directory set in sbatch header

### Different accuracies than expected
- If random-label accuracies are high (>50%), check:
  - Label permutation is actually happening (verify seed in logs)
  - Class distribution is balanced (3 classes should each be ~33% after shuffle)
  - Validation/test aren't accidentally using permuted labels

### Memory errors
- Reduce `--batch_size` in sbatch file
- Set `--num_workers 0` if data loading is slow

## See Also

- [LOOCV_KDE_VISUALIZATION_GUIDE.md](LOOCV_KDE_VISUALIZATION_GUIDE.md) - Interpreting the visualization output
- [LOOCV_PREDICTIONS_GUIDE.md](LOOCV_PREDICTIONS_GUIDE.md) - Per-sample prediction storage and analysis
- [LOOCV_PERMUTATION_TESTING_GUIDE.md](LOOCV_PERMUTATION_TESTING_GUIDE.md) - Original workflow (subject-shuffling; now superseded)

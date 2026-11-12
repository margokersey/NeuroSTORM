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

In `_set_data()`, after loading all training data, each subject's block labels are
shuffled **among that subject's own blocks** (`np.random.RandomState(seed)`, subjects
processed in sorted order so the result is deterministic).

Key properties:
- **Only affects training** (checked via `self.train`)
- **Does not affect validation/test** (they always use true labels)
- **Deterministic** (seed-based shuffling)
- **Within-subject**: every training subject keeps its 10/10/10 class balance; only the block-to-label assignment is permuted
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

Primary inference is the **aggregate test**: the statistic is mean accuracy over all folds;
the null for permutation k is the mean over folds of permutation-k accuracy;
`p = (1 + #{null >= real}) / (1 + n_permutations)`. With n permutations the smallest
attainable p is 1/(1+n), so you need **>= 19 permutations for p < 0.05** and ~100+ to report
a precise p. Per-fold p-values (one held-out subject) are descriptive only.

Each held-out subject is the unit of inference. For every fold the script computes
accuracy, **balanced accuracy (primary)**, macro-F1, Cohen's κ, macro one-vs-rest AUROC,
log-loss, Brier score and ECE (calibration). It averages each metric over subjects and
reports a subject-level bootstrap 95% CI (`--n_bootstrap`, default 10000) and the
aggregate permutation p-value above, applied to that metric (for log-loss, Brier and
ECE, lower is better). As a supplementary check, it also runs a one-sided Wilcoxon
signed-rank test on per-subject (real − mean permuted).

Output (default `output/loocv_permutation/`, override with `--output_file` / `--plot_file`):
- `loocv_permutation_results.json` - `aggregate` (accuracy test), `metrics` (all metrics: mean, CI, null, p), `per_fold`
- `loocv_permutation_results_per_fold.csv` - per-subject metric table (real and mean permuted)
- `loocv_permutation_figure.{png,pdf,svg}` - figure, 183 mm wide (double column), Arial 7–10 pt, editable text in PDF/SVG. Real = blue, permuted = orange (seaborn "deep" colors). Only panels with data are drawn:
  - **a** per-subject balanced accuracy (real vs. that subject's permutation range), sorted
  - **b** KDE of per-subject test accuracy: all real runs vs. all permuted runs (subjects × permutations), reflected at 0/1, with means and a rug of real values
  - **c** permutation null of the mean balanced accuracy, with the real value and p
  - **d** summary metrics: real mean ± 95% CI vs. null, with chance markers and p per metric
  - **e** pooled row-normalized confusion matrix
  - **f** pooled one-vs-rest ROC and **g** calibration (reliability) diagram with ECE. These need `softmax_probs` from per-sample `test_predictions_fold_*.json`; they are skipped when only `test_summary_fold_*.json` exists.
- Console summary with the metric table and per-fold breakdown

Per-sample data is read from, in order: `test_predictions_fold_*.json`, then
`lightning_logs/version_N/test_predictions.pt` (written at test time by
`models/lightning_model.py`; needs `torch`; has the softmax probabilities), then the
confusion matrix in `test_summary_fold_*.json`. The script prints how many runs have
labels / probabilities.

Runs with only `test_summary_fold_*.json` are still fully usable for accuracy, balanced accuracy, macro-F1, κ and the confusion matrix, because the script rebuilds labels from the saved confusion matrix. AUROC, log-loss, Brier and ECE need the per-sample predictions.

The KDE (panel b) uses one accuracy per run, not per sample: per-sample correctness is binary (0/1), so a KDE of it carries no information beyond accuracy.

Panels e–g concatenate predictions across folds (different models), so they are
descriptive; the numbers to report are the subject-mean metrics in panel d / `metrics`.

## Expected Results

**For task-imagery_classification with real labels:**
- Real-label model accuracy: ~0.82 (80-82% per fold)
- Randomized-label model accuracy: ~0.33-0.40 (chance is 1/3; slightly above because the best checkpoint is picked on the true-label validation set, exactly as for the real models)
- P-value: < 0.05 (highly significant)
- Interpretation: Model clearly learned task structure, not random patterns

**For random labels:**
- All models should cluster near chance, a bit above 0.33 due to best-val checkpoint selection
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

- **Within-subject shuffle**: labels are permuted among each subject's own blocks (block = exchangeable unit), so per-subject class balance matches the real labels. (Earlier runs pooled-shuffled across subjects; those `_perm*` outputs are not comparable and should be re-run.)

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
- (older subject-shuffling workflow was removed; see git history)

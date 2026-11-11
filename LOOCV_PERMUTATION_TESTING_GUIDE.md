# LOOCV Proper Permutation Testing: Correct Implementation

## The Correct Approach

**Proper permutation testing requires retraining models on randomized labels.**

For each fold:
1. Train model #1 on **real labels** → test accuracy A_real
2. For each of N permutations:
   - Shuffle training labels randomly
   - Train model #(1+i) on **randomized labels** → test accuracy A_perm_i
3. Null distribution = [A_perm_1, A_perm_2, ..., A_perm_N]
4. Test hypothesis: Is A_real significantly higher than the null?

This answers: **"Does the model learn real structure, or would it perform equally well on meaningless labels?"**

---

## Workflow

### Step 1: Train Real Models (Already Complete)
You've already trained folds 101-126 on real labels. Each fold has test predictions saved in `test_predictions/`.

### Step 2: Generate Permuted Label Datasets
```bash
bash scripts/loocv_permutation_testing.sh \
  --n_permutations 3 \
  --start_index 101 \
  --n_folds 26 \
  --dry_run  # First, check what would run
```

This generates permuted split files like:
- `nas/splits/MICHIGAN/split_fixed_101_perm1.txt`
- `nas/splits/MICHIGAN/split_fixed_101_perm2.txt`
- `nas/splits/MICHIGAN/split_fixed_101_perm3.txt`

### Step 3: Train Models on Permuted Labels
```bash
bash scripts/loocv_permutation_testing.sh \
  --n_permutations 3 \
  --start_index 101 \
  --n_folds 26
```

This trains:
- 26 folds × 3 permutations = **78 additional models**
- Each trained from scratch (same pretrained checkpoint)
- Each trained for `--max_epochs 50` (default, or adjust)
- Results saved to: `nas/output/supervised/xcpd_nosmooth_general_fold-101_perm1/`, etc.

**Timeline:**
- N=3 permutations: ~36 hours (3× the time of one full LOOCV run)
- N=5 permutations: ~60 hours
- Runs in parallel if submitted as job array

### Step 4: Analyze Results
After all permutation models complete:

```bash
python scripts/analyze_loocv_permutation_results.py \
  --real_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
  --perm_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*_perm*" \
  --n_permutations 3
```

**Outputs:**
- `loocv_permutation_results.json` — Per-fold statistics with p-values
- `loocv_permutation_comparison.png` — Bar plot: real vs. permutation accuracies
- Console summary with effect sizes and significance

---

## Expected Results

**Real labels train well, random labels train poorly:**
```
Real accuracy (mean):      0.823
Randomized accuracy (mean): 0.345
Improvement:               0.478 (47.8 percentage points)
Folds significant at p<0.05: 25/26 (96%)
```

This shows the model learns real structure in labels, not noise.

---

## Cost vs. Reliability

| N Permutations | Training Time | Statistical Confidence | Use Case |
|---|---|---|---|
| 2 | 24 hours | Low (coarse estimate) | Quick sanity check |
| 3 | 36 hours | Medium (reasonable CI) | **Recommended** |
| 5 | 60 hours | High (tight CI) | Publication-quality |
| 10 | 120 hours | Very High | Thorough analysis |

**Recommendation:** Start with N=3 for ~36 hours. If results are marginal, increase to N=5 for tighter estimates.

---

## Interpretation

### Per-Fold P-Values
- **p < 0.05**: Fold significantly outperforms random labels
- **p ≥ 0.05**: Fold not significantly better than random (possible overfitting or weak signal)

### Aggregate Statistics
- **Effect size (real - random)**: How much better is real than random?
  - 0.3+: Strong signal (model clearly learns real structure)
  - 0.1-0.3: Moderate signal
  - <0.1: Weak signal (close to random)

- **Significant folds (p<0.05)**: What fraction of folds beat random?
  - 100%: All folds robust
  - 80-90%: Most folds robust
  - <50%: Many folds not significant (potential overfitting)

---

## Why This Matters

**Before:** "Model achieves 82% accuracy" — but on what?
**After permutation testing:** "Model achieves 82% (vs. 33% on random), p<0.001" — clear evidence of learning

This is especially important for:
- Small datasets (potential overfitting to noise)
- Complex models (high capacity = easy to memorize)
- Publication-quality results (reviewers expect this)

---

## Files Structure

### New Scripts
- **`scripts/loocv_permutation_testing.sh`** — Main script to train models on permuted labels
- **`scripts/analyze_loocv_permutation_results.py`** — Analyze results and compare

### New Output Files (Per Fold)
```
nas/output/supervised/xcpd_nosmooth_general_fold-101_perm1/
├── checkpt-XX-valid_acc-YY.ckpt
├── test_predictions/
│   ├── test_predictions_fold_101.json
│   └── test_summary_fold_101.json
```

Repeat for `_perm1`, `_perm2`, `_perm3`, etc.

### Analysis Output
- `loocv_permutation_results.json` — All per-fold stats
- `loocv_permutation_comparison.png` — Visualization

---

## Quick Reference

```bash
# Check what would run (no actual training)
bash scripts/loocv_permutation_testing.sh --n_permutations 3 --dry_run

# Train permutation models (takes ~36 hours for N=3)
bash scripts/loocv_permutation_testing.sh --n_permutations 3

# Analyze when complete
python scripts/analyze_loocv_permutation_results.py \
  --real_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
  --perm_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*_perm*" \
  --n_permutations 3
```

---

## Limitations & Notes

1. **Assumes same architecture/hyperparameters:** Models train with same max_epochs, batch_size, learning rate
2. **Train/val split difference:** Real labels get good validation signal; random labels all get ~33% (imbalanced validation)
   - This is fine; we're testing generalization, not optimization
3. **Multiple comparisons:** If running many permutations, consider multiple comparison correction (alpha=0.05/N_permutations)
4. **Label permutation strategy:** Currently shuffles ALL labels equally. Could stratify by class if needed.

---

## See Also

- `LOOCV_PREDICTIONS_GUIDE.md` — Per-sample predictions and inspection
- `LOOCV_PREDICTIONS_QUICKSTART.md` — Quick reference for main commands
- `scripts/inspect_fold_predictions.py` — Inspect individual fold predictions

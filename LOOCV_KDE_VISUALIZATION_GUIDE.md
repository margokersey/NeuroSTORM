# LOOCV Permutation Testing: KDE Visualization Guide

## What's New

The `analyze_loocv_permutation_results.py` script now generates **two complementary visualizations**:

1. **Per-Fold Bar Chart** (left panel)
   - Real-label model accuracy per fold (blue bars)
   - Randomized-label model accuracy per fold with error bars (coral dots + error bars)
   - Shows fold-by-fold breakdown

2. **Per-Sample KDE Distribution** (right panel)
   - Kernel Density Estimation plot of **all individual sample accuracies**
   - Real-label model: all correct/incorrect predictions pooled across all folds
   - Randomized-label model: all correct/incorrect predictions pooled across all permutations
   - Shows overall separation between real and random distributions

## Why This Matters

The KDE plot directly visualizes the **null hypothesis test**:

- **Real labels**: Should cluster near 1.0 (high accuracy across samples)
- **Randomized labels**: Should cluster near 0.33 (1/3 for 3-class task)
- **Separation**: Clear gap indicates the model learned real structure, not random patterns

## Output Files

After running the analysis script:

```
loocv_permutation_comparison.png
├── Panel 1: Per-fold bar chart (existing functionality)
└── Panel 2: Per-sample KDE distributions (NEW)

loocv_permutation_results.json
└── Numerical statistics (p-values, effect sizes, etc.)
```

## Usage

```bash
# Run after all permutation models complete
python scripts/analyze_loocv_permutation_results.py \
    --real_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
    --perm_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*_perm*" \
    --n_permutations 3 \
    --plot_file loocv_permutation_analysis.png

# View the generated plot
display loocv_permutation_analysis.png  # or open in your image viewer
```

## Data Sources

The KDE plot uses **per-sample predictions** automatically saved by the `ImageryPredictionsCallback`:

- Each model saves: `test_predictions_fold_N.json` (or `test_predictions_fold_N_permK.json`)
- Contains array of samples with `'correct'` field (0 or 1)
- Script pools all samples across:
  - All 26 real-label folds
  - All N×26 permutation runs (N=3 permutations per fold)

Example per-sample data:
```json
[
  {
    "sample_idx": 0,
    "subject": "sub-02",
    "true_label": 1,
    "predicted_label": 1,
    "correct": 1
  },
  {
    "sample_idx": 1,
    "subject": "sub-02",
    "true_label": 0,
    "predicted_label": 2,
    "correct": 0
  }
]
```

## Interpretation

### Expected Behavior (Real > Random)

```
Density
  │
  │     ╔════╗
  │     ║Real║         ╔══════════╗
  │     ║    ║         ║ Randomized
  │     ║ (≈0.82)      ║ (≈0.33)
  ├─────╫────╫─────────╫──────────╫─────
  │     0.3  0.8       0.0        0.5   1.0
  │                Accuracy
  v
```

**This pattern confirms:**
- Model learns real task structure
- Real accuracies significantly exceed random chance
- Permutation test is valid and statistically significant

### If Distributions Overlap

If distributions heavily overlap:
- Model may not have learned task structure
- Consider data quality issues
- Check class balance in training
- Verify label quality

## Statistics Shown

The plot includes summary statistics in the upper right:

```
Real: μ=0.821, σ=0.065
Random: μ=0.331, σ=0.042
```

- **μ** (mu): Mean accuracy
- **σ** (sigma): Standard deviation across all samples

These reflect across-sample variability, not fold-level statistics.

## Technical Details

### Jittering
Small random noise (σ=0.01) is added to binary samples (0/1) for better KDE visualization.

### Bandwidth Method
KDE bandwidth set to 0.1 for smooth curves while preserving distribution shape.

### Density Estimation
Uses `scipy.stats.gaussian_kde` for robust, automatic bandwidth selection.

## Requirements

- matplotlib (for visualization)
- scipy (for KDE)
- numpy (for statistics)
- json (built-in)

```bash
# Ensure dependencies are installed
pip install matplotlib scipy numpy
```

## Common Issues

### "Per-sample predictions not available"

**Cause**: Model outputs lack test predictions JSON files.

**Solution**: 
1. Verify models ran with updated `main.py` (has callback)
2. Check that `nas/output/supervised/*/test_predictions/` directories exist
3. Re-run models with callback enabled

### Plot looks wrong

**Causes**:
- Predictions directory has different name (check glob pattern)
- Fold naming doesn't match expected format (fold-101, fold-102, etc.)
- Permutation naming missing `_permN` suffix

**Solution**: Add `--verbose` flag (if available) and check directory structure:
```bash
find nas/output/supervised -name "test_summary_fold_*.json" | head -5
```

## See Also

- [LOOCV_PERMUTATION_TESTING_GUIDE.md](LOOCV_PERMUTATION_TESTING_GUIDE.md) — Full permutation testing workflow
- [LOOCV_PREDICTIONS_GUIDE.md](LOOCV_PREDICTIONS_GUIDE.md) — Per-sample predictions details
- [callbacks/imagery_predictions_callback.py](callbacks/imagery_predictions_callback.py) — Callback implementation

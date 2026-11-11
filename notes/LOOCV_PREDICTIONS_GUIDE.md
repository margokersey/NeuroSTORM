# LOOCV Per-Sample Predictions & Permutation Testing

## Overview

The LOOCV fine-tuning pipeline now automatically saves **per-sample test predictions** to disk after each fold completes. This enables:

1. **Permutation testing** — Generate null distributions via label shuffling
2. **Per-subject performance** — See which subjects generalize well vs. poorly  
3. **Per-class breakdown** — Understand model performance on tennis/navigation/squeeze
4. **Confusion analysis** — Identify systematic misclassifications
5. **Reproducible post-hoc analysis** — All data saved, not just aggregated metrics

---

## Implementation Details

### 1. Automatic Prediction Saving (ImageryPredictionsCallback)

**Location:** `callbacks/imagery_predictions_callback.py`

A PyTorch Lightning callback that:
- Intercepts test predictions during `test_epoch_end()`
- Saves per-sample logits, softmax probabilities, and labels
- Computes summary statistics (accuracy, confusion matrix, per-subject breakdown)
- **Does NOT modify training/validation** — only outputs test results

**Activated automatically** for:
- Task: `imagery_task_classification`
- Mode: `finetune` (downstream, not pretraining)
- Only saves when test set is evaluated

### 2. Output Files per Fold

After each LOOCV fold completes, you'll find:

```
nas/output/supervised/xcpd_nosmooth_general_fold-101/
├── checkpt-XX-valid_acc-YY.ckpt          (best model)
├── last.ckpt                              (last epoch)
└── test_predictions/
    ├── test_predictions_fold_101.json     ← Per-sample data (6-15 KB)
    └── test_summary_fold_101.json         ← Summary statistics
```

**test_predictions_fold_101.json** structure:
```json
[
  {
    "sample_idx": 0,
    "subject": "sub-02",
    "true_label": 1,
    "true_label_name": "navigation",
    "predicted_label": 1,
    "predicted_label_name": "navigation",
    "logits": [-0.543, 2.104, -1.562],
    "softmax_probs": [0.024, 0.923, 0.053],
    "correct": true
  },
  ...
]
```

**test_summary_fold_101.json** structure:
```json
{
  "fold_id": 101,
  "num_samples": 847,
  "overall_accuracy": 0.823,
  "per_class_accuracy": {
    "tennis": 0.834,
    "navigation": 0.812,
    "squeeze": 0.823
  },
  "per_subject_accuracy": {
    "sub-02": {"accuracy": 0.823, "num_samples": 847}
  },
  "confusion_matrix": [[42, 5, 1], [3, 38, 2], [2, 1, 39]],
  "confusion_matrix_labels": ["tennis", "navigation", "squeeze"]
}
```

---

## Analysis Scripts

### 1. Quick Fold Inspection

Inspect predictions from a single fold:

```bash
python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101
```

**Output:**
- Overview accuracy and per-class breakdown
- Per-subject results
- Confusion matrix
- Top misclassifications with confidence scores

**Options:**
- `--misclass_top_n 30` — Show top 30 misclassifications (default: 20)

### 2. Aggregate Analysis Across All Folds

After all 26 folds complete, analyze permutation chance:

```bash
python scripts/analyze_loocv_predictions.py \
  --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
  --n_permutations 1000
```

**Output:**
- `loocv_permutation_analysis.json` — Detailed per-fold metrics
- `loocv_real_vs_permutation.png` — Comparison plots
- Console summary with:
  - Mean accuracy ± SEM
  - Permutation chance distribution
  - P-values and effect sizes
  - Significance count (p < 0.05)

**Options:**
- `--n_permutations 10000` — More permutations = more stable null distribution (slower)
- `--output_file results.json` — Custom output filename
- `--no_plot` — Skip plot generation (e.g., if matplotlib unavailable)

---

## Permutation Testing Logic

### The Test

For each fold:
1. **Real prediction:** Model predicts class X when true label is Y
2. **Permutation:** Randomly shuffle all true labels, keeping predictions fixed
3. **Chance accuracy:** What accuracy do the (fixed) predictions get under random labels?
4. **Repeat:** Generate 1000 permutations to build null distribution

### Interpretation

- **Real accuracy:** e.g., 0.823 (82.3% correct)
- **Permutation mean:** e.g., 0.335 (random chance ~33% for 3 classes)
- **P-value:** Fraction of permutations with accuracy ≥ real accuracy
  - p < 0.001: Model learned far beyond chance
  - p < 0.05: Significant improvement over chance
  - p ≥ 0.05: Not significantly better than chance

### Example Output

```
Across 26 folds:
  Real accuracy (mean ± SEM):          0.823 ± 0.018
  Real accuracy range:                 0.745 - 0.891
  Permutation chance (mean ± std):     0.335 ± 0.052
  Expected random chance:              0.333
  Improvement over chance:             147.2%
  
  Fraction significant (p < 0.05):    25 / 26 (96.2%)
  Mean effect size (Cohen's d-like):  9.33
  Median p-value:                      0.001
```

---

## Comparison Plots

The analysis script generates a 2-panel figure:

**Panel 1: Per-fold real vs. permutation mean**
- Blue bars: Model accuracy on real labels
- Coral bars: Mean accuracy under random label permutations
- Red dashed line: Expected random chance (1/3 ≈ 0.333)
- Shows each fold individually, easily spot check fold-specific issues

**Panel 2: Distribution comparison**
- Histogram: All permutation accuracies pooled across folds
- Blue line: Mean real accuracy
- Red dashed line: Random chance
- Shows that model accuracies are far from permutation distribution

---

## Workflow: End-to-End

### Step 1: Submit All Folds
```bash
sbatch --array=101-126 run_finetune_scripts/greatlakes/loocv/sbatch_loocv_xcpd.sh
```

### Step 2: Monitor Progress
```bash
squeue -u mkersey
tail -f logs/xcpd_loocv_<jobid>_101.log  # Watch fold 101
```

After each fold finishes:
```bash
python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101
# Verify accuracy looks reasonable
```

### Step 3: After All Folds Complete
```bash
python scripts/analyze_loocv_predictions.py \
  --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
  --n_permutations 1000
```

Inspect outputs:
- `loocv_permutation_analysis.json` — Machine-readable results
- `loocv_real_vs_permutation.png` — For papers/presentations
- Console summary — Quick sanity check

### Step 4: Deep Dive (Optional)

For specific folds or custom analysis:

```python
# Load and explore predictions directly
import json

with open('nas/output/supervised/xcpd_nosmooth_general_fold-101/test_predictions/test_predictions_fold_101.json') as f:
    preds = json.load(f)

# Per-subject performance
subj_perfs = {}
for p in preds:
    subj = p['subject']
    if subj not in subj_perfs:
        subj_perfs[subj] = {'correct': 0, 'total': 0}
    subj_perfs[subj]['total'] += 1
    if p['correct']:
        subj_perfs[subj]['correct'] += 1

for subj, perf in sorted(subj_perfs.items()):
    acc = perf['correct'] / perf['total']
    print(f"{subj}: {acc:.3f} ({perf['correct']}/{perf['total']})")
```

---

## Troubleshooting

### No `test_predictions/` directory after fold completes

**Possible causes:**
1. **Task name mismatch:** Callback only activates for `task_name='imagery_task_classification'`
   - Check: `cat nas/output/supervised/xcpd_nosmooth_general_fold-101/.hydra/config.yaml | grep task_name`

2. **Mode issue:** Callback only activates in finetune mode (not pretraining)
   - Check: `grep "mode:" config.yaml`

3. **Test set empty:** If validation split is empty or drop_last=True with small splits
   - Check: `ls nas/output/supervised/xcpd_nosmooth_general_fold-101/events.out.tfevents.*`

### test_summary_fold_X.json but no test_predictions_fold_X.json

The callback saves both; if only summary exists, predictions might have been truncated. Check file sizes:
```bash
ls -lh nas/output/supervised/xcpd_nosmooth_general_fold-*/test_predictions/
```

If predictions file is 0 bytes, the test phase had 0 samples. Check split files.

### analyze_loocv_predictions.py finds 0 folds

**Check:**
1. Glob pattern is correct: `ls nas/output/supervised/xcpd_nosmooth_general_fold-*`
2. Folds completed (not still running): `squeue -u mkersey`
3. Fold directories exist and contain output: `ls nas/output/supervised/xcpd_nosmooth_general_fold-101/`

### ImportError: No module named 'matplotlib'

The analysis script detects this and skips plots. To enable plots:
```bash
pip install matplotlib
```

---

## Customization

### Adjusting Permutation Parameters

Edit `scripts/analyze_loocv_predictions.py`:

```python
# Increase for more stable null distribution (slower)
n_permutations=10000  # Instead of 1000
```

### Modifying Summary Statistics

Edit `callbacks/imagery_predictions_callback.py` in `_compute_summary()`:
- Add confidence intervals
- Compute balanced accuracy (macro F1)
- Add per-class AUC

### Saving Additional Data

Modify `ImageryPredictionsCallback` to save:
- Attention maps / saliency
- Model features (bottleneck activations)
- Per-block timings

---

## Citations & Theory

**Permutation testing:**
- Good, P. (2005). Permutation, Parametric, and Bootstrap Tests of Hypotheses
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero

**LOOCV generalization:**
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation

---

## Quick Reference

| Task | Command |
|------|---------|
| Inspect fold 101 | `python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101` |
| Analyze all folds | `python scripts/analyze_loocv_predictions.py --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*"` |
| Increase permutations | `--n_permutations 10000` |
| Load predictions in Python | `json.load(open('nas/output/.../test_predictions_fold_101.json'))` |
| Per-subject breakdown | Check `test_summary_fold_101.json` → `per_subject_accuracy` |

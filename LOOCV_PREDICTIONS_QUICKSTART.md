# LOOCV Predictions: Quick Start (TL;DR)

## What's New
✅ Predictions are **automatically saved** after each fold completes  
✅ Saves to `nas/output/supervised/xcpd_nosmooth_general_fold-XXX/test_predictions/`  
✅ Ready for permutation testing and per-subject analysis

---

## One-Liner Commands

### After Fold 101 Completes (Quick Check)
```bash
python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101
```

### After All 26 Folds Complete (Full Analysis)
```bash
python scripts/analyze_loocv_predictions.py --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*"
```

**Outputs:**
- `loocv_permutation_analysis.json` — All statistics
- `loocv_real_vs_permutation.png` — Comparison plot
- Console summary with p-values and effect sizes

---

## Output Files

Per fold, in `<output_dir>/test_predictions/`:
- `test_predictions_fold_101.json` — Per-sample logits + labels (6-15 KB)
- `test_summary_fold_101.json` — Accuracy stats + confusion matrix + per-subject breakdown (1-2 KB)

---

## What Each Script Does

| Script | When | Usage | Output |
|--------|------|-------|--------|
| `inspect_fold_predictions.py` | After fold 101 completes | `--fold_dir <path>` | Console table with per-subject, per-class, confusion matrix |
| `analyze_loocv_predictions.py` | After all folds complete | `--output_dir "nas/output/supervised/xcpd*"` | JSON file + comparison plot + console summary |

---

## Python: Load & Analyze Manually

```python
import json
import numpy as np

# Load fold 101
with open('nas/output/supervised/xcpd_nosmooth_general_fold-101/test_predictions/test_predictions_fold_101.json') as f:
    preds = json.load(f)

# Overall accuracy
acc = np.mean([p['correct'] for p in preds])
print(f"Accuracy: {acc:.3f}")

# Per-subject
for subj in set(p['subject'] for p in preds):
    subj_preds = [p for p in preds if p['subject'] == subj]
    subj_acc = np.mean([p['correct'] for p in subj_preds])
    print(f"{subj}: {subj_acc:.3f}")

# Permutation test
true_labels = np.array([p['true_label'] for p in preds])
pred_labels = np.array([p['predicted_label'] for p in preds])
real_acc = (pred_labels == true_labels).mean()

perm_accs = []
for _ in range(1000):
    perm_labels = np.random.permutation(true_labels)
    perm_acc = (pred_labels == perm_labels).mean()
    perm_accs.append(perm_acc)

perm_accs = np.array(perm_accs)
p_value = (perm_accs >= real_acc).sum() / 1000
print(f"\nReal accuracy: {real_acc:.3f}")
print(f"Permutation mean: {perm_accs.mean():.3f}")
print(f"P-value: {p_value:.3f}")
```

---

## How It Works

1. **Training:** TensorBoard logs per-epoch metrics (accuracy, AUROC, F1, confusion matrix) ← existing
2. **Testing:** Callback intercepts test predictions and saves to JSON ← **NEW**
3. **Analysis:** Scripts load JSON files, compute permutation tests, generate plots ← **NEW**

No changes to training behavior. Only affects what gets saved to disk after test.

---

## When to Use

| Task | Command |
|------|---------|
| Check fold 101 looks OK before all 26 | `inspect_fold_predictions.py --fold_dir xcpd...fold-101` |
| After all 26 folds: Get summary stats | `analyze_loocv_predictions.py --output_dir "xcpd...fold-*"` |
| Generate permutation plots for paper | `analyze_loocv_predictions.py ... && open loocv_real_vs_permutation.png` |
| Check which subjects generalize poorly | Load JSON + Python script above |
| Debug model on specific subject | `grep sub-02 test_predictions_fold_101.json` |

---

## Key Statistics

**Per fold:**
- Real accuracy: e.g., 0.823
- Permutation mean: e.g., 0.335 (≈1/3 for 3 classes)
- P-value: e.g., 0.001 (significant if < 0.05)

**Across 26 folds:**
- Mean accuracy: (sum of all fold accuracies) / 26
- Significance: Fraction of folds with p < 0.05

---

## Troubleshooting in 30 Seconds

**No `test_predictions/` directory?**
- Check task_name is `imagery_task_classification`
- Check running in finetune mode (not pretraining)

**Empty JSON file?**
- Test set had 0 samples (check split file)

**Analyze script finds 0 folds?**
- Check glob: `ls nas/output/supervised/xcpd_nosmooth_general_fold-*`
- Check fold completed: `squeue -u mkersey`

**Full guide:** See `LOOCV_PREDICTIONS_GUIDE.md`

---

## Next Steps

1. ✅ Fold 101 finishes
2. Run: `python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101`
3. Verify: Accuracy looks reasonable, subjects present
4. Submit all 26: `sbatch --array=101-126 sbatch_loocv_xcpd.sh`
5. When done: `python scripts/analyze_loocv_predictions.py --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*"`
6. Review: `loocv_real_vs_permutation.png` + `loocv_permutation_analysis.json`

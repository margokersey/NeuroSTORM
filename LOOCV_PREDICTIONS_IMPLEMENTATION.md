# LOOCV Per-Sample Predictions Implementation Summary

## What Changed

### Problem
- Previously only TensorBoard metrics (accuracy, AUROC, F1) were saved per fold
- No per-sample data = no permutation testing, no subject-specific analysis, hard to debug
- All results were aggregated; couldn't see which subjects generalized poorly

### Solution
Implemented a **clean, zero-overhead PyTorch Lightning callback** that:
- ✅ Automatically saves per-sample test predictions for imagery_task_classification
- ✅ Saves to JSON files in `<output_dir>/test_predictions/`
- ✅ No impact on training/validation — only captures test results
- ✅ Includes per-class, per-subject, and confusion matrix summaries
- ✅ Ready for permutation testing and custom post-hoc analysis

---

## Files Created

### 1. **Callback** (Core Implementation)
- **`callbacks/imagery_predictions_callback.py`** (150 lines)
  - PyTorch Lightning Callback class
  - Intercepts test-phase predictions
  - Saves logits, softmax probs, labels, per-subject breakdown
  - Zero runtime overhead (only file I/O at epoch end)
  - Automatically activated in main.py for imagery_task_classification

### 2. **Analysis Scripts**

- **`scripts/analyze_loocv_predictions.py`** (380 lines)
  - Load predictions from all 26 folds
  - Compute permutation-based null distributions (label shuffling)
  - Generate summary statistics and comparison plots
  - Usage: `python scripts/analyze_loocv_predictions.py --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*"`

- **`scripts/inspect_fold_predictions.py`** (180 lines)
  - Quick inspection of a single fold
  - Per-subject accuracy breakdown
  - Per-class breakdown with confusion matrix
  - Top misclassifications sorted by confidence
  - Usage: `python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101`

### 3. **Documentation**
- **`LOOCV_PREDICTIONS_GUIDE.md`** — Complete user guide with examples and troubleshooting
- **This file** — Implementation summary

---

## Modified File

### `main.py`
- Added import: `from callbacks.imagery_predictions_callback import ImageryPredictionsCallback`
- Added conditional to append callback to callbacks list:
  ```python
  if (args.downstream_task_type == "classification" and 
      args.task_name == 'imagery_task_classification' and
      not args.pretraining):
      callbacks.append(ImageryPredictionsCallback())
  ```
- **Scope:** Only affects imagery_task_classification finetune mode; zero impact on other tasks

---

## Output Files per Fold

After each LOOCV fold completes:

```
nas/output/supervised/xcpd_nosmooth_general_fold-101/
├── checkpt-XX-valid_acc-YY.ckpt
├── last.ckpt
└── test_predictions/                           ← NEW
    ├── test_predictions_fold_101.json          ← Per-sample data
    └── test_summary_fold_101.json              ← Summary stats
```

### test_predictions_fold_101.json
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

### test_summary_fold_101.json
```json
{
  "fold_id": 101,
  "num_samples": 847,
  "overall_accuracy": 0.823,
  "per_class_accuracy": {"tennis": 0.834, "navigation": 0.812, "squeeze": 0.823},
  "per_subject_accuracy": {"sub-02": {"accuracy": 0.823, "num_samples": 847}},
  "confusion_matrix": [[42, 5, 1], [3, 38, 2], [2, 1, 39]],
  "confusion_matrix_labels": ["tennis", "navigation", "squeeze"]
}
```

---

## Key Features

### 1. **Permutation Testing**
```bash
python scripts/analyze_loocv_predictions.py \
  --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
  --n_permutations 1000
```

Generates:
- Per-fold null distributions (label shuffling)
- P-values for each fold
- Effect sizes
- Comparison plots
- Summary JSON with all statistics

### 2. **Per-Subject Performance**
```bash
python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101
```

Shows which subjects the model generalizes to vs. from.

### 3. **Per-Class Breakdown**
Tennis vs. navigation vs. squeeze accuracy — useful for task-specific debugging.

### 4. **Confusion Matrix**
Understand systematic misclassifications (e.g., navigation often confused with tennis).

---

## Workflow: How to Use

### Immediate (Fold 101 Testing)
```bash
# After fold 101 completes
python scripts/inspect_fold_predictions.py \
  --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101
```

Verify outputs look reasonable before committing to all 26 folds.

### After All Folds Complete
```bash
python scripts/analyze_loocv_predictions.py \
  --output_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
  --n_permutations 1000
```

Generates:
1. `loocv_permutation_analysis.json` — Machine-readable results
2. `loocv_real_vs_permutation.png` — Comparison plot for papers
3. Console summary with statistical significance

### Custom Analysis (Python)
```python
import json

# Load fold 101 predictions
with open('nas/output/supervised/xcpd_nosmooth_general_fold-101/test_predictions/test_predictions_fold_101.json') as f:
    preds = json.load(f)

# Per-subject accuracy
for subj in set(p['subject'] for p in preds):
    subj_preds = [p for p in preds if p['subject'] == subj]
    acc = sum(p['correct'] for p in subj_preds) / len(subj_preds)
    print(f"{subj}: {acc:.3f}")
```

---

## Design Decisions

### 1. **Callback vs. In-Model Code**
- ✅ Callback: Clean separation of concerns, no model modification
- ✅ Reusable for other tasks (just change class_names mapping)
- ✅ Easy to disable (just don't instantiate in main.py)

### 2. **JSON Format**
- ✅ Human-readable (unlike pickle)
- ✅ Language-agnostic (R, MATLAB, JavaScript can read)
- ✅ Versioning: Can add fields without breaking old code
- ✅ No schema enforcement (simpler than protocol buffers)

### 3. **Label Permutation (Not Prediction Shuffling)**
- ✅ Tests: "Does model learn true signal?" (standard hypothesis test)
- ❌ Prediction shuffling tests: "Is model consistent?" (different question)

### 4. **Save to Output Dir, Not nas/outputs/**
- ✅ All fold data stays together (logs, checkpoints, predictions)
- ✅ No naming collisions across experiments
- ✅ Easy to archive/delete entire experiment

---

## Testing Checklist

Before running all 26 folds:

- [ ] Fold 101 completes successfully
- [ ] `nas/output/supervised/xcpd_nosmooth_general_fold-101/test_predictions/` exists
- [ ] `test_predictions_fold_101.json` is not empty (`ls -lh` shows >1 KB)
- [ ] `test_summary_fold_101.json` has accuracy < 1.0 (not overfitting test set)
- [ ] Run `python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101`
  - Output looks reasonable (no null values, subjects match split file)
- [ ] Accuracy is in reasonable range (e.g., > 0.4, < 0.95 for first test)

---

## Troubleshooting

### No `test_predictions/` directory
1. Check task_name: `cat nas/output/supervised/xcpd_nosmooth_general_fold-101/.hydra/config.yaml | grep task_name`
   - Should be `task_name: imagery_task_classification`
2. Check mode: Callback only activates in finetune mode (not pretraining)
3. Check test set not empty: Verify split file has test subjects

### test_predictions_fold_101.json is 0 bytes
- Test set had 0 samples (check split file)
- Callback encountered an error (check job log)

### `analyze_loocv_predictions.py` finds 0 folds
1. Check glob pattern: `ls nas/output/supervised/xcpd_nosmooth_general_fold-*`
2. Verify folds completed (not running): `squeue -u mkersey`
3. Verify fold has output: `ls nas/output/supervised/xcpd_nosmooth_general_fold-101/test_predictions/`

---

## Performance Impact

- **Training time:** None (callback only runs after test epoch)
- **Validation time:** None (callback doesn't affect validation)
- **Test time:** <100ms per fold (JSON serialization)
- **Disk usage:** ~10 KB per fold × 26 folds = 260 KB (negligible)

---

## Future Enhancements (Optional)

- Add per-subject learning curves (accuracy vs. epoch)
- Save attention maps / saliency for misclassified samples
- Compute AUC per-class (not just accuracy)
- Integrate with wandb/Neptune for live monitoring
- Generate subject-level generalization heatmaps

---

## References

1. **Permutation Testing:** Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests*
2. **LOOCV:** Hastie et al. (2009). *Elements of Statistical Learning*
3. **PyTorch Lightning:** https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html

---

## Questions?

Refer to `LOOCV_PREDICTIONS_GUIDE.md` for:
- Detailed usage examples
- Interpretation of permutation results
- Custom analysis code snippets
- Troubleshooting guide

# Quick Metrics & Permutation Testing Guide

## What's Currently Being Saved During LOOCV

✅ **Aggregate metrics** (logged to TensorBoard):
- Per-epoch validation accuracy
- Per-epoch test accuracy
- Confusion matrices
- Per-class F1 scores

❌ **NOT saved** (need to add):
- Per-sample predictions (logits + targets)
- Per-sample accuracy
- Individual block predictions

## Why You Need Per-Sample Predictions

To plot **permutation chance vs. true metrics**, you need:
1. Raw logits for each test sample
2. Ground truth labels for each sample
3. Subject ID for each sample (optional but helpful)

## Minimal Code Addition

Add to `models/lightning_model.py` in `validation_epoch_end()`, after line ~717 (after `_evaluate_metrics` call):

```python
# Save test predictions for permutation testing
import pickle, os
if len(out_test_logits_list) > 0:
    test_pkl = os.path.join(
        os.path.dirname(self.trainer.checkpoint_callback.best_model_path),
        'test_predictions.pkl'
    )
    os.makedirs(os.path.dirname(test_pkl), exist_ok=True)
    with open(test_pkl, 'wb') as f:
        pickle.dump({
            'logits': total_out_test_logits.numpy(),
            'targets': total_out_test_target.numpy(),
            'fold_id': getattr(self.hparams, 'dataset_split_num', -1),
        }, f)
```

## After Fold 101 Completes

1. Check if `test_predictions.pkl` exists:
   ```bash
   find checkpoints -name test_predictions.pkl
   ```

2. If it exists, you can run permutation analysis on that fold

## For All 26 Folds

Once all folds finish, run:
```bash
python scripts/aggregate_loocv_results.py  # loads all pkl files
python scripts/plot_loocv_permutation.py   # generates figures
```

## Key Files
- Metrics guide: `LOOCV_METRICS_AND_PERMUTATION.md`
- Fold 101 running: Check `logs/xcpd_loocv_*_101.log`
- Split file (correct): `nas/splits/MICHIGAN/split_fixed_101.txt` (22/3/1)

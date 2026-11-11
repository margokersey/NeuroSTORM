"""
Callback to save per-sample test predictions for imagery_task_classification.

This callback saves test-phase predictions to JSON files for post-hoc analysis,
permutation testing, and per-subject generalization analysis. It does NOT
modify the model's training or validation behavior.

Predictions are saved to: <output_dir>/test_predictions/
"""

import json
import numpy as np
import torch
from pathlib import Path
import pytorch_lightning as pl


class ImageryPredictionsCallback(pl.Callback):
    """Save per-sample test predictions to disk after test phase."""

    def __init__(self):
        super().__init__()
        self.test_predictions = []
        self.test_subjects = []
        self.test_logits = None
        self.test_targets = None

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Accumulate test batches for saving at epoch end."""
        if outputs is not None:
            subj, (logits, targets, _) = outputs
            self.test_subjects.extend(subj)
            if self.test_logits is None:
                self.test_logits = logits
                self.test_targets = targets
            else:
                self.test_logits = torch.cat([self.test_logits, logits], dim=0)
                self.test_targets = torch.cat([self.test_targets, targets], dim=0)

    def on_test_epoch_end(self, trainer, pl_module):
        """Save accumulated test predictions to disk."""
        if not self.test_subjects or self.test_logits is None:
            return

        # Only save for imagery_task_classification task
        if pl_module.hparams.task_name != 'imagery_task_classification':
            return

        # Create output directory
        output_dir = Path(trainer.logger.log_dir) / "test_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get fold ID from hparams or use "unknown"
        fold_id = getattr(pl_module.hparams, 'dataset_split_num', 'unknown')

        # Convert tensors to numpy
        logits_np = self.test_logits.numpy() if isinstance(self.test_logits, torch.Tensor) else self.test_logits
        targets_np = self.test_targets.numpy() if isinstance(self.test_targets, torch.Tensor) else self.test_targets

        # Class name mapping
        imagery_class_names = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}

        # Build per-sample predictions list
        predictions = []
        for i, (subj, logit_vec, true_label) in enumerate(zip(
            self.test_subjects, logits_np, targets_np
        )):
            pred_label = int(np.argmax(logit_vec))
            true_label_int = int(true_label)

            # Compute softmax probabilities
            exp_logits = np.exp(logit_vec - np.max(logit_vec))  # numerical stability
            softmax_probs = exp_logits / np.sum(exp_logits)

            predictions.append({
                'sample_idx': int(i),
                'subject': str(subj),
                'true_label': true_label_int,
                'true_label_name': imagery_class_names.get(true_label_int, 'unknown'),
                'predicted_label': pred_label,
                'predicted_label_name': imagery_class_names.get(pred_label, 'unknown'),
                'logits': logit_vec.tolist(),
                'softmax_probs': softmax_probs.tolist(),
                'correct': int(pred_label == true_label_int),
            })

        # Save detailed predictions to JSON
        predictions_file = output_dir / f"test_predictions_fold_{fold_id}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        # Compute and save summary statistics
        summary = self._compute_summary(predictions, fold_id, imagery_class_names)
        summary_file = output_dir / f"test_summary_fold_{fold_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Saved {len(predictions)} test predictions to {predictions_file}")
        print(f"✓ Saved test summary to {summary_file}")

        # Reset for next test phase (if any)
        self.test_predictions = []
        self.test_subjects = []
        self.test_logits = None
        self.test_targets = None

    @staticmethod
    def _compute_summary(predictions, fold_id, class_names):
        """Compute summary statistics from predictions."""
        if not predictions:
            return {}

        # Overall accuracy
        overall_acc = np.mean([p['correct'] for p in predictions])

        # Per-class accuracy
        class_accs = {}
        for cname in class_names.values():
            class_preds = [p for p in predictions if p['true_label_name'] == cname]
            if class_preds:
                class_accs[cname] = np.mean([p['correct'] for p in class_preds])
            else:
                class_accs[cname] = None

        # Per-subject accuracy
        unique_subjects = set(p['subject'] for p in predictions)
        per_subject_acc = {}
        for subj in sorted(unique_subjects):
            subj_preds = [p for p in predictions if p['subject'] == subj]
            per_subject_acc[subj] = {
                'accuracy': float(np.mean([p['correct'] for p in subj_preds])),
                'num_samples': len(subj_preds),
            }

        # Confusion matrix (simple counts)
        cm = np.zeros((3, 3), dtype=int)
        for p in predictions:
            true_idx = p['true_label']
            pred_idx = p['predicted_label']
            cm[true_idx, pred_idx] += 1

        summary = {
            'fold_id': fold_id,
            'num_samples': len(predictions),
            'overall_accuracy': float(overall_acc),
            'per_class_accuracy': class_accs,
            'per_subject_accuracy': per_subject_acc,
            'confusion_matrix': cm.tolist(),  # [true_label, pred_label]
            'confusion_matrix_labels': ['tennis', 'navigation', 'squeeze'],
        }

        return summary

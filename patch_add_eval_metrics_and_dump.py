"""
Patch models/lightning_model.py's imagery_task_classification branch of
_evaluate_metrics() to add the pieces needed for a thorough, paper-style
evaluation on top of what's already logged (test_acc, test_AUROC,
per-class pred_frac, confusion matrix):

  1. Per-class precision / recall / F1, logged as plain scalars (robust
     regardless of whether matplotlib is installed -- the existing
     confusion matrix is only recoverable as *numbers* if matplotlib is
     ABSENT, since add_figure() renders a picture, not data. These new
     scalars are always parseable from TensorBoard, matching the
     confusion-matrix-derived breakdown the paper implicitly reports).

  2. A per-sample test-time dump (softmax probabilities, predicted class,
     true class) written to <fold's log_dir>/test_predictions.pt whenever
     mode == 'test'. This is what's needed to reproduce something like
     Fig. 6b in Wang et al. 2026 (NeuroSTORM paper) -- a violin plot of
     the model's confidence (softmax probability assigned to the TRUE
     class), split by correct vs. incorrect predictions, showing whether
     the model is well-calibrated (confident when right, hesitant when
     wrong) rather than just reporting a single accuracy number.

Both additions are purely additive -- existing logged metrics/behavior
(test_acc, test_AUROC, pred_frac, confusion matrix image/text) are
unchanged. Since this only takes effect on runs made AFTER patching, to
backfill these on your already-completed 5 folds, use `--test_only` with
`--test_ckpt_path <fold's checkpoint>` to cheaply re-evaluate each fold's
saved checkpoint (no retraining needed) -- see evaluate_cv_thorough.py's
docstring for the exact commands.

Run from the repo root:
    python patch_add_eval_metrics_and_dump.py
"""
import io

TARGET = "models/lightning_model.py"

OLD_METRICS = '''            acc = acc_func(total_out_logits, targets)
            auroc = auroc_func(total_out_logits, targets)
            cm = cm_func(total_out_logits, targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)'''

NEW_METRICS = '''            acc = acc_func(total_out_logits, targets)
            auroc = auroc_func(total_out_logits, targets)
            cm = cm_func(total_out_logits, targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

            # Per-class precision / recall / F1, derived from the confusion matrix
            # (rows = true class, cols = predicted class). Logged as plain scalars
            # so they're always recoverable from TensorBoard -- unlike the
            # confusion matrix itself, which is only numeric when matplotlib is
            # ABSENT (add_figure() below renders a picture, not data, when it IS
            # installed). Classes with zero predictions/support get precision/
            # recall 0 rather than NaN (same convention as sklearn's
            # zero_division=0), since a class that never appears in a small test
            # fold shouldn't crash the eval.
            cm_f = cm.float()
            tp = cm_f.diag()
            predicted_count = cm_f.sum(dim=0)
            support = cm_f.sum(dim=1)
            precision = tp / predicted_count.clamp(min=1)
            recall = tp / support.clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
            for ci, cname in enumerate(class_names):
                self.log(f"{mode}_precision_{cname}", precision[ci], sync_dist=True)
                self.log(f"{mode}_recall_{cname}", recall[ci], sync_dist=True)
                self.log(f"{mode}_f1_{cname}", f1[ci], sync_dist=True)'''

OLD_TAIL = '''            except ImportError:
                if not getattr(self, '_warned_no_matplotlib', False):
                    print('[warn] matplotlib not installed -- skipping confusion matrix figure logging. '
                          'Run `pip install matplotlib` in this env to enable it. Logging a text dump instead.')
                    self._warned_no_matplotlib = True
                if self.logger is not None and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_text(f'{mode}_confusion_matrix', str(cm.cpu().numpy().tolist()), global_step=self.current_epoch)

            return'''

NEW_TAIL = '''            except ImportError:
                if not getattr(self, '_warned_no_matplotlib', False):
                    print('[warn] matplotlib not installed -- skipping confusion matrix figure logging. '
                          'Run `pip install matplotlib` in this env to enable it. Logging a text dump instead.')
                    self._warned_no_matplotlib = True
                if self.logger is not None and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_text(f'{mode}_confusion_matrix', str(cm.cpu().numpy().tolist()), global_step=self.current_epoch)

            # Per-sample test-time dump for confidence/calibration analysis
            # (Wang et al. 2026 Fig. 6b-style: distribution of the softmax
            # probability assigned to the TRUE class, split correct vs.
            # incorrect). Only written at test time -- not every val epoch --
            # to <fold's log_dir>/test_predictions.pt.
            if mode == 'test':
                try:
                    probs = torch.softmax(total_out_logits, dim=1)
                    log_dir = getattr(self.trainer, 'log_dir', None) if self.trainer is not None else None
                    if log_dir is not None:
                        dump_path = os.path.join(log_dir, 'test_predictions.pt')
                        torch.save({
                            'probs': probs.detach().cpu(),
                            'preds': preds.detach().cpu(),
                            'targets': targets.detach().cpu(),
                            'class_names': class_names,
                        }, dump_path)
                        print(f'[test predictions] saved per-sample probs/preds/targets to {dump_path}')
                    else:
                        print('[test predictions] skipped saving dump: trainer.log_dir is None')
                except Exception as e:
                    print(f'[test predictions] skipped saving dump: {e}')

            return'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    checks = (
        (OLD_METRICS, "acc/auroc/cm computation + logging block"),
        (OLD_TAIL, "matplotlib-ImportError fallback tail block"),
    )
    for old, label in checks:
        assert src.count(old) == 1, (
            "Expected exactly one match for the %s in %s -- file may already "
            "be patched or has changed unexpectedly." % (label, TARGET)
        )

    src = src.replace(OLD_METRICS, NEW_METRICS)
    src = src.replace(OLD_TAIL, NEW_TAIL)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - added per-class precision/recall/F1 scalar logging (test_precision_<class>, etc.)")
    print("  - added test-time per-sample dump to <log_dir>/test_predictions.pt for calibration analysis")


if __name__ == "__main__":
    main()

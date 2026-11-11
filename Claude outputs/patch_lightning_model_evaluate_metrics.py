"""
Patch for models/lightning_model.py: fixes the AUROC=0 flatline bug for
the Michigan imagery_task_classification task, and adds:
  - per-class prediction fraction scalars ({mode}_pred_frac_tennis/navigation/squeeze)
  - a 3x3 confusion matrix logged to TensorBoard (as a figure if matplotlib
    is installed, otherwise as a text dump so it never crashes training)

Run this from the repo root on the Linux machine:
    python3 patch_lightning_model_evaluate_metrics.py

It is idempotent-safe in the sense that it will refuse to apply (via the
assert) if the expected original text isn't found verbatim -- e.g. if it's
already been applied, or if lightning_model.py has since changed upstream.
"""
import os

path = os.path.join("models", "lightning_model.py")
if not os.path.exists(path):
    raise SystemExit(
        f"Could not find {path} -- run this script from the NeuroSTORM repo root "
        "(the directory containing models/lightning_model.py)."
    )

with open(path, "r") as f:
    content = f.read()

old = '''        if self.hparams.task_name == 'movie_classification':
            acc_func = MulticlassAccuracy(num_classes=self.hparams.num_classes).to(total_out_logits.device)
            auroc_func = MulticlassAUROC(num_classes=self.hparams.num_classes).to(total_out_logits.device)

            acc = acc_func(total_out_logits, total_out_target.long())

            self.log(f"{mode}_acc", acc, sync_dist=True)
            return

        if self.hparams.num_classes == 1:'''

count = content.count(old)
if count != 1:
    raise SystemExit(
        f"Expected exactly one match for the anchor text, found {count}. "
        "The file may have already been patched, or differs from what this "
        "script expects -- inspect models/lightning_model.py manually around "
        "the movie_classification branch of _evaluate_metrics before proceeding."
    )

new = '''        if self.hparams.task_name == 'movie_classification':
            acc_func = MulticlassAccuracy(num_classes=self.hparams.num_classes).to(total_out_logits.device)
            auroc_func = MulticlassAUROC(num_classes=self.hparams.num_classes).to(total_out_logits.device)

            acc = acc_func(total_out_logits, total_out_target.long())

            self.log(f"{mode}_acc", acc, sync_dist=True)
            return

        if self.hparams.task_name == 'imagery_task_classification':
            # Michigan blocks are per-block labeled (tennis/navigation/squeeze), so a
            # single subject contributes many blocks spanning ALL classes -- unlike
            # sex/age/diagnosis-style tasks where one subject has one label. The
            # subject-averaging path below (built for the latter) would average
            # together logits from all 3 classes per subject and compare against only
            # the first block's label, silently degenerating AUROC/accuracy. Evaluate
            # directly over every block-level prediction instead.
            imagery_class_names = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}
            class_names = [imagery_class_names.get(i, str(i)) for i in range(self.hparams.num_classes)]

            targets = total_out_target.long()
            preds = total_out_logits.argmax(dim=1)

            acc_func = MulticlassAccuracy(num_classes=self.hparams.num_classes).to(total_out_logits.device)
            auroc_func = MulticlassAUROC(num_classes=self.hparams.num_classes).to(total_out_logits.device)
            cm_func = MulticlassConfusionMatrix(num_classes=self.hparams.num_classes).to(total_out_logits.device)

            acc = acc_func(total_out_logits, targets)
            auroc = auroc_func(total_out_logits, targets)
            cm = cm_func(total_out_logits, targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

            # Per-class prediction fraction: how often each class is predicted,
            # regardless of true label. Shows up as scalar line charts in TensorBoard
            # and flags a model that has collapsed onto predicting one class.
            n = preds.numel()
            for ci, cname in enumerate(class_names):
                frac = (preds == ci).float().mean() if n > 0 else torch.tensor(0.0, device=preds.device)
                self.log(f"{mode}_pred_frac_{cname}", frac, sync_dist=True)

            # Log the confusion matrix (rows = true class, cols = predicted class) as
            # a TensorBoard figure. matplotlib isn't in requirements.txt, so fall back
            # to a plain-text dump under the same tag rather than crashing training.
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                cm_np = cm.cpu().numpy()
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(cm_np, cmap='Blues')
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right')
                ax.set_yticklabels(class_names)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'{mode} confusion matrix (epoch {self.current_epoch})')
                cm_max = cm_np.max() if cm_np.size > 0 else 0
                for i in range(cm_np.shape[0]):
                    for j in range(cm_np.shape[1]):
                        ax.text(j, i, int(cm_np[i, j]), ha='center', va='center',
                                color='white' if cm_np[i, j] > cm_max / 2 else 'black')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()

                if self.logger is not None and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_figure(f'{mode}_confusion_matrix', fig, global_step=self.current_epoch)
                plt.close(fig)
            except ImportError:
                if not getattr(self, '_warned_no_matplotlib', False):
                    print('[warn] matplotlib not installed -- skipping confusion matrix figure logging. '
                          'Run `pip install matplotlib` in this env to enable it. Logging a text dump instead.')
                    self._warned_no_matplotlib = True
                if self.logger is not None and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_text(f'{mode}_confusion_matrix', str(cm.cpu().numpy().tolist()), global_step=self.current_epoch)

            return

        if self.hparams.num_classes == 1:'''

content = content.replace(old, new)

old_import = "from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC"
if content.count(old_import) != 1:
    raise SystemExit("Could not find the expected torchmetrics.classification import line -- aborting without writing.")
new_import = "from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix"
content = content.replace(old_import, new_import)

with open(path, "w") as f:
    f.write(content)

print(f"Patched {path} OK.")
print("Run `python3 -m py_compile models/lightning_model.py` to sanity-check it compiles.")

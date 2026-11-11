"""
Make main.py's automatic post-fit trainer.test() call evaluate the BEST
validation checkpoint (per the ModelCheckpoint callback's monitor metric),
instead of whatever weights happen to still be in memory after the last
training epoch.

Background (established earlier in this project by reading both this repo's
main.py and the upstream CUHK-AIM-Group/NeuroSTORM main.py -- identical
logic in both): ModelCheckpoint is already configured to save the best-val
checkpoint to disk throughout training (monitor="valid_acc"/"valid_loss"/etc,
mode="max"/"min", filename="checkpt-{epoch:02d}-...") alongside last.ckpt.
But the final `trainer.test(model, dataloaders=data_module)` call was made
with NO ckpt_path, which means PyTorch Lightning tests whatever weights are
currently loaded in `model` -- the LAST training epoch's weights, not a
reload of the best checkpoint. This silently contradicts Wang et al. 2026's
own stated protocol: "the checkpoint with the lowest validation loss (or
highest relevant metric) was subsequently evaluated on the test set."

Previously the fix for an already-completed run was a manual, separate
`--test_only --test_ckpt_path <path>` invocation. This patch makes it
automatic and permanent for every FUTURE run (pretraining and
finetuning/downstream alike, since checkpoint_callback is defined for every
branch): after fit(), trainer.test() now reloads
`checkpoint_callback.best_model_path` if one was recorded, and only falls
back to the old (in-memory, last-epoch) behavior if that's empty (e.g. the
monitor metric was somehow never logged).

IMPORTANT: this does NOT retroactively fix runs already in flight or
completed before this patch was applied -- they already loaded the old
main.py into their process. For those, keep using the manual
`--test_only --test_ckpt_path <path>` backfill. Only runs started AFTER
this patch get the corrected behavior automatically.

Run from the repo root:
    python patch_test_best_checkpoint.py
"""
import io

TARGET = "main.py"

OLD = "        trainer.test(model, dataloaders=data_module)"

NEW = (
    "        # Evaluate the BEST-validation checkpoint (per checkpoint_callback's\n"
    "        # monitor metric above), not whatever weights happen to be in memory\n"
    "        # after the last training epoch -- matching Wang et al. 2026's own\n"
    "        # stated protocol (\"the checkpoint with the lowest validation loss\n"
    "        # (or highest relevant metric) was subsequently evaluated on the\n"
    "        # test set\"). Without this, trainer.test() below would silently\n"
    "        # evaluate last-epoch weights even though ModelCheckpoint has been\n"
    "        # saving a separate best-val checkpoint to disk the whole time.\n"
    "        best_ckpt_path = getattr(checkpoint_callback, 'best_model_path', '') or ''\n"
    "        if best_ckpt_path:\n"
    "            print(f'[diag] TEST_START using best-val checkpoint: {best_ckpt_path}', flush=True)\n"
    "            trainer.test(model, dataloaders=data_module, ckpt_path=best_ckpt_path)\n"
    "        else:\n"
    "            print('[diag] TEST_START no best_model_path recorded (monitor metric never '\n"
    "                  'logged?) -- falling back to in-memory (last-epoch) weights', flush=True)\n"
    "            trainer.test(model, dataloaders=data_module)"
)


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    assert src.count(OLD) == 1, (
        "Expected exactly one match for the trainer.test(...) call in %s -- "
        "file may already be patched or has changed unexpectedly." % TARGET
    )

    src = src.replace(OLD, NEW)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s: trainer.test() after fit() now reloads the best-val checkpoint." % TARGET)


if __name__ == "__main__":
    main()

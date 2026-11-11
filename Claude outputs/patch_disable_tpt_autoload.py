"""
Patch main.py to remove the "TPT auto-load" checkpoint-peeking block.

The bug: before building the model, main.py opens --load_model_path and, if
the checkpoint's own saved hyperparameters contain a non-'none' tpt_strategy
(e.g. a checkpoint that was itself produced by a TPT/prompt-tuning run),
it silently overwrites args.tpt_strategy with that value -- but ONLY checks
`args.tpt_strategy == "none"` to decide whether to do this. Since "none" is
ALSO argparse's default, this can't distinguish "the user didn't set it"
from "the user explicitly asked for full fine-tune" -- so an explicit
`--tpt_strategy none` gets silently clobbered whenever the checkpoint
carries a different value.

Concretely this hit `neurostorm_hcpya_task_state.ckpt`, which was itself
trained with tpt_strategy=linear, prompt_len=50: loading it with
`--tpt_strategy none` silently switched the run to TPT-linear (867
trainable params, 0.02% of the model) instead of the requested full
fine-tune. It then hard-crashes anyway, because the LightningDataModule was
already constructed earlier in main.py using the ORIGINAL args.tpt_strategy
('none'), while the LightningModule is built AFTER this override (now
'linear') -- so the two objects' saved hparams disagree and Lightning's
hparam-merge step raises MisconfigurationException at trainer.fit().

Fix: remove the auto-load block entirely. --tpt_strategy on the command
line (or its 'none' default) is now always respected exactly as passed,
regardless of what any given checkpoint's own hyperparameters say. This
also protects future runs that reuse a checkpoint produced by a TPT/prompt
run of this project's own -- same trap, same fix.

Run from the repo root:
    python patch_disable_tpt_autoload.py
"""
import io

TARGET = "main.py"

OLD = '''    # ------------ model -------------
    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        try:
            _peek = torch.load(args.load_model_path, map_location="cpu", weights_only=False)
            _hp = _peek.get("hyper_parameters", {})
            _ckpt_tpt = _hp.get("tpt_strategy", "none")
            if _ckpt_tpt and _ckpt_tpt != "none" and getattr(args, "tpt_strategy", "none") == "none":
                args.tpt_strategy = _ckpt_tpt
                args.prompt_len = int(_hp.get("prompt_len", args.prompt_len))
                print(f"[TPT auto-load] inherited from ckpt: tpt_strategy={args.tpt_strategy}, prompt_len={args.prompt_len}")
            del _peek, _hp
        except Exception as e:
            print(f"[TPT auto-load] skipped (could not peek ckpt hparams): {e}")

    model = LightningModel(data_module = data_module, **vars(args))'''

NEW = '''    # ------------ model -------------
    # (TPT auto-load-from-checkpoint block removed -- it silently overrode an
    # explicit --tpt_strategy none whenever the checkpoint's own saved hparams
    # carried a different value, since "none" is indistinguishable from "not
    # set". --tpt_strategy is now always exactly what was passed on the CLI.)
    model = LightningModel(data_module = data_module, **vars(args))'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    assert src.count(OLD) == 1, (
        "Expected exactly one match for the TPT auto-load block in %s -- "
        "file may already be patched or has changed unexpectedly." % TARGET
    )

    src = src.replace(OLD, NEW)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s: removed the TPT auto-load-from-checkpoint block." % TARGET)
    print("--tpt_strategy is now always taken literally from the CLI, never overridden by a checkpoint's own saved hparams.")


if __name__ == "__main__":
    main()

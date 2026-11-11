# Local (no-GPU) testing sandbox

Test code changes on a laptop in ~10 seconds before submitting big jobs to Great Lakes.

## What it is

- **Env:** `scripts/sandbox/environment-local.yml` -> conda env `neurostorm-sandbox`. Cross-platform, CPU/MPS torch 2.7.1 + Lightning 1.9.4. (`neurostorm_hpc.yml` is a linux-64/CUDA export and does not resolve on a Mac.)
- **`mamba_ssm` shim:** `scripts/sandbox/shims/mamba_ssm/` is a pure-PyTorch `Mamba` (same parameter names as the real one, so GPU checkpoints load). It is slow and only on `PYTHONPATH` for sandbox runs; it never shadows the real package if that is installed.
- **Synthetic data:** `scripts/sandbox/make_synthetic_michigan.py` writes a tiny Michigan-style dataset (6 subjects, 24^3 voxels, per-class signal) and a matching split file into `output/sandbox/` (gitignored).
- **Smoke run:** `scripts/sandbox/run_smoke.sh` runs the **real `main.py`** (data module -> NeuroSTORM -> train/val/test -> prediction callback) with a tiny model for 2 epochs.

## Local vs. cluster switch

`NEUROSTORM_ENV` decides which machine the code thinks it is on (`utils/runtime_env.py`). `set_env.sh` exports `greatlakes` (all `sbatch_*` files source it); `run_smoke.sh` / `run_greatlakes_checks.sh` export `local`. **Only the exact value `local`** skips the CUDA-only NVTX markers and CUDA event timers; unset or any other value runs the original code, so a missing variable on the cluster can't change GPU behaviour. If you run the sandbox by hand, `export NEUROSTORM_ENV=local` first.

## Use

```bash
make sandbox-env                 # once
conda activate neurostorm-sandbox
make sandbox                     # end-to-end smoke run (~10 s)
make sandbox-test                # pytest suite with the shim

# try a flag/option on the tiny run before submitting it to Slurm
# (args after `--` are appended to the main.py command; later flags win)
bash scripts/sandbox/run_smoke.sh -- --max_epochs 3 --tpt_strategy none
```

The script refuses to run where `sbatch` exists (the cluster); set `SANDBOX_ALLOW_CLUSTER=1` to override.

## What it catches / does not catch

Catches: import errors, argparse/yaml wiring, split-file and data-loading bugs, tensor-shape errors, loss/metric/callback crashes, CUDA-only calls that break on other devices.

Does **not** tell you anything about: accuracy, speed, memory/OOM, multi-GPU/DDP behaviour, real Mamba CUDA-kernel numerics, or real-data preprocessing. Always do a short real-data run on the cluster before a full job.

## Notes

- Sandbox runs execute from `output/sandbox/`, so `nas/splits` and `nas/output` are created there and never touch the repo's real `nas/`.
- Split files for all Michigan datasets (incl. `MICHIGAN_FULL_BLOCK`) live in `nas/splits/MICHIGAN/`; a missing or invalid (e.g. val == test) split aborts the run instead of generating a random one.
- Other datasets: add a generator next to `make_synthetic_michigan.py`; the on-disk format is `img/<subject>/data.pt` = `{frames: int8 [T,H,W,D], scale, num_frames}`.
- Known repo issue found while building this (not fixed): `LightningModel._compute_logits` does `logits.squeeze()`, so an eval batch with exactly one sample (split size % `eval_batch_size` == 1) becomes 1-D and `validation_epoch_end` crashes in `torch.cat`. The sandbox uses sizes that avoid it.

## Testing the Great Lakes launchers locally

`make sandbox-gl` (= `bash scripts/sandbox/run_greatlakes_checks.sh all`, ~10 min) runs the **unmodified** `run_finetune_scripts/greatlakes/{5fold,loocv}` runners and the three `sbatch_*` files against 26 synthetic subjects, with a wrapper that only shrinks `main.py`'s model/data/epochs (original args are logged and asserted). Stubs replace `srun`, `module`, `conda`, `nvidia-smi`; the job files' hard-coded conda path and `/nfs` `DATA_ROOT` are redirected to the sandbox. Stages: `setup | static | 5fold | loocv | sbatch`.

`verify_run.py` checks each fold: finished all epochs, best + last checkpoints saved, encoder initialised from the pretrained ckpt (and then updated), saved hparams match the launcher's intent, and predictions cover exactly the split file's held-out subjects (right count, balanced classes, probs/preds consistent). `check_real_ckpt_compat.py` checks your real `pt_neurostorm_mae_5ds.ckpt` against the real fine-tune architecture (`strict=False` would otherwise hide a key mismatch).

Not covered: Slurm itself (partition, time/memory limits, array behaviour), `set_env.sh`, real CUDA/Mamba, accuracy.

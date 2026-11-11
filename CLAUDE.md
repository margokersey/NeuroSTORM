# NeuroSTORM (fork) — Project Guide for Claude

Fork of CUHK-AIM-Group/NeuroSTORM (fMRI foundation model, PyTorch Lightning), used here for fine-tuning / LOOCV / permutation-testing experiments on the UMich Great Lakes HPC cluster. Upstream docs are in [README.md](README.md), [USER_GUIDE.md](USER_GUIDE.md), [INSTALLATION.md](INSTALLATION.md).

## Layout

- [main.py](main.py) — entry point (train / fine-tune / test via Lightning CLI args)
- [models/](models/) — NeuroSTORM, SwiFT, BNT, BrainGNN, LG-GNN, etc.; `lightning_model.py` is the LightningModule, `load_model.py` the model factory, `heads/` the task heads
- [datasets/](datasets/) — `data_module.py` (LightningDataModule, splits), `fmri_datasets.py` (voxel 4D), `roi_datasets.py` (ROI/FC), preprocessing + atlas utilities. Large data dirs here are gitignored.
- [callbacks/](callbacks/) — custom Lightning callbacks (e.g. per-fold prediction saving)
- [utils/](utils/) — argument parser (`parser.py`), losses, metrics, LR schedulers, Neptune logging
- [scripts/](scripts/) — split generation (`make_*_splits.py`), result aggregation/plotting (`extract_loocv_metrics.py`, `plot_loocv_results.py`, `analyze_loocv_permutation_results.py`), baselines, dataset download/preprocessing
- [run_finetune_scripts/](run_finetune_scripts/) — launch scripts; `greatlakes/5fold/` and `greatlakes/loocv/` hold Slurm (`sbatch_*`) and runner (`run_*`) scripts for Great Lakes
- [analysis/](analysis/) — our custom analysis/backfill scripts (not upstream): `compare_conditions*.py`, `backfill_*best_ckpt_test.sh`
- [notes/](notes/) — our LOOCV / permutation / label workflow docs (not upstream)
- [tests/](tests/) — pytest suite (`make test`, `make test-fast`; see [Makefile](Makefile))
- [docs/](docs/) — upstream project website (HTML)
- `CHANGES.md` is the running change log. Put new experiment docs in `notes/`, new custom analysis scripts in `analysis/`, never at the repo root.
- [pretrained_models/](pretrained_models/), `output/`, `data/` — weights/outputs/data; not for committing.

## Environment

- Work happens on the Great Lakes login node (shared!). Do not run training/heavy compute on the login node; submit with `sbatch`.
- Env setup: `source set_env.sh`; deps in `requirements.txt` / `neurostorm_hpc.yml`.

## Shared-filesystem rules

Follow the site policy in `/etc/claude-code/CLAUDE.md`: never search recursively outside this project or `$HOME`; use `-maxdepth`/`-xdev`; ask for exact paths elsewhere; don't poll `squeue`/`sacct` in loops (at most once a minute, or use `sbatch --wait`); stop any background process you start.

## Where things go (keep the repo organized)

| What | Where | Tracked? |
|---|---|---|
| Throwaway / exploratory scripts, one-off tests | `scratch/` | No (gitignored) |
| Results, figures, CSV/JSON outputs, logs | `output/` or `figure_outputs/` | No |
| Our reusable analysis / backfill scripts | `analysis/` | Yes |
| Reusable pipeline scripts (splits, preprocessing, aggregation) | `scripts/` | Yes |
| Our workflow docs / quickstarts | `notes/` (one doc per topic; update, don't add a new "quickstart/guide/implementation" variant) | Yes |
| Change log | `CHANGES.md` | Yes |

Rules for Claude:
- Never create files in the repo root. Root is for upstream entry points and config only.
- Put experiments and debugging scripts in `scratch/` first; promote to `analysis/` or `scripts/` only if the user wants to keep them.
- Before creating a new doc, check `notes/` for one to extend. Do not create docs the user didn't ask for.
- A pre-commit hook (`.githooks/pre-commit`, enable with `make hooks`) blocks files >5 MB, checkpoints/logs/pickles/npy, and result-type files at the root. Don't bypass with `--no-verify` without asking.

## Git management

- Remote: `origin` = `git@github.com:margokersey/NeuroSTORM.git` (the user's fork). Default branch: `main`. Push over SSH from the cluster.
- This cluster checkout is the primary working copy and the user also keeps a laptop copy. **Keep GitHub in sync**: when asked to commit, also push, and `git pull --rebase origin main` first if the remote is ahead. Never force-push to `main` without explicit instruction.
- Only commit/push when the user asks. When they do, commit all intended work, not just the file last edited.
- Before committing: run `git status` and `git diff --stat`; stage specific paths (or `git add -A` after confirming nothing unwanted is untracked).
- **Never commit**: data, checkpoints (`*.ckpt`, `*.pt`, `*.pth`), Slurm logs (`slurm-*.out`), `output/`, `figure_outputs/`, wheels, `__pycache__`, credentials/tokens (Neptune API keys etc.). Check `.gitignore` and add patterns rather than committing such files. Flag any file over ~5 MB before adding it.
- Commit messages: short imperative summary line (≤72 chars, e.g. `Add LOOCV permutation testing scripts`), optional body explaining why. Group unrelated changes into separate commits (code vs. docs vs. scripts).
- Moves/renames: use `git mv` or stage both sides so history shows a rename.
- Pull requests: end the description with the attribution line the harness specifies. Commits end with the Co-Authored-By trailer the harness specifies.
- Upstream: `https://github.com/CUHK-AIM-Group/NeuroSTORM` is not configured as a remote. Add it as `upstream` only if the user asks to sync from it.
- Documentation: when changing experiment behavior (LOOCV, permutation tests, prediction outputs), add a short entry to `CHANGES.md` in the same commit.

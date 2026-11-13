# Custom analysis scripts

Our own scripts (not part of upstream NeuroSTORM) for evaluating fine-tuning runs.

- `compare_conditions.py`, `compare_conditions_v2.py` — compare conditions across LOOCV folds (v2 adds raincloud plots).
- `backfill_best_ckpt_test.sh`, `backfill_loocv_best_ckpt_test.sh` — re-run test evaluation with the best checkpoint for finished folds.
- `two_stage_decoding.py`, `plot_two_stage_decoding.py` — compose Stage A (Task vs. No Task) with the 3-way LOOCV predictions, and plot the result (works on partial folds; re-run as folds finish).

Default outputs go to `output/` (gitignored). Workflow notes live in `../notes/`.

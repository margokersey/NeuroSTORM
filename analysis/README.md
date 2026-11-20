# Custom analysis scripts

Our own scripts (not part of upstream NeuroSTORM) for evaluating fine-tuning runs.

- `compare_conditions.py`, `compare_conditions_v2.py` — OLD condition-comparison scripts, kept for reference only; not used by any current workflow.
- `backfill_best_ckpt_test.sh`, `backfill_loocv_best_ckpt_test.sh` — re-run test evaluation with the best checkpoint for finished folds.
- `two_stage_decoding.py`, `plot_two_stage_decoding.py` — compose Stage A (Task vs. No Task) with the 3-way LOOCV predictions, and plot the result (works on partial folds; re-run as folds finish).
- `predict_stage_b_on_stage_a_blocks.py`, `sbatch_stage_b_predict.sh` — run each fold's existing 3-way checkpoint on all Stage A test blocks (real task + mock rest); inference only, GPU node.
- `benchmark_decoders.py`, `sbatch_benchmark_decoders.sh` — benchmark GLM template matching, logistic regression, linear SVM, PCA+LDA and random forest against NeuroSTORM on the 3-way task, trained on run 1 and scored per propofol state (run 1, pre-LOR, LOR, ROR, recovery). CPU only; features are cached, so re-evaluation is cheap.
- `plot_benchmark_decoders.py` — accuracy-by-state line plot of every benchmark model vs. NeuroSTORM, from the benchmark's `summary.csv`.
- `classify_squeeze_force.py`, `plot_missed_squeezes.py`, `plot_squeeze_missed_vs_model.py`, `plot_squeeze_force_traces.py`, `plot_squeeze_notask_vs_force.py` — classify each squeeze block from the force traces (squeezed / early / missed, weak flag), plot missed squeezes per subject and run, and compare with the 3-way model's per-block correctness (the traces script draws force lines with the model's call under each squeeze block).

- `qc_dashboard/` — interactive QC dashboard (Streamlit) for fMRIPrep, XCP-D and `data.pt`: `scan.py` builds a cache, `app.py` serves it, `sbatch_qc_dashboard.sh` runs both on a compute node behind an SSH tunnel. See `../notes/QC_DASHBOARD.md`.

Default outputs go to `output/` (gitignored). Workflow notes live in `../notes/`.

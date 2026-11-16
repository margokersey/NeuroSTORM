# LOR task decoding: scripts and motion notes

Each fold's existing 3-way (tennis / navigation / squeeze) LOOCV checkpoint is run, inference only, on the held-out subject's imagery runs 2-4, tagged by LOR segment.

## Scripts

| Step | Script |
|---|---|
| Predict runs 2/3 (and 4 with `LOR_RUNS=4`) | `analysis/predict_lor_task_decoding.py`, submitted with `analysis/sbatch_lor_task_decoding.sh` |
| Accuracy by segment (run 1 awake baseline, pre-LOR, LOR, ROR, run 4) | `analysis/plot_lor_task_decoding.py` |
| Accuracy vs head motion, one panel per run | `analysis/plot_lor_accuracy_vs_motion.py` |

See each script's docstring for commands. Run 1 accuracy comes from the fold's own saved test predictions.

## Motion data (XCP-D)

Derivatives: `/nfs/turbo/umms-huanglab/pipeline/postproc/fmriprep-25.2.5_xcp_d-26.1.1/Michigan_Dataset_1/noGSR_highpass_fd-0.8_nosmooth/sub-*/func/`. Per run: `*_task-imagery_run-0N_motion.tsv` (columns: rot/trans, `rmsd`, `framewise_displacement`) and `*_outliers.tsv` (one 0/1 censor-flag column, also named `framewise_displacement`). Run tags are zero-padded (`run-01`).

- **Censor threshold = FD >= 0.8 mm** (verified: in every run, min FD among flagged volumes >= 0.80 and max FD among kept volumes <= 0.80; `% FD > 0.8` equals `% censored` everywhere). XCP-D's default is 0.3 mm, so this is lenient.
- **Run lengths:** run 1 and run 4 = 1145 volumes, run 2 and run 3 = 2270 (TR 0.8 s). sub-29 run 3 has only 1417 (excluded from decoding, frame count != timing file); sub-29 has no run 4.
- **Censoring by run (% of volumes):** run 1 mostly < 1% (max 2.5%); run 2 mostly < 6% (max sub-19, 10.5%); run 3 is the worst (sub-16 44%, sub-19 28%, sub-20 26%, sub-29 25%, sub-25 22%); run 4 mostly < 5% (max sub-17, 7.8%). Run 3 is also the after-ROR run, so motion and the LOR-recovery effect may be confounded.
- **Do not use mean FD.** Run 3 has extreme single-volume spikes (max FD: sub-16 1007 mm, sub-25 355 mm, sub-29 203 mm), which dominate the mean. Use % volumes with FD > 0.3 mm (default in the plot script), % censored, or median FD.

## Censored volumes are still in the data the model saw

The denoised BOLD (`*_desc-denoised_bold.nii.gz`) has the full run length even for the most-censored subjects (sub-16, -19, -20, -25 run 3: 2270 volumes, same as `n_vol`). So censored volumes were not removed, which is also why the block timing still lines up. XCP-D's `--output-type` is either `censored` (flagged volumes removed) or `interpolated` (kept, filled from neighbours); since volumes are present this is most likely `interpolated`, **but this is not confirmed**. The denoised-BOLD JSON lists only the high-pass filter (0.008 Hz, order 2) and names `outliers.tsv` as a source; it does not say which. To confirm: look for `--output-type` in the XCP-D submission script / `dataset_description.json`, or compare a flagged volume with its neighbours for one subject (sub-16 run 3).

Consequences for interpretation: the model decoded the high-motion time points (interpolated or otherwise), and the motion metrics above describe the pre-interpolation data. State this in the caption of the accuracy-vs-motion figure.

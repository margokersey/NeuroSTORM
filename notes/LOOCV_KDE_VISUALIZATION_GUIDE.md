# LOOCV Permutation Testing: KDE Visualization (superseded)

The per-sample KDE plot is gone from `scripts/analyze_loocv_permutation_results.py`. It
smoothed binary correct/incorrect values with a KDE, which carries no information
beyond accuracy, and it treated samples from the same subject as independent.

The script now draws a 6-panel figure from subject-level metrics: per-subject
performance, the permutation null, summary metrics with CIs, the confusion matrix,
ROC curves and calibration. See the "Analyze results" section of
[LABEL_PERMUTATION_TESTING.md](LABEL_PERMUTATION_TESTING.md).

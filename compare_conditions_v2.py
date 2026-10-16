"""
compare_conditions.py

Generic comparison of ANY set of NeuroSTORM fine-tuning conditions -- each
condition is just a project_name glob (5 folds or however many you ran) --
so a future test that doesn't match today's naming convention at all needs
no code changes, only one more --condition flag.

WHAT COUNTS AS A "CONDITION"
  --condition "label|project_name_glob[|group[|facet]]"   (repeatable)
    label:  display name shown on the figures / in the CSV
              (e.g. "No smooth, none")
    glob:   glob pattern (relative to --output_root) matching every fold's
            project directory for this condition, e.g.
              "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_fold-*"
    group:  optional (default: same as label). Conditions sharing a group
            get the same color, and the training-curve figure facets by
            group -- use it to cluster e.g. "none" vs "prompt_ln" within
            one preprocessing bucket.
    facet:  optional (default: "", i.e. everything in one facet). Splits
            the RAINCLOUD and PER-CLASS figures into separate subplot rows
            -- use this when two sets of conditions sit on very different
            scales (e.g. "none" strategies scoring ~0.8 acc vs "prompt_ln"
            collapsing near chance) so one doesn't visually flatten the
            other.

Fold numbers are read from a "fold-<N>" substring in each matched directory
name if present (for CSV ordering only); if absent, folds are just numbered
in the order glob() returns them.

WHICH METRICS THIS SCRIPT TREATS AS "THE" METRICS, AND WHY
  - CHOOSING among conditions uses VALIDATION metrics (valid_AUROC and
    valid_acc, co-primary), not test -- picking whichever condition has the
    best TEST number is itself a form of test-set leakage (you'd be using
    held-out data to make a modeling decision), which matters a lot with
    only ~26 subjects and 10 conditions to compare. Test is for reporting
    the winner's final, honest number once the pick is already made -- see
    "Statistical comparison" below and run_stat_tests()'s docstring.
  - Validation numbers are read at each fold's BEST/CHECKPOINTED epoch
    (whichever epoch main.py's ModelCheckpoint(monitor="valid_acc",
    mode="max") actually saved), not the final epoch of training -- see
    load_best_epoch_validation()'s docstring. Using the final epoch instead
    would mix in however training happened to end, which can understate a
    condition that peaked early and was correctly checkpointed there.
  - valid_pred_frac_<class> is the class-collapse GUARDRAIL, not a ranking
    metric: class collapse (the model predicting one class for nearly
    everything) has been the single most recurring failure mode in this
    project, and it can hide behind an OK-looking aggregate accuracy.
    Checked on validation data (never test) via check_collapse() -- a
    condition that fails this is excluded from being called "best"
    regardless of its AUROC/accuracy, not merely down-ranked.
  - Fold-to-fold consistency (the SEM annotations / point spread already in
    the raincloud plot, and the SEM tie-break in run_stat_tests) is a real
    tiebreaker: with only ~26 subjects, a condition whose mean is carried by
    one great fold and one terrible fold is a shakier bet than a
    slightly-lower but more consistent one.
  - Train/valid loss and accuracy CURVES (the per-epoch trajectories, not
    the single best-epoch scalar above) are diagnostic only (did training
    converge smoothly? overfit? diverge?) -- not part of the "best
    condition" decision.
  - Statistical significance (Friedman omnibus + pairwise Wilcoxon vs. the
    top candidate, Holm-corrected) is a SUPPLEMENTARY sanity check, not a
    hard gate: with only 5 folds the tests are very underpowered (a paired
    Wilcoxon can't reach p<0.05 with 5 folds no matter the effect size), so
    treat "not significant" as "can't tell from this little data", not
    "no real difference". See run_stat_tests()'s docstring.

WHAT IT PRODUCES
  1. condition_raw.csv        -- tidy long-format per-fold test AND
                                  validation metrics (condition, group,
                                  facet, fold_id, metric, value; metric names
                                  are prefixed test_/valid_). Source of truth
                                  for any figure you want later that this
                                  script doesn't make.
  2. Console summary           -- mean +/- SEM per condition per test AND
                                  validation metric.
  3. Statistical comparison    -- (validation metrics; skip with
                                  --skip_stats) class-collapse guardrail
                                  check, a Friedman omnibus test across all
                                  non-collapsed conditions, and pairwise
                                  Wilcoxon signed-rank tests (Holm-corrected)
                                  between the top candidate and every other
                                  non-collapsed condition. Also saved to
                                  condition_stats.csv. See run_stat_tests()'s
                                  docstring for the full method and its
                                  (significant) small-n power caveats.
  4. condition_raincloud.png/.pdf
     condition_validation_raincloud.png/.pdf
                               -- one row of subplots per facet, one column
                                  per metric (default: acc, AUROC): horizontal
                                  half-violin + boxplot + individual fold
                                  points per condition, colored by group
                                  (marker shape cycles within a group), a
                                  dashed "chance" reference line, and each
                                  row's mean +/- SEM printed next to it. Test
                                  and validation versions are separate files;
                                  validation is what "Statistical comparison"
                                  above is actually computed from.
  5. condition_perclass.png/.pdf
     condition_perclass_validation.png/.pdf
                               -- one row of subplots per facet, one column
                                  per per-class metric (default: pred_frac,
                                  f1): horizontal grouped bars, one bar per
                                  class, mean +/- SEM across folds, with a
                                  dashed chance line at 1/num_classes. This
                                  is the class-collapse check -- a condition
                                  with one class's pred_frac far from the
                                  others (even if aggregate acc/AUROC look
                                  fine) is not actually the best model. The
                                  validation version is what check_collapse()
                                  actually gates on; the test version is
                                  purely for your own visual confirmation.
                                  Best-effort: these tags only exist for
                                  folds re-tested after
                                  patch_add_eval_metrics_and_dump.py: a
                                  facet/metric with no data is skipped with
                                  a printed note, not a crash.
  6. condition_curves_loss.png/.pdf
     condition_curves_acc.png/.pdf
                               -- per-epoch train (dashed) vs valid (solid)
                                  curves, mean +/- SEM band across folds,
                                  faceted by group (all conditions in a
                                  group overlaid in one panel, shared y-axis
                                  across panels for comparability). Test
                                  metrics are a single end-of-training
                                  number, not a curve over epochs, so they
                                  are deliberately left out of these figures
                                  -- see the raincloud plot / condition_raw.csv
                                  for those.

                                  IMPORTANT: for each condition, this reads
                                  train_*/valid_* tags from WHICHEVER
                                  lightning_logs/version_* under that
                                  project has the most datapoints for that
                                  tag -- NOT just the latest version. A
                                  later version can be a --mode test -only
                                  best-checkpoint backfill pass (see
                                  backfill_best_ckpt_test.sh) whose event
                                  file contains ONLY test_* scalars and no
                                  training history at all; grabbing "the
                                  latest version" naively would silently
                                  produce a blank curve for exactly the
                                  conditions you backfilled. Test metrics
                                  themselves still come from the LATEST
                                  version only (that's the correct,
                                  backfilled best-ckpt value).

                                  Also: train_* tags are logged per training
                                  STEP in this codebase (Lightning's
                                  on_epoch=False for train), not per epoch
                                  like valid_*, so each fold's train curve is
                                  re-binned onto the same number of points as
                                  that fold's valid curve (by step position,
                                  not true epoch boundaries) before averaging
                                  across folds -- an approximation, fine for
                                  comparing convergence shape across
                                  conditions, not for reading off an exact
                                  epoch number.

USAGE (Margo's current 10-condition comparison, split into "none" vs
"prompt_ln" facets):
    python "Claude outputs/compare_conditions.py" \
      --condition "No smooth, none|FIXED_mich_xcpd_nosmooth_tpt_strategy-none_fold-*|No smooth|none" \
      --condition "No smooth, prompt_ln|FIXED_mich_xcpd_nosmooth_tpt_strategy-prompt_ln_fold-*|No smooth|prompt_ln" \
      --condition "Smooth, none|FIXED_mich_xcpd_smooth_tpt_strategy-none_fold-*|Smooth|none" \
      --condition "Smooth, prompt_ln|FIXED_mich_xcpd_smooth_tpt_strategy-prompt_ln_fold-*|Smooth|prompt_ln" \
      --condition "No smooth + voxel-znorm, none|FIXED_mich_xcpd_nosmooth_voxelznorm_tpt_strategy-none_fold-*|No smooth + voxel-znorm|none" \
      --condition "No smooth + voxel-znorm, prompt_ln|FIXED_mich_xcpd_nosmooth_voxelznorm_tpt_strategy-prompt_ln_fold-*|No smooth + voxel-znorm|prompt_ln" \
      --condition "Smooth + voxel-znorm, none|FIXED_mich_xcpd_smooth_voxelznorm_tpt_strategy-none_fold-*|Smooth + voxel-znorm|none" \
      --condition "Smooth + voxel-znorm, prompt_ln|FIXED_mich_xcpd_smooth_voxelznorm_tpt_strategy-prompt_ln_fold-*|Smooth + voxel-znorm|prompt_ln" \
      --condition "AFNI, none|FIXED_afni_smooth_voxelznorm_tpt_strategy-none_fold-*|AFNI|none" \
      --condition "AFNI, prompt_ln|FIXED_afni_smooth_voxelznorm_tpt_strategy-prompt_ln_fold-*|AFNI|prompt_ln"

To add a future test with a totally different naming scheme, just add one
more --condition "label|glob[|group[|facet]]" -- nothing else needs to
change. Omit group/facet entirely and it behaves like one flat, ungrouped,
unfaceted comparison.

Optional:
    --output_root nas/output/neurostorm     (default)
    --out_dir "Claude outputs/condition_sweep"   (default; created if missing)
    --test_metrics acc,AUROC                (default; comma-separated,
                                              matched against test_<metric>
                                              scalar tags; raincloud plot)
    --perclass_metrics pred_frac,f1         (default; comma-separated,
                                              matched against
                                              test_<metric>_<class> scalar
                                              tags; per-class figure)
    --curve_metrics loss,acc                (default; comma-separated,
                                              matched against train_<metric>
                                              / valid_<metric> scalar tags)
    --class_names tennis,navigation,squeeze (default; used for the
                                              per-class figure)
    --chance "acc=0.333,AUROC=0.5"          (default shown; dashed reference
                                              line per metric in the
                                              raincloud plot -- pass
                                              "metric=" with no value, or
                                              omit a metric, to skip its
                                              line)
    --axis_range "acc=0:1,AUROC=0:1"        (default shown; fixed x-axis
                                              range per metric in the
                                              raincloud plot -- a metric not
                                              listed falls back to an
                                              auto-fit range with padding)
"""
import argparse
import glob
import os
import re
import statistics
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("This script needs the `tensorboard` package: pip install tensorboard", file=sys.stderr)
    raise

try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    HAVE_SCIPY_STATS = True
except ImportError:
    HAVE_SCIPY_STATS = False

FOLD_RE = re.compile(r"fold-(\d+)")

DEFAULT_CHANCE = {"acc": 1.0 / 3.0, "AUROC": 0.5}
DEFAULT_AXIS_RANGE = {"acc": (0.0, 1.0), "AUROC": (0.0, 1.0)}
DEFAULT_COLLAPSE_THRESHOLD = 0.5  # flag a condition if (max-min) mean pred_frac across classes exceeds this


# ----------------------------------------------------------------------
# Condition parsing
# ----------------------------------------------------------------------
def parse_condition(spec):
    parts = spec.split("|")
    if len(parts) == 2:
        label, glob_pat = parts
        group, facet = label, ""
    elif len(parts) == 3:
        label, glob_pat, group = parts
        facet = ""
    elif len(parts) == 4:
        label, glob_pat, group, facet = parts
    else:
        raise SystemExit(
            f"--condition must be 'label|glob', 'label|glob|group', or "
            f"'label|glob|group|facet', got: {spec!r}"
        )
    return {"label": label.strip(), "glob": glob_pat.strip(),
            "group": group.strip(), "facet": facet.strip()}


def parse_kv_floats(spec, defaults):
    out = dict(defaults)
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, val = chunk.partition("=")
        key = key.strip()
        val = val.strip()
        if val == "":
            out.pop(key, None)
        else:
            out[key] = float(val)
    return out


def parse_axis_range(spec, defaults):
    out = dict(defaults)
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, val = chunk.partition("=")
        key = key.strip()
        if ":" not in val:
            raise SystemExit(f"--axis_range entries must be 'metric=lo:hi', got: {chunk!r}")
        lo, hi = val.split(":")
        out[key] = (float(lo), float(hi))
    return out


def resolve_folds(output_root, glob_pat):
    matches = sorted(glob.glob(os.path.join(output_root, glob_pat)))
    folds = []
    for m in matches:
        name = os.path.basename(m)
        mobj = FOLD_RE.search(name)
        fold_id = int(mobj.group(1)) if mobj else len(folds) + 1
        folds.append((fold_id, m))
    folds.sort(key=lambda x: x[0])
    return folds


# ----------------------------------------------------------------------
# TensorBoard event reading
# ----------------------------------------------------------------------
def find_all_version_dirs(project_dir):
    pattern = os.path.join(project_dir, "lightning_logs", "version_*")
    return sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_", 1)[-1]))


def find_event_file(project_dir):
    version_dirs = find_all_version_dirs(project_dir)
    if not version_dirs:
        return None
    latest = version_dirs[-1]
    events = glob.glob(os.path.join(latest, "events.out.tfevents.*"))
    return events[0] if events else None


def find_best_event_for_tags(project_dir, wanted_tags):
    """Pick the best event file for each requested tag across all version_*
    directories. This avoids a common issue where the newest version is a
    test-only backfill with no validation history, while an earlier version has
    the actual valid_* metrics.
    """
    best = {}
    for vdir in find_all_version_dirs(project_dir):
        events = glob.glob(os.path.join(vdir, "events.out.tfevents.*"))
        if not events:
            continue
        ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        for tag in wanted_tags:
            if tag not in available:
                continue
            series = [(s.step, s.value) for s in ea.Scalars(tag)]
            prev = best.get(tag)
            if prev is None or len(series) > len(prev[1]):
                best[tag] = (events[0], series)
    return {tag: event_path for tag, (event_path, _) in best.items()}


def load_scalars_from_version(version_dir, tags=None):
    events = glob.glob(os.path.join(version_dir, "events.out.tfevents.*"))
    if not events:
        return {}
    ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    wanted = available if tags is None else (available & set(tags))
    return {tag: [(s.step, s.value) for s in ea.Scalars(tag)] for tag in wanted}


def load_all_scalars(event_path):
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return {tag: [(s.step, s.value) for s in ea.Scalars(tag)] for tag in tags}


def load_best_epoch_validation(project_dir, val_metrics, perclass_metrics, class_names, monitor="valid_acc"):
    """Find the epoch (TensorBoard step) that maximizes `monitor` -- exactly
    the epoch main.py's ModelCheckpoint(monitor="valid_acc", mode="max")
    actually checkpointed for every classification task in this codebase
    (see main.py's ModelCheckpoint block) -- and read every other
    requested valid_<metric> / valid_<perclass>_<class> tag AT THAT SAME
    STEP, from the SAME version_dir `monitor`'s series came from.

    This is what makes it valid to use these numbers (instead of test
    metrics) to choose among conditions: they reflect the actual
    checkpointed model for each fold, not whatever epoch training happened
    to end on. Using the FINAL epoch instead (as this script did before)
    silently mixes in however training ended, which can be well past the
    checkpointed peak once validation performance starts degrading --
    understating a condition that peaked early and was correctly
    checkpointed there.

    All of a fold's valid_* tags (loss, acc, AUROC, per-class
    precision/recall/f1/pred_frac) are logged together inside the same
    validation-epoch hook (models/lightning_model.py's _evaluate_metrics
    call with mode="valid"), so they all live in the SAME version_dir and
    share the SAME step numbering as `monitor` -- no cross-version lookup
    needed here (unlike test metrics, which is the reason
    find_best_event_for_tags exists at all: a --mode test-only backfill
    pass creates a newer version with ONLY test_* tags and logs no valid_*
    at all, so it can never win this search).

    Returns (dict of "valid_<tag>" -> value, best_step), or (None, None)
    if `monitor` isn't found in any version_dir for this project.
    """
    best_series = None
    best_vdir = None
    for vdir in find_all_version_dirs(project_dir):
        scalars = load_scalars_from_version(vdir, tags=[monitor])
        series = scalars.get(monitor)
        if series and (best_series is None or len(series) > len(best_series)):
            best_series = series
            best_vdir = vdir
    if best_series is None:
        return None, None

    best_step, _best_value = max(best_series, key=lambda sv: sv[1])

    wanted_tags = [f"valid_{m}" for m in val_metrics]
    for prefix in perclass_metrics:
        for cname in class_names:
            wanted_tags.append(f"valid_{prefix}_{cname}")
    all_scalars = load_scalars_from_version(best_vdir, tags=wanted_tags)

    out = {}
    for tag, series in all_scalars.items():
        if not series:
            continue
        step_map = dict(series)
        if best_step in step_map:
            out[tag] = step_map[best_step]
        else:
            # epoch-aligned logging should match exactly; tolerate an
            # off-by-one step number between tags rather than dropping data
            nearest = min(series, key=lambda sv: abs(sv[0] - best_step))
            out[tag] = nearest[1]
    return out, best_step


def holm_bonferroni(pvals):
    """Holm's step-down multiple-comparisons correction. `pvals` is a list
    or array; returns adjusted p-values in the SAME order as the input,
    each monotonic and capped at 1.0. Preferred over a flat Bonferroni
    here because it's less conservative while still controlling the
    family-wise error rate -- worth it given how few comparisons and how
    little power there already is with 5 folds."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def load_curve_scalars(project_dir, curve_tags):
    """Search ALL lightning_logs/version_* dirs for train_*/valid_* curve
    tags (NOT just the latest -- see module docstring for why), keeping
    whichever version has the most datapoints for each tag independently."""
    best = {}
    for vdir in find_all_version_dirs(project_dir):
        scalars = load_scalars_from_version(vdir, tags=curve_tags)
        for tag, series in scalars.items():
            if tag not in best or len(series) > len(best[tag]):
                best[tag] = series
    return best


def final_value(series):
    return series[-1][1] if series else None


def bin_to_n_points(series, n_bins):
    """Average a (step, value) series into n_bins equal-width bins spanning
    its own step range -- used to align per-step train_* logging onto the
    same number of points as per-epoch valid_* logging, for the SAME fold."""
    if not series or n_bins <= 0:
        return np.full(max(n_bins, 0), np.nan)
    steps = np.array([s for s, _ in series], dtype=float)
    values = np.array([v for _, v in series], dtype=float)
    lo, hi = steps.min(), steps.max()
    if hi == lo:
        return np.full(n_bins, values.mean())
    bin_idx = np.clip(((steps - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = values[bin_idx == b]
        if len(sel):
            out[b] = sel.mean()
    return out


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def collect(conditions, output_root, test_metrics, val_metrics, curve_metrics,
            class_names, perclass_metrics, val_checkpoint_monitor="acc"):
    raw_rows = []          # per-fold test/validation metrics -> CSV / raincloud / per-class figure
    curve_rows = []        # per-fold, per-epoch-bin curve values -> CSV
    fold_curves = {}       # (label, tag) -> list of np.array, one per fold (for averaging)

    curve_tags = []
    for m in curve_metrics:
        curve_tags.append(f"train_{m}")
        curve_tags.append(f"valid_{m}")

    for cond in conditions:
        folds = resolve_folds(output_root, cond["glob"])
        if not folds:
            print(f"WARNING: no directories matched for condition '{cond['label']}' "
                  f"(glob: {cond['glob']!r} under {output_root})", file=sys.stderr)
            continue

        for fold_id, project_dir in folds:
            # --- test metrics: LATEST version only (the correct, possibly
            #     best-ckpt-backfilled, value) ---
            event_path = find_event_file(project_dir)
            if event_path is None:
                print(f"  SKIPPED (no event file): {project_dir}", file=sys.stderr)
                continue
            test_scalars = load_all_scalars(event_path)

            for m in test_metrics:
                tag = f"test_{m}"
                v = final_value(test_scalars.get(tag, []))
                if v is None:
                    print(f"  WARNING: '{tag}' missing for {project_dir}", file=sys.stderr)
                    continue
                raw_rows.append({"condition": cond["label"], "group": cond["group"],
                                  "facet": cond["facet"], "fold_id": fold_id,
                                  "metric": f"test_{m}", "value": v})

            for prefix in perclass_metrics:
                for cname in class_names:
                    tag = f"test_{prefix}_{cname}"
                    v = final_value(test_scalars.get(tag, []))
                    if v is not None:
                        raw_rows.append({"condition": cond["label"], "group": cond["group"],
                                          "facet": cond["facet"], "fold_id": fold_id,
                                          "metric": f"test_{prefix}_{cname}", "value": v})

            # --- validation metrics: BEST-EPOCH values (the epoch actually
            #     checkpointed, monitor="valid_acc") -- see
            #     load_best_epoch_validation()'s docstring. This is what
            #     makes it valid to use these numbers, rather than test
            #     metrics, to choose among conditions without touching
            #     held-out test data (see run_stat_tests / the module
            #     docstring's metrics section). Also pulls valid_pred_frac_*
            #     for the class-collapse guardrail, checked on validation
            #     data for the same reason.
            if val_metrics:
                best_vals, best_step = load_best_epoch_validation(
                    project_dir, val_metrics, perclass_metrics, class_names,
                    monitor=f"valid_{val_checkpoint_monitor}")
                if best_vals is None:
                    print(f"  WARNING: 'valid_{val_checkpoint_monitor}' (checkpoint monitor) missing "
                          f"for {project_dir} -- skipping validation metrics for this fold", file=sys.stderr)
                else:
                    for m in val_metrics:
                        tag = f"valid_{m}"
                        v = best_vals.get(tag)
                        if v is None:
                            print(f"  WARNING: '{tag}' missing at the best epoch for {project_dir}", file=sys.stderr)
                            continue
                        raw_rows.append({"condition": cond["label"], "group": cond["group"],
                                          "facet": cond["facet"], "fold_id": fold_id,
                                          "metric": tag, "value": v})
                    for prefix in perclass_metrics:
                        for cname in class_names:
                            tag = f"valid_{prefix}_{cname}"
                            v = best_vals.get(tag)
                            if v is not None:
                                raw_rows.append({"condition": cond["label"], "group": cond["group"],
                                                  "facet": cond["facet"], "fold_id": fold_id,
                                                  "metric": tag, "value": v})

            # --- curves: best version PER TAG across the whole project dir
            #     (see load_curve_scalars docstring) ---
            curve_scalars = load_curve_scalars(project_dir, curve_tags)

            for m in curve_metrics:
                valid_series = curve_scalars.get(f"valid_{m}", [])
                train_series = curve_scalars.get(f"train_{m}", [])
                n_epochs = len(valid_series)
                if n_epochs == 0:
                    continue
                valid_curve = np.array([v for _, v in valid_series], dtype=float)
                train_curve = bin_to_n_points(train_series, n_epochs)

                key_valid = (cond["label"], f"valid_{m}")
                key_train = (cond["label"], f"train_{m}")
                fold_curves.setdefault(key_valid, []).append(valid_curve)
                fold_curves.setdefault(key_train, []).append(train_curve)

                for epoch_idx, v in enumerate(valid_curve):
                    curve_rows.append({"condition": cond["label"], "group": cond["group"],
                                        "facet": cond["facet"], "fold_id": fold_id,
                                        "tag": f"valid_{m}", "epoch": epoch_idx, "value": v})
                for epoch_idx, v in enumerate(train_curve):
                    curve_rows.append({"condition": cond["label"], "group": cond["group"],
                                        "facet": cond["facet"], "fold_id": fold_id,
                                        "tag": f"train_{m}", "epoch": epoch_idx, "value": v})

    return pd.DataFrame(raw_rows), pd.DataFrame(curve_rows), fold_curves


def summarize(raw_df):
    print("\n" + "=" * 78)
    print("Per-condition summary (mean +/- SEM across folds)")
    print("=" * 78)
    for (condition, metric), g in raw_df.groupby(["condition", "metric"]):
        values = g["value"].dropna().tolist()
        n = len(values)
        if n == 0:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else float("nan")
        sem = std / (n ** 0.5) if n > 1 else float("nan")
        print(f"{condition:45s} {metric:14s} mean={mean:.4f}  sem={sem:.4f}  (n={n})")
    print("\nSee 'Statistical comparison' below for the significance tests used to "
          "pick a best condition from validation metrics -- this descriptive table "
          "alone isn't a ranking decision, and neither is the significance test alone; "
          "read them together with the per-class guardrail figure.")


# ----------------------------------------------------------------------
# Statistical comparison -- choosing a best condition
# ----------------------------------------------------------------------
def check_collapse(raw_df, conditions, class_names, perclass_metrics, source_prefix="valid",
                    threshold=DEFAULT_COLLAPSE_THRESHOLD):
    """Flag conditions where the model collapsed onto one class: for each
    condition, take the mean pred_frac per class across folds and flag it
    if (max - min) across classes exceeds `threshold`. Checked on
    VALIDATION pred_frac by design (see run_stat_tests docstring) -- this
    never touches test data.

    Returns {label: {"collapsed": bool or None, "spread": float or None,
    "pred_frac": {class: mean_value}}}. "collapsed" is None (not False)
    when there's no pred_frac data at all for a condition (best-effort tag,
    only present for folds re-tested after patch_add_eval_metrics_and_dump.py)
    -- treat None as "unknown, go check the test-side per-class figure
    yourself" rather than "passed the guardrail".
    """
    out = {}
    if "pred_frac" not in perclass_metrics or not class_names:
        for c in conditions:
            out[c["label"]] = {"collapsed": None, "spread": None, "pred_frac": {}}
        return out

    for c in conditions:
        label = c["label"]
        means = {}
        for cname in class_names:
            col = f"{source_prefix}_pred_frac_{cname}"
            values = raw_df.loc[(raw_df["condition"] == label) & (raw_df["metric"] == col), "value"].dropna()
            if len(values):
                means[cname] = float(values.mean())
        if len(means) < len(class_names):
            out[label] = {"collapsed": None, "spread": None, "pred_frac": means}
        else:
            spread = max(means.values()) - min(means.values())
            out[label] = {"collapsed": spread > threshold, "spread": spread, "pred_frac": means}
    return out


def run_stat_tests(raw_df, conditions, primary_metrics, collapse_info, out_dir, alpha=0.05,
                    source_prefix="valid"):
    """Statistical comparison across conditions, used to pick a best one --
    see the module docstring's metrics section for the framework this
    implements (AUROC+acc co-primary, collapse as a guardrail/veto rather
    than a ranking metric, fold consistency as a tiebreaker). Runs on
    VALIDATION metrics (source_prefix="valid" by default), not test --
    picking among conditions using test performance is itself a form of
    test-set leakage (you'd be using held-out data to make a modeling
    decision), especially with only ~26 subjects and 10 conditions to
    compare -- test is for reporting the winner's final number, once,
    after the pick is already made from validation.

    Steps:
      1. Any condition the collapse guardrail flags (or couldn't check --
         see check_collapse) is excluded from being called "best", though
         its numbers are still printed for transparency.
      2. Friedman test (paired, non-parametric, across ALL non-excluded
         conditions at once) per primary metric -- an omnibus check for
         whether there's any real difference among conditions before
         drilling into pairwise comparisons.
      3. The top condition is the non-excluded one with the highest mean
         primary_metrics[0] (AUROC by default), tie-broken by
         primary_metrics[1] (acc), then by lowest SEM (fold consistency).
      4. Pairwise Wilcoxon signed-rank test (paired by fold_id, since every
         condition shares the SAME 5-fold subject split -- only
         preprocessing/tpt_strategy differs) between the top condition and
         every other non-excluded condition, per primary metric, with
         Holm-Bonferroni correction across that family of comparisons.

    IMPORTANT CAVEAT, printed with the results and worth restating here:
    with ~5 folds, the smallest two-sided p-value a paired Wilcoxon test
    can even produce is ~0.06 -- ABOVE the conventional 0.05 threshold no
    matter how large the true effect is. So "not significant" here very
    likely means "underpowered", not "no real difference". Use this test
    as a supplementary sanity check (does the apparent ranking hold up to
    ANY formal test, even a weak one?), not as a hard gate -- the
    descriptive means/SEM, the collapse guardrail, and fold-to-fold
    consistency remain the primary decision tools.

    Returns a pandas DataFrame (also saved to out_dir/condition_stats.csv)
    and prints a full report to stdout.
    """
    print("\n" + "=" * 78)
    print("Statistical comparison (validation metrics) -- choosing a best condition")
    print("=" * 78)

    if not HAVE_SCIPY_STATS:
        print("scipy.stats.wilcoxon / friedmanchisquare not available -- install scipy "
              "to get significance tests (`pip install scipy`). Skipping.")
        return pd.DataFrame()

    all_labels = [c["label"] for c in conditions]
    excluded = [lbl for lbl in all_labels if collapse_info.get(lbl, {}).get("collapsed") in (True, None)]
    included = [lbl for lbl in all_labels if lbl not in excluded]

    for lbl in excluded:
        reason = "class collapse detected" if collapse_info.get(lbl, {}).get("collapsed") else \
                 "no per-class validation data to check (best-effort tag) -- verify manually"
        print(f"  EXCLUDED from 'best' consideration: {lbl!r} ({reason})")
    if len(included) < 2:
        print("\nFewer than 2 non-excluded conditions -- nothing to compare statistically.")
        return pd.DataFrame()

    # --- build a fold x condition wide table per primary metric, restricted
    #     to folds present for EVERY included condition (Friedman/paired
    #     Wilcoxon both require complete, matched blocks) ---
    wide = {}
    common_folds = None
    for metric in primary_metrics:
        tag = f"{source_prefix}_{metric}"
        sub = raw_df[(raw_df["metric"] == tag) & (raw_df["condition"].isin(included))]
        pivot = sub.pivot_table(index="fold_id", columns="condition", values="value")
        pivot = pivot[included]  # keep a stable column order
        wide[metric] = pivot
        folds_here = set(pivot.dropna().index)
        common_folds = folds_here if common_folds is None else (common_folds & folds_here)

    common_folds = sorted(common_folds) if common_folds else []
    n_folds = len(common_folds)
    print(f"\nComparing {len(included)} non-excluded condition(s) on {n_folds} fold(s) "
          f"common to all of them: {common_folds}")
    if n_folds < len(set(raw_df["fold_id"])):
        print("  (some folds were dropped from this comparison because at least one "
              "condition is missing validation data for them)")
    if n_folds < 5:
        print(f"  WARNING: only {n_folds} paired folds -- treat any p-value here as "
              f"very rough; see the power caveat in run_stat_tests' docstring.")

    # --- omnibus Friedman test per metric ---
    friedman_results = {}
    if n_folds >= 3 and len(included) >= 3:
        for metric in primary_metrics:
            pivot = wide[metric].loc[common_folds]
            try:
                stat, p = friedmanchisquare(*[pivot[c].values for c in included])
                friedman_results[metric] = (stat, p)
                print(f"\nFriedman test on {source_prefix}_{metric} across all {len(included)} conditions: "
                      f"chi2={stat:.3f}, p={p:.4f}"
                      + ("  (below alpha={:.2f} -- some real difference among conditions is likely)".format(alpha)
                         if p < alpha else "  (above alpha -- underpowered at this n, not necessarily 'no difference')"))
            except Exception as e:
                print(f"\nFriedman test on {source_prefix}_{metric} failed ({e}) -- skipping omnibus test for this metric.")
    else:
        print(f"\nSkipping Friedman omnibus test (needs >=3 conditions and >=3 common folds; "
              f"have {len(included)} conditions, {n_folds} folds).")

    # --- pick the top condition: highest mean of primary_metrics[0], tie-broken
    #     by primary_metrics[1], then by lowest SEM of primary_metrics[0] ---
    stats_by_cond = {}
    for lbl in included:
        row = {"condition": lbl}
        for metric in primary_metrics:
            values = wide[metric][lbl].dropna().values
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_sem"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            row[f"{metric}_n"] = len(values)
        stats_by_cond[lbl] = row

    sort_key = lambda lbl: (
        -stats_by_cond[lbl].get(f"{primary_metrics[0]}_mean", float("-inf")),
        -stats_by_cond[lbl].get(f"{primary_metrics[1]}_mean", float("-inf")) if len(primary_metrics) > 1 else 0,
        stats_by_cond[lbl].get(f"{primary_metrics[0]}_sem", float("inf")),
    )
    ranked = sorted(included, key=sort_key)
    top = ranked[0]
    print(f"\nTop candidate by {primary_metrics[0]} (tie-break: {primary_metrics[1] if len(primary_metrics) > 1 else 'n/a'}, "
          f"then lowest SEM): {top!r}")
    if len(primary_metrics) > 1:
        top_by_second = sorted(included, key=lambda lbl: -stats_by_cond[lbl].get(f"{primary_metrics[1]}_mean", float("-inf")))[0]
        if top_by_second != top:
            print(f"  NOTE: {top_by_second!r} actually has the highest mean {primary_metrics[1]} instead -- "
                  f"the two co-primary metrics disagree on the winner. Look at both numbers yourself "
                  f"before trusting either one alone.")

    # --- pairwise Wilcoxon: top vs every other included condition, per metric ---
    pairwise_rows = []
    for metric in primary_metrics:
        pivot = wide[metric].loc[common_folds] if n_folds else wide[metric]
        others = [lbl for lbl in included if lbl != top]
        raw_pvals = []
        for other in others:
            a = pivot[top].values
            b = pivot[other].values
            if n_folds < 2 or np.allclose(a, b):
                raw_pvals.append(float("nan"))
                continue
            try:
                stat, p = wilcoxon(a, b)
            except ValueError as e:
                # e.g. all differences are zero, or too few non-zero diffs
                print(f"  Wilcoxon {top!r} vs {other!r} on {metric} could not be computed ({e})")
                stat, p = float("nan"), float("nan")
            raw_pvals.append(p)
        valid_mask = [not np.isnan(p) for p in raw_pvals]
        adjusted = np.full(len(raw_pvals), np.nan)
        if any(valid_mask):
            adjusted[valid_mask] = holm_bonferroni([p for p, ok in zip(raw_pvals, valid_mask) if ok])
        for other, p, p_adj in zip(others, raw_pvals, adjusted):
            pairwise_rows.append({"metric": metric, "condition_a": top, "condition_b": other,
                                   "wilcoxon_p": p, "holm_adjusted_p": p_adj})

    pairwise_df = pd.DataFrame(pairwise_rows)

    # --- assemble + print the final table ---
    summary_rows = []
    for lbl in ranked:
        row = dict(stats_by_cond[lbl])
        row["is_top"] = (lbl == top)
        row["collapsed"] = collapse_info.get(lbl, {}).get("collapsed")
        for metric in primary_metrics:
            match = pairwise_df[(pairwise_df["condition_b"] == lbl) & (pairwise_df["metric"] == metric)]
            row[f"{metric}_p_vs_top"] = float(match["wilcoxon_p"].iloc[0]) if len(match) else float("nan")
            row[f"{metric}_holm_p_vs_top"] = float(match["holm_adjusted_p"].iloc[0]) if len(match) else float("nan")
        summary_rows.append(row)
    for lbl in excluded:
        row = {"condition": lbl, "is_top": False, "collapsed": collapse_info.get(lbl, {}).get("collapsed")}
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    csv_path = os.path.join(out_dir, "condition_stats.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    print(f"\n{'condition':45s} " + " ".join(f"{m + '_mean':>14s}" for m in primary_metrics) +
          " " + " ".join(f"{m + '_holm_p':>12s}" for m in primary_metrics) + "  collapsed")
    for lbl in ranked:
        r = stats_by_cond[lbl]
        marker = " <-- top" if lbl == top else ""
        means_str = " ".join(f"{r.get(m + '_mean', float('nan')):14.4f}" for m in primary_metrics)
        pvals_str = " ".join(
            f"{'--':>12s}" if lbl == top else f"{next((row[f'{m}_holm_p_vs_top'] for row in summary_rows if row['condition'] == lbl), float('nan')):12.4f}"
            for m in primary_metrics)
        collapsed = collapse_info.get(lbl, {}).get("collapsed")
        collapsed_str = "yes" if collapsed else ("unknown" if collapsed is None else "no")
        print(f"{lbl:45s} {means_str} {pvals_str}  {collapsed_str}{marker}")
    for lbl in excluded:
        collapsed = collapse_info.get(lbl, {}).get("collapsed")
        collapsed_str = "yes" if collapsed else "unknown"
        print(f"{lbl:45s} {'excluded':>{14*len(primary_metrics)+1}s} {'':>{12*len(primary_metrics)+len(primary_metrics)-1}s}  {collapsed_str}")

    print("\nReminder: with this few folds, a non-significant p-value most likely means "
          "'underpowered', not 'no real difference' -- read the ranking + guardrail + "
          "SEM together, don't gate purely on p < 0.05.")

    return summary_df


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*"]


def plot_raincloud(raw_df, conditions, metrics, out_path_base, chance, axis_range,
                   source_label="Test", source_prefix="test", y_min=None):
    groups = list(dict.fromkeys(c["group"] for c in conditions))
    group_palette = dict(zip(groups, sns.color_palette("colorblind", n_colors=len(groups))))
    group_marker_counter = {g: 0 for g in groups}
    condition_style = {}
    for c in conditions:
        g = c["group"]
        marker = MARKERS[group_marker_counter[g] % len(MARKERS)]
        group_marker_counter[g] += 1
        condition_style[c["label"]] = {"color": group_palette[g], "marker": marker}

    facets = list(dict.fromkeys(c["facet"] for c in conditions))
    nrows = len(facets)
    ncols = len(metrics)
    max_conditions = max((sum(1 for c in conditions if c["facet"] == f) for f in facets), default=1)

    sns.set_theme(style="whitegrid", context="talk")
    # adapt figure width to number of conditions per facet so x-axis labels fit
    width_per_col = max(3, 0.9 * max_conditions + 1.0)
    fig_width = width_per_col * ncols + 0.6
    fig_height = 4.0 * nrows + 1.0 + 0.35 * max_conditions
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=False,
    )

    for fi, facet in enumerate(facets):
        facet_conditions = [c for c in conditions if c["facet"] == facet]
        order = [c["label"] for c in facet_conditions]
        n = len(order)

        for mi, metric in enumerate(metrics):
            ax = axes[fi, mi]
            # draw axes spines under plotted artists so points at the axis
            # limits (e.g. y==1.0) are not hidden by the border lines
            try:
                for spine in ax.spines.values():
                    spine.set_zorder(0)
                # put grids below artists as well
                ax.set_axisbelow(True)
                # ensure the axes patch (background/border) is drawn below artists
                ax.patch.set_zorder(-1)
            except Exception:
                pass
            sub = raw_df[raw_df["metric"] == metric]
            x_positions = np.arange(1, n + 1)

            for i, label in enumerate(order):
                style = condition_style[label]
                values = sub.loc[sub["condition"] == label, "value"].dropna().values
                if len(values) == 0:
                    continue

                bp = ax.boxplot(
                    [values], positions=[x_positions[i]], widths=0.6,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="white", edgecolor=style["color"], linewidth=1.8),
                    medianprops=dict(color=style["color"], linewidth=2.1),
                    whiskerprops=dict(color=style["color"], linewidth=1.5),
                    capprops=dict(color=style["color"], linewidth=1.5),
                )
                # make sure boxplot artists are above the spines and not clipped
                try:
                    for k, artists in bp.items():
                        if isinstance(artists, list):
                            for a in artists:
                                a.set_zorder(90)
                                try:
                                    a.set_clip_on(False)
                                except Exception:
                                    pass
                except Exception:
                    pass
                rng = np.random.default_rng(sum(ord(ch) for ch in label) % (2 ** 32))
                jitter = x_positions[i] + rng.normal(0.0, 0.07, size=len(values))
                ax.scatter(jitter, values, color=style["color"], marker=style["marker"],
                           s=70, alpha=0.9, edgecolor="white", linewidth=0.8,
                           zorder=95, clip_on=False)

                mean_v = float(np.mean(values))
                sem_v = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                text = f"{mean_v:.3f}" if np.isfinite(sem_v) else f"{mean_v:.3f}" # removed the ± {sem_v:.3f} from figure to save space 
                ax.text(x_positions[i], np.nanmax(values) + (0.01 if metric in axis_range else 0.01),
                        text, ha="center", va="bottom", fontsize=12,
                        color=style["color"], fontweight="bold")

            ax.set_xticks(x_positions)
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=12)
            ax.set_xlim(0.5, n + 0.5)
            ax.set_xlabel("Condition")

            metric_key = metric[len(source_prefix) + 1:] if metric.startswith(source_prefix + "_") else metric
            if metric_key in axis_range:
                lo, hi = axis_range[metric_key]
                # allow caller to override the lower bound across metrics
                if y_min is not None:
                    lo = y_min
                ax.set_ylim(lo, hi)
                lo_set, hi_set = lo, hi
            else:
                # no fixed axis range provided for this metric — allow a
                # caller-provided y_min to adjust the autoscaled lower bound
                if y_min is not None:
                    cur_hi = ax.get_ylim()[1]
                    ax.set_ylim(y_min, cur_hi)
                    lo_set, hi_set = y_min, cur_hi
                else:
                    lo_set, hi_set = ax.get_ylim()

            # place ticks every 0.1 including the starting lower bound
            try:
                ticks = np.arange(lo_set, hi_set + 1e-8, 0.1)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:.1f}" for t in ticks])
            except Exception:
                # fallback: do nothing if tick placement fails for any reason
                pass
            # if metric_key in chance:
            #     ax.axhline(chance[metric_key], color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
            #     ax.text(n + 0.15, chance[metric_key], "chance", va="bottom", ha="left",
            #             fontsize=10, color="black", alpha=0.75)

            ax.set_ylabel(metric_key)
            title = f"{source_label} {metric_key}" + (f" — {facet}" if facet else "")
            ax.set_title(title, fontweight="bold")
            # Move title higher above the figure
            ax.title.set_y(1.15)
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", visible=False)
            for tick, label in zip(ax.get_xticklabels(), order):
                tick.set_color(condition_style[label]["color"])
                tick.set_fontweight("bold")

    fig.suptitle(f"Condition comparison -- {source_label.lower()}-set performance across folds\n"
                 "(box = median/IQR, points = individual folds"
                 "color = group; separate panels for each tpt strategy)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path_base}.png / .pdf")


def plot_perclass(raw_df, conditions, class_names, perclass_metrics, out_path_base,
                   source_label="Test", source_prefix="test"):
    """Horizontal grouped bars, one row of subplots per facet, one column per
    per-class metric -- the class-collapse guardrail check (see module
    docstring). Skips gracefully (with a printed note) if a metric/facet has
    no data, since these tags are best-effort (only present for folds
    re-tested after patch_add_eval_metrics_and_dump.py). `source_prefix`
    selects "test" or "valid" -- metric names in raw_df are stored as
    "<source>_<perclass_metric>_<class>" (e.g. "valid_pred_frac_tennis")."""
    if not class_names or not perclass_metrics:
        return

    facets = list(dict.fromkeys(c["facet"] for c in conditions))
    nrows = len(facets)
    ncols = len(perclass_metrics)
    rows_per_facet = [sum(1 for c in conditions if c["facet"] == f) for f in facets]
    max_rows = max(rows_per_facet)
    class_palette = dict(zip(class_names, sns.color_palette("colorblind", n_colors=len(class_names))))
    chance_value = 1.0 / len(class_names)
    bar_h = 0.8 / len(class_names)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(8.5 * ncols, 1.35 * max_rows + 1.6 * nrows + 1.2),
                              squeeze=False)

    any_drawn = False
    for fi, facet in enumerate(facets):
        facet_conditions = [c for c in conditions if c["facet"] == facet]
        order = [c["label"] for c in facet_conditions]
        n = len(order)

        for mi, prefix in enumerate(perclass_metrics):
            ax = axes[fi][mi]
            for i, label in enumerate(order):
                pos = n - i
                for ci, cname in enumerate(class_names):
                    col = f"{source_prefix}_{prefix}_{cname}"
                    values = raw_df.loc[(raw_df["condition"] == label) & (raw_df["metric"] == col),
                                         "value"].dropna().values
                    if len(values) == 0:
                        continue
                    mean_v = float(np.mean(values))
                    sem_v = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                    y = pos + (ci - (len(class_names) - 1) / 2.0) * bar_h
                    ax.barh(y, mean_v, height=bar_h * 0.9,
                            xerr=sem_v if sem_v else None,
                            color=class_palette[cname], alpha=0.9,
                            error_kw=dict(ecolor=class_palette[cname], elinewidth=1.4, capsize=3))
                    any_drawn = True

            ax.set_yticks([n - i for i in range(n)])
            ax.set_yticklabels(order)
            ax.set_ylim(0.3, n + 1.0)
            # ax.axvline(chance_value, color="black", linestyle="--", linewidth=1.4, alpha=0.6, zorder=1)
            # ax.text(chance_value, n + 0.65, "chance", va="bottom", ha="center", fontsize=10, alpha=0.7)
            if prefix == "pred_frac":
                ax.set_xlim(0, 1)
            ax.set_xlabel(prefix)
            title = prefix.replace("_", " ").title() + (f" — {facet}" if facet else "")
            ax.set_title(title, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.grid(axis="y", visible=False)

    if not any_drawn:
        print(f"  (no {source_label.lower()} per-class data found -- skipping {out_path_base}; "
              f"these tags only exist for folds re-tested after patch_add_eval_metrics_and_dump.py)")
        plt.close(fig)
        return

    class_handles = [Line2D([0], [0], marker="s", linestyle="", markersize=14,
                             color=class_palette[c], label=c) for c in class_names]
    fig.legend(handles=class_handles, loc="upper right", fontsize=12, frameon=True,
               bbox_to_anchor=(0.995, 1.02), title="Class")
    fig.suptitle(f"{source_label} per-class breakdown across folds (mean ± SEM) — the class-collapse check",
                 fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path_base}.png / .pdf")


def plot_curves(fold_curves, conditions, metric, out_path_base):
    groups = list(dict.fromkeys(c["group"] for c in conditions))
    ncols = min(3, len(groups))
    nrows = int(np.ceil(len(groups) / ncols))

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows),
                              squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    any_drawn = False
    for gi, group in enumerate(groups):
        ax = axes_flat[gi]
        conds_in_group = [c for c in conditions if c["group"] == group]
        palette = sns.color_palette("colorblind", n_colors=max(len(conds_in_group), 1))

        for ci, cond in enumerate(conds_in_group):
            color = palette[ci]
            for mode, linestyle in (("valid", "-"), ("train", "--")):
                key = (cond["label"], f"{mode}_{metric}")
                curves = fold_curves.get(key)
                if not curves:
                    continue
                min_len = min(len(c) for c in curves)
                if min_len == 0:
                    continue
                stacked = np.stack([c[:min_len] for c in curves])
                mean_curve = np.nanmean(stacked, axis=0)
                if stacked.shape[0] > 1:
                    sem_curve = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                else:
                    sem_curve = np.zeros_like(mean_curve)
                epochs = np.arange(min_len)
                ax.plot(epochs, mean_curve, linestyle=linestyle, color=color, linewidth=2.4)
                ax.fill_between(epochs, mean_curve - sem_curve, mean_curve + sem_curve,
                                 color=color, alpha=0.15, linewidth=0)
                any_drawn = True

        color_handles = [Line2D([0], [0], color=palette[ci], lw=2.6, label=cond["label"])
                          for ci, cond in enumerate(conds_in_group)]
        if color_handles:
            ax.legend(handles=color_handles, fontsize=10, loc="best")
        ax.set_title(group, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

    for j in range(len(groups), len(axes_flat)):
        axes_flat[j].axis("off")

    if not any_drawn:
        print(f"  (no data to plot for curve metric '{metric}' -- skipping {out_path_base})")
        plt.close(fig)
        return

    style_handles = [Line2D([0], [0], color="gray", lw=2.6, linestyle="-", label="valid"),
                      Line2D([0], [0], color="gray", lw=2.6, linestyle="--", label="train")]
    fig.legend(handles=style_handles, loc="upper right", fontsize=12, frameon=True,
               bbox_to_anchor=(0.995, 1.03))
    fig.suptitle(f"Training dynamics — {metric} (mean ± SEM across folds; "
                 "dashed = train, solid = valid)", fontsize=15, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path_base}.png / .pdf")


# ----------------------------------------------------------------------
def aggregate_and_plot_confusions(conditions, out_dir, class_names, output_root):
    """Search each condition's fold dirs for `lightning_logs/version_*/test_predictions.pt`
    (written by models/lightning_model.py's _evaluate_metrics only when mode=='test',
    and only on runs made AFTER patch_add_eval_metrics_and_dump.py was applied -- see
    that patch's docstring), aggregate per-fold `preds`/`targets` and plot one
    confusion matrix per condition. Requires `torch` and
    `sklearn.metrics.confusion_matrix`; prints a clear diagnostic and returns
    (no exception) if these are unavailable or no prediction dumps are found --
    see the two most likely causes below, both of which are DATA-availability
    issues, not bugs in this function: the matching logic itself is correct.

    MOST LIKELY REASON THIS PRODUCES NOTHING: test_predictions.pt is only
    written for a fold that was *tested* (mode=='test') using a
    lightning_model.py that already had patch_add_eval_metrics_and_dump.py
    applied. A fold trained/tested before that patch existed has NO such file
    in ANY of its lightning_logs/version_* dirs, no matter how many times you
    rerun this script -- rerunning compare_conditions_v2.py never generates
    it, since this script only READS existing event/dump files, it never runs
    the model. Fix: re-test those folds with the patch applied, e.g.
      bash "Claude outputs/backfill_best_ckpt_test.sh" \\
        --project_base <condition's project name minus _fold-N> \\
        --dataset <dataset> --task <task> --tpt_strategy <strategy>
    which reloads each fold's saved checkpoint (no retraining) and writes a
    new lightning_logs/version_N containing test_predictions.pt. The
    diagnostics below print exactly which folds/versions were searched and
    whether the file turned up, so this is visible immediately instead of a
    silent, unexplained no-op.
    """
    try:
        import torch
        from sklearn.metrics import confusion_matrix
    except Exception as e:
        print(f"Cannot plot confusions: missing torch or sklearn.metrics.confusion_matrix ({e}). "
              f"Run this script with a Python that has both installed (your training env, "
              f"typically) if you want --plot_confusions to work.")
        return

    cms_by_condition = {}
    for cond in conditions:
        folds = resolve_folds(output_root, cond['glob'])
        all_true = []
        all_pred = []
        n_versions_total = 0
        n_versions_with_dump = 0
        for fold_id, project_dir in folds:
            vdirs = find_all_version_dirs(project_dir)
            n_versions_total += len(vdirs)
            # Use only the LATEST version dir that actually has the dump, not
            # every version that has one -- a fold can have multiple version_*
            # dirs (e.g. the original training run's auto-test, THEN a later
            # backfill re-test), and if more than one happens to contain
            # test_predictions.pt, summing all of them would double-count that
            # fold's test set and silently mix stale last-epoch predictions in
            # with the correct best-checkpoint ones.
            dump_path = None
            for v in reversed(vdirs):
                candidate = os.path.join(v, 'test_predictions.pt')
                if os.path.isfile(candidate):
                    dump_path = candidate
                    break
            if dump_path is None:
                continue
            n_versions_with_dump += 1
            try:
                d = torch.load(dump_path, map_location='cpu')
                preds = d.get('preds')
                targets = d.get('targets')
                if preds is None or targets is None:
                    print(f"  WARNING: {dump_path} exists but is missing 'preds'/'targets' keys -- skipping")
                    continue
                # convert tensors or lists to python lists
                if hasattr(preds, 'tolist'):
                    preds_list = preds.tolist()
                else:
                    preds_list = list(preds)
                if hasattr(targets, 'tolist'):
                    targets_list = targets.tolist()
                else:
                    targets_list = list(targets)
                all_pred.extend(preds_list)
                all_true.extend(targets_list)
            except Exception as e:
                print(f"  WARNING: failed to load {dump_path} ({e}) -- skipping")
                continue

        print(f"  '{cond['label']}': {len(folds)} fold dir(s), {n_versions_total} version dir(s) searched, "
              f"test_predictions.pt found for {n_versions_with_dump}/{len(folds)} fold(s)"
              + (f", {len(all_true)} total samples aggregated" if all_true else ""))

        if all_true:
            cm = confusion_matrix(all_true, all_pred, labels=list(range(len(class_names))))
            cms_by_condition[cond['label']] = cm

    if not cms_by_condition:
        print("\nNo per-fold prediction dumps found for ANY condition -- nothing to plot.")
        print("See this function's docstring: test_predictions.pt only exists for folds")
        print("tested (mode=='test') AFTER patch_add_eval_metrics_and_dump.py was applied.")
        print("If your folds predate that patch, re-test them first, e.g.:")
        print('  bash "Claude outputs/backfill_best_ckpt_test.sh" \\')
        print("    --project_base <project name minus _fold-N> --dataset <dataset> \\")
        print("    --task <task> --tpt_strategy <strategy>")
        return

    sns.set_theme(style='whitegrid', context='talk')
    nconds = len(cms_by_condition)
    fig, axes = plt.subplots(1, nconds, figsize=(6 * nconds, 6), squeeze=False)
    for i, (label, cm) in enumerate(cms_by_condition.items()):
        ax = axes[0, i]
        im = ax.imshow(cm, cmap='Blues')
        # ensure no matplotlib/seaborn grid overlays the matrix
        ax.grid(False)
        ax.set_axisbelow(False)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(label)
        cm_max = cm.max() if cm.size > 0 else 0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, int(cm[r, c]), ha='center', va='center',
                        color='white' if cm[r, c] > cm_max / 2 else 'black')
        fig.colorbar(im, ax=ax)
    outpath = os.path.join(out_dir, 'condition_confusion_matrices.png')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved aggregated confusion matrices to {outpath}")

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", action="append", required=True, dest="conditions",
                     help="Repeatable. 'label|project_name_glob[|group[|facet]]'. See module docstring.")
    ap.add_argument("--output_root", default="nas/output/neurostorm")
    ap.add_argument("--out_dir", default="figure_outputs/condition_sweep")
    ap.add_argument("--test_metrics", default="acc,AUROC")
    ap.add_argument("--val_metrics", default="acc,AUROC",
                     help="Comma-separated validation metrics to plot as a separate boxplot figure."
                     )
    ap.add_argument("--perclass_metrics", default="pred_frac,f1")
    ap.add_argument("--curve_metrics", default="loss,acc")
    ap.add_argument("--class_names", default="tennis,navigation,squeeze")
    ap.add_argument("--chance", default="acc=0.333,AUROC=0.5",
                     help="'metric=value,...' dashed reference line per metric in the raincloud plot.")
    ap.add_argument("--axis_range", default="acc=0:1,AUROC=0:1",
                     help="'metric=lo:hi,...' fixed x-axis range per metric in the raincloud plot.")
    ap.add_argument("--val_checkpoint_monitor", default="acc",
                     help="Which valid_<metric> main.py's ModelCheckpoint monitors (mode='max') -- "
                          "used to find each fold's best/checkpointed epoch for validation metrics. "
                          "Matches main.py's ModelCheckpoint(monitor='valid_acc', ...) default; change "
                          "only if you changed that in main.py.")
    ap.add_argument("--primary_metrics", default="AUROC,acc",
                     help="Comma-separated validation metrics used for the statistical comparison / "
                          "best-condition pick, in priority order (first = primary ranking key, "
                          "second = tie-break). Must be a subset of --val_metrics.")
    ap.add_argument("--stats_tpt_subset", choices=("all", "none"), default="all",
                     help="Run an additional statistical comparison restricted to conditions whose facet is 'none' (useful to compare only tpt_strategy=none variants).")
    ap.add_argument("--collapse_threshold", type=float, default=DEFAULT_COLLAPSE_THRESHOLD,
                     help="Flag a condition's class-collapse guardrail if (max-min) mean validation "
                          "pred_frac across classes exceeds this (default 0.5).")
    ap.add_argument("--alpha", type=float, default=0.05,
                     help="Significance threshold used only for labeling the Friedman omnibus result "
                          "in the printed report (default 0.05).")
    ap.add_argument("--skip_stats", action="store_true",
                     help="Skip the statistical comparison / best-condition selection entirely.")
    ap.add_argument("--plot_confusions", action="store_true",
                     help="Aggregate per-fold test_predictions.pt and plot per-condition confusion matrices")
    args = ap.parse_args()

    conditions = [parse_condition(c) for c in args.conditions]
    test_metrics = [m.strip() for m in args.test_metrics.split(",") if m.strip()]
    val_metrics = [m.strip() for m in args.val_metrics.split(",") if m.strip()]
    perclass_metrics = [m.strip() for m in args.perclass_metrics.split(",") if m.strip()]
    curve_metrics = [m.strip() for m in args.curve_metrics.split(",") if m.strip()]
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    chance = parse_kv_floats(args.chance, DEFAULT_CHANCE)
    axis_range = parse_axis_range(args.axis_range, DEFAULT_AXIS_RANGE)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Comparing {len(conditions)} condition(s):")
    for c in conditions:
        print(f"  - {c['label']!r}  (group={c['group']!r}, facet={c['facet']!r}, glob={c['glob']!r})")

    raw_df, curve_df, fold_curves = collect(conditions, args.output_root, test_metrics,
                                           val_metrics, curve_metrics,
                                           class_names, perclass_metrics,
                                           val_checkpoint_monitor=args.val_checkpoint_monitor)

    if raw_df.empty:
        raise SystemExit("No test metrics could be read for any condition -- nothing to report.")

    raw_csv_path = os.path.join(args.out_dir, "condition_raw.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"\nSaved {raw_csv_path} ({len(raw_df)} rows)")

    if not curve_df.empty:
        curve_csv_path = os.path.join(args.out_dir, "condition_curves.csv")
        curve_df.to_csv(curve_csv_path, index=False)
        print(f"Saved {curve_csv_path} ({len(curve_df)} rows)")

    summarize(raw_df)

    plot_raincloud(raw_df, conditions,
                   [f"test_{m}" for m in test_metrics],
                   os.path.join(args.out_dir, "condition_raincloud"),
                   chance, axis_range, source_label="Test", source_prefix="test", y_min=0.6)

    if val_metrics:
        plot_raincloud(raw_df, conditions,
                       [f"valid_{m}" for m in val_metrics],
                       os.path.join(args.out_dir, "condition_validation_raincloud"),
                       chance, axis_range, source_label="Validation", source_prefix="valid", y_min=0.6)

    plot_perclass(raw_df, conditions, class_names, perclass_metrics,
                  os.path.join(args.out_dir, "condition_perclass"),
                  source_label="Test", source_prefix="test")
    plot_perclass(raw_df, conditions, class_names, perclass_metrics,
                  os.path.join(args.out_dir, "condition_perclass_validation"),
                  source_label="Validation", source_prefix="valid")

    if not args.skip_stats:
        primary_metrics = [m.strip() for m in args.primary_metrics.split(",") if m.strip()]
        missing = [m for m in primary_metrics if m not in val_metrics]
        if missing:
            raise SystemExit(f"--primary_metrics {missing} not in --val_metrics {val_metrics} -- "
                              f"add them to --val_metrics or pick different --primary_metrics.")
        collapse_info = check_collapse(raw_df, conditions, class_names, perclass_metrics,
                                        source_prefix="valid", threshold=args.collapse_threshold)
        run_stat_tests(raw_df, conditions, primary_metrics, collapse_info, args.out_dir,
                        alpha=args.alpha, source_prefix="valid")

    # Optionally aggregate and plot confusion matrices per condition
    if args.plot_confusions:
        aggregate_and_plot_confusions(conditions, args.out_dir, class_names, args.output_root)

        # Optional: run the same stats restricted to only conditions with facet == 'none'
        if args.stats_tpt_subset == "none":
            import re
            subset_conditions = []
            pattern = re.compile(r"tpt[_-]strategy[=_-]?none")
            for c in conditions:
                facet = (c.get("facet") or "").strip().lower()
                label = (c.get("label") or "").strip().lower()
                glob_pat = (c.get("glob") or "").lower()
                if facet == "none" or pattern.search(glob_pat) or label == "none" or label.endswith(" none"):
                    subset_conditions.append(c)
            if len(subset_conditions) < 2:
                found = [c.get("label") for c in subset_conditions]
                print(f"Not enough 'none' tpt_strategy conditions to run tpt-none-only stats (need >=2). Found: {found}. Skipping.")
            else:
                labels = [c["label"] for c in subset_conditions]
                print(f"Running tpt-none-only stats on conditions: {labels}")
                raw_sub = raw_df[raw_df["condition"].isin(labels)]
                collapse_info_sub = check_collapse(raw_sub, subset_conditions, class_names, perclass_metrics,
                                                   source_prefix="valid", threshold=args.collapse_threshold)
                out_dir_sub = os.path.join(args.out_dir, "tpt_none_stats")
                os.makedirs(out_dir_sub, exist_ok=True)
                print("\nRunning statistical comparison restricted to tpt_strategy='none' conditions...")
                run_stat_tests(raw_sub, subset_conditions, primary_metrics, collapse_info_sub, out_dir_sub,
                               alpha=args.alpha, source_prefix="valid")

    for m in curve_metrics:
        plot_curves(fold_curves, conditions, m,
                    os.path.join(args.out_dir, f"condition_curves_{m}"))


if __name__ == "__main__":
    main()

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
  - test_AUROC and test_acc are both plotted as co-primary (raincloud plot)
    -- AUROC because it's threshold-independent and far more robust to a
    model that's gotten lucky on the decision boundary, acc because it's
    what the paper and this project have both been quoting informally.
    Trust a condition only if BOTH look good, not just one.
  - test_pred_frac_<class> and test_f1_<class> are plotted separately
    (per-class figure) as a GUARDRAIL, not a ranking metric: class collapse
    (the model predicting one class for nearly everything) has been the
    single most recurring failure mode in this project, and it can hide
    behind an OK-looking aggregate accuracy. Check this before trusting a
    "best" condition, don't rank by it.
  - Fold-to-fold consistency (the SEM annotations / point spread already in
    the raincloud plot) is a real tiebreaker: with only ~26 subjects, a
    condition whose mean is carried by one great fold and one terrible fold
    is a shakier bet than a slightly-lower but more consistent one.
  - Train/valid loss and accuracy curves are diagnostic (did training
    converge smoothly? overfit? diverge?), not part of the "best condition"
    decision -- that decision is about held-out test performance only.

WHAT IT PRODUCES
  1. condition_raw.csv        -- tidy long-format per-fold test metrics
                                  (condition, group, facet, fold_id, metric,
                                  value). Source of truth for any figure you
                                  want later that this script doesn't make.
  2. Console summary           -- mean +/- SEM per condition per test metric.
                                  NOTE: no significance testing is run here
                                  (by request) -- your conditions aren't
                                  necessarily a clean factorial design, so a
                                  generic "compare everything" script would
                                  either run a fishing-expedition of pairwise
                                  tests or have to guess your intended
                                  contrasts. `evaluate_cv_thorough.py
                                  --project_name_glob ... --compare_project_name_glob ...`
                                  already does a proper paired ttest_rel for
                                  any two globs with the same fold count.
  3. condition_raincloud.png/.pdf
                               -- one row of subplots per facet, one column
                                  per test metric (default: acc, AUROC):
                                  horizontal half-violin + boxplot +
                                  individual fold points per condition,
                                  colored by group (marker shape cycles
                                  within a group), a dashed "chance"
                                  reference line, and each row's mean +/- SEM
                                  printed next to it.
  4. condition_perclass.png/.pdf
                               -- one row of subplots per facet, one column
                                  per per-class metric (default: pred_frac,
                                  f1): horizontal grouped bars, one bar per
                                  class, mean +/- SEM across folds, with a
                                  dashed chance line at 1/num_classes. This
                                  is the class-collapse check -- a condition
                                  with one class's pred_frac far from the
                                  others (even if aggregate acc/AUROC look
                                  fine) is not actually the best model.
                                  Best-effort: these tags only exist for
                                  folds re-tested after
                                  patch_add_eval_metrics_and_dump.py: a
                                  facet/metric with no data is skipped with
                                  a printed note, not a crash.
  5. condition_curves_loss.png/.pdf
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

FOLD_RE = re.compile(r"fold-(\d+)")

DEFAULT_CHANCE = {"acc": 1.0 / 3.0, "AUROC": 0.5}
DEFAULT_AXIS_RANGE = {"acc": (0.0, 1.0), "AUROC": (0.0, 1.0)}


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
def collect(conditions, output_root, test_metrics, curve_metrics, class_names, perclass_metrics):
    raw_rows = []          # per-fold test metrics -> CSV / raincloud / per-class figure
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
                                  "metric": m, "value": v})

            for prefix in perclass_metrics:
                for cname in class_names:
                    tag = f"test_{prefix}_{cname}"
                    v = final_value(test_scalars.get(tag, []))
                    if v is not None:
                        raw_rows.append({"condition": cond["label"], "group": cond["group"],
                                          "facet": cond["facet"], "fold_id": fold_id,
                                          "metric": f"{prefix}_{cname}", "value": v})

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
    print("\nNo significance testing is run by this script -- see the module "
          "docstring for why, and use evaluate_cv_thorough.py's --compare_* "
          "flags for a specific paired comparison once you know which pair "
          "matters.")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*"]


def half_violin(ax, data, pos, color, width=0.38, alpha=0.55):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if not HAVE_SCIPY or len(data) < 3 or np.allclose(data, data[0]):
        return
    kde = gaussian_kde(data)
    lo, hi = data.min(), data.max()
    pad = (hi - lo) * 0.3 if hi > lo else max(abs(lo) * 0.05, 0.01)
    ys = np.linspace(lo - pad, hi + pad, 200)
    dens = kde(ys)
    dens = dens / dens.max() * width
    ax.fill_between(ys, pos, pos + dens, color=color, alpha=alpha, linewidth=0, zorder=2)


def plot_raincloud(raw_df, conditions, metrics, out_path_base, chance, axis_range, y_min=None):
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
    rows_per_facet = [sum(1 for c in conditions if c["facet"] == f) for f in facets]
    max_rows = max(rows_per_facet)

    sns.set_theme(style="whitegrid", context="talk")
    # adapt figure width to number of conditions per facet so x-axis labels fit
    max_conditions = max(rows_per_facet) if rows_per_facet else 1
    width_per_col = max(6.0, 0.9 * max_conditions + 1.0)
    fig_width = width_per_col * ncols
    fig_height = 1.35 * max_rows + 1.6 * nrows + 1.2
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fig_width, fig_height),
                              squeeze=False)

    for fi, facet in enumerate(facets):
        facet_conditions = [c for c in conditions if c["facet"] == facet]
        order = [c["label"] for c in facet_conditions]
        n = len(order)

        for mi, metric in enumerate(metrics):
            ax = axes[fi][mi]
            sub = raw_df[raw_df["metric"] == metric]

            for i, label in enumerate(order):
                pos = n - i
                style = condition_style[label]
                values = sub.loc[sub["condition"] == label, "value"].dropna().values
                if len(values) == 0:
                    continue

                half_violin(ax, values, pos, style["color"])

                ax.boxplot([values], positions=[pos - 0.02], vert=False, widths=0.14,
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="white", edgecolor=style["color"], linewidth=1.6),
                           medianprops=dict(color=style["color"], linewidth=2),
                           whiskerprops=dict(color=style["color"], linewidth=1.4),
                           capprops=dict(color=style["color"], linewidth=1.4), zorder=3)

                rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))
                jitter = pos - 0.16 - rng.uniform(0, 0.10, size=len(values))
                ax.scatter(values, jitter, color=style["color"], marker=style["marker"],
                           s=90, alpha=0.9, edgecolor="white", linewidth=0.8, zorder=4)

                mean_v = float(np.mean(values))
                sem_v = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
                txt = f"{mean_v:.3f}" if np.isnan(sem_v) else f"{mean_v:.3f} ± {sem_v:.3f}"
                ax.text(0.985, pos / (n + 1.0), txt, transform=ax.transAxes,
                        ha="right", va="center", fontsize=12, color=style["color"], fontweight="bold")

            ax.set_yticks([n - i for i in range(n)])
            ax.set_yticklabels(order)
            for tick, label in zip(ax.get_yticklabels(), order):
                tick.set_color(condition_style[label]["color"])
            ax.set_ylim(0.3, n + 1.0)

            if metric in axis_range:
                lo, hi = axis_range[metric]
                # allow caller to override the lower bound across metrics
                if y_min is not None:
                    lo = y_min
                ax.set_xlim(lo, hi)
                lo_set, hi_set = lo, hi
            else:
                # no fixed axis range provided for this metric — allow a
                # caller-provided y_min to adjust the autoscaled lower bound
                if y_min is not None:
                    cur_hi = ax.get_xlim()[1]
                    ax.set_xlim(y_min, cur_hi)
                    lo_set, hi_set = y_min, cur_hi
                else:
                    lo_set, hi_set = ax.get_xlim()

            # place ticks every 0.1 including the starting lower bound
            try:
                ticks = np.arange(lo_set, hi_set + 1e-8, 0.1)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:.1f}" for t in ticks])
            except Exception:
                pass

            if metric in chance:
                ax.axvline(chance[metric], color="black", linestyle="--", linewidth=1.6,
                           alpha=0.7, zorder=1)
                ax.text(chance[metric], n + 0.65, "chance", rotation=0, va="bottom", ha="center",
                        fontsize=11, color="black", alpha=0.75)

            ax.set_xlabel(metric)
            title = f"Test {metric}" + (f" — {facet}" if facet else "")
            ax.set_title(title, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.grid(axis="y", visible=False)

    fig.suptitle("Condition comparison -- test-set performance across folds\n"
                  "(color = group, marker shape cycles within a group; box = median/IQR, "
                  "points = individual folds, shaded area = kernel density, "
                  "text = mean ± SEM)",
                  fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path_base}.png / .pdf")


def plot_perclass(raw_df, conditions, class_names, perclass_metrics, out_path_base):
    """Horizontal grouped bars, one row of subplots per facet, one column per
    per-class metric -- the class-collapse guardrail check (see module
    docstring). Skips gracefully (with a printed note) if a metric/facet has
    no data, since these tags are best-effort (only present for folds
    re-tested after patch_add_eval_metrics_and_dump.py)."""
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
                    col = f"{prefix}_{cname}"
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
            ax.axvline(chance_value, color="black", linestyle="--", linewidth=1.4, alpha=0.6, zorder=1)
            ax.text(chance_value, n + 0.65, "chance", va="bottom", ha="center", fontsize=10, alpha=0.7)
            if prefix == "pred_frac":
                ax.set_xlim(0, 1)
            ax.set_xlabel(prefix)
            title = prefix.replace("_", " ").title() + (f" — {facet}" if facet else "")
            ax.set_title(title, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.grid(axis="y", visible=False)

    if not any_drawn:
        print(f"  (no per-class data found -- skipping {out_path_base}; these tags only exist "
              f"for folds re-tested after patch_add_eval_metrics_and_dump.py)")
        plt.close(fig)
        return

    class_handles = [Line2D([0], [0], marker="s", linestyle="", markersize=14,
                             color=class_palette[c], label=c) for c in class_names]
    fig.legend(handles=class_handles, loc="upper right", fontsize=12, frameon=True,
               bbox_to_anchor=(0.995, 1.02), title="Class")
    fig.suptitle("Per-class breakdown across folds (mean ± SEM) — the class-collapse check",
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
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", action="append", required=True, dest="conditions",
                     help="Repeatable. 'label|project_name_glob[|group[|facet]]'. See module docstring.")
    ap.add_argument("--output_root", default="nas/output/neurostorm")
    ap.add_argument("--out_dir", default="Claude outputs/condition_sweep")
    ap.add_argument("--test_metrics", default="acc,AUROC")
    ap.add_argument("--perclass_metrics", default="pred_frac,f1")
    ap.add_argument("--curve_metrics", default="loss,acc")
    ap.add_argument("--class_names", default="tennis,navigation,squeeze")
    ap.add_argument("--chance", default="acc=0.333,AUROC=0.5",
                     help="'metric=value,...' dashed reference line per metric in the raincloud plot.")
    ap.add_argument("--axis_range", default="acc=0:1,AUROC=0:1",
                     help="'metric=lo:hi,...' fixed x-axis range per metric in the raincloud plot.")
    args = ap.parse_args()

    conditions = [parse_condition(c) for c in args.conditions]
    test_metrics = [m.strip() for m in args.test_metrics.split(",") if m.strip()]
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
                                             curve_metrics, class_names, perclass_metrics)

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

    plot_raincloud(raw_df, conditions, test_metrics,
                   os.path.join(args.out_dir, "condition_raincloud"), chance, axis_range)

    plot_perclass(raw_df, conditions, class_names, perclass_metrics,
                  os.path.join(args.out_dir, "condition_perclass"))

    for m in curve_metrics:
        plot_curves(fold_curves, conditions, m,
                    os.path.join(args.out_dir, f"condition_curves_{m}"))


if __name__ == "__main__":
    main()

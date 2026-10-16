"""
scripts/evaluate_cv_thorough.py

A thorough, Wang-et-al.-2026 (NeuroSTORM paper)-style evaluation on top of
what aggregate_cv_results.py already reports, for a 5-fold CV sweep of
run_experiment.sh --mode finetune/test runs.

What the paper does, and how this maps onto your 5-fold Michigan setup:

  - "Variances were estimated from five technical replicates."
    -> Your 5 folds ARE your five replicates. This script reports mean,
       sample std, AND standard error of the mean (SEM = std/sqrt(n)) --
       the paper's plots use +/- s.e.m. error bars, not std, so SEM is the
       directly comparable number if you're building a similar figure.

  - "Pairwise significance markers were computed using a two-sided paired
    t-test (ttest_rel) without multiple-comparison correction."
    -> Use --compare_project_names/--compare_project_name_glob to hand this
       script a SECOND condition (e.g. the sex-specific checkpoint, or the
       fMRIPrep-preprocessed run) with the same 5 folds in the same
       dataset_split_num order, and it runs scipy.stats.ttest_rel fold-by-fold
       exactly the way the paper does, for test_AUROC and test_acc (and any
       per-class metrics both conditions have).

  - Per-class precision/recall/F1 (this project's analogue of the paper's
    per-task/per-class accuracy breakdowns): read from test_precision_<class>,
    test_recall_<class>, test_f1_<class> scalars IF they exist (added by
    patch_add_eval_metrics_and_dump.py). If a fold's event file predates that
    patch, this script says so and skips those tags for that fold rather than
    crashing -- see the module docstring in patch_add_eval_metrics_and_dump.py
    for how to cheaply backfill them via --test_only --test_ckpt_path
    (no retraining needed).

  - Per-class prediction fraction (test_pred_frac_<class>): flags class
    collapse, a recurring failure mode in this project -- reported alongside
    everything else, same as before.

Usage (single condition, no comparison):
    python scripts/evaluate_cv_thorough.py \
      --project_name_glob "michigan_xcpd_fi_neurostorm_task6_imagery_classification_fold*_general_ckpt_tpt_none"

Usage (with a paired-t-test comparison against a second condition, e.g. the
sex-specific checkpoint run on the same 5 folds):
    python scripts/evaluate_cv_thorough.py \
      --project_name_glob "michigan_xcpd_fi_neurostorm_task6_imagery_classification_fold*_general_ckpt_tpt_none" \
      --compare_project_name_glob "michigan_xcpd_fi_neurostorm_task6_imagery_classification_fold*_sex_ckpt_tpt_none"

IMPORTANT: --compare requires the SAME NUMBER of folds in the SAME
dataset_split_num order in both globs (sorted alphabetically here, so name
your project_names so that sorting puts fold1..fold5 in order, as this
project's naming convention already does) -- ttest_rel is a PAIRED test and
pairs by position, not by matching values.
"""
import argparse
import glob
import os
import statistics
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("This script needs the `tensorboard` package: pip install tensorboard", file=sys.stderr)
    raise

DEFAULT_CLASS_NAMES = ["tennis", "navigation", "squeeze"]


def find_event_file(project_dir):
    pattern = os.path.join(project_dir, "lightning_logs", "version_*")
    version_dirs = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_", 1)[-1]))
    if not version_dirs:
        raise FileNotFoundError(f"No lightning_logs/version_* found under {project_dir}")
    latest = version_dirs[-1]
    events = glob.glob(os.path.join(latest, "events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No events.out.tfevents.* found under {latest}")
    if len(version_dirs) > 1:
        print(f"  note: {len(version_dirs)} version_* dirs found under {project_dir}, using latest ({latest})")
    return events[0]


def scalar_series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def final_scalar(ea, tag):
    series = scalar_series(ea, tag)
    return series[-1][1] if series else None


def fold_metrics(project_dir, class_names):
    event_path = find_event_file(project_dir)
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()

    out = {}
    for tag in ("test_AUROC", "test_acc"):
        v = final_scalar(ea, tag)
        if v is not None:
            out[tag] = v
        else:
            print(f"  WARNING: '{tag}' not found in {event_path}")

    for cname in class_names:
        for prefix in ("test_pred_frac", "test_precision", "test_recall", "test_f1"):
            tag = f"{prefix}_{cname}"
            v = final_scalar(ea, tag)
            if v is not None:
                out[tag] = v
            # silent skip if missing -- expected for precision/recall/f1 on
            # folds run before patch_add_eval_metrics_and_dump.py

    return out


def resolve_project_names(project_names, project_name_glob, output_root, label):
    if project_names:
        names = project_names
    elif project_name_glob:
        matches = sorted(glob.glob(os.path.join(output_root, project_name_glob)))
        names = [os.path.basename(m) for m in matches]
        if not names:
            raise SystemExit(f"No directories under {output_root} matched glob {project_name_glob!r} ({label})")
    else:
        return None
    return names


def summarize(tag, values):
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else float("nan")
    sem = std / (n ** 0.5) if n > 1 else float("nan")
    print(f"{tag:28s} mean={mean:.4f}  std={std:.4f}  sem={sem:.4f}  (n={n})  values={['%.4f' % v for v in values]}")
    return mean, std, sem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project_names", nargs="+")
    ap.add_argument("--project_name_glob")
    ap.add_argument("--compare_project_names", nargs="+")
    ap.add_argument("--compare_project_name_glob")
    ap.add_argument("--output_root", default="nas/output/neurostorm")
    ap.add_argument("--class_names", nargs="+", default=DEFAULT_CLASS_NAMES)
    args = ap.parse_args()

    names_a = resolve_project_names(args.project_names, args.project_name_glob, args.output_root, "condition A")
    if names_a is None:
        raise SystemExit("Pass either --project_names or --project_name_glob for condition A")

    print(f"Condition A: {len(names_a)} fold(s)")
    metrics_a = []
    for name in names_a:
        print(f"\n[A: {name}]")
        try:
            m = fold_metrics(os.path.join(args.output_root, name), args.class_names)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        metrics_a.append(m)
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    if not metrics_a:
        raise SystemExit("No folds could be read for condition A -- nothing to report.")

    print("\n" + "=" * 70)
    print(f"Condition A aggregate across {len(metrics_a)} fold(s)  (mean / std / SEM)")
    print("=" * 70)
    all_tags = sorted({k for m in metrics_a for k in m})
    summary_a = {}
    for tag in all_tags:
        values = [m[tag] for m in metrics_a if tag in m]
        if len(values) < len(metrics_a):
            print(f"{tag:28s} only present in {len(values)}/{len(metrics_a)} folds -- "
                  f"skipping aggregate (likely pre-patch folds mixed with post-patch folds)")
            continue
        summary_a[tag] = summarize(tag, values)

    # ------------------------------------------------------------------
    # Optional: paired comparison against a second condition, matching
    # Wang et al. 2026's "two-sided paired t-test (ttest_rel) without
    # multiple-comparison correction" methodology.
    # ------------------------------------------------------------------
    names_b = resolve_project_names(args.compare_project_names, args.compare_project_name_glob,
                                     args.output_root, "condition B")
    if names_b is None:
        return

    if len(names_b) != len(names_a):
        raise SystemExit(
            f"--compare requires the same number of folds as condition A "
            f"(A has {len(names_a)}, B has {len(names_b)}) -- ttest_rel is a paired test."
        )

    try:
        from scipy.stats import ttest_rel
    except ImportError:
        print("\nscipy not available (pip install scipy) -- skipping paired t-test, "
              "but here are condition B's raw numbers for a manual comparison:")
        ttest_rel = None

    print(f"\nCondition B: {len(names_b)} fold(s)")
    metrics_b = []
    for name in names_b:
        print(f"\n[B: {name}]")
        try:
            m = fold_metrics(os.path.join(args.output_root, name), args.class_names)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        metrics_b.append(m)
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    if len(metrics_b) != len(metrics_a):
        raise SystemExit(
            f"Condition B only yielded {len(metrics_b)} readable fold(s) vs. condition A's "
            f"{len(metrics_a)} -- can't pair them 1:1 for ttest_rel. Fix the missing fold(s) above."
        )

    print("\n" + "=" * 70)
    print("Paired comparison: A vs. B (matched fold-by-fold, in the order listed above)")
    print("=" * 70)
    shared_tags = sorted(set(all_tags) & {k for m in metrics_b for k in m})
    for tag in shared_tags:
        a_vals = [m.get(tag) for m in metrics_a]
        b_vals = [m.get(tag) for m in metrics_b]
        if any(v is None for v in a_vals) or any(v is None for v in b_vals):
            print(f"{tag:28s} not present in every fold of both conditions -- skipping")
            continue
        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        mean_diff = statistics.mean(diffs)
        sem_diff = (statistics.stdev(diffs) / (len(diffs) ** 0.5)) if len(diffs) > 1 else float("nan")
        line = f"{tag:28s} A-B mean_diff={mean_diff:+.4f}  sem_diff={sem_diff:.4f}"
        if ttest_rel is not None:
            t_stat, p_value = ttest_rel(a_vals, b_vals)
            line += f"  t={t_stat:.3f}  p={p_value:.4g}"
        print(line)

    print("\nNote: no multiple-comparison correction applied across the tags above, "
          "matching the paper's own stated methodology -- treat multiple p-values here "
          "with the same caution the paper's own reporting implies.")


if __name__ == "__main__":
    main()

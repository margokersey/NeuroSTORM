"""
scripts/aggregate_cv_results.py

Aggregate test metrics across the 5 folds of a cross-validation sweep (i.e.
5 separate run_experiment.sh invocations differing only by
--dataset_split_num 1..5, each with its own --project_name).

Reads each fold's TensorBoard event file directly (no need to copy numbers
out of the terminal by hand) and reports:
  - each fold's final test_AUROC / test_acc (the value trainer.test() logged
    at the end of that run -- i.e. the LAST-epoch model, see the earlier
    discussion of trainer.test(model, ...) with ckpt_path=None)
  - each fold's BEST validation epoch for the same metrics (max valid_AUROC,
    min valid_loss and the epoch each occurred at) -- useful context, since
    the final-epoch test score is not necessarily the run's best checkpoint
  - mean +/- sample std (ddof=1) across folds for every requested metric

Usage (run from the repo root, needs `pip install tensorboard` if not
already present -- it's a light dependency, no torch/GPU needed):

    python scripts/aggregate_cv_results.py \\
      --project_names \\
        michigan_gender_fi_neurostorm_task7_sex_train1.0_fold1 \\
        michigan_gender_fi_neurostorm_task7_sex_train1.0_fold2 \\
        michigan_gender_fi_neurostorm_task7_sex_train1.0_fold3 \\
        michigan_gender_fi_neurostorm_task7_sex_train1.0_fold4 \\
        michigan_gender_fi_neurostorm_task7_sex_train1.0_fold5

or, if your 5 folds' --project_name values all share a common prefix:

    python scripts/aggregate_cv_results.py \\
      --project_name_glob "michigan_gender_fi_neurostorm_task7_sex_train1.0_fold*"

By default looks for logs under nas/output/neurostorm/<project_name>/lightning_logs/
(matching the "Missing logger folder: nas/output/neurostorm/.../lightning_logs"
path seen in run_experiment.sh's own output) -- override with --output_root
if your logs land somewhere else.

IMPORTANT: this reads whatever project_name directories you point it at. If
you haven't been giving each fold a distinct --project_name (e.g. reusing
the same name for all 5 dataset_split_num values), each run will have
overwritten/appended to the same TensorBoard log dir and this script cannot
tell the folds apart -- use a distinct --project_name per fold going forward.
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


def find_event_file(project_dir):
    """Return the event file from the highest-numbered lightning_logs/version_N/."""
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


def load_scalars(event_path):
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def scalar_series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def fold_summary(project_dir, test_metrics, val_metrics):
    event_path = find_event_file(project_dir)
    ea = load_scalars(event_path)

    result = {"project_dir": project_dir, "test": {}, "val_best": {}}

    for tag in test_metrics:
        series = scalar_series(ea, tag)
        if not series:
            print(f"  WARNING: tag '{tag}' not found in {event_path}")
            continue
        # trainer.test() logs a single point (or one per test call); take the last.
        result["test"][tag] = series[-1][1]

    for tag, mode in val_metrics.items():
        series = scalar_series(ea, tag)
        if not series:
            print(f"  WARNING: tag '{tag}' not found in {event_path}")
            continue
        if mode == "max":
            step, value = max(series, key=lambda sv: sv[1])
        else:
            step, value = min(series, key=lambda sv: sv[1])
        result["val_best"][tag] = (value, step)

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project_names", nargs="+", help="explicit list of --project_name values, one per fold")
    ap.add_argument("--project_name_glob", help="glob pattern matched against directory names under --output_root")
    ap.add_argument("--output_root", default="nas/output/neurostorm",
                     help="parent directory containing <project_name>/lightning_logs/ (default: nas/output/neurostorm)")
    ap.add_argument("--test_metrics", nargs="+", default=["test_AUROC", "test_acc"],
                     help="scalar tags to report the final test-time value for")
    ap.add_argument("--val_auroc_tag", default="valid_AUROC", help="val AUROC scalar tag (reports best/max epoch)")
    ap.add_argument("--val_loss_tag", default="valid_loss", help="val loss scalar tag (reports best/min epoch)")
    args = ap.parse_args()

    if args.project_names:
        project_names = args.project_names
    elif args.project_name_glob:
        matches = sorted(glob.glob(os.path.join(args.output_root, args.project_name_glob)))
        project_names = [os.path.basename(m) for m in matches]
        if not project_names:
            raise SystemExit(f"No directories under {args.output_root} matched glob {args.project_name_glob!r}")
    else:
        raise SystemExit("Pass either --project_names (explicit list) or --project_name_glob")

    val_metrics = {args.val_auroc_tag: "max", args.val_loss_tag: "min"}

    fold_results = []
    print(f"Found {len(project_names)} fold(s):")
    for name in project_names:
        project_dir = os.path.join(args.output_root, name)
        print(f"\n[{name}]")
        try:
            res = fold_summary(project_dir, args.test_metrics, val_metrics)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        fold_results.append(res)
        for tag in args.test_metrics:
            if tag in res["test"]:
                print(f"  final test {tag}: {res['test'][tag]:.4f}")
        for tag, mode in val_metrics.items():
            if tag in res["val_best"]:
                value, step = res["val_best"][tag]
                print(f"  best val {tag} ({'max' if mode == 'max' else 'min'}): {value:.4f} at step {step}")

    if not fold_results:
        raise SystemExit("\nNo folds could be read -- nothing to aggregate.")

    print("\n" + "=" * 60)
    print(f"Aggregate across {len(fold_results)} fold(s)")
    print("=" * 60)
    for tag in args.test_metrics:
        values = [r["test"][tag] for r in fold_results if tag in r["test"]]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        print(f"final test {tag}: mean={mean:.4f}  std={std:.4f}  (n={len(values)})  values={['%.4f' % v for v in values]}")

    for tag in val_metrics:
        values = [r["val_best"][tag][0] for r in fold_results if tag in r["val_best"]]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        print(f"best val {tag}:   mean={mean:.4f}  std={std:.4f}  (n={len(values)})  values={['%.4f' % v for v in values]}")


if __name__ == "__main__":
    main()

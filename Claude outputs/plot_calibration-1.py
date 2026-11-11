"""
scripts/plot_calibration.py

Confidence-calibration plot in the style of Wang et al. 2026 (NeuroSTORM
paper) Fig. 6b: "The violin plots display the distribution of predicted
probabilities assigned to the true class for each participant; correct
predictions cluster at high-confidence values, while errors occur at
markedly lower probabilities, indicating well-calibrated predictions."

Requires patch_add_eval_metrics_and_dump.py to have been applied AND the
folds re-run (or re-tested via --test_only --test_ckpt_path) AFTER that
patch, since it reads the per-sample dump
<project_dir>/lightning_logs/version_*/test_predictions.pt that patch adds.
Folds tested before the patch won't have this file -- this script skips
them with a warning rather than crashing, so partial coverage still plots
what's available.

Usage:
    python scripts/plot_calibration.py \\
      --project_name_glob "michigan_xcpd_fi_neurostorm_task6_imagery_classification_fold*_general_ckpt_tpt_none" \\
      --out calibration_general_ckpt.png

Pools every sample (every block, from every subject, from every fold) into
one plot -- with only ~6 test subjects per fold this is far more informative
than a single fold's handful of blocks. Prints summary numbers (mean
confidence when correct vs. incorrect, sample counts) to the console too.
"""
import argparse
import glob
import os
import sys

import torch


def find_dump_files(project_dir):
    pattern = os.path.join(project_dir, "lightning_logs", "version_*")
    version_dirs = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_", 1)[-1]))
    if not version_dirs:
        return None
    latest = version_dirs[-1]
    dump_path = os.path.join(latest, "test_predictions.pt")
    return dump_path if os.path.isfile(dump_path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project_names", nargs="+")
    ap.add_argument("--project_name_glob")
    ap.add_argument("--output_root", default="nas/output/neurostorm")
    ap.add_argument("--out", default="calibration.png")
    args = ap.parse_args()

    if args.project_names:
        names = args.project_names
    elif args.project_name_glob:
        matches = sorted(glob.glob(os.path.join(args.output_root, args.project_name_glob)))
        names = [os.path.basename(m) for m in matches]
        if not names:
            raise SystemExit(f"No directories under {args.output_root} matched glob {args.project_name_glob!r}")
    else:
        raise SystemExit("Pass either --project_names or --project_name_glob")

    all_probs, all_preds, all_targets = [], [], []
    class_names = None
    n_missing = 0
    for name in names:
        dump_path = find_dump_files(os.path.join(args.output_root, name))
        if dump_path is None:
            print(f"  SKIPPED (no test_predictions.pt -- re-run this fold's test after applying "
                  f"patch_add_eval_metrics_and_dump.py): {name}")
            n_missing += 1
            continue
        d = torch.load(dump_path, map_location="cpu", weights_only=False)
        all_probs.append(d["probs"])
        all_preds.append(d["preds"])
        all_targets.append(d["targets"])
        class_names = d.get("class_names", class_names)
        print(f"  loaded {d['probs'].shape[0]} samples from {dump_path}")

    if not all_probs:
        raise SystemExit("No test_predictions.pt dumps found for any requested fold -- nothing to plot.")
    if n_missing:
        print(f"\nNote: {n_missing}/{len(names)} fold(s) skipped (missing dump) -- "
              f"plot below reflects only the {len(all_probs)} fold(s) with a dump.")

    probs = torch.cat(all_probs, dim=0)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    true_class_prob = probs[torch.arange(probs.shape[0]), targets]
    correct_mask = (preds == targets)

    correct_conf = true_class_prob[correct_mask].numpy()
    incorrect_conf = true_class_prob[~correct_mask].numpy()

    print(f"\nTotal samples pooled: {probs.shape[0]}  "
          f"(correct: {correct_mask.sum().item()}, incorrect: {(~correct_mask).sum().item()})")
    if correct_conf.size:
        print(f"  mean P(true class) | correct:   {correct_conf.mean():.4f}  (std={correct_conf.std():.4f})")
    if incorrect_conf.size:
        print(f"  mean P(true class) | incorrect: {incorrect_conf.mean():.4f}  (std={incorrect_conf.std():.4f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    data = [v for v in (correct_conf, incorrect_conf) if v.size]
    labels = [lbl for lbl, v in (("Correct", correct_conf), ("Incorrect", incorrect_conf)) if v.size]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Softmax probability assigned to true class")
    ax.set_ylim(0, 1)
    ax.set_title(f"Confidence calibration (n={probs.shape[0]} samples, "
                 f"{len(all_probs)} fold{'s' if len(all_probs) != 1 else ''})")
    ax.axhline(1.0 / probs.shape[1], color="gray", linestyle="--", linewidth=1, label="chance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()

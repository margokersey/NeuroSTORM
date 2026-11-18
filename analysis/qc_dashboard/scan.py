#!/usr/bin/env python
"""Scan fMRIPrep / XCP-D / data.pt folders and write a small QC cache for the dashboard.

Run on a compute node (the data.pt and --deep passes read a lot of data), e.g. via
sbatch_qc_dashboard.sh. Re-running is incremental: rows whose source file mtime is
unchanged are reused (use --force to recompute).

  python scan.py --name my_dataset \
      --fmriprep /nfs/turbo/.../fmriprep --xcpd /nfs/turbo/.../xcp_d \
      --datapt /nfs/turbo/.../Dataset_for_NeuroSTORM/<run_folder> \
      --cache_root output/qc_dashboard --workers 8 [--deep]

Any stage can be omitted. Cache layout: <cache_root>/<name>/{manifest.json,
fmriprep_runs.csv, xcpd_runs.csv, datapt_subjects.csv, datapt_series/<subject>.npz}
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qc_lib as Q  # noqa: E402


def _mtime(*paths):
    return max((os.path.getmtime(p) for p in paths if p and os.path.exists(p)), default=0.0)


def _bids_job(args):
    kind, rid, files, fd_thr, deep = args
    row = {"run_id": rid, **files["ents"]}
    try:
        row.update(Q.fmriprep_row(files, fd_thr) if kind == "fmriprep" else Q.xcpd_row(files, fd_thr))
        if deep and files.get("bold"):
            row.update(Q.bold_stats(files["bold"], files.get("mask")))
    except Exception as e:  # keep scanning; surface in the table
        row["error"] = f"{type(e).__name__}: {e}"
    row["src_mtime"] = _mtime(*(files.get(k) for k in ("confounds", "motion", "outliers", "qc", "bold")))
    for k in ("confounds", "motion", "outliers", "qc", "bold", "mask"):
        if files.get(k):
            row[f"path_{k}"] = files[k]
    row["deep"] = bool(deep)
    return row


def _datapt_job(args):
    subj, path, series_dir = args
    row = {"subject": subj, "path": path, "src_mtime": os.path.getmtime(path)}
    try:
        import torch
        torch.set_num_threads(1)
        stats, series = Q.datapt_stats(path)
        row.update(stats)
        np.savez_compressed(Path(series_dir) / f"{subj}.npz", **series)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def _reuse(old: pd.DataFrame, key: str, items: dict, mtime_of, force: bool):
    """Split items into (cached rows, todo keys) based on stored src_mtime."""
    if force or old is None or old.empty or "src_mtime" not in old:
        return [], list(items)
    have = old.drop_duplicates(key).set_index(key)
    has_err = "error" in have.columns
    keep, todo = [], []
    for k in items:
        ok = (k in have.index and abs(float(have.at[k, "src_mtime"]) - mtime_of(items[k])) < 1e-6
              and not (has_err and pd.notna(have.at[k, "error"])))
        if ok:
            keep.append(have.loc[[k]].reset_index().iloc[0].to_dict())
        else:
            todo.append(k)
    return keep, todo


def _read_old(p):
    return Q.read_cache_csv(p) if p.is_file() else None


def run_pool(fn, jobs, workers, label):
    out, t0 = [], time.time()
    if not jobs:
        return out
    if workers <= 1:
        it = map(fn, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        it = pool.map(fn, jobs, chunksize=1)
    for i, r in enumerate(it, 1):
        out.append(r)
        if i % 5 == 0 or i == len(jobs):
            print(f"  [{label}] {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)
    if pool:
        pool.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="dataset label (cache subfolder)")
    ap.add_argument("--fmriprep"), ap.add_argument("--xcpd"), ap.add_argument("--datapt")
    ap.add_argument("--cache_root", default="output/qc_dashboard")
    ap.add_argument("--fd_thr", type=float, default=0.5, help="FD threshold (mm) for %% volumes above")
    ap.add_argument("--deep", action="store_true",
                    help="also load preprocessed/denoised BOLD NIfTIs (tSNR, DVARS, ...); slow, ~2-4 GB RAM/worker")
    ap.add_argument("--datapt_subject_regex", default=None,
                    help="regex with one group to turn a data.pt folder name into a subject id")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (a.fmriprep or a.xcpd or a.datapt):
        ap.error("give at least one of --fmriprep / --xcpd / --datapt")

    out = Path(a.cache_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"name": a.name, "fd_thr": a.fd_thr, "deep": a.deep, "scanned": time.strftime("%Y-%m-%d %H:%M:%S"),
                "paths": {"fmriprep": a.fmriprep, "xcpd": a.xcpd, "datapt": a.datapt}, "counts": {}, "warnings": []}

    # BIDS rows depend on the FD threshold and on --deep, so changing either invalidates them
    prev_m = {}
    if (out / "manifest.json").is_file():
        prev_m = json.loads((out / "manifest.json").read_text())
    stale_bids = bool(prev_m) and (prev_m.get("fd_thr") != a.fd_thr or (a.deep and not prev_m.get("deep")))

    for kind, root_arg, pats in (("fmriprep", a.fmriprep, Q.FMRIPREP_PATTERNS), ("xcpd", a.xcpd, Q.XCPD_PATTERNS)):
        if not root_arg:
            continue
        root = Q.resolve_bids_root(root_arg)
        runs = Q.index_runs(root, pats)
        print(f"{kind}: {len(runs)} runs under {root}", flush=True)
        if not runs:
            manifest["warnings"].append(f"{kind}: no runs found under {root} (expected sub-*/[ses-*/]func/)")
            continue
        csv = out / f"{kind}_runs.csv"
        old = _read_old(csv)
        force = a.force or stale_bids
        keep, todo = _reuse(old, "run_id", runs,
                            lambda f: _mtime(*(f.get(k) for k in ("confounds", "motion", "outliers", "qc", "bold"))), force)
        print(f"  reusing {len(keep)}, computing {len(todo)}", flush=True)
        new = run_pool(_bids_job, [(kind, r, runs[r], a.fd_thr, a.deep) for r in todo], a.workers, kind)
        df = pd.DataFrame(keep + new).sort_values("run_id")
        df.to_csv(csv, index=False)
        manifest["counts"][kind] = len(df)
        if "error" in df and df["error"].notna().any():
            manifest["warnings"].append(f"{kind}: {int(df['error'].notna().sum())} runs failed (see 'error' column)")

    if a.datapt:
        root = Q.resolve_datapt_root(a.datapt)
        subs = Q.datapt_subject_dirs(root, a.datapt_subject_regex)
        print(f"datapt: {len(subs)} subjects under {root}", flush=True)
        if not subs:
            manifest["warnings"].append(f"datapt: no <subject>/data.pt under {root}")
        else:
            sdir = out / "datapt_series"
            sdir.mkdir(exist_ok=True)
            csv = out / "datapt_subjects.csv"
            old = _read_old(csv)
            keep, todo = _reuse(old, "subject", subs, os.path.getmtime, a.force)
            keep = [k for k in keep if (sdir / f"{k['subject']}.npz").is_file()]
            todo += [s for s in subs if s not in todo and s not in {k["subject"] for k in keep}]
            print(f"  reusing {len(keep)}, computing {len(todo)}", flush=True)
            new = run_pool(_datapt_job, [(s, subs[s], str(sdir)) for s in todo], a.workers, "datapt")
            df = pd.DataFrame(keep + new).sort_values("subject")
            df.to_csv(csv, index=False)
            manifest["counts"]["datapt"] = len(df)
            if "error" in df and df["error"].notna().any():
                manifest["warnings"].append(f"datapt: {int(df['error'].notna().sum())} subjects failed (see 'error' column)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}\n{json.dumps(manifest['counts'])}  warnings: {manifest['warnings']}", flush=True)


if __name__ == "__main__":
    main()

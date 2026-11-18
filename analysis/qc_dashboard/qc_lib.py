"""Shared helpers for the QC dashboard: stage discovery, metric extraction, loaders.

Three stages are handled, each given only by a root path:
  fmriprep : BIDS derivatives dir  (sub-*/[ses-*/]func/*_desc-confounds_timeseries.tsv, ...)
  xcpd     : BIDS derivatives dir  (sub-*/[ses-*/]func/*_motion.tsv, *_outliers.tsv, *qc*.tsv, ...)
  datapt   : NeuroSTORM dataset    (<root>/img/<subject>/data.pt = {frames int8 [T,H,W,D], scale, num_frames})

Everything is glob-based and tolerant of missing files, so other datasets work by
only changing the paths.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ENTITY_ORDER = ("sub", "ses", "task", "acq", "run")
SPACE_PREF = ("MNI152NLin2009cAsym", "MNI152NLin6Asym", "MNI")


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def parse_entities(name: str) -> dict:
    ents = {}
    for part in os.path.basename(name).split("_"):
        m = re.fullmatch(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", part)
        if m:
            ents[m.group(1)] = m.group(2)
    return ents


def run_id(ents: dict) -> str:
    return "_".join(f"{k}-{ents[k]}" for k in ENTITY_ORDER if k in ents)


def resolve_bids_root(root) -> Path:
    """Accept either the derivatives dir itself or a parent holding it one level down."""
    p = Path(os.path.abspath(Path(root).expanduser()))
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    if any(p.glob("sub-*")):
        return p
    for c in sorted(p.iterdir()):
        if c.is_dir() and any(c.glob("sub-*")):
            return c
    return p


def resolve_datapt_root(root) -> Path:
    p = Path(os.path.abspath(Path(root).expanduser()))
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    return p / "img" if (p / "img").is_dir() else p


def _func_dirs(root: Path):
    return sorted(set(root.glob("sub-*/func")) | set(root.glob("sub-*/ses-*/func")))


def _space_rank(name: str) -> int:
    sp = parse_entities(name).get("space", "")
    for i, s in enumerate(SPACE_PREF):
        if sp.startswith(s):
            return i
    return len(SPACE_PREF)


def index_runs(root: Path, patterns: dict) -> dict:
    """{run_id: {kind: path, 'ents': {...}}}; if several files match a kind, prefer MNI space."""
    runs: dict = {}
    for d in _func_dirs(root):
        for kind, pats in patterns.items():
            for pat in ([pats] if isinstance(pats, str) else pats):
                for f in sorted(d.glob(pat)):
                    ents = parse_entities(f.name)
                    if "sub" not in ents:
                        continue
                    rid = run_id(ents)
                    r = runs.setdefault(rid, {"ents": {k: ents[k] for k in ENTITY_ORDER if k in ents}})
                    cur = r.get(kind)
                    if cur is None or _space_rank(f.name) < _space_rank(cur):
                        r[kind] = str(f)
    return runs


FMRIPREP_PATTERNS = {
    "confounds": "*_desc-confounds_timeseries.tsv",
    "bold": "*_desc-preproc_bold.nii.gz",
    "mask": "*_desc-brain_mask.nii.gz",
}
XCPD_PATTERNS = {
    "motion": "*_motion.tsv",
    "outliers": "*_outliers.tsv",
    "qc": ["*_qc.tsv", "*_qc.csv", "*qc_bold.csv", "*qc_bold.tsv"],
    "bold": ["*_desc-denoised_bold.nii.gz", "*_desc-denoised*_bold.nii.gz"],
}


# --------------------------------------------------------------------------- #
# TSV-derived metrics (cheap)
# --------------------------------------------------------------------------- #
def read_table(path):
    return pd.read_csv(path, sep=None, engine="python") if str(path).endswith(".csv") else pd.read_csv(path, sep="\t")


def _fd_stats(fd: pd.Series, thr: float) -> dict:
    fd = pd.to_numeric(fd, errors="coerce").dropna()
    if fd.empty:
        return {}
    return {"mean_fd": fd.mean(), "median_fd": fd.median(), "max_fd": fd.max(),
            "pct_fd_gt_thr": 100 * (fd > thr).mean()}


def _longest_run(flag: np.ndarray) -> int:
    best = cur = 0
    for v in flag:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def fmriprep_row(files: dict, fd_thr: float) -> dict:
    row = {"n_vol": np.nan}
    if "confounds" not in files:
        return row
    df = read_table(files["confounds"])
    row["n_vol"] = len(df)
    if "framewise_displacement" in df:
        row.update(_fd_stats(df["framewise_displacement"], fd_thr))
    for col, name in (("std_dvars", "mean_std_dvars"), ("dvars", "mean_dvars"), ("rmsd", "mean_rmsd")):
        if col in df:
            row[name] = pd.to_numeric(df[col], errors="coerce").mean()
    if "std_dvars" in df:
        row["max_std_dvars"] = pd.to_numeric(df["std_dvars"], errors="coerce").max()
    for col in ("global_signal", "csf", "white_matter"):
        if col in df:
            row[f"{col}_std"] = pd.to_numeric(df[col], errors="coerce").std()
    trans = [c for c in ("trans_x", "trans_y", "trans_z") if c in df]
    rot = [c for c in ("rot_x", "rot_y", "rot_z") if c in df]
    if trans:
        row["max_abs_trans_mm"] = df[trans].abs().max().max()
    if rot:
        row["max_abs_rot_rad"] = df[rot].abs().max().max()
    row["n_nonsteady"] = sum(c.startswith("non_steady_state_outlier") for c in df.columns)
    row["n_motion_outlier_cols"] = sum(c.startswith("motion_outlier") for c in df.columns)
    row["n_acompcor"] = sum(c.startswith("a_comp_cor") for c in df.columns)
    return row


def xcpd_row(files: dict, fd_thr: float) -> dict:
    row = {"n_vol": np.nan}
    if "motion" in files:
        df = read_table(files["motion"])
        row["n_vol"] = len(df)
        if "framewise_displacement" in df:
            row.update(_fd_stats(df["framewise_displacement"], fd_thr))
    if "outliers" in files:
        df = read_table(files["outliers"])
        flag = None
        for c in df.columns:  # first binary column is the censor flag
            v = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if set(v.unique()) <= {0, 1}:
                flag = v.values > 0
                break
        if flag is not None:
            row["n_vol"] = len(flag)
            row["n_censored"] = int(flag.sum())
            row["pct_censored"] = 100 * flag.mean()
            row["n_retained"] = int((~flag).sum())
            row["longest_censor_run"] = _longest_run(flag)
    if "qc" in files:
        q = read_table(files["qc"])
        if len(q):
            for c in q.columns:
                v = pd.to_numeric(q[c].iloc[0], errors="coerce")
                if pd.notna(v):
                    row[f"qc_{c}"] = float(v)
    return row


# --------------------------------------------------------------------------- #
# NIfTI-derived metrics (deep scan; memory-heavy: loads the 4D file)
# --------------------------------------------------------------------------- #
def bold_stats(bold_path: str, mask_path: str | None = None) -> dict:
    import nibabel as nib

    img = nib.load(bold_path)
    if len(img.shape) != 4:
        return {}
    data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
    mean, std = data.mean(-1), data.std(-1)
    mask = std > 0
    if mask_path and os.path.isfile(mask_path):
        m = np.asanyarray(nib.load(mask_path).dataobj) > 0
        if m.shape == mask.shape:
            mask &= m
    if mask.sum() == 0:
        return {"bold_mask_vox": 0}
    ts = data[mask]  # [N, T]
    gs = ts.mean(0)
    dv = np.sqrt(np.mean(np.diff(ts, axis=1) ** 2, axis=0))
    mn, sd = ts.mean(1), ts.std(1)
    out = {
        "bold_mask_vox": int(mask.sum()),
        "bold_mean_signal": float(mn.mean()),
        "bold_tstd_median": float(np.median(sd)),
        "bold_gs_std": float(gs.std()),
        "bold_dvars_mean": float(dv.mean()),
        "bold_vox_size_mm": float(np.mean(img.header.get_zooms()[:3])),
        "bold_tr": float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else np.nan,
    }
    pos = mn > 0.25 * np.percentile(mn, 98)
    if pos.any() and np.median(mn) > 2 * np.median(sd):  # tSNR only meaningful on non-demeaned data
        out["bold_tsnr_median"] = float(np.median(mn[pos] / np.maximum(sd[pos], 1e-8)))
        out["bold_gs_cv_pct"] = float(100 * gs.std() / abs(gs.mean())) if gs.mean() else np.nan
    return out


# --------------------------------------------------------------------------- #
# data.pt
# --------------------------------------------------------------------------- #
def load_blob(path, mmap=True):
    """Same loading policy as datasets/fmri_datasets.py (trusted, locally produced files)."""
    import pickle
    import torch

    def _load(mm):
        try:
            return torch.load(path, mmap=mm, weights_only=True)
        except pickle.UnpicklingError:
            return torch.load(path, mmap=mm, weights_only=False)

    if mmap and os.environ.get("NEUROSTORM_BLOB_MMAP", "1") != "0":
        try:
            return _load(True)
        except RuntimeError as e:
            if "Operation not permitted" not in str(e):
                raise
    return _load(False)


def datapt_stats(path: str, chunk: int = 32):
    """One pass over a data.pt blob. Returns (row dict, per-frame series dict)."""
    blob = load_blob(path)
    fr = blob["frames"]
    scale = float(blob["scale"])
    T = int(fr.shape[0])
    n_vox = int(np.prod(fr.shape[1:]))
    row = {
        "n_frames": T, "recorded_num_frames": int(blob.get("num_frames", T)),
        "shape": "x".join(map(str, fr.shape[1:])), "scale": scale,
        "dtype": str(fr.dtype), "file_mb": os.path.getsize(path) / 1e6,
        "scale_finite": bool(np.isfinite(scale)),
    }
    s1 = np.zeros(fr.shape[1:], np.float64)
    s2 = np.zeros(fr.shape[1:], np.float64)
    for a in range(0, T, chunk):
        c = fr[a:a + chunk].numpy().astype(np.float64)
        s1 += c.sum(0)
        s2 += (c ** 2).sum(0)
    mu = s1 / T
    sd = np.sqrt(np.maximum(s2 / T - mu ** 2, 0))
    fg = sd > 1e-9
    nfg = int(fg.sum())
    row["n_fg_voxels"] = nfg
    row["fg_fraction"] = nfg / n_vox
    corner = fr[:, :5, :5, :5].numpy()
    row["corner_const"] = bool(corner.std(0).max() == 0)
    row["bg_value"] = float(np.median(corner[0])) * scale
    if nfg:
        idx = np.argwhere(fg)
        row["fg_com_x"], row["fg_com_y"], row["fg_com_z"] = idx.mean(0)
        dims = np.array(fr.shape[1:])
        shell = ((idx < 2) | (idx >= dims - 2)).any(1)
        row["fg_edge_fraction"] = float(shell.mean())
        row["mean_abs_fingerprint"] = float(np.abs(mu[fg]).mean() * scale)
        row["tstd_mean"] = float(sd[fg].mean() * scale)
        nz = sd[fg] > 0
        row["tsnr_median"] = float(np.median(np.abs(mu[fg][nz]) / sd[fg][nz])) if nz.any() else np.nan
    gs, fstd, dvars, sat = (np.zeros(T) for _ in range(4))
    prev = None
    for a in range(0, T, chunk):
        c = fr[a:a + chunk].numpy()
        n = len(c)
        f = c[:, fg].astype(np.float32) * scale  # [t, nfg]
        gs[a:a + n] = f.mean(1)
        fstd[a:a + n] = f.std(1)
        sat[a:a + n] = ((c[:, fg] >= 127) | (c[:, fg] <= -127)).mean(1) * 100
        full = f if prev is None else np.concatenate([prev, f])
        d = np.sqrt(np.mean(np.diff(full, axis=0) ** 2, axis=1)) if len(full) > 1 else np.array([])
        if prev is None:
            dvars[a] = np.nan
            dvars[a + 1:a + n] = d
        else:
            dvars[a:a + n] = d
        prev = f[-1:]
    row["gs_mean"], row["gs_std"] = float(gs.mean()), float(gs.std())
    row["dvars_mean"], row["dvars_max"] = float(np.nanmean(dvars)), float(np.nanmax(dvars)) if T > 1 else np.nan
    row["pct_saturated_mean"], row["pct_saturated_max"] = float(sat.mean()), float(sat.max())
    row["value_abs_max"] = float(np.abs(mu).max() * scale)
    row["has_nan"] = bool(not np.isfinite(gs).all())
    return row, {"global_signal": gs, "frame_std": fstd, "dvars": dvars, "pct_saturated": sat}


def datapt_maps(path: str, chunk: int = 32) -> dict:
    """Per-voxel temporal mean / std / tSNR volumes (dequantized) for the viewer."""
    blob = load_blob(path)
    fr, scale = blob["frames"], float(blob["scale"])
    T = fr.shape[0]
    s1 = np.zeros(fr.shape[1:], np.float64)
    s2 = np.zeros(fr.shape[1:], np.float64)
    for a in range(0, T, chunk):
        c = fr[a:a + chunk].numpy().astype(np.float64)
        s1 += c.sum(0)
        s2 += (c ** 2).sum(0)
    mu = s1 / T
    sd = np.sqrt(np.maximum(s2 / T - mu ** 2, 0))
    tsnr = np.where(sd > 1e-9, np.abs(mu) / np.maximum(sd, 1e-9), 0)
    return {"mean": (mu * scale).astype(np.float32), "std": (sd * scale).astype(np.float32),
            "tsnr": tsnr.astype(np.float32), "fg": sd > 1e-9}


def datapt_subject_dirs(root: Path, regex: str | None = None) -> dict:
    """{subject_key: path-to-data.pt}. Key = folder name minus 'sub-' (or regex group 1)."""
    out = {}
    for d in sorted(root.iterdir()):
        f = d / "data.pt"
        if d.is_dir() and f.is_file():
            key = d.name
            if regex:
                m = re.search(regex, key)
                if m:
                    key = m.group(1)
            out[re.sub(r"^sub-", "", key)] = str(f)
    return out


ID_COLS = {c: str for c in ("sub", "ses", "task", "acq", "run", "subject")}


def read_cache_csv(path):
    """Read a cache CSV keeping BIDS ids (e.g. sub-'01', run-'1') as strings."""
    return pd.read_csv(path, dtype=ID_COLS)

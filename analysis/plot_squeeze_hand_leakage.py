"""Check for neighbour-block leakage around squeeze blocks in left motor-hand parcels.

Claim under test: the high-pass filter leaks a negative copy of each block's
response into the blocks on either side, so left-hand-area signal dips just
BEFORE a squeeze block (which cannot be a real response).

What this does
  1. Loads each subject's XCP-D-derived data.pt (img/<sub>/data.pt), averages the
     voxels of the chosen Schaefer-400 (7-network) parcels -> one ROI trace/subject.
     Default parcels (atlas label values): 50, 51, 53, 57, 59 (LH_SomMot_19/20/22/
     26/28, hand motor cortex) and 76 (LH_DorsAttn_Post_8).
  2. Cuts epochs from the fixed task order in metadata/timing_onehot_task1.tsv.
     An epoch = block A (task run + its trailing rest) followed by the next block B
     (task run + trailing rest), aligned to A's onset.
  3. Contrast, per subject then averaged across subjects (+/- SEM):
        (A blocks whose successor B is squeeze) - (A blocks whose successor is
        another imagery task)
     Squeeze never follows squeeze, so A is tennis or navigation. Panels:
        A = tennis      : [-> squeeze] - [-> navigation]
        A = navigation  : [-> squeeze] - [-> tennis]
        A = any         : pooled
     Leakage predicts a dip in the final seconds of A (before B = squeeze starts),
     and a mirror-image artefact after the squeeze.
  4. Background shading = block type (tennis / navigation / squeeze, with squeeze
     action darker than squeeze imagery); white = rest. In the type-specific panels
     the upper half of the B region shows the squeeze-successor condition, the lower
     half the comparison condition.

Input: either a data.pt dataset root (has img/<sub>/data.pt and metadata/), or a
directory of raw XCP-D denoised NIfTIs (pass --timing pointing at
timing_onehot_task1.tsv; the atlas is resampled onto each BOLD grid).

Run it once per export (e.g. the 0.008 Hz high-pass export and the no-high-pass
export) to compare:
    python analysis/plot_squeeze_hand_leakage.py \
        --root highpass=/nfs/turbo/.../FIXED_xcp_nosmooth_task-imagery_run-1 \
        --root nohp=/nfs/turbo/.../<no-highpass export>

The work is light (reads ~1.5k voxels/subject) but run it under srun/sbatch, not
on the login node, if there are many subjects. Output goes to output/ (gitignored).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_PATH = os.path.join(REPO, "datasets", "atlas", "schaefer_400_7net", "atlas.nii.gz")
DEFAULT_ROOT = ("/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/"
                "FIXED_xcp_nosmooth_task-imagery_run-1")
DEFAULT_PARCELS = [50, 51, 53, 57, 59, 76]

# Okabe-Ito colour-blind-safe hues
COLORS = {"tennis": "#0072B2", "navigation": "#009E73",
          "squeeze_imagery": "#E69F00", "squeeze_action": "#D55E00"}


# ----------------------------------------------------------------------------- atlas
def atlas_in_blob_space(size=96):
    """Atlas label volume in the same 96^3 space as data.pt.

    Mirrors datasets/compute_atlas_map.py (2 mm nearest resample -> select_middle_96
    -> centred zero-pad), the transform used for NeuroSTORM's atlas masking.
    """
    import math
    import nibabel as nib
    vol = np.asarray(nib.load(ATLAS_PATH).dataobj).astype(np.int32)
    out = vol
    crop = []
    for n in out.shape[:3]:
        s = (n - size) // 2 if n > size else 0
        crop.append(slice(s, s + min(n, size)))
    out = out[tuple(crop)]
    padded = np.zeros((size,) * 3, dtype=out.dtype)
    starts = [math.ceil((size - s) / 2) for s in out.shape]
    padded[tuple(slice(st, st + s) for st, s in zip(starts, out.shape))] = out
    return padded


# ----------------------------------------------------------------------------- data
def roi_trace(blob_path, masks, chunk=64):
    """Mean over each parcel -> [n_parcels, T], dequantised. Reads only masked voxels."""
    import torch
    blob = torch.load(blob_path, mmap=True, weights_only=False)
    frames, scale = blob["frames"], float(blob["scale"])  # int8 [T,H,W,D]
    T = frames.shape[0]
    out = np.zeros((len(masks), T), dtype=np.float32)
    union = np.any(masks, axis=0)
    idx = torch.from_numpy(np.flatnonzero(union.ravel()))
    # position of each parcel's voxels within the union list
    pos = np.cumsum(union.ravel()) - 1
    sub = [pos[m.ravel()] for m in masks]
    for t0 in range(0, T, chunk):
        x = frames[t0:t0 + chunk].reshape(min(chunk, T - t0), -1)[:, idx].float().numpy() * scale
        for k, s in enumerate(sub):
            out[k, t0:t0 + x.shape[0]] = x[:, s].mean(axis=1)
    return out, frames.shape[1:]


def load_subjects(root, parcels, cache):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["parcels"]) == list(parcels):
            print(f"[cache] {cache}")
            return list(z["subjects"]), z["traces"]
    atlas = atlas_in_blob_space()
    masks = np.stack([atlas == p for p in parcels])
    print("voxels per parcel:", dict(zip(parcels, masks.reshape(len(parcels), -1).sum(1))))
    img = os.path.join(root, "img")
    subs, traces, T0 = [], [], None
    for s in sorted(os.listdir(img)):
        p = os.path.join(img, s, "data.pt")
        if not os.path.isfile(p):
            continue
        tr, shape = roi_trace(p, masks)
        if tuple(shape) != (96, 96, 96):
            sys.exit(f"{s}: blob spatial shape {tuple(shape)} != 96^3, atlas alignment not valid")
        if T0 is None:
            T0 = tr.shape[1]
        if tr.shape[1] != T0:
            print(f"skip {s}: {tr.shape[1]} frames != {T0}")
            continue
        # sanity: parcels should lie in foreground (non-zero signal)
        if not subs:
            fg = [(np.abs(tr[k]) > 0).mean() for k in range(len(parcels))]
            print("fraction non-zero ROI frames (sanity, first subject):", np.round(fg, 2))
        subs.append(s)
        traces.append(tr)
        print(f"  {s}: {tr.shape[1]} frames")
    traces = np.stack(traces)  # [S, P, T]
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, subjects=np.array(subs), traces=traces, parcels=np.array(parcels))
    return subs, traces


# ----------------------------------------------------------------------------- timing
import re as _re
NII_RE = _re.compile(r"^(sub-[^_]+)_task-imagery_run-0?1_.*desc-denoised_bold\.nii(\.gz)?$")


def load_subjects_nii(root, parcels, cache):
    """XCP-D denoised BOLD NIfTIs (no data.pt needed).

    Finds sub-*_task-imagery_run-1_*_desc-denoised_bold.nii.gz under `root`,
    resamples the Schaefer atlas (nearest) onto each BOLD file's own grid via the
    affines, and averages each parcel. Unsmoothed (desc-denoised, not denoisedSmoothed).
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["parcels"]) == list(parcels):
            print(f"[cache] {cache}")
            return list(z["subjects"]), z["traces"]
    atlas_img = nib.load(ATLAS_PATH)
    files = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            m = NII_RE.match(f)
            if m:
                files[m.group(1)] = os.path.join(d, f)
    if not files:
        sys.exit(f"no *task-imagery_run-1*desc-denoised_bold.nii(.gz) found under {root}")
    subs, traces, T0, grid = [], [], None, None
    for s in sorted(files):
        img = nib.load(files[s])
        tr_hdr = float(img.header.get_zooms()[3])
        if abs(tr_hdr - 0.8) > 0.01:
            sys.exit(f"{s}: TR={tr_hdr}, timing file assumes 0.8 s")
        key = (img.shape[:3], tuple(np.round(img.affine.ravel(), 3)))
        if key != grid:
            lab = np.asarray(resample_from_to(
                atlas_img, (img.shape[:3], img.affine), order=0).dataobj).astype(np.int32)
            masks = np.stack([lab == p for p in parcels])
            grid = key
            print("voxels per parcel on BOLD grid:", dict(zip(parcels, masks.reshape(len(parcels), -1).sum(1))))
        data = np.asarray(img.dataobj, dtype=np.float32)  # [X,Y,Z,T]
        tr = np.stack([data[m].mean(axis=0) for m in masks])  # [P,T]
        if T0 is None:
            T0 = tr.shape[1]
        if tr.shape[1] != T0:
            print(f"skip {s}: {tr.shape[1]} frames != {T0}")
            continue
        if not subs:
            print("fraction non-zero ROI frames (sanity, first subject):",
                  np.round([(np.abs(tr[k]) > 0).mean() for k in range(len(parcels))], 2))
        subs.append(s)
        traces.append(tr)
        print(f"  {s}: {tr.shape[1]} frames")
    traces = np.stack(traces)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, subjects=np.array(subs), traces=traces, parcels=np.array(parcels))
    return subs, traces


def read_blocks(tsv):
    """Blocks in run order: dict(type, start, end(task run), rest_end, phases)."""
    df = pd.read_csv(tsv, sep="\t")
    df["squeeze"] = df["squeeze_imagery"].astype(int) | df["squeeze_action"].astype(int)
    lab = df[["rest", "squeeze", "navigation", "tennis"]].idxmax(axis=1).tolist()
    runs, prev, st = [], None, 0
    for i, l in enumerate(lab + [None]):
        if l != prev:
            if prev is not None:
                runs.append((st, i, prev))
            st, prev = i, l
    blocks = []
    for k, (s, e, l) in enumerate(runs):
        if l == "rest":
            continue
        re_ = runs[k + 1][1] if k + 1 < len(runs) and runs[k + 1][2] == "rest" else e
        blocks.append(dict(type=l, start=s, end=e, rest_end=re_))
    return df, blocks


# ----------------------------------------------------------------------------- plot
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", metavar="[LABEL=]PATH",
                    help="FIXED_xcp_* dataset root (has img/ and metadata/). Repeatable to compare exports.")
    ap.add_argument("--timing", default=None,
                    help="timing_onehot_task1.tsv. Needed for NIfTI roots (no img/ dir); "
                         "default <root>/metadata/timing_onehot_task1.tsv")
    ap.add_argument("--parcels", type=int, nargs="+", default=DEFAULT_PARCELS,
                    help="Schaefer-400 atlas label values to average (default: 50 51 53 57 59 76)")
    ap.add_argument("--tr", type=float, default=0.8)
    ap.add_argument("--pre", type=int, default=0, help="TRs before block-A onset to include")
    ap.add_argument("--dip-window", type=float, default=8.0, help="seconds before B onset for the dip stat")
    ap.add_argument("--out", default=os.path.join(REPO, "output", "squeeze_hand_leakage.png"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    roots = []
    for r in (args.root or [DEFAULT_ROOT]):
        lab, _, path = r.partition("=") if "=" in r else ("", "", r)
        roots.append((lab or os.path.basename(path.rstrip("/")), path))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    results = {}
    for lab, root in roots:
        cache = None if args.no_cache else os.path.join(REPO, "output", "cache", f"hand_roi_{lab}.npz")
        is_blob = os.path.isdir(os.path.join(root, "img"))
        subs, traces = (load_subjects if is_blob else load_subjects_nii)(root, args.parcels, cache)
        roi = traces.mean(axis=1)  # [S, T] mean over the chosen parcels
        df, blocks = read_blocks(args.timing or os.path.join(root, "metadata", "timing_onehot_task1.tsv"))
        T = roi.shape[1]
        print(f"[{lab}] {len(subs)} subjects, T={T}, timing rows={len(df)}, blocks={len(blocks)}")
        print("  order:", " ".join(b["type"][0].upper() for b in blocks))
        results[lab] = (subs, roi, df, blocks)

    # panel definitions: (A type or None, comparison type or None)
    panels = [("tennis", "navigation"), ("navigation", "tennis"), (None, None)]
    fig, axes = plt.subplots(len(panels), len(roots), figsize=(8.5 * len(roots), 3.3 * len(panels)),
                             squeeze=False, sharey="row")
    dt = args.tr
    for ci, (lab, _) in enumerate(roots):
        subs, roi, df, blocks = results[lab]
        T = roi.shape[1]
        for ri, (a_type, other) in enumerate(panels):
            ax = axes[ri][ci]
            sq_ep, ot_ep = [], []  # (block index) lists
            for i in range(len(blocks) - 1):
                a, b = blocks[i], blocks[i + 1]
                if a["type"] == "squeeze":
                    continue
                if a_type is not None and a["type"] != a_type:
                    continue
                if b["type"] == "squeeze":
                    sq_ep.append(i)
                elif other is None or b["type"] == other:
                    ot_ep.append(i)
            if not sq_ep or not ot_ep:
                ax.set_title(f"{lab} A={a_type or 'any'}: no epochs ({len(sq_ep)} vs {len(ot_ep)})")
                continue
            # common epoch length: A onset -> end of B (incl. trailing rest)
            L = min(blocks[i + 1]["rest_end"] - blocks[i]["start"] for i in sq_ep + ot_ep)
            t = (np.arange(-args.pre, L)) * dt

            def epochs(idx):
                out = []
                for i in idx:
                    s = blocks[i]["start"]
                    lo, hi = s - args.pre, s + L
                    if lo < 0 or hi > T:
                        continue
                    out.append(roi[:, lo:hi])  # [S, len]
                return np.stack(out).mean(axis=0)  # per-subject mean over epochs -> [S, len]

            m_sq, m_ot = epochs(sq_ep), epochs(ot_ep)
            diff = m_sq - m_ot
            mean, sem = diff.mean(0), diff.std(0, ddof=1) / np.sqrt(diff.shape[0])

            # shading: block A (from its own type) and B (split: top=squeeze cond, bottom=comparison)
            ref = (sq_ep + ot_ep)[0]
            def shade_a():
                a = blocks[ref]
                x0, x1 = 0, (a["end"] - a["start"]) * dt
                ax.axvspan(x0, x1, color=COLORS[a["type"]] if a_type else "#999999", alpha=0.18, lw=0)
            shade_a()
            tr_sq = blocks[sq_ep[0] + 1]
            b_off = (tr_sq["start"] - blocks[sq_ep[0]]["start"]) * dt
            ymin, ymax = None, None
            ax.set_xlim(t[0], t[-1])
            ax.axvline(b_off, color="k", lw=0.8, ls=":")
            ax.axhline(0, color="#888", lw=0.6)
            ax.fill_between(t, mean - sem, mean + sem, color="k", alpha=0.18, lw=0)
            ax.plot(t, mean, color="k", lw=2)
            ax.margins(y=0.15)
            lo, hi = ax.get_ylim()
            mid = (lo + hi) / 2
            # B shading: squeeze (imagery lighter, action darker) in the upper half
            sq_i = df["squeeze_imagery"].values; sq_a = df["squeeze_action"].values
            s0 = tr_sq["start"]
            for (c0, c1, col) in _phase_spans(sq_i, sq_a, s0, tr_sq["end"]):
                ax.fill_between([b_off + c0 * dt, b_off + c1 * dt], mid, hi, color=COLORS[col], alpha=0.25, lw=0)
            oth = blocks[ot_ep[0] + 1]
            if a_type is not None:
                ob = (oth["end"] - oth["start"]) * dt
                ax.fill_between([b_off, b_off + ob], lo, mid, color=COLORS[oth["type"]], alpha=0.25, lw=0)
            else:
                ax.fill_between([b_off, b_off + (oth["end"] - oth["start"]) * dt], lo, mid,
                                color="#999999", alpha=0.25, lw=0)
            ax.set_ylim(lo, hi)

            # dip statistic: last `dip-window` seconds of A (just before B onset)
            n = int(round(args.dip_window / dt))
            k_b = int(round(b_off / dt)) + args.pre
            win = diff[:, k_b - n:k_b].mean(axis=1)
            tt = stats.ttest_1samp(win, 0.0)
            ax.set_title(f"{lab}: A={a_type or 'tennis/navigation'}  "
                         f"[B=squeeze, n={len(sq_ep)}] - [B={other or 'non-squeeze'}, n={len(ot_ep)}]\n"
                         f"pre-B {args.dip_window:.0f}s mean={win.mean():+.3f}, t({len(win)-1})={tt.statistic:.2f}, p={tt.pvalue:.3g}",
                         fontsize=8, loc="left")
            print(f"[{lab}] A={a_type or 'any'}: pre-squeeze window diff mean={win.mean():+.4f} "
                  f"t={tt.statistic:.2f} p={tt.pvalue:.3g} (n_sub={len(win)}, epochs {len(sq_ep)} vs {len(ot_ep)})")
            ax.set_ylabel("Δ signal (a.u.)")
            if ri == len(panels) - 1:
                ax.set_xlabel("time from block-A onset (s)   (dotted = start of next block B)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.35) for k in COLORS]
    fig.legend(handles, ["tennis", "navigation", "squeeze (imagery)", "squeeze (action)"],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Left hand-motor + DorsAttn parcels " + "/".join(map(str, args.parcels)) +
                 "\nblocks followed by squeeze − blocks followed by other imagery (mean ± SEM across subjects; white = rest)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("saved", args.out)


def _phase_spans(imag, act, s0, e0):
    """(start_offset, end_offset, colour-key) TR spans of squeeze imagery/action inside a block."""
    spans, cur, st = [], None, 0
    for i in range(s0, e0):
        c = "squeeze_action" if act[i] else "squeeze_imagery"
        if c != cur:
            if cur is not None:
                spans.append((st, i - s0, cur))
            cur, st = c, i - s0
    spans.append((st, e0 - s0, cur))
    return spans


if __name__ == "__main__":
    main()

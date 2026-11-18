"""Interactive QC dashboard for fMRIPrep -> XCP-D -> data.pt.

Reads the cache written by scan.py (run that first) and loads individual
subjects' series / volumes lazily. Launch (see sbatch_qc_dashboard.sh, which also
sets up localhost-only binding and the access token):

  streamlit run analysis/qc_dashboard/app.py --server.address 127.0.0.1 -- --cache_root output/qc_dashboard
"""
import argparse
import hmac
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qc_lib as Q  # noqa: E402

st.set_page_config(page_title="fMRI QC dashboard", layout="wide")

# Give every plotly chart a unique, run-stable key (identical-looking charts otherwise collide).
if not getattr(DeltaGenerator.plotly_chart, "_qc_keyed", False):
    _orig_plotly_chart = DeltaGenerator.plotly_chart

    def _keyed_plotly_chart(self, fig, *a, **k):
        k.setdefault("key", f"pc{next(DeltaGenerator._qc_ctr)}")
        return _orig_plotly_chart(self, fig, *a, **k)

    _keyed_plotly_chart._qc_keyed = True
    DeltaGenerator.plotly_chart = _keyed_plotly_chart
DeltaGenerator._qc_ctr = itertools.count()  # reset on every script run

_ap = argparse.ArgumentParser()
_ap.add_argument("--cache_root", default=os.environ.get("QC_CACHE_ROOT", "output/qc_dashboard"))
ARGS, _ = _ap.parse_known_args(sys.argv[1:])

# ------------------------------------------------------------------ access gate
_TOKEN = os.environ.get("QC_DASH_TOKEN")
if _TOKEN and not st.session_state.get("auth"):
    given = st.query_params.get("token", "")
    if not given:
        given = st.text_input("Access token (printed in the Slurm log)", type="password")
    if given and hmac.compare_digest(str(given), _TOKEN):
        st.session_state["auth"] = True
        st.rerun()
    st.stop()

# ------------------------------------------------------------------ data loading
NON_METRIC = {"src_mtime", "scale", "recorded_num_frames"}
ID_LIKE = {"run_id", "sub", "ses", "task", "acq", "run", "subject", "group", "error"}


@st.cache_data(show_spinner=False)
def load_csv(path, mtime):
    return Q.read_cache_csv(path)


@st.cache_data(show_spinner=False)
def load_table(path):
    return Q.read_table(path)


@st.cache_data(show_spinner=False)
def load_series_npz(path, mtime):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@st.cache_resource(max_entries=2)
def get_blob(path):
    return Q.load_blob(path)


@st.cache_data(show_spinner="Computing temporal mean/std/tSNR maps...", max_entries=3)
def get_maps(path):
    return Q.datapt_maps(path)


def load_dataset(d: Path):
    import json
    out = {"manifest": json.loads((d / "manifest.json").read_text())}
    for k, f in (("fmriprep", "fmriprep_runs.csv"), ("xcpd", "xcpd_runs.csv"), ("datapt", "datapt_subjects.csv")):
        p = d / f
        out[k] = load_csv(str(p), p.stat().st_mtime) if p.is_file() else None
    for k in ("fmriprep", "xcpd"):
        if out[k] is not None:
            out[k] = out[k].assign(subject=out[k]["sub"])
    return out


def metric_cols(df):
    return [c for c in df.select_dtypes("number").columns if c not in NON_METRIC | ID_LIKE and not c.startswith("path_")]


# ------------------------------------------------------------------ sidebar
root = Path(ARGS.cache_root)
datasets = sorted(p.parent.name for p in root.glob("*/manifest.json")) if root.is_dir() else []
st.sidebar.title("fMRI QC")
if not datasets:
    st.error(f"No scanned datasets under `{root}`. Run `scan.py --name <dataset> --fmriprep ... --xcpd ... --datapt ...` first.")
    st.stop()
ds_name = st.sidebar.selectbox("Dataset", datasets)
D = load_dataset(root / ds_name)
man = D["manifest"]
FD_THR = man.get("fd_thr", 0.5)
with st.sidebar.expander("Dataset info"):
    st.caption(f"scanned {man.get('scanned')}  |  deep={man.get('deep')}  |  FD threshold {FD_THR} mm")
    for k, v in man["paths"].items():
        st.text(f"{k}: {v}")
    for w in man.get("warnings", []):
        st.warning(w)

fp, xc, dp = D["fmriprep"], D["xcpd"], D["datapt"]
bids = [d for d in (fp, xc) if d is not None]
tasks = sorted(set().union(*[set(d["task"].dropna()) for d in bids if "task" in d])) if bids else []
runs = sorted(set().union(*[set(d["run"].dropna()) for d in bids if "run" in d])) if bids else []
sel_tasks = st.sidebar.multiselect("Task", tasks, default=tasks) if len(tasks) > 1 else tasks
sel_runs = st.sidebar.multiselect("Run", runs, default=runs) if len(runs) > 1 else runs


def filt(df):
    if df is None:
        return None
    if "task" in df and sel_tasks:
        df = df[df["task"].isin(sel_tasks)]
    if "run" in df and sel_runs:
        df = df[df["run"].isin(sel_runs)]
    return df


fp, xc = filt(fp), filt(xc)
all_subj = sorted(set().union(*[set(d["subject"]) for d in (fp, xc, dp) if d is not None]))
subjects = st.sidebar.multiselect("Subjects included", all_subj, default=all_subj)

# groups: participants.tsv columns, task/run, or a custom group A
part_guess = ""
for cand in [man["paths"].get("fmriprep"), man["paths"].get("xcpd")]:
    if cand:
        for up in (Path(cand), Path(cand).parent, Path(cand).parent.parent):
            if (up / "participants.tsv").is_file():
                part_guess = str(up / "participants.tsv")
                break
    if part_guess:
        break
part_path = st.sidebar.text_input("participants.tsv (optional, for grouping)", part_guess)
part = None
if part_path and Path(part_path).is_file():
    part = pd.read_csv(part_path, sep="\t", dtype=str)
    part["subject"] = part[part.columns[0]].str.replace(r"^sub-", "", regex=True)
    part = part.set_index("subject")
group_opts = ["(none)", "Custom group A vs rest"] + (["task"] if len(tasks) > 1 else []) + \
             (["run"] if len(runs) > 1 else []) + ([c for c in part.columns] if part is not None else [])
group_by = st.sidebar.selectbox("Group by", group_opts)
group_a = st.sidebar.multiselect("Group A subjects", subjects) if group_by == "Custom group A vs rest" else []


def add_group(df, subj_col="subject"):
    df = df.copy()
    if group_by == "(none)":
        df["group"] = "all"
    elif group_by == "Custom group A vs rest":
        df["group"] = np.where(df[subj_col].isin(group_a), "A", "rest")
    elif group_by in ("task", "run") and group_by in df:
        df["group"] = df[group_by].astype(str)
    elif part is not None and group_by in part.columns:
        df["group"] = df[subj_col].map(part[group_by]).fillna("n/a")
    else:
        df["group"] = "all"
    return df


with st.sidebar.expander("Flag thresholds"):
    t_fd = st.number_input("fMRIPrep mean FD >", 0.0, 5.0, 0.3, 0.05)
    t_cens = st.number_input("XCP-D % censored >", 0.0, 100.0, 20.0, 1.0)
    t_sat = st.number_input("data.pt max % saturated voxels >", 0.0, 100.0, 1.0, 0.1)
    t_edge = st.number_input("data.pt foreground at FOV edge >", 0.0, 1.0, 0.02, 0.01)

# apply subject filter + group
if fp is not None:
    fp = add_group(fp[fp["subject"].isin(subjects)])
if xc is not None:
    xc = add_group(xc[xc["subject"].isin(subjects)])
if dp is not None:
    dp = add_group(dp[dp["subject"].isin(subjects)])

focus = st.sidebar.selectbox("Focus subject (Subject detail tab)", subjects) if subjects else None

tabs = st.tabs(["Overview", "fMRIPrep", "XCP-D", "data.pt", "Cross-stage", "Subject detail"])


# ------------------------------------------------------------------ helpers
def flag_table():
    rows = {s: [] for s in subjects}
    if fp is not None and "mean_fd" in fp:
        for s, v in fp.groupby("subject")["mean_fd"].mean().items():
            if v > t_fd:
                rows[s].append(f"fMRIPrep mean FD {v:.2f}")
    if xc is not None and "pct_censored" in xc:
        for s, v in xc.groupby("subject")["pct_censored"].max().items():
            if v > t_cens:
                rows[s].append(f"XCP-D {v:.0f}% censored")
    if dp is not None:
        for s, r in dp.set_index("subject").iterrows():
            if r.get("pct_saturated_max", 0) > t_sat:
                rows[s].append(f"data.pt saturated {r['pct_saturated_max']:.1f}%")
            if r.get("fg_edge_fraction", 0) > t_edge:
                rows[s].append(f"data.pt FG at FOV edge {r['fg_edge_fraction']:.1%}")
            if r.get("has_nan", False) or not r.get("scale_finite", True):
                rows[s].append("data.pt NaN/inf")
            if pd.notna(r.get("error")):
                rows[s].append("data.pt scan error")
    for name, d in (("fMRIPrep", fp), ("XCP-D", xc)):
        if d is not None and "error" in d:
            for s in d.loc[d["error"].notna(), "subject"].unique():
                rows[s].append(f"{name} scan error")
    return rows


def stage_tab(df, name, idcol, defaults, caption=""):
    if df is None or df.empty:
        st.info(f"No {name} data in this dataset/selection.")
        return
    if caption:
        st.caption(caption)
    cols = metric_cols(df)
    pick = st.multiselect(f"{name} metrics", cols, default=[c for c in defaults if c in cols], key=f"m_{name}")
    kind = st.radio("Plot", ["Box + points", "Histogram", "Per-subject bars"], horizontal=True, key=f"k_{name}")
    ncol = 2
    for i in range(0, len(pick), ncol):
        for c, m in zip(st.columns(ncol), pick[i:i + ncol]):
            if kind == "Box + points":
                fig = px.box(df, x="group", y=m, color="group", points="all", hover_data=[idcol])
            elif kind == "Histogram":
                fig = px.histogram(df, x=m, color="group", barmode="overlay", opacity=.7, nbins=25)
            else:
                fig = px.bar(df.sort_values(m), x=idcol, y=m, color="group")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=len(df["group"].unique()) > 1,
                              title=m)
            c.plotly_chart(fig, use_container_width=True)
    if pick:
        st.subheader("Summary by group")
        g = df.groupby("group")[pick].agg(["count", "mean", "std", "median", "min", "max"]).round(3)
        st.dataframe(g, use_container_width=True)
        groups = [x for x in df["group"].unique()]
        if len(groups) == 2:
            from scipy.stats import mannwhitneyu
            rows = []
            for m in pick:
                a, b = (df.loc[df["group"] == k, m].dropna() for k in groups)
                if len(a) > 1 and len(b) > 1:
                    rows.append({"metric": m, f"median {groups[0]}": a.median(), f"median {groups[1]}": b.median(),
                                 "Mann-Whitney p": mannwhitneyu(a, b).pvalue})
            if rows:
                st.caption("Two groups: unpaired Mann-Whitney U (uncorrected; runs from one subject are not independent).")
                st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
    st.subheader("Table")
    st.dataframe(df.drop(columns=[c for c in df if c.startswith("path_")], errors="ignore"), use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{name}_qc.csv")


# ------------------------------------------------------------------ Overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Subjects", len(subjects))
    c[1].metric("fMRIPrep runs", 0 if fp is None else len(fp))
    c[2].metric("XCP-D runs", 0 if xc is None else len(xc))
    c[3].metric("data.pt subjects", 0 if dp is None else len(dp))
    cov = pd.DataFrame(index=subjects)
    if fp is not None:
        cov["fMRIPrep runs"] = fp.groupby("subject").size().reindex(subjects).fillna(0).astype(int)
    if xc is not None:
        cov["XCP-D runs"] = xc.groupby("subject").size().reindex(subjects).fillna(0).astype(int)
    if dp is not None:
        cov["data.pt"] = pd.Series(1, index=dp["subject"]).reindex(subjects).fillna(0).astype(int)
    fl = flag_table()
    cov["flags"] = ["; ".join(fl[s]) for s in cov.index]
    cov["missing stage"] = [any(cov.loc[s, k] == 0 for k in cov.columns if k not in ("flags",)) for s in cov.index]
    st.subheader("Coverage and flags")
    only_flag = st.checkbox("Only show subjects with flags or missing stages")
    show = cov[(cov["flags"] != "") | cov["missing stage"]] if only_flag else cov
    st.dataframe(show, use_container_width=True)
    st.caption("Thresholds are in the sidebar. A 0 means that stage has no file for the subject (or the subject id does not "
               "match across stages -- ids are compared with any 'sub-' prefix removed).")

with tabs[1]:
    stage_tab(fp, "fMRIPrep", "run_id",
              ["mean_fd", "pct_fd_gt_thr", "max_fd", "mean_std_dvars", "max_abs_trans_mm", "n_vol", "bold_tsnr_median"],
              f"pct_fd_gt_thr = % volumes with FD > {FD_THR} mm. bold_* metrics exist only if scanned with --deep.")
with tabs[2]:
    stage_tab(xc, "XCP-D", "run_id",
              ["mean_fd", "pct_censored", "n_retained", "longest_censor_run"] +
              [c for c in (xc.columns if xc is not None else []) if c.startswith("qc_")][:6],
              "pct_censored comes from the *_outliers.tsv censor flags; qc_* columns are whatever the XCP-D QC file holds.")
with tabs[3]:
    stage_tab(dp, "data.pt", "subject",
              ["n_frames", "gs_std", "dvars_mean", "pct_saturated_max", "tsnr_median", "mean_abs_fingerprint", "fg_fraction",
               "fg_edge_fraction"],
              "Computed over foreground voxels (temporal std > 0) on the dequantized int8 data. pct_saturated = voxels at "
              "+/-127 (clipped by quantization). mean_abs_fingerprint = mean |voxel temporal mean|.")

# ------------------------------------------------------------------ Cross-stage
with tabs[4]:
    parts = {}
    for pre, d in (("fmriprep", fp), ("xcpd", xc), ("datapt", dp)):
        if d is not None and not d.empty:
            m = d.groupby("subject")[metric_cols(d)].mean().add_prefix(pre + ".")
            parts[pre] = m
    if len(parts) < 1:
        st.info("Nothing to compare.")
    else:
        sub_df = pd.concat(parts.values(), axis=1).reset_index().rename(columns={"index": "subject"})
        sub_df = add_group(sub_df)
        st.caption("Subject-level table: runs are averaged within a subject; columns are `<stage>.<metric>`.")
        mc = [c for c in sub_df.columns if "." in c]
        x = st.selectbox("X", mc, index=0)
        y = st.selectbox("Y", mc, index=min(1, len(mc) - 1))
        fig = px.scatter(sub_df, x=x, y=y, color="group", hover_name="subject")
        ok = sub_df[[x, y]].dropna()
        if len(ok) > 2:
            from scipy.stats import spearmanr
            r, p = spearmanr(ok[x], ok[y])
            fig.update_layout(title=f"Spearman rho={r:.2f}, p={p:.3g}, n={len(ok)}")
        st.plotly_chart(fig, use_container_width=True)
        sel = st.multiselect("Correlation matrix metrics", mc, default=mc[:8])
        if len(sel) > 1:
            cm = sub_df[sel].corr(method="spearman")
            st.plotly_chart(px.imshow(cm, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto"),
                            use_container_width=True)
        st.download_button("Download subject-level CSV", sub_df.to_csv(index=False), file_name="subject_level_qc.csv")


# ------------------------------------------------------------------ Subject detail
def line_fig(series: dict, title, thr=None, shade=None, height=240):
    fig = go.Figure()
    for k, v in series.items():
        fig.add_trace(go.Scatter(y=v, mode="lines", name=k, line=dict(width=1.2)))
    if thr is not None:
        fig.add_hline(y=thr, line_dash="dash", line_color="grey")
    if shade is not None:
        idx = np.where(shade)[0]
        for i in idx:
            fig.add_vrect(x0=i - .5, x1=i + .5, fillcolor="red", opacity=.25, line_width=0)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="volume")
    return fig


def kv(series):
    return series.dropna().astype(str).to_frame("value")


def pick_row(df, label):
    rows = df[df["subject"] == focus]
    if rows.empty:
        st.info(f"{label}: no data for sub-{focus}.")
        return None
    rid = st.selectbox(f"{label} run", rows["run_id"], key=f"run_{label}_{focus}") if len(rows) > 1 else rows["run_id"].iloc[0]
    return rows[rows["run_id"] == rid].iloc[0]


def vol_view(path):
    blob = get_blob(path)
    fr, sc = blob["frames"], float(blob["scale"])
    T, X, Y, Z = fr.shape
    mode = st.radio("Show", ["Single frame", "Temporal mean", "Temporal std", "tSNR (|mean|/std)"], horizontal=True)
    if mode == "Single frame":
        t = st.slider("Frame", 0, T - 1, 0)
        vol = fr[t].numpy().astype(np.float32) * sc
        lo, hi = np.percentile(vol[vol != 0], [1, 99]) if (vol != 0).any() else (0, 1)
    else:
        mp = get_maps(path)
        vol = mp[{"Temporal mean": "mean", "Temporal std": "std"}.get(mode, "tsnr")]
        fgv = vol[mp["fg"]]
        lo, hi = np.percentile(fgv, [1, 99]) if fgv.size else (0, 1)
        if mode == "Temporal mean":
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
    c = st.columns(3)
    x = c[0].slider("x (sagittal)", 0, X - 1, X // 2)
    y = c[1].slider("y (coronal)", 0, Y - 1, Y // 2)
    z = c[2].slider("z (axial)", 0, Z - 1, Z // 2)
    scale = "RdBu_r" if mode == "Temporal mean" else "gray"
    for col, img, ttl in zip(st.columns(3), (vol[x].T, vol[:, y].T, vol[:, :, z].T), ("Sagittal", "Coronal", "Axial")):
        f = px.imshow(img, origin="lower", zmin=lo, zmax=hi, color_continuous_scale=scale, aspect="equal")
        f.update_layout(title=ttl, height=330, margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False)
        f.update_xaxes(visible=False), f.update_yaxes(visible=False)
        col.plotly_chart(f, use_container_width=True)
    ts = fr[:, x, y, z].numpy().astype(np.float32) * sc
    c1, c2 = st.columns(2)
    c1.plotly_chart(line_fig({f"voxel ({x},{y},{z})": ts}, "Voxel time series"), use_container_width=True)
    v = vol[vol != 0] if mode == "Single frame" else vol[get_maps(path)["fg"]]
    c2.plotly_chart(px.histogram(v, nbins=80, title="Foreground value histogram").update_layout(
        height=240, margin=dict(l=10, r=10, t=35, b=10), showlegend=False), use_container_width=True)


with tabs[5]:
    if not focus:
        st.info("No subjects selected.")
    else:
        st.header(f"sub-{focus}")
        if fp is not None:
            st.subheader("fMRIPrep")
            r = pick_row(fp, "fMRIPrep")
            if r is not None and pd.notna(r.get("path_confounds")) and Path(r["path_confounds"]).is_file():
                df = load_table(r["path_confounds"])
                c = st.columns(2)
                if "framewise_displacement" in df:
                    c[0].plotly_chart(line_fig({"FD (mm)": df["framewise_displacement"]}, "Framewise displacement", FD_THR),
                                      use_container_width=True)
                dv = {k: df[k] for k in ("std_dvars", "dvars") if k in df}
                if dv:
                    c[1].plotly_chart(line_fig(dv, "DVARS"), use_container_width=True)
                c = st.columns(2)
                sig = {k: df[k] for k in ("global_signal", "csf", "white_matter") if k in df}
                if sig:
                    c[0].plotly_chart(line_fig(sig, "Global / CSF / WM signal"), use_container_width=True)
                mot = {k: df[k] for k in ("trans_x", "trans_y", "trans_z") if k in df}
                if mot:
                    c[1].plotly_chart(line_fig(mot, "Translations (mm)"), use_container_width=True)
                rot = {k: df[k] for k in ("rot_x", "rot_y", "rot_z") if k in df}
                if rot:
                    st.plotly_chart(line_fig(rot, "Rotations (rad)"), use_container_width=True)
                with st.expander("fMRIPrep metrics"):
                    st.dataframe(kv(r.drop(labels=[i for i in r.index if str(i).startswith("path_")])))
        if xc is not None:
            st.subheader("XCP-D")
            r = pick_row(xc, "XCP-D")
            if r is not None:
                cens = None
                if pd.notna(r.get("path_outliers")) and Path(r["path_outliers"]).is_file():
                    o = load_table(r["path_outliers"])
                    for cname in o.columns:
                        v = pd.to_numeric(o[cname], errors="coerce").fillna(0)
                        if set(v.unique()) <= {0, 1}:
                            cens = v.values > 0
                            break
                if pd.notna(r.get("path_motion")) and Path(r["path_motion"]).is_file():
                    m = load_table(r["path_motion"])
                    if "framewise_displacement" in m:
                        st.plotly_chart(line_fig({"FD (mm)": m["framewise_displacement"]},
                                                 "XCP-D FD (red = censored volumes)", FD_THR, cens), use_container_width=True)
                with st.expander("XCP-D metrics (incl. QC file)", expanded=True):
                    st.dataframe(kv(r.drop(labels=[i for i in r.index if str(i).startswith("path_")])))
        if dp is not None:
            st.subheader("data.pt")
            row = dp[dp["subject"] == focus]
            if row.empty:
                st.info(f"No data.pt for sub-{focus}.")
            else:
                row = row.iloc[0]
                npz = root / ds_name / "datapt_series" / f"{focus}.npz"
                if npz.is_file():
                    s = load_series_npz(str(npz), npz.stat().st_mtime)
                    c = st.columns(3)
                    c[0].plotly_chart(line_fig({"global signal": s["global_signal"]}, "Foreground global signal"),
                                      use_container_width=True)
                    c[1].plotly_chart(line_fig({"DVARS-like": s["dvars"], "frame std": s["frame_std"]}, "Frame-to-frame change"),
                                      use_container_width=True)
                    c[2].plotly_chart(line_fig({"% saturated": s["pct_saturated"]}, "% foreground voxels at int8 limit"),
                                      use_container_width=True)
                with st.expander("data.pt metrics"):
                    st.dataframe(kv(row.drop(labels=["path"], errors="ignore")))
                if st.checkbox("Load volume viewer (reads this subject's data.pt)", value=False):
                    vol_view(row["path"])

#!/usr/bin/env python
"""Plot Test 3 (task decoding during LOR): per-subject block accuracy by segment.

Reads lor_predictions_fold_<N>.json from every fold (written by
analysis/predict_lor_task_decoding.py). For each held-out subject computes block accuracy
(3-way tennis / navigation / squeeze) in pre_lor (responsive control), lor and ror, drops
boundary blocks, and draws paired per-subject dots/lines with group means (95% bootstrap CI).
A segment is only used for a subject if it has >= --min_blocks blocks. Paired Wilcoxon
signed-rank tests (pre_lor vs lor, lor vs ror) and a one-sample test of lor vs chance (1/3)
are annotated (needs scipy; skipped if missing). Also writes the per-subject table as CSV.

Awake baseline: unless --no_run1, a first column shows each subject's accuracy on the imagery
run-1 blocks of the SAME 3-way model (its own held-out test predictions, from
<output_root>/<stage_b_prefix>_fold-<N>/lightning_logs/version_*/test_predictions). The latest
version is used; a warning is printed if it was not made with final_test_ckpt=best (the LOR
predictions use the best checkpoint). Run 1 vs pre_lor is reported in the console/caption.

    python analysis/plot_lor_task_decoding.py \\
        --pred_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_lor_task_decoding_fold-*/lor_predictions_fold_*.json" \\
        --output_file figure_outputs/lor_task_decoding/lor_accuracy_by_segment

Writes .png (600 dpi), .pdf, .svg and .csv.
"""
import argparse
import csv
import importlib.util
import json
from glob import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    'perm_style', HERE.parent / 'scripts' / 'analyze_loocv_permutation_results.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from predict_stage_b_on_stage_a_blocks import STAGE_B_PREFIX, latest_saved_predictions  # noqa: E402

plt = S.plt if hasattr(S, 'plt') else None
if plt is None:
    import matplotlib.pyplot as plt  # noqa: E402

SEGMENTS = [('run1', 'Run 1\n(awake baseline)'), ('pre_lor', 'Before LOR\n(responsive control)'), ('lor', 'During LOR'), ('ror', 'After ROR\n(responsive again)'),
            ('recovery', 'Run 4\n(recovery)')]
SEG_COLORS = {'run1': '#8172B3', 'pre_lor': '#4C72B0', 'lor': '#C44E52', 'ror': '#55A868', 'recovery': '#CCB974'}
CHANCE = 1 / 3


def load_run1(output_root, prefix, fold):
    """Per-block predictions of the 3-way run's own test set (imagery run 1), or (None, info)."""
    return latest_saved_predictions(str(Path(output_root) / f"{prefix}_fold-{fold}"))


def load(pred_glob, exclude, output_root=None, prefix=None, run4_glob=None):
    files = sorted(glob(pred_glob))
    if not files:
        raise SystemExit(f"no files match {pred_glob}")
    run4 = {}
    for f in (sorted(glob(run4_glob)) if run4_glob else []):
        d4 = json.load(open(f))
        run4[d4['meta']['fold']] = d4
    rows = {}
    for f in files:
        d = json.load(open(f))
        sub = d['meta']['subjects'][0]
        if sub in exclude:
            continue
        rec = {'subject': sub, 'fold': d['meta']['fold'], 'ror_na': d['predictions'][0]['ror_tr'] is None}
        d4 = run4.get(rec['fold'])
        if d4 is not None and d4['meta']['subjects'] != [sub]:
            print(f"WARNING fold {rec['fold']}: run-4 file is for {d4['meta']['subjects']}, not {sub}; ignored")
            d4 = None
        if output_root:
            r1, info = load_run1(output_root, prefix, d['meta']['fold'])
            if r1 is not None and info.get('final_test_ckpt') != 'best':
                print(f"WARNING fold {rec['fold']}: run-1 predictions used final_test_ckpt={info.get('final_test_ckpt')!r}, not 'best'")
            if r1 is not None and {p['subject'] for p in r1} != {sub}:
                print(f"WARNING fold {rec['fold']}: run-1 predictions are for {sorted({p['subject'] for p in r1})}, not {sub}; ignored")
                r1 = None
            if r1 is None:
                print(f"WARNING fold {rec['fold']}: no saved run-1 test predictions")
        else:
            r1 = None
        for seg, _ in SEGMENTS:
            if seg == 'run1':
                ps = r1 or []
            elif seg == 'recovery':
                ps = [p for p in d4['predictions'] if p['segment'] == 'recovery'] if d4 is not None else []
            else:
                ps = [p for p in d['predictions'] if p['segment'] == seg]
            rec[f'{seg}_n'] = len(ps)
            rec[f'{seg}_acc'] = float(np.mean([p['predicted_label'] == p['true_label'] for p in ps])) if ps else np.nan
        rows[sub] = rec
    return [rows[k] for k in sorted(rows)]


def boot_ci(x, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def wilcoxon(a, b=None, mu=None):
    try:
        from scipy.stats import wilcoxon as w
    except ImportError:
        return None
    try:
        return float(w(a - b if b is not None else a - mu).pvalue)
    except ValueError:
        return None


def fmt_p(p):
    return 'p n/a' if p is None else ('p < 0.001' if p < 0.001 else f'p = {p:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred_glob', required=True)
    ap.add_argument('--output_file', default='figure_outputs/lor_task_decoding/lor_accuracy_by_segment')
    ap.add_argument('--output_root', default='/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output',
                    help='parent of the 3-way LOOCV run folders (for the run-1 baseline)')
    ap.add_argument('--stage_b_prefix', default=STAGE_B_PREFIX, help='3-way run folder prefix (<prefix>_fold-<N>)')
    ap.add_argument('--run4_glob', default=None,
                    help='run-4 (recovery) prediction files, lor_predictions_run4_fold_*.json (default: derived from '
                         '--pred_glob). The column is dropped if no files exist.')
    ap.add_argument('--no_run1', action='store_true', help='omit the run-1 awake baseline column')
    ap.add_argument('--min_blocks', type=int, default=5, help='min blocks in a segment for a subject to count there')
    ap.add_argument('--exclude_subjects', nargs='*', default=[])
    args = ap.parse_args()

    run4_glob = args.run4_glob or args.pred_glob.replace('lor_predictions_fold_', 'lor_predictions_run4_fold_')
    rows = load(args.pred_glob, set(args.exclude_subjects), None if args.no_run1 else args.output_root,
                args.stage_b_prefix, run4_glob)
    if args.no_run1:
        SEGMENTS[:] = [x for x in SEGMENTS if x[0] != 'run1']
    if not any(r['recovery_n'] for r in rows):
        print(f"no run-4 predictions found ({run4_glob}); omitting the recovery column")
        SEGMENTS[:] = [x for x in SEGMENTS if x[0] != 'recovery']
    ror_na = {r['subject']: r['ror_na'] for r in rows}
    acc = {}
    for seg, _ in SEGMENTS:
        acc[seg] = {r['subject']: r[f'{seg}_acc'] for r in rows if r[f'{seg}_n'] >= args.min_blocks}

    def paired(a, b):
        subs = sorted(set(acc[a]) & set(acc[b]))
        return subs, np.array([acc[a][s] for s in subs]), np.array([acc[b][s] for s in subs])

    S._style()
    fig, ax = plt.subplots(figsize=(5.6 + 1.1 * ('recovery' in [x[0] for x in SEGMENTS]), 3.6))
    xs = {seg: i for i, (seg, _) in enumerate(SEGMENTS)}
    rng = np.random.RandomState(1)
    # paired lines between adjacent segments
    for a, b in (('run1', 'pre_lor'), ('pre_lor', 'lor'), ('lor', 'ror'), ('ror', 'recovery')):
        if a not in xs or b not in xs:
            continue
        for s in set(acc[a]) & set(acc[b]):
            ax.plot([xs[a], xs[b]], [acc[a][s], acc[b][s]], color=S.CHANCE, lw=0.5, alpha=0.5, zorder=1)
    for seg, _ in SEGMENTS:
        v = np.array(list(acc[seg].values()))
        subs_seg = list(acc[seg].keys())
        jit = rng.uniform(-0.06, 0.06, len(v))
        na = np.array([seg == 'recovery' and ror_na[x] for x in subs_seg], bool)
        ax.scatter((xs[seg] + jit)[~na], v[~na], s=14, color=SEG_COLORS[seg], alpha=0.75,
                   edgecolor='white', linewidth=0.4, zorder=3, clip_on=False)
        if na.any():   # open circles: ROR = N/A (woken manually before run 4)
            ax.scatter((xs[seg] + jit)[na], v[na], s=16, facecolor='white', edgecolor=SEG_COLORS[seg],
                       linewidth=1.0, zorder=3, clip_on=False)
        lo, hi = boot_ci(v)
        ax.plot([xs[seg] + 0.22] * 2, [lo, hi], color=S.INK, lw=1.2, zorder=4)
        ax.scatter([xs[seg] + 0.22], [v.mean()], marker='D', s=22, color=S.INK, zorder=5)
    ax.axhline(CHANCE, color=S.CHANCE, ls='--', lw=0.8, zorder=0)
    ax.text(len(SEGMENTS) - 0.55, CHANCE + 0.01, 'chance', fontsize=6.5, color=S.INK_2, ha='right', va='bottom')

    stats = []
    if 'run1' in xs:
        sb, va, vb = paired('run1', 'pre_lor')
        stats.append(('run1 vs pre_lor', len(sb), wilcoxon(va, vb)))
    for (a, b), y in ((('pre_lor', 'lor'), 1.09), (('lor', 'ror'), 1.17)):
        subs, va, vb = paired(a, b)
        p = wilcoxon(va, vb)
        stats.append((f'{a} vs {b}', len(subs), p))
        ax.plot([xs[a], xs[a], xs[b], xs[b]], [y - 0.02, y, y, y - 0.02], color=S.INK_2, lw=0.7, clip_on=False)
        ax.text((xs[a] + xs[b]) / 2, y + 0.005, f'{fmt_p(p)} (n={len(subs)})', ha='center', va='bottom', fontsize=6.5,
                color=S.INK_2)
    if 'recovery' in xs:
        for a in ('pre_lor', 'lor'):
            sb, va, vb = paired(a, 'recovery')
            stats.append((f'{a} vs recovery', len(sb), wilcoxon(va, vb)))
    lv = np.array(list(acc['lor'].values()))
    p_chance = wilcoxon(lv, mu=CHANCE)
    stats.append(('lor vs chance', len(lv), p_chance))

    ax.set_xticks(range(len(SEGMENTS)))
    ax.set_xticklabels([lab for _, lab in SEGMENTS])
    ax.set_xlim(-0.5, len(SEGMENTS) - 0.5)
    ax.set_ylim(0, 1.03)   # headroom so dots at 1.0 are not cut off
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Block accuracy (3-way task decoding)')
    ax.set_title('Task decoding by responsiveness segment', pad=38)

    n_all = len(rows)
    all_subs = [r['subject'] for r in rows]
    no_ror = [x for x in all_subs if x not in acc['ror']] if 'ror' in acc else []
    fig.text(0.0, -0.04,
             f"{n_all} held-out subjects (one LOOCV fold each); each dot = one subject's block accuracy;\n"
             f"lines join the same subject. Diamond + bar: mean, 95% bootstrap CI. Dashed line: chance.\n"
             f"Run 1 = the same 3-way model's own held-out test blocks (awake); other columns are\n"
             f"runs 2-4 of the same subject. Subjects need >= {args.min_blocks} blocks in a segment to be\n"
             f"included there; boundary blocks excluded. Paired Wilcoxon signed-rank tests.\n"
             f"During LOR vs chance (1/3): {fmt_p(p_chance)} (n={len(lv)}).\n"
             f"No after-ROR point (ROR N/A, run 3 unusable, or < {args.min_blocks} blocks): "
             f"{', '.join(s.replace('sub-', '') for s in no_ror) or 'none'}."
             + ("\nSub-29: run 3 excluded (frame count != timing file)." if 'sub-29' in all_subs else "")
             + ("\nRun 4: all blocks; open circles = ROR N/A (woken manually before run 4). Lines join run 3\n"
                "(after ROR) to run 4 only for subjects with both." if 'recovery' in xs else ""),
             fontsize=6.5, color=S.INK_2, va='top', linespacing=1.4)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with open(out.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg, .csv): {n_all} subjects")
    for name, n, p in stats:
        print(f"  {name}: n={n} {fmt_p(p)}")
    for seg, _ in SEGMENTS:
        v = np.array(list(acc[seg].values()))
        print(f"  {seg}: n={len(v)} mean acc={v.mean():.3f}")


if __name__ == '__main__':
    main()

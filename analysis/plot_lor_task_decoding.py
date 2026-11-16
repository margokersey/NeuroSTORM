#!/usr/bin/env python
"""Plot Test 3 (task decoding during LOR): per-subject block accuracy by segment.

Reads lor_predictions_fold_<N>.json from every fold (written by
analysis/predict_lor_task_decoding.py). For each held-out subject computes block accuracy
(3-way tennis / navigation / squeeze) in pre_lor (responsive control), lor and ror, drops
boundary blocks, and draws paired per-subject dots/lines with group means (95% bootstrap CI).
A segment is only used for a subject if it has >= --min_blocks blocks. Paired Wilcoxon
signed-rank tests (pre_lor vs lor, lor vs ror) and a one-sample test of lor vs chance (1/3)
are annotated (needs scipy; skipped if missing). Also writes the per-subject table as CSV.

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
plt = S.plt if hasattr(S, 'plt') else None
if plt is None:
    import matplotlib.pyplot as plt  # noqa: E402

SEGMENTS = [('pre_lor', 'Before LOR\n(responsive control)'), ('lor', 'During LOR'), ('ror', 'After ROR\n(responsive again)')]
SEG_COLORS = {'pre_lor': '#4C72B0', 'lor': '#C44E52', 'ror': '#55A868'}
CHANCE = 1 / 3


def load(pred_glob, exclude):
    files = sorted(glob(pred_glob))
    if not files:
        raise SystemExit(f"no files match {pred_glob}")
    rows = {}
    for f in files:
        d = json.load(open(f))
        sub = d['meta']['subjects'][0]
        if sub in exclude:
            continue
        rec = {'subject': sub, 'fold': d['meta']['fold']}
        for seg, _ in SEGMENTS:
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
    ap.add_argument('--min_blocks', type=int, default=5, help='min blocks in a segment for a subject to count there')
    ap.add_argument('--exclude_subjects', nargs='*', default=[])
    args = ap.parse_args()

    rows = load(args.pred_glob, set(args.exclude_subjects))
    acc = {}
    for seg, _ in SEGMENTS:
        acc[seg] = {r['subject']: r[f'{seg}_acc'] for r in rows if r[f'{seg}_n'] >= args.min_blocks}

    def paired(a, b):
        subs = sorted(set(acc[a]) & set(acc[b]))
        return subs, np.array([acc[a][s] for s in subs]), np.array([acc[b][s] for s in subs])

    S._style()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    xs = {seg: i for i, (seg, _) in enumerate(SEGMENTS)}
    rng = np.random.RandomState(1)
    # paired lines between adjacent segments
    for a, b in (('pre_lor', 'lor'), ('lor', 'ror')):
        for s in set(acc[a]) & set(acc[b]):
            ax.plot([xs[a], xs[b]], [acc[a][s], acc[b][s]], color=S.CHANCE, lw=0.5, alpha=0.5, zorder=1)
    for seg, _ in SEGMENTS:
        v = np.array(list(acc[seg].values()))
        ax.scatter(xs[seg] + rng.uniform(-0.06, 0.06, len(v)), v, s=14, color=SEG_COLORS[seg], alpha=0.75,
                   edgecolor='white', linewidth=0.4, zorder=3)
        lo, hi = boot_ci(v)
        ax.plot([xs[seg] + 0.22] * 2, [lo, hi], color=S.INK, lw=1.2, zorder=4)
        ax.scatter([xs[seg] + 0.22], [v.mean()], marker='D', s=22, color=S.INK, zorder=5)
    ax.axhline(CHANCE, color=S.CHANCE, ls='--', lw=0.8, zorder=0)
    ax.text(2.45, CHANCE + 0.01, 'chance', fontsize=6.5, color=S.INK_2, ha='right', va='bottom')

    stats = []
    for (a, b), y in ((('pre_lor', 'lor'), 1.04), (('lor', 'ror'), 1.12)):
        subs, va, vb = paired(a, b)
        p = wilcoxon(va, vb)
        stats.append((f'{a} vs {b}', len(subs), p))
        ax.plot([xs[a], xs[a], xs[b], xs[b]], [y - 0.02, y, y, y - 0.02], color=S.INK_2, lw=0.7, clip_on=False)
        ax.text((xs[a] + xs[b]) / 2, y + 0.005, f'{fmt_p(p)} (n={len(subs)})', ha='center', va='bottom', fontsize=6.5,
                color=S.INK_2)
    lv = np.array(list(acc['lor'].values()))
    p_chance = wilcoxon(lv, mu=CHANCE)
    stats.append(('lor vs chance', len(lv), p_chance))

    ax.set_xticks(range(len(SEGMENTS)))
    ax.set_xticklabels([lab for _, lab in SEGMENTS])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Block accuracy (3-way task decoding)')
    ax.set_title('Task decoding by responsiveness segment', pad=38)

    n_all = len(rows)
    fig.text(0.0, -0.04,
             f"{n_all} held-out subjects (one LOOCV fold each); each dot = one subject's block accuracy; lines join the same subject.\n"
             f"Subjects need >= {args.min_blocks} blocks in a segment to be included there. Boundary blocks excluded. "
             f"Diamond + bar: mean, 95% bootstrap CI.\n"
             f"Paired Wilcoxon signed-rank tests. During LOR vs chance (1/3): {fmt_p(p_chance)} (n={len(lv)}). "
             f"Dashed line: chance.",
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

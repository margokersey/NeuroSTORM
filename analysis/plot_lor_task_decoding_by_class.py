#!/usr/bin/env python
"""Plot Test 3 (task decoding during LOR): every subject's per-class accuracy across segments.

One panel per class (tennis / navigation / squeeze; the last is absent for --task tennisnav). In each
panel every held-out subject is a thin line with dots in the class colour (accuracy on that class's
blocks only, i.e. recall), joined across segments (run 1, before LOR, during LOR, after ROR, run 4),
and the thick line is the group mean with a 95% bootstrap CI. A subject is only drawn in a segment if
it has >= --min_blocks blocks there in total (same rule and ROR include list as
analysis/plot_lor_task_decoding.py); a subject with no blocks of a class in a segment has no point for
that class there. Per-class values rest on only a few blocks per subject, so they move in coarse steps.

    python analysis/plot_lor_task_decoding_by_class.py \\
        --pred_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_lor_task_decoding_fold-*/lor_predictions_fold_*.json" \\
        --output_file output/lor_task_decoding/lor_accuracy_by_segment_per_class

Add --label_subjects to write each subject's ID next to its dots.

2-class model: add --task tennisnav and the tennisnav --pred_glob (see plot_lor_task_decoding.py).
Writes .png (600 dpi), .pdf, .svg and a long-format .csv (subject, segment, class, accuracy).
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('lor_plot', HERE / 'plot_lor_task_decoding.py')
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
S, plt = L.S, L.plt

SHORT = {'run1': 'Run 1\n(awake)', 'pre_lor': 'Before\nLOR', 'lor': 'During\nLOR', 'ror': 'After\nROR',
         'recovery': 'Run 4\n(recovery)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred_glob', required=True)
    ap.add_argument('--output_file', default='output/lor_task_decoding/lor_accuracy_by_segment_per_class')
    ap.add_argument('--output_root', default='/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output',
                    help='parent of the 3-way LOOCV run folders (for the run-1 baseline)')
    ap.add_argument('--task', choices=('3way', 'tennisnav'), default='3way')
    ap.add_argument('--stage_b_prefix', default=None)
    ap.add_argument('--run4_glob', default=None)
    ap.add_argument('--no_run1', action='store_true', help='omit the run-1 awake baseline column')
    ap.add_argument('--min_blocks', type=int, default=5, help='min blocks in a segment for a subject to count there')
    ap.add_argument('--label_subjects', action='store_true',
                    help='write each subject\'s ID (e.g. 04) next to its dots; crowded where values coincide')
    ap.add_argument('--exclude_subjects', nargs='*', default=[])
    ap.add_argument('--ror_include', default=str(L.DEFAULT_ROR_INCLUDE))
    ap.add_argument('--no_ror_filter', action='store_true')
    args = ap.parse_args()
    if args.task == 'tennisnav':
        L.CHANCE, L.TASK_LABEL = 1 / 2, '2-class tennis vs navigation'
        L.CLASS_NAMES[:] = L.CLASS_NAMES[:2]
        args.stage_b_prefix = args.stage_b_prefix or 'FIXED_mich_xcpd_nosmooth_tennisnav_tpt_strategy-none_loocv'
    else:
        args.stage_b_prefix = args.stage_b_prefix or L.STAGE_B_PREFIX

    run4_glob = args.run4_glob or args.pred_glob.replace('lor_predictions_fold_', 'lor_predictions_run4_fold_')
    rows = L.load(args.pred_glob, set(args.exclude_subjects), None if args.no_run1 else args.output_root,
                  args.stage_b_prefix, run4_glob)
    if args.no_run1:
        L.SEGMENTS[:] = [x for x in L.SEGMENTS if x[0] != 'run1']
    if not any(r['recovery_n'] for r in rows):
        print(f"no run-4 predictions found ({run4_glob}); omitting the recovery column")
        L.SEGMENTS[:] = [x for x in L.SEGMENTS if x[0] != 'recovery']
    segs = [seg for seg, _ in L.SEGMENTS]
    ror_na = {r['subject']: r['ror_na'] for r in rows}
    by_sub = {r['subject']: r for r in rows}
    in_seg = {seg: {r['subject'] for r in rows if r[f'{seg}_n'] >= args.min_blocks} for seg in segs}
    ror_ok = None if args.no_ror_filter else L.load_ror_include(args.ror_include)
    if ror_ok is not None and 'ror' in in_seg:
        print(f"ROR filter ({args.ror_include}): dropped {sorted(in_seg['ror'] - ror_ok)}")
        in_seg['ror'] &= ror_ok

    classes = list(L.CLASS_NAMES)
    S._style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 11.5, 'xtick.labelsize': 9.5,
                         'ytick.labelsize': 9.5, 'axes.linewidth': 0.9, 'xtick.major.width': 0.9,
                         'ytick.major.width': 0.9})
    fig, axes = plt.subplots(1, len(classes), sharey=True, layout='constrained',
                             figsize=(len(classes) * (0.85 * len(segs) + 1.0) + 0.6, 4.0))
    axes = np.atleast_1d(axes)
    rng = np.random.RandomState(1)
    xs = {seg: i for i, seg in enumerate(segs)}
    long_rows = []
    for ax, name in zip(axes, classes):
        color = L.CLASS_LINE_COLORS[name]
        val = {seg: {s: by_sub[s][f'{seg}_{name}_acc'] for s in in_seg[seg]
                     if not np.isnan(by_sub[s][f'{seg}_{name}_acc'])} for seg in segs}
        if args.label_subjects:   # dots with the same value sit in an even row so each ID label has its own slot
            xpos = {}
            for seg in segs:
                groups = {}
                for s_ in sorted(val[seg]):
                    groups.setdefault(round(val[seg][s_], 4), []).append(s_)
                xpos[seg] = {s_: xs[seg] + (i - (len(g) - 1) / 2) * 0.085
                             for g in groups.values() for i, s_ in enumerate(g)}
        else:
            xpos = {seg: {s: xs[seg] + rng.uniform(-0.08, 0.08) for s in val[seg]} for seg in segs}
        for a, b in zip(segs[:-1], segs[1:]):
            for s in sorted(set(val[a]) & set(val[b])):
                ax.plot([xpos[a][s], xpos[b][s]], [val[a][s], val[b][s]], color=color, lw=0.6, alpha=0.4, zorder=1)
        for seg in segs:
            subs = sorted(val[seg])
            if not subs:
                continue
            v = np.array([val[seg][s] for s in subs])
            px = np.array([xpos[seg][s] for s in subs])
            na = np.array([seg == 'recovery' and ror_na[s] for s in subs], bool)
            ax.scatter(px[~na], v[~na], s=22, color=color, alpha=0.6, edgecolor='white', linewidth=0.4, zorder=3,
                       clip_on=False)
            if na.any():   # open circles: ROR N/A (woken manually before run 4)
                ax.scatter(px[na], v[na], s=24, facecolor='white', edgecolor=color, linewidth=1.0, zorder=3,
                           clip_on=False)
            if args.label_subjects:
                for s, x0, y in zip(subs, px, v):   # vertical IDs above each dot (below it at the top edge)
                    top = y > 0.95
                    ax.annotate(s.replace('sub-', ''), (x0, y), xytext=(0, -4 if top else 4), textcoords='offset points',
                                fontsize=5.5, color=color, rotation=90, ha='center', va='top' if top else 'bottom',
                                zorder=4, clip_on=False)
            for s, y in zip(subs, v):
                long_rows.append(dict(subject=s, segment=seg, cls=name, accuracy=y))
        ms = [seg for seg in segs if val[seg]]
        means = [np.mean(list(val[seg].values())) for seg in ms]
        ax.plot([xs[s] for s in ms], means, color=color, lw=2.4, marker='D', ms=6.5, mec='white', mew=0.6, zorder=6)
        for seg in ms:
            v = np.array(list(val[seg].values()))
            lo, hi = L.boot_ci(v)
            ax.plot([xs[seg]] * 2, [lo, hi], color=color, lw=1.6, zorder=5, solid_capstyle='butt')
            print(f"  {name:<10} {seg:<9} n={len(v):<3} mean={v.mean():.3f}")
        ax.axhline(L.CHANCE, color=S.CHANCE, ls='--', lw=0.8, zorder=0)
        ax.set_title(name.capitalize(), color=color, fontweight='bold', pad=8)
        ax.set_xticks(range(len(segs)))
        ax.set_xticklabels([SHORT[s] for s in segs])
        ax.set_xlim(-0.5, len(segs) - 0.5)
        ax.set_ylim(0, 1.03)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    axes[0].set_ylabel(f'Per-class accuracy ({L.TASK_LABEL})')
    axes[-1].text(len(segs) - 0.55, L.CHANCE + 0.01, 'chance', fontsize=9, color=S.INK_2, ha='right', va='bottom')
    fig.supxlabel("Thin lines and dots: individual subjects. Thick line: group mean, 95% bootstrap CI. Dashed: chance."
                  + (" Open circles: woken manually before run 4." if 'recovery' in segs else ""),
                  fontsize=9, color=S.INK_2)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with open(out.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject', 'segment', 'cls', 'accuracy'])
        w.writeheader()
        w.writerows(long_rows)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg, .csv): {len(rows)} subjects")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Plot Test 3 (task decoding during LOR) for ONE subject: block accuracy by segment, overall and per class.

Same data and styling as analysis/plot_lor_task_decoding.py, but for a single held-out subject
(e.g. P04 / sub-04 / 4), so there is no group mean or statistics: the black line is the subject's
overall block accuracy in each segment and the coloured lines are the per-class accuracy (recall).
The number of blocks in each segment is printed under its tick label. Segments with no blocks are
left out; the after-ROR segment is also left out if the subject is not in the ROR include list
(use --no_ror_filter to keep it). Per-class accuracy rests on only a handful of blocks, so it moves in
coarse steps.

    python analysis/plot_lor_task_decoding_subject.py P04 \\
        --pred_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_lor_task_decoding_fold-*/lor_predictions_fold_*.json" \\
        --output_file output/lor_task_decoding/lor_accuracy_by_segment_P04

2-class model: add --task tennisnav and the tennisnav --pred_glob (see plot_lor_task_decoding.py).
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
_spec = importlib.util.spec_from_file_location('lor_plot', HERE / 'plot_lor_task_decoding.py')
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
S, plt = L.S, L.plt


def norm_subject(s):
    t = s.strip()
    if t.lower().startswith('sub-'):
        t = t[4:]
    return 'sub-' + t.lstrip('PpSsUuBb-').zfill(2)


def find_fold_file(pred_glob, subject):
    for f in sorted(glob(pred_glob)):
        if json.load(open(f))['meta']['subjects'][0] == subject:
            return f
    raise SystemExit(f"no prediction file for {subject} matches {pred_glob}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', help='subject ID: P04, sub-04 or 4')
    ap.add_argument('--pred_glob', required=True)
    ap.add_argument('--output_file', default=None,
                    help='default: output/lor_task_decoding/lor_accuracy_by_segment_<subject>')
    ap.add_argument('--output_root', default='/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output',
                    help='parent of the 3-way LOOCV run folders (for the run-1 baseline)')
    ap.add_argument('--task', choices=('3way', 'tennisnav'), default='3way')
    ap.add_argument('--stage_b_prefix', default=None)
    ap.add_argument('--run4_glob', default=None, help='default: derived from the subject\'s prediction file')
    ap.add_argument('--no_run1', action='store_true', help='omit the run-1 awake baseline column')
    ap.add_argument('--ror_include', default=str(L.DEFAULT_ROR_INCLUDE))
    ap.add_argument('--no_ror_filter', action='store_true')
    args = ap.parse_args()

    sub = norm_subject(args.subject)
    if args.task == 'tennisnav':
        L.CHANCE, L.TASK_LABEL = 1 / 2, '2-class tennis vs navigation'
        L.CLASS_NAMES[:] = L.CLASS_NAMES[:2]
        args.stage_b_prefix = args.stage_b_prefix or 'FIXED_mich_xcpd_nosmooth_tennisnav_tpt_strategy-none_loocv'
    else:
        args.stage_b_prefix = args.stage_b_prefix or L.STAGE_B_PREFIX
    out = Path(args.output_file or f'output/lor_task_decoding/lor_accuracy_by_segment_{sub}')

    fold_file = find_fold_file(args.pred_glob, sub)
    run4_file = args.run4_glob or fold_file.replace('lor_predictions_fold_', 'lor_predictions_run4_fold_')
    rec = L.load(fold_file, set(), None if args.no_run1 else args.output_root, args.stage_b_prefix, run4_file)[0]

    ror_ok = None if args.no_ror_filter else L.load_ror_include(args.ror_include)
    if args.no_run1:
        L.SEGMENTS[:] = [x for x in L.SEGMENTS if x[0] != 'run1']
    segs = []
    for seg, label in L.SEGMENTS:
        if not rec[f'{seg}_n']:
            print(f"  {seg}: no blocks; column omitted")
        elif seg == 'ror' and ror_ok is not None and sub not in ror_ok:
            print(f"  ror: {sub} is not in the ROR include list ({args.ror_include}); column omitted "
                  f"(--no_ror_filter to keep it)")
        else:
            segs.append((seg, label))
    if not segs:
        raise SystemExit(f"{sub}: no segment has any blocks")

    S._style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 11.5, 'xtick.labelsize': 9.5,
                         'ytick.labelsize': 9.5, 'legend.fontsize': 9.5, 'axes.linewidth': 0.9,
                         'xtick.major.width': 0.9, 'ytick.major.width': 0.9})
    fig, ax = plt.subplots(figsize=(1.25 * len(segs) + 2.1, 4.3))
    x = np.arange(len(segs))
    for name in L.CLASS_NAMES:
        y = np.array([rec[f'{seg}_{name}_acc'] for seg, _ in segs])
        ax.plot(x, y, color=L.CLASS_LINE_COLORS[name], lw=1.8, marker='o', ms=5, mec='white', mew=0.5,
                zorder=4, label=name.capitalize(), clip_on=False)
    y = np.array([rec[f'{seg}_acc'] for seg, _ in segs])
    ax.plot(x, y, color=S.INK, lw=2.2, marker='D', ms=6.5, mec='white', mew=0.5, zorder=6, label='Average',
            clip_on=False)
    for xi, yi in zip(x, y):
        ax.text(xi + 0.1, yi, f'{yi:.2f}', ha='left', va='center', fontsize=9.5, fontweight='bold', zorder=7,
                clip_on=False, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.axhline(L.CHANCE, color=S.CHANCE, ls='--', lw=0.8, zorder=0)
    ax.text(len(segs) - 0.55, L.CHANCE + 0.01, 'chance', fontsize=9, color=S.INK_2, ha='right', va='bottom')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, title='Accuracy', title_fontsize=9.5,
              handlelength=1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nn = {rec[f'{seg}_n']} blocks" for seg, lab in segs])
    ax.set_xlim(-0.5, len(segs) - 0.5)
    ax.set_ylim(0, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(f'Block accuracy ({L.TASK_LABEL} task decoding)')
    ax.set_title(f'Task decoding by responsiveness segment: {sub.replace("sub-", "P")}', pad=12)
    if 'recovery' in dict(segs) and rec['ror_na']:
        ax.set_xlabel('Run 4: subject woken manually before run 4.', fontsize=9, color=S.INK_2, loc='left', labelpad=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with open(out.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader()
        w.writerow(rec)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg, .csv) for {sub}")
    for seg, _ in segs:
        per = ', '.join(f"{n} {rec[f'{seg}_{n}_acc']:.2f}" for n in L.CLASS_NAMES)
        print(f"  {seg}: n={rec[f'{seg}_n']} blocks, acc={rec[f'{seg}_acc']:.3f} ({per})")


if __name__ == '__main__':
    main()

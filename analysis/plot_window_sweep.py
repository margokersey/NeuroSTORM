#!/usr/bin/env python
"""Plot LOSO accuracy vs window length (one line per shift) from analysis/window_sweep_cached.py.

  python analysis/plot_window_sweep.py --csv output/window_sweep/window_sweep_loso.csv \
      --out output/window_sweep/window_sweep_accuracy.png

Error bars are the SEM across held-out subjects. Dashed line = chance (1/3).
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

TR = 0.8
# shift is ordinal -> one hue, light to dark (shift 0 lightest)
SHIFT_COLORS = ['#9ecae1', '#4292c6', '#08519c', '#08306b']
MARKERS = ['o', 's', '^', 'D']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default='output/window_sweep/window_sweep_loso.csv')
    ap.add_argument('--out', default='output/window_sweep/window_sweep_accuracy.png')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for i, (sh, g) in enumerate(sorted(df.groupby('shift'))):
        g = g.sort_values('window')
        ax.errorbar(g['window'], g['loso_acc'], yerr=g['acc_subj_sem'], color=SHIFT_COLORS[i % 4],
                    marker=MARKERS[i % 4], ms=7, mec='white', mew=1.2, lw=2, capsize=3, elinewidth=1,
                    label=f'start +{sh} TR ({sh * TR:.1f} s)')
    ax.axhline(1 / 3, color='#6b7280', ls='--', lw=1)
    ax.text(df['window'].max(), 1 / 3 + 0.012, 'chance', ha='right', va='bottom', color='#6b7280', fontsize=9)
    ax.set_xticks(sorted(df['window'].unique()))
    ax.set_xlabel('Window length (TR; 1 TR = 0.8 s)   [13 = imagery only, 20 = imagery + squeeze, 37 = full block]',
                  fontsize=8.5)
    ax.set_ylabel('Leave-one-subject-out accuracy')
    ax.set_ylim(0.3, 0.9)
    ax.set_title('3-way imagery decoding (linear probe): accuracy vs window length', fontsize=11, loc='left')
    ax.grid(axis='y', color='#e5e7eb', lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.08), title='Window start after block onset', title_fontsize=9)
    fig.savefig(args.out, dpi=200)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Figure: which squeeze blocks did each participant miss, in each imagery run.

Reads the CSV from analysis/classify_squeeze_force.py. One panel per run (1-4); one row per
subject, one cell per squeeze block in time order (10 blocks in runs 1 and 4, 20 in runs 2 and 3).
Colour = what the force trace did: squeezed (green), early squeeze during imagery (orange),
missed (red), no data (grey). A white dot marks a weak squeeze. The right-hand margin gives each
subject's number of missed blocks per run.

    python analysis/plot_missed_squeezes.py --out output/squeeze_force/missed_squeezes_by_run
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
COLORS = {'squeezed': '#55A868', 'early': '#E69F00', 'missed': '#C44E52', 'no_data': '#d9d9d9'}
RUN_TITLES = {1: 'Run 1', 2: 'Run 2', 3: 'Run 3', 4: 'Run 4'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(REPO / 'output/squeeze_force/squeeze_block_response.csv'))
    ap.add_argument('--out', default=str(REPO / 'output/squeeze_force/missed_squeezes_by_run'))
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    subs = sorted(df.subject.unique())
    runs = sorted(df.run.unique())
    nblk = {r: int(df[df.run == r].squeeze_block.max()) for r in runs}
    fig, axes = plt.subplots(1, len(runs), figsize=(17, 9), sharey=True,
                             gridspec_kw={'width_ratios': [nblk[r] + 4 for r in runs], 'wspace': 0.06})
    for ax, r in zip(axes, runs):
        d = df[df.run == r]
        for i, s in enumerate(subs):
            ds = d[d.subject == s].set_index('squeeze_block')
            for k in range(1, nblk[r] + 1):
                if k not in ds.index:
                    cat, weak = 'no_data', False
                else:
                    cat, weak = ds.loc[k, 'category'], bool(ds.loc[k, 'weak'])
                ax.add_patch(Rectangle((k - 0.5, i - 0.5), 0.92, 0.88, color=COLORS[cat], lw=0))
                if weak:
                    ax.plot(k, i, 'o', ms=3.5, color='white', mec='none')
            nm = int((ds.category == 'missed').sum()) if len(ds) else 0
            ax.text(nblk[r] + 0.9, i, f'{nm}/{nblk[r]}', va='center', ha='left', fontsize=8,
                    color=COLORS['missed'] if nm else '#888888')
        n_missed = int((d.category == 'missed').sum())
        n_tot = int((d.category != 'no_data').sum())
        ax.set_title(f'{RUN_TITLES[r]}\n{n_missed}/{n_tot} blocks missed ({100 * n_missed / n_tot:.0f}%)', fontsize=11)
        ax.set_xlim(0.5, nblk[r] + 3.2)
        ax.set_ylim(len(subs) - 0.5, -0.5)
        ax.set_xticks(range(1, nblk[r] + 1, 1 if nblk[r] <= 10 else 2))
        ax.set_xlabel('Squeeze block (in time order)')
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(left=False)
    axes[0].set_yticks(range(len(subs)))
    axes[0].set_yticklabels(subs, fontsize=9)
    handles = [Patch(color=COLORS['squeezed'], label='Squeezed at the cue'),
               Patch(color=COLORS['early'], label='Early squeeze (during imagery)'),
               Patch(color=COLORS['missed'], label='Missed (no squeeze)'),
               Patch(color=COLORS['no_data'], label='No force data'),
               Line2D([], [], marker='o', ls='', color='#777777', ms=5, label='White dot = weak squeeze')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=10)
    fig.suptitle('Missed squeezes per participant and imagery run', fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), dpi=200 if ext == 'png' else None)
    print(f'saved {out}.png/.pdf')


if __name__ == '__main__':
    main()

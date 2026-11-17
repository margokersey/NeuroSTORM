#!/usr/bin/env python
"""Plot the classical-decoder benchmark against NeuroSTORM, accuracy by propofol state.

Reads summary.csv from analysis/benchmark_decoders.py. One panel per normalisation (default raw |
run_z), one line per model, NeuroSTORM in bold black with +/- 1 SEM across subjects, dashed chance line.

    python analysis/plot_benchmark_decoders.py                      # beta features, raw + run_z
    python analysis/plot_benchmark_decoders.py --feature winmean    # like-for-like window features
    python analysis/plot_benchmark_decoders.py --norms raw run_z spatial_z

Output: output/benchmark_decoders/benchmark_vs_neurostorm_<feature>.png (gitignored).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
STATES = [('run1', 'Run 1\n(awake)'), ('pre_lor', 'Pre-LOR\n(propofol,\nresponsive)'), ('lor', 'LOR'),
          ('ror', 'ROR'), ('recovery', 'Run 4\n(recovery)')]
# fixed model order -> categorical slots 1-5 of the reference palette (light mode)
MODELS = [('template', 'GLM template matching', '#2a78d6'), ('logreg', 'Logistic regression', '#eb6834'),
          ('linsvm', 'Linear SVM', '#1baf7a'), ('lda', 'PCA + LDA', '#eda100'), ('rf', 'Random forest', '#e87ba4')]
NORM_TITLES = {'raw': 'Raw features', 'run_z': 'Z-scored within each scan (run_z)', 'spatial_z': 'Spatially z-scored per block'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', default=str(REPO / 'output' / 'benchmark_decoders' / 'summary.csv'))
    ap.add_argument('--feature', default='beta', choices=['beta', 'winmean'])
    ap.add_argument('--norms', nargs='+', default=['raw', 'run_z'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    ns = df[df['model'] == 'NeuroSTORM'].set_index('state')
    if ns.empty:
        print('note: no NeuroSTORM rows in summary.csv (re-run benchmark_decoders.py with --pred_glob); plotting baselines only')
    xs = {s: i for i, (s, _) in enumerate(STATES)}

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(1, len(args.norms), figsize=(5.6 * len(args.norms), 4.6), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(args.norms) == 1 else list(axes)
    for ax, norm in zip(axes, args.norms):
        ax.set_facecolor(SURFACE)
        ax.axhline(1 / 3, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(len(STATES) - 0.55, 1 / 3 + 0.012, 'chance', color=MUTED, ha='right', va='bottom', fontsize=8.5)
        for key, label, color in MODELS:
            g = df[(df['model'] == key) & (df['feature'] == args.feature) & (df['norm'] == norm)].set_index('state')
            st = [s for s, _ in STATES if s in g.index]
            if not st:
                continue
            ax.plot([xs[s] for s in st], g.loc[st, 'acc_mean'], color=color, lw=1.8, marker='o', ms=6,
                    mec=SURFACE, mew=1.5, label=label, zorder=3)
        if not ns.empty:
            st = [s for s, _ in STATES if s in ns.index]
            ax.errorbar([xs[s] for s in st], ns.loc[st, 'acc_mean'], yerr=ns.loc[st, 'acc_sem'], color=INK, lw=3,
                        marker='o', ms=8, mec=SURFACE, mew=1.5, capsize=3, elinewidth=1.4, label='NeuroSTORM', zorder=5)
        ax.set_xticks(range(len(STATES)))
        ax.set_xticklabels([l for _, l in STATES], fontsize=9)
        ax.set_xlim(-0.35, len(STATES) - 0.65)
        ax.set_ylim(0.2, 1.04)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(NORM_TITLES.get(norm, norm), fontsize=10.5, color=INK, loc='left', pad=8)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel('Accuracy (mean across held-out subjects)')
    h, l = axes[0].get_legend_handles_labels()
    order = ([i for i, x in enumerate(l) if x == 'NeuroSTORM'] + [i for i, x in enumerate(l) if x != 'NeuroSTORM'])
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=len(order), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.4)
    feat = 'GLM single-trial betas' if args.feature == 'beta' else '32-frame window means'
    fig.suptitle(f'3-way imagery decoding (tennis / navigation / squeeze) across propofol states  -  {feat}',
                 x=0.01, ha='left', fontsize=11.5, color=INK)
    fig.text(0.01, 0.905, 'All models trained on awake run 1 and tested on held-out subjects (leave-one-subject-out).',
             fontsize=9, color=MUTED, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    out = Path(args.out or REPO / 'output' / 'benchmark_decoders' / f'benchmark_vs_neurostorm_{args.feature}.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

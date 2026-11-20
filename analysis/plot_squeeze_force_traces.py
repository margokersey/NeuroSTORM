#!/usr/bin/env python
"""Per-subject force traces with every squeeze block labelled, and the 3-way model's call underneath.

One figure per subject, one row per imagery run (1-4). Each row has
  top   : the force line (relative to that run's resting level). Each squeeze block (imagine + cue) is
          shaded: green = squeezed, red = missed (flat line), orange = early squeeze (during imagery).
          A "WEAK" tag marks a weak squeeze, "MISSED" tags a missed one.
  below : one box per squeeze block, aligned under the shading: green tick = the 3-way model classified
          that block correctly (as squeeze), red cross = wrong (the predicted class is written in it),
          grey = no prediction.

Inputs: output/squeeze_force/squeeze_block_response.csv (analysis/classify_squeeze_force.py) and, for the
model strip, the per-block CSV written by analysis/plot_squeeze_missed_vs_model.py (--model_csv). Without
--model_csv the strip is omitted and only the force/missed labelling is drawn.

    python analysis/plot_squeeze_force_traces.py \\
        --model_csv output/squeeze_force/squeeze_missed_vs_model.csv
Writes <out_dir>/<subject>.png per subject and <out_dir>/all_subjects.pdf (one page per subject).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TR = 0.8
CAT_COLOR = {'squeezed': '#55A868', 'missed': '#C44E52', 'early': '#E69F00'}
PRED_NAME = {0: 'tennis', 1: 'nav', 2: 'squeeze'}
IMAGERY, CUE = 13, 7   # TR per squeeze block


def load_lor_ror(path):
    """{subject: (LOR TR in run 2 or None, ROR TR in run 3 or None)} from LOR_ROR_timing.csv."""
    out = {}
    if path and Path(path).exists():
        d = pd.read_csv(path, encoding='utf-8-sig')
        for _, r in d.iterrows():
            v = [pd.to_numeric(r.iloc[1], errors='coerce'), pd.to_numeric(r.iloc[2], errors='coerce')]
            out[str(r.iloc[0]).strip()] = tuple(None if pd.isna(x) else float(x) for x in v)
    return out


def draw_subject(sub, blocks, model, force_dir, has_model, lor_ror=None):
    runs = [r for r in (1, 2, 3, 4) if (force_dir / sub / f'{sub}_task{r}_cut.1D').exists()
            and (force_dir / sub / f'{sub}_task{r}_cut.1D').stat().st_size > 0]
    h = [3, 0.9, 0.95] if has_model else [3, 0.7]   # force, model strip, spacer
    fig = plt.figure(figsize=(17, 4.4 * len(runs)))
    gs = fig.add_gridspec(len(runs) * len(h), 1, height_ratios=h * len(runs), hspace=0.08)
    xmax = 2270 * TR / 60
    for i, r in enumerate(runs):
        x = np.loadtxt(force_dir / sub / f'{sub}_task{r}_cut.1D')
        y = x - np.median(x)
        t = np.arange(len(y)) * TR / 60
        ax = fig.add_subplot(gs[i * len(h)])
        axm = fig.add_subplot(gs[i * len(h) + 1], sharex=ax) if has_model else None
        ax.plot(t, y, color='k', lw=0.8, zorder=3)
        # same y-range in every run of a subject, scaled to their own squeeze size (so a missed squeeze
        # is a visibly flat line, not magnified drift)
        ref = float(blocks.ref_amp.iloc[0])
        lo, hi = min(float(y.min()), -0.25 * ref), max(float(y.max()), 0.8 * ref)
        ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.18 * (hi - lo))
        b = blocks[(blocks.run == r)]
        for _, row in b.iterrows():
            if row.category == 'no_data':
                continue
            s, c = row.imagery_start_tr * TR / 60, row.action_start_tr * TR / 60
            e = (row.action_start_tr + CUE) * TR / 60
            col = CAT_COLOR[row.category]
            ax.axvspan(s, e, color=col, alpha=0.22, lw=0, zorder=1)
            ax.axvline(c, color=col, lw=0.6, ls=':', zorder=2)
            tag = {'missed': 'MISSED', 'early': 'EARLY', 'squeezed': ''}[row.category]
            if row.weak:
                tag = (tag + ' ' if tag else '') + 'WEAK'
            if tag:
                ax.text((s + e) / 2, hi + 0.04 * (hi - lo + 1e-9), tag, ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color=col if row.category != 'squeezed' else '#8a6d00',
                        rotation=90 if len(b) > 12 else 0, clip_on=False)
            if axm is not None:
                m = model[(model.run == r) & (model.squeeze_block == row.squeeze_block)]
                xc = (s + e) / 2
                if len(m):
                    ok = bool(m.correct.iloc[0])
                    axm.plot(xc, 0.62, 'o', ms=15, color='#55A868' if ok else '#C44E52', zorder=3)
                    axm.text(xc, 0.62, '✓' if ok else '✗', ha='center', va='center', color='white',
                             fontsize=11, fontweight='bold', zorder=4)
                    if not ok:
                        axm.text(xc, 0.1, PRED_NAME.get(int(m.pred.iloc[0]), '?'), ha='center', va='center',
                                 fontsize=7, color='#C44E52')
                else:
                    axm.plot(xc, 0.62, 'o', ms=15, color='#d9d9d9', zorder=3)
        # LOR (loss of responsiveness, in run 2) and ROR (return, in run 3), TR -> minutes
        mark = {2: (lor_ror or (None, None))[0], 3: (lor_ror or (None, None))[1]}.get(r)
        if mark is not None:
            lab, c = ('LOR', '#7b1fa2') if r == 2 else ('ROR', '#1565c0')
            for a_ in [ax] + ([axm] if axm is not None else []):
                a_.axvline(mark * TR / 60, color=c, lw=2, ls='--', zorder=5)
            ax.text(mark * TR / 60, ax.get_ylim()[1], f' {lab}', color=c, fontsize=10, fontweight='bold',
                    va='top', ha='left', zorder=6, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
        ax.set_xlim(0, xmax)
        ax.set_ylabel(f'Run {r}\nforce (vs rest)', fontsize=9)
        nm = int((b.category == 'missed').sum())
        ax.set_title(f'Run {r}: {nm}/{len(b)} squeeze blocks missed', fontsize=11, fontweight='bold', loc='left', pad=22)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if axm is not None:
            axm.set_ylim(0, 1)
            axm.set_yticks([])
            axm.set_ylabel('3-way\nmodel', fontsize=8)
            axm.axhline(0.62, color='#e5e5e5', lw=0.6, zorder=0)
            for sp in axm.spines.values():
                sp.set_visible(False)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(bottom=False)
            if i < len(runs) - 1:
                axm.tick_params(labelbottom=False)
        elif i < len(runs) - 1:
            ax.tick_params(labelbottom=False)
    last = fig.axes[-1]
    last.set_xlabel('Time in run (minutes)')
    handles = [Patch(color=CAT_COLOR['squeezed'], alpha=0.4, label='Squeezed at the cue'),
               Patch(color=CAT_COLOR['missed'], alpha=0.4, label='Missed (flat line)'),
               Patch(color=CAT_COLOR['early'], alpha=0.4, label='Early squeeze (during imagery)'),
               Line2D([], [], color='#7b1fa2', lw=2, ls='--', label='LOR (run 2)'),
               Line2D([], [], color='#1565c0', lw=2, ls='--', label='ROR (run 3)'),
               Patch(color='none', label='WEAK tag = small squeeze (still counts as squeezed)')]
    if has_model:
        handles += [Line2D([], [], marker='o', ms=9, ls='', color='#55A868', label='Model correct (said squeeze)'),
                    Line2D([], [], marker='o', ms=9, ls='', color='#C44E52', label='Model wrong (predicted class below)')]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 4), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f'{sub}: squeeze force and 3-way model result on each squeeze block', y=1.005, fontsize=13)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(REPO / 'output/squeeze_force/squeeze_block_response.csv'))
    ap.add_argument('--model_csv', default=None, help='per-block CSV from plot_squeeze_missed_vs_model.py')
    ap.add_argument('--force_dir', default=str(REPO / 'data/squeeze_force'))
    ap.add_argument('--lor_csv', default=str(REPO / 'data/lor_ror/LOR_ROR_timing.csv'))
    ap.add_argument('--out_dir', default=str(REPO / 'output/squeeze_force/traces'))
    ap.add_argument('--subjects', nargs='*', default=None)
    a = ap.parse_args()

    blocks = pd.read_csv(a.csv)
    model = pd.read_csv(a.model_csv) if a.model_csv else None
    lor = load_lor_ror(a.lor_csv)
    subs = a.subjects or sorted(blocks.subject.unique())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / 'all_subjects.pdf') as pdf:
        for s in subs:
            m = model[model.subject == s] if model is not None else None
            fig = draw_subject(s, blocks[blocks.subject == s], m, Path(a.force_dir), model is not None, lor.get(s))
            fig.savefig(out / f'{s}.png', dpi=110, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print('drew', s)
    print(f'saved to {out}')


if __name__ == '__main__':
    main()

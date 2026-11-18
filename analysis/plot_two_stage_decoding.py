#!/usr/bin/env python
"""Figure for the two-stage decoding test (Task vs. No Task, then which task).

Re-runnable at any point: it plots whatever folds exist. Panels whose data are
missing are skipped, so a Stage-A-only run (no Stage B folds yet, no permutations)
still gives a figure; re-run with the same command later to fill in the rest.

  a  Stage summary: Stage A bal. acc (with permuted range if available), Stage B acc | called
     Task, end-to-end 4-way acc, each with its chance level (one dot per held-out subject,
     mean +/- SEM)
  b  Stage A confusion matrix (pooled over subjects)
  c  Stage A permutation null (needs --stage_a_perm_glob)
  d  With --stage_b_pred_glob (Stage B run on all Stage A blocks, incl. mock):
     Stage B predicted-class mix for real vs mock blocks, and
  e  Stage B confidence per block and per held-out subject, by block type x
     Stage A call (real/mock x called Task / No Task)
(Letters shift if optional panels are absent.)

With --stage_b_pred_glob a second, branching figure (<output_file>_branching.*) is also
written: all blocks -> Stage A (Task vs. No Task, 2x2 confusion) -> blocks called Task
split into real task blocks (Stage B confusion matrix) and mock blocks (Stage B
predicted-class mix) -> blocks called No Task. Every arrow/box shows count and %.

Styling is imported from scripts/analyze_loocv_permutation_results.py so the
figure matches the 3-way one.

Usage (repo root):
    python analysis/plot_two_stage_decoding.py \\
      --stage_a_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*" \\
      --stage_b_pred_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_stageB_allblocks_fold-*" \\
      --tsv "$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv" \\
      --stage_a_perm_glob "/nfs/turbo/umms-huanglab/mkersey/neurostorm_output/output/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*_perm*" \\
      --output_file output/two_stage/two_stage_figure

--stage_b_pred_glob / --tsv / --stage_a_perm_glob are optional (--stage_b_glob is the older lookup mode). Writes .png/.pdf/.svg.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import two_stage_decoding as tsd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'perm_style', HERE.parent / 'scripts' / 'analyze_loocv_permutation_results.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
plt, Line2D = S.plt, S.Line2D

E2E_CHANCE = 0.5   # majority-class baseline (half of the blocks are No Task); see two_stage_decoding.py


def _balacc(preds):
    return tsd.stage_a_metrics(preds)['balanced_accuracy']


def _top_legend(ax, handles):
    S._top_legend(ax, handles, fontsize=6.5)


def _single_confusion(ax, fig, cm, rows, cols, title, colorbar=True):
    cm = np.asarray(cm, float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = S.LinearSegmentedColormap.from_list('seq_blue', S.SEQ_BLUE)
    im = ax.pcolormesh(cmn, cmap=cmap, vmin=0, vmax=1, edgecolors='white', linewidth=1.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cmn[i, j]
            ax.text(j + 0.5, i + 0.5, f"{100 * v:.0f}%\n({int(cm[i, j])})", ha='center', va='center',
                    fontsize=6.5, color='white' if v > 0.55 else S.INK,
                    fontweight='bold' if i == j else 'normal', linespacing=1.15)
    ax.set_xlim(0, cm.shape[1]); ax.set_ylim(cm.shape[0], 0)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_xticklabels(cols, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_yticklabels(rows, va='center')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_title(title, loc='left', pad=S.TITLE_PAD)
    ax.set_anchor('N')
    if not colorbar:
        return
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1], panchor=False)
    cb.ax.set_yticklabels(['0%', '50%', '100%'])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)


def plot_stages(ax, rows, rng):
    """rows: [(label, per-subject values, chance, null values or None)] drawn top-to-bottom."""
    n = len(rows)
    for k, (label, vals, chance, null) in enumerate(rows):
        yi = n - 1 - k
        vals = np.asarray(vals, float)
        vals = vals[~np.isnan(vals)]
        if null is not None and len(null):
            ax.plot([np.min(null), np.max(null)], [yi, yi], color=S.PERM, alpha=0.7, lw=6,
                    solid_capstyle='butt', zorder=1)
        ax.plot([chance] * 2, [yi - 0.42, yi + 0.42], color=S.CHANCE, lw=1.0, ls=(0, (2, 1.5)), zorder=0)
        ax.scatter(vals, yi - 0.22 + rng.uniform(-0.06, 0.06, len(vals)), s=9, color=S.REAL,
                   alpha=0.7, edgecolor='white', linewidth=0.3, zorder=3)
        m = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ax.plot([m - sem, m + sem], [yi + 0.08] * 2, color=S.REAL, lw=1.4, zorder=4)
        ax.scatter([m], [yi + 0.08], s=26, color=S.REAL, edgecolor='white', linewidth=0.6, zorder=5)
        ax.text(0.99, yi + 0.36, f"{m:.3f}", transform=ax.get_yaxis_transform(), ha='right',
                va='center', fontsize=6.5, color=S.INK_2)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.6, n - 0.35)
    ax.set_xlabel('Score per held-out subject')
    ax.set_title('Stage summary (one dot per subject)', loc='left', pad=S.TITLE_PAD)
    handles = [Line2D([], [], marker='o', color=S.REAL, lw=1.4, markersize=4.5, label='Mean ± SEM'),
               Line2D([], [], color=S.CHANCE, lw=1.0, ls=(0, (2, 1.5)), label='Chance / baseline')]
    if any(r[3] is not None for r in rows):
        handles.insert(1, Line2D([], [], color=S.PERM, alpha=0.7, lw=6, label='Permuted (range)'))
    _top_legend(ax, handles)


def plot_null(ax, real_mean, null_means):
    null = np.asarray(null_means)
    n = len(null)
    p = (1 + (null >= real_mean).sum()) / (1 + n)
    lo, hi = min(null.min(), real_mean, 0.5), max(null.max(), real_mean)
    pad = 0.08 * (hi - lo + 1e-6)
    if n >= 10:
        ax.hist(null, bins=np.linspace(lo - pad, hi + pad, 30), color=S.PERM, alpha=0.8,
                edgecolor='white', linewidth=0.5, zorder=1)
        ax.set_ylabel('Permutations')
    else:
        ax.scatter(null, np.random.default_rng(0).uniform(-0.25, 0.25, n), s=22, color=S.PERM,
                   edgecolor='white', linewidth=0.5, zorder=2)
        ax.set_ylim(-1, 1.6); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.axvline(0.5, color=S.CHANCE, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.axvline(real_mean, color=S.REAL, lw=1.8, zorder=3)
    right_side = real_mean < hi - 0.3 * (hi - lo)
    ax.annotate(f"Real = {real_mean:.3f}\n{S._fmt_p(p, n)}", xy=(real_mean, ax.get_ylim()[1]),
                xytext=(4 if right_side else -4, -2), textcoords='offset points',
                ha='left' if right_side else 'right', va='top', fontsize=7, color=S.INK, linespacing=1.3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel('Mean accuracy (Stage A)')
    ax.set_title(f'Permutation null (n = {n})', loc='left', pad=S.TITLE_PAD)


def plot_class_mix(ax, per_fold):
    """Stage B predicted-class mix (100% stacked bars): real task blocks vs mock blocks."""
    rows = [('Real task\n(all blocks)', lambda r: r['block_type'] == 'task'),
            ('Mock\n(all blocks)', lambda r: r['block_type'] == 'mock'),
            ('Mock called\nTask by A', lambda r: r['group'] == 'mock_called_task')]
    nb = tsd.N_B
    colors = [S.SEQ_BLUE[2], S.SEQ_BLUE[4], S.SEQ_BLUE[6]] if nb == 3 else [S.SEQ_BLUE[2], S.SEQ_BLUE[5]]
    names = [tsd.STAGE_B_NAMES[k] for k in range(nb)]
    n = len(rows)
    for k, (lab, sel) in enumerate(rows):
        yi = n - 1 - k
        recs = [r for fr in per_fold for r in fr['records'] if sel(r)]
        tot = max(len(recs), 1)
        left = 0.0
        for c in range(nb):
            frac = sum(r['stage_b_pred'] == c for r in recs) / tot
            ax.barh(yi, frac, left=left, color=colors[c], edgecolor='white', linewidth=0.8, height=0.62)
            if frac > 0.08:
                ax.text(left + frac / 2, yi, f"{100 * frac:.0f}%", ha='center', va='center', fontsize=6.5,
                        color='white' if c >= 1 else S.INK)
            left += frac
        ax.text(1.01, yi, f"n={len(recs)}", va='center', ha='left', fontsize=6.5, color=S.INK_2)
    ax.set_yticks(np.arange(n)[::-1]); ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Fraction of blocks predicted as each class\n(a prediction mix, not accuracy)')
    ax.set_title('Stage B predicted-class mix', loc='left', pad=S.TITLE_PAD)
    _top_legend(ax, [Line2D([], [], marker='s', ls='', color=colors[c], markersize=5, label=names[c])
                     for c in range(nb)])


def plot_confidence(ax, per_fold, measure='logit_margin'):
    """Stage B confidence per block (small faint dots), with each subject's mean over the group's
    blocks overlaid (larger dots) and the mean of those subject means as a black bar. x = block type
    (real task / mock task) crossed with Stage A's call."""
    groups = [('real_called_task', 'Task\n(hit)'), ('real_missed', 'No Task\n(miss)'),
              ('mock_rejected', 'No Task\n(correct\nreject)'), ('mock_called_task', 'Task\n(false\nalarm)')]
    means = tsd.subject_group_means(per_fold, measure)
    jitter = np.random.default_rng(0)
    colors = [S.REAL, S.REAL, S.PERM, S.PERM]
    ns = []
    nb = []
    for k, (g, lab) in enumerate(groups):
        blocks = np.array([float(r[measure]) for fr in per_fold for r in fr['records']
                           if r['group'] == g and r[measure] is not None])
        vals = np.array([d[g] for d in means.values() if g in d])
        ns.append(len(vals))
        nb.append(len(blocks))
        if len(blocks) == 0:
            continue
        ax.scatter(k + jitter.uniform(-0.3, 0.3, len(blocks)), blocks, s=4, color=colors[k],
                   alpha=0.25, linewidth=0, zorder=2)
        ax.scatter(k + jitter.uniform(-0.1, 0.1, len(vals)), vals, s=14, color=colors[k],
                   alpha=0.95, edgecolor='black', linewidth=0.5, zorder=3)
        ax.plot([k - 0.33, k + 0.33], [vals.mean()] * 2, color=S.INK, lw=1.4, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[1]}\n{nbk} blocks\n{n} subj" for g, n, nbk in zip(groups, ns, nb)], fontsize=6.5)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(tsd.CONF_MEASURES[measure])
    # Second row under the ticks: block type, with a bracket over its two groups.
    tr = ax.get_xaxis_transform()
    for x0, x1, label, c in [(-0.4, 1.4, 'Real task', S.REAL), (1.6, 3.4, 'Mock task', S.PERM)]:
        ax.plot([x0, x1], [-0.285, -0.285], transform=tr, color=c, lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, -0.305, label, transform=tr, ha='center', va='top', fontsize=6.5,
                color=S.INK, fontweight='bold')
    ax.text(0.5, -0.46, "x labels: Stage A's call (and whether it was right).\nFaint dot = one block; outlined dot = one subject's mean; bar = mean of subject means",
            transform=ax.transAxes, ha='center', va='top', fontsize=6, color=S.INK_2, linespacing=1.3)
    ax.set_title('Stage B confidence by block type', loc='left', pad=S.TITLE_PAD)
    _top_legend(ax, [Line2D([], [], color=S.INK, lw=1.2, label='Mean over subjects')])


def _flow_matrix(ax, cm, row_labels, col_labels, title, colors, highlight=(), fs=8.5):
    """Confusion matrix for the branching figure: cell = % of its row (count); row labels carry
    the row total; cells in `highlight` get a heavy outline (the cells that feed the next stage)."""
    cm = np.asarray(cm, float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = S.LinearSegmentedColormap.from_list('flow', colors)
    ax.pcolormesh(cmn, cmap=cmap, vmin=0, vmax=1, edgecolors='white', linewidth=2)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cmn[i, j]
            ax.text(j + 0.5, i + 0.5, f"{100 * v:.0f}%\n({int(cm[i, j])})", ha='center', va='center',
                    fontsize=fs, color='white' if v > 0.55 else S.INK,
                    fontweight='bold' if i == j else 'normal', linespacing=1.2)
    for (i, j) in highlight:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, ec=S.INK, lw=2.4, zorder=5, clip_on=False))
    ax.set_xlim(0, cm.shape[1]); ax.set_ylim(cm.shape[0], 0)
    ax.set_aspect('equal'); ax.set_anchor('NW')
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5); ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_yticklabels([f"{r}\nn = {int(t)}" for r, t in zip(row_labels, cm.sum(1))], fontsize=7.5, va='center')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold', pad=8)


def plot_branching(per_fold, out):
    """Branching-logic figure. Reads left to right:
         all blocks -> Stage A confusion matrix -> (cells 'called Task') -> Stage B matrices
    plus a strip showing where each kind of block ends up end-to-end.

    The two outlined cells of the Stage A matrix (real/mock blocks called Task) are exactly the
    blocks handed to Stage B; arrows run from those cells to their Stage B matrices. Built from
    the per-block records of folds that have Stage B, so all numbers agree across the figure.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle
    nb = tsd.N_B
    names = list(tsd.STAGE_B_NAMES)
    recs = [r for fr in per_fold for r in fr['records']]
    n_all = len(recs)
    real = [r for r in recs if r['block_type'] == 'task']
    mock = [r for r in recs if r['block_type'] == 'mock']
    called = [r for r in recs if r['stage_a_called_task']]
    real_called = [r for r in real if r['stage_a_called_task']]
    mock_called = [r for r in mock if r['stage_a_called_task']]
    n_missed = len(real) - len(real_called)
    n_rej = len(mock) - len(mock_called)

    def pct(k, n):
        return f"{100 * k / n:.0f}%" if n else "n/a"

    cm_a = np.array([[n_rej, len(mock_called)], [n_missed, len(real_called)]])
    cm_b = np.zeros((nb, nb), int)
    for r in real_called:
        cm_b[r['stage_b_true'], r['stage_b_pred']] += 1
    mock_mix = np.array([[sum(r['stage_b_pred'] == c for r in mock_called) for c in range(nb)]])
    b_correct = int(np.trace(cm_b))
    b_acc = b_correct / cm_b.sum() if cm_b.sum() else float('nan')
    bal = np.nanmean([cm_a[1, 1] / max(cm_a[1].sum(), 1), cm_a[0, 0] / max(cm_a[0].sum(), 1)])
    mock_cmap = ['#ffffff', S.PERM]

    fig = plt.figure(figsize=(13.5, 7.8))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis('off')

    def head(x, text, sub):
        bg.text(x, 0.955, text, fontsize=11, fontweight='bold', color=S.INK, ha='left', va='center')
        bg.text(x, 0.925, sub, fontsize=7.5, color=S.INK_2, ha='left', va='center')
        bg.plot([x, x + 0.27], [0.905, 0.905], color=S.INK_2, lw=0.8)

    head(0.02, 'All blocks', 'every held-out block')
    head(0.215, 'Stage A: Task vs. No task', 'run on every block')
    head(0.60, 'Stage B: which task?', 'only blocks Stage A called Task')

    # root
    bg.add_patch(FancyBboxPatch((0.02, 0.50), 0.12, 0.18, boxstyle='round,pad=0.003,rounding_size=0.008',
                                fc='white', ec=S.INK_2, lw=1.0))
    bg.text(0.08, 0.59, f"n = {n_all}\n\nreal task  {len(real)} ({pct(len(real), n_all)})\n"
            f"mock  {len(mock)} ({pct(len(mock), n_all)})", ha='center', va='center', fontsize=8.5,
            color=S.INK, linespacing=1.5)

    # Stage A matrix
    ax_a = fig.add_axes([0.235, 0.34, 0.25, 0.50])
    _flow_matrix(ax_a, cm_a, ['No task\n(mock)', 'Task\n(real)'],
                 [f"called No task\nn = {n_rej + n_missed} ({pct(n_rej + n_missed, n_all)})",
                  f"called Task\nn = {len(called)} ({pct(len(called), n_all)})"],
                 f"balanced acc. = {bal:.3f}", S.SEQ_BLUE, highlight=[(0, 1), (1, 1)])
    ax_a.set_xlabel('Stage A prediction  (outlined cells go on to Stage B;\n"called No task" blocks stop here)',
                    fontsize=7.5, color=S.INK_2)

    # Stage B matrices
    ax_m = fig.add_axes([0.68, 0.68, 0.27, 0.15])
    _flow_matrix(ax_m, mock_mix, ['Mock blocks\ncalled Task'], names,
                 "Mock blocks called Task", mock_cmap)
    ax_m.set_xticks([])
    ax_m.set_xlabel("what Stage B called them: a prediction mix (no true class exists, so not an accuracy)",
                    fontsize=7, color=S.INK_2, labelpad=3)
    ax_b = fig.add_axes([0.68, 0.24, 0.27, 0.34])
    _flow_matrix(ax_b, cm_b, names, names,
                 f"Real task blocks called Task: acc. = {b_acc:.3f}  (chance = {1 / nb:.2f})", S.SEQ_BLUE)
    ax_b.set_xlabel('Stage B prediction', fontsize=8, color=S.INK_2)
    ax_b.set_ylabel('True task', fontsize=8)

    # arrows from the outlined Stage A cells to the Stage B matrices
    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    def cell_edge(ax, col_edge, row_center):
        return inv.transform(ax.transData.transform((col_edge, row_center)))

    def left_mid(ax):
        ax.apply_aspect()
        bb = ax.get_position()
        return bb.x0 - 0.075, bb.y0 + bb.height / 2

    for (ci, cj), ax_t, lab in [((0, 1), ax_m, f"mock called Task\n{len(mock_called)} ({pct(len(mock_called), len(mock))} of mock)"),
                                ((1, 1), ax_b, f"real called Task\n{len(real_called)} ({pct(len(real_called), len(real))} of real)")]:
        x0, y0 = cell_edge(ax_a, 2.0, ci + 0.5)
        x1, y1 = left_mid(ax_t)
        bg.annotate('', xy=(x1, y1), xytext=(x0 + 0.004, y0),
                    arrowprops=dict(arrowstyle='-|>', color=S.INK, lw=1.6, shrinkA=0, shrinkB=0,
                                    connectionstyle='arc3,rad=0'), zorder=1)
        bg.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.018 if ci == 0 else 0.018), lab, fontsize=7.5, color=S.INK,
                ha='center', va='bottom', linespacing=1.3,
                bbox=dict(fc='white', ec='none', pad=1.5), zorder=4)
    bg.annotate('', xy=(0.19, 0.59), xytext=(0.145, 0.59),
                arrowprops=dict(arrowstyle='-|>', color=S.INK, lw=1.6, shrinkA=0, shrinkB=0))

    # bottom strip: end-to-end outcome of each kind of block
    ax_o = fig.add_axes([0.235, 0.045, 0.72, 0.115])
    segs_real = [('Correct end-to-end\n(passed A, right task)', b_correct, S.SEQ_BLUE[6], 'white'),
                 ('Passed A, wrong task\n(Stage B error)', int(cm_b.sum() - b_correct), S.SEQ_BLUE[3], S.INK),
                 ('Missed by Stage A\n(called No task)', n_missed, '#cfd4da', S.INK)]
    segs_mock = [('Correctly rejected', n_rej, S.SEQ_BLUE[6], 'white'),
                 ('False alarm (passed to Stage B)', len(mock_called), '#cfd4da', S.INK)]
    for yi, (lab, n_tot, segs) in enumerate([(f"Real task blocks (n = {len(real)})", len(real), segs_real),
                                              (f"Mock blocks (n = {len(mock)})", len(mock), segs_mock)][::-1]):
        left = 0.0
        for sl, k, col, tc in segs:
            frac = k / max(n_tot, 1)
            ax_o.barh(yi, frac, left=left, color=col, edgecolor='white', linewidth=1.5, height=0.8)
            if frac > 0.07:
                ax_o.text(left + frac / 2, yi, f"{sl}\n{100 * frac:.0f}% ({k})", ha='center', va='center',
                          fontsize=7, color=tc, linespacing=1.25)
            left += frac
        ax_o.text(-0.01, yi, lab, ha='right', va='center', fontsize=8, color=S.INK)
    ax_o.set_xlim(0, 1); ax_o.set_ylim(-0.5, 1.5); ax_o.axis('off')
    ax_o.set_title('Where each kind of block ends up (end-to-end)', loc='left', fontsize=9, fontweight='bold', pad=6)

    n_sub = len({fr['subject'] for fr in per_fold})
    fig.text(0.02, 0.005,
             f"{n_sub} held-out subjects (LOOCV), blocks pooled. Matrix cells: % of the row's true class (count); "
             f"darker = higher %. Stage B is run on every block, but only blocks Stage A called Task are passed on.",
             fontsize=6.8, color=S.INK_2, va='bottom')
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage_a_glob', required=True)
    ap.add_argument('--stage_b_pred_glob', default=None,
                    help='Stage B run on ALL Stage A blocks (predict_stage_b_on_stage_a_blocks.py output dirs)')
    ap.add_argument('--stage_b_glob', default=None, help='older lookup mode (3-way LOOCV run dirs)')
    ap.add_argument('--tsv', default=None, help='timing_onehot_task1.tsv (needed with either Stage B option)')
    ap.add_argument('--stage_a_perm_glob', default=None)
    ap.add_argument('--stage_b_classes', type=int, choices=(2, 3), default=3,
                    help='2 = tennis-vs-navigation Stage B (see two_stage_decoding.py)')
    ap.add_argument('--exclude_subjects', nargs='*', default=['sub-29'])
    ap.add_argument('--output_file', default='output/two_stage/two_stage_figure',
                    help='path stem; .png/.pdf/.svg are written')
    args = ap.parse_args()

    tsd.configure(args.stage_b_classes)
    nb = tsd.N_B
    excl = set(args.exclude_subjects)
    a_runs = tsd.collect(args.stage_a_glob, excl, want_perm=False)
    if not a_runs:
        sys.exit(f"No Stage A folds with predictions matched {args.stage_a_glob}")
    folds = sorted(a_runs)
    a_preds = {f: a_runs[f][0][1] for f in folds}
    real_bal = [_balacc(a_preds[f]) for f in folds]

    # Stage A permutations (complete permutation indices only, same rule as two_stage_decoding.py)
    perm_by_fold, null_means = {}, None
    if args.stage_a_perm_glob:
        perms = tsd.collect(args.stage_a_perm_glob, excl, want_perm=True)
        by_perm = {}
        for f in folds:
            for k, preds in perms.get(f, []):
                by_perm.setdefault(k, {})[f] = preds
        for f in folds:
            vals = [_balacc(v[f]) for v in by_perm.values() if f in v]
            if vals:
                perm_by_fold[f] = vals
        complete = [v for v in by_perm.values() if set(v) == set(folds)]
        if complete:
            null_means = [np.mean([tsd.stage_a_metrics(v[f])['accuracy'] for f in folds]) for v in complete]
        else:
            print("  [warn] no permutation index covers every fold; null panel skipped", file=sys.stderr)

    # Stage B / end-to-end (optional)
    composed = []
    all_blocks = bool(args.stage_b_pred_glob)
    if args.stage_b_pred_glob and args.stage_b_glob:
        print("  [warn] pass only one of --stage_b_pred_glob / --stage_b_glob; skipping Stage B panels", file=sys.stderr)
    elif (args.stage_b_pred_glob or args.stage_b_glob) and not args.tsv:
        print("  [warn] --tsv is required for the Stage B panels; skipping them", file=sys.stderr)
    elif args.stage_b_pred_glob or args.stage_b_glob:
        block_labels = tsd.tsv_block_labels(args.tsv)
        if all_blocks:
            b_runs = tsd.collect_stage_b_all(args.stage_b_pred_glob, excl)
        else:
            b_runs = tsd.collect(args.stage_b_glob, excl, want_perm=False)
        for f in folds:
            if f not in b_runs:
                continue
            try:
                if all_blocks:
                    composed.append(tsd.compose_fold_all(a_preds[f], b_runs[f], block_labels, f))
                else:
                    composed.append(tsd.compose_fold(a_preds[f], b_runs[f][0][1], block_labels, f))
            except ValueError as e:
                print(f"  [warn] {e}", file=sys.stderr)
        missing = [f for f in folds if f not in {c['fold'] for c in composed}]
        if missing:
            print(f"  [note] Stage B unavailable for folds {missing}", file=sys.stderr)

    rng = np.random.default_rng(0)
    stage_rows = [('Stage A\nTask vs. No Task\n(bal. acc.)', real_bal, 0.5,
                   [x for v in perm_by_fold.values() for x in v] or None)]
    if composed:
        stage_rows.append((f'Stage B | called Task\n({nb}-way acc.)',
                           [c['stage_b_called_task_acc'] for c in composed], 1 / nb, None))
        stage_rows.append((f'End-to-end\n({nb + 1}-way acc.)', [c['end_to_end_acc'] for c in composed], E2E_CHANCE, None))

    S._style()
    cm_a = np.sum([np.array(tsd_cm(a_preds[f])) for f in folds], axis=0)
    panels = [
        lambda ax, fig: plot_stages(ax, stage_rows, rng),
        lambda ax, fig: _single_confusion(ax, fig, cm_a, ['No task', 'Task'], ['No task', 'Task'],
                                          'Stage A confusion (pooled)'),
    ]
    if null_means is not None:
        real_acc = float(np.mean([tsd.stage_a_metrics(a_preds[f])['accuracy'] for f in folds]))
        panels.append(lambda ax, fig: plot_null(ax, real_acc, null_means))
    if composed and all_blocks:
        panels.append(lambda ax, fig: plot_class_mix(ax, composed))
        panels.append(lambda ax, fig: plot_confidence(ax, composed))

    ncols = 3 if (len(panels) == 3 or len(panels) >= 5) else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(7.2 if ncols == 2 else 11.0, 3.0 * nrows + 0.3))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.75, left=0.08, right=0.97, top=0.93, bottom=0.08)
    for i, draw in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        draw(ax, fig)
        S._panel_letter(ax, 'abcdefgh'[i])
    n_b = len(composed)
    excl_txt = f" (excluded: {', '.join(sorted(excl))})" if excl else ""
    perm_txt = f"{len(null_means)} Stage A permutations" if null_means is not None else "no permutations run"
    fig.text(0.08, -0.03,
             f"{len(folds)} held-out subjects (one LOOCV fold each){excl_txt}; {n_b} with Stage B; {perm_txt}.\n"
             f"Dashed lines: chance (Stage A 0.5, Stage B 1/{nb}). End-to-end baseline 0.5 = always answering 'No task'.",
             fontsize=6.5, color=S.INK_2, va='top', linespacing=1.4)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg): {len(folds)} Stage A folds, {n_b} with Stage B")
    if composed and all_blocks:
        plot_branching(composed, out.with_name(out.name + '_branching'))
    elif composed:
        print("  [note] branching figure needs --stage_b_pred_glob (Stage B predictions on mock blocks); skipped", file=sys.stderr)


def tsd_cm(preds):
    """2x2 confusion counts [[TN, FP], [FN, TP]] from Stage A predictions."""
    y = np.array([int(p['true_label']) for p in preds])
    yh = np.array([int(p['predicted_label']) for p in preds])
    return [[int(((y == 0) & (yh == 0)).sum()), int(((y == 0) & (yh == 1)).sum())],
            [int(((y == 1) & (yh == 0)).sum()), int(((y == 1) & (yh == 1)).sum())]]


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Figure for the two-stage decoding test (Task vs. No Task, then which task).

Re-runnable at any point: it plots whatever folds exist. Panels whose data are
missing are skipped, so a Stage-A-only run (no Stage B folds yet, no permutations)
still gives a figure; re-run with the same command later to fill in the rest.

  a  Per-subject Stage A balanced accuracy (real dot; permuted range if available)
  b  Stage summary: Stage A bal. acc, Stage B acc | called Task, end-to-end 4-way
     acc, each with its chance level (one dot per held-out subject, mean +/- SEM)
  c  Stage A confusion matrix (pooled over subjects)
  d  Stage A permutation null (needs --stage_a_perm_glob)
  e  Pooled end-to-end confusion matrix (needs Stage B folds)
  f  With --stage_b_pred_glob (Stage B run on all Stage A blocks, incl. mock):
     Stage B predicted-class mix for real vs mock blocks, and
  g  Stage B confidence (logit margin) per held-out subject, by block type x
     Stage A call (real/mock x called Task / No Task)

Styling is imported from scripts/analyze_loocv_permutation_results.py so the
figure matches the 3-way one.

Usage (repo root):
    python analysis/plot_two_stage_decoding.py \\
      --stage_a_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*" \\
      --stage_b_pred_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_stageB_allblocks_fold-*" \\
      --tsv "$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv" \\
      --stage_a_perm_glob "nas/output/neurostorm/FIXED_mich_xcpd_nosmooth_taskvsrest_loocv_fold-*_perm*" \\
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


def _single_confusion(ax, fig, cm, rows, cols, title):
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
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['0%', '50%', '100%'])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)


def plot_per_subject(ax, folds, real, perm_by_fold):
    order = np.argsort(real)
    x = np.arange(len(folds))
    for xi, i in zip(x, order):
        pv = perm_by_fold.get(folds[i])
        if pv is not None and len(pv):
            ax.plot([xi, xi], [np.min(pv), np.max(pv)], color=S.PERM, alpha=0.55, lw=2.6,
                    solid_capstyle='round', zorder=1)
            ax.plot([xi - 0.34, xi + 0.34], [np.mean(pv)] * 2, color=S.PERM_DARK, lw=1.0, zorder=2)
    ax.scatter(x, np.asarray(real)[order], s=20, color=S.REAL, edgecolor='white', linewidth=0.6, zorder=3)
    ax.axhline(0.5, color=S.CHANCE, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.8, max(len(x) - 0.2, 0.8))
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([str(folds[i]) for i in order], rotation=90, fontsize=6)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_xlabel('Held-out subject (fold, sorted)')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Stage A: Task vs. No Task', loc='left', pad=S.TITLE_PAD)
    handles = [Line2D([], [], marker='o', ls='', color=S.REAL, markersize=4.5, label='Real')]
    if perm_by_fold:
        handles.append(Line2D([], [], color=S.PERM, alpha=0.7, lw=2.6, label='Permuted (range)'))
    handles.append(Line2D([], [], color=S.CHANCE, lw=0.8, ls=(0, (3, 2)), label='Chance'))
    _top_legend(ax, handles)


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
        ax.plot([chance] * 2, [yi - 0.3, yi + 0.3], color=S.CHANCE, lw=0.8, ls=(0, (2, 1.5)), zorder=0)
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
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, n - 0.35)
    ax.set_xlabel('Score per held-out subject')
    ax.set_title('Stage summary', loc='left', pad=S.TITLE_PAD)
    handles = [Line2D([], [], marker='o', color=S.REAL, lw=1.4, markersize=4.5, label='Mean ± SEM'),
               Line2D([], [], color=S.CHANCE, lw=0.8, ls=(0, (2, 1.5)), label='Chance')]
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
    rows = [('Real task\n(all)', lambda r: r['block_type'] == 'task'),
            ('Mock\n(all)', lambda r: r['block_type'] == 'mock'),
            ('Mock called\nTask by A', lambda r: r['group'] == 'mock_called_task')]
    colors = [S.SEQ_BLUE[2], S.SEQ_BLUE[4], S.SEQ_BLUE[6]]
    names = [tsd.STAGE_B_NAMES[k] for k in range(3)]
    n = len(rows)
    for k, (lab, sel) in enumerate(rows):
        yi = n - 1 - k
        recs = [r for fr in per_fold for r in fr['records'] if sel(r)]
        tot = max(len(recs), 1)
        left = 0.0
        for c in range(3):
            frac = sum(r['stage_b_pred'] == c for r in recs) / tot
            ax.barh(yi, frac, left=left, color=colors[c], edgecolor='white', linewidth=0.8, height=0.62)
            if frac > 0.08:
                ax.text(left + frac / 2, yi, f"{100 * frac:.0f}%", ha='center', va='center', fontsize=6.5,
                        color='white' if c >= 1 else S.INK)
            left += frac
        ax.text(1.01, yi, f"n={len(recs)}", va='center', ha='left', fontsize=6.5, color=S.INK_2)
    ax.axvline(1 / 3, color=S.CHANCE, lw=0.6, ls=(0, (1, 2)), zorder=0)
    ax.set_yticks(np.arange(n)[::-1]); ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Fraction of blocks (Stage B prediction)')
    ax.set_title('Stage B predicted class', loc='left', pad=S.TITLE_PAD)
    _top_legend(ax, [Line2D([], [], marker='s', ls='', color=colors[c], markersize=5, label=names[c])
                     for c in range(3)])


def plot_confidence(ax, per_fold, measure='logit_margin'):
    """Per-subject mean Stage B confidence. x = block type (real task / mock rest) crossed with
    Stage A's call; one dot per held-out subject (that subject's mean over the group's blocks)."""
    groups = [('real_called_task', 'Task\n(correct)'), ('real_missed', 'No Task\n(missed)'),
              ('mock_rejected', 'No Task\n(correct)'), ('mock_called_task', 'Task\n(false\nalarm)')]
    means = tsd.subject_group_means(per_fold, measure)
    jitter = np.random.default_rng(0)
    colors = [S.REAL, S.REAL, S.PERM, S.PERM]
    ns = []
    for k, (g, lab) in enumerate(groups):
        vals = np.array([d[g] for d in means.values() if g in d])
        ns.append(len(vals))
        if len(vals) == 0:
            continue
        ax.scatter(k + jitter.uniform(-0.12, 0.12, len(vals)), vals, s=10, color=colors[k],
                   alpha=0.75, edgecolor='white', linewidth=0.3, zorder=3)
        ax.plot([k - 0.27, k + 0.27], [vals.mean()] * 2, color=S.INK, lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[1]}\nn={n}" for g, n in zip(groups, ns)], fontsize=6.5)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(tsd.CONF_MEASURES[measure])
    # Second row under the ticks: block type, with a bracket over its two groups.
    tr = ax.get_xaxis_transform()
    for x0, x1, label, c in [(-0.4, 1.4, 'Real task blocks', S.REAL), (1.6, 3.4, 'Mock rest blocks', S.PERM)]:
        ax.plot([x0, x1], [-0.285, -0.285], transform=tr, color=c, lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, -0.305, label, transform=tr, ha='center', va='top', fontsize=6.5,
                color=S.INK, fontweight='bold')
    ax.text(0.5, -0.43, "x labels: Stage A's call. Dot = one subject's mean;\nn = subjects with at least 1 block in the group",
            transform=ax.transAxes, ha='center', va='top', fontsize=6, color=S.INK_2, linespacing=1.3)
    ax.set_title('Stage B confidence', loc='left', pad=S.TITLE_PAD)
    _top_legend(ax, [Line2D([], [], marker='o', ls='', color=S.REAL, markersize=4.5, label='Real task'),
                     Line2D([], [], marker='o', ls='', color=S.PERM, markersize=4.5, label='Mock'),
                     Line2D([], [], color=S.INK, lw=1.2, label='Mean of subjects')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage_a_glob', required=True)
    ap.add_argument('--stage_b_pred_glob', default=None,
                    help='Stage B run on ALL Stage A blocks (predict_stage_b_on_stage_a_blocks.py output dirs)')
    ap.add_argument('--stage_b_glob', default=None, help='older lookup mode (3-way LOOCV run dirs)')
    ap.add_argument('--tsv', default=None, help='timing_onehot_task1.tsv (needed with either Stage B option)')
    ap.add_argument('--stage_a_perm_glob', default=None)
    ap.add_argument('--exclude_subjects', nargs='*', default=['sub-29'])
    ap.add_argument('--output_file', default='output/two_stage/two_stage_figure',
                    help='path stem; .png/.pdf/.svg are written')
    args = ap.parse_args()

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
        stage_rows.append(('Stage B | called Task\n(3-way acc.)',
                           [c['stage_b_called_task_acc'] for c in composed], 1 / 3, None))
        stage_rows.append(('End-to-end\n(4-way acc.)', [c['end_to_end_acc'] for c in composed], E2E_CHANCE, None))

    S._style()
    cm_a = np.sum([np.array(tsd_cm(a_preds[f])) for f in folds], axis=0)
    panels = [
        lambda ax, fig: plot_per_subject(ax, folds, real_bal, perm_by_fold),
        lambda ax, fig: plot_stages(ax, stage_rows, rng),
        lambda ax, fig: _single_confusion(ax, fig, cm_a, ['No task', 'Task'], ['No task', 'Task'],
                                          'Stage A confusion (pooled)'),
    ]
    if null_means is not None:
        real_acc = float(np.mean([tsd.stage_a_metrics(a_preds[f])['accuracy'] for f in folds]))
        panels.append(lambda ax, fig: plot_null(ax, real_acc, null_means))
    if composed:
        pooled = np.sum([np.array(c['end_to_end_confusion']) for c in composed], axis=0)
        e2e_rows = ['No task', 'tennis', 'navigation', 'squeeze']
        e2e_cols = e2e_rows if all_blocks else e2e_rows + ['Task\n(unclassified)']
        panels.append(lambda ax, fig: _single_confusion(ax, fig, pooled, e2e_rows, e2e_cols,
                                                         'End-to-end confusion (pooled)'))
    if composed and all_blocks:
        panels.append(lambda ax, fig: plot_class_mix(ax, composed))
        panels.append(lambda ax, fig: plot_confidence(ax, composed))

    ncols = 3 if (len(panels) == 3 or len(panels) >= 5) else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(7.2 if ncols == 2 else 9.0, 3.0 * nrows + 0.3))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.75, wspace=0.75, left=0.08, right=0.97, top=0.93, bottom=0.08)
    for i, draw in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        draw(ax, fig)
        S._panel_letter(ax, 'abcdefgh'[i])
    n_b = len(composed)
    fig.text(0.08, -0.03, f"{len(folds)} Stage A fold(s); {n_b} with Stage B; "
             f"{'permutations: ' + str(len(null_means)) if null_means is not None else 'no permutations'}",
             fontsize=6, color=S.INK_2, va='top')

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out.with_suffix(ext), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out.with_suffix('.png')} (+ .pdf, .svg): {len(folds)} Stage A folds, {n_b} with Stage B")


def tsd_cm(preds):
    """2x2 confusion counts [[TN, FP], [FN, TP]] from Stage A predictions."""
    y = np.array([int(p['true_label']) for p in preds])
    yh = np.array([int(p['predicted_label']) for p in preds])
    return [[int(((y == 0) & (yh == 0)).sum()), int(((y == 0) & (yh == 1)).sum())],
            [int(((y == 1) & (yh == 0)).sum()), int(((y == 1) & (yh == 1)).sum())]]


if __name__ == '__main__':
    main()

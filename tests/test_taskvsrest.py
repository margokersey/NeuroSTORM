"""Stage A (Task vs. No Task) dataset + two-stage composition, on tiny synthetic data."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.fmri_datasets import MichiganTaskVsRest

SEQ = 37
IMAGERY_TAG = "FIXED_xcp_nosmooth_task-imagery_run-1"


def _write_subject(root, run_dir, sub, n_frames):
    d = Path(root) / run_dir / "img" / sub
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": torch.zeros(n_frames, 4, 4, 4, dtype=torch.int8),
                "scale": 1.0, "num_frames": n_frames}, d / "data.pt")


def _write_tsv(root):
    """30 task blocks (10 per class, shuffled), each 30 task + 7/8 rest TRs, from TR 20."""
    rng = np.random.RandomState(0)
    order = np.repeat(["tennis", "navigation", "squeeze"], 10)
    rng.shuffle(order)
    rows = [dict(rest=1, squeeze_imagery=0, squeeze_action=0, navigation=0, tennis=0)] * 20
    for i, name in enumerate(order):
        row = dict(rest=0, squeeze_imagery=int(name == "squeeze"), squeeze_action=0,
                   navigation=int(name == "navigation"), tennis=int(name == "tennis"))
        rest = dict(rest=1, squeeze_imagery=0, squeeze_action=0, navigation=0, tennis=0)
        rows += [row] * 30 + [rest] * (7 + (i % 2))
    meta = Path(root) / IMAGERY_TAG / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # pad/trim to exactly 1145 TRs like the real run
    df = pd.concat([df, pd.DataFrame([rows[0]] * max(0, 1145 - len(df)))]).iloc[:1145]
    df.to_csv(meta / "timing_onehot_task1.tsv", sep="\t", index=False)
    return list(order)


@pytest.fixture()
def dataset_root(tmp_path):
    order = _write_tsv(tmp_path)
    for sub in ("sub-01", "sub-02", "sub-29"):
        _write_subject(tmp_path, IMAGERY_TAG, sub, 1145)
    _write_subject(tmp_path, "FIXED_xcp_nosmooth_task-rest_run-1", "sub-01", 770)
    _write_subject(tmp_path, "FIXED_xcp_nosmooth_task-rest_run-1", "sub-02", 1145)  # sub-22-like
    _write_subject(tmp_path, "FIXED_xcp_nosmooth_task-rest_run-1", "sub-29", 770)
    _write_subject(tmp_path, "FIXED_xcp_nosmooth_task-rest_run-2", "sub-01", 770)
    _write_subject(tmp_path, "FIXED_xcp_nosmooth_task-rest_run-2", "sub-02", 770)
    # sub-29 has no rest-2
    return tmp_path, order


def _make(root, subjects, train, **kw):
    return MichiganTaskVsRest(
        root=str(root / IMAGERY_TAG), subject_dict={s: [0, 0] for s in subjects},
        sequence_length=SEQ, stride_within_seq=1, stride_between_seq=1.0, train=train, **kw)


@pytest.mark.unit
def test_blocks_labels_and_bounds(dataset_root):
    root, _ = dataset_root
    ds = _make(root, ["sub-01", "sub-02", "sub-29"], train=False)
    by_sub = {}
    for (_i, subj, path, start, _stride, nframes, label, _sex) in ds.data:
        by_sub.setdefault(subj, []).append((path, start, nframes, label))
        assert start + SEQ <= nframes                      # window always fits
        if label == 0:
            assert nframes <= 770 and start + SEQ <= 770    # mock blocks never use TRs >= 770
            assert "task-rest_run-" in path
    for subj in ("sub-01", "sub-02"):
        labels = [x[3] for x in by_sub[subj]]
        assert labels.count(1) == 30 and labels.count(0) == 30
    # sub-29 has no rest-2 -> only 15 mock blocks
    labels29 = [x[3] for x in by_sub["sub-29"]]
    assert labels29.count(1) == 30 and labels29.count(0) == 15


@pytest.mark.unit
def test_mock_onsets_match_first_15_imagery_blocks(dataset_root):
    root, _ = dataset_root
    ds = _make(root, ["sub-01"], train=False)
    blocks = ds._blocks_from_tsv(str(root / IMAGERY_TAG / "metadata" / "timing_onehot_task1.tsv"))
    expected = [s for s, _e, _l in blocks[:15]]
    for run in (1, 2):
        got = [t[3] for t in ds.data if t[6] == 0 and f"task-rest_run-{run}" in t[2]]
        assert got == expected
    assert max(expected) + SEQ <= 770   # the premise of the design: first 15 onsets fit in 770 TRs


@pytest.mark.unit
def test_sub22_like_long_rest_is_capped(dataset_root):
    root, _ = dataset_root
    ds = _make(root, ["sub-02"], train=False)
    assert all(t[5] == 770 for t in ds.data if t[6] == 0 and "run-1" in t[2])


@pytest.mark.unit
def test_permutation_within_subject_preserves_balance(dataset_root):
    root, _ = dataset_root
    real = _make(root, ["sub-01", "sub-02"], train=True)
    perm = _make(root, ["sub-01", "sub-02"], train=True, permute_labels=True, permutation_seed=7)
    assert [t[:6] for t in real.data] == [t[:6] for t in perm.data]   # same windows
    changed = sum(a[6] != b[6] for a, b in zip(real.data, perm.data))
    assert changed > 0
    for subj in ("sub-01", "sub-02"):
        r = sorted(t[6] for t in real.data if t[1] == subj)
        p = sorted(t[6] for t in perm.data if t[1] == subj)
        assert r == p                                                 # per-subject balance kept
    # eval splits are never permuted
    ev = _make(root, ["sub-01"], train=False, permute_labels=True, permutation_seed=7)
    assert [t[6] for t in ev.data] == [t[6] for t in _make(root, ["sub-01"], train=False).data]


@pytest.mark.unit
def test_missing_rest_dir_raises(tmp_path):
    _write_tsv(tmp_path)
    _write_subject(tmp_path, IMAGERY_TAG, "sub-01", 1145)
    with pytest.raises(FileNotFoundError):
        _make(tmp_path, ["sub-01"], train=False)


# --------------------------------------------------------- composition ----
def _load_two_stage():
    path = Path(__file__).resolve().parents[1] / "analysis" / "two_stage_decoding.py"
    spec = importlib.util.spec_from_file_location("two_stage_decoding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _a_pred(true, pred, p):
    return {"subject": "sub-01", "true_label": true, "predicted_label": pred,
            "softmax_probs": [1 - p, p]}


def _b_pred(true, pred):
    return {"subject": "sub-01", "true_label": true, "predicted_label": pred}


@pytest.mark.unit
def test_compose_fold_counts():
    m = _load_two_stage()
    labels = [0, 1, 2] * 10
    # Stage A: calls 28/30 task blocks "Task" (misses blocks 0 and 1); 25/30 mock blocks "NoTask"
    a = [_a_pred(1, 0 if i < 2 else 1, 0.3 if i < 2 else 0.9) for i in range(30)]
    a += [_a_pred(0, 1 if i < 5 else 0, 0.8 if i < 5 else 0.1) for i in range(30)]
    # Stage B: right on every block except block 5
    b = [_b_pred(t, t if i != 5 else (t + 1) % 3) for i, t in enumerate(labels)]
    r = m.compose_fold(a, b, labels, fold=101)
    assert r["stage_a"]["accuracy"] == pytest.approx((28 + 25) / 60)
    assert r["stage_b_called_task_n"] == 28
    assert r["stage_b_called_task_acc"] == pytest.approx(27 / 28)
    assert r["stage_b_all_task_blocks_acc"] == pytest.approx(29 / 30)
    cm = np.array(r["end_to_end_confusion"])
    assert cm.sum() == 60
    assert cm[0, 0] == 25 and cm[0, 4] == 5            # false alarms land in "task_unclassified"
    assert cm[1:, 0].sum() == 2                         # the two missed task blocks
    assert r["end_to_end_acc"] == pytest.approx((25 + 27) / 60)
    assert m.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert m.auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)


@pytest.mark.unit
def test_compose_fold_rejects_misalignment():
    m = _load_two_stage()
    labels = [0, 1, 2] * 10
    a = [_a_pred(1, 1, 0.9)] * 30 + [_a_pred(0, 0, 0.1)] * 30
    wrong_order = [_b_pred(t, t) for t in reversed(labels)]
    with pytest.raises(ValueError, match="order"):
        m.compose_fold(a, wrong_order, labels, 101)
    with pytest.raises(ValueError, match="cannot align"):
        m.compose_fold(a, [_b_pred(t, t) for t in labels[:-1]], labels, 101)

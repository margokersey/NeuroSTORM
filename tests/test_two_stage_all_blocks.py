"""Tests for the Stage-B-on-all-blocks composition in analysis/two_stage_decoding.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import two_stage_decoding as tsd  # noqa: E402


def _fold(subject, a_called, b_logits, n_task=3, n_mock=3):
    """Stage A/B prediction lists for one subject: n_task real blocks then n_mock mock blocks."""
    labels = [1] * n_task + [0] * n_mock
    a = [{'sample_idx': i, 'subject': subject, 'true_label': y, 'predicted_label': c, 'logits': [2.0 if c else -2.0],
          'softmax_probs': [0.1, 0.9] if c else [0.9, 0.1]} for i, (y, c) in enumerate(zip(labels, a_called))]
    b = [{'sample_idx': i, 'subject': subject, 'stage_a_label': y, 'predicted_label': int(np.argmax(lg)),
          'logits': lg, 'softmax_probs': [1 / 3] * 3} for i, (y, lg) in enumerate(zip(labels, b_logits))]
    return a, {'predictions': b}


def test_gating_groups_and_confusion():
    # real blocks 0,1,2 are tennis, navigation, squeeze; Stage A misses real block 2 and false-alarms on mock block 5
    a, b = _fold('sub-01', [1, 1, 0, 0, 0, 1],
                 [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 5], [5, 0, 0], [0, 5, 0]])
    r = tsd.compose_fold_all(a, b, [0, 1, 2], 101)
    groups = [x['group'] for x in r['records']]
    assert groups == ['real_called_task'] * 2 + ['real_missed'] + ['mock_rejected'] * 2 + ['mock_called_task']
    cm = np.array(r['end_to_end_confusion'])
    assert cm[1, 1] == 1 and cm[2, 2] == 1          # tennis, navigation right
    assert cm[3, 0] == 1                              # missed squeeze -> "No task"
    assert cm[0, 0] == 2 and cm[0, 2] == 1            # 2 rejections; false alarm labelled navigation by Stage B
    assert r['stage_b_called_task_acc'] == 1.0 and r['stage_b_all_task_blocks_acc'] == 1.0
    assert cm.sum() == 6


def test_misaligned_labels_are_rejected():
    a, b = _fold('sub-01', [1, 1, 1, 0, 0, 0], [[1, 0, 0]] * 6)
    b['predictions'][0]['stage_a_label'] = 0
    with pytest.raises(ValueError):
        tsd.compose_fold_all(a, b, [0, 1, 2], 101)


def test_two_subjects_in_one_fold_rejected():
    a, b = _fold('sub-01', [1, 1, 1, 0, 0, 0], [[1, 0, 0]] * 6)
    b['predictions'][5]['subject'] = 'sub-02'
    with pytest.raises(ValueError):
        tsd.compose_fold_all(a, b, [0, 1, 2], 101)


def test_within_subject_difference_uses_subject_means():
    per_fold = []
    for k, (hi, lo) in enumerate([(5.0, 1.0), (4.0, 2.0), (6.0, 2.0)]):
        # real blocks all called Task; mock: block 3 rejected (low margin), block 5 false alarm (high margin)
        a, b = _fold(f'sub-{k}', [1, 1, 1, 0, 0, 1],
                     [[hi, 0, 0]] * 3 + [[lo, 0, 0]] * 2 + [[hi, 0, 0]])
        per_fold.append(tsd.compose_fold_all(a, b, [0, 0, 0], 101 + k))
    t = tsd.within_subject_test(per_fold, 'logit_margin', 'mock_called_task', 'mock_rejected')
    assert t['n_subjects'] == 3
    assert t['mean_diff'] == pytest.approx(np.mean([4.0, 2.0, 4.0]))

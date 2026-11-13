#!/usr/bin/env python
"""Test 3: task decoding (tennis / navigation / squeeze) during loss of responsiveness (LOR).

For ONE LOOCV fold this loads that fold's best 3-way checkpoint (trained on imagery
run 1 with the held-out subject excluded) and predicts every block of the held-out
subject's imagery run 2 and run 3. Nothing is trained. Each block is tagged with a
segment (pre_lor / boundary_lor / lor / boundary_ror / ror -- see datasets.fmri_datasets.
MichiganLOR) so the analysis can compare LOR accuracy with the responsive control
segments of the SAME subject and scanner session. Selecting segments happens later.

Leakage guard (same as predict_stage_b_on_stage_a_blocks.py): fold N's checkpoint is
only ever applied to fold N's test subject. The script asserts that (a) the checkpoint's
hparams say dataset_split_num == N, (b) the blocks come from exactly the split file's
test subject, (c) the checkpoint lives in the fold-N 3-way directory.

Run on a GPU node (sbatch analysis/sbatch_lor_task_decoding.sh), not the login node:
    python analysis/predict_lor_task_decoding.py --fold 101 \\
        --lor_csv $DATA_ROOT/metadata/lor_ror/LOR_ROR_timing.csv

Output (per fold):
    <output_root>/FIXED_mich_xcpd_nosmooth_lor_task_decoding_fold-<N>/lor_predictions_fold_<N>.json
    {"meta": {...}, "predictions": [{sample_idx, subject, run, block_idx, start,
      true_label, true_label_name, segment, lor_tr, ror_tr, predicted_label,
      predicted_label_name, logits[3], softmax_probs[3]}, ...]}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
from predict_stage_b_on_stage_a_blocks import (  # noqa: E402  (reuse the Stage B helpers)
    STAGE_B_NAMES, STAGE_B_PREFIX, find_checkpoint, read_split_test_subjects)

OUT_PREFIX = "FIXED_mich_xcpd_nosmooth_lor_task_decoding"
DATASET = "MICHIGAN_LOR"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True, help='LOOCV fold / split number, e.g. 101')
    ap.add_argument('--lor_csv', default=os.environ.get('NEUROSTORM_LOR_CSV'),
                    help='LOR_ROR_timing.csv (columns: Subject, LOR time (TR in task2), ROR time (TR in task3))')
    ap.add_argument('--output_root', default='nas/output/neurostorm')
    ap.add_argument('--stage_b_prefix', default=STAGE_B_PREFIX,
                    help='project-name prefix of the 3-way LOOCV runs (<prefix>_fold-<N>)')
    ap.add_argument('--ckpt', default=None, help='override the 3-way checkpoint (default: the one in the fold dir)')
    ap.add_argument('--image_path', default=None,
                    help='override the imagery run-1 data root stored in the checkpoint hparams '
                         '(runs 2/3 are the sibling task-imagery_run-N dirs)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if not args.lor_csv:
        sys.exit("--lor_csv (or env NEUROSTORM_LOR_CSV) is required")

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch
    from models.lightning_model import LightningModel
    from datasets.data_module import fMRIDataModule
    from datasets.fmri_datasets import MichiganLOR

    stage_b_dir = os.path.join(args.output_root, f"{args.stage_b_prefix}_fold-{args.fold}")
    ckpt_path = args.ckpt or find_checkpoint(stage_b_dir)
    if f"fold-{args.fold}" not in os.path.abspath(ckpt_path) and args.ckpt is None:
        raise RuntimeError(f"checkpoint {ckpt_path} is not from fold {args.fold}")
    print(f"[lor] fold {args.fold}: checkpoint {ckpt_path}", flush=True)

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = dict(ck['hyper_parameters'])
    del ck
    problems = []
    if int(hp.get('dataset_split_num', -1)) != args.fold:
        problems.append(f"checkpoint dataset_split_num={hp.get('dataset_split_num')} != fold {args.fold}")
    if hp.get('task_name') != 'imagery_task_classification' or int(hp.get('num_classes', 0)) != 3:
        problems.append(f"checkpoint is not the 3-way model (task_name={hp.get('task_name')}, num_classes={hp.get('num_classes')})")
    if hp.get('permute_labels'):
        problems.append("checkpoint was trained with permuted labels")
    if hp.get('tpt_strategy', 'none') != 'none':
        problems.append(f"tpt_strategy={hp.get('tpt_strategy')} is not supported by this script")
    if problems:
        sys.exit("Refusing to run: " + "; ".join(problems))

    # Use the same onset trim as the dataset the 3-way model was trained on.
    train_ds_name = hp.get('dataset_name')
    trim = {'MICHIGAN': 2, 'MICHIGAN_FULL_BLOCK': 0, 'MICHIGAN_TASK_VS_REST': 0}.get(train_ds_name)
    if trim is None:
        sys.exit(f"unexpected training dataset_name {train_ds_name!r}; set the block trim explicitly")
    MichiganLOR.TRIM_START = trim
    MichiganLOR.LOR_CSV = args.lor_csv
    print(f"[lor] trained on {train_ds_name} -> block trim {trim}; LOR table {args.lor_csv}", flush=True)

    from argparse import ArgumentParser
    defaults = {}
    for add in (LightningModel.add_model_specific_args, fMRIDataModule.add_data_specific_args):
        defaults.update(vars(add(ArgumentParser(add_help=False)).parse_known_args([])[0]))
    missing_hp = {k: v for k, v in defaults.items() if k not in hp}
    if missing_hp:
        print(f"[lor] hparams missing from the checkpoint, filled with defaults: {sorted(missing_hp)}", flush=True)
    dm_hp = {**missing_hp, **hp}
    dm_hp['dataset_name'] = DATASET
    dm_hp['eval_batch_size'] = args.batch_size
    dm_hp['num_workers'] = args.num_workers
    if args.image_path:
        dm_hp['image_path'] = args.image_path
    if 'task-imagery_run-1' not in str(dm_hp['image_path']):
        sys.exit(f"image_path must be the imagery run-1 root, got {dm_hp['image_path']}")
    dm = fMRIDataModule(**dm_hp)

    expected_test = read_split_test_subjects(dm.split_file_path)
    print(f"[lor] split file {dm.split_file_path}: test subjects {expected_test}", flush=True)

    model = LightningModel.load_from_checkpoint(ckpt_path, data_module=None, map_location='cpu', **missing_hp)
    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model.to(dev).eval()

    loader = dm.test_dataloader()
    test_ds = dm.test_dataset
    info = test_ds.sample_info          # parallel to the (unshuffled) loader order
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            subj, logits, target, _, _ = model._compute_logits(batch)
            logits = logits.detach().float().cpu().reshape(-1, 3).numpy()
            target = target.detach().cpu().reshape(-1).numpy()
            for s, lg, t in zip(subj, logits, target):
                meta = info[len(preds)]
                # alignment guard: loader order must equal dataset order
                if str(s) != meta['subject'] or int(t) != test_ds.LABEL_MAP[meta['true_label_name']]:
                    raise RuntimeError(f"sample {len(preds)}: loader order differs from dataset order "
                                       f"({s}/{t} vs {meta['subject']}/{meta['true_label_name']})")
                e = np.exp(lg - lg.max())
                pred = int(np.argmax(lg))
                preds.append({'sample_idx': len(preds), **meta,
                              'true_label': int(t),
                              'predicted_label': pred, 'predicted_label_name': STAGE_B_NAMES[pred],
                              'logits': lg.tolist(), 'softmax_probs': (e / e.sum()).tolist()})
    if len(preds) != len(info):
        raise RuntimeError(f"{len(preds)} predictions for {len(info)} dataset samples")

    subjects = sorted({p['subject'] for p in preds})
    if subjects != sorted(expected_test):
        raise RuntimeError(f"predicted subjects {subjects} != split-file test subjects {expected_test}: refusing to save")
    seg_counts = {}
    for p in preds:
        key = f"run{p['run']}:{p['segment']}"
        seg_counts[key] = seg_counts.get(key, 0) + 1
    acc = {}
    for seg in ('pre_lor', 'lor', 'ror'):
        ps = [p for p in preds if p['segment'] == seg]
        if ps:
            acc[seg] = [float(np.mean([p['predicted_label'] == p['true_label'] for p in ps])), len(ps)]
    print(f"[lor] {len(preds)} blocks for {subjects}: {seg_counts}", flush=True)
    print(f"[lor] block accuracy (acc, n) by segment: {acc}", flush=True)

    out_dir = Path(args.output_root) / f"{OUT_PREFIX}_fold-{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"lor_predictions_fold_{args.fold}.json"
    with open(out, 'w') as f:
        json.dump({'meta': {'fold': args.fold, 'subjects': subjects, 'checkpoint': os.path.abspath(ckpt_path),
                            'split_file': dm.split_file_path, 'trained_on_dataset': train_ds_name,
                            'block_trim': trim, 'lor_csv': os.path.abspath(args.lor_csv),
                            'segment_counts': seg_counts, 'class_names': STAGE_B_NAMES},
                   'predictions': preds}, f, indent=2)
    print(f"[lor] wrote {out}", flush=True)


if __name__ == '__main__':
    main()

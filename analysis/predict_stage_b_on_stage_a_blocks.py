#!/usr/bin/env python
"""Stage B of the two-stage decoding test: run the EXISTING 3-way LOOCV model on
every Stage A test block (real task blocks AND mock rest blocks).

For one LOOCV fold this loads that fold's best 3-way checkpoint (trained with the
held-out subject excluded) and predicts tennis / navigation / squeeze for all 60
blocks of the held-out subject that Stage A saw (30 task + up to 30 mock). Nothing
is trained. Stage A gating ("only look at blocks called Task") is applied later in
analysis/two_stage_decoding.py, because a block's 3-way output does not depend on
Stage A.

Leakage guard: fold N's checkpoint is only ever applied to fold N's test subject.
The script asserts that (a) the checkpoint's own hparams say dataset_split_num == N,
(b) the blocks come from exactly the split file's test subject, and (c) the
checkpoint lives in the fold-N 3-way directory.

Self-check: for the true task blocks the new predictions must reproduce the 3-way
run's saved test predictions (same model, same clips). A mismatch is reported.

Run on a GPU node (sbatch analysis/sbatch_stage_b_predict.sh), not the login node:
    python analysis/predict_stage_b_on_stage_a_blocks.py --fold 101

Output (per fold):
    <out_root>/FIXED_mich_xcpd_nosmooth_stageB_allblocks_fold-<N>/stage_b_predictions_fold_<N>.json
    {"meta": {...}, "predictions": [{sample_idx, subject, block_type, stage_a_label,
      predicted_label, predicted_label_name, logits[3], softmax_probs[3]}, ...]}
sample_idx is the position in the Stage A test set, so it lines up one-to-one with
the Stage A prediction JSON of the same fold.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
STAGE_B_PREFIX = "FIXED_mich_xcpd_nosmooth_tpt_strategy-none_loocv"
STAGE_B_NAMES = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}
STAGE_A_DATASET = "MICHIGAN_TASK_VS_REST"


def read_split_test_subjects(split_file):
    """Test subjects listed in a split_fixed_N.txt (same layout DataModule.load_split reads)."""
    lines = [x.rstrip('\n') for x in open(split_file)]
    idx = {k: next(i for i, l in enumerate(lines) if k in l) for k in ('train', 'val', 'test')}
    return [s.strip() for s in lines[idx['test'] + 1:] if s.strip()]


def find_checkpoint(stage_b_dir):
    ckpts = sorted(glob.glob(os.path.join(stage_b_dir, 'checkpt-*.ckpt')))
    if len(ckpts) != 1:
        raise FileNotFoundError(
            f"expected exactly one checkpt-*.ckpt in {stage_b_dir} (ModelCheckpoint save_top_k=1), "
            f"found {len(ckpts)}: {ckpts}. Pass --ckpt to choose one explicitly.")
    return ckpts[0]


def latest_saved_predictions(stage_b_dir):
    """(predictions, info) of the 3-way run's own latest saved test predictions, or (None, info).
    info['final_test_ckpt'] is read from that version's hparams.yaml ('best' or 'last'): the
    saved predictions only match the best checkpoint if it says 'best'."""
    best = None
    for d in glob.glob(os.path.join(stage_b_dir, 'lightning_logs', 'version_*', 'test_predictions')):
        try:
            v = int(Path(d).parent.name.split('_')[-1])
        except ValueError:
            continue
        files = sorted(glob.glob(os.path.join(d, 'test_predictions_fold_*.json')))
        if files and (best is None or v > best[0]):
            best = (v, files[0], Path(d).parent / 'hparams.yaml')
    if best is None:
        return None, {}
    info = {'version': best[0], 'final_test_ckpt': None}
    try:
        for line in open(best[2]):
            if line.startswith('final_test_ckpt:'):
                info['final_test_ckpt'] = line.split(':', 1)[1].strip()
    except OSError:
        pass
    return json.load(open(best[1])), info


def check_reproduces_saved(preds, saved, atol=1e-2):
    """Compare new predictions on the true task blocks to the 3-way run's saved ones.
    Passes if every predicted label is identical and the largest logit difference is <= atol.
    Small float differences are expected (different batch grouping, non-deterministic GPU
    kernels); a wrong checkpoint or clip would flip labels or differ by O(1)."""
    new_task = [p for p in preds if p['stage_a_label'] == 1]
    if saved is None:
        return {'status': 'skipped', 'reason': 'no saved 3-way predictions found'}
    if len(new_task) != len(saved):
        return {'status': 'mismatch', 'reason': f'{len(new_task)} task blocks vs {len(saved)} saved'}
    same_label = sum(int(a['predicted_label']) == int(b['predicted_label']) for a, b in zip(new_task, saved))
    new_lg = np.array([a['logits'] for a in new_task], float)
    old_lg = np.array([b['logits'] for b in saved], float)
    max_diff = float(np.max(np.abs(new_lg - old_lg)))
    max_rel = float(np.max(np.abs(new_lg - old_lg) / np.maximum(np.abs(old_lg), 1e-6)))
    ok = same_label == len(saved) and max_diff <= atol
    return {'status': 'ok' if ok else 'mismatch', 'n': len(saved),
            'same_predicted_label': same_label, 'max_abs_logit_diff': max_diff, 'max_rel_logit_diff': max_rel}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True, help='LOOCV fold / split number, e.g. 101')
    ap.add_argument('--output_root', default='nas/output/neurostorm')
    ap.add_argument('--stage_b_prefix', default=STAGE_B_PREFIX,
                    help='project-name prefix of the 3-way LOOCV runs (<prefix>_fold-<N>)')
    ap.add_argument('--ckpt', default=None, help='override the 3-way checkpoint (default: the one in the fold dir)')
    ap.add_argument('--image_path', default=None,
                    help='override the imagery-run-1 data root stored in the checkpoint hparams')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--atol', type=float, default=1e-2, help='tolerance for the reproduce-saved-predictions check')
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch
    from models.lightning_model import LightningModel
    from datasets.data_module import fMRIDataModule

    stage_b_dir = os.path.join(args.output_root, f"{args.stage_b_prefix}_fold-{args.fold}")
    ckpt_path = args.ckpt or find_checkpoint(stage_b_dir)
    if f"fold-{args.fold}" not in os.path.abspath(ckpt_path) and args.ckpt is None:
        raise RuntimeError(f"checkpoint {ckpt_path} is not from fold {args.fold}")
    print(f"[stage B] fold {args.fold}: checkpoint {ckpt_path}", flush=True)

    # ---- hparams of the 3-way run (these built both its model and its data module)
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

    # ---- Stage A test set (same dataset class/splits/clips as the Stage A run) for the 3-way model
    from argparse import ArgumentParser
    defaults = {}
    for add in (LightningModel.add_model_specific_args, fMRIDataModule.add_data_specific_args):
        defaults.update(vars(add(ArgumentParser(add_help=False)).parse_known_args([])[0]))
    missing_model_hp = {k: v for k, v in defaults.items() if k not in hp}
    if missing_model_hp:
        print(f"[stage B] hparams missing from the checkpoint, filled with defaults: {sorted(missing_model_hp)}", flush=True)
    dm_hp = {**missing_model_hp, **hp}
    dm_hp['dataset_name'] = STAGE_A_DATASET
    dm_hp['eval_batch_size'] = args.batch_size
    dm_hp['num_workers'] = args.num_workers
    if args.image_path:
        dm_hp['image_path'] = args.image_path
    if 'task-imagery_run-1' not in str(dm_hp['image_path']):
        sys.exit(f"image_path must be the imagery run-1 root, got {dm_hp['image_path']}")
    dm = fMRIDataModule(**dm_hp)

    expected_test = read_split_test_subjects(dm.split_file_path)
    print(f"[stage B] split file {dm.split_file_path}: test subjects {expected_test}", flush=True)

    # Older checkpoints may lack hparams added since they were trained. Fill ONLY missing keys with the
    # current argparse defaults (never override a saved value).
    model = LightningModel.load_from_checkpoint(ckpt_path, data_module=None, map_location='cpu', **missing_model_hp)
    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model.to(dev).eval()

    # block type per sample (Stage A label: 1 = real imagery block, 0 = mock rest block)
    preds = []
    loader = dm.test_dataloader()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            subj, logits, target, _, _ = model._compute_logits(batch)
            logits = logits.detach().float().cpu().reshape(-1, 3).numpy()   # reshape: a 1-block batch squeezes to (3,)
            target = target.detach().cpu().reshape(-1).numpy()
            for s, lg, t in zip(subj, logits, target):
                e = np.exp(lg - lg.max())
                probs = e / e.sum()
                pred = int(np.argmax(lg))
                preds.append({
                    'sample_idx': len(preds),
                    'subject': str(s),
                    'block_type': 'task' if int(t) == 1 else 'mock',
                    'stage_a_label': int(t),
                    'predicted_label': pred,
                    'predicted_label_name': STAGE_B_NAMES[pred],
                    'logits': lg.tolist(),
                    'softmax_probs': probs.tolist(),
                })

    # ---- guards
    subjects = sorted({p['subject'] for p in preds})
    if subjects != sorted(expected_test):
        raise RuntimeError(f"predicted subjects {subjects} != split-file test subjects {expected_test}: refusing to save")
    n_task = sum(p['stage_a_label'] == 1 for p in preds)
    n_mock = len(preds) - n_task
    saved, saved_info = latest_saved_predictions(stage_b_dir)
    check = check_reproduces_saved(preds, saved, atol=args.atol)
    check['saved_run'] = saved_info
    if check['status'] == 'mismatch' and saved_info.get('final_test_ckpt') != 'best':
        check['note'] = ("the saved 3-way predictions were not made with final_test_ckpt=best "
                         f"(found {saved_info.get('final_test_ckpt')!r}); a mismatch is expected if they used last-epoch weights")
    print(f"[stage B] {len(preds)} blocks ({n_task} task, {n_mock} mock) for {subjects}", flush=True)
    print(f"[stage B] reproduces saved 3-way predictions on task blocks: {check}", flush=True)
    if check['status'] == 'mismatch':
        print("[stage B] WARNING: task-block predictions differ from the saved 3-way run -- "
              "check the checkpoint and data root before using these results", flush=True)

    out_dir = Path(args.output_root) / f"FIXED_mich_xcpd_nosmooth_stageB_allblocks_fold-{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"stage_b_predictions_fold_{args.fold}.json"
    with open(out, 'w') as f:
        json.dump({'meta': {'fold': args.fold, 'subjects': subjects, 'checkpoint': os.path.abspath(ckpt_path),
                            'split_file': dm.split_file_path, 'n_task_blocks': n_task, 'n_mock_blocks': n_mock,
                            'reproduces_saved_3way': check, 'class_names': STAGE_B_NAMES},
                   'predictions': preds}, f, indent=2)
    print(f"[stage B] wrote {out}", flush=True)


if __name__ == '__main__':
    main()

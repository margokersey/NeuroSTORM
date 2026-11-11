#!/usr/bin/env python
"""Verify one fine-tune run produced by the Great Lakes launchers (sandbox).

Run from the sandbox working dir (the one holding nas/). Exit 0 = all checks pass.

Checks:
  * run dir has last.ckpt and a best (checkpt-*) checkpoint; training reached the final epoch
  * saved hparams match the launcher's intent: dataset_split_num, load_model_path, permute_labels
  * encoder was INITIALISED from the pretrained ckpt (small relative distance, not a fresh init)
    and was actually UPDATED (not identical)
  * test_predictions.pt / test_predictions_fold_*.json are internally consistent and cover exactly
    the split file's held-out subjects, with the expected clips-per-subject and class balance
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--split", required=True, help="split_fixed_N.txt the launcher should have used")
ap.add_argument("--ckpt", required=True, help="pretrained ckpt the launcher passed")
ap.add_argument("--category", default="neurostorm")
ap.add_argument("--clips_per_subject", type=int, default=30)
ap.add_argument("--epochs", type=int, default=int(os.environ.get("SANDBOX_MAX_EPOCHS", 2)))
a = ap.parse_args()

errs = []


def expect(cond, msg):
    if not cond:
        errs.append(msg)


run = os.path.join("nas", "output", a.category, a.project)
expect(os.path.isdir(run), f"run dir missing: {run}")
if errs:
    print(f"[{a.project}] " + "; ".join(errs)); sys.exit(1)

# ---- split file -> expected test subjects
lines = [x.strip() for x in open(a.split) if x.strip()]
ti = lines.index("test_subjects")
test_subjects = lines[ti + 1:]
train_subjects = lines[1:lines.index("val_subjects")]
expect(set(test_subjects).isdisjoint(train_subjects), "split: test subjects overlap train")

# ---- checkpoints
last = os.path.join(run, "last.ckpt")
best = glob.glob(os.path.join(run, "checkpt-*.ckpt"))
expect(os.path.isfile(last), "last.ckpt missing")
expect(len(best) >= 1, "no best checkpt-*.ckpt (valid_acc checkpoint) saved")
ck = torch.load(last, map_location="cpu", weights_only=False)
expect(ck["epoch"] == a.epochs - 1, f"last.ckpt epoch={ck['epoch']} (expected {a.epochs - 1}) -> training did not finish")
hp = ck.get("hyper_parameters", {})
m = re.search(r"split_fixed_(\d+)\.txt", a.split)
expect(hp.get("dataset_split_num") == int(m.group(1)), f"hparams.dataset_split_num={hp.get('dataset_split_num')} != {m.group(1)}")
expect(hp.get("load_model_path") == a.ckpt, f"hparams.load_model_path={hp.get('load_model_path')} != {a.ckpt}")
is_perm = "_perm" in a.project
expect(bool(hp.get("permute_labels", False)) == is_perm, f"hparams.permute_labels={hp.get('permute_labels')} but project is {'perm' if is_perm else 'real-label'}")
expect(hp.get("project_name") == a.project, f"hparams.project_name={hp.get('project_name')}")

# ---- init-from-pretrained + actually updated
if not os.path.isfile(a.ckpt):
    print(f"[{a.project}] FAILED: pretrained ckpt not found: {a.ckpt}"); sys.exit(1)
pre = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
ft = ck["state_dict"]
rel, same = [], 0
for k, w0 in pre.items():
    if k in ft and w0.ndim >= 2 and w0.is_floating_point():
        w1 = ft[k]
        d = (w1 - w0).norm() / (w0.norm() + 1e-12)
        rel.append(float(d)); same += int(d == 0)
expect(len(rel) > 3, "no overlapping encoder tensors between pretrained and fine-tuned ckpt")
if rel:
    med = sorted(rel)[len(rel) // 2]
    expect(med < 0.7, f"encoder median rel. distance from pretrained = {med:.2f} (>=0.7 => looks like a fresh init, ckpt not loaded?)")
    expect(same < len(rel), "encoder identical to pretrained -> no training updates")

# ---- predictions
vdirs = sorted(glob.glob(os.path.join(run, "lightning_logs", "version_*")))
expect(len(vdirs) >= 1, "no lightning_logs/version_*")
v = vdirs[-1]
pt = os.path.join(v, "test_predictions.pt")
js = glob.glob(os.path.join(v, "test_predictions", "test_predictions_fold_*.json"))
sm = glob.glob(os.path.join(v, "test_predictions", "test_summary_fold_*.json"))
expect(os.path.isfile(pt) and len(js) == 1 and len(sm) == 1, "prediction files missing (pt/json/summary)")
n_expect = a.clips_per_subject * len(test_subjects)
if os.path.isfile(pt):
    d = torch.load(pt, weights_only=False)
    expect(d["probs"].shape == (n_expect, 3), f"probs shape {tuple(d['probs'].shape)} != ({n_expect}, 3)")
    expect(torch.allclose(d["probs"].sum(1), torch.ones(len(d["probs"])), atol=1e-3), "probs rows do not sum to 1")
    expect(bool(((d["targets"] >= 0) & (d["targets"] < 3)).all()), "targets out of range")
    counts = torch.bincount(d["targets"].long(), minlength=3).tolist()
    expect(counts == [n_expect // 3] * 3, f"test class balance {counts} (labels must NOT be permuted on test)")
    expect(bool(torch.equal(d["probs"].argmax(1), d["preds"].long())), "preds != argmax(probs)")
if len(js) == 1:
    rows = json.load(open(js[0]))
    expect(len(rows) == n_expect, f"json rows {len(rows)} != {n_expect}")
    got = {r["subject"] for r in rows}
    expect(got == set(test_subjects), f"predicted subjects {sorted(got)} != split test subjects {test_subjects}")
    expect(all(r["correct"] == int(r["true_label"] == r["predicted_label"]) for r in rows), "json 'correct' flag inconsistent")
    expect(not (got & set(train_subjects)), "predictions include TRAIN subjects (split leakage)")
    s = json.load(open(sm[0]))
    expect(s["fold_id"] == int(m.group(1)), f"summary fold_id={s['fold_id']} != {m.group(1)}")
    expect(s["num_samples"] == n_expect, f"summary num_samples={s['num_samples']}")

if errs:
    print(f"[{a.project}] FAILED:\n   - " + "\n   - ".join(errs)); sys.exit(1)
print(f"[{a.project}] ok: epoch {ck['epoch']}, best ckpt {os.path.basename(best[0])}, "
      f"{n_expect} test clips from {test_subjects}, encoder median rel-dist from pretrained "
      f"{sorted(rel)[len(rel) // 2]:.3f}")

#!/usr/bin/env python
"""Check the REAL pretrained checkpoint against the REAL fine-tune architecture.

main.py loads the checkpoint with load_state_dict(strict=False), which silently
ignores missing/unexpected keys -- a key mismatch would mean "fine-tuning" is
secretly training from scratch with no error. This builds the full-size model
the launchers use (scripts/configs/models/neurostorm.yaml: embed 36, depths
2 2 6 2, heads 3 6 12 24, 96^3 x 37, window 4) on CPU and reports key coverage
and shape mismatches. Needs repo (+ mamba shim if mamba_ssm is absent) on PYTHONPATH.

Usage: check_real_ckpt_compat.py [CKPT]
"""
import sys

import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt"

from models.neurostorm import NeuroSTORM  # noqa: E402

net = NeuroSTORM(
    img_size=[96, 96, 96, 37], in_chans=1, embed_dim=36, window_size=[4, 4, 4, 4],
    first_window_size=[4, 4, 4, 4], patch_size=[6, 6, 6, 1], depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24], c_multiplier=2, last_layer_full_MSA=True,
    drop_rate=0.1, drop_path_rate=0.1, attn_drop_rate=0.1,
)
target = net.state_dict()

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
src = {k.removeprefix("model."): v for k, v in ck["state_dict"].items() if "model." in k}

missing = sorted(set(target) - set(src))
unexpected = sorted(set(src) - set(target))
shape_bad = [(k, tuple(src[k].shape), tuple(target[k].shape))
             for k in set(src) & set(target) if src[k].shape != target[k].shape]
# time_embed is interpolated by main.py when T differs, so it is not an error
shape_bad_real = [x for x in shape_bad if not x[0].endswith("time_embed")]
n_loaded = len(set(src) & set(target)) - len(shape_bad_real)

print(f"checkpoint: {ckpt_path}")
print(f"  checkpoint tensors under 'model.': {len(src)} | model tensors: {len(target)}")
print(f"  loadable (key+shape match): {n_loaded}/{len(target)}")
print(f"  missing from ckpt  : {len(missing)} {missing[:6]}")
print(f"  unexpected in ckpt : {len(unexpected)} {unexpected[:6]}")
print(f"  shape mismatches   : {len(shape_bad_real)} {shape_bad_real[:4]}  (+{len(shape_bad) - len(shape_bad_real)} time_embed, auto-interpolated)")
sys.exit(0 if not shape_bad_real and n_loaded > 0.9 * len(target) else 1)

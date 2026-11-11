#!/usr/bin/env python
"""Write a tiny stand-in for pretrained_models/pretraining/pt_neurostorm_mae_5ds.ckpt.

Same file format main.py expects ({'state_dict': {'model.<param>': tensor}}),
but sized to match the sandbox's tiny architecture so the launchers' real
--load_model_path code path (main.py: "loading model from ...") is exercised.
Weights are seeded random, so a fine-tune that really loaded them starts from
these values (run_greatlakes_checks.sh verifies this).

Usage: make_tiny_pretrained_ckpt.py --out PATH [architecture flags, same as main.py]
"""
import argparse
import os

import torch

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--embed_dim", type=int, default=12)
p.add_argument("--depths", type=int, nargs="+", default=[1, 1])
p.add_argument("--num_heads", type=int, nargs="+", default=[2, 4])
p.add_argument("--c_multiplier", type=int, default=2)
p.add_argument("--img_size", type=int, nargs="+", default=[24, 24, 24, 6])
p.add_argument("--window_size", type=int, nargs="+", default=[2, 2, 2, 2])
p.add_argument("--first_window_size", type=int, nargs="+", default=None)
p.add_argument("--patch_size", type=int, nargs="+", default=[6, 6, 6, 1])
a = p.parse_args()

from models.neurostorm import NeuroSTORM  # noqa: E402  (needs repo + shim on PYTHONPATH)

torch.manual_seed(1234)
net = NeuroSTORM(
    img_size=a.img_size, in_chans=1, embed_dim=a.embed_dim, window_size=a.window_size,
    first_window_size=a.first_window_size or a.window_size, patch_size=a.patch_size, depths=a.depths,
    num_heads=a.num_heads, c_multiplier=a.c_multiplier, last_layer_full_MSA=False,
    drop_rate=0.1, drop_path_rate=0.1, attn_drop_rate=0.1,
)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save({"state_dict": {"model." + k: v for k, v in net.state_dict().items()}}, a.out)
print(f"tiny pretrained ckpt: {a.out} ({sum(v.numel() for v in net.state_dict().values())} values)")

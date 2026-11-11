"""Pure-PyTorch stand-in for ``mamba_ssm`` -- LOCAL SANDBOX USE ONLY.

The real ``mamba_ssm`` package needs CUDA kernels and cannot be installed on a
Mac. This shim implements the same ``Mamba`` module (selective SSM, Gu & Dao
2023) with a plain sequential scan so ``models/neurostorm.py`` can be imported
and trained on CPU/MPS for smoke tests.

Properties:
  * Parameter names/shapes match ``mamba_ssm.Mamba`` (in_proj, conv1d, x_proj,
    dt_proj, A_log, D, out_proj), so GPU-trained checkpoints load into it.
  * Numerically equivalent in spirit but NOT bit-identical to the CUDA kernel.
  * Slow (python loop over the sequence). Use tiny configs only.
  * Never put this directory on PYTHONPATH on the cluster -- it would shadow
    the real, fast implementation. ``scripts/sandbox/run_smoke.sh`` adds it
    only for the local run.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Mamba", "Mamba2", "IS_SANDBOX_SHIM"]

IS_SANDBOX_SHIM = True


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        **_ignored,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=conv_bias, **factory,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory)

        # dt bias init so softplus(dt) lands in [dt_min, dt_max] (as upstream)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory)

    def forward(self, hidden_states, inference_params=None):
        # hidden_states: (B, L, d_model)
        _, L, _ = hidden_states.shape
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)  # (B, L, d_inner) each

        x = self.conv1d(x.transpose(1, 2))[..., :L]  # causal depthwise conv
        x = F.silu(x).transpose(1, 2)                # (B, L, d_inner)

        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))            # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())           # (d_inner, d_state)

        # Discretise: dA = exp(dt*A), dBx = dt*B*x   -> (B, L, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = (dt * x).unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D     # (B, L, d_inner)

        return self.out_proj(y * F.silu(z))


class Mamba2(nn.Module):
    """Placeholder so ``from mamba_ssm import Mamba, Mamba2`` succeeds."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2 is not implemented in the local sandbox shim. "
            "NeuroSTORM uses Mamba (v1); run Mamba2 code on the cluster."
        )

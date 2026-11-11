"""Tests for the pure-PyTorch mamba_ssm stand-in used by the local sandbox."""
import importlib.util
import os

import pytest
import torch

SHIM = os.path.join(os.path.dirname(__file__), "..", "scripts", "sandbox", "shims", "mamba_ssm", "__init__.py")


@pytest.fixture(scope="module")
def shim():
    # Load by path so this works whether or not the real mamba_ssm is installed.
    spec = importlib.util.spec_from_file_location("_mamba_shim", SHIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_shape_and_grad(shim):
    m = shim.Mamba(d_model=8, d_state=4, d_conv=4, expand=2)
    x = torch.randn(2, 10, 8, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.unit
def test_is_causal(shim):
    m = shim.Mamba(d_model=8, d_state=4).eval()
    x = torch.randn(1, 10, 8)
    x2 = x.clone()
    x2[:, 6:] += 1.0  # perturb the future only
    assert torch.allclose(m(x)[:, :6], m(x2)[:, :6], atol=1e-5)


@pytest.mark.unit
def test_param_names_match_real_mamba(shim):
    names = {n for n, _ in shim.Mamba(d_model=8).named_parameters()}
    assert names == {"in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight",
                     "dt_proj.weight", "dt_proj.bias", "A_log", "D", "out_proj.weight"}

"""NEUROSTORM_ENV: only the exact value "local" may change behaviour."""
import pytest
import torch

from utils.runtime_env import is_local


@pytest.mark.unit
@pytest.mark.parametrize("val,expected", [("local", True), ("LOCAL", True), ("greatlakes", False),
                                          ("", False), ("localhost", False), ("cuda", False)])
def test_only_local_enables_local_mode(monkeypatch, val, expected):
    monkeypatch.setenv("NEUROSTORM_ENV", val)
    assert is_local() is expected


@pytest.mark.unit
def test_unset_is_not_local(monkeypatch):
    monkeypatch.delenv("NEUROSTORM_ENV", raising=False)
    assert is_local() is False


@pytest.mark.unit
@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA failure mode")
@pytest.mark.parametrize("val", [None, "greatlakes"])
def test_cluster_mode_keeps_original_nvtx_calls(monkeypatch, val):
    """Off-cluster, non-local mode must hit the ORIGINAL (unguarded) NVTX call."""
    from models.patchembedding import PatchEmbed
    if val is None:
        monkeypatch.delenv("NEUROSTORM_ENV", raising=False)
    else:
        monkeypatch.setenv("NEUROSTORM_ENV", val)
    pe = PatchEmbed(img_size=(12, 12, 12, 2), patch_size=(6, 6, 6, 1), in_chans=1, embed_dim=8)
    with pytest.raises(RuntimeError, match="NVTX"):
        pe(torch.zeros(1, 1, 12, 12, 12, 2))


@pytest.mark.unit
def test_local_mode_skips_nvtx(monkeypatch):
    from models.patchembedding import PatchEmbed
    monkeypatch.setenv("NEUROSTORM_ENV", "local")
    pe = PatchEmbed(img_size=(12, 12, 12, 2), patch_size=(6, 6, 6, 1), in_chans=1, embed_dim=8)
    assert pe(torch.zeros(1, 1, 12, 12, 12, 2)).shape[0] == 1

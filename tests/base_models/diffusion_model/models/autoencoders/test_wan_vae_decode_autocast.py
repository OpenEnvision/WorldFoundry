"""Tests for WanVideoVAE.decode_autocast_dtype opt-in mixed-precision decode.

Profiling the real Wan2.2 decode showed conv is 76.8% of CUDA time and runs in
FP32. Routing conv+matmul through autocast (bf16/fp16) while keeping the fragile
RMS_norm/normalize reductions in FP32 gives ~1.3x decode at cos>=0.9999. These
tests lock the seam contract without needing a GPU or the real checkpoint:
- default (None) is a no-op nullcontext so the shipped FP32 path is unchanged;
- setting a dtype produces a torch.autocast context targeting that dtype.
"""

from __future__ import annotations

import contextlib

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE


def _bare_vae() -> WanVideoVAE:
    # __new__ + manual attr set avoids constructing the heavy VideoVAE_ backbone;
    # we only exercise the autocast-context seam, which depends on one attribute.
    vae = WanVideoVAE.__new__(WanVideoVAE)
    vae.decode_autocast_dtype = None
    return vae


def test_default_is_nullcontext() -> None:
    vae = _bare_vae()
    ctx = vae._decode_autocast()
    assert isinstance(ctx, contextlib.nullcontext)


def test_missing_attribute_falls_back_to_nullcontext() -> None:
    # getattr default guards checkpoints deserialized before the attr existed.
    vae = WanVideoVAE.__new__(WanVideoVAE)
    assert isinstance(vae._decode_autocast(), contextlib.nullcontext)


def test_dtype_yields_autocast_context() -> None:
    for dt in (torch.float16, torch.bfloat16):
        vae = _bare_vae()
        vae.decode_autocast_dtype = dt
        ctx = vae._decode_autocast()
        assert isinstance(ctx, torch.autocast)
        assert ctx.fast_dtype == dt
        assert ctx.device == "cuda"


def test_redundant_resident_dtype_elides_autocast_context() -> None:
    vae = _bare_vae()
    torch.nn.Module.__init__(vae)
    vae.model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    vae.decode_autocast_dtype = torch.bfloat16

    assert vae._decode_autocast_is_redundant(torch.bfloat16) is True
    assert isinstance(
        vae._decode_autocast(torch.bfloat16),
        contextlib.nullcontext,
    )
    assert isinstance(vae._decode_autocast(torch.float32), torch.autocast)


def test_init_default_is_off() -> None:
    # A real (light z_dim) construct: the shipped default must keep FP32 decode.
    vae = WanVideoVAE(z_dim=4)
    assert vae.decode_autocast_dtype is None
    assert isinstance(vae._decode_autocast(), contextlib.nullcontext)


def _resolve(**options):
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
        _resolve_vae_decode_autocast,
    )
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    return _resolve_vae_decode_autocast(RuntimePolicy(options=options))


def test_policy_absent_keeps_fp32() -> None:
    assert _resolve() is None
    assert _resolve(vae_decode_autocast=None) is None
    assert _resolve(vae_decode_autocast=False) is None
    assert _resolve(vae_decode_autocast="") is None


def test_policy_string_aliases() -> None:
    assert _resolve(vae_decode_autocast="fp16") is torch.float16
    assert _resolve(vae_decode_autocast="half") is torch.float16
    assert _resolve(vae_decode_autocast="BF16") is torch.bfloat16
    assert _resolve(vae_decode_autocast="bfloat16") is torch.bfloat16


def test_policy_accepts_torch_dtype() -> None:
    assert _resolve(vae_decode_autocast=torch.float16) is torch.float16


def test_policy_true_uses_runtime_low_precision_or_bf16() -> None:
    assert _resolve(vae_decode_autocast=True) is torch.bfloat16


def test_policy_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError):
        _resolve(vae_decode_autocast="int8")


def test_policy_rejects_fp32() -> None:
    import pytest

    with pytest.raises(ValueError, match="only fp16 or bf16"):
        _resolve(vae_decode_autocast="fp32")

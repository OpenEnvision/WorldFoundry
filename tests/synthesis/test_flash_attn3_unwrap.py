from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.unit


def _make_fa3_stub(return_tuple: bool):
    def fake_varlen_func(*, q, k, v, **kwargs):
        del k, v, kwargs
        out = q + 1.0
        return (out, None) if return_tuple else out

    return SimpleNamespace(flash_attn_varlen_func=fake_varlen_func)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required by varlen flash_attention device assertion"
)
@pytest.mark.parametrize("return_tuple", [False, True])
def test_flash_attn3_varlen_handles_tensor_or_tuple_return(monkeypatch: pytest.MonkeyPatch, return_tuple: bool) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import (
        dreamzero,
        geometry_attention,
        linear_attention,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamzero import (
        attention as dreamzero_attention,
    )
    from worldfoundry.synthesis.visual_generation.alayaworld_v11.alayaworld_v11_runtime.ltx2.modules import (
        attention as alaya_attention,
    )
    from worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime.models import (
        dit_model as infinite_dit,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.wan22.runtime.modeling import (
        attention as solarwm_attention,
    )

    del dreamzero
    stub = _make_fa3_stub(return_tuple)
    q = torch.randn((1, 4, 2, 16), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 4, 2, 16), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 4, 2, 16), device="cuda", dtype=torch.bfloat16)

    for module, fa3_attr, flag_attr in (
        (infinite_dit, "flash_attn_interface", "FLASH_ATTN_3_AVAILABLE"),
        (geometry_attention, "flash_attn_interface", "FLASH_ATTN_3_AVAILABLE"),
        (linear_attention, "flash_attn_interface", "FLASH_ATTN_3_AVAILABLE"),
        (solarwm_attention, "flash_attn_interface", "FLASH_ATTN_3_AVAILABLE"),
        (alaya_attention, "_flash_attn_interface", "_FLASH_ATTN_3_AVAILABLE"),
    ):
        monkeypatch.setattr(module, flag_attr, True)
        monkeypatch.setattr(module, fa3_attr, stub, raising=False)
        out = module.flash_attention(q, k, v, version=3)
        assert out.shape == q.shape
        torch.testing.assert_close(out, q + 1.0)

    # dreamzero/attention delegates version=3 through the shared varlen dispatcher; verify it runs on CUDA cleanly.
    out = dreamzero_attention.flash_attention(q, k, v, version=3)
    assert out.shape == q.shape
    assert out.dtype == q.dtype

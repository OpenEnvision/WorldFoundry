"""CPU camera attention stays usable when CUDA providers are installed."""

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import camera_attention


@pytest.mark.parametrize("provider", ["fa2", "fa3", "sage"])
@pytest.mark.parametrize("entry", ["attention", "flash_attention"])
def test_cpu_camera_attention_with_installed_cuda_provider_matches_sdpa(monkeypatch, provider, entry):
    monkeypatch.setattr(camera_attention, "FLASH_ATTN_2_AVAILABLE", provider == "fa2")
    monkeypatch.setattr(camera_attention, "FLASH_ATTN_3_AVAILABLE", provider == "fa3")
    monkeypatch.setattr(camera_attention, "SAGEATTN_AVAILABLE", provider == "sage")

    def cuda_only_provider(*args, **kwargs):
        raise AssertionError("CPU camera attention dispatched to a CUDA provider")

    monkeypatch.setattr(camera_attention, "flash_attn_func", cuda_only_provider, raising=False)
    monkeypatch.setattr(camera_attention, "sageattn_func", cuda_only_provider)
    generator = torch.Generator().manual_seed(37)
    query = torch.randn(2, 7, 2, 8, generator=generator).bfloat16()
    key, value = [torch.randn(2, 9, 2, 8, generator=generator).bfloat16() for _ in range(2)]
    expected = F.scaled_dot_product_attention(
        (query * 0.75).float().transpose(1, 2),
        key.float().transpose(1, 2),
        value.float().transpose(1, 2),
        scale=0.2,
    ).transpose(1, 2).to(query.dtype)

    with torch.no_grad():
        actual = getattr(camera_attention, entry)(query, key, value, q_scale=0.75, softmax_scale=0.2)

    assert actual.device.type == "cpu" and actual.dtype == query.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

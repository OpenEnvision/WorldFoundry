from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.attention import native


@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive", "head", "batch"])
@pytest.mark.parametrize("broadcast", [False, True])
def test_old_sdpa_gqa_matches_explicit_heads(monkeypatch, mask_kind, broadcast):
    torch.manual_seed(3)
    q = torch.randn(2, 6, 5, 8)
    k = torch.randn(1 if broadcast else 2, 2, 7, 8)
    v = torch.randn(1 if broadcast else 2, 2, 7, 4)
    mask = None
    if mask_kind != "none":
        shape = {"head": (2, 6, 5, 7), "batch": (2, 1, 5, 7)}.get(mask_kind, (5, 7))
        mask = torch.rand(shape) > 0.3
        mask[..., 0, :] = False
        if mask_kind == "additive":
            mask = torch.zeros(shape).masked_fill(~mask, float("-inf"))
    sdpa = F.scaled_dot_product_attention
    expected = sdpa(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=mask, scale=0.25)
    calls = []

    def old_sdpa(query, key, value, **kwargs):
        if "enable_gqa" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_gqa'")
        calls.append((key, value))
        return sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", old_sdpa)
    actual = native.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, backend="math", scale=0.25)
    torch.testing.assert_close(actual, expected)
    if not broadcast and mask_kind in {"none", "boolean", "additive"}:
        assert calls[0][0].untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        assert calls[0][1].untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        assert calls[0][0].stride(1) == 0


def test_unrelated_type_error_is_not_retried(monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise TypeError("invalid mask")

    monkeypatch.setattr(F, "scaled_dot_product_attention", broken)
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(TypeError, match="invalid mask"):
        native.scaled_dot_product_attention(q, q, q, enable_gqa=True)
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mask_kind", ["none", "boolean", "head"])
def test_efficient_cuda_gqa_matches_math(mask_kind):
    q = torch.randn(2, 6, 16, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, 24, 32, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    mask = None
    if mask_kind != "none":
        shape = (16, 24) if mask_kind == "boolean" else (2, 6, 16, 24)
        mask = torch.rand(shape, device="cuda") > 0.2
    with torch.no_grad():
        actual = native.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, backend="efficient")
        expected = native.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, backend="math")
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)

"""CPU-only tests for the CC-09 fix: xFormers bool-mask semantics.

``XFormersAttention`` receives SDPA-style masks. A bool mask must be
converted into an additive 0/-inf bias before being handed to
``memory_efficient_attention``; the previous code cast True/False to 1.0/0.0,
which effectively disabled masking. xformers itself is not required: the
kernel entry point is replaced with a NumPy-free reference implementation.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch


@pytest.fixture()
def fake_xformers(monkeypatch):
    """Install a fake ``xformers.ops.memory_efficient_attention`` that records calls."""

    recorded: dict[str, object] = {}

    def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0):
        # xformers layout: [B, M, H, K]
        assert p == 0.0
        recorded["attn_bias"] = attn_bias
        scale = 1.0 / math.sqrt(q.shape[-1])
        qh = q.permute(0, 2, 1, 3).float()
        kh = k.permute(0, 2, 1, 3).float()
        vh = v.permute(0, 2, 1, 3).float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        if attn_bias is not None:
            scores = scores + attn_bias.float()
        out = torch.matmul(torch.softmax(scores, dim=-1), vh)
        return out.permute(0, 2, 1, 3).to(v.dtype)

    ops_module = types.ModuleType("xformers.ops")
    ops_module.memory_efficient_attention = memory_efficient_attention
    xformers_module = types.ModuleType("xformers")
    xformers_module.ops = ops_module
    monkeypatch.setitem(sys.modules, "xformers", xformers_module)
    monkeypatch.setitem(sys.modules, "xformers.ops", ops_module)
    return recorded


def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)


def test_bool_mask_becomes_additive_neg_inf_bias(fake_xformers):
    from worldfoundry.core.attention.model_backends import XFormersAttention

    torch.manual_seed(0)
    batch, heads, seq, dim = 1, 2, 5, 4
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    bool_mask = torch.ones(seq, seq, dtype=torch.bool)
    bool_mask[:, -2:] = False  # drop the last two keys, keep >=1 True per row

    out = XFormersAttention()(
        _flatten_heads(q),
        _flatten_heads(k),
        _flatten_heads(v),
        heads,
        mask=bool_mask,
    )

    bias = fake_xformers["attn_bias"]
    assert bias is not None and bias.dtype == q.dtype
    assert bias.shape == (batch, heads, seq, seq)
    assert torch.isneginf(bias[..., -2:]).all(), "masked-out keys must receive -inf"
    assert (bias[..., :-2] == 0).all(), "attended keys must receive exactly 0 bias"

    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
    out_heads = out.view(batch, seq, heads, dim).permute(0, 2, 1, 3)
    torch.testing.assert_close(out_heads, reference, rtol=1e-4, atol=1e-5)


def test_bool_mask_actually_masks(fake_xformers):
    """Changing the content of masked-out keys must not change the output."""

    from worldfoundry.core.attention.model_backends import XFormersAttention

    torch.manual_seed(1)
    batch, heads, seq, dim = 1, 1, 4, 8
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    bool_mask = torch.ones(seq, seq, dtype=torch.bool)
    bool_mask[:, 0] = False

    attention = XFormersAttention()
    out_a = attention(_flatten_heads(q), _flatten_heads(k), _flatten_heads(v), heads, mask=bool_mask)

    k_perturbed = k.clone()
    v_perturbed = v.clone()
    k_perturbed[:, :, 0] += 100.0
    v_perturbed[:, :, 0] -= 50.0
    out_b = attention(
        _flatten_heads(q), _flatten_heads(k_perturbed), _flatten_heads(v_perturbed), heads, mask=bool_mask
    )

    torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-6)


def test_float_mask_values_pass_through_unchanged(fake_xformers):
    from worldfoundry.core.attention.model_backends import XFormersAttention

    torch.manual_seed(2)
    batch, heads, seq, dim = 1, 2, 6, 4
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    float_mask = torch.randn(seq, seq)

    XFormersAttention()(
        _flatten_heads(q),
        _flatten_heads(k),
        _flatten_heads(v),
        heads,
        mask=float_mask,
    )
    bias = fake_xformers["attn_bias"]
    assert bias.shape == (batch, heads, seq, seq)
    torch.testing.assert_close(bias[0, 0], float_mask, rtol=0, atol=0)


def test_mask_width_not_multiple_of_8_is_padded_storage_only(fake_xformers):
    """Logical mask width stays intact when the key length is not 8-aligned."""

    from worldfoundry.core.attention.model_backends import XFormersAttention

    torch.manual_seed(3)
    batch, heads, seq, dim = 1, 1, 5, 4  # 5 keys -> storage padded to 8
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    bool_mask = torch.ones(seq, seq, dtype=torch.bool)
    bool_mask[0, 1] = False

    out = XFormersAttention()(
        _flatten_heads(q), _flatten_heads(k), _flatten_heads(v), heads, mask=bool_mask
    )
    bias = fake_xformers["attn_bias"]
    assert bias.shape[-1] == seq
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
    out_heads = out.view(batch, seq, heads, dim).permute(0, 2, 1, 3)
    torch.testing.assert_close(out_heads, reference, rtol=1e-4, atol=1e-5)

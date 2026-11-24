"""Symbolic causal masks must retain native SDPA and fully-masked-row semantics."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import CausalBias, causal_lower_right, causal_upper_left

from worldfoundry.core.attention import varlen_scaled_dot_product_attention
from worldfoundry.core.attention.backends import native


def _causal_masks(variant: str, q_length: int, k_length: int) -> tuple[CausalBias, torch.Tensor]:
    factory = causal_lower_right if variant == "lower_right" else causal_upper_left
    diagonal = k_length - q_length if variant == "lower_right" else 0
    return factory(q_length, k_length), torch.ones(q_length, k_length, dtype=torch.bool).tril(diagonal)


@pytest.mark.parametrize("variant", ["lower_right", "upper_left"])
@pytest.mark.parametrize(("q_length", "k_length"), [(3, 5), (5, 3), (3, 3)])
@pytest.mark.parametrize("flattened", [False, True])
def test_native_causal_bias_matches_boolean_mask(
    variant: str,
    q_length: int,
    k_length: int,
    flattened: bool,
) -> None:
    torch.manual_seed(13)
    query = torch.randn(2, 2, q_length, 8)
    key = torch.randn(2, 2, k_length, 8)
    value = torch.randn(2, 2, k_length, 4)
    bias, mask = _causal_masks(variant, q_length, k_length)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=0.25)
    # Establish that upstream SDPA accepts the very same symbolic object.
    torch.testing.assert_close(F.scaled_dot_product_attention(query, key, value, attn_mask=bias, scale=0.25), expected)
    if flattened:
        actual = native.flattened_multihead_attention(
            query.transpose(1, 2).reshape(2, q_length, 16),
            key.transpose(1, 2).reshape(2, k_length, 16),
            value.transpose(1, 2).reshape(2, k_length, 8),
            num_heads=2,
            attn_mask=bias,
            backend="math",
            scale=0.25,
        )
        actual = actual.reshape(2, q_length, 2, 4).transpose(1, 2)
    else:
        actual = native.scaled_dot_product_attention(query, key, value, attn_mask=bias, backend="math", scale=0.25)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    if variant == "lower_right" and q_length > k_length:
        assert torch.count_nonzero(actual[..., : q_length - k_length, :]) == 0


@pytest.mark.parametrize("variant", ["lower_right", "upper_left"])
@pytest.mark.parametrize(("q_length", "k_length"), [(3, 5), (5, 3), (3, 3)])
@pytest.mark.parametrize("masked_value", [float("nan"), 7.0])
def test_normalize_causal_bias_uses_geometry_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    q_length: int,
    k_length: int,
    masked_value: float,
) -> None:
    bias, mask = _causal_masks(variant, q_length, k_length)
    query = torch.empty(2, 2, q_length, 8)
    key = torch.empty(2, 2, k_length, 8)
    output = torch.randn(2, 2, q_length, 4)
    empty_rows = ~mask.any(dim=-1)
    output[..., empty_rows, :] = masked_value
    expected = output.clone()
    expected[..., empty_rows, :] = 0

    def reject_materialize(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("normalization must not materialize an L x S mask")

    monkeypatch.setattr(CausalBias, "_materialize", reject_materialize)
    actual = native.normalize_fully_masked_rows(output, bias, query, key)
    torch.testing.assert_close(actual, expected)
    if not empty_rows.any():
        assert actual is output


@pytest.mark.parametrize("mask_rank", [2, 3, 4])
def test_flattened_boolean_mask_preserves_zero_rows(mask_rank: int) -> None:
    torch.manual_seed(17)
    query = torch.randn(2, 5, 16)
    key = torch.randn(2, 3, 16)
    value = torch.randn_like(key)
    mask = torch.ones(5, 3, dtype=torch.bool).tril(-2)
    if mask_rank >= 3:
        mask = mask.expand(2, 5, 3).clone()
        mask[1, 3] = False
    reference_mask = mask.unsqueeze(1) if mask_rank == 3 else mask
    if mask_rank == 4:
        mask = mask.unsqueeze(1).expand(2, 2, 5, 3).clone()
        mask[0, 1, 4] = False
        reference_mask = mask
    expected = (
        F.scaled_dot_product_attention(
            query.reshape(2, 5, 2, 8).transpose(1, 2),
            key.reshape(2, 3, 2, 8).transpose(1, 2),
            value.reshape(2, 3, 2, 8).transpose(1, 2),
            attn_mask=reference_mask,
        )
        .transpose(1, 2)
        .reshape(2, 5, 16)
    )
    actual = native.flattened_multihead_attention(query, key, value, num_heads=2, attn_mask=mask, backend="math")
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, :2]) == 0


@pytest.mark.parametrize("variant", ["lower_right", "upper_left"])
@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
def test_native_causal_bias_gqa_matches_explicit_heads(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    compatible: bool,
    broadcast: bool,
) -> None:
    torch.manual_seed(14)
    query = torch.randn(2, 6, 5, 8)
    key = torch.randn(1 if broadcast else 2, 2, 3, 8)
    value = torch.randn(1 if broadcast else 2, 2, 3, 4)
    bias, mask = _causal_masks(variant, 5, 3)
    expected = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(3, 1),
        value.repeat_interleave(3, 1),
        attn_mask=mask,
    )
    if compatible:
        # Exercise the efficient-only adapter with CPU SDPA as its kernel.
        monkeypatch.setattr(native, "_sdpa_kernel_context", lambda **kwargs: nullcontext())
    actual = native.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=bias,
        enable_gqa=True,
        backend="efficient" if compatible else "math",
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("variant", ["lower_right", "upper_left"])
@pytest.mark.parametrize("flattened", [False, True])
def test_native_causal_bias_fullgraph(variant: str, flattened: bool) -> None:
    torch.manual_seed(15)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn_like(key)
    bias, mask = _causal_masks(variant, 5, 3)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    if flattened:
        query = query.transpose(1, 2).reshape(1, 5, 16)
        key = key.transpose(1, 2).reshape(1, 3, 16)
        value = value.transpose(1, 2).reshape(1, 3, 16)
        expected = expected.transpose(1, 2).reshape(1, 5, 16)

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if flattened:
            return native.flattened_multihead_attention(q, k, v, num_heads=2, attn_mask=bias, backend="math")
        return native.scaled_dot_product_attention(q, k, v, attn_mask=bias, backend="math")

    compiled = torch.compile(attention, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(query, key, value), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_varlen_packed_causal_automatically_uses_symbolic_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.attention.sequence import varlen

    monkeypatch.setenv("WORLDFOUNDRY_VARLEN_JAGGED_MIN_BATCH", "32")
    torch.manual_seed(16)
    lengths = [(3, 5), (5, 3), (3, 3)]
    query = torch.randn(11, 2, 32, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seen: list[CausalBias] = []
    sdpa = varlen.scaled_dot_product_attention

    def record_bias(*args: Any, **kwargs: Any) -> torch.Tensor:
        assert isinstance(kwargs["attn_mask"], CausalBias)
        seen.append(kwargs["attn_mask"])
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(varlen, "scaled_dot_product_attention", record_bias)
    actual = varlen_scaled_dot_product_attention(
        query,
        key,
        value,
        cu_seqlens_q=torch.tensor([0, 3, 8, 11], device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 5, 8, 11], device="cuda", dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_k=5,
        causal=True,
    )
    expected = []
    q_start = k_start = 0
    for q_length, k_length in lengths:
        mask = torch.ones(q_length, k_length, device="cuda", dtype=torch.bool).tril(k_length - q_length)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                query[q_start : q_start + q_length].transpose(0, 1).unsqueeze(0).float(),
                key[k_start : k_start + k_length].transpose(0, 1).unsqueeze(0).float(),
                value[k_start : k_start + k_length].transpose(0, 1).unsqueeze(0).float(),
                attn_mask=mask,
            )
        expected.append(output.squeeze(0).transpose(0, 1).to(query.dtype))
        q_start += q_length
        k_start += k_length
    assert [(bias.seq_len_q, bias.seq_len_kv) for bias in seen] == lengths
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.002, rtol=0.002)
    assert torch.count_nonzero(actual[3:5]) == 0
    assert torch.isfinite(actual).all()

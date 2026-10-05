from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.magi2 import ops


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    head_dim = q.shape[-1]
    sink_logits = torch.logsumexp(sinks.float(), dim=0) if sinks.ndim == 2 else sinks.float()
    chunks = []
    bounds = cu_seqlens.cpu().tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        qs = q[start:stop].float().transpose(0, 1)
        ks = k[start:stop].float().transpose(0, 1)
        vs = v[start:stop].float().transpose(0, 1)
        scores = qs @ ks.transpose(-1, -2) * (head_dim**-0.5)
        sink_col = sink_logits[:, None, None].expand(-1, stop - start, 1)
        probs = torch.softmax(torch.cat((scores, sink_col), dim=-1), dim=-1)[..., : stop - start]
        chunks.append((probs @ vs).transpose(0, 1))
    return torch.cat(chunks)


def test_attention_with_sink_cpu_matches_full_reference() -> None:
    torch.manual_seed(7)
    q = torch.randn(9, 3, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sinks = torch.randn(2, 3)
    cu_seqlens = torch.tensor([0, 5, 9], dtype=torch.int32)

    actual = ops.attention_with_sink(q, k, v, sinks, cu_seqlens=cu_seqlens)
    expected = _reference_attention(q, k, v, sinks, cu_seqlens)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the FA3 path")
def test_attention_with_sink_fa3_matches_full_reference() -> None:
    if ops._flash_attn_varlen_func() is None:
        pytest.skip("flash_attn_interface is not installed")
    torch.manual_seed(11)
    device = torch.device("cuda")
    q = torch.randn(12, 4, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sinks = torch.randn(2, 4, device=device)
    cu_seqlens = torch.tensor([0, 7, 12], device=device, dtype=torch.int32)

    actual = ops.attention_with_sink(q, k, v, sinks, cu_seqlens=cu_seqlens)
    expected = _reference_attention(q, k, v, sinks, cu_seqlens)

    assert ops.can_use_fused_attn_sink(q, k, v)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def _moe_inputs(device: torch.device, dtype: torch.dtype):
    torch.manual_seed(19)
    tokens, heads, experts, top_k = 6, 2, 3, 2
    d_head, d_expert = 8, 12
    x = torch.randn(tokens, heads, d_head, device=device, dtype=dtype)
    topk_indices = torch.tensor(
        [
            [[0, 2], [1, 0], [2, 1], [0, 1], [2, 0], [1, 2]],
            [[2, 0], [0, 1], [1, 2], [2, 1], [0, 2], [1, 0]],
        ],
        device=device,
    )
    topk_probs = torch.rand(heads, tokens, top_k, device=device)
    topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
    gather_ids, probs, offsets = ops.flash_mh_moe_global_sort(
        topk_probs, topk_indices, experts
    )
    weights = (
        torch.randn(heads * experts, d_head, d_expert, device=device, dtype=dtype),
        torch.randn(heads * experts, d_head, d_expert, device=device, dtype=dtype),
        torch.randn(heads * experts, d_expert, d_head, device=device, dtype=dtype),
    )
    return x, gather_ids, probs, offsets, weights


def _explicit_moe_reference(x, gather_ids, probs, offsets, weights):
    W_gate, W_up, W_down = weights
    heads = x.shape[1]
    experts = W_gate.shape[0] // heads
    result = torch.zeros_like(x, dtype=torch.float32)
    bounds = offsets.cpu().tolist()
    for expert, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        if stop <= start:
            continue
        rows = gather_ids[start:stop].long()
        head = expert // experts
        xe = x[rows, head].float()
        gate = xe @ W_gate[expert].float()
        up = xe @ W_up[expert].float()
        gate = gate.clamp(max=ops._SWIGLU7_LIMIT)
        up = up.clamp(min=-ops._SWIGLU7_LIMIT, max=ops._SWIGLU7_LIMIT)
        hidden = gate * torch.sigmoid(ops._SWIGLU7_ALPHA * gate) * (up + 1.0)
        contribution = hidden @ W_down[expert].float()
        contribution *= probs[start:stop, None].float()
        result[:, head].index_add_(0, rows, contribution)
    return result


def test_mh_moe_cpu_scatter_adds_routed_experts() -> None:
    inputs = _moe_inputs(torch.device("cpu"), torch.float32)
    x, gather_ids, probs, offsets, weights = inputs

    actual = ops.flash_mh_moe_fwd(
        x, gather_ids, probs, offsets, *weights
    )
    expected = _explicit_moe_reference(x, gather_ids, probs, offsets, weights)

    assert torch.count_nonzero(actual) > 0
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for grouped MoE")
def test_mh_moe_grouped_cuda_matches_explicit_reference() -> None:
    inputs = _moe_inputs(torch.device("cuda"), torch.bfloat16)
    x, gather_ids, probs, offsets, weights = inputs

    actual = ops.flash_mh_moe_fwd(
        x, gather_ids, probs, offsets, *weights
    )
    expected = _explicit_moe_reference(x, gather_ids, probs, offsets, weights)

    assert ops.can_use_fused_mh_moe(x, *weights)
    torch.testing.assert_close(actual.float(), expected, atol=0.5, rtol=2e-2)

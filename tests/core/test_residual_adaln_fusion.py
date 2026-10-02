from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.kernels import residual_layer_norm_scale_shift


def reference(residual, update, scale, shift, gate=None, output_gate=None):
    summed = residual + (update if gate is None else update * gate)
    normalized = F.layer_norm(summed, (summed.shape[-1],), eps=1e-6) * (1 + scale) + shift
    result = (normalized, summed)
    return result if output_gate is None else (*result, output_gate.expand_as(summed).contiguous())


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_fallback_preserves_values_gradients_and_inputs(monkeypatch, gated, strided):
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")
    torch.manual_seed(913)
    tensors = [torch.randn(2, 7, 34, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    residual, update = [t[..., ::2] if strided else t[..., :17].contiguous() for t in tensors]
    scale, shift, gate = [torch.randn(2, 1, 17, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    before = residual.detach().clone()
    kwargs = {"gate": gate if gated else None, "output_gate": gate}
    expected = reference(residual, update, scale, shift, **kwargs)
    actual = residual_layer_norm_scale_shift(residual, update, scale, shift, **kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual[2].is_contiguous()
    torch.testing.assert_close(residual, before, rtol=0, atol=0)
    leaves = (*tensors, scale, shift, gate)
    grads_a = torch.autograd.grad(sum(t.square().sum() for t in actual), leaves, retain_graph=True)
    grads_b = torch.autograd.grad(sum(t.square().sum() for t in expected), leaves)
    for a, b in zip(grads_a, grads_b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness fixture")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("features", [17, 2240, 3072, 5120])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("emit_gate", [False, True])
def test_cuda_fusion_matches_unfused_boundary(dtype, features, gated, emit_gate):
    from worldfoundry.core.kernels.triton_diffusion import layer_norm_scale_shift, residual_gate
    from worldfoundry.core.kernels.triton_residual_adaln import residual_layer_norm_scale_shift as fused

    torch.manual_seed(914)
    with torch.inference_mode():
        residual, update = [torch.randn(2, 67, features, device="cuda", dtype=dtype) for _ in range(2)]
        scale, shift, gate = [torch.randn(2, 1, features, device="cuda", dtype=dtype) * 0.1 for _ in range(3)]
        before = residual.clone()
        summed = residual_gate(residual, update, gate) if gated else residual + update
        expected = (layer_norm_scale_shift(summed, scale, shift, 1e-6), summed)
        if emit_gate:
            expected = (*expected, gate.expand_as(summed).contiguous())
        actual = fused(residual, update, scale, shift, 1e-6, gate if gated else None, gate if emit_gate else None)
        # Fusion must not change the pre-existing residual or modulation cuts.
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(residual, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness fixture")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gated", [False, True])
def test_cuda_mixed_dtype_strided_per_token_modulation(dtype, gated):
    from worldfoundry.core.kernels.triton_diffusion import layer_norm_scale_shift, residual_gate
    from worldfoundry.core.kernels.triton_residual_adaln import residual_layer_norm_scale_shift as fused

    torch.manual_seed(918)
    with torch.inference_mode():
        residual = torch.randn(2, 3, 7, 17, device="cuda", dtype=dtype)
        update = torch.randn_like(residual, dtype=torch.float32)
        scale = torch.randn(2, 3, 1, 34, device="cuda", dtype=dtype)[..., ::2]
        shift = torch.randn(1, 3, 1, 17, device="cuda", dtype=torch.float32)
        gate = torch.randn(2, 1, 7, 17, device="cuda", dtype=dtype)
        summed = residual_gate(residual, update, gate) if gated else residual + update
        expected = (layer_norm_scale_shift(summed, scale, shift, 1e-6), summed)
        actual = fused(residual, update, scale, shift, 1e-6, gate if gated else None, None)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_wan_block_opt_in_disable_partial_and_training(monkeypatch):
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import DiTBlock
    from worldfoundry.base_models.diffusion_model.optimizations.residual_adaln import enable_residual_adaln_fusion

    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    torch.manual_seed(915)
    block = DiTBlock(False, dim=32, num_heads=4, ffn_dim=64)
    # Keep real norms, modulation, residual and FFN; isolate attention unrelated
    # to the fusion so this regression runs without CUDA or attention packages.
    monkeypatch.setattr(block.self_attn, "forward", lambda value, *args, **kw: value.sin())
    monkeypatch.setattr(block.cross_attn, "forward", lambda value, *args, **kw: value.cos())
    x, modulation = torch.randn(2, 7, 32), torch.randn(2, 6, 32)
    with torch.inference_mode():
        original = block(x.clone(), t_mod=modulation)
        runtime = enable_residual_adaln_fusion(block)
        candidate = block(x.clone(), t_mod=modulation)
        torch.testing.assert_close(candidate, original, rtol=0, atol=0)
        assert runtime.calls == 1 and runtime.fallback_calls == 1
        partial, modifiers = block(x.clone(), t_mod=modulation, return_partial=True)
        resumed = block(partial, run_remaining=True, modifiers=modifiers)
        torch.testing.assert_close(resumed, original, rtol=0, atol=0)
        assert runtime.calls == 1
    block(x.clone().requires_grad_(), t_mod=modulation).sum().backward()
    assert runtime.calls == 1
    enable_residual_adaln_fusion(block, False)
    assert block._worldfoundry_residual_adaln_runtime is None

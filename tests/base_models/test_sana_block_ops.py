"""Numerical and training contracts for SANA's shared block glue."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.sana.block_ops import (
    SanaBlockFusionPolicy,
    gated_residual,
    modulated_norm,
)
from worldfoundry.core.kernels import clear_kernel_dispatch_cache
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.core.nn.blocks.layers import DropPath


@pytest.mark.parametrize("frames", [None, 3])
@pytest.mark.parametrize("affine", [False, True])
def test_norm_forward_and_backward_preserve_reference(monkeypatch, frames, affine):
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")
    torch.manual_seed(42)
    value = torch.randn(2, 9, 16, requires_grad=True)
    shape = (2, 1, 16) if frames is None else (2, frames, 1, 16)
    shift = torch.randn(shape, requires_grad=True)
    scale = torch.randn(shape, requires_grad=True)
    norm = nn.LayerNorm(16, eps=3e-5, elementwise_affine=affine)
    normalized = norm(value)
    if frames is not None:
        normalized = normalized.reshape(2, frames, -1, 16)
    expected = (normalized * (1 + scale) + shift).reshape(value.shape)
    actual = modulated_norm(value, norm, shift, scale, frames=frames)
    probe = torch.randn_like(value)
    leaves = (value, shift, scale, *norm.parameters())
    expected_grads = torch.autograd.grad(expected, leaves, probe)
    actual_grads = torch.autograd.grad(actual, leaves, probe)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_custom_norm_receives_flat_video_input():
    class FlatNorm(nn.LayerNorm):
        def forward(self, value):
            assert value.ndim == 3
            return super().forward(value) + 2

    value = torch.randn(2, 9, 16)
    norm = FlatNorm(16, elementwise_affine=False)
    shift, scale = torch.randn(2, 3, 1, 16), torch.randn(2, 3, 1, 16)
    expected = (norm(value).reshape(2, 3, 3, 16) * (1 + scale) + shift).reshape(value.shape)
    torch.testing.assert_close(modulated_norm(value, norm, shift, scale, frames=3), expected, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [None, 3])
def test_stochastic_depth_and_gate_gradients_keep_original_order(frames):
    residual = torch.randn(4, 9, 16, requires_grad=True)
    update = torch.randn_like(residual, requires_grad=True)
    gate = torch.randn((4, 1, 16) if frames is None else (4, frames, 1, 16), requires_grad=True)
    drop_path = DropPath(0.4).train()
    shaped_update = update if frames is None else update.reshape(4, frames, -1, 16)
    torch.manual_seed(5)
    expected = residual + drop_path((gate * shaped_update).reshape(residual.shape))
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(5)
    actual = gated_residual(residual, update, gate, drop_path, frames=frames)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    leaves = (residual, update, gate)
    probe = torch.randn_like(actual)
    for left, right in zip(torch.autograd.grad(actual, leaves, probe), torch.autograd.grad(expected, leaves, probe)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_custom_drop_path_runs_even_in_eval_and_receives_flat_tokens():
    class CustomDrop(nn.Module):
        def forward(self, value):
            assert value.ndim == 3
            return value * 0.5

    residual, update = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
    gate = torch.randn(2, 3, 1, 16)
    expected = residual + (gate * update.reshape(2, 3, 3, 16)).reshape(residual.shape) * 0.5
    actual = gated_residual(residual, update, gate, CustomDrop().eval(), frames=3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["image", "multiscale", "video", "camera"])
@pytest.mark.parametrize("frames", [None, 3])
def test_native_blocks_match_original_forward(family, frames):
    if frames is not None and family in {"image", "multiscale"}:
        pytest.skip("image graphs have no frame-aware conditioning")
    from worldfoundry.base_models.diffusion_model.models.networks.sana.sana import SanaBlock
    from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale import SanaMSBlock
    from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale_video import SanaVideoMSBlock
    from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale_video_camctrl import (
        SanaVideoMSCamCtrlBlock,
    )

    class Projection(nn.Linear):
        def forward(self, value, **kwargs):
            return super().forward(value)

    class CrossProjection(nn.Linear):
        def forward(self, value, context, mask=None, **kwargs):
            return super().forward(value + context.mean(dim=1, keepdim=True))

    constructors = {
        "image": SanaBlock,
        "multiscale": SanaMSBlock,
        "video": SanaVideoMSBlock,
        "camera": SanaVideoMSCamCtrlBlock,
    }
    torch.manual_seed(7)
    options = {"attn_type": "BidirectionalGDN", "linear_head_dim": 8} if family == "camera" else {}
    block = constructors[family](hidden_size=24, num_heads=3, drop_path=0.3, **options).train()
    block._worldfoundry_block_fusion = SanaBlockFusionPolicy(min_elements=0)
    block.attn = Projection(24, 24)
    block.mlp = Projection(24, 24)
    block.cross_attn = CrossProjection(24, 24)
    value = torch.randn(4, 12, 24)
    context = torch.randn(4, 5, 24)
    timestep = torch.randn(4, 6 * 24) if frames is None else torch.randn(4, 6 * 24, frames)
    if frames is None:
        modulation = (block.scale_shift_table[None] + timestep.reshape(4, 6, 24)).chunk(6, dim=1)
    else:
        modulation = (block.scale_shift_table[None, None] + timestep.reshape(4, frames, 6, 24)).chunk(6, dim=2)
    shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = modulation

    def normalized(value, norm, shift, scale):
        result = norm(value)
        if frames is not None:
            result = result.reshape(4, frames, -1, 24)
        return (result * (1 + scale) + shift).reshape(value.shape)

    def residual(value, update, gate):
        if frames is not None:
            update = update.reshape(4, frames, -1, 24)
        return value + block.drop_path((gate * update).reshape(value.shape))

    torch.manual_seed(8)
    expected = residual(value, block.attn(normalized(value, block.norm1, shift_sa, scale_sa)), gate_sa)
    expected = expected + block.cross_attn(expected, context)
    expected = residual(expected, block.mlp(normalized(expected, block.norm2, shift_ff, scale_ff)), gate_ff)
    torch.manual_seed(8)
    actual = block(value, context, timestep)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("frames", [None, 3])
@pytest.mark.parametrize("fuse_layer_norm", [False, True])
def test_cuda_fusion_matches_unfused_math_and_executes(monkeypatch, dtype, frames, fuse_layer_norm):
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE", "0")
    clear_kernel_dispatch_cache()
    torch.manual_seed(42)
    value = torch.randn(2, 12, 2240, device="cuda", dtype=dtype)
    update = torch.randn_like(value)
    shape = (2, 1, 2240) if frames is None else (2, frames, 1, 2240)
    shift, scale, gate = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    norm = nn.LayerNorm(2240, elementwise_affine=False, eps=1e-6).cuda()
    normalized = norm(value)
    shaped_update = update
    if frames is not None:
        normalized = normalized.reshape(2, frames, -1, 2240)
        shaped_update = update.reshape(normalized.shape)
    expected_norm = (normalized * (1 + scale) + shift).reshape(value.shape)
    expected_gate = value + (shaped_update * gate).reshape(value.shape)
    receipt = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        policy = SanaBlockFusionPolicy(backend="triton", min_elements=0, fuse_layer_norm=fuse_layer_norm)
        actual_norm = modulated_norm(value, norm, shift, scale, frames=frames, policy=policy)
        actual_gate = gated_residual(value, update, gate, nn.Identity(), frames=frames, policy=policy)
    tolerance = {torch.bfloat16: (0.02, 0.04), torch.float16: (0.002, 0.004), torch.float32: (1e-5, 2e-6)}
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(
        actual_norm, expected_norm, rtol=rtol if fuse_layer_norm else 0, atol=atol if fuse_layer_norm else 0
    )
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)
    assert torch.isfinite(actual_norm).all()
    executed = {item["op"]: item for item in receipt["dispatches"]}
    assert executed["layer_norm_scale_shift" if fuse_layer_norm else "scale_shift"]["backend"] == "triton"
    assert executed["residual_gate_add"]["backend"] == "triton"

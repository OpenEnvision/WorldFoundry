"""Canonical Wan blocks preserve the validated PyTorch modulation semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    DiTBlock,
    GateModule,
)


class _ConstantOutput(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden, *_args, **_kwargs):
        return torch.full_like(hidden, self.value)


def _constant_block() -> DiTBlock:
    block = DiTBlock(
        has_image_input=False,
        dim=8,
        num_heads=1,
        ffn_dim=16,
    ).eval()
    block.self_attn = _ConstantOutput(2.0)
    block.cross_attn = _ConstantOutput(3.0)
    block.ffn = _ConstantOutput(4.0)
    with torch.no_grad():
        block.modulation.zero_()
    return block


def _constant_block_inputs() -> tuple[torch.Tensor, ...]:
    hidden = torch.randn(1, 5, 8)
    context = torch.empty(1, 0, 8)
    t_mod = torch.zeros(1, 6, 8)
    t_mod[:, 2] = 0.5
    t_mod[:, 5] = 1.5
    freqs = torch.empty(0)
    return hidden, context, t_mod, freqs


def test_gate_module_matches_reference() -> None:
    residual = torch.randn(2, 7, 32)
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 32)

    expected = residual + gate * update
    actual = GateModule()(residual, gate, update)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gate_module_inplace_is_opt_in_and_reuses_storage() -> None:
    residual = torch.randn(2, 7, 32)
    original = residual.clone()
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 32)
    expected = original + gate * update

    with torch.no_grad():
        actual = GateModule()(residual, gate, update, inplace=True)

    assert actual.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)


def test_gate_module_requested_inplace_preserves_autograd_functional_path() -> None:
    residual = torch.randn(2, 7, 8, requires_grad=True)
    original = residual.detach().clone()
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 8)

    actual = GateModule()(residual, gate, update, inplace=True)
    actual.sum().backward()

    torch.testing.assert_close(residual.detach(), original, rtol=0, atol=0)
    assert residual.grad is not None


def test_dit_block_inplace_residual_matches_functional_and_mutates_only_opt_in() -> None:
    torch.manual_seed(4)
    block = _constant_block()
    hidden, context, t_mod, freqs = _constant_block_inputs()
    original = hidden.clone()

    with torch.no_grad():
        functional = block(hidden, context, t_mod, freqs)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)

    inplace_input = original.clone()
    with torch.no_grad():
        inplace = block(
            inplace_input,
            context,
            t_mod,
            freqs,
            _worldfoundry_inplace_residual=True,
        )

    assert inplace.data_ptr() == inplace_input.data_ptr()
    torch.testing.assert_close(inplace, functional, rtol=0, atol=0)
    torch.testing.assert_close(inplace_input, functional, rtol=0, atol=0)


def test_dit_phase_forward_supports_same_inplace_residual_contract() -> None:
    torch.manual_seed(5)
    block = _constant_block()
    hidden, context, t_mod, freqs = _constant_block_inputs()
    phases = {
        "self_attn_out": torch.full_like(hidden, 2.0),
        "cross_attn_out": torch.full_like(hidden, 3.0),
        "ffn_out": torch.full_like(hidden, 4.0),
    }

    with torch.no_grad():
        functional, _ = block.forward_with_phase_cache(
            hidden.clone(),
            context,
            t_mod,
            freqs,
            cached_phases=phases,
        )
        inplace_input = hidden.clone()
        inplace, returned = block.forward_with_phase_cache(
            inplace_input,
            context,
            t_mod,
            freqs,
            cached_phases=phases,
            _worldfoundry_inplace_residual=True,
        )

    assert returned == phases
    assert inplace.data_ptr() == inplace_input.data_ptr()
    torch.testing.assert_close(inplace, functional, rtol=0, atol=0)


def test_forward_remaining_matches_unfused_reference() -> None:
    torch.manual_seed(0)
    block = DiTBlock(
        has_image_input=False,
        dim=32,
        num_heads=4,
        ffn_dim=64,
    ).eval()
    hidden = torch.randn(2, 7, 32)
    shift = torch.randn(2, 1, 32)
    scale = torch.randn(2, 1, 32)
    gate = torch.randn(2, 1, 32)

    normalized = F.layer_norm(
        hidden,
        (hidden.shape[-1],),
        weight=None,
        bias=None,
        eps=block.norm2.eps,
    )
    expected = hidden + gate * block.ffn(normalized * (1 + scale) + shift)
    actual = block.forward_remaining(hidden, shift, scale, gate)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

"""Portable mixture-of-experts reference operators.

These operators intentionally favor correctness and broad device support over
specialized grouped-GEMM performance.  Model integrations can keep their
checkpoint-compatible packed expert weights and use this module when an
optional CUDA/Triton kernel is unavailable on the current GPU.

Not this module:
    Grouped Triton matmuls live in :mod:`.triton_moe` and are imported only
    after the registry predicate passes. Capability probing lives in
    :mod:`.capabilities`.

Public surface:

- :func:`routed_swiglu_moe` — registry dispatch; Triton when eligible.
- :func:`routed_swiglu_moe_pytorch` — exact per-expert ``index_select`` path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from worldfoundry.core.kernels.capabilities import triton_tensor_eligible
from worldfoundry.core.kernels.registry import KERNEL_REGISTRY


# ──────────────────────────────────────────────────────────────────────────
# Portable reference — evaluate only routed tokens so unused experts stay cold
# ──────────────────────────────────────────────────────────────────────────


def routed_swiglu_moe_pytorch(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a token-routed SwiGLU MoE using portable PyTorch operators.

    Args:
        hidden_states: Flattened token activations with shape ``[T, D]``.
        routing_weights: Per-route weights with shape ``[T, K]``.
        selected_experts: Expert indices with shape ``[T, K]``.
        gate_weight: Packed gate weights with shape ``[E, I, D]``.
        up_weight: Packed up-projection weights with shape ``[E, I, D]``.
        down_weight: Packed down-projection weights with shape ``[E, D, I]``.

    Only tokens routed to an expert are evaluated for that expert.  This keeps
    the fallback substantially smaller than materializing every expert output,
    while retaining autograd support and working on CPU, CUDA, and other
    PyTorch devices.
    """

    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must have shape [T, D], got {tuple(hidden_states.shape)}")
    if routing_weights.ndim != 2 or selected_experts.ndim != 2:
        raise ValueError("routing_weights and selected_experts must have shape [T, K]")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            "routing_weights and selected_experts must have identical shapes, "
            f"got {tuple(routing_weights.shape)} and {tuple(selected_experts.shape)}"
        )
    if routing_weights.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tensors and hidden_states must contain the same number of tokens")
    if gate_weight.ndim != 3 or up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("packed expert weights must be rank-3 tensors")

    num_experts, intermediate_size, hidden_size = gate_weight.shape
    expected_up = (num_experts, intermediate_size, hidden_size)
    expected_down = (num_experts, hidden_size, intermediate_size)
    if hidden_states.shape[1] != hidden_size:
        raise ValueError(
            f"hidden size mismatch: activations have {hidden_states.shape[1]}, weights expect {hidden_size}"
        )
    if tuple(up_weight.shape) != expected_up or tuple(down_weight.shape) != expected_down:
        raise ValueError(
            "unexpected packed expert shapes: "
            f"gate={tuple(gate_weight.shape)}, up={tuple(up_weight.shape)}, "
            f"down={tuple(down_weight.shape)}"
        )
    devices = {
        hidden_states.device,
        routing_weights.device,
        selected_experts.device,
        gate_weight.device,
        up_weight.device,
        down_weight.device,
    }
    if len(devices) != 1:
        raise ValueError(f"all MoE tensors must share one device, got {sorted(map(str, devices))}")
    if gate_weight.dtype != up_weight.dtype or gate_weight.dtype != down_weight.dtype:
        raise ValueError("all packed expert weights must share one dtype")

    output = torch.zeros_like(hidden_states)
    for expert_idx in range(num_experts):
        token_indices, route_slots = torch.where(selected_experts == expert_idx)
        if token_indices.numel() == 0:
            continue

        expert_input = hidden_states.index_select(0, token_indices).to(gate_weight.dtype)
        gate = F.linear(expert_input, gate_weight[expert_idx])
        up = F.linear(expert_input, up_weight[expert_idx])
        expert_output = F.linear(F.silu(gate) * up, down_weight[expert_idx])
        route_weight = routing_weights[token_indices, route_slots].unsqueeze(-1)
        weighted_output = (expert_output * route_weight.to(expert_output.dtype)).to(output.dtype)
        output = output.index_add(0, token_indices, weighted_output)

    return output


# ──────────────────────────────────────────────────────────────────────────
# Registry adapters — lazy Triton import; signature keys exclude workspace
# ──────────────────────────────────────────────────────────────────────────


def _triton_candidate_eligible(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    workspace: Any = None,
    allow_triton: bool = True,
) -> bool:
    """Return whether packed weights and activations may enter the Triton path.

    Routing tensors are ignored here: packing validates them inside the kernel
    wrapper. Workspace is optional and checked only after launch is chosen.
    """

    del routing_weights, selected_experts, workspace
    return (
        allow_triton
        and triton_tensor_eligible(hidden_states)
        and gate_weight.device == hidden_states.device
        and up_weight.device == hidden_states.device
        and down_weight.device == hidden_states.device
        and gate_weight.dtype == hidden_states.dtype
        and up_weight.dtype == hidden_states.dtype
        and down_weight.dtype == hidden_states.dtype
        and hidden_states.is_contiguous()
        and gate_weight.is_contiguous()
        and up_weight.is_contiguous()
        and down_weight.is_contiguous()
    )


def _routed_swiglu_moe_triton_candidate(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    workspace: dict[str, torch.Tensor] | Callable[[], dict[str, torch.Tensor]] | None = None,
    allow_triton: bool = True,
) -> torch.Tensor:
    """Import and launch the grouped Triton MoE after the predicate has passed."""

    del allow_triton
    from worldfoundry.core.kernels.triton_moe import routed_swiglu_moe_triton

    resolved_workspace = workspace() if callable(workspace) else workspace
    return routed_swiglu_moe_triton(
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
        workspace=resolved_workspace,
    )


def _routed_swiglu_moe_fallback(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    workspace: Any = None,
    allow_triton: bool = True,
) -> torch.Tensor:
    """Drop dispatch-only kwargs and run the portable PyTorch reference."""

    del workspace, allow_triton
    return routed_swiglu_moe_pytorch(
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
    )


KERNEL_REGISTRY.register(
    "routed_swiglu_moe",
    backend="triton",
    name="worldfoundry_triton_packed_swiglu_moe",
    implementation=_routed_swiglu_moe_triton_candidate,
    predicate=_triton_candidate_eligible,
    priority=100,
)


# ──────────────────────────────────────────────────────────────────────────
# Public dispatch — disable Triton under autograd; cache key is shape/dtype
# ──────────────────────────────────────────────────────────────────────────


def routed_swiglu_moe(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    workspace: dict[str, torch.Tensor] | Callable[[], dict[str, torch.Tensor]] | None = None,
    allow_triton: bool | None = None,
) -> torch.Tensor:
    """Dispatch packed SwiGLU MoE to Triton or the portable PyTorch reference."""

    if allow_triton is None:
        allow_triton = not torch.is_grad_enabled() and not any(
            tensor.requires_grad for tensor in (hidden_states, gate_weight, up_weight, down_weight)
        )
    signature = (
        str(hidden_states.device),
        str(hidden_states.dtype),
        tuple(hidden_states.shape),
        tuple(routing_weights.shape),
        tuple(gate_weight.shape),
        tuple(down_weight.shape),
        bool(allow_triton),
    )
    return KERNEL_REGISTRY.dispatch(
        "routed_swiglu_moe",
        _routed_swiglu_moe_fallback,
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
        workspace=workspace,
        allow_triton=bool(allow_triton),
        signature=signature,
    )


__all__ = ["routed_swiglu_moe", "routed_swiglu_moe_pytorch"]

"""Runtime proof for Wan's optional fused Q/K RMSNorm + 3D RoPE path.

Installing a fused-table flag proves only that the model was configured.  This
module binds receipts to the concrete ``SelfAttention`` instances and records
either an eager accelerated-kernel dispatch or a Dynamo graph trace.  Formal
benchmark gates can therefore reject a PyTorch registry fallback even when the
fused public operator returned numerically correct tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class FusedRoPERuntimeState:
    """Shared installation and execution receipts for one Wan model."""

    installed_blocks: int = 0
    eager_calls: int = 0
    compiled_graph_traces: int = 0
    provider_calls: int = 0
    torch_fallback_calls: int = 0
    provider_failures: int = 0
    quarantined_skips: int = 0
    malformed_receipts: int = 0
    provider_paths: set[str] = field(default_factory=set)
    last_dispatch: dict[str, Any] | None = None

    def reset_request_window(self) -> None:
        """Clear eager request counters without invalidating compiled graphs."""

        self.eager_calls = 0
        self.provider_calls = 0
        self.torch_fallback_calls = 0
        self.provider_failures = 0
        self.quarantined_skips = 0
        self.malformed_receipts = 0
        self.provider_paths.clear()
        self.last_dispatch = None

    def record_eager_dispatch(self, receipt: Mapping[str, Any]) -> None:
        self.eager_calls += 1
        raw_dispatches = receipt.get("dispatches")
        if not isinstance(raw_dispatches, list):
            self.malformed_receipts += 1
            return
        dispatches = [
            item
            for item in raw_dispatches
            if isinstance(item, Mapping)
            and item.get("op") == "hidden_qk_rmsnorm_rope_3d"
        ]
        if len(dispatches) != 1:
            self.malformed_receipts += 1
            return
        dispatch = dict(dispatches[0])
        self.last_dispatch = dispatch
        failures = dispatch.get("failures")
        quarantined = dispatch.get("quarantined")
        if not isinstance(failures, list) or not isinstance(quarantined, list):
            self.malformed_receipts += 1
            return
        self.provider_failures += len(failures)
        self.quarantined_skips += len(quarantined)
        provider = dispatch.get("implementation")
        accelerated = dispatch.get("accelerated") is True
        fallback = dispatch.get("fallback") is True
        if accelerated and not fallback and isinstance(provider, str) and provider:
            self.provider_calls += 1
            self.provider_paths.add(provider)
        else:
            self.torch_fallback_calls += 1


@torch.compiler.assume_constant_result
def record_compiled_fused_rope_graph_trace(
    state: FusedRoPERuntimeState,
) -> int:
    """Record that Dynamo embedded the fused primitive in this model graph."""

    state.compiled_graph_traces += 1
    return state.compiled_graph_traces


def install_fused_rope_runtime(model: nn.Module) -> FusedRoPERuntimeState:
    """Attach one shared receipt state to every Wan self-attention block."""

    existing = getattr(model, "_worldfoundry_fused_rope_runtime", None)
    if isinstance(existing, FusedRoPERuntimeState):
        return existing
    state = FusedRoPERuntimeState()
    for module in model.modules():
        if type(module).__name__ != "SelfAttention":
            continue
        module._worldfoundry_fused_rope_runtime = state
        state.installed_blocks += 1
    model._worldfoundry_fused_rope_runtime = state
    return state


def fused_rope_runtime_report(
    state: FusedRoPERuntimeState,
) -> dict[str, Any]:
    """Return strict installation, graph, and eager provider receipts."""

    if (
        state.provider_calls > 0
        and state.torch_fallback_calls == 0
        and state.provider_failures == 0
        and state.quarantined_skips == 0
        and state.malformed_receipts == 0
    ):
        effective = "accelerated-provider-executed"
    elif state.compiled_graph_traces > 0:
        effective = "compiled-graph-traced"
    elif state.eager_calls > 0:
        effective = "torch-or-failed-provider"
    else:
        effective = "installed (runtime-pending)"
    return {
        "installed_blocks": state.installed_blocks,
        "effective": effective,
        "eager_calls": state.eager_calls,
        "compiled_graph_traces": state.compiled_graph_traces,
        "provider_calls": state.provider_calls,
        "torch_fallback_calls": state.torch_fallback_calls,
        "provider_failures": state.provider_failures,
        "quarantined_skips": state.quarantined_skips,
        "malformed_receipts": state.malformed_receipts,
        "provider_paths": sorted(state.provider_paths),
        "last_dispatch": (
            None if state.last_dispatch is None else dict(state.last_dispatch)
        ),
    }


__all__ = [
    "FusedRoPERuntimeState",
    "fused_rope_runtime_report",
    "install_fused_rope_runtime",
    "record_compiled_fused_rope_graph_trace",
]

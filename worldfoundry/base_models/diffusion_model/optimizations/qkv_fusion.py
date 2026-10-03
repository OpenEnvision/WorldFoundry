"""Post-load merged-QKV build transform for separate-projection attention.

Wan's ``SelfAttention`` projects Q/K/V with three independent
``nn.Linear(dim, dim)`` layers (``self.q``/``self.k``/``self.v``) and applies
QK-norm + RoPE *after* projection. Because the norms and RoPE act on the split
outputs, the three projections can be fused into one ``nn.Linear(dim, 3*dim)``
followed by a split — this is bitwise-equivalent up to GEMM reduction order and
removes two kernel launches (and two weight reads) per attention block.

This is a load-time transform (run after the dense checkpoint is loaded, before
offload hooks), matching the FastVideo/LightX2V "replace at load, not per
request" pattern. It duck-types the target so it does not import the Wan class,
and it is a no-op on already-fused modules (a single ``qkv`` Linear, as in
Sana). Cross-attention is intentionally skipped: its Q and K/V consume different
inputs and cannot share one projection.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakValueDictionary

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class QKVFusionState:
    """Runtime receipts shared by every fused projection in one model.

    ``fused_blocks`` is only an installation receipt.  Eager projection calls
    and successful Dynamo graph traces are kept separately so certification
    can require actual execution (and, for a compiled trace, pair it with the
    owning compiled-wrapper call counter).  Counting at the projection rather
    than at the replacement attention ``forward`` also covers the supported
    QKV-fusion + sequence-parallel path, whose processor invokes ``qkv``
    directly.
    """

    fused_blocks: int = 0
    eager_projection_calls: int = 0
    compiled_graph_traces: int = 0
    request_eager_projection_calls: int = 0
    request_compiled_graph_traces: int = 0
    eager_packed_projection_calls: int = 0
    eager_split_projection_calls: int = 0
    compiled_packed_graph_traces: int = 0
    compiled_split_graph_traces: int = 0
    request_eager_packed_projection_calls: int = 0
    request_eager_split_projection_calls: int = 0
    request_compiled_packed_graph_traces: int = 0
    request_compiled_split_graph_traces: int = 0
    strategy: str = "packed"
    split_threshold: int = 8192

    def reset_request_window(self) -> None:
        """Start a new generation-request telemetry window.

        The lifetime counters intentionally survive.  In particular, Dynamo
        traces a fused projection only while building a graph; later requests
        can execute that cached graph without tracing it again.  Certification
        therefore pairs the lifetime graph receipt with the compile wrapper's
        request-local execution counter, while eager certification uses the
        request-local projection counter directly.
        """

        self.request_eager_projection_calls = 0
        self.request_compiled_graph_traces = 0
        self.request_eager_packed_projection_calls = 0
        self.request_eager_split_projection_calls = 0
        self.request_compiled_packed_graph_traces = 0
        self.request_compiled_split_graph_traces = 0


_QKV_FUSION_STATES: WeakValueDictionary[str, QKVFusionState] = WeakValueDictionary()
_QKV_FUSION_STATES_LOCK = Lock()


def _register_qkv_fusion_state(state: QKVFusionState) -> str:
    token = str(id(state))
    with _QKV_FUSION_STATES_LOCK:
        _QKV_FUSION_STATES[token] = state
    return token


@torch.compiler.assume_constant_result
def _record_compiled_qkv_graph_trace(receipt_token: str, path_code: int) -> int:
    """Record that a fused projection was embedded in a compiled graph.

    Dynamo evaluates this helper while tracing and replaces its return value
    with a constant, so no counter operation is emitted into the generated
    graph.  A trace receipt alone is not runtime proof; callers must combine it
    with a positive compiled-wrapper execution count.
    """

    with _QKV_FUSION_STATES_LOCK:
        state = _QKV_FUSION_STATES.get(receipt_token)
        if state is None:
            raise RuntimeError("QKV fusion compile receipt state expired during tracing")
        state.compiled_graph_traces += 1
        state.request_compiled_graph_traces += 1
        if int(path_code) == 0:
            state.compiled_packed_graph_traces += 1
            state.request_compiled_packed_graph_traces += 1
        elif int(path_code) == 1:
            state.compiled_split_graph_traces += 1
            state.request_compiled_split_graph_traces += 1
        else:
            raise ValueError(f"unknown QKV projection path code: {path_code}")
        return state.compiled_graph_traces


class _FusedQKVLinear(nn.Linear):
    """``nn.Linear``-compatible merged projection with execution telemetry."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: torch.device,
        dtype: torch.dtype,
        fusion_state: QKVFusionState,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._worldfoundry_qkv_fusion_state = fusion_state
        # Text identities stay specialized across models. Integer identities
        # can become symbolic during recompilation and cannot be passed to an
        # assume_constant_result helper.
        self._worldfoundry_qkv_receipt_token = _register_qkv_fusion_state(
            fusion_state
        )

    def _record_projection(self, path: str) -> None:
        state = self._worldfoundry_qkv_fusion_state
        if torch.compiler.is_compiling():
            _record_compiled_qkv_graph_trace(
                self._worldfoundry_qkv_receipt_token,
                0 if path == "packed" else 1,
            )
        else:
            state.eager_projection_calls += 1
            state.request_eager_projection_calls += 1
            if path == "packed":
                state.eager_packed_projection_calls += 1
                state.request_eager_packed_projection_calls += 1
            else:
                state.eager_split_projection_calls += 1
                state.request_eager_split_projection_calls += 1

    def _projection_strategy(self, input: torch.Tensor) -> str:
        state = self._worldfoundry_qkv_fusion_state
        if state.strategy != "auto":
            return state.strategy
        flattened_rows = input.numel() // self.in_features
        return "split" if flattened_rows >= state.split_threshold else "packed"

    def project_qkv(
        self,
        input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q/K/V through the configured packed/split workload path.

        ``split`` calls three GEMMs over zero-copy row slices of the merged
        parameter.  It therefore keeps the load-time memory saving while
        avoiding the slower very-wide GEMM observed for production Wan token
        counts.  ``auto`` uses a shape-only cutoff, which is deterministic and
        safe to specialize inside a compiled graph.
        """

        path = self._projection_strategy(input)
        self._record_projection(path)
        if path == "packed":
            return F.linear(input, self.weight, self.bias).chunk(3, dim=-1)
        width = self.out_features // 3
        outputs = []
        for index in range(3):
            start = index * width
            end = start + width
            bias = None if self.bias is None else self.bias[start:end]
            outputs.append(F.linear(input, self.weight[start:end], bias))
        return outputs[0], outputs[1], outputs[2]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Preserve the nn.Linear-compatible public contract. Wan consumers use
        # ``project_qkv`` so split mode does not pay an immediate concat/chunk.
        self._record_projection("packed")
        return F.linear(input, self.weight, self.bias)


def project_fused_qkv(
    projection: nn.Module,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use workload dispatch when available, otherwise a packed projection.

    Generic quantization can replace the dense fused Linear after installation.
    Those wrappers intentionally keep their existing packed kernel; this helper
    lets every Wan processor share the same compatibility behavior.
    """

    project = getattr(projection, "project_qkv", None)
    if callable(project):
        return project(input)
    packed = projection(input)
    if not isinstance(packed, torch.Tensor) or packed.shape[:-1] != input.shape[:-1]:
        raise RuntimeError("fused Wan qkv projection must preserve leading dimensions")
    if packed.shape[-1] != input.shape[-1] * 3:
        raise RuntimeError(
            "fused Wan qkv projection must return exactly 3 * hidden width"
        )
    return packed.chunk(3, dim=-1)


def _is_fusible_self_attention(module: nn.Module) -> bool:
    """Duck-type a separate-projection self-attention block.

    Requires ``q``/``k``/``v`` to be square ``nn.Linear(dim, dim)`` with matching
    in/out features and consistent bias, plus the QK-norm attributes the fused
    forward will reuse. A merged module (no separate ``q``) returns False.
    """

    q = getattr(module, "q", None)
    k = getattr(module, "k", None)
    v = getattr(module, "v", None)
    if not (isinstance(q, nn.Linear) and isinstance(k, nn.Linear) and isinstance(v, nn.Linear)):
        return False
    if getattr(module, "_qkv_fused", False):
        return False
    # A non-default processor may opt in to projection-only fusion by exposing
    # ``supports_fused_qkv = True``.  Those processors keep ownership of the
    # attention algorithm and consume ``attention.qkv`` themselves; _fuse_one
    # therefore leaves the module's processor-dispatching forward intact.  A
    # processor without that explicit contract is skipped fail-closed so a
    # load-time fusion can never silently bypass a sparse/custom algorithm.
    get_processor = getattr(module, "get_processor", None)
    if callable(get_processor):
        processor = get_processor()
        is_default = type(processor).__name__ == "SelfAttentionProcessor"
        supports_fused_qkv = getattr(processor, "supports_fused_qkv", False) is True
        if not is_default and not supports_fused_qkv:
            return False
    # Cross-attention exposes the same names but its forward is driven by a
    # processor over a separate context; only fuse the freqs-based self-attn.
    if not hasattr(module, "norm_q") or not hasattr(module, "norm_k"):
        return False
    native_projection_dispatch = getattr(module, "supports_fused_qkv", False) is True
    if not native_projection_dispatch and (
        type(module).__name__ != "SelfAttention" or not hasattr(module, "attn")
    ):
        return False
    dim = q.in_features
    for layer in (q, k, v):
        if layer.in_features != dim or layer.out_features != dim:
            return False
    bias_flags = {layer.bias is not None for layer in (q, k, v)}
    if len(bias_flags) != 1:
        return False
    return True


def _fused_self_attention_forward(
    self: nn.Module,
    x: torch.Tensor,
    freqs: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Drop-in replacement for ``SelfAttention.forward`` using one QKV GEMM."""

    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
        apply_wan_qk_norm_rope,
    )

    q, k, v = project_fused_qkv(self.qkv, x)
    q, k = apply_wan_qk_norm_rope(
        self,
        q,
        k,
        freqs,
        fused_table=kwargs.pop("_worldfoundry_rope_table", None),
        fused_grid=kwargs.pop("_worldfoundry_rope_grid", None),
        precision=kwargs.pop("_worldfoundry_rope_precision", "fp64"),
    )
    x = self.attn(q, k, v)
    return self.o(x)


def _fuse_one(module: nn.Module, state: QKVFusionState) -> None:
    get_processor = getattr(module, "get_processor", None)
    processor = get_processor() if callable(get_processor) else None
    preserve_projection_dispatch = getattr(module, "supports_fused_qkv", False) is True or (
        processor is not None
        and type(processor).__name__ != "SelfAttentionProcessor"
        and getattr(processor, "supports_fused_qkv", False) is True
    )
    q, k, v = module.q, module.k, module.v
    dim = q.in_features
    has_bias = q.bias is not None
    # Build the merged projection where the loaded weights already live.  The
    # default ``nn.Linear`` constructor creates an FP32 CPU allocation; for a
    # GPU-resident Wan block that used to pull Q/K/V back to host during
    # ``copy_`` and then upload the merged tensor again.  Across 30 blocks and
    # several ranks that needless round trip dominates model-load time and can
    # exhaust host memory.  Direct device/dtype construction is both cheaper
    # and numerically identical because the destination has the final dtype
    # before the copy in either implementation.
    fused = _FusedQKVLinear(
        dim,
        3 * dim,
        bias=has_bias,
        device=q.weight.device,
        dtype=q.weight.dtype,
        fusion_state=state,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([q.weight, k.weight, v.weight], dim=0))
        if has_bias:
            fused.bias.copy_(torch.cat([q.bias, k.bias, v.bias], dim=0))
    module.add_module("qkv", fused)
    # Drop the originals so their weights are freed and no stale path remains.
    del module.q, module.k, module.v
    module._qkv_fused = True
    if not preserve_projection_dispatch:
        module.forward = types.MethodType(_fused_self_attention_forward, module)


def fuse_qkv_projections(
    model: nn.Module,
    *,
    strategy: str = "packed",
    split_threshold: int = 8192,
) -> int:
    """Fuse separate Q/K/V self-attention projections in ``model`` in place.

    Returns the number of attention blocks fused. Safe to call on models that
    are already fused (Sana) or have no matching blocks — it simply returns 0.
    """

    normalized_strategy = str(strategy).strip().casefold()
    if normalized_strategy not in {"auto", "packed", "split"}:
        raise ValueError("QKV strategy must be 'auto', 'packed', or 'split'")
    resolved_threshold = int(split_threshold)
    if resolved_threshold <= 0:
        raise ValueError("QKV split_threshold must be a positive integer")

    state = getattr(model, "_worldfoundry_qkv_fusion", None)
    if not isinstance(state, QKVFusionState):
        state = QKVFusionState()
        model._worldfoundry_qkv_fusion = state
    state.strategy = normalized_strategy
    state.split_threshold = resolved_threshold
    _register_qkv_fusion_state(state)

    fused = 0
    for module in model.modules():
        if _is_fusible_self_attention(module):
            _fuse_one(module, state)
            fused += 1
    state.fused_blocks += fused
    return fused


def restore_fused_qkv_processor_dispatch(module: nn.Module) -> bool:
    """Route a previously fused attention module through its new processor.

    Dense QKV fusion installs an instance-local forward for the default
    processor. A later model-specific processor must explicitly support the
    merged projection and regain ownership of ``forward``; otherwise changing
    ``module.processor`` would be a silent no-op. Returns whether a fused
    module was restored and fails closed for an incompatible processor.
    """

    if not bool(getattr(module, "_qkv_fused", False)):
        return False
    get_processor = getattr(module, "get_processor", None)
    processor = get_processor() if callable(get_processor) else None
    if getattr(processor, "supports_fused_qkv", False) is not True:
        raise RuntimeError(
            "cannot restore fused QKV processor dispatch: the installed "
            "processor does not declare supports_fused_qkv=True"
        )
    class_forward = getattr(type(module), "forward", None)
    if not callable(class_forward):
        raise TypeError(
            "cannot restore fused QKV processor dispatch without a class forward"
        )
    module.forward = types.MethodType(class_forward, module)
    return True


def qkv_fusion_report(model: nn.Module) -> dict[str, int | str] | None:
    """Return installation and runtime receipts for one fused model."""

    state = getattr(model, "_worldfoundry_qkv_fusion", None)
    if not isinstance(state, QKVFusionState):
        return None
    eager_packed = state.request_eager_packed_projection_calls
    eager_split = state.request_eager_split_projection_calls
    compiled_packed = state.request_compiled_packed_graph_traces
    compiled_split = state.request_compiled_split_graph_traces
    if eager_packed and eager_split:
        execution = "eager-mixed-projection-executed"
    elif eager_split:
        execution = "eager-split-projection-executed"
    elif eager_packed:
        execution = "eager-packed-projection-executed"
    elif state.request_compiled_graph_traces > 0:
        execution = "compiled-graph-traced (execution-pending-wrapper-receipt)"
    elif state.compiled_graph_traces > 0:
        execution = "compiled-graph-cached (execution-pending-wrapper-receipt)"
    else:
        execution = "installed (runtime-pending)"
    return {
        "fused_blocks": state.fused_blocks,
        # Eager calls describe the active request.  A compiled trace is a
        # structural lifetime receipt (matching fused-RoPE/attention receipts):
        # cached graphs do not retrace on every request, so certification pairs
        # it with the compile wrapper's request-local call count.
        "eager_projection_calls": state.request_eager_projection_calls,
        "compiled_graph_traces": state.compiled_graph_traces,
        "request_compiled_graph_traces": state.request_compiled_graph_traces,
        "lifetime_eager_projection_calls": state.eager_projection_calls,
        "lifetime_compiled_graph_traces": state.compiled_graph_traces,
        "strategy": state.strategy,
        "split_threshold": state.split_threshold,
        "eager_packed_projection_calls": eager_packed,
        "eager_split_projection_calls": eager_split,
        "compiled_packed_graph_traces": compiled_packed,
        "compiled_split_graph_traces": compiled_split,
        "lifetime_eager_packed_projection_calls": state.eager_packed_projection_calls,
        "lifetime_eager_split_projection_calls": state.eager_split_projection_calls,
        "lifetime_compiled_packed_graph_traces": state.compiled_packed_graph_traces,
        "lifetime_compiled_split_graph_traces": state.compiled_split_graph_traces,
        "execution": execution,
    }


__all__ = [
    "QKVFusionState",
    "fuse_qkv_projections",
    "project_fused_qkv",
    "qkv_fusion_report",
    "restore_fused_qkv_processor_dispatch",
]

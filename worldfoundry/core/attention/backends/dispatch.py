"""Layout-agnostic Attention dispatch: normalize any einops QKV pattern, then pick a kernel.

Why this split exists — instead of "use Flash whenever it is installed":

- **Do not resolve a device at import.** ``initialize_attention_priority`` only
  reads the env preference (usually ``auto``). Calling
  ``cuda.get_device_capability`` at import would create a CUDA context on GPU 0
  and break fork workers. Capability bits are probed lazily through
  ``__getattr__``.
- **``auto`` does not enable external packages.** The in-tree contract is
  PyTorch SDPA (which already dispatches to bundled Flash/cuDNN).
  FlashAttention, SageAttention, and xFormers are explicit opt-ins.
- **``attn_mask`` or ``compatibility_mode`` go to ``torch_sdpa``.** Fused
  kernels do not share one mask contract; forcing FlashAttention can silently
  compute the wrong scores.
- **Short sequences** below ``WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE`` (or
  the SM defaults 64/128/256) pay more in fused-kernel launch than they save,
  so they stay on SDPA.
- **Non-fp16/bf16** is almost never supported by fused kernels; the path is
  fixed to ``torch``.
- **Failure quarantine.** Optional-package load failures are marked
  unavailable. ``kernel not supported`` RuntimeErrors are isolated by
  ``(backend, device, dtype, shape)``. OOM must propagate — never retry with
  an explicit O(S²) score tensor that can OOM again.
- The candidate list always ends with ``torch``. Use
  ``attention_dispatch_report`` / ``clear_attention_dispatch_cache`` to debug
  or reset after a hot-reload in a long-lived process.
"""

import math
import os
import threading
import warnings
from collections import deque
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
from einops import rearrange

from worldfoundry.core.attention.backends.native import (
    native_sdpa_priority,
)
from worldfoundry.core.attention.backends.native import (
    scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention,
)
from worldfoundry.core.attention.backends.probe import (
    attention_backend_capability,
    attention_backend_from_env,
    gpu_supports_flash_attention,
    normalize_attention_backend,
    probe_attention_backends,
    require_generic_attention_backend,
    resolve_attention_backend,
)

# ──────────────────────────────────────────────────────────────────────────
# Import-time policy — record the env preference; do not bind a CUDA device
# ──────────────────────────────────────────────────────────────────────────


def initialize_attention_priority():
    """Record the env preference (usually ``auto``) without binding a device backend at import."""
    # Keep the user's preference (usually ``auto``) unresolved until a tensor
    # device is known. Resolving at import time can permanently select CPU SDPA
    # before a worker calls torch.cuda.set_device().
    return attention_backend_from_env()


ATTENTION_IMPLEMENTATION = initialize_attention_priority()

# CC-07: capability probing calls ``torch.cuda.get_device_capability`` which
# lazily creates a CUDA context. Doing that at import time breaks fork-based
# workers and pins a context on GPU 0 before ``torch.cuda.set_device``.
# ``_CAPABILITIES`` and the ``*_AVAILABLE`` flags are therefore resolved on
# first attribute access instead of at import, and are never frozen into
# module globals (``probe_attention_backends`` already caches per runtime
# signature).
_LAZY_CAPABILITY_EXPORTS = {
    "FLASH_ATTN_4_AVAILABLE": "flash_attention_4",
    "FLASH_ATTN_3_AVAILABLE": "flash_attention_3",
    "FLASH_ATTN_2_AVAILABLE": "flash_attention_2",
    "SAGE_ATTN_AVAILABLE": "sage_attention",
    "XFORMERS_AVAILABLE": "xformers",
}


def __getattr__(name: str):
    """Lazily export ``_CAPABILITIES`` and ``*_AVAILABLE`` so import does not create a CUDA context."""
    if name == "_CAPABILITIES":
        return probe_attention_backends()
    backend_name = _LAZY_CAPABILITY_EXPORTS.get(name)
    if backend_name is not None:
        return probe_attention_backends()[backend_name].available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _gpu_supports_flash_attention():
    """Return whether the current device is a validated in-tree FA2 target."""
    return gpu_supports_flash_attention()


# ──────────────────────────────────────────────────────────────────────────
# Layout adapters — normalize any einops pattern, then restore the caller
# ──────────────────────────────────────────────────────────────────────────


def rearrange_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    required_in_pattern="b n s d",
    dims=None,
):
    """Rearrange Q/K/V from the caller layout into the backend ``required_in_pattern``."""
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    """Restore backend output from the internal layout to the caller ``out_pattern``."""
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    attn_mask=None,
    scale=None,
):
    """Exact path: in-tree ``scaled_dot_product_attention``. Failures do not retry with dense O(S²)."""
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    # The core helper already supplies a math implementation when PyTorch does
    # not expose SDPA. Runtime failures from an available implementation must
    # propagate: retrying with an explicit O(S^2) score tensor can amplify OOM.
    out = _worldfoundry_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask,
        scale=scale,
        enable_gqa=q.shape[1] != k.shape[1],
        backends=native_sdpa_priority(q.device, has_mask=attn_mask is not None),
    )
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Explicit providers — one callable per opt-in kernel; never selected by auto
# ──────────────────────────────────────────────────────────────────────────


def flash_attention_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    """FlashAttention 3 entry (Hopper ``flash_attn_interface``); layout ``b s n d``."""
    import flash_attn_interface

    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def _flash_attention_4_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Call either the public or pinned private CuTeDSL FA4 interface."""

    try:
        from flash_attn.cute import flash_attn_func
    except ImportError:
        from flash_attn.cute.interface import _flash_attn_fwd

        result = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=None,
            window_size_right=None,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
        )
    else:
        result = flash_attn_func(q, k, v, softmax_scale=softmax_scale)
    return result[0] if isinstance(result, tuple) else result


@torch.library.custom_op(
    "worldfoundry::flash_attention_4_forward",
    mutates_args=(),
    device_types="cuda",
)
def _flash_attention_4_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Opaque compile boundary around FA4's CuTeDSL/JIT implementation."""

    return _flash_attention_4_impl(q, k, v, softmax_scale)


@torch.library.register_fake("worldfoundry::flash_attention_4_forward")
def _flash_attention_4_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Fake tensor for Dynamo tracing; output width follows ``v``, not ``q``."""
    del k, softmax_scale
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


def flash_attention_4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    """Explicit inference-only FlashAttention 4 entry; layout ``b s n d``."""

    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
        raise RuntimeError("WorldFoundry FlashAttention 4 currently only supports inference")
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(
        q,
        k,
        v,
        q_pattern,
        k_pattern,
        v_pattern,
        required_in_pattern,
        dims,
    )
    out = torch.ops.worldfoundry.flash_attention_4_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        scale,
    )
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def flash_attention_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    """FlashAttention 2 entry (``flash_attn``); layout ``b s n d``."""
    import flash_attn

    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    """SageAttention entry (approximate; explicit opt-in); layout ``b n s d``."""
    from sageattention import sageattn

    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    """SageAttention 3 (Blackwell FP4). Does not support GQA/MQA or a custom softmax scale."""

    from sageattn3 import sageattn3_blackwell

    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    if q.shape[1] != k.shape[1] or k.shape[1] != v.shape[1]:
        raise RuntimeError("SageAttention 3 does not support GQA/MQA head layouts")
    default_scale = q.shape[-1] ** -0.5
    if scale is not None and not math.isclose(float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError("SageAttention 3 does not support a custom softmax scale")
    out = sageattn3_blackwell(q, k, v)
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def xformers_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    """xFormers memory-efficient attention. GQA is rejected until a 5D layout exists."""
    import xformers.ops as xops

    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Provider telemetry — prove which kernel ran without distorting the graph
# ──────────────────────────────────────────────────────────────────────────

_PROVIDER_RUNTIME_EVENT_NAMES = (
    "attempts",
    "successes",
    "fallbacks",
    "errors",
    "quarantined_skips",
    "compiled_graph_traces",
)
_PROVIDER_RUNTIME: dict[str, dict[str, int]] = {}
_PROVIDER_RUNTIME_LOCK = threading.Lock()
_PROVIDER_COMPILE_RECEIPT_SINK: ContextVar[
    MutableMapping[str, int] | None
] = ContextVar("worldfoundry_attention_compile_receipt_sink", default=None)


@contextmanager
def attention_compile_receipt_scope(
    sink: MutableMapping[str, int],
) -> Iterator[None]:
    """Bind compiled-provider receipts to the wrapper currently executing.

    The process-global report is useful for eager runs, but a compiled graph
    can be cached before request-scoped counters are reset.  The owning module
    therefore keeps its own persistent receipt map.  A context variable ties a
    Dynamo trace to that exact compiled wrapper without placing telemetry in
    the generated graph.
    """

    token = _PROVIDER_COMPILE_RECEIPT_SINK.set(sink)
    try:
        yield
    finally:
        _PROVIDER_COMPILE_RECEIPT_SINK.reset(token)


def _record_provider_runtime(selected: str, **increments: int) -> None:
    """Record provider events without putting locks around kernel execution.

    Dynamo/Inductor must not trace a Python lock or mutable process-global
    telemetry. The compiled-forward counter owned by the module loader proves
    compiled execution; eager and CUDA-Graph capture calls continue to prove
    the concrete attention provider here.
    """

    if torch.compiler.is_compiling():
        return
    unknown = set(increments) - set(_PROVIDER_RUNTIME_EVENT_NAMES)
    if unknown:
        raise ValueError(f"unknown attention provider runtime events {sorted(unknown)}")
    with _PROVIDER_RUNTIME_LOCK:
        counters = _PROVIDER_RUNTIME.setdefault(
            selected,
            {name: 0 for name in _PROVIDER_RUNTIME_EVENT_NAMES},
        )
        for event, increment in increments.items():
            counters[event] += int(increment)


@torch.compiler.assume_constant_result
def _record_compiled_provider_graph_trace(selected: str) -> int:
    """Write one compile-time receipt without adding an op to the runtime graph.

    Dynamo evaluates ``assume_constant_result`` functions while constructing a
    graph and replaces their return value with a constant.  Recording only
    after the provider callable traced successfully proves which provider is
    embedded in that graph; the module loader's compiled-wrapper call counter
    separately proves the graph was executed.  This avoids a Python callback
    or tiny counter kernel on every attention call, either of which would
    distort the benchmark being audited.
    """

    with _PROVIDER_RUNTIME_LOCK:
        counters = _PROVIDER_RUNTIME.setdefault(
            selected,
            {name: 0 for name in _PROVIDER_RUNTIME_EVENT_NAMES},
        )
        counters["compiled_graph_traces"] += 1
        count = counters["compiled_graph_traces"]
    sink = _PROVIDER_COMPILE_RECEIPT_SINK.get()
    if sink is not None:
        sink[selected] = int(sink.get(selected, 0)) + 1
    return count


def attention_provider_runtime_report() -> dict[str, dict[str, int]]:
    """Return provider execution counters copied under the telemetry lock.

    ``attempts`` and ``successes`` prove that a selected provider callable ran;
    ``fallbacks`` records only classified load/shape failures that proceeded to
    another backend. ``compiled_graph_traces`` is a zero-runtime-overhead
    receipt that the provider callable successfully entered a compiled graph;
    it becomes execution proof only together with the owning compiled
    wrapper's positive runtime call count. An installed wrapper with neither
    proof is therefore not misreported as runtime-effective.
    """

    with _PROVIDER_RUNTIME_LOCK:
        return {
            provider: dict(counters)
            for provider, counters in sorted(_PROVIDER_RUNTIME.items())
        }


def reset_attention_provider_runtime() -> None:
    """Reset provider call telemetry without changing capability caches."""

    with _PROVIDER_RUNTIME_LOCK:
        _PROVIDER_RUNTIME.clear()


def _invoke_attention_provider(
    selected: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str,
    k_pattern: str,
    v_pattern: str,
    out_pattern: str,
    dims,
    scale,
) -> torch.Tensor:
    """Invoke one qualified non-Torch provider or fail for an unwired name."""

    if selected == "cudnn_fp8":
        from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa

        q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, "b n s d", dims)
        output = native_cudnn_fp8_sdpa(q, k, v, scale=scale)
        return rearrange_out(output, out_pattern, "b n s d", dims)
    if selected == "flash_attention_4":
        return flash_attention_4(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if selected == "flash_attention_3":
        return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if selected == "flash_attention_2":
        return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if selected == "sage_attention":
        return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if selected == "sage_attention_3":
        return sage_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if selected == "xformers":
        return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    # Gives VSA/FlexBlock/V-MoBA/SLA/SageSLA an actionable model-contract
    # error. Any other resolved-but-unwired provider is an invariant breach.
    require_generic_attention_backend(selected)
    raise RuntimeError(
        f"Attention backend {selected!r} resolved as executable but has no generic provider callable"
    )


def _invoke_attention_provider_audited(selected: str, *args, **kwargs) -> torch.Tensor:
    """Run one non-Torch provider and record attempt/success/error/trace."""
    try:
        output = _invoke_attention_provider(selected, *args, **kwargs)
    except Exception:
        _record_provider_runtime(selected, attempts=1, errors=1)
        raise
    if torch.compiler.is_compiling():
        _record_compiled_provider_graph_trace(selected)
    else:
        _record_provider_runtime(selected, attempts=1, successes=1)
    return output


@torch.compiler.disable
def _invoke_uncompiled_attention_provider_audited(selected: str, *args, **kwargs) -> torch.Tensor:
    """Record actual opaque provider calls without claiming Dynamo graph traces."""

    return _invoke_attention_provider_audited(selected, *args, **kwargs)


def _invoke_torch_sdpa_audited(*args, **kwargs) -> torch.Tensor:
    """Run in-tree SDPA and record the same telemetry as optional providers."""
    try:
        output = torch_sdpa(*args, **kwargs)
    except Exception:
        _record_provider_runtime("torch", attempts=1, errors=1)
        raise
    if torch.compiler.is_compiling():
        _record_compiled_provider_graph_trace("torch")
    else:
        _record_provider_runtime("torch", attempts=1, successes=1)
    return output


# ──────────────────────────────────────────────────────────────────────────
# Dispatch — quarantine load/shape failures; torch SDPA is always last
# ──────────────────────────────────────────────────────────────────────────


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    attn_mask=None,
    scale=None,
    compatibility_mode=False,
    backend: str | None = None,
):
    """Dispatch QKV Attention across qualified backends for this device, dtype, and shape.

    Layouts are einops patterns and are normalized before execution. ``auto``
    only tries *explicit* candidates that are usable on the current hardware
    (the default list is empty, so the path is SDPA). Unsupported workload
    signatures are remembered, and in-tree PyTorch SDPA is always the exact
    fallback.

    Args:
        q / k / v: Tensors in ``q_pattern`` / ``k_pattern`` / ``v_pattern``.
        q_pattern / k_pattern / v_pattern: Input layouts.
        out_pattern: Requested output layout.
        dims: Named dimensions needed to expand grouped terms such as ``(n d)``.
        attn_mask: Boolean or additive mask. A mask forces SDPA because fused
            kernels do not share one mask contract.
        scale: Softmax scale; ``None`` uses the backend default.
        compatibility_mode: Skip optional providers and run PyTorch SDPA.
        backend: Optional request-scoped provider. When omitted, retain the
            process default resolved from ``WORLDFOUNDRY_ATTENTION_BACKEND``.

    Returns:
        Attention output rearranged into ``out_pattern``.

    Raises:
        ModelSpecificAttentionBackendError: A sparse attention system was
            selected without its required model/checkpoint metadata graph.
        RuntimeError: The selected provider failed for a reason other than an
            unsupported kernel or a missing optional library.

    Notes:
        Inspect the choice with ``attention_dispatch_report``. After changing
        provider availability in a long-lived process, call
        ``clear_attention_dispatch_cache``.
    """
    preferred = ATTENTION_IMPLEMENTATION if backend is None else normalize_attention_backend(backend)
    # Validate before compatibility/mask short-circuiting. An explicit sparse
    # request must never look successful merely because it silently took SDPA.
    require_generic_attention_backend(preferred)
    if preferred == "cudnn_fp8":
        if compatibility_mode or attn_mask is not None:
            raise ValueError("Explicit cudnn_fp8 does not support attention masks or compatibility-mode fallback")
        return _invoke_uncompiled_attention_provider_audited(
            preferred, q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale,
        )
    if compatibility_mode or (attn_mask is not None):
        return _invoke_torch_sdpa_audited(
            q,
            k,
            v,
            q_pattern,
            k_pattern,
            v_pattern,
            out_pattern,
            dims,
            attn_mask=attn_mask,
            scale=scale,
        )
    signature = _attention_signature(q, k, v, q_pattern, k_pattern, v_pattern, dims)
    candidates = _select_attention_backends_cached(
        preferred,
        str(q.device),
        str(q.dtype),
        signature,
        _short_attention_threshold(q.device),
        _unavailable_backends_snapshot(),
    )
    for selected in candidates:
        if selected == "torch":
            break
        failure_key = (selected, str(q.device), str(q.dtype), signature)
        if failure_key in _FAILED_ATTENTION_SIGNATURES:
            _record_provider_runtime(selected, quarantined_skips=1)
            continue
        try:
            return _invoke_attention_provider_audited(
                selected,
                q,
                k,
                v,
                q_pattern,
                k_pattern,
                v_pattern,
                out_pattern,
                dims,
                scale,
            )
        except (ImportError, OSError) as exc:
            if not _is_backend_load_error(selected, exc):
                raise
            _quarantine_unavailable_backend(selected)
            _record_provider_runtime(selected, fallbacks=1)
            _warn_backend_fallback(selected, exc)
        except RuntimeError as exc:
            if not _is_unsupported_kernel_error(exc):
                raise
            _remember_failed_attention_signature(failure_key)
            _record_provider_runtime(selected, fallbacks=1)
            _warn_backend_fallback(selected, exc)
    return _invoke_torch_sdpa_audited(
        q,
        k,
        v,
        q_pattern,
        k_pattern,
        v_pattern,
        out_pattern,
        dims,
        scale=scale,
    )


_FAILED_ATTENTION_SIGNATURE_LIMIT = 1024
_FAILED_ATTENTION_SIGNATURES: set[tuple[object, ...]] = set()
_FAILED_ATTENTION_SIGNATURE_ORDER: deque[tuple[object, ...]] = deque()
_FAILED_ATTENTION_SIGNATURE_LOCK = threading.Lock()
_UNAVAILABLE_ATTENTION_BACKENDS: set[str] = set()
_UNAVAILABLE_ATTENTION_BACKENDS_LOCK = threading.Lock()


def _unavailable_backends_snapshot() -> tuple[str, ...]:
    """Return a stable snapshot while other threads quarantine providers."""

    with _UNAVAILABLE_ATTENTION_BACKENDS_LOCK:
        return tuple(sorted(_UNAVAILABLE_ATTENTION_BACKENDS))


def _quarantine_unavailable_backend(selected: str) -> None:
    """Atomically mark an optional provider unavailable for this process."""

    with _UNAVAILABLE_ATTENTION_BACKENDS_LOCK:
        _UNAVAILABLE_ATTENTION_BACKENDS.add(selected)


# ──────────────────────────────────────────────────────────────────────────
# Workload eligibility — short / non-half / GQA constraints per provider
# ──────────────────────────────────────────────────────────────────────────


def _short_attention_threshold(device: torch.device | str | None = None) -> int:
    """Minimum sequence length that may try a fused kernel.

    Volta (SM70) pays more launch cost, so the default is 256. Hopper
    (SM90) is 64. Everything else is 128.
    ``WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE`` overrides; a non-integer
    value falls back to 128 rather than crashing dispatch.
    """
    try:
        configured = os.getenv("WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE")
        if configured is not None:
            return max(int(configured or "0"), 0)
    except ValueError:
        return 128
    capability = _device_compute_capability(device)
    if capability is not None and capability[0] == 7:
        return 256
    if capability is not None and capability[0] == 9:
        return 64
    return 128


def _device_compute_capability(device: torch.device | str | None = None) -> tuple[int, int] | None:
    """Return a validated NVIDIA compute capability for ``device``.

    HIP exposes a CUDA-compatible PyTorch device API, so checking only
    ``torch.cuda.is_available()`` can accidentally classify an AMD GPU as an
    NVIDIA architecture. Unknown and non-CUDA devices deliberately return
    ``None`` and use the exact PyTorch SDPA path.
    """

    if getattr(torch.version, "hip", None) is not None or not torch.cuda.is_available():
        return None
    try:
        resolved = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if resolved.type != "cuda":
            return None
        major, minor = torch.cuda.get_device_capability(resolved)
        return int(major), int(minor)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None


def _auto_attention_backends(device: torch.device | str | None) -> tuple[str, ...]:
    """Use the in-tree exact PyTorch provider for automatic dispatch.

    External FlashAttention/xFormers packages and approximate SageAttention
    remain explicit opt-ins. PyTorch SDPA already dispatches to its bundled
    Flash/cuDNN kernels and preserves WorldFoundry's no-external-repo contract.
    """

    del device
    return ()


def _tensor_layout_shape(tensor: torch.Tensor, pattern: str, dims: dict | None) -> tuple[int, int, int] | None:
    """Return ``(sequence, heads, head_dim)`` for common dispatcher layouts."""

    normalized = " ".join(str(pattern).split())
    if tensor.ndim == 4 and normalized == "b n s d":
        return int(tensor.shape[2]), int(tensor.shape[1]), int(tensor.shape[3])
    if tensor.ndim == 4 and normalized == "b s n d":
        return int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3])
    if tensor.ndim == 3 and normalized in {"b s (n d)", "b s (h d)"}:
        names = {} if dims is None else dims
        heads = int(names.get("n") or names.get("h") or 0)
        if heads > 0 and tensor.shape[-1] % heads == 0:
            return int(tensor.shape[1]), heads, int(tensor.shape[-1] // heads)
    return None


def _attention_signature(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str,
    k_pattern: str,
    v_pattern: str,
    dims: dict | None,
) -> tuple[object, ...]:
    """Hash QKV layout into a quarantine key, or ``unknown`` when unpackable.

    Unknown layouts still dispatch; they just cannot be shape-gated or
    remembered as a failed fused signature.
    """
    q_shape = _tensor_layout_shape(q, q_pattern, dims)
    k_shape = _tensor_layout_shape(k, k_pattern, dims)
    v_shape = _tensor_layout_shape(v, v_pattern, dims)
    if q_shape is None or k_shape is None or v_shape is None:
        return ("unknown", q.ndim, k.ndim, v.ndim, q.shape[-1], k.shape[-1], v.shape[-1])
    q_seq, q_heads, head_dim = q_shape
    k_seq, k_heads, k_dim = k_shape
    v_seq, v_heads, v_dim = v_shape
    return (
        "known",
        q_seq,
        k_seq,
        v_seq,
        q_heads,
        k_heads,
        v_heads,
        head_dim,
        k_dim,
        v_dim,
        str(k.dtype),
        str(v.dtype),
        str(k.device),
        str(v.device),
    )


@lru_cache(maxsize=2048)
def _select_attention_backends_cached(
    preferred: str,
    device: str,
    dtype: str,
    signature: tuple[object, ...],
    short_threshold: int,
    unavailable: tuple[str, ...],
) -> tuple[str, ...]:
    """Return usable backends in priority order for one workload signature."""

    if dtype not in {"torch.float16", "torch.bfloat16"}:
        return ("torch",)
    if preferred == "auto" and signature[0] == "known":
        q_seq, k_seq = int(signature[1]), int(signature[2])
        if max(q_seq, k_seq) < short_threshold:
            return ("torch",)

    if preferred == "auto":
        # Keep the automatic policy centralized in ``resolve_attention_backend``.
        # Its production contract resolves auto to Torch, while tests or future
        # offline selectors can inject an exact, qualified provider here.
        resolved = resolve_attention_backend(preferred, device)
        requested = () if resolved == "torch" else (resolved,)
    elif preferred == "flash_attention":
        requested = ("flash_attention_3", "flash_attention_2")
    else:
        resolved = resolve_attention_backend(preferred, device)
        requested = () if resolved == "torch" else (resolved,)

    blocked = set(unavailable)
    selected = tuple(
        candidate
        for candidate in requested
        if candidate not in blocked
        and attention_backend_capability(candidate, device).usable
        and _attention_backend_shape_eligible(candidate, device, dtype, signature)
    )
    return (*selected, "torch")


def _attention_backend_shape_eligible(
    selected: str,
    device: str,
    dtype: str,
    signature: tuple[object, ...],
) -> bool:
    """Return whether ``selected`` can serve this signature without a kernel error.

    Rejects mismatched K/V, GQA on xFormers/Sage3, illegal head dims, and
    Sage3 sequences below 512. Unknown signatures return True so the
    kernel itself is the source of truth.
    """
    if signature[0] != "known":
        return True

    (
        _,
        _q_seq,
        k_seq,
        v_seq,
        q_heads,
        k_heads,
        v_heads,
        head_dim,
        k_dim,
        v_dim,
        k_dtype,
        v_dtype,
        k_device,
        v_device,
    ) = signature
    if k_seq != v_seq:
        return False
    if k_dtype != dtype or v_dtype != dtype or k_device != device or v_device != device:
        return False
    if k_heads != v_heads or head_dim != k_dim or head_dim != v_dim:
        return False
    if k_heads <= 0 or q_heads % k_heads:
        return False
    if selected in {"flash_attention_2", "flash_attention_3", "flash_attention_4"}:
        if int(head_dim) <= 0 or int(head_dim) > 256 or int(head_dim) % 8:
            return False
    if selected == "sage_attention" and int(head_dim) not in {64, 128}:
        return False
    if selected == "sage_attention_3":
        if int(head_dim) not in {64, 128}:
            return False
        if q_heads != k_heads or k_heads != v_heads:
            return False
        if max(int(_q_seq), int(k_seq)) < 512:
            return False
    if selected == "xformers" and (q_heads != k_heads or k_heads != v_heads):
        # Common xFormers releases require explicit 5D grouped layout for
        # GQA/MQA. Until that layout is built here, keep the safe MHA path.
        return False
    return True


def _remember_failed_attention_signature(key: tuple[object, ...]) -> None:
    """Record a failed fused signature, evicting the oldest past the cap.

    The cap keeps a long-lived process from growing an unbounded set of
    shape keys. Eviction is FIFO, not LRU — a rare shape may be retried.
    """
    with _FAILED_ATTENTION_SIGNATURE_LOCK:
        if key in _FAILED_ATTENTION_SIGNATURES:
            return
        if len(_FAILED_ATTENTION_SIGNATURE_ORDER) >= _FAILED_ATTENTION_SIGNATURE_LIMIT:
            expired = _FAILED_ATTENTION_SIGNATURE_ORDER.popleft()
            _FAILED_ATTENTION_SIGNATURES.discard(expired)
        _FAILED_ATTENTION_SIGNATURE_ORDER.append(key)
        _FAILED_ATTENTION_SIGNATURES.add(key)


def attention_dispatch_report() -> dict[str, object]:
    """Return lightweight operator-selection and quarantine state."""

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda", torch.cuda.current_device())
        except (AssertionError, RuntimeError, ValueError):
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    capability = _device_compute_capability(device)
    cache_info = getattr(_select_attention_backends_cached, "cache_info", None)
    return {
        "requested": ATTENTION_IMPLEMENTATION,
        "device": str(device),
        "compute_capability": capability,
        "auto_priority": list(_auto_attention_backends(device)) or ["torch"],
        "native_sdpa_priority": list(native_sdpa_priority(device)) or ["pytorch-auto"],
        "selection_cache": (
            cache_info()._asdict()
            if callable(cache_info)
            else {"injected_selector": True}
        ),
        "unavailable_backends": list(_unavailable_backends_snapshot()),
        "failed_signatures": len(_FAILED_ATTENTION_SIGNATURES),
        "provider_calls": attention_provider_runtime_report(),
        "min_fused_sequence": _short_attention_threshold(device),
    }


def clear_attention_dispatch_cache() -> None:
    """Clear workload decisions and runtime failure quarantine."""

    _select_attention_backends_cached.cache_clear()
    with _FAILED_ATTENTION_SIGNATURE_LOCK:
        _FAILED_ATTENTION_SIGNATURES.clear()
        _FAILED_ATTENTION_SIGNATURE_ORDER.clear()
    with _UNAVAILABLE_ATTENTION_BACKENDS_LOCK:
        _UNAVAILABLE_ATTENTION_BACKENDS.clear()


# ──────────────────────────────────────────────────────────────────────────
# Failure classification — quarantine load/shape errors; never quarantine OOM
# ──────────────────────────────────────────────────────────────────────────

_UNSUPPORTED_KERNEL_MARKERS = (
    "not supported",
    "unsupported",
    "only supports",
    "requires sm",
    "no kernel image is available",
    "invalid device function",
    "not compiled with",
)

_BACKEND_IMPORT_ROOTS = {
    "flash_attention_4": ("flash_attn",),
    "flash_attention_3": ("flash_attn_interface",),
    "flash_attention_2": ("flash_attn", "flash_attn_2_cuda"),
    "sage_attention": ("sageattention",),
    "sage_attention_3": ("sageattn3",),
    "xformers": ("xformers",),
}
_DYNAMIC_LIBRARY_ERROR_MARKERS = (
    "cannot open shared object file",
    "dll load failed",
    "image not found",
    "undefined symbol",
    "symbol not found",
    "library not loaded",
)


def _is_backend_load_error(selected: str, exc: ImportError | OSError) -> bool:
    """Classify only optional-package or dynamic-loader availability errors."""

    roots = _BACKEND_IMPORT_ROOTS.get(selected, ())
    if isinstance(exc, ImportError):
        # Once execution is inside a selected optional provider, a missing
        # transitive Python/CUDA module is a provider availability failure too.
        return bool(roots)
    message = str(exc).lower()
    if not any(marker in message for marker in _DYNAMIC_LIBRARY_ERROR_MARKERS):
        return False
    # Loader errors commonly mention only a transitive dependency such as
    # libcudart/libtorch rather than the importing backend module.
    return True


def _is_unsupported_kernel_error(exc: RuntimeError) -> bool:
    """Return True only for explicit kernel-availability failures."""

    message = str(exc).lower()
    if "out of memory" in message or "alloc_failed" in message:
        return False
    return any(marker in message for marker in _UNSUPPORTED_KERNEL_MARKERS)


def _warn_backend_fallback(selected: str, exc: BaseException) -> None:
    """Warn that ``selected`` failed and the next eligible backend will run."""
    warnings.warn(
        f"Attention backend {selected!r} is unavailable for this workload; "
        f"trying the next eligible backend: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )


def packed_sequence_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
    scale=None,
    backend: str | None = None,
):
    """Apply the shared dispatcher to flattened packed-sequence Q/K/V.

    Args:
        q: Query tensor shaped ``(batch, sequence, heads * head_dim)``.
        k: Key tensor with the same flattened-head convention.
        v: Value tensor with the same flattened-head convention.
        num_heads: Head count used to split the final dimension.
        compatibility_mode: Force the exact PyTorch SDPA path.
        scale: Optional attention softmax scale.
        backend: Optional request-scoped provider name.

    Returns:
        Tensor shaped ``(batch, query_sequence, heads * value_head_dim)``.

    Notes:
        This adapter is appropriate when the model already packs heads into
        the hidden dimension. Use ``attention_forward`` directly for custom
        layouts or explicit masks.
    """
    return attention_forward(
        q,
        k,
        v,
        q_pattern="b s (n d)",
        k_pattern="b s (n d)",
        v_pattern="b s (n d)",
        out_pattern="b s (n d)",
        dims={"n": num_heads},
        scale=scale,
        compatibility_mode=compatibility_mode,
        backend=backend,
    )


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "attention_compile_receipt_scope",
    "attention_dispatch_report",
    "attention_forward",
    "attention_provider_runtime_report",
    "clear_attention_dispatch_cache",
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "initialize_attention_priority",
    "packed_sequence_attention",
    "rearrange_out",
    "rearrange_qkv",
    "reset_attention_provider_runtime",
    "sage_attention",
    "sage_attention_3",
    "torch_sdpa",
    "xformers_attention",
]

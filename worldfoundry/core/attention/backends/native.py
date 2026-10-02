# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact SDPA attention with selectable QKV layout and kernel backend.

This is the in-tree precise path used when ``dispatch`` declines a fused
provider (mask, short sequence, non-half, or ``auto``). :func:`native_sdpa_priority`
picks cuDNN-first on Blackwell and lets Ampere/Ada/Hopper use PyTorch's
shape-aware order. :func:`normalize_fully_masked_rows` makes all-false
boolean rows backend-independent (cuDNN vs math disagree there).

:class:`NativeAttention` can lease torch Context Parallel options (disable
load-balance, allgather rotate) for the lifetime of a CP group, then restore
them so other libraries are not stuck with our settings.

Not this module:
    Fused Flash / Sage / xFormers probing lives in
    :mod:`worldfoundry.core.attention.backends.probe`. Packed / varlen kernels
    live in :mod:`.varlen`. Sequence-parallel Ulysses exchange lives in
    :mod:`.ulysses_attention`.

Public surface:

- :func:`scaled_dot_product_attention` — exact SDPA with backend pin and
  GQA compatibility for older PyTorch.
- :func:`flattened_multihead_attention` — ``[B, S, H*D]`` reshape adapter.
- :func:`native_sdpa_priority` / :func:`normalize_fully_masked_rows` —
  architecture order and all-false-row quarantine.
- :class:`NativeAttention` — module wrapper that can lease torch CP.
"""

import importlib
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

try:
    from torch.distributed.tensor.device_mesh import DeviceMesh
except ImportError:  # torch 2.4 exposes DeviceMesh from this module.
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor.experimental import context_parallel as _context_parallel
except ImportError:
    _context_parallel = None


_CP_OPTIONS_LOCK = threading.RLock()
_CP_OPTIONS_USERS = 0
_CP_OPTIONS_SNAPSHOT: tuple[Any, Any, Any] | None = None


# ──────────────────────────────────────────────────────────────────────────
# Torch CP option lease — process-global flags must restore after last user
# ──────────────────────────────────────────────────────────────────────────


def _acquire_torch_cp_options() -> bool:
    """Lease the private torch CP options while NativeAttention needs them."""

    global _CP_OPTIONS_SNAPSHOT, _CP_OPTIONS_USERS
    try:
        attention_module = importlib.import_module("torch.distributed.tensor.experimental._attention")
    except ImportError:
        return False
    options = getattr(attention_module, "_cp_options", None)
    if options is None or not hasattr(options, "enable_load_balance"):
        # PyTorch 2.4/2.5 does not expose this newer option object.
        return False

    with _CP_OPTIONS_LOCK:
        if _CP_OPTIONS_USERS == 0:
            _CP_OPTIONS_SNAPSHOT = (
                attention_module,
                options.enable_load_balance,
                getattr(options, "rotate_method", None),
            )
        _CP_OPTIONS_USERS += 1
        try:
            options.enable_load_balance = False
            set_rotate_method = getattr(attention_module, "set_rotate_method", None)
            if callable(set_rotate_method):
                set_rotate_method("allgather")
        except BaseException:
            _CP_OPTIONS_USERS -= 1
            if _CP_OPTIONS_USERS == 0:
                _CP_OPTIONS_SNAPSHOT = None
            raise
    return True


def _release_torch_cp_options() -> None:
    """Release one lease and restore the pre-first-user torch CP options."""

    global _CP_OPTIONS_SNAPSHOT, _CP_OPTIONS_USERS
    with _CP_OPTIONS_LOCK:
        if _CP_OPTIONS_USERS <= 0:
            return
        _CP_OPTIONS_USERS -= 1
        if _CP_OPTIONS_USERS != 0 or _CP_OPTIONS_SNAPSHOT is None:
            return
        attention_module, enable_load_balance, rotate_method = _CP_OPTIONS_SNAPSHOT
        options = getattr(attention_module, "_cp_options", None)
        if options is not None:
            options.enable_load_balance = enable_load_balance
            if rotate_method is not None and hasattr(options, "rotate_method"):
                options.rotate_method = rotate_method
        _CP_OPTIONS_SNAPSHOT = None


# ──────────────────────────────────────────────────────────────────────────
# Backend metadata and exact SDPA — shared by dispatch and model adapters
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttentionBackendInfo:
    """Resolved attention backend metadata."""

    backend: str
    uses_torch_sdpa: bool


def attention_backend_info() -> AttentionBackendInfo:
    """Return the available generic PyTorch attention backend."""

    has_sdpa = callable(getattr(F, "scaled_dot_product_attention", None))
    backend = "torch.scaled_dot_product_attention" if has_sdpa else "torch.einsum"
    return AttentionBackendInfo(backend=backend, uses_torch_sdpa=has_sdpa)


def attention_backend_context(
    backend: Literal["math", "efficient", "cudnn", "flash"] | Any | None = None,
    *,
    backends: Any = None,
) -> Any:
    """Return a context manager that selects PyTorch SDPA backends via core."""

    return _sdpa_kernel_context(backend=backend, backends=backends)


def native_sdpa_priority(
    device: torch.device | str | None = None,
    *,
    has_mask: bool = False,
) -> tuple[str, ...]:
    """Return an exact PyTorch SDPA order for one workload.

    Ampere, Ada, and Hopper use PyTorch's shape-aware dispatcher: cuDNN and
    FlashAttention cross over at different sequence/head/mask shapes even on
    one GPU. Blackwell data-center and client targets prefer cuDNN before the
    generic fallbacks because it is the architecture-native exact path; SM110
    embedded targets retain PyTorch's automatic choice until physically
    calibrated. The environment override remains available for offline tuning
    on a concrete deployment.
    """

    configured = os.getenv("WORLDFOUNDRY_NATIVE_SDPA_PRIORITY", "").strip()
    if configured:
        known = {"cudnn", "flash", "efficient", "math"}
        requested = tuple(
            item.strip().lower().replace("_attention", "") for item in configured.split(",") if item.strip()
        )
        invalid = tuple(item for item in requested if item not in known)
        if invalid:
            raise ValueError(f"Unknown native SDPA backends: {invalid}")
        return requested

    parsed = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None and torch.cuda.is_available()
        else torch.device(device or "cpu")
    )
    if parsed.type != "cuda":
        return ("math",)
    if getattr(torch.version, "hip", None) is not None:
        # Let PyTorch/AOTriton choose on ROCm until CDNA/RDNA-specific orders
        # are physically qualified by the project.
        return ()
    try:
        major, _minor = torch.cuda.get_device_capability(parsed)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return ()
    if major >= 8:
        if major in {10, 12}:
            # PyTorch's default order can reach efficient/math before cuDNN on
            # Blackwell even though cuDNN is the architecture-native exact
            # path. Ampere, Ada, Hopper, and SM110 embedded Blackwell retain
            # shape-dispatched selection for masked and unmasked workloads.
            return ("cudnn", "flash", "efficient", "math")
        return ()
    return ("efficient", "math")


def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    *args: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    backend: Literal["math", "efficient", "cudnn", "flash"] | Any | None = None,
    backends: Any = None,
) -> Any:
    """Compute exact scaled dot-product attention through one stable core API.

    The function mirrors PyTorch SDPA, adds explicit backend selection, and
    preserves compatibility with PyTorch versions that do not yet accept
    ``enable_gqa``. When native SDPA is unavailable it evaluates the same
    attention equation with matmul, softmax, and an optional dropout.

    Args:
        query: Query tensor shaped ``(..., query_length, head_dim)``. The
            common layout is ``(batch, heads, query_length, head_dim)``.
        key: Key tensor shaped ``(..., key_length, head_dim)``.
        value: Value tensor shaped ``(..., key_length, value_dim)``.
        *args: Backward-compatible positional values for ``attn_mask``,
            ``dropout_p``, ``is_causal``, and ``scale``, in that order.
        attn_mask: Boolean keep-mask or additive attention bias broadcastable
            to the attention score shape.
        dropout_p: Probability applied to attention weights. Pass ``0.0`` at
            inference time; SDPA applies a non-zero value even in eval mode.
        is_causal: Apply a lower-triangular causal mask.
        scale: Softmax scale. ``None`` uses ``1 / sqrt(head_dim)``.
        enable_gqa: Expand key/value heads when query has a compatible larger
            head count.
        backend: One requested PyTorch SDPA backend: ``math``, ``efficient``,
            ``cudnn``, or ``flash``.
        backends: Ordered backend collection. Takes precedence over
            ``backend`` and is useful when an explicit fallback order is
            required.

    Returns:
        Attention values with the query prefix shape and ``value.shape[-1]``
        as the final dimension.

    Raises:
        TypeError: More than four compatibility positional options are given.
        ValueError: Grouped-query attention head counts are incompatible.

    Notes:
        Prefer this function for already split heads. For flattened
        ``(batch, sequence, hidden)`` tensors, use
        ``flattened_multihead_attention`` so reshape and mask normalization
        stay centralized.
    """

    if args:
        if len(args) > 4:
            raise TypeError(f"scaled_dot_product_attention expected at most 4 positional options, got {len(args)}")
        if len(args) >= 1:
            attn_mask = args[0]
        if len(args) >= 2:
            dropout_p = args[1]
        if len(args) >= 3:
            is_causal = args[2]
        if len(args) >= 4:
            scale = args[3]

    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if callable(sdpa):
        kwargs: dict[str, Any] = {
            "attn_mask": attn_mask,
            "dropout_p": float(dropout_p),
            "is_causal": bool(is_causal),
        }
        if scale is not None:
            kwargs["scale"] = float(scale)
        if enable_gqa:
            kwargs["enable_gqa"] = True
        # ``torch.nn.attention.sdpa_kernel`` is a Python context manager and
        # cannot appear inside a Dynamo fullgraph. Inductor already lowers the
        # SDPA op to an eligible device kernel, so keep the eager-only provider
        # constraint outside compiled graphs and trace the operator directly.
        compiling = torch.compiler.is_compiling()
        context = nullcontext() if compiling else _sdpa_kernel_context(backend=backend, backends=backends)
        requested = backends if backends is not None else backend
        efficient_only = _only_efficient_backend(requested)
        with context:
            if enable_gqa and efficient_only:
                kwargs.pop("enable_gqa", None)
                output = _compatible_gqa(sdpa, query, key, value, kwargs)
            else:
                try:
                    output = sdpa(query, key, value, **kwargs)
                except TypeError as error:
                    if "enable_gqa" not in kwargs or "enable_gqa" not in str(error):
                        raise
                    kwargs.pop("enable_gqa")
                    output = _compatible_gqa(sdpa, query, key, value, kwargs)
        return normalize_fully_masked_rows(output, attn_mask, query, key)

    if enable_gqa:
        key, value = _repeat_key_value_for_gqa(key, value, query)
    dim = int(query.shape[-1])
    factor = (dim**-0.5) if scale is None else float(scale)
    scores = torch.matmul(query, key.transpose(-2, -1)) * factor
    if attn_mask is not None:
        if getattr(attn_mask, "dtype", None) is torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    if is_causal:
        q_len = int(query.shape[-2])
        k_len = int(key.shape[-2])
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, p=float(dropout_p), training=True)
    output = torch.matmul(weights, value)
    return normalize_fully_masked_rows(output, attn_mask, query, key)


def normalize_fully_masked_rows(
    output: Any,
    attn_mask: Any,
    query: Any,
    key: Any,
) -> Any:
    """Make all-false boolean rows backend-independent.

    CUDA SDPA backends do not all agree on the result of a query row whose
    boolean keep-mask contains no keys. In particular, cuDNN attention can
    return non-zero values while the math backend returns zeros. Rectangular
    bottom-right causal windows naturally create such leading rows when the
    query is longer than the key, so normalize them at the shared boundary.
    """

    if not isinstance(attn_mask, torch.Tensor) or attn_mask.dtype is not torch.bool:
        return output
    score_shape = (*query.shape[:-1], key.shape[-2])
    valid_rows = torch.broadcast_to(attn_mask, score_shape).any(dim=-1, keepdim=True)
    return torch.where(valid_rows, output, torch.zeros_like(output))


def flattened_multihead_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    backend: Literal["math", "efficient", "cudnn", "flash"] | Any | None = None,
    backends: Any = None,
) -> Tensor:
    """Apply SDPA to flattened ``[B, S, H*D]`` Q/K/V tensors.

    This is the shared layout adapter for diffusion transformers. Model code
    should keep projections/RoPE/mask construction locally and delegate the
    generic head reshape, mask canonicalization, backend context, and output
    merge here.
    """

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("flattened attention expects [B, S, hidden] Q/K/V tensors")
    if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
        raise ValueError("query, key and value must have compatible batch/sequence shapes")
    heads = int(num_heads)
    if heads <= 0:
        raise ValueError("num_heads must be positive")
    if query.shape[-1] % heads or key.shape[-1] % heads or value.shape[-1] % heads:
        raise ValueError("all hidden dimensions must be divisible by num_heads")
    query_head_dim = query.shape[-1] // heads
    key_head_dim = key.shape[-1] // heads
    value_head_dim = value.shape[-1] // heads
    if query_head_dim != key_head_dim:
        raise ValueError("query and key head dimensions must match")

    batch = query.shape[0]
    query_heads = query.view(batch, -1, heads, query_head_dim).transpose(1, 2)
    key_heads = key.view(batch, -1, heads, key_head_dim).transpose(1, 2)
    value_heads = value.view(batch, -1, heads, value_head_dim).transpose(1, 2)
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)
    output = scaled_dot_product_attention(
        query_heads,
        key_heads,
        value_heads,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        backend=backend,
        backends=backends,
    )
    return output.transpose(1, 2).reshape(batch, -1, heads * value_head_dim)


def _only_efficient_backend(requested: Any) -> bool:
    if isinstance(requested, (list, tuple)):
        if len(requested) != 1:
            return False
        requested = requested[0]
    if isinstance(requested, str):
        return requested.strip().lower() == "efficient"
    return getattr(requested, "name", None) == "EFFICIENT_ATTENTION"


def _compatible_gqa(sdpa, query: Tensor, key: Tensor, value: Tensor, kwargs: dict[str, Any]) -> Tensor:
    """Adapt GQA to kernels without native support, sharing KV storage when possible.

    Flatten batch and KV heads, and express each group of query heads against
    a zero-stride KV view. Head/batch-specific masks retain the explicit-head
    path because their axes cannot be passed unchanged to the grouped layout.
    No RuntimeError (including OOM) is caught or retried here.
    """

    mask = kwargs.get("attn_mask")
    if (
        query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0]
        and key.shape[1] == value.shape[1]
        and (mask is None or mask.ndim <= 2)
    ):
        batch, heads, length, width = query.shape
        kv_heads = key.shape[1]
        if kv_heads <= 0 or heads % kv_heads:
            raise ValueError(f"Cannot expand {kv_heads} KV heads to {heads} query heads.")
        groups = heads // kv_heads
        q = query.reshape(batch * kv_heads, groups, length, width)
        k = key.reshape(batch * kv_heads, 1, *key.shape[2:]).expand(-1, groups, -1, -1)
        v = value.reshape(batch * kv_heads, 1, *value.shape[2:]).expand(-1, groups, -1, -1)
        return sdpa(q, k, v, **kwargs).reshape(batch, heads, length, value.shape[-1])
    key, value = _repeat_key_value_for_gqa(key, value, query)
    return sdpa(query, key, value, **kwargs)


def _repeat_key_value_for_gqa(key: Any, value: Any, query: Any) -> tuple[Any, Any]:
    """Expand KV heads so older SDPA that lacks ``enable_gqa`` still matches Q.

    Failure: ``ValueError`` when query heads are not an integer multiple of
    KV heads — repeating would invent a layout the checkpoint never trained.
    """

    query_heads = int(query.shape[1])
    key_value_heads = int(key.shape[1])
    if key_value_heads == query_heads:
        return key, value
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise ValueError(f"Cannot expand {key_value_heads} KV heads to {query_heads} query heads.")
    repeats = query_heads // key_value_heads
    return (
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
    )


def _sdpa_kernel_context(*, backend: Any = None, backends: Any = None) -> Any:
    """Return ``sdpa_kernel`` when eager, else ``nullcontext`` inside Dynamo.

    A contextlib selector is not representable in a fullgraph. Compiled
    callers already lower ``F.scaled_dot_product_attention``; pinning
    providers there would break the graph rather than change the kernel.
    Missing ``sdpa_kernel`` or an empty resolved list is a no-op, not an
    error — the caller still has the math fallback.
    """

    requested = backends if backends is not None else backend
    if requested is None:
        return nullcontext()
    attention = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention, "sdpa_kernel", None) if attention is not None else None
    if not callable(sdpa_kernel):
        return nullcontext()
    resolved = _resolve_sdpa_backends(requested)
    if not resolved:
        return nullcontext()
    # A contextlib-based SDPA selector is not representable inside a Dynamo
    # fullgraph. Compiled callers trace F.scaled_dot_product_attention directly;
    # eager execution below still preserves the requested provider priority.
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None) if compiler is not None else None
    if callable(is_compiling) and is_compiling():
        return nullcontext()
    try:
        return sdpa_kernel(backends=resolved, set_priority=True)
    except TypeError:
        try:
            return sdpa_kernel(backends=resolved, set_priority_order=True)
        except TypeError:
            try:
                return sdpa_kernel(backends=resolved)
            except TypeError:
                return sdpa_kernel(resolved)


def _resolve_sdpa_backends(requested: Any) -> list[Any]:
    """Map string aliases onto ``SDPBackend`` enums; drop unknown names.

    Unknown strings are skipped rather than raised so an env-configured
    priority list still works on a PyTorch that lacks ``CUDNN_ATTENTION``.
    """

    if isinstance(requested, (str, bytes)) or not isinstance(requested, (list, tuple, set, frozenset)):
        values = [requested]
    else:
        values = list(requested)

    backend_type = getattr(getattr(torch.nn, "attention", None), "SDPBackend", None)
    backend_map = {
        "math": getattr(backend_type, "MATH", None) if backend_type is not None else None,
        "efficient": getattr(backend_type, "EFFICIENT_ATTENTION", None) if backend_type is not None else None,
        "cudnn": getattr(backend_type, "CUDNN_ATTENTION", None) if backend_type is not None else None,
        "flash": getattr(backend_type, "FLASH_ATTENTION", None) if backend_type is not None else None,
    }
    resolved: list[Any] = []
    for value in values:
        if isinstance(value, str):
            value = backend_map.get(value.strip().lower().replace("_", "-"))
        if value is not None:
            resolved.append(value)
    return resolved


# ──────────────────────────────────────────────────────────────────────────
# Module wrapper — optional torch CP lease around the same exact SDPA path
# ──────────────────────────────────────────────────────────────────────────


class NativeAttention(torch.nn.Module):
    """Native attention module with configurable QKV layout and SDPA backend."""

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        backend: Literal["math", "efficient", "cudnn", "flash"] = "cudnn",
    ) -> None:
        """Configure attention format and backend.

        Args:
            qkv_format: Layout of the QKV tensors; ``"bhsd"`` is ``(B, H, S, D)``,
                ``"bshd"`` is ``(B, S, H, D)``.
            backend: SDPA backend selected via ``sdpa_kernel``.
        """
        super().__init__()
        assert qkv_format in ["bhsd", "bshd"], f"Invalid qkv format: {qkv_format}"
        assert backend in ["math", "efficient", "cudnn", "flash"], f"Invalid backend: {backend}"
        self.qkv_format = qkv_format
        self.backend = backend
        self.device_mesh: DeviceMesh | None = None
        self._torch_cp_options_leased = False

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Enable or disable context parallelism for ring attention.

        Args:
            cp_group: Process group for context parallel; use None to disable.
        """
        if cp_group is None:
            self.device_mesh = None
            if self._torch_cp_options_leased:
                _release_torch_cp_options()
                self._torch_cp_options_leased = False
        else:
            if _context_parallel is None:
                raise RuntimeError(
                    "NativeAttention context parallel requires torch.distributed.tensor.experimental.context_parallel."
                )
            device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")
            if not self._torch_cp_options_leased:
                self._torch_cp_options_leased = _acquire_torch_cp_options()
            self.device_mesh = device_mesh

    def clear_context_parallel_group(self) -> None:
        """Disable context parallelism and release its process-global option lease."""

        self.set_context_parallel_group(None)

    def is_context_parallel_enabled(self) -> bool:
        """Return True if context parallelism is active."""
        return self.device_mesh is not None

    def context_parallel_size(self) -> int:
        """Return the context parallel world size, or 1 if disabled."""
        return self.device_mesh.size() if self.device_mesh is not None else 1

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Run context-parallel SDPA (or single-rank SDPA when CP is disabled).

        Args:
            query: Query tensor in configured ``qkv_format``.
            key: Key tensor in configured ``qkv_format``.
            value: Value tensor in configured ``qkv_format``.

        Returns:
            Attention output in the same format as inputs.
        """
        # SDPA / low-level ops expect (B, H, S, D). "bshd" is (B, S, H, D) → transpose once.
        if self.qkv_format == "bshd":
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        out = self._impl(query=query, key=key, value=value)
        if self.qkv_format == "bshd":
            out = out.transpose(1, 2)
        return out

    def _impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Attention implementation.

        Args:
            query: Query tensor, shape ``[B, H, S, D]`` (CP-shared).
            key: Key tensor, shape ``[B, H, S, D]`` (CP-sharded).
            value: Value tensor, shape ``[B, H, S, D]`` (CP-sharded).

        Returns:
            Attention output.
        """
        sdpa_backend = {
            "math": torch.nn.attention.SDPBackend.MATH,
            "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        }[self.backend]

        with torch.nn.attention.sdpa_kernel(sdpa_backend):
            if self.device_mesh is not None:
                # Pass a dummy buffer to satisfy context_parallel's buffers[0].device
                # check (required in PyTorch 2.9+ where buffers cannot be empty).
                _dummy = torch.empty(self.device_mesh.size(), device=query.device)
                with _context_parallel(
                    self.device_mesh,
                    buffers=[
                        _dummy,
                    ],
                    buffer_seq_dims=[
                        0,
                    ],
                    no_restore_buffers={_dummy},
                ):
                    out = F.scaled_dot_product_attention(query, key, value)
            else:
                out = F.scaled_dot_product_attention(query, key, value)

        return out


__all__ = [
    "AttentionBackendInfo",
    "NativeAttention",
    "attention_backend_info",
    "flattened_multihead_attention",
    "native_sdpa_priority",
    "normalize_fully_masked_rows",
    "scaled_dot_product_attention",
]

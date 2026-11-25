"""Instance-local steady-state CUDA Graph runner for inference.

The existing :mod:`worldfoundry.core.execution.graphs.cuda_graph` module is autograd- and
training-oriented (RNG juggling, backward capture). Diffusion inference wants
something leaner: a per-callable runner that captures one CUDA Graph per input
signature during warmup, then replays it by ``copy_``-ing new inputs into the
captured static buffers. This removes per-launch CPU overhead on the stable
denoise-block loop and composes with FP8 linears and FA3 attention, which run
*inside* the captured region.

Contract and safety (mirrors ``plan/inference_operator_optimization_plan.md``
§6.3, §7 and the risk table):

- **Instance-local.** No process globals; each runner owns its graphs and pool.
- **Signature-bucketed.** A graph is keyed by the full tensor identity (shape,
  stride, dtype, device, layout) plus the ambient math/autocast policy, reusing
  ``_tensor_tree_signature`` / ``_cuda_graph_cache_key`` semantics. Unseen
  signatures beyond ``max_graphs`` fall back to eager rather than capturing
  without bound.
- **Static storage.** Inputs are copied into pre-allocated buffers; ``replay``
  never changes storage. Outputs are returned as the captured tensors' clones
  by default so callers cannot alias the static output across replays.
- **Eager fallback.** CPU tensors, capture failure, non-tensor leaves, or
  ``requires_grad`` inputs run the wrapped callable directly. OOM during capture
  propagates (retrying usually raises peak memory).
- **Warmup.** A few eager iterations run on a side stream before capture so lazy
    allocations and autotune settle, per the plan's warmup rule.

Public surface
    :class:`InferenceCUDAGraphRunner`.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from worldfoundry.core.execution.graphs.cuda_graph import _tensor_tree_signature, graph_pool_handle

_DEFAULT_WARMUP = 3
_DEFAULT_MAX_GRAPHS = 16

_EXACT_SCALAR_TYPES = frozenset((bool, int, float, str, torch.dtype, torch.device, type(None)))
_ELIGIBLE_LEAF_TYPES = (torch.Tensor, bool, int, float, str, torch.dtype, torch.device)

# Resolve version-dependent autocast and TF32 getters once at import time so
# steady-state replay skips per-call try/except and __getattr__ wrappers.
try:
    torch.is_autocast_enabled("cuda")
    _AUTOCAST_ENABLED_TAKES_DEVICE = True
except TypeError:  # pragma: no cover - PyTorch < 2.4
    _AUTOCAST_ENABLED_TAKES_DEVICE = False

_HAS_GET_AUTOCAST_DTYPE = hasattr(torch, "get_autocast_dtype")


def _read_matmul_allow_tf32() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32)


def _read_cudnn_allow_tf32() -> bool:
    return bool(torch.backends.cudnn.allow_tf32)


_c_get_cublas_allow_tf32 = getattr(torch._C, "_get_cublas_allow_tf32", None)
_c_get_cudnn_allow_tf32 = getattr(torch._C, "_get_cudnn_allow_tf32", None)
_c_get_float32_matmul_precision = getattr(torch._C, "_get_float32_matmul_precision", None)

_get_matmul_allow_tf32: Callable[[], bool] = (
    _c_get_cublas_allow_tf32 if callable(_c_get_cublas_allow_tf32) else _read_matmul_allow_tf32
)
_get_cudnn_allow_tf32: Callable[[], bool] = (
    _c_get_cudnn_allow_tf32 if callable(_c_get_cudnn_allow_tf32) else _read_cudnn_allow_tf32
)
_get_float32_matmul_precision: Callable[[], str] = (
    _c_get_float32_matmul_precision if callable(_c_get_float32_matmul_precision) else torch.get_float32_matmul_precision
)
_is_grad_enabled = torch.is_grad_enabled
_is_inference_mode_enabled = torch.is_inference_mode_enabled
_are_deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled


# ──────────────────────────────────────────────────────────────────────────
# Signature helpers — Graph identity binds tensors + ambient math policy
# ──────────────────────────────────────────────────────────────────────────


def _flatten_tensors(args: Sequence[Any], kwargs: Mapping[str, Any] | None = None) -> list[torch.Tensor]:
    """Collect tensor leaves, including explicit per-layer K/V tuples."""
    leaves, _ = tree_flatten((args, kwargs or {}))
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _ambient_policy_key_tuple() -> tuple[bool, bool, torch.dtype, str, bool, bool, bool]:
    """Compact hashable ambient policy tuple for eligible (no-grad) calls.

    ``grad_enabled`` is guaranteed ``False`` by the caller's eligibility check.
    Keeping ``autocast_dtype`` as a ``torch.dtype`` avoids per-replay string
    allocation while preserving exact bucket identity.
    """
    autocast_enabled = (
        torch.is_autocast_enabled("cuda") if _AUTOCAST_ENABLED_TAKES_DEVICE else torch.is_autocast_enabled()
    )
    autocast_dtype = torch.get_autocast_dtype("cuda") if _HAS_GET_AUTOCAST_DTYPE else torch.get_autocast_gpu_dtype()
    return (
        _is_inference_mode_enabled(),
        bool(autocast_enabled),
        autocast_dtype,
        _get_float32_matmul_precision(),
        _get_matmul_allow_tf32(),
        _get_cudnn_allow_tf32(),
        _are_deterministic_algorithms_enabled(),
    )


def _ambient_policy_key() -> dict[str, Any]:
    """Math/autocast/grad policy that a captured graph implicitly binds."""
    (
        inference_mode,
        autocast_enabled,
        autocast_dtype,
        float32_matmul_precision,
        matmul_allow_tf32,
        cudnn_allow_tf32,
        deterministic_algorithms,
    ) = _ambient_policy_key_tuple()
    return {
        "grad_enabled": _is_grad_enabled(),
        "inference_mode": inference_mode,
        "autocast_enabled": autocast_enabled,
        "autocast_dtype": str(autocast_dtype),
        "float32_matmul_precision": float32_matmul_precision,
        "matmul_allow_tf32": matmul_allow_tf32,
        "cudnn_allow_tf32": cudnn_allow_tf32,
        "deterministic_algorithms": deterministic_algorithms,
    }


def _scalar_leaf_key(leaf: Any) -> Any:
    """Build a collision-free hashable key for an eligible non-tensor leaf.

    Includes ``type(leaf)`` so ``1``, ``1.0``, and ``True`` never collide in
    Python tuple/dict hashing, and normalizes signed zero (``+0.0`` vs ``-0.0``)
    and ``NaN`` via ``repr`` so repeated ``NaN`` constants still cache-hit.
    """
    if leaf is None:
        return None
    leaf_type = type(leaf)
    if leaf_type is float:
        if leaf == 0.0 or leaf != leaf:
            return (float, repr(leaf))
        return (float, leaf)
    if leaf_type in _EXACT_SCALAR_TYPES:
        return (leaf_type, leaf)
    return (leaf_type, repr(leaf))


_INELIGIBLE = object()
_PYTREE_FALLBACK = object()


def _inspect_builtin_value(
    value: Any,
    tensors: list[torch.Tensor],
    device_box: list[torch.device | None],
) -> Any:
    """Single-pass walker for plain tensors, exact scalars, and built-in containers.

    Preserves exact container types (``tuple``, ``list``, ``dict``) and ``dict``
    insertion order to match :func:`torch.utils._pytree.tree_flatten` leaf order.
    Returns a hashable key, ``_INELIGIBLE`` when the input cannot be graphed, or
    ``_PYTREE_FALLBACK`` for subclasses / custom pytree nodes.
    """
    v_type = type(value)
    if v_type is torch.Tensor:
        if not value.is_cuda or value.requires_grad:
            return _INELIGIBLE
        dev = value.device
        first = device_box[0]
        if first is None:
            device_box[0] = dev
        elif dev != first:
            return _INELIGIBLE
        tensors.append(value)
        return (
            value.shape,
            value.stride(),
            value.dtype,
            value.layout,
            value.names if value.has_names() else None,
            value.is_conj(),
            value.is_neg(),
        )
    if v_type in _EXACT_SCALAR_TYPES:
        return _scalar_leaf_key(value)
    if v_type is tuple or v_type is list:
        out: list[Any] = [v_type]
        for item in value:
            item_key = _inspect_builtin_value(item, tensors, device_box)
            if item_key is _INELIGIBLE or item_key is _PYTREE_FALLBACK:
                return item_key
            out.append(item_key)
        return tuple(out)
    if v_type is dict:
        out = [dict]
        for dk, dv in value.items():
            dk_type = type(dk)
            if dk_type not in _EXACT_SCALAR_TYPES:
                return _PYTREE_FALLBACK
            key_sig = _scalar_leaf_key(dk)
            val_sig = _inspect_builtin_value(dv, tensors, device_box)
            if val_sig is _INELIGIBLE or val_sig is _PYTREE_FALLBACK:
                return val_sig
            out.append((key_sig, val_sig))
        return tuple(out)
    return _PYTREE_FALLBACK


def _tensor_leaf_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Hashable tensor-layout key matching ``_tensor_tree_signature`` fields.

    Device equality and ``requires_grad is False`` are enforced once per call;
    CUDA capability is a pure function of ``device``. Also binds ``is_conj``
    and ``is_neg`` so conjugated/negated views stay in distinct buckets.
    """
    names = tensor.names if tensor.has_names() else None
    t_type = type(tensor)
    if t_type is torch.Tensor:
        return (
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
            tensor.layout,
            names,
            tensor.is_conj(),
            tensor.is_neg(),
        )
    return (
        t_type,
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.layout,
        names,
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _arg_signature(value: Any) -> Any:
    """Describe pytree structure, tensor layouts and immutable constants."""
    leaves, spec = tree_flatten(value)
    return {
        "tree": str(spec),
        "leaves": [
            _tensor_tree_signature(leaf)
            if isinstance(leaf, torch.Tensor)
            else {"type": type(leaf).__qualname__, "const": repr(leaf)}
            for leaf in leaves
        ],
    }


def _signature_key(args: Sequence[Any], kwargs: Mapping[str, Any], extra_key: str | None) -> str:
    """Hash args, kwargs, ambient policy, and optional salt into one cache key."""
    payload = {
        "version": 3,
        "args": [_arg_signature(a) for a in args],
        "kwargs": {name: _arg_signature(kwargs[name]) for name in sorted(kwargs)},
        "policy": _ambient_policy_key(),
        "extra_key": extra_key,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"infer-graph-v3:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


class _CapturedGraph:
    """One captured Graph plus the static buffers replay must copy into."""

    __slots__ = ("graph", "input_indices", "static_inputs", "static_outputs")

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_inputs: list[torch.Tensor],
        static_outputs: Any,
        input_indices: list[int],
    ) -> None:
        """Store graph, static I/O, and indices in the flattened input tree."""
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self.input_indices = input_indices


class InferenceCUDAGraphRunner:
    """Capture-and-replay a pure-inference callable over stable input shapes.

    ``fn`` must be a side-effect-free forward over its tensor arguments; it may
    accept non-tensor arguments (kept constant per signature). Call the runner
    like the wrapped function.

    Why capture/replay instead of launching eager kernels every step:

    - Diffusion denoise blocks have a *steady* CUDA work graph once shapes,
      dtypes, and autocast policy settle. Recreating that launch stream each
      step burns CPU time that CUDA Graph ``replay`` removes.
    - FP8 linears and FlashAttention 3 already run *inside* the captured
      region. The runner does not replace those kernels; it only freezes the
      outer launch order.

    Why there is an eager fallback (and when it fires):

    - CPU tensors, empty tensor lists, ``requires_grad``, or ``grad_enabled``
      cannot be captured safely — Graph replay would alias autograd storage.
    - Capture warmup or ``torch.cuda.graph`` raising a non-OOM ``RuntimeError``
      disables that signature until invalidation. Allocator OOM propagates;
      an eager retry must not conceal a failed memory budget.
    - Unseen signatures after ``max_graphs`` stay eager. Unbounded capture
      would pin a Graph (and its static buffers) per rare shape.

    Replay copies fresh inputs into the captured static buffers with
    ``copy_`` so storage addresses stay identical. Outputs are cloned by
    default so callers cannot alias the static output across two replays.
    Pair with :class:`~worldfoundry.core.execution.graphs.cuda_graph_dispatch.CUDAGraphDispatch`
    when the *policy* of which AR index to graph lives outside this runner.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        warmup: int = _DEFAULT_WARMUP,
        max_graphs: int = _DEFAULT_MAX_GRAPHS,
        clone_outputs: bool = True,
        enabled: bool = True,
        extra_key: str | None = None,
        pool: Any | None = None,
    ) -> None:
        """Wrap ``fn`` with an instance-local Graph cache.

        Args:
            fn: Pure-inference callable. Tensor arguments are rebound on
                replay; non-tensor arguments must stay constant per signature.
            warmup: Eager iterations on a side stream before capture so lazy
                allocations and Triton autotune are not baked into the Graph.
            max_graphs: LRU cap. Extra signatures fall back to eager.
            clone_outputs: If True (default), clone captured outputs so the
                static buffers are not exposed to the caller.
            enabled: Master switch. False always runs eager.
            extra_key: Optional salt folded into the signature (for example a
                model revision or compile mode) so two policies do not share
                one Graph.
            pool: Optional CUDA Graph memory pool handle (from
                :func:`torch.cuda.graph_pool_handle`). When ``None`` (default),
                the runner lazily allocates its own private pool on first
                capture. Pass one shared pool handle across mutually-exclusive
                sequential runners to reuse intermediate activation buffers.
        """
        self._fn = fn
        self._warmup = int(warmup)
        self._max_graphs = int(max_graphs)
        self._clone_outputs = bool(clone_outputs)
        self._enabled = bool(enabled)
        self._extra_key = extra_key
        self._graphs: OrderedDict[Any, _CapturedGraph] = OrderedDict()
        self._pool: Any | None = pool
        self._owns_pool: bool = pool is None
        self._disabled_keys: set[Any] = set()
        self.stats = {"capture": 0, "replay": 0, "eager": 0, "capture_failed": 0}
        self._request_window_id = 0
        self._request_window_baseline = dict(self.stats)
        self._capture_on_miss = True

    @property
    def pool(self) -> Any | None:
        """CUDA Graph memory pool handle (None until supplied or first capture)."""
        return self._pool

    def freeze(self) -> None:
        """Use eager execution for uncaptured signatures after service warmup.

        Explicit :meth:`capture` remains available between requests. Existing
        signatures still replay; request execution never starts a new capture.
        """
        self._capture_on_miss = False

    @contextmanager
    def capture_scope(self) -> Generator[None, None, None]:
        """Allow capture during explicit warmup, then restore the serving policy."""
        previous = self._capture_on_miss
        self._capture_on_miss = True
        try:
            yield
        finally:
            self._capture_on_miss = previous

    @property
    def capture_on_miss(self) -> bool:
        """Whether an ordinary call may capture an unseen signature."""
        return self._capture_on_miss

    def capture(self, *args: Any, **kwargs: Any) -> bool:
        """Prepare one signature between requests; return whether it is captured.

        This does not return graph-owned outputs or count a replay. CPU and
        unsupported inputs return False; allocator OOM propagates.
        """
        if not torch.cuda.is_available():
            return False
        inspected = self._inspect_call(args, kwargs)
        if inspected is None:
            return False
        key, _, _ = inspected
        if key in self._graphs:
            return True
        if key in self._disabled_keys or len(self._graphs) >= self._max_graphs:
            return False
        return self._capture(key, args, kwargs) is not None

    # ──────────────────────────────────────────────────────────────────────
    # Eligibility / capture / replay — eager on CPU, grad, or capture failure
    # ──────────────────────────────────────────────────────────────────────

    def begin_request_window(self) -> None:
        """Start request-scoped telemetry while retaining captured graphs.

        Graph objects are intentionally lifetime-scoped: a later diffusion
        request should replay an already captured stable signature. Counters
        used for certification are not lifetime-scoped, however, because a
        capture/replay during warmup or a previous request must not prove that
        the current request used CUDA Graph execution.
        """

        self._request_window_id += 1
        self._request_window_baseline = dict(self.stats)

    def _eligible(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        """Reject CPU, empty, or autograd inputs — Graph replay would alias storage."""
        if not self._enabled or not torch.cuda.is_available():
            return False
        tensors = _flatten_tensors(args, kwargs)
        if not tensors:
            return False
        if any(t.device.type != "cuda" or t.device != tensors[0].device for t in tensors):
            return False
        if any(t.requires_grad for t in tensors) or _is_grad_enabled():
            return False
        leaves, _ = tree_flatten((args, kwargs))
        return all(isinstance(leaf, _ELIGIBLE_LEAF_TYPES) or leaf is None for leaf in leaves)

    def _inspect_call(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], Sequence[torch.Tensor], bool] | None:
        """Single-pass eligibility validation, key construction, and tensor extraction.

        Returns ``(key, tensor_leaves, inference_mode)`` when the call is
        CUDA-Graph eligible, or ``None`` when it must run eagerly. Avoids
        ``tree_flatten``, JSON serialization, and SHA-256 hashing on flat tensor
        and scalar calls, and flattens nested pytrees at most once.
        """
        if not self._enabled or _is_grad_enabled():
            return None
        # Only query torch.cuda.is_available() before the first capture; once
        # graphs exist, every eligible call still checks tensor.is_cuda below.
        if not self._graphs and not torch.cuda.is_available():
            return None

        # Fast path 0 & 1: positional-only calls (most block-stack segments).
        if not kwargs:
            if len(args) == 1:
                arg0 = args[0]
                if type(arg0) is torch.Tensor:
                    if not arg0.is_cuda or arg0.requires_grad:
                        return None
                    policy = _ambient_policy_key_tuple()
                    key: tuple[Any, ...] = (
                        0,
                        self._extra_key,
                        arg0.device,
                        policy,
                        arg0.shape,
                        arg0.stride(),
                        arg0.dtype,
                        arg0.layout,
                        arg0.names if arg0.has_names() else None,
                        arg0.is_conj(),
                        arg0.is_neg(),
                    )
                    return key, args, policy[0]

            device_box: list[torch.device | None] = [None]
            builtin_ok = True
            arg_keys: list[Any] = []
            tensors: list[torch.Tensor] = []

            for arg in args:
                arg_type = type(arg)
                if arg_type is torch.Tensor:
                    if not arg.is_cuda or arg.requires_grad:
                        return None
                    dev = arg.device
                    first = device_box[0]
                    if first is None:
                        device_box[0] = dev
                    elif dev != first:
                        return None
                    tensors.append(arg)
                    arg_keys.append(
                        (
                            arg.shape,
                            arg.stride(),
                            arg.dtype,
                            arg.layout,
                            arg.names if arg.has_names() else None,
                            arg.is_conj(),
                            arg.is_neg(),
                        )
                    )
                elif arg_type in _EXACT_SCALAR_TYPES:
                    arg_keys.append(_scalar_leaf_key(arg))
                elif arg_type is tuple or arg_type is list or arg_type is dict:
                    nested_key = _inspect_builtin_value(arg, tensors, device_box)
                    if nested_key is _INELIGIBLE:
                        return None
                    if nested_key is _PYTREE_FALLBACK:
                        builtin_ok = False
                        break
                    arg_keys.append(nested_key)
                else:
                    builtin_ok = False
                    break

            if builtin_ok:
                first_device = device_box[0]
                if first_device is None:
                    return None
                policy = _ambient_policy_key_tuple()
                key = (1, self._extra_key, first_device, policy, tuple(arg_keys))
                return key, tensors, policy[0]

        # Fast path 2: positional + keyword args over plain tensors, scalars, and built-in containers.
        else:
            device_box = [None]
            builtin_ok = True
            arg_keys = []
            tensors = []

            for arg in args:
                arg_type = type(arg)
                if arg_type is torch.Tensor:
                    if not arg.is_cuda or arg.requires_grad:
                        return None
                    dev = arg.device
                    first = device_box[0]
                    if first is None:
                        device_box[0] = dev
                    elif dev != first:
                        return None
                    tensors.append(arg)
                    arg_keys.append(
                        (
                            arg.shape,
                            arg.stride(),
                            arg.dtype,
                            arg.layout,
                            arg.names if arg.has_names() else None,
                            arg.is_conj(),
                            arg.is_neg(),
                        )
                    )
                elif arg_type in _EXACT_SCALAR_TYPES:
                    arg_keys.append(_scalar_leaf_key(arg))
                elif arg_type is tuple or arg_type is list or arg_type is dict:
                    nested_key = _inspect_builtin_value(arg, tensors, device_box)
                    if nested_key is _INELIGIBLE:
                        return None
                    if nested_key is _PYTREE_FALLBACK:
                        builtin_ok = False
                        break
                    arg_keys.append(nested_key)
                else:
                    builtin_ok = False
                    break

            if builtin_ok:
                kw_names = tuple(kwargs) if len(kwargs) == 1 else tuple(sorted(kwargs))
                kw_keys: list[Any] = []
                for name in kw_names:
                    val = kwargs[name]
                    val_type = type(val)
                    if val_type is torch.Tensor:
                        if not val.is_cuda or val.requires_grad:
                            return None
                        dev = val.device
                        first = device_box[0]
                        if first is None:
                            device_box[0] = dev
                        elif dev != first:
                            return None
                        tensors.append(val)
                        kw_keys.append(
                            (
                                val.shape,
                                val.stride(),
                                val.dtype,
                                val.layout,
                                val.names if val.has_names() else None,
                                val.is_conj(),
                                val.is_neg(),
                            )
                        )
                    elif val_type in _EXACT_SCALAR_TYPES:
                        kw_keys.append(_scalar_leaf_key(val))
                    elif val_type is tuple or val_type is list or val_type is dict:
                        nested_key = _inspect_builtin_value(val, tensors, device_box)
                        if nested_key is _INELIGIBLE:
                            return None
                        if nested_key is _PYTREE_FALLBACK:
                            builtin_ok = False
                            break
                        kw_keys.append(nested_key)
                    else:
                        builtin_ok = False
                        break

                if builtin_ok:
                    first_device = device_box[0]
                    if first_device is None:
                        return None
                    policy = _ambient_policy_key_tuple()
                    key = (2, self._extra_key, first_device, policy, tuple(arg_keys), kw_names, tuple(kw_keys))
                    return key, tensors, policy[0]

        # General path 3: single pytree flatten for custom pytree nodes / tensor subclasses.
        canonical_kwargs = {name: kwargs[name] for name in sorted(kwargs)} if kwargs else {}
        leaves, spec = tree_flatten((args, canonical_kwargs))
        first_device = None
        leaf_keys: list[Any] = []
        tensors = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                if not leaf.is_cuda or leaf.requires_grad:
                    return None
                dev = leaf.device
                if first_device is None:
                    first_device = dev
                elif dev != first_device:
                    return None
                tensors.append(leaf)
                leaf_keys.append(_tensor_leaf_key(leaf))
            elif leaf is None or isinstance(leaf, _ELIGIBLE_LEAF_TYPES):
                leaf_keys.append(_scalar_leaf_key(leaf))
            else:
                return None

        if first_device is None:
            return None
        policy = _ambient_policy_key_tuple()
        key = (3, self._extra_key, first_device, policy, spec, tuple(leaf_keys))
        return key, tensors, policy[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Replay a captured Graph for this signature, or run ``fn`` eagerly.

        Order: eligibility check → cache hit / replay → cap / disabled →
        capture → replay. Any capture failure increments ``capture_failed``
        and subsequent calls for that key stay eager.
        """
        inspected = self._inspect_call(args, kwargs)
        if inspected is None:
            self.stats["eager"] += 1
            return self._fn(*args, **kwargs)

        key, tensor_leaves, inference_mode = inspected
        graphs = self._graphs
        captured = graphs.get(key)
        if captured is not None:
            if len(graphs) > 1:
                graphs.move_to_end(key)
            return self._replay_leaves(captured, tensor_leaves, inference_mode)
        if not self._capture_on_miss or key in self._disabled_keys or len(graphs) >= self._max_graphs:
            self.stats["eager"] += 1
            return self._fn(*args, **kwargs)

        captured = self._capture(key, args, kwargs)
        if captured is None:
            self.stats["eager"] += 1
            return self._fn(*args, **kwargs)
        return self._replay_leaves(captured, tensor_leaves, inference_mode)

    def _capture(self, key: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _CapturedGraph | None:
        """Warm up and capture; remember unsupported signatures, propagate OOM."""
        from worldfoundry.core.execution.inference_runtime import is_accelerator_out_of_memory

        # Every tensor leaf becomes an explicit replay input, including nested
        # per-layer conditions. Warm these exact buffers: cloning a sliced or
        # expanded view changes its strides and therefore compiler guards.
        canonical_kwargs = {name: kwargs[name] for name in sorted(kwargs)} if kwargs else {}
        leaves, spec = tree_flatten((args, canonical_kwargs))
        input_indices = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        device = leaves[input_indices[0]].device
        static_inputs = [leaves[index].clone() for index in input_indices]
        for index, buffer in zip(input_indices, static_inputs, strict=False):
            leaves[index] = buffer
        call_args, call_kwargs = tree_unflatten(leaves, spec)

        # Warm up on a side stream so lazy allocations / autotune settle and do
        # not get baked into the captured graph.
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            # Autocast's weight cache dies at the enclosing context exit.
            # Capture casts as graph operations instead of retaining pointers
            # to that temporary cache across requests.
            with (
                torch.cuda.stream(side),
                torch.autocast(
                    "cuda",
                    enabled=torch.is_autocast_enabled("cuda"),
                    cache_enabled=False,
                ),
            ):
                for _ in range(max(self._warmup, 1)):
                    self._fn(*call_args, **call_kwargs)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
        except RuntimeError as error:
            if is_accelerator_out_of_memory(error):
                raise
            self.stats["capture_failed"] += 1
            self._disabled_keys.add(key)
            return None

        if self._pool is None:
            self._pool = graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        inference_mode = _is_inference_mode_enabled()
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            # CUDA Graph control updates process-owned generator tensors, which
            # compilation may have created in inference mode. Keep those updates
            # inference-safe while preserving the caller's mode for compute.
            with (
                torch.cuda.device(device),
                torch.inference_mode(),
                torch.cuda.graph(graph, pool=self._pool),
                # Inside the capture, compute runs in the caller's inference mode.
                torch.inference_mode(inference_mode),
                torch.no_grad(),
                torch.autocast("cuda", enabled=torch.is_autocast_enabled("cuda"), cache_enabled=False),
            ):
                static_outputs = self._fn(*call_args, **call_kwargs)
        except RuntimeError as error:
            if is_accelerator_out_of_memory(error):
                raise
            self.stats["capture_failed"] += 1
            self._disabled_keys.add(key)
            return None
        finally:
            if gc_was_enabled:
                gc.enable()

        captured = _CapturedGraph(graph, static_inputs, static_outputs, input_indices)
        self._graphs[key] = captured
        self.stats["capture"] += 1
        return captured

    def _replay_leaves(
        self,
        captured: _CapturedGraph,
        tensor_leaves: Sequence[torch.Tensor],
        already_inference_mode: bool,
    ) -> Any:
        """``copy_`` pre-extracted tensor leaves, replay, then materialize outputs."""
        static_inputs = captured.static_inputs
        if len(static_inputs) == 1:
            buf = static_inputs[0]
            src = tensor_leaves[0]
            if buf is not src:
                buf.copy_(src)
        else:
            for buf, src in zip(static_inputs, tensor_leaves, strict=False):
                if buf is not src:
                    buf.copy_(src)
        if already_inference_mode:
            captured.graph.replay()
        else:
            with torch.inference_mode():
                captured.graph.replay()
        self.stats["replay"] += 1
        outputs = captured.static_outputs
        if not self._clone_outputs:
            return outputs
        if type(outputs) is torch.Tensor:
            return outputs.clone()
        return self._materialize(outputs)

    def _replay(self, captured: _CapturedGraph, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """``copy_`` fresh tensor leaves, replay, then materialize owned outputs."""
        canonical_kwargs = {name: kwargs[name] for name in sorted(kwargs)} if kwargs else {}
        leaves, _ = tree_flatten((args, canonical_kwargs))
        tensor_leaves = [leaves[index] for index in captured.input_indices]
        return self._replay_leaves(captured, tensor_leaves, _is_inference_mode_enabled())

    def _materialize(self, outputs: Any) -> Any:
        """Clone tensor leaves so callers cannot alias static Graph output storage."""
        if not self._clone_outputs:
            return outputs
        if type(outputs) is torch.Tensor:
            return outputs.clone()
        if type(outputs) is tuple:
            for value in outputs:
                if type(value) is not torch.Tensor:
                    return tree_map(lambda v: v.clone() if isinstance(v, torch.Tensor) else v, outputs)
            return tuple(value.clone() for value in outputs)
        return tree_map(lambda value: value.clone() if isinstance(value, torch.Tensor) else value, outputs)

    def report(self) -> dict[str, Any]:
        """Return request-window counters plus separate lifetime diagnostics."""

        window = {
            name: int(value) - int(self._request_window_baseline.get(name, 0)) for name, value in self.stats.items()
        }
        return {
            "enabled": self._enabled,
            "capture_on_miss": self._capture_on_miss,
            "window_id": self._request_window_id,
            "graphs": len(self._graphs),
            "max_graphs": self._max_graphs,
            "disabled_signatures": len(self._disabled_keys),
            **window,
            "lifetime": dict(self.stats),
        }

    def invalidate(self) -> None:
        """Drain and discard captures after weights or execution policy change.

        Request telemetry resets retain graphs. Model replacement, LoRA
        installation and toggling low-precision execution require this explicit
        invalidation: replay cannot reevaluate Python policy or cache decisions.
        Call between requests, with no concurrent calls to this runner.
        """
        devices = {
            tensor.device for captured in self._graphs.values() for tensor in captured.static_inputs if tensor.is_cuda
        }
        for device in devices:
            torch.cuda.synchronize(device)
        self._graphs.clear()
        self._disabled_keys.clear()
        # Only drop the pool handle when this runner owns it privately; when a
        # pool is shared across segments, sibling runners may still hold active
        # captures in that pool.
        if self._owns_pool:
            self._pool = None


__all__ = ["InferenceCUDAGraphRunner"]

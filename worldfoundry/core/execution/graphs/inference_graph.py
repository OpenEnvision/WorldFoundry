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


# ──────────────────────────────────────────────────────────────────────────
# Signature helpers — Graph identity binds tensors + ambient math policy
# ──────────────────────────────────────────────────────────────────────────


def _flatten_tensors(args: Sequence[Any], kwargs: Mapping[str, Any] | None = None) -> list[torch.Tensor]:
    """Collect tensor leaves, including explicit per-layer K/V tuples."""
    leaves, _ = tree_flatten((args, kwargs or {}))
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _ambient_policy_key() -> dict[str, Any]:
    """Math/autocast/grad policy that a captured graph implicitly binds."""

    try:
        autocast_enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch < 2.4
        autocast_enabled = torch.is_autocast_enabled()
    try:
        autocast_dtype = torch.get_autocast_dtype("cuda")
    except AttributeError:  # PyTorch < 2.4
        autocast_dtype = torch.get_autocast_gpu_dtype()
    return {
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "autocast_enabled": bool(autocast_enabled),
        "autocast_dtype": str(autocast_dtype),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


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

    __slots__ = ("graph", "static_inputs", "static_outputs", "input_indices")

    def __init__(
        self,
        graph: "torch.cuda.CUDAGraph",
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
        """
        self._fn = fn
        self._warmup = int(warmup)
        self._max_graphs = int(max_graphs)
        self._clone_outputs = bool(clone_outputs)
        self._enabled = bool(enabled)
        self._extra_key = extra_key
        self._graphs: OrderedDict[str, _CapturedGraph] = OrderedDict()
        self._pool: Any | None = None
        self._disabled_keys: set[str] = set()
        self.stats = {"capture": 0, "replay": 0, "eager": 0, "capture_failed": 0}
        self._request_window_id = 0
        self._request_window_baseline = dict(self.stats)
        self._capture_on_miss = True

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
        if not self._eligible(args, kwargs):
            return False
        key = _signature_key(args, kwargs, self._extra_key)
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
        if any(t.requires_grad for t in tensors) or torch.is_grad_enabled():
            return False
        leaves, _ = tree_flatten((args, kwargs))
        return all(
            isinstance(leaf, (torch.Tensor, bool, int, float, str, torch.dtype, torch.device)) or leaf is None
            for leaf in leaves
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Replay a captured Graph for this signature, or run ``fn`` eagerly.

        Order: eligibility check → cache hit / replay → cap / disabled →
        capture → replay. Any capture failure increments ``capture_failed``
        and subsequent calls for that key stay eager.
        """
        if not self._eligible(args, kwargs):
            self.stats["eager"] += 1
            return self._fn(*args, **kwargs)

        key = _signature_key(args, kwargs, self._extra_key)
        captured = self._graphs.get(key)
        if captured is not None:
            self._graphs.move_to_end(key)
            return self._replay(captured, args, kwargs)
        if not self._capture_on_miss or key in self._disabled_keys or len(self._graphs) >= self._max_graphs:
            self.stats["eager"] += 1
            return self._fn(*args, **kwargs)

        captured = self._capture(key, args, kwargs)
        if captured is None:
            self.stats["eager"] += 1
            return self._fn(*args, **kwargs)
        return self._replay(captured, args, kwargs)

    def _capture(self, key: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _CapturedGraph | None:
        """Warm up and capture; remember unsupported signatures, propagate OOM."""
        from worldfoundry.core.execution.inference_runtime import is_accelerator_out_of_memory

        device = _flatten_tensors(args, kwargs)[0].device
        # Every tensor leaf becomes an explicit replay input, including nested
        # per-layer conditions. Warm these exact buffers: cloning a sliced or
        # expanded view changes its strides and therefore compiler guards.
        leaves, spec = tree_flatten((args, {name: kwargs[name] for name in sorted(kwargs)}))
        input_indices = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        static_inputs = [leaves[index].clone() for index in input_indices]
        for index, buffer in zip(input_indices, static_inputs):
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
        inference_mode = torch.is_inference_mode_enabled()
        try:
            # CUDA Graph control updates process-owned generator tensors, which
            # compilation may have created in inference mode. Keep those updates
            # inference-safe while preserving the caller's mode for compute.
            with torch.cuda.device(device), torch.inference_mode(), torch.cuda.graph(graph, pool=self._pool):
                with (
                    torch.inference_mode(inference_mode),
                    torch.no_grad(),
                    torch.autocast(
                        "cuda",
                        enabled=torch.is_autocast_enabled("cuda"),
                        cache_enabled=False,
                    ),
                ):
                    static_outputs = self._fn(*call_args, **call_kwargs)
        except RuntimeError as error:
            if is_accelerator_out_of_memory(error):
                raise
            self.stats["capture_failed"] += 1
            self._disabled_keys.add(key)
            return None

        captured = _CapturedGraph(graph, static_inputs, static_outputs, input_indices)
        self._graphs[key] = captured
        self.stats["capture"] += 1
        return captured

    def _replay(self, captured: _CapturedGraph, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """``copy_`` fresh tensor leaves, replay, then materialize owned outputs."""
        leaves, _ = tree_flatten((args, {name: kwargs[name] for name in sorted(kwargs)}))
        for buf, index in zip(captured.static_inputs, captured.input_indices):
            buf.copy_(leaves[index])
        with torch.inference_mode():
            captured.graph.replay()
        self.stats["replay"] += 1
        return self._materialize(captured.static_outputs)

    def _materialize(self, outputs: Any) -> Any:
        """Clone tensor leaves so callers cannot alias static Graph output storage."""
        if not self._clone_outputs:
            return outputs
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


__all__ = ["InferenceCUDAGraphRunner"]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Low-level CUDA Graph helpers (pool handle, tensor-tree signatures).

Responsibility
    Flatten tensor pytrees, build a capture cache key, and wrap modules with
    :func:`torch.cuda.graph` for training-oriented / TE graphing.

Boundaries
    This module does not decide *when* to capture — that is
    :class:`~worldfoundry.core.execution.graphs.inference_graph.InferenceCUDAGraphRunner`
    and ``acceleration.cuda_graph_dispatch``. Autograd-aware; inference
    callers that want eager fallback should use the runner, not
    :func:`create_cuda_graph`.

Public surface
    :func:`create_cuda_graph` (``__all__``). Also used internally:
    ``graph_pool_handle``, ``_tensor_tree_signature``, capture flags.
"""

import hashlib
import json
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import torch

try:
    # PyTorch has not yet promoted pytree to a stable public namespace on all
    # supported versions. Guard the compatibility import so this module stays
    # importable and reports an actionable runtime error if it disappears.
    from torch.utils import _pytree as _torch_pytree
except (ImportError, AttributeError):  # pragma: no cover - future torch guard
    _torch_pytree = None


# ──────────────────────────────────────────────────────────────────────────
# Pytree + TE optional imports — capture must fail loudly if flatten is gone
# ──────────────────────────────────────────────────────────────────────────


def _tree_flatten(value):
    """Flatten a tensor pytree; raise if this PyTorch build has no ``_pytree``."""
    if _torch_pytree is None:
        raise RuntimeError("This PyTorch build does not provide pytree helpers required by CUDA Graph capture")
    return _torch_pytree.tree_flatten(value)


def _tree_unflatten(values, spec):
    """Rebuild a pytree from flattened tensors and a capture-time spec."""
    if _torch_pytree is None:
        raise RuntimeError("This PyTorch build does not provide pytree helpers required by CUDA Graph capture")
    return _torch_pytree.tree_unflatten(values, spec)

try:
    from transformer_engine.pytorch.distributed import get_all_rng_states, graph_safe_rng_available
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
except ImportError:

    class TransformerEngineBaseModule(torch.nn.Module):
        """Sentinel used when Transformer Engine is not installed."""

    def graph_safe_rng_available() -> bool:
        """TE-absent stub: Graph cannot register extra RNG states."""
        return False

    def get_all_rng_states() -> dict:
        """TE-absent stub: no extra generator states to snapshot around warmup."""
        return {}


from worldfoundry.core.distributed.runtime.logging import log

__all__ = ["create_cuda_graph"]


_IS_GRAPH_CAPTURING = False

_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]


# ──────────────────────────────────────────────────────────────────────────
# Capture identity — shape-only keys are unsafe; bind stride / dtype / policy
# ──────────────────────────────────────────────────────────────────────────


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Return the execution-relevant identity of a graph input tensor."""
    device = tensor.device
    capability: tuple[int, int] | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability(device))
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(device),
        "layout": str(tensor.layout),
        "requires_grad": tensor.requires_grad,
        "names": None if tensor.names is None else list(tensor.names),
        "cuda_capability": capability,
    }


def _tensor_tree_signature(value: Any) -> dict[str, Any]:
    """Describe both the pytree structure and every tensor leaf."""
    flat, spec = _tree_flatten(value)
    tensors: list[dict[str, Any]] = []
    for leaf in flat:
        if not isinstance(leaf, torch.Tensor):
            raise TypeError(
                "create_cuda_graph only supports pytrees of torch.Tensor leaves; "
                f"got leaf type {type(leaf)}"
            )
        tensors.append(_tensor_signature(leaf))
    return {"tree_spec": str(spec), "tensors": tensors}


def _cuda_graph_cache_key(
    blocks: torch.nn.ModuleList,
    tensor_args: list[Any],
    tensor_kwargs: dict[str, Any],
    extra_key: str | None,
) -> str:
    """Build a stable cache key for one CUDA Graph input contract.

    Shape-only keys are unsafe: tensors with equal shapes can still differ in
    dtype, device, stride, tree grouping, or distributed topology.  Captured
    graphs bind all of those properties, so they are part of the identity.
    """
    distributed = torch.distributed
    distributed_context = {
        "initialized": distributed.is_available() and distributed.is_initialized(),
        "rank": None,
        "world_size": None,
    }
    if distributed_context["initialized"]:
        distributed_context["rank"] = distributed.get_rank()
        distributed_context["world_size"] = distributed.get_world_size()

    try:
        autocast_enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch < 2.4
        autocast_enabled = torch.is_autocast_enabled()
    try:
        autocast_dtype = torch.get_autocast_dtype("cuda")
    except AttributeError:  # PyTorch < 2.4
        autocast_dtype = torch.get_autocast_gpu_dtype()

    payload = {
        "version": 2,
        "args": [_tensor_tree_signature(arg) for arg in tensor_args],
        "kwargs": [
            {"name": name, "value": _tensor_tree_signature(tensor_kwargs[name])}
            for name in sorted(tensor_kwargs)
        ],
        "blocks": [
            {
                "class": f"{block.__class__.__module__}:{block.__class__.__qualname__}",
                "training": block.training,
            }
            for block in blocks
        ],
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "autocast": {"enabled": autocast_enabled, "dtype": str(autocast_dtype)},
        "math_policy": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "distributed": distributed_context,
        "extra_key": extra_key,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"cuda-graph-v2:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


# ──────────────────────────────────────────────────────────────────────────
# Capture flags / pool — process-wide capturing bit for TE modules
# ──────────────────────────────────────────────────────────────────────────


def set_capture_start() -> None:
    """Record beginning of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def set_capture_end() -> None:
    """Record end of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_capturing() -> bool:
    """Return whether within `make_graphed_callables`."""
    return _IS_GRAPH_CAPTURING


def graph_pool_handle():
    """
    Returns an opaque token representing the id of a graph memory pool.
    """
    public_factory = getattr(torch.cuda, "graph_pool_handle", None)
    if callable(public_factory):
        return public_factory()
    private_factory = getattr(torch._C, "_graph_pool_handle", None)
    if callable(private_factory):  # pragma: no cover - legacy torch fallback
        return private_factory()
    raise RuntimeError("This PyTorch build does not expose CUDA Graph memory-pool handles")


# ──────────────────────────────────────────────────────────────────────────
# Graphed callables — warmup then capture; replay copies into static buffers
# ──────────────────────────────────────────────────────────────────────────


def _make_graphed_callables(
    callables: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    pool: Optional[Tuple[int, ...]] = None,
) -> SingleOrTuple[Callable]:
    """
    Helper method for `make_graphed_callables`
    """

    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    # Default is to pass no kwargs to callables
    if sample_kwargs is None:
        if isinstance(callables, tuple):
            sample_kwargs = tuple({} for _ in range(len(sample_args)))
        else:
            sample_kwargs = {}

    # Canonicalize args as tuples
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
        sample_kwargs = (sample_kwargs,)

    # Check sizes of args
    assert len(sample_args) == len(callables)
    assert len(sample_kwargs) == len(callables)

    # Check callables
    for c in callables:
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and len(c._forward_pre_hooks) == 0, (
                "Modules must not have hooks registered at the time they are passed. "
                + "However, registering hooks on modules after passing them "
                + "through make_graphed_callables is allowed."
            )
            assert all(b.requires_grad is False for b in c.buffers()), (
                "In any :class:`~torch.nn.Module` passed to "
                + ":func:`~make_graphed_callables`, only parameters may be trainable. "
                + "All buffers must have ``requires_grad=False``."
            )

    # Flatten callable arguments
    per_callable_kwargs_keys = [list(kwargs.keys()) for kwargs in sample_kwargs]
    flatten_sample_args = []
    for args, kwargs, kwargs_keys in zip(sample_args, sample_kwargs, per_callable_kwargs_keys):
        flatten_arg, _ = _tree_flatten(args)
        flatten_kwarg, _ = _tree_flatten([kwargs[key] for key in kwargs_keys])
        flatten_sample_args.append(tuple(flatten_arg + flatten_kwarg))
        assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), (
            "In the beta API, sample_args "
            + "for each callable must contain only Tensors. Other types are not allowed."
        )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))
    ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    graph_callables = [None for _ in range(len(flatten_sample_args))]

    # For cases with multiple active RNG states, e.g. TP.
    if graph_safe_rng_available():
        for _, state in get_all_rng_states().items():
            for fwd_graph in fwd_graphs:
                fwd_graph.register_generator_state(state)

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()

    # Get warmup func and func_idx.
    warmup_func_idx = []
    warmup_func = []
    for func_idx, func in enumerate(callables):
        warmup_func_idx.append(func_idx)
        warmup_func.append(func)
    assert len(warmup_func) == len(sample_args), f"Warmup runs {len(warmup_func)} don't match args {len(sample_args)}."
    assert len(warmup_func_idx) == len(set(warmup_func_idx)), (
        f"Warmup runs {len(warmup_func)} but only {len(set(warmup_func_idx))} are unique."
    )

    # Filter the TE modules that cudagraph can access.
    visited_te_modules = set()

    def hook_fn(module, inputs, outputs):  # pylint: disable=unused-argument
        """Hook fn.

        Args:
            module: The module.
            inputs: The inputs.
            outputs: The outputs.
        """
        if isinstance(module, TransformerEngineBaseModule):
            visited_te_modules.add(module)

    # Run warmup and do the above filtering.
    with torch.cuda.stream(torch.cuda.Stream()):
        for func_idx, func in zip(warmup_func_idx, warmup_func):
            args = sample_args[func_idx]
            kwargs = sample_kwargs[func_idx]
            for _ in range(num_warmup_iters):
                hooks = []
                for module in func.modules():
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
                outputs, _ = _tree_flatten(func(*args, **kwargs))
                for hook in hooks:
                    hook.remove()
                del outputs
            # The following code is added specifically for MCore's special requirements,
            # aimed at preventing warmup from altering the control flow.
            for module in func.modules():
                if hasattr(module, "is_first_microbatch"):
                    module.is_first_microbatch = True
    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    graph_id = 0
    for func, args, kwargs, fwd_graph in zip(callables, sample_args, sample_kwargs, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args, **kwargs)
        graph_callables[graph_id] = func
        graph_id += 1

        flatten_outputs, spec = _tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    def make_graphed_autograd_function(
        fwd_graph,
        module_params,
        kwargs_keys,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
    ):
        """Make graphed autograd function.

        Args:
            fwd_graph: The fwd graph.
            module_params: The module params.
            kwargs_keys: The kwargs keys.
            len_user_args: The len user args.
            output_unflatten_spec: The output unflatten spec.
            static_input_surface: The static input surface.
            static_outputs: The static outputs.
        """

        class Graphed(torch.autograd.Function):
            """Autograd function for graph replay."""

            @staticmethod
            def forward(ctx, *inputs):
                """Forward.

                Args:
                    ctx: The ctx.
                """
                # pylint: disable=missing-function-docstring

                # Copy values from new tensors into static tensors
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])

                # Replay forward graph
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple(o.detach() for o in static_outputs)

        def functionalized(*user_args, **user_kwargs):
            """Functionalized."""
            # Check that required kwargs are provided
            for key in kwargs_keys:
                if key not in user_kwargs:
                    raise TypeError(
                        f"Graphed callable was initialized with kwarg {key} ,but it was not provided in graph replay"
                    )

            # Runs the autograd function with inputs == all inputs to
            # the graph that might require grad (explicit user args +
            # module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args, _ = _tree_flatten(user_args)
            flatten_user_kwargs, _ = _tree_flatten([user_kwargs[key] for key in kwargs_keys])
            func_args = tuple(flatten_user_args) + tuple(flatten_user_kwargs) + module_params
            out = Graphed.apply(*func_args)
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret = []
    for i in range(len(sample_args)):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            per_callable_module_params[i],
            per_callable_kwargs_keys[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
        )

        func = graph_callables[i]
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                """Make graphed forward.

                Args:
                    func: The func.
                    graph_training_state: The graph training state.
                    graphed: The graphed.
                    orig_fwd: The orig fwd.
                """

                def new_fwd(*user_args, **user_kwargs):
                    """New fwd."""
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        return graphed(*user_args, **user_kwargs)
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            forward = make_graphed_forward(func, func.training, graphed, func.forward)
            ret.append(forward)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)


def make_graphed_callables_forward(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    pool: Optional[Tuple[int, ...]] = None,
) -> Union[Callable, Tuple[Callable, ...]]:
    """
    Make CUDA graph version of Transformer Engine modules
    A variation of PyTorch's `make_graphed_callables` utility function.
    `original PyTorch implementation <https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html>`_
    for more documentation.
    Graphing parameters
    -------------------
    modules: (tuple of) callable
             Callable or callables to graph.
    sample_args: (tuple of) tuple of torch.Tensor
                 Positional arguments to callable(s).
    num_warmup_iters: int, default = 3
                      Number of warmup iterations.
    sample_kwargs: (tuple of) dict, optional
                   Keyword arguments to callable(s)
    pool: (tuple of) int, default = `None`, optional
          An instance returned from function `torch.cuda.graph_pool_handle` that hints
          this graph may share memory with the indicated pool.
    """
    set_capture_start()
    try:
        # Handle single module.
        just_one_callable = False
        if not isinstance(modules, tuple):
            just_one_callable = True
            modules = (modules,)

        forward_funcs = []
        for module in modules:
            assert isinstance(module, torch.nn.Module), f"Graphing for {type(module)} is not supported."
            forward_funcs.append(module)

        if just_one_callable:
            forward_funcs = forward_funcs[0]
        else:
            forward_funcs = tuple(forward_funcs)

        # Save RNG state so graph warmup never changes subsequent sampling.
        graph_safe_rng = graph_safe_rng_available()
        if graph_safe_rng:
            generators = [
                torch.cuda.default_generators[torch.cuda.current_device()],
                *get_all_rng_states().values(),
            ]
            original_rng_states = [state.get_state() for state in generators]
        else:
            generators = []
            original_rng_states = torch.cuda.get_rng_state()

        try:
            return _make_graphed_callables(
                forward_funcs,
                sample_args,
                num_warmup_iters=num_warmup_iters,
                sample_kwargs=sample_kwargs,
                pool=pool,
            )
        finally:
            # Ensures warmup does not affect numerics for ops such as dropout,
            # including when capture itself raises.
            if graph_safe_rng:
                for generator, state in zip(generators, original_rng_states):
                    generator.set_state(state)
            else:
                torch.cuda.set_rng_state(original_rng_states)
    finally:
        set_capture_end()


# ──────────────────────────────────────────────────────────────────────────
# Public factory — dummy trees match stride/dtype so capture does not alias
# ──────────────────────────────────────────────────────────────────────────


def create_cuda_graph(
    cuda_graphs_storage: dict,
    blocks: torch.nn.ModuleList,
    tensor_args: list[Any],
    tensor_kwargs: dict[str, Any],
    extra_key: Optional[str] = None,
) -> str:
    """Create cuda graph.

    Args:
        cuda_graphs_storage: The cuda graphs storage.
        blocks: The blocks.
        tensor_args: The tensor args.
        tensor_kwargs: The tensor kwargs.
        extra_key: The extra key.

    Returns:
        The return value.
    """

    def _make_dummy_tensor_like(t: torch.Tensor) -> torch.Tensor:
        """Helper function to make dummy tensor like.

        Args:
            t: The t.

        Returns:
            The return value.
        """
        if t.layout != torch.strided:
            raise TypeError(f"create_cuda_graph only supports strided tensors; got {t.layout}")
        if any(name is not None for name in t.names):
            raise TypeError("create_cuda_graph does not support named tensors")
        dummy = torch.empty_strided(t.shape, t.stride(), device=t.device, dtype=t.dtype)
        with torch.no_grad():
            if t.dtype.is_floating_point or t.dtype.is_complex:
                try:
                    dummy.normal_()
                except RuntimeError:
                    dummy.zero_()
            elif t.dtype == torch.bool:
                dummy.zero_()
            elif t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                if t.numel() > 0:
                    low = int(t.min().item())
                    maximum = int(t.max().item())
                    if maximum == low:
                        dummy.fill_(low)
                        return dummy.requires_grad_(t.requires_grad)
                    dtype_max = torch.iinfo(t.dtype).max
                    high = maximum + 1 if maximum < dtype_max else maximum
                else:
                    low, high = 0, 1
                dummy.random_(low, high)
            else:
                dummy.zero_()
        return dummy.requires_grad_(t.requires_grad)

    def _make_dummy_tree(x: Any) -> Any:
        """Helper function to make dummy tree.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        flat, spec = _tree_flatten(x)
        dummy_flat: list[torch.Tensor] = []
        for leaf in flat:
            if not isinstance(leaf, torch.Tensor):
                raise TypeError(
                    f"create_cuda_graph only supports pytrees of torch.Tensor leaves; got leaf type {type(leaf)}"
                )
            dummy_flat.append(_make_dummy_tensor_like(leaf))
        return _tree_unflatten(dummy_flat, spec)

    if any(arg is None for arg in tensor_args):
        raise TypeError(
            "create_cuda_graph cannot omit positional arguments containing None; "
            "pass optional tensor inputs by keyword"
        )
    real_args = list(tensor_args)
    real_kwargs = {k: v for k, v in tensor_kwargs.items() if v is not None}

    graph_key = _cuda_graph_cache_key(blocks, real_args, real_kwargs, extra_key)
    if graph_key not in cuda_graphs_storage:
        callables = []
        sample_args = []
        sample_kwargs = []
        for block in blocks:
            callables.append(block)
            args = []
            kwargs = {}
            for arg in real_args:
                args.append(_make_dummy_tree(arg))
            for name, kwarg in real_kwargs.items():
                kwargs[name] = _make_dummy_tree(kwarg)
            sample_args.append(tuple(args))
            sample_kwargs.append(kwargs)

        log.critical(f"Creating CUDA Graph {graph_key}")
        cuda_graphs_storage[graph_key] = make_graphed_callables_forward(
            tuple(callables),
            tuple(sample_args),
            sample_kwargs=tuple(sample_kwargs),
            num_warmup_iters=11,
        )
        log.critical(f"Created CUDA Graph {graph_key}")
    return graph_key

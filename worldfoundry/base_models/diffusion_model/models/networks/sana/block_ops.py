"""SANA block glue using shared kernels without changing checkpoint modules.

The dispatcher retains PyTorch for autograd, unsupported inputs, small eager
workloads, and compiled graphs. Custom normalization and stochastic depth
keep their own forward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.kernels import layer_norm_scale_shift, residual_gate_add, scale_shift
from worldfoundry.core.nn.blocks.layers import DropPath


@dataclass(frozen=True, slots=True)
class SanaBlockFusionPolicy:
    """Per-model eager fusion; small workloads and autograd keep original math."""

    backend: str = "auto"
    min_elements: int = 12 * 1024**2
    fuse_layer_norm: bool = False
    repack_tokens: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "torch", "triton"}:
            raise ValueError("SANA block fusion backend must be auto, torch or triton")
        if isinstance(self.min_elements, bool) or not isinstance(self.min_elements, int):
            raise TypeError("min_elements must be an integer")
        if self.min_elements < 0:
            raise ValueError("min_elements must be non-negative")
        if not isinstance(self.fuse_layer_norm, bool):
            raise TypeError("fuse_layer_norm must be bool")
        if not isinstance(self.repack_tokens, bool):
            raise TypeError("repack_tokens must be bool")

    def eligible(self, value: torch.Tensor) -> bool:
        return not torch.is_grad_enabled() and value.numel() >= self.min_elements


def modulated_norm(
    value: torch.Tensor,
    norm: nn.Module,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    frames: int | None = None,
    policy: SanaBlockFusionPolicy | None = None,
) -> torch.Tensor:
    """Apply ``norm(value) * (1 + scale) + shift`` with eligible fusion.

    Only the standard affine-free last-dimension LayerNorm may be bypassed.
    In particular, FP32LayerNorm subclasses and offload wrappers still run.
    ``frames`` broadcasts ``[B, F, 1, C]`` modulation over flat video tokens.
    Custom norms still receive the original flat input.
    """

    fused = policy is not None and policy.eligible(value)
    if (
        fused
        and policy.fuse_layer_norm
        and type(norm) is nn.LayerNorm
        and not norm.elementwise_affine
        and norm.normalized_shape == (value.shape[-1],)
    ):
        shaped = value if frames is None else value.reshape(value.shape[0], frames, -1, value.shape[-1])
        result = layer_norm_scale_shift(shaped, scale, shift, eps=norm.eps, backend=policy.backend)
    else:
        normalized = norm(value)
        shaped = normalized if frames is None else normalized.reshape(value.shape[0], frames, -1, value.shape[-1])
        result = scale_shift(shaped, scale, shift, backend=policy.backend) if fused else shaped * (1 + scale) + shift
    return result.reshape(value.shape)


def gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    drop_path: nn.Module,
    *,
    frames: int | None = None,
    policy: SanaBlockFusionPolicy | None = None,
) -> torch.Tensor:
    """Apply ``residual + drop_path(gate * update)`` preserving training order."""

    shaped_residual = residual
    shaped_update = update
    if frames is not None:
        shaped_residual = residual.reshape(residual.shape[0], frames, -1, residual.shape[-1])
        shaped_update = update.reshape(shaped_residual.shape)
    if (
        policy is not None
        and policy.eligible(residual)
        and (type(drop_path) is nn.Identity or (type(drop_path) is DropPath and not drop_path.training))
    ):
        return residual_gate_add(shaped_residual, shaped_update, gate, backend=policy.backend).reshape(residual.shape)
    return residual + drop_path((gate * shaped_update).reshape(residual.shape))


def prepare_sana_block_input(blocks: nn.ModuleList, value: torch.Tensor) -> torch.Tensor:
    """Convert token layout only under a separately opted-in numerical budget.

    Patch embedding yields transposed tokens; the native pointwise path keeps
    that stride through every residual. Repacking once allows the existing
    contiguous fusion kernels to execute without a copy at every block. Layout
    changes also change vendor reduction schedules, so defaults retain strides.
    """
    if value.is_cuda and any(
        (policy := getattr(block, "_worldfoundry_block_fusion", None)) is not None
        and policy.repack_tokens
        and policy.backend != "torch"
        and policy.eligible(value)
        for block in blocks
    ):
        return value.contiguous()
    return value


def run_sana_blocks(blocks: nn.ModuleList, value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """One block-stack seam for stateless image/video graphs, request caches, and piecewise CUDA Graphs."""
    cache = kwargs.pop("feature_cache", None)
    step = kwargs.pop("feature_cache_step", 0)
    total_steps = kwargs.pop("feature_cache_total_steps", None)
    piecewise_graph = kwargs.pop("piecewise_graph", None)
    on_stack_execute = kwargs.pop("on_stack_execute", None)
    if piecewise_graph is None:
        piecewise_graph = getattr(blocks, "_worldfoundry_piecewise_graph", None)
    value = prepare_sana_block_input(blocks, value)

    use_piecewise = piecewise_graph is not None and not torch.is_grad_enabled() and value.is_cuda and len(blocks) > 0
    seg_name: str | None = None
    flat_tensors: list[torch.Tensor] = []
    arg_specs: list[tuple[str, object]] = []
    kw_specs: list[tuple[str, str, object]] = []
    if use_piecewise:
        # Piecewise CUDA Graph captures tensor and tuple-of-tensor inputs while
        # keeping non-tensor metadata (such as THW tuple) in the segment key.
        for item in args:
            if isinstance(item, torch.Tensor):
                arg_specs.append(("tensor", len(flat_tensors)))
                flat_tensors.append(item)
            elif isinstance(item, tuple) and item and all(isinstance(elem, torch.Tensor) for elem in item):
                start = len(flat_tensors)
                flat_tensors.extend(item)
                arg_specs.append(("tensor_tuple", (start, len(flat_tensors))))
            else:
                arg_specs.append(("const", item))

        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                kw_specs.append((k, "tensor", len(flat_tensors)))
                flat_tensors.append(v)
            elif isinstance(v, tuple) and v and all(isinstance(elem, torch.Tensor) for elem in v):
                start = len(flat_tensors)
                flat_tensors.extend(v)
                kw_specs.append((k, "tensor_tuple", (start, len(flat_tensors))))
            else:
                kw_specs.append((k, "const", v))

        seg_name = f"sana_blocks:{tuple(arg_specs)}:{tuple(kw_specs)}"

    def _run_stack_eager(hidden: torch.Tensor, *stack_args) -> torch.Tensor:
        for block in blocks:
            hidden = block(hidden, *stack_args, **kwargs)
        return hidden

    def _run_stack(hidden: torch.Tensor) -> torch.Tensor:
        if callable(on_stack_execute):
            on_stack_execute()
        if not use_piecewise or piecewise_graph is None or seg_name is None:
            return _run_stack_eager(hidden, *args)

        def _segment_fn(seg_hidden: torch.Tensor, *seg_tensors: torch.Tensor) -> torch.Tensor:
            rebuilt_args: list[object] = []
            for kind, payload in arg_specs:
                if kind == "tensor":
                    rebuilt_args.append(seg_tensors[int(payload)])  # type: ignore[arg-type]
                elif kind == "tensor_tuple":
                    start_i, end_i = payload  # type: ignore[misc]
                    rebuilt_args.append(tuple(seg_tensors[start_i:end_i]))
                else:
                    rebuilt_args.append(payload)
            rebuilt_kwargs: dict[str, object] = {}
            for k, kind, payload in kw_specs:
                if kind == "tensor":
                    rebuilt_kwargs[k] = seg_tensors[int(payload)]  # type: ignore[arg-type]
                elif kind == "tensor_tuple":
                    start_i, end_i = payload  # type: ignore[misc]
                    rebuilt_kwargs[k] = tuple(seg_tensors[start_i:end_i])
                else:
                    rebuilt_kwargs[k] = payload
            cur = seg_hidden
            for block in blocks:
                cur = block(cur, *rebuilt_args, **rebuilt_kwargs)
            return cur

        piecewise_graph.get_or_register_segment(seg_name, _segment_fn)
        return piecewise_graph.run_segment(seg_name, hidden, *flat_tensors)

    if cache is None:
        return _run_stack(value)

    # Support both fast whole-stack cache controllers (run_stack) and standard
    # EasyCache (run_blocks), recording piecewise skips cleanly on cache hits.
    run_stack_hook = getattr(cache, "run_stack", None)
    if callable(run_stack_hook):
        output, hit = run_stack_hook(
            step,
            value,
            _run_stack,
            block_count=len(blocks),
            total_steps=total_steps,
        )
        if hit and use_piecewise and piecewise_graph is not None and seg_name is not None:
            piecewise_graph.record_skip(seg_name)
        return output

    if use_piecewise and piecewise_graph is not None and seg_name is not None:
        # Wrap EasyCache so a dense miss still replays the captured piecewise graph.
        # EasyCache calls run_block(0..N-1) sequentially on a miss; intercept index 0
        # to execute the captured whole stack and short-circuit indices 1..N-1.
        stack_ran = False

        def _run_block_piecewise(index: int, hidden: torch.Tensor) -> torch.Tensor:
            nonlocal stack_ran
            if index == 0:
                stack_ran = True
                return _run_stack(hidden)
            return hidden

        output = cache.run_blocks(
            step,
            value,
            _run_block_piecewise,
            block_count=len(blocks),
            total_steps=total_steps,
        )
        if not stack_ran:
            piecewise_graph.record_skip(seg_name)
        return output

    def _run_block_direct(index: int, hidden: torch.Tensor) -> torch.Tensor:
        if index == 0 and callable(on_stack_execute):
            on_stack_execute()
        return blocks[index](hidden, *args, **kwargs)

    return cache.run_blocks(
        step,
        value,
        _run_block_direct,
        block_count=len(blocks),
        total_steps=total_steps,
    )

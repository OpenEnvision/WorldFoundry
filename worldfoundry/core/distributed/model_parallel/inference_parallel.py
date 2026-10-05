"""Instance-owned tensor/context groups and inference projection shards.

Callers initialize and supervise the world process group; this module creates only
its own axes. No global model-parallel state or single-rank behavior changes.
All participating ranks must construct, call, and close a context together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as functional


def validate_parallel_degrees(tensor_parallel: int, context_parallel: int) -> tuple[int, int]:
    for name, value in (("tensor_parallel", tensor_parallel), ("context_parallel", context_parallel)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    return tensor_parallel, context_parallel


def balanced_ranges(total: int, parts: int) -> tuple[tuple[int, int], ...]:
    """Contiguous, non-empty ranges with the remainder on lower ranks."""

    if not 1 <= parts <= total:
        raise ValueError(f"parts must be between 1 and the {total} available tokens/features")
    base, extra = divmod(total, parts)
    sizes = [base + (index < extra) for index in range(parts)]
    start = 0
    ranges = []
    for size in sizes:
        ranges.append((start, start + size))
        start += size
    return tuple(ranges)


@dataclass(slots=True)
class InferenceParallelContext:
    """A rectangular TP × CP mesh, independent of training/global meshes."""

    tp_size: int = 1
    cp_size: int = 1
    tp_rank: int = 0
    cp_rank: int = 0
    tp_group: object = None
    cp_group: object = None
    control_group: object = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _owned_groups: tuple = field(default=(), repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def world_size(self) -> int:
        return self.tp_size * self.cp_size

    @property
    def is_main(self) -> bool:
        return self.tp_rank == self.cp_rank == 0

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("inference parallel context is closed")
        if self.world_size > 1 and not dist.is_initialized():
            raise RuntimeError("inference parallel context requires its live process group")
        if not (0 <= self.tp_rank < self.tp_size and 0 <= self.cp_rank < self.cp_size):
            raise ValueError("inference parallel coordinates are outside the mesh")
        if self.world_size > 1 and self.control_group is None:
            raise RuntimeError("multi-rank context is missing its control group")
        if self.tp_size > 1 and self.tp_group is None:
            raise RuntimeError("tensor-parallel context is missing its data group")
        if self.cp_size > 1 and self.cp_group is None:
            raise RuntimeError("context-parallel context is missing its data group")

    @torch.inference_mode(False)
    @torch.no_grad()
    def agree(self, signature: object) -> None:
        """Reject inconsistent request/config metadata before tensor collectives."""

        self._check()
        if self.world_size == 1:
            return
        received = [None] * self.world_size
        dist.all_gather_object(received, signature, group=self.control_group)
        if any(value != received[0] for value in received[1:]):
            raise ValueError("inference parallel ranks disagree on the request or model configuration")

    def local_range(self, total: int) -> tuple[int, int]:
        self._check()
        return balanced_ranges(total, self.cp_size)[self.cp_rank]

    @torch.inference_mode(False)
    @torch.no_grad()
    def broadcast_seed(self, seed: int) -> int:
        """Use one root-generated random seed for a supervised multi-rank request."""

        import secrets

        self._check()
        values = [secrets.randbits(63) if self.is_main and seed < 0 else seed]
        if self.world_size > 1:
            dist.broadcast_object_list(values, src=0, group=self.control_group)
        return values[0]

    @torch.inference_mode(False)
    @torch.no_grad()
    def gather_tokens(self, value: torch.Tensor, *, total: int, dim: int = 1) -> torch.Tensor:
        """Gather unequal CP token shards in original order, without padding leakage."""

        self._check()
        if self.cp_size == 1:
            return value
        ranges = balanced_ranges(total, self.cp_size)
        lengths = [end - start for start, end in ranges]
        if value.shape[dim] != lengths[self.cp_rank]:
            raise ValueError("local token shape does not match its context-parallel range")
        moved = value.movedim(dim, 0).contiguous()
        maximum = max(lengths)
        if moved.shape[0] < maximum:
            moved = torch.cat((moved, moved.new_zeros((maximum - moved.shape[0], *moved.shape[1:]))))
        gathered = [torch.empty_like(moved) for _ in lengths]
        dist.all_gather(gathered, moved, group=self.cp_group)
        return torch.cat([part[:length] for part, length in zip(gathered, lengths)]).movedim(0, dim).contiguous()

    @torch.inference_mode(False)
    @torch.no_grad()
    def sum_tensor_shards(self, value: torch.Tensor) -> torch.Tensor:
        self._check()
        if self.tp_size > 1:
            # Collective workers write outside the caller's inference scope.
            if value.is_inference():
                value = value.clone()
            dist.all_reduce(value, group=self.tp_group)
        return value

    def close(self) -> None:
        """Release only owned subgroups after all inference work has finished."""

        if self._closed:
            return
        self._closed = True
        error = None
        if dist.is_initialized():
            for group in reversed(self._owned_groups):
                try:
                    dist.destroy_process_group(group)
                except Exception as caught:
                    error = error or caught
        if error is not None:
            raise error


def build_inference_parallel_context(
    *, tensor_parallel: int, context_parallel: int, device: str | torch.device,
) -> InferenceParallelContext:
    """Reuse a caller-owned world; never initialize or destroy it implicitly."""

    tp_size, cp_size = validate_parallel_degrees(tensor_parallel, context_parallel)
    device = torch.device(device)
    if tp_size * cp_size == 1:
        return InferenceParallelContext(device=device)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("multi-rank inference requires an initialized torch.distributed world")
    if dist.get_world_size() != tp_size * cp_size:
        raise ValueError("tensor_parallel × context_parallel must equal the initialized world size")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("inference parallelism supports CPU/Gloo or CUDA/NCCL")
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("multi-rank CUDA inference requires an explicit per-rank device index")
        torch.cuda.set_device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"
    rank = dist.get_rank()
    owned = []
    timeout = timedelta(seconds=60)

    def group_for(ranks, group_backend):
        group = dist.new_group(ranks=ranks, backend=group_backend, timeout=timeout)
        if rank in ranks:
            owned.append(group)
            return group
        return None

    tp_group = cp_group = None
    try:
        # Agree on axes before creating differently shaped subgroup lists.
        # The control plane always contains the entire caller-owned world.
        control = group_for(list(range(tp_size * cp_size)), "gloo")
        configurations = [None] * (tp_size * cp_size)
        dist.all_gather_object(configurations, (tp_size, cp_size, device.type), group=control)
        if any(value != configurations[0] for value in configurations[1:]):
            raise ValueError("inference parallel ranks disagree on model configuration")
        if tp_size > 1:
            for cp_index in range(cp_size):
                ranks = list(range(cp_index * tp_size, (cp_index + 1) * tp_size))
                group = group_for(ranks, backend)
                if rank in ranks:
                    tp_group = group
        if cp_size > 1:
            for tp_index in range(tp_size):
                ranks = list(range(tp_index, tp_size * cp_size, tp_size))
                group = group_for(ranks, backend)
                if rank in ranks:
                    cp_group = group
        context = InferenceParallelContext(
            tp_size=tp_size, cp_size=cp_size, tp_rank=rank % tp_size, cp_rank=rank // tp_size,
            tp_group=tp_group, cp_group=cp_group, control_group=control, device=device,
            _owned_groups=tuple(owned),
        )
        return context
    except BaseException:
        for group in reversed(owned):
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        raise


def validate_inference_parallel_policy(policy) -> None:
    """Reject combinations that have no distributed inference validation."""

    from worldfoundry.core.model_loading.policy import OffloadMode, QuantizationMode

    if policy.offload.mode is not OffloadMode.NONE:
        raise ValueError("multi-rank Cosmos inference requires offload_mode=none")
    if policy.dtype is not torch.float32:
        raise ValueError("multi-rank Cosmos inference currently requires torch_dtype=float32 for numerical stability")
    if torch.is_autocast_enabled(policy.device.type):
        raise ValueError("multi-rank Cosmos inference requires autocast to be disabled")
    if policy.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("multi-rank Cosmos inference requires CUDA matmul TF32 to be disabled")
    if policy.quantization.mode is not QuantizationMode.NONE or policy.compile or policy.options:
        raise ValueError("multi-rank Cosmos inference requires dense, uncompiled weights without runtime options")


class ColumnParallelLinear(torch.nn.Module):
    """A checkpoint-loaded output feature slice; no constructor RNG consumption."""

    def __init__(self, linear: torch.nn.Linear, *, start: int, end: int) -> None:
        super().__init__()
        if not 0 <= start < end <= linear.out_features:
            raise ValueError("column projection range is outside its output features")
        self.in_features = linear.in_features
        self.out_features = end - start
        self.weight = torch.nn.Parameter(linear.weight[start:end].detach().clone(), requires_grad=False)
        self.bias = (
            torch.nn.Parameter(linear.bias[start:end].detach().clone(), requires_grad=False)
            if linear.bias is not None else None
        )
        self.train(linear.training)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.linear(value, self.weight, self.bias)


class RowParallelLinear(torch.nn.Module):
    """Accumulate low-precision projection shards in FP32 before one final cast.

    Casting each partial dot product to BF16/FP16 before reduction discards
    contributions that survive in a full projection, especially cancellation
    across feature shards. Bias is added once after the FP32 reduction.
    """

    def __init__(
        self, linear: torch.nn.Linear, *, start: int, end: int, parallel: InferenceParallelContext,
    ) -> None:
        super().__init__()
        if not 0 <= start < end <= linear.in_features:
            raise ValueError("row projection range is outside its input features")
        self.in_features = end - start
        self.out_features = linear.out_features
        self.weight = torch.nn.Parameter(linear.weight[:, start:end].detach().clone(), requires_grad=False)
        self.bias = (
            torch.nn.Parameter(linear.bias.detach().clone(), requires_grad=False) if linear.bias is not None else None
        )
        self.parallel = parallel
        self.train(linear.training)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and value.requires_grad:
            raise RuntimeError("row-parallel inference projections do not support autograd")
        compute_dtype = torch.float32 if value.dtype in {torch.float16, torch.bfloat16} else value.dtype
        output = self.parallel.sum_tensor_shards(
            functional.linear(value.to(compute_dtype), self.weight.to(compute_dtype)),
        )
        if self.bias is not None:
            output = output + self.bias.to(compute_dtype)
        return output.to(value.dtype)


__all__ = [
    "ColumnParallelLinear", "InferenceParallelContext", "RowParallelLinear", "balanced_ranges",
    "build_inference_parallel_context", "validate_parallel_degrees",
    "validate_inference_parallel_policy",
]

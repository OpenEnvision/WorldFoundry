"""Sequence-parallel runtime: SP groups, 4D all-to-all, and xFuser shims.

Initializes the SP world used by Wan long-context attention. 4D all-to-all
swaps head and sequence dims for Ulysses. Optional ``xFuserLongContextAttention``
is imported when xfuser is present; otherwise in-tree attention modules
use :func:`all_to_all_4D` directly.

This is SP, not FSDP — parameters stay replicated (or TP-sharded) on every
SP rank.
"""

import datetime
import os
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F

from worldfoundry.core.distributed.collectives.device_mesh import all_to_all_tensor

# ──────────────────────────────────────────────────────────────────────────
# Process-local SP snapshot — not the vLLM GroupCoordinator source of truth
# ──────────────────────────────────────────────────────────────────────────


class SequenceParallelInfo:
    """Mutable bag of the native Ulysses group this process belongs to."""

    def __init__(self):
        """Start ungrouped: ``sp_size=1`` until :func:`initialize_sequence_parallel_group`."""

        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0


nccl_info = SequenceParallelInfo()
_SEQUENCE_PARALLEL_STATE = False
_SEQUENCE_PARALLEL_GROUPS: dict[str, dist.ProcessGroup] = {}
_CFG_PARALLEL_GROUPS: dict[str, dist.ProcessGroup] = {}
_COLLECTIVE_SHAPE_CACHE: "OrderedDict[tuple[object, ...], tuple[tuple[int, ...], ...]]" = OrderedDict()
_COLLECTIVE_SHAPE_CACHE_LOCK = threading.Lock()
_COLLECTIVE_SHAPE_CACHE_LIMIT = 128
_COLLECTIVE_COUNTERS_LOCK = threading.Lock()
_COLLECTIVE_COUNTERS: dict[str, int] = {
    "all_to_all_calls": 0,
    "all_to_all_input_bytes": 0,
    "fused_multi_tensor_all_to_all_calls": 0,
    "unfused_multi_tensor_all_to_all_calls": 0,
    "shape_metadata_all_gather_calls": 0,
    "shape_metadata_cache_hits": 0,
    "sequence_output_all_gather_calls": 0,
}


def _increment_collective_counter(name: str, value: int = 1) -> None:
    """Add ``value`` to a process-local counter under the shared lock."""

    with _COLLECTIVE_COUNTERS_LOCK:
        _COLLECTIVE_COUNTERS[name] += int(value)


def reset_sequence_parallel_collective_counters() -> None:
    """Reset process-local SP counters before a measured inference region."""

    with _COLLECTIVE_COUNTERS_LOCK:
        for name in _COLLECTIVE_COUNTERS:
            _COLLECTIVE_COUNTERS[name] = 0


def get_sequence_parallel_collective_counters() -> dict[str, int]:
    """Return a stable process-local snapshot of logical SP collectives."""

    with _COLLECTIVE_COUNTERS_LOCK:
        return dict(_COLLECTIVE_COUNTERS)


def _resolve_group(group: Optional[dist.ProcessGroup] = None):
    """Use the caller group, or fall back to the native SP group."""

    return group if group is not None else get_sequence_parallel_group()


def _rank_in_group(group: Optional[dist.ProcessGroup] = None):
    """Group-local rank; WORLD rank when ``group`` is ``None``."""

    if group is None:
        return dist.get_rank()
    return dist.get_group_rank(group, dist.get_rank())


def _collective_shape_cache_enabled() -> bool:
    """Opt-in only: cached shapes are unsafe when ranks change resolution independently."""

    # Dynamic video resolutions can leave one rank's local shape unchanged
    # while another rank changes. Cache only when the deployment explicitly
    # promises a fixed-shape workload.
    return os.getenv("WORLDFOUNDRY_CACHE_COLLECTIVE_SHAPES", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _gather_shapes_cached(
    local_shape: torch.Size | tuple[int, ...],
    group: Optional[dist.ProcessGroup],
    device: torch.device,
) -> tuple[tuple[int, ...], ...]:
    """Gather shape metadata on-device, optionally caching fixed workloads."""

    shape = tuple(int(value) for value in local_shape)
    world_size = dist.get_world_size(group)
    key = (id(group), world_size, _rank_in_group(group), shape)
    if _collective_shape_cache_enabled():
        with _COLLECTIVE_SHAPE_CACHE_LOCK:
            cached = _COLLECTIVE_SHAPE_CACHE.get(key)
            if cached is not None:
                _COLLECTIVE_SHAPE_CACHE.move_to_end(key)
                _increment_collective_counter("shape_metadata_cache_hits")
                return cached

    _increment_collective_counter("shape_metadata_all_gather_calls")
    local = torch.tensor(shape, dtype=torch.int64, device=device)
    if hasattr(dist, "all_gather_into_tensor"):
        gathered_tensor = local.new_empty(world_size * local.numel())
        dist.all_gather_into_tensor(gathered_tensor, local, group=group)
        gathered_tensor = gathered_tensor.reshape(world_size, local.numel())
    else:
        gathered_list = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered_list, local, group=group)
        gathered_tensor = torch.stack(gathered_list, dim=0)
    result = tuple(tuple(int(value) for value in item) for item in gathered_tensor.cpu().tolist())
    if _collective_shape_cache_enabled():
        with _COLLECTIVE_SHAPE_CACHE_LOCK:
            _COLLECTIVE_SHAPE_CACHE[key] = result
            _COLLECTIVE_SHAPE_CACHE.move_to_end(key)
            while len(_COLLECTIVE_SHAPE_CACHE) > _COLLECTIVE_SHAPE_CACHE_LIMIT:
                _COLLECTIVE_SHAPE_CACHE.popitem(last=False)
    return result


def clear_collective_shape_cache() -> None:
    """Drop the LRU of gathered shapes after a group rebuild or resolution change."""

    with _COLLECTIVE_SHAPE_CACHE_LOCK:
        _COLLECTIVE_SHAPE_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────────
# Native SP / CFG groups — contiguous rank blocks; world_size must divide
# ──────────────────────────────────────────────────────────────────────────


def initialize_sequence_parallel_group(sp_size: int) -> None:
    """Create contiguous SP subgroups of size ``sp_size`` and bind this rank's group.

    ``world_size`` must divide by ``sp_size``. The shape cache is cleared so a
    previous group's metadata cannot be reused on a new communicator.
    """

    clear_collective_shape_cache()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, "world_size must be divisible by sequence_parallel_size"

    nccl_info.sp_size = sp_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups = world_size // sp_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sp_size, (i + 1) * sp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            set_sequence_parallel_group(group)
            nccl_info.rank_within_group = rank - i * sp_size
            nccl_info.group_id = i


def set_sequence_parallel_group(group: dist.ProcessGroup) -> None:
    """Install an already-created SP group and refresh ``nccl_info``."""

    clear_collective_shape_cache()
    _SEQUENCE_PARALLEL_GROUPS["sequence"] = group
    nccl_info.group = group
    nccl_info.sp_size = dist.get_world_size(group)
    nccl_info.rank_within_group = dist.get_rank(group)


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Return the native SP group, or ``None`` when SP is not initialized."""

    return _SEQUENCE_PARALLEL_GROUPS.get("sequence", nccl_info.group)


def initialize_cfg_parallel_group(sp_size: int, cfg_size: int) -> None:
    """Create groups joining equal SP shards across CFG replicas."""

    if cfg_size < 1:
        raise ValueError("cfg parallel size must be positive")
    world_size = dist.get_world_size()
    if world_size != int(sp_size) * int(cfg_size):
        raise ValueError(
            f"world size {world_size} must equal sp_size*cfg_size="
            f"{int(sp_size) * int(cfg_size)}"
        )
    rank = dist.get_rank()
    for sequence_rank in range(int(sp_size)):
        ranks = [cfg_rank * int(sp_size) + sequence_rank for cfg_rank in range(int(cfg_size))]
        group = dist.new_group(ranks)
        if rank in ranks:
            _CFG_PARALLEL_GROUPS["cfg"] = group


def get_cfg_parallel_group() -> Optional[dist.ProcessGroup]:
    """Return the CFG replica group, or ``None`` when CFG parallel is off."""

    return _CFG_PARALLEL_GROUPS.get("cfg")


def get_cfg_parallel_rank() -> int:
    """CFG-group rank, or 0 when the group is missing / dist is down."""

    group = get_cfg_parallel_group()
    return int(dist.get_rank(group)) if dist.is_initialized() and group is not None else 0


def get_cfg_parallel_world_size() -> int:
    """CFG-group size, or 1 when the group is missing / dist is down."""

    group = get_cfg_parallel_group()
    return int(dist.get_world_size(group)) if dist.is_initialized() and group is not None else 1


def initialize_sequence_parallel_state(sequence_parallel_size: int):
    """Enable native SP when ``sequence_parallel_size > 1``; otherwise stay local."""

    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        _SEQUENCE_PARALLEL_STATE = False
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def get_sequence_parallel_state():
    """True after :func:`initialize_sequence_parallel_state` enabled SP."""

    return _SEQUENCE_PARALLEL_STATE


def initialize_distributed(seed):
    """NCCL env:// init (or single-GPU skip) plus SP groups of size ``WORLD_SIZE``.

    Timeout is ``2**31-1`` seconds so a slow rank-0 download cannot abort the
    handshake. CUDA is required when ``world_size > 1``.
    """

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    # Set defaults for distributed env vars required by env:// init method
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if world_size == 1 and not dist.is_initialized():
        # Single-GPU mode: skip full init, use default device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL sequence parallelism requires CUDA, but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=2**31 - 1),
            world_size=world_size,
            rank=rank,
            device_id=torch.device("cuda", local_rank),
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initialize_sequence_parallel_state(world_size)


def broadcast(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None):
    """Broadcast from group-local rank 0; return a contiguous view."""

    group = _resolve_group(group)
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast(input_, src=src, group=group)
    return input_.contiguous()


# ──────────────────────────────────────────────────────────────────────────
# Uneven 4D hop — last rank pads a short sequence; assume_even skips the probe
# ──────────────────────────────────────────────────────────────────────────


def _all_to_all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    *,
    assume_even: bool = False,
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    group = _resolve_group(group)
    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # Wan pads the token sequence before sharding, so its Ulysses path can
        # skip one metadata all-gather per transformer block. Keep the dynamic
        # probe as the safe default for other callers that permit uneven shards.
        seq_lens = (
            (int(input.shape[1]),) * seq_world_size
            if assume_even
            else tuple(
                shape[0]
                for shape in _gather_shapes_cached(
                    (input.shape[1],), group, input.device
                )
            )
        )
        # uneven
        if seq_lens[-1] != seq_lens[0]:
            assert seq_lens[0] > seq_lens[-1]
            gap = seq_lens[0] - seq_lens[-1]
            if _rank_in_group(group) == seq_world_size - 1:
                assert input.shape[1] == seq_lens[-1]
                input = F.pad(input, (0, 0, 0, 0, 0, gap))
        else:
            gap = 0

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert hc % seq_world_size == 0, f"Invalid size: {hc}, which should be divisible by {seq_world_size}"
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            _increment_collective_counter("all_to_all_calls")
            _increment_collective_counter(
                "all_to_all_input_bytes",
                input_t.numel() * input_t.element_size(),
            )
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if gap > 0:
            output = output[:, :-gap]

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        if seqlen % seq_world_size != 0:
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
            bs, seqlen, shard_hc, hs = input.shape
        else:
            gap = 0

        assert seqlen % seq_world_size == 0

        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        # (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            _increment_collective_counter("all_to_all_calls")
            _increment_collective_counter(
                "all_to_all_input_bytes",
                input_t.numel() * input_t.element_size(),
            )
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        if gap > 0 and _rank_in_group(group) == seq_world_size - 1:
            output = output[:, :-gap]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


# ──────────────────────────────────────────────────────────────────────────
# 4D Ulysses all-to-all — swap sequence (dim 1) and head (dim 2) shards
# ──────────────────────────────────────────────────────────────────────────


class SeqAllToAll4D(torch.autograd.Function):
    """Autograd wrapper around :func:`_all_to_all_4D`.

    Backward swaps scatter/gather indices so the inverse hop restores the
    original layout. ``assume_even`` is reused so training matches the
    padded-sequence shortcut used in forward.
    """

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        assume_even: bool,
    ) -> torch.Tensor:
        """Exchange 4D Q/K/V shards; cache dims for the inverse backward hop."""

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.assume_even = assume_even

        return _all_to_all_4D(
            input,
            scatter_idx,
            gather_idx,
            group=group,
            assume_even=assume_even,
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None, None, None]:
        """Invert the 4D exchange by swapping scatter and gather indices."""

        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.assume_even,
            ),
            None,
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    *,
    assume_even: bool = False,
):
    """Public 4D Ulysses hop; ``assume_even`` skips a metadata all-gather when safe."""

    group = _resolve_group(group)
    return SeqAllToAll4D.apply(
        group,
        input_,
        scatter_dim,
        gather_dim,
        assume_even,
    )


def all_to_all_4d_many(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    *,
    assume_even: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Fuse equal-shaped 4D Q/K/V exchanges along their batch dimension."""

    values = tuple(tensors)
    if len(values) < 2:
        return values
    first = values[0]
    compatible = all(
        value.ndim == 4
        and value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        for value in values[1:]
    )
    try:
        max_bytes = max(
            int(float(os.getenv("WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB", "512") or "512") * 1024**2),
            0,
        )
    except ValueError:
        max_bytes = 512 * 1024**2
    total_bytes = sum(value.numel() * value.element_size() for value in values)
    if not compatible or max_bytes == 0 or total_bytes > max_bytes:
        _increment_collective_counter("unfused_multi_tensor_all_to_all_calls")
        return tuple(
            all_to_all_4D(
                value,
                group,
                scatter_dim,
                gather_dim,
                assume_even=assume_even,
            )
            for value in values
        )
    _increment_collective_counter("fused_multi_tensor_all_to_all_calls")
    packed = torch.cat(values, dim=0)
    exchanged = all_to_all_4D(
        packed,
        group,
        scatter_dim,
        gather_dim,
        assume_even=assume_even,
    )
    return exchanged.split(first.shape[0], dim=0)


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        """Scatter along ``scatter_dim`` and concatenate along ``gather_dim``."""

        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = all_to_all_tensor(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Invert the hop by swapping scatter and gather dimensions."""

        grad_output = all_to_all_tensor(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    """Autograd all-to-all on the native SP group (or ``group``)."""

    group = _resolve_group(group)
    return _AllToAll.apply(input_, group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim, group):
        """All-gather possibly uneven shards along ``dim`` using cached shapes."""

        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)

        sizes = _gather_shapes_cached(input_.shape, group, input_.device)

        ctx.gathered_dim_sizes = tuple(shape[dim] for shape in sizes)

        tensor_list = [torch.empty(sizes[i], dtype=input_.dtype, device=input_.device) for i in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Keep this rank's slice of the concatenated gradient."""

        group = ctx.group
        rank = _rank_in_group(group)
        dim = ctx.dim

        grad_input_list = torch.split(grad_output, ctx.gathered_dim_sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, _resolve_group(group))


def _split(input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Keep this rank's even slice along ``dim``; ``dim`` must divide by world size."""

    group = _resolve_group(group)
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size})"
    )
    output_list = torch.split(input_, dim_size // world_size, dim=dim)
    return output_list[rank].contiguous()


def _gather(input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """All-gather equal tensors and concatenate along ``dim``."""

    group = _resolve_group(group)
    world_size = dist.get_world_size(group)
    input_ = input_.contiguous()
    output_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, input_, group=group)
    return torch.cat(output_list, dim=dim).contiguous()


class _SplitForwardGatherBackward(torch.autograd.Function):
    """Keep a local shard in forward; all-gather grads so the full tensor trains."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]):
        """Split ``input_`` along ``dim`` to this rank's shard."""

        ctx.dim = dim
        ctx.group = group
        return _split(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """All-gather local grads to reconstruct the full-tensor gradient."""

        return _gather(grad_output, ctx.dim, ctx.group), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Assemble the full tensor in forward; split grads back to this shard."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]):
        """All-gather shards along ``dim`` for decode / loss."""

        ctx.dim = dim
        ctx.group = group
        return _gather(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Keep this rank's slice of the full-tensor gradient."""

        return _split(grad_output, ctx.dim, ctx.group), None, None


def split_forward_gather_backward(
    input_: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Public split-forward / gather-backward used by Ulysses residual paths."""

    return _SplitForwardGatherBackward.apply(input_, dim, _resolve_group(group))


def gather_forward_split_backward(
    input_: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Public gather-forward / split-backward used by decode / loss."""

    return _GatherForwardSplitBackward.apply(input_, dim, _resolve_group(group))


# ──────────────────────────────────────────────────────────────────────────
# Optional xFuser shims — native collectives above remain the WorldFoundry path
# ──────────────────────────────────────────────────────────────────────────

# Optional xFuser compatibility used by official Wan-family runtimes.  The
# native sequence-parallel collectives above remain the canonical
# implementation for WorldFoundry models.
try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("paifuser") is not None:
        from paifuser.xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
            get_world_group,
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from paifuser.xfuser.core.long_ctx_attention import xFuserLongContextAttention
    else:
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
            get_world_group,
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except Exception:
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    xFuserLongContextAttention = None
    get_sp_group = None
    get_world_group = None
    init_distributed_environment = None
    initialize_model_parallel = None
    model_parallel_is_initialized = None


# ──────────────────────────────────────────────────────────────────────────
# Runtime bring-up — xFuser ring vs native Ulysses; refuse WORLD_SIZE mismatch
# ──────────────────────────────────────────────────────────────────────────


def set_multi_gpus_devices(
    ulysses_degree: int,
    ring_degree: int,
    classifier_free_guidance_degree: int = 1,
) -> torch.device:
    """Initialize an xFuser or native Ulysses mesh and return the local device."""

    if ulysses_degree > 1 or ring_degree > 1 or classifier_free_guidance_degree > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU sequence parallelism requires CUDA")
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
        if not 0 <= local_rank < torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the {torch.cuda.device_count()} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                device_id=torch.device("cuda", local_rank),
            )
        expected_world_size = (
            int(ring_degree)
            * int(ulysses_degree)
            * int(classifier_free_guidance_degree)
        )
        if dist.get_world_size() != expected_world_size:
            raise ValueError(
                f"world size {dist.get_world_size()} does not match requested "
                f"parallel degree {expected_world_size}"
            )
        if ring_degree != 1:
            if get_sp_group is None:
                raise RuntimeError("ring parallelism requires xFuser")
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=ring_degree * ulysses_degree,
                classifier_free_guidance_degree=classifier_free_guidance_degree,
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree,
            )
            return torch.device(f"cuda:{get_world_group().local_rank}")
        initialize_sequence_parallel_group(int(ulysses_degree))
        initialize_cfg_parallel_group(
            int(ulysses_degree),
            int(classifier_free_guidance_degree),
        )
        return torch.device("cuda", local_rank)
    return torch.device("cuda", torch.cuda.current_device())


def ensure_sequence_parallel_runtime(sp_degree: int) -> torch.device:
    """Initialize one idempotent Ulysses runtime from a torchrun environment."""

    degree = int(sp_degree)
    if degree < 2:
        raise ValueError("sequence_parallel must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("sequence_parallel requires CUDA/NCCL")

    xfuser_active = (
        get_sequence_parallel_world_size is not None
        and model_parallel_is_initialized is not None
        and model_parallel_is_initialized()
    )
    if xfuser_active:
        actual = int(get_sequence_parallel_world_size())
        if actual != degree:
            raise ValueError(
                f"sequence_parallel={degree} but active xFuser group has size {actual}"
            )
        local_rank = int(os.getenv("LOCAL_RANK", dist.get_rank()))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    native_group = get_sequence_parallel_group()
    if dist.is_initialized() and native_group is not None:
        actual = int(dist.get_world_size(native_group))
        if actual != degree:
            raise ValueError(
                f"sequence_parallel={degree} but active native group has size {actual}"
            )
        local_rank = int(os.getenv("LOCAL_RANK", dist.get_rank()))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    declared_world_size = int(os.getenv("WORLD_SIZE", "1"))
    if declared_world_size != degree:
        raise RuntimeError(
            f"sequence_parallel={degree} requires torchrun --nproc_per_node={degree}; "
            f"WORLD_SIZE={declared_world_size}"
        )
    return set_multi_gpus_devices(degree, 1)


def ensure_parallel_runtime(sp_degree: int, cfg_degree: int) -> torch.device:
    """Initialize the native SP×CFG mesh from a torchrun environment."""

    sp = int(sp_degree)
    cfg = int(cfg_degree)
    if sp < 1 or cfg < 1:
        raise ValueError("parallel degrees must be positive")
    if sp == 1 and cfg == 1:
        if not torch.cuda.is_available():
            raise RuntimeError("parallel runtime requires CUDA")
        return torch.device("cuda", torch.cuda.current_device())
    declared_world_size = int(os.getenv("WORLD_SIZE", "1"))
    expected = sp * cfg
    if declared_world_size != expected:
        raise RuntimeError(
            f"sequence_parallel={sp}, cfg_parallel={cfg} requires torchrun "
            f"--nproc_per_node={expected}; WORLD_SIZE={declared_world_size}"
        )
    if dist.is_initialized():
        active_sp = get_sequence_parallel_group()
        active_cfg = get_cfg_parallel_group()
        if active_sp is not None and active_cfg is not None:
            if dist.get_world_size(active_sp) != sp or dist.get_world_size(active_cfg) != cfg:
                raise ValueError("active distributed mesh differs from requested SP×CFG degrees")
            local_rank = int(os.getenv("LOCAL_RANK", dist.get_rank()))
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
    return set_multi_gpus_devices(sp, 1, classifier_free_guidance_degree=cfg)


def sequence_parallel_chunk(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Return the xFuser/native sequence shard, or input in local mode."""

    xfuser_active = (
        get_sequence_parallel_world_size is not None
        and model_parallel_is_initialized is not None
        and model_parallel_is_initialized()
    )
    group = get_sequence_parallel_group()
    native_active = dist.is_initialized() and group is not None
    if xfuser_active:
        world_size = int(get_sequence_parallel_world_size())
        rank = int(get_sequence_parallel_rank())
    elif native_active:
        world_size = int(dist.get_world_size(group))
        rank = int(dist.get_rank(group))
    else:
        return x
    if world_size <= 1:
        return x
    if x.size(dim) % world_size:
        raise ValueError(
            f"dimension {dim} ({x.size(dim)}) must be divisible by "
            f"sequence-parallel world size {world_size}"
        )
    return torch.chunk(x, world_size, dim=dim)[rank]


def sequence_parallel_all_gather(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather xFuser/native sequence shards, or return input in local mode."""

    xfuser_active = (
        get_sequence_parallel_world_size is not None
        and model_parallel_is_initialized is not None
        and model_parallel_is_initialized()
    )
    if xfuser_active:
        if int(get_sequence_parallel_world_size()) <= 1:
            return x
        _increment_collective_counter("sequence_output_all_gather_calls")
        return get_sp_group().all_gather(x, dim=dim)
    group = get_sequence_parallel_group()
    if not dist.is_initialized() or group is None or dist.get_world_size(group) <= 1:
        return x
    _increment_collective_counter("sequence_output_all_gather_calls")
    shards = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, x.contiguous(), group=group)
    return torch.cat(shards, dim=dim)

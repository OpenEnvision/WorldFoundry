"""Legacy DP/CP group snapshot used by Ulysses-style attention.

This module keeps an independently initialized copy of data-parallel and
context-parallel groups in module-level globals. It is **not** the source of
truth for process-group topology.

The intended source of truth is
``worldfoundry.core.distributed.sequence_parallel.parallel_state``
(``GroupCoordinator`` singletons, with destroy/patch support). A third snapshot
lives in ``sequence_parallel_runtime``. The three parallel-state singletons are
intentionally not merged: combining them is a cross-runtime behavior change.

Call :func:`reset_context_parallel` between tests to clear these globals.

Public surface: :func:`init_context_parallel`, the ``get_cp_*`` / ``get_dp_*``
accessors, gather/split/reduce helpers, and :func:`dynamic_switch`.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed.device_mesh import init_device_mesh

from worldfoundry.core.distributed.collectives.device_mesh import all_to_all_tensor

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Module-level DP/CP snapshot — independent of model_parallel_groups
# ──────────────────────────────────────────────────────────────────────────

dp_size = None
cp_size = None
dp_group = None
cp_group = None
cp_stream = None
dp_ranks = None
cp_ranks = None
dp_rank = None
cp_rank = None


def reset_context_parallel() -> None:
    """Clear module-level DP/CP globals so tests can re-initialize cleanly.

    Does not destroy torch process groups; it only drops this module's
    references to them.
    """

    global dp_size, cp_size, dp_group, cp_group, cp_stream, dp_ranks, cp_ranks, dp_rank, cp_rank
    dp_size = None
    cp_size = None
    dp_group = None
    cp_group = None
    cp_stream = None
    dp_ranks = None
    cp_ranks = None
    dp_rank = None
    cp_rank = None


def init_context_parallel(context_parallel_size: int = 1, global_rank: int = 0, world_size: int = 1):
    """Build a 2-D DeviceMesh and cache this rank's DP/CP groups.

    Layout is ``[dp_size, cp_size]`` so ranks that share a sample sit on the
    CP axis. ``world_size`` must be a multiple of ``context_parallel_size``;
    otherwise the mesh would invent a leftover rank that never joins the
    collective and hang the first Ulysses all-to-all.

    Failure semantics: raises :class:`RuntimeError` on a non-divisible world.
    Does not destroy a previous mesh — call :func:`reset_context_parallel`
    first or leftover globals silently alias the old groups.
    """

    global dp_size, cp_size, dp_group, cp_group, dp_ranks, cp_ranks, dp_rank, cp_rank

    if world_size % context_parallel_size != 0:
        raise RuntimeError(f"world_size {world_size} must be multiple of context_parallel_size {context_parallel_size}")

    cp_size = context_parallel_size
    dp_size = world_size // context_parallel_size

    logger.info(
        "[rank %s] init_device_mesh [dp_size x cp_size]: [%s x %s]",
        global_rank,
        dp_size,
        cp_size,
    )
    mesh_2d = init_device_mesh("cuda", (dp_size, cp_size), mesh_dim_names=("dp", "cp"))
    logger.info("[rank %s] mesh_2d: %s", global_rank, mesh_2d)

    dp_group = mesh_2d.get_group(mesh_dim="dp")
    cp_group = mesh_2d.get_group(mesh_dim="cp")
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
    dp_rank = dist.get_rank(group=dp_group)
    cp_rank = dist.get_rank(group=cp_group)

    current_global_rank = torch.distributed.get_rank()
    logger.info(
        "[rank %s] [dp_rank, cp_rank]: [%s, %s], dp_ranks: %s, cp_ranks: %s",
        current_global_rank,
        dp_rank,
        cp_rank,
        dp_ranks,
        cp_ranks,
    )


# ──────────────────────────────────────────────────────────────────────────
# Snapshot accessors — None until init_context_parallel runs
# ──────────────────────────────────────────────────────────────────────────


def get_cp_size():
    """Return cached CP world size, or ``None`` before initialization."""

    global cp_size
    return cp_size


def get_dp_size():
    """Return cached DP world size, or ``None`` before initialization."""

    global dp_size
    return dp_size


def get_cp_stream():
    """Return a lazily created CUDA stream for overlapping CP collectives.

    The stream is process-local and is *not* synchronized with the default
    compute stream unless the caller waits on it.
    """

    global cp_stream
    if cp_stream is None:
        cp_stream = torch.cuda.Stream()
    return cp_stream


def get_dp_group():
    """Return this rank's DP process group, or ``None`` before initialization."""

    global dp_group
    return dp_group


def get_cp_group():
    """Return this rank's CP process group, or ``None`` before initialization."""

    global cp_group
    return cp_group


def get_dp_rank():
    """Return this rank's index inside the DP group, or ``None`` before init."""

    global dp_rank
    return dp_rank


def get_cp_rank():
    """Return this rank's index inside the CP group, or ``None`` before init."""

    global cp_rank
    return cp_rank


def get_cp_rank_list():
    """Return global ranks in this CP group, lazily filled from ``cp_group``."""

    global cp_ranks
    if cp_ranks is None:
        cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
    return cp_ranks


# ──────────────────────────────────────────────────────────────────────────
# CP broadcast / split — sequence shards must stay aligned with RoPE
# ──────────────────────────────────────────────────────────────────────────


def cp_broadcast(tensor, cp_index=0):
    """Broadcast ``tensor`` from ``cp_ranks[cp_index]`` on the CP group.

    ``cp_index`` is a *group-local* index, not a global rank. Using a global
    rank here would pick the wrong source and silently corrupt every shard.
    """

    cp_ranks = get_cp_rank_list()
    torch.distributed.broadcast(tensor, cp_ranks[cp_index], group=cp_group)


def cp_broadcast_objects(tensor):
    """Object-list CP broadcast — not implemented; pickle must not ride NCCL."""

    raise NotImplementedError("cp_broadcast_objects method is not yet implemented")


def split_tensor_in_cp(input, seq_dim):
    """Keep this rank's contiguous slice along ``seq_dim``.

    Sequence length must be divisible by ``cp_size``. A remainder would leave
    the last rank with a shorter shard and break all-gather concat on the
    backward path.
    """

    global cp_size

    seq_size = input.shape[seq_dim]
    if seq_size % cp_size != 0:
        raise RuntimeError(f"seq_length {seq_size} in dim {seq_dim} must be multiple of cp_size {cp_size}")

    split_seq_size = seq_size // cp_size
    tensor_splits = input.split(split_seq_size, dim=seq_dim)
    return tensor_splits[get_cp_rank()]


def split_tensor_in_cp_2d(input, dim_hw, split_hw):
    """Keep this rank's 2-D spatial tile (height × width) for 2-D Ulysses.

    Ranks are numbered row-major: ``rank = h_idx * split_w + w_idx``.
    ``cp_size`` must equal ``split_h * split_w`` so every rank owns exactly
    one tile; leftover ranks would hang on the matching gather.
    """

    global cp_size

    dim_h, dim_w = dim_hw
    split_h, split_w = split_hw
    if cp_size != split_h * split_w:
        raise RuntimeError(f"cp_size {cp_size} must equal split_h * split_w ({split_h} * {split_w})")

    seq_size_h = input.shape[dim_h]
    seq_size_w = input.shape[dim_w]
    if seq_size_h % split_h != 0:
        raise RuntimeError(f"seq_size_h {seq_size_h} in dim_h {dim_h} must be multiple of split_h {split_h}")
    if seq_size_w % split_w != 0:
        raise RuntimeError(f"seq_size_w {seq_size_w} in dim_w {dim_w} must be multiple of split_w {split_w}")

    split_seq_size_h = seq_size_h // split_h
    split_seq_size_w = seq_size_w // split_w

    tensor_splits = []
    for tensor_split_h in input.split(split_seq_size_h, dim=dim_h):
        tensor_splits.extend(tensor_split_h.split(split_seq_size_w, dim=dim_w))

    return tensor_splits[get_cp_rank()]


# ──────────────────────────────────────────────────────────────────────────
# Autograd gather/split — scale grads by cp_size so mean-reduction is honest
# ──────────────────────────────────────────────────────────────────────────


class GatherFunction(torch.autograd.Function):
    """All-gather sequence shards; backward splits and rescales by ``cp_size``.

    Forward rearranges ``B (T S) C → B T S C`` so the gather concatenates the
    spatial axis, not the packed token axis. Backward multiplies by
    ``cp_size`` because the matching :class:`SplitFunction` divides — the pair
    is a mean-preserving identity when stacked.
    """

    @staticmethod
    def forward(ctx, input, process_group, seq_dim, frames):
        """All-gather CP shards after unpacking the packed token axis."""

        ctx.cp_group = process_group
        ctx.seq_dim = seq_dim
        ctx.frames = frames
        ctx.cp_size = get_cp_size()

        input = rearrange(input, "B (T S) C -> B T S C", T=frames)
        with torch.no_grad():
            input = input.contiguous()
            output_tensors = [torch.zeros_like(input) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, input, group=ctx.cp_group)
            output_tensor = torch.cat(output_tensors, dim=seq_dim)

        return rearrange(output_tensor, "B T S C -> B (T S) C", T=frames)

    @staticmethod
    def backward(ctx, grad_output):
        """Split the gathered grad and undo the forward mean-scale."""

        with torch.no_grad():
            grad_output = grad_output * ctx.cp_size
            grad_output = rearrange(grad_output, "B (T S) C -> B T S C", T=ctx.frames)
            grad_input = split_tensor_in_cp(grad_output, ctx.seq_dim)
            grad_input = rearrange(grad_input, "B T S C -> B (T S) C", T=ctx.frames)

        return grad_input, None, None, None


class SplitFunction(torch.autograd.Function):
    """Keep this rank's sequence shard; backward all-gathers the grad.

    Backward divides by ``cp_size`` so stacking with :class:`GatherFunction`
    does not inflate the loss gradient by the number of CP ranks.
    """

    @staticmethod
    def forward(ctx, input, process_group, seq_dim):
        """Drop every CP shard except this rank's slice along ``seq_dim``."""

        ctx.cp_group = process_group
        ctx.seq_dim = seq_dim
        ctx.cp_size = get_cp_size()
        return split_tensor_in_cp(input, ctx.seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """All-gather local grads and undo the forward mean-scale."""

        with torch.no_grad():
            grad_output = grad_output / ctx.cp_size
            output_tensors = [torch.zeros_like(grad_output) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, grad_output, group=ctx.cp_group)
            grad_input = torch.cat(output_tensors, dim=ctx.seq_dim)

        return grad_input, None, None


class GatherFunction2D(torch.autograd.Function):
    """2-D Ulysses gather: reassemble height×width tiles in row-major rank order.

    ``shape`` is the *full* ``(T, H, W)`` before the CP split. Input sequence
    length must be ``T * (H / split_h) * (W / split_w)``. Concat along width
    first (inner rank index), then height, matching :func:`split_tensor_in_cp_2d`.
    """

    @staticmethod
    def forward(ctx, input, process_group, seq_dim_hw, shape, split_hw):
        """All-gather 2-D tiles and stitch them in row-major CP rank order."""

        ctx.cp_group = process_group
        ctx.seq_dim_hw = seq_dim_hw
        ctx.split_hw = split_hw
        ctx.shape = shape
        ctx.cp_size = get_cp_size()

        t, h, w = shape
        dim_h, dim_w = seq_dim_hw
        split_h, split_w = split_hw
        if h % split_h != 0 or w % split_w != 0:
            raise RuntimeError(f"shape {(t, h, w)} is not divisible by split_hw {split_hw}")
        if t * (h // split_h) * (w // split_w) != input.shape[1]:
            raise RuntimeError("input sequence length does not match shape and split_hw")

        input = rearrange(input, "B (T H W) C -> B T H W C", T=t, H=h // split_h, W=w // split_w)
        with torch.no_grad():
            input = input.contiguous()
            output_tensors = [torch.zeros_like(input) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, input, group=ctx.cp_group)
            output_tensors_hs = []
            if ctx.cp_size % split_w != 0:
                raise RuntimeError(f"cp_size {ctx.cp_size} must be divisible by split_w {split_w}")
            for i in range(ctx.cp_size // split_w):
                output_tensors_hs.append(torch.cat(output_tensors[i * split_w : (i + 1) * split_w], dim=dim_w))
            output_tensor = torch.cat(output_tensors_hs, dim=dim_h)

        return rearrange(output_tensor, "B T H W C -> B (T H W) C")

    @staticmethod
    def backward(ctx, grad_output):
        """Split the full spatial grad back to this rank's 2-D tile."""

        t, h, w = ctx.shape
        with torch.no_grad():
            grad_output = grad_output * ctx.cp_size
            grad_output = rearrange(grad_output, "B (T H W) C -> B T H W C", T=t, H=h, W=w)
            grad_input = split_tensor_in_cp_2d(grad_output, ctx.seq_dim_hw, ctx.split_hw)
            grad_input = rearrange(grad_input, "B T H W C -> B (T H W) C")

        return grad_input, None, None, None, None


class SplitFunction2D(torch.autograd.Function):
    """Keep this rank's 2-D tile; backward restitches the full spatial map."""

    @staticmethod
    def forward(ctx, input, process_group, seq_dim_hw, split_hw):
        """Drop every 2-D tile except this CP rank's height×width shard."""

        ctx.cp_group = process_group
        ctx.seq_dim_hw = seq_dim_hw
        ctx.split_hw = split_hw
        ctx.cp_size = get_cp_size()
        return split_tensor_in_cp_2d(input, ctx.seq_dim_hw, split_hw)

    @staticmethod
    def backward(ctx, grad_output):
        """All-gather 2-D tiles and stitch them in the same row-major order."""

        with torch.no_grad():
            grad_output = grad_output / ctx.cp_size
            output_tensors = [torch.zeros_like(grad_output) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, grad_output, group=ctx.cp_group)

            split_h, split_w = ctx.split_hw
            dim_h, dim_w = ctx.seq_dim_hw
            if ctx.cp_size % split_w != 0:
                raise RuntimeError(f"cp_size {ctx.cp_size} must be divisible by split_w {split_w}")
            output_tensors_hs = []
            for i in range(ctx.cp_size // split_w):
                output_tensors_hs.append(torch.cat(output_tensors[i * split_w : (i + 1) * split_w], dim=dim_w))
            grad_input = torch.cat(output_tensors_hs, dim=dim_h)

        return grad_input, None, None, None


def gather_cp(input, frames):
    """Gather 1-D CP shards; ``frames`` unpacks the packed ``(T S)`` token axis."""

    cp_process_group = get_cp_group()
    return GatherFunction.apply(input, cp_process_group, 2, frames)


def split_cp(input, seq_dim):
    """Keep this rank's 1-D sequence shard with a matching gather backward."""

    cp_process_group = get_cp_group()
    return SplitFunction.apply(input, cp_process_group, seq_dim)


def gather_cp_2d(input, shape, split_hw):
    """Gather 2-D tiles; ``shape`` is the full ``(T, H, W)`` before the split."""

    cp_process_group = get_cp_group()
    return GatherFunction2D.apply(input, cp_process_group, (2, 3), shape, split_hw)


def split_cp_2d(input, seq_dim_hw, split_hw):
    """Keep this rank's 2-D tile with a matching gather backward."""

    cp_process_group = get_cp_group()
    return SplitFunction2D.apply(input, cp_process_group, seq_dim_hw, split_hw)


# ──────────────────────────────────────────────────────────────────────────
# CP reduce / replicate — LayerNorm stats that must see the full sequence
# ──────────────────────────────────────────────────────────────────────────


class ReduceFunction(torch.autograd.Function):
    """All-reduce in forward; identity in backward (stats already reduced)."""

    @staticmethod
    def forward(ctx, input, process_group):
        """Sum ``input`` across the CP group without mutating the caller's tensor."""

        ctx.cp_group = process_group
        output = input.detach().clone()
        dist.all_reduce(output, group=ctx.cp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Pass the grad through: the forward sum already owned the full value."""

        return grad_output.detach().clone(), None


class ReplicateFunction(torch.autograd.Function):
    """Identity in forward; all-reduce grads so replicated stats stay consistent."""

    @staticmethod
    def forward(ctx, input, process_group):
        """Clone ``input`` so later in-place LayerNorm writes do not alias."""

        ctx.cp_group = process_group
        return input.detach().clone()

    @staticmethod
    def backward(ctx, grad_output):
        """Sum grads across CP so every rank sees the same mean/var gradient."""

        grad_input = grad_output.detach().clone()
        dist.all_reduce(grad_input, group=ctx.cp_group)
        return grad_input, None


def reduce_cp(partial_sum, partial_square_sum):
    """All-reduce first and second moments computed on a local sequence shard."""

    cp_process_group = get_cp_group()
    all_sum = ReduceFunction.apply(partial_sum, cp_process_group)
    all_square_sum = ReduceFunction.apply(partial_square_sum, cp_process_group)
    return all_sum, all_square_sum


def replicate_cp(all_mean, all_var):
    """Mark already-reduced mean/var as replicated so their grads all-reduce."""

    cp_process_group = get_cp_group()
    all_mean = ReplicateFunction.apply(all_mean, cp_process_group)
    all_var = ReplicateFunction.apply(all_var, cp_process_group)
    return all_mean, all_var


# ──────────────────────────────────────────────────────────────────────────
# Ulysses all-to-all — swap sequence and head shards (pad then strip)
# ──────────────────────────────────────────────────────────────────────────


class _AllToAll(torch.autograd.Function):
    """Autograd wrapper around :func:`all_to_all_tensor`.

    Backward swaps ``scatter_dim`` and ``gather_dim`` so the inverse exchange
    restores the original layout. Calling :meth:`apply` recursively (instead
    of a raw collective) keeps the second hop on the autograd graph.
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        """Scatter along ``scatter_dim`` and gather along ``gather_dim``."""

        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)
        return all_to_all_tensor(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        """Invert the exchange by swapping scatter and gather dimensions."""

        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return return_grad, None, None, None


def all_to_all_with_pad(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_pad: int = 0,
    gather_pad: int = 0,
):
    """All-to-all after optional padding so the scatter dim is divisible.

    ``scatter_pad`` is appended *before* the collective; ``gather_pad`` is
    stripped from the *gathered* dim afterwards. Both must be agreed by every
    rank — a mismatched pad hangs NCCL.
    """

    if scatter_pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[scatter_dim] = scatter_pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_, pad_tensor], dim=scatter_dim)

    world_size = dist.get_world_size(process_group)
    if input_.shape[scatter_dim] % world_size != 0:
        raise RuntimeError(
            f"Dimension to scatter ({input_.shape[scatter_dim]}) is not divisible by world size ({world_size})"
        )
    input_ = _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

    if gather_pad > 0:
        input_ = input_.narrow(gather_dim, 0, input_.size(gather_dim) - gather_pad)

    return input_


def dynamic_switch(x, scatter_dim, gather_dim):
    """Ulysses reshape: swap sequence and head shards on the cached CP group."""

    return all_to_all_with_pad(
        x,
        get_cp_group(),
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        scatter_pad=0,
        gather_pad=0,
    )


def get_optimal_split(size):
    """Return the most square factor pair of ``size`` for a 2-D CP tile grid.

    Prefers ``(h, w)`` closest to a square so spatial all-to-alls stay balanced.
    ``size`` must be at least 1 (always true for a process-group world size).
    """

    factors = []
    for i in range(1, int(size**0.5) + 1):
        if size % i == 0:
            factors.append([i, size // i])
    return min(factors, key=lambda x: abs(x[0] - x[1]))

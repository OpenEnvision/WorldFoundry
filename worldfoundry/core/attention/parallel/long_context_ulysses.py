"""Ulysses context-parallel (CP) communication primitive.

This module implements Ulysses context-parallel attention using
PyTorch Distributed All-to-All communication primitives.

Core Design Philosophy:
1. Dimension Transformation: Before local attention calculation, input tensors are
   gathered across the sequence dimension (S) and scattered across the head dimension (H)
   to all active ranks. This allows each rank to hold the full sequence of a subset of heads,
   enabling the use of high-performance single-GPU attention kernels (e.g., FlashAttention).
2. Zero-Redundant Communication: Compared to tensor parallelism (TP), Ulysses
   requires only two All-to-All communication phases (pre-attention and post-attention),
   scaling linearly with the tensor size and matching high-bandwidth NVLink/RoCE networks.
3. Compiler Barriers: PyTorch Distributed primitives (especially `all_to_all_single`)
   often trigger graph breaks or compilation errors when used with `torch.compile`.
   Thus, entry points are decorated with `@torch.compiler.disable`.
"""

import torch
import torch.distributed as dist

from worldfoundry.core.distributed.model_parallel import context_state as context_parallel_util
from worldfoundry.core.distributed.sequence_parallel.ops import all_to_all_many

# ──────────────────────────────────────────────────────────────────────────
# All-to-all reshape — permute so all_to_all_single can slice dim 0
# ──────────────────────────────────────────────────────────────────────────


def all_to_all(tensor, scatter_idx, gather_idx, group=None, gather=True):
    """Perform precise All-to-All communication across a distributed process group.

    This function wraps PyTorch `all_to_all_single` by dynamically reshaping and
    permuting the input tensor to bypass the physical limitation of the primitive,
    which only supports partitioning and concatenating along the outermost dimension (dim 0).

    Args:
        tensor (torch.Tensor): Input tensor for all-to-all communication.
        scatter_idx (int): Dimension to split and scatter across the process group.
        gather_idx (int): Dimension along which to concatenate gathered data.
        group (ProcessGroup, optional): Process group to use for communication.
        gather (bool): Whether to perform physical gather. If False, slices the local rank's
            portion, which is helpful for gradient or memory optimization.

    Returns:
        torch.Tensor: The communicated and restructured tensor.
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group)
    ulysses_rank = context_parallel_util.get_cp_rank()
    if world_size == 1:
        return tensor

    if scatter_idx == gather_idx:
        raise ValueError("scatter_idx and gather_idx must be different")

    def chunk_tensor(tensor, scatter_idx):
        """Split scatter_idx dimension and permute it to the front (dim 0) for all-to-all compatibility."""
        t_shape = list(tensor.shape)
        if t_shape[scatter_idx] % world_size != 0:
            raise ValueError(f"Dimension {scatter_idx} must be divisible by world size {world_size}")
        chunk_size = t_shape[scatter_idx] // world_size

        # Split scatter_idx into [world_size, chunk_size]
        new_shape = list()
        for i in range(len(t_shape)):
            if i != scatter_idx:
                new_shape.append(t_shape[i])
            else:
                new_shape.extend([world_size, chunk_size])
        tensor = tensor.reshape(*new_shape)

        # Move the world_size dimension to dim 0 and make memory contiguous.
        # This is critical as `all_to_all_single` slices the tensor along dim 0 to dispatch to ranks.
        tensor = tensor.permute(scatter_idx, *[i for i in range(len(new_shape)) if i != scatter_idx]).contiguous()
        return tensor

    # Chunk tensor for all-to-all dispatch
    tensor = chunk_tensor(tensor, scatter_idx)

    # Allocate receive buffer of equivalent shape for in-place slice exchange
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)

    # After communication, output dim 0 represents sender rank. We must restructure it back.
    # E.g., for scatter_idx==1, gather_idx==2:
    # Input shape: [B, H, S/N, D] -> chunked: [N, B, H/N, S/N, D]
    # Output shape: [N, B, H/N, S/N, D] (where each local rank received S/N slice from N ranks)
    # Target shape: [B, H/N, S, D] where sequence dimension S = N * (S/N) is merged.
    def reorder_tensor(tensor, gather_idx):
        """Move rank (dim 0) next to ``gather_idx`` and merge, or keep one rank."""
        t_shape = list(tensor.shape)
        world_size = t_shape[0]

        # Build permutation indices, inserting world_size dimension (dim 0) right after gather_idx.
        permute_idx = list()
        for i in range(1, len(t_shape)):
            if i != gather_idx + 1:
                permute_idx.append(i)
            else:
                permute_idx.extend([0, i])
        tensor = tensor.permute(*permute_idx).contiguous()

        # Reshape to merge the inserted world_size with the chunk dimension, restoring sequence/head length.
        new_shape = list()
        if gather:
            for i in range(1, len(t_shape)):
                if i != gather_idx + 1:
                    new_shape.append(t_shape[i])
                else:
                    new_shape.append(world_size * t_shape[i])

            tensor = tensor.reshape(*new_shape)
        else:
            # Optimize: directly retrieve the slice corresponding to the local rank.
            tensor = tensor[:, ulysses_rank]

        return tensor

    output = reorder_tensor(output, gather_idx)

    return output


# ──────────────────────────────────────────────────────────────────────────
# Ulysses phases — gather sequence before local attn, scatter heads after
# ──────────────────────────────────────────────────────────────────────────


@torch.compiler.disable
def ulysses_a2a_in(query, key, value):
    """Pre-attention All-to-All communication phase.

    Transforms Query, Key, and Value tensors from [B, Head_Original, Seq_Split, Dim]
    (sequence-partitioned state) to [B, Head_Split, Seq_Original, Dim] (head-partitioned state).
    This ensures each rank possesses the entire sequence length for its allocated heads,
    allowing standard self-attention calculations locally.

    Disabled for torch.compile to avoid distributed graph breaks.
    """
    if context_parallel_util.get_cp_size() == 1:
        return query, key, value

    # Partition along Head dimension (scatter_idx=1), reconstruct/gather along Seq dimension (gather_idx=2)
    query, key, value = all_to_all_many(
        (query, key, value),
        scatter_dim=1,
        gather_dim=2,
        group=context_parallel_util.get_cp_group(),
    )
    return query, key, value


@torch.compiler.disable
def ulysses_a2a_out(output):
    """Post-attention All-to-All communication phase.

    Transforms the attention output tensor back from [B, Head_Split, Seq_Original, Dim]
    to [B, Head_Original, Seq_Split, Dim] to match the expected sequence-partitioned
    layout of subsequent transformer layers.
    """
    if context_parallel_util.get_cp_size() == 1:
        return output

    # Partition along Seq dimension (scatter_idx=2), reconstruct/gather along Head dimension (gather_idx=1)
    output = all_to_all(output, scatter_idx=2, gather_idx=1, group=context_parallel_util.get_cp_group())
    return output


def ulysses_wrapper(func):
    """Decorator to automatically wrap a standard self-attention forward call for Ulysses compatibility.

    The wrapper:
    1. Intercepts inputs, routing Q, K, V through `ulysses_a2a_in`.
    2. Invokes the underlying attention kernel (e.g., FlashAttention, SDPA) on the fully gathered sequence.
    3. Transforms output back to sequence-partitioned space via `ulysses_a2a_out`.
    """

    def wrapper(self, query, key, value, shape):
        """Run ``func`` between Ulysses gather-in and scatter-out."""
        query, key, value = ulysses_a2a_in(query, key, value)
        output = func(self, query, key, value, shape)
        output = ulysses_a2a_out(output)
        return output

    return wrapper

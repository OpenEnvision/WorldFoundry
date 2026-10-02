# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-parallel runtime: CSO helper, Ulysses scheduler, and CP pre/post process.

Long video sequences do not fit one GPU's attention workspace. This module
splits packed Q/K/V along the sequence (or head) dim, runs local attention,
then concatenates. :class:`UlyssesScheduler` orders the all-to-alls so RoPE
positions stay aligned with :mod:`worldfoundry.core.distributed.model_parallel.context`
splits. :class:`CSOHelper` is the communication-then-compute helper used by
packed core / cross attention.

Do not call these collectives without the matching RoPE CP shard — positions
would silently rotate the wrong tokens.
"""

import math
from typing import Callable, List, Tuple, Union

import torch
import torch.distributed

from worldfoundry.core.attention.sequence.packed_sequence import ModelMetaArgs, PackedCoreAttnParams, PackedCrossAttnParams
from worldfoundry.core.distributed.model_parallel import groups as mpu
from worldfoundry.core.utils.python.misc_utils import divide


def _rearrange(*args, **kwargs):
    """Import einops lazily so CPU-only import of this module stays cheap."""
    from einops import rearrange

    return rearrange(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Scatter / gather — slice packed sequences onto CP ranks (or gather them back)
# ──────────────────────────────────────────────────────────────────────────


def scatter_to_context_parallel_region(input_, cp_split_sizes, cp_shuffle_num=1, cp_pad_size=0):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = mpu.get_cp_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension with padding.
    rank = mpu.get_cp_rank()
    if cp_shuffle_num > 1:
        cp_pad_size = divide(cp_pad_size, cp_shuffle_num)
        cp_split_sizes = [divide(s, cp_shuffle_num) for s in cp_split_sizes]
        dim_offset = sum(cp_split_sizes[:rank])
        xs = []
        for x in torch.chunk(input_, cp_shuffle_num, dim=0):
            x = torch.nn.functional.pad(x, [0, 0] * (x.dim() - 1) + [0, cp_pad_size], mode="constant", value=0)
            xs.append(x[dim_offset : dim_offset + cp_split_sizes[rank]])
        output = torch.concat(xs, dim=0)
    else:
        dim_offset = sum(cp_split_sizes[:rank])
        x = torch.nn.functional.pad(input_, [0, 0] * (input_.dim() - 1) + [0, cp_pad_size], mode="constant", value=0)
        output = x[dim_offset : dim_offset + cp_split_sizes[rank]].contiguous()
    return output


def gather_from_context_parallel_region(input_, cp_split_sizes, cp_shuffle_num=1, cp_pad_size=0):
    """Gather tensors and concatinate along the first dimension."""

    world_size = mpu.get_cp_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    input_ = input_.contiguous()
    total_seq_len = sum(cp_split_sizes)
    dim_size = list(input_.size())
    dim_size[0] = total_seq_len

    output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    outputs = list(torch.split(output, cp_split_sizes, dim=0))
    torch.distributed.all_gather(outputs, input_, group=mpu.get_cp_group())
    if cp_shuffle_num > 1:
        total_seq_len = divide(total_seq_len, cp_shuffle_num)
        cp_pad_size = divide(cp_pad_size, cp_shuffle_num)
        chunks = [torch.chunk(o, cp_shuffle_num, dim=0) for o in outputs]
        output = torch.concat(
            [
                torch.concat([chunk[i] for chunk in chunks], dim=0)[: total_seq_len - cp_pad_size]
                for i in range(cp_shuffle_num)
            ],
            dim=0,
        )
    else:
        output = torch.concat(outputs, dim=0)[: total_seq_len - cp_pad_size]

    return output


class FakeHandle:
    """Completed work handle used when CP world size is 1 (no collective)."""

    def __init__(self):
        """Construct a no-op handle that is already complete."""
        pass

    def wait(self):
        """No-op; there is no outstanding collective to join."""
        pass


# ──────────────────────────────────────────────────────────────────────────
# Packed-parameter rewrite — keep varlen ranges aligned after a CP split
# ──────────────────────────────────────────────────────────────────────────


def update_packed_seq_params_for_cuda_graph(cross_attn_params: PackedCrossAttnParams, xattn_mask: torch.Tensor):
    """Rewrite KV ranges onto the static caption layout CUDA Graphs captured.

    Live ``cu_seqlens_k`` change with the mask; the graph captured a fixed
    caption length. Map each dynamic start onto that static grid so the
    packed kernel still indexes the captured buffer.
    """
    assert xattn_mask is not None
    # xattn_mask: (N * denoising_range_num, L, 1, 1)
    xattn_mask = xattn_mask.reshape(xattn_mask.shape[0], -1)
    batch_size, static_caption_length = xattn_mask.shape

    # Get index_map for kv_range injection, map y_index to static_caption_length
    y_index = torch.sum(xattn_mask, dim=-1)
    cu_seqlens_k = torch.cat([y_index.new_tensor([0]), y_index]).to(torch.int32).to(xattn_mask.device)
    cu_seqlens_k = cu_seqlens_k.cumsum(-1).to(torch.int32)
    static_cu_seqlens_k = torch.arange(0, (batch_size + 1) * static_caption_length, static_caption_length)
    assert cu_seqlens_k.shape[0] == batch_size + 1 == static_cu_seqlens_k.shape[0]
    start_index_map = dict(zip(cu_seqlens_k.flatten().tolist(), static_cu_seqlens_k.flatten().tolist()))

    # Move kv_range to the right position
    kv_range_start_list = cross_attn_params.kv_ranges[:, 0].flatten().tolist()
    static_kv_range_start = [start_index_map[kv_range_start_list[i]] for i in range(len(kv_range_start_list))]
    static_kv_range_start = torch.tensor(static_kv_range_start, dtype=torch.int32, device=xattn_mask.device)
    assert static_kv_range_start.shape[0] == cross_attn_params.kv_ranges.shape[0]
    static_kv_range_diff = cross_attn_params.kv_ranges[:, 1] - cross_attn_params.kv_ranges[:, 0]
    static_kv_range_end = static_kv_range_start + static_kv_range_diff
    static_kv_range = torch.stack((static_kv_range_start, static_kv_range_end), dim=1)

    assert static_kv_range.shape == cross_attn_params.kv_ranges.shape
    return PackedCrossAttnParams(
        q_ranges=cross_attn_params.q_ranges,
        kv_ranges=static_kv_range,
        cu_seqlens_q=cross_attn_params.cu_seqlens_q,
        cu_seqlens_kv=cross_attn_params.cu_seqlens_kv,
        max_seqlen_q=cross_attn_params.max_seqlen_q,
        max_seqlen_kv=cross_attn_params.max_seqlen_kv,
    )


def cp_update_cross_attn_qkv_range(
    cross_attn_params: PackedCrossAttnParams,
    batch_size: int,
    cp_split_sizes: List[int],
    device: torch.device,
    cp_shuffle_num: int = 1,
    cp_pad_size: int = 0,
):
    """
    Update cross_attn_params for cross_attn in context parallel.

    Input:
        cross_attn_params: PackedCrossAttnParams. Packed sequence parameters for cross_atten
        batch_size: int. Batch size
        cp_split_sizes: List[int]. Split sizes for each rank
        device: torch.device. Device

    Output:
        cross_attn_params: PackedCrossAttnParams. Updated packed parameters for cross_atten
    """
    # Update cu_seqlens_q and max_seqlen_q because split x maybe unbalanced
    cp_rank = mpu.get_cp_rank()
    seq_len_cur_rank = cp_split_sizes[cp_rank]
    cp_split_sizes = [divide(x, cp_shuffle_num) for x in cp_split_sizes]
    cp_split_sizes = torch.tensor(cp_split_sizes, dtype=torch.int32, device=device)
    base_cp_boundaries = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), cp_split_sizes.cumsum(0)))
    total_seq_len = base_cp_boundaries[-1]

    cu_seqlens_q = cross_attn_params.cu_seqlens_q
    cu_seqlens_k = cross_attn_params.cu_seqlens_kv
    cu_seqlens_pad = torch.arange(cu_seqlens_q.shape[0], dtype=torch.int32, device=device) * divide(
        cp_pad_size, cp_shuffle_num
    )
    cu_seqlens_q = cu_seqlens_q + cu_seqlens_pad

    q_seg_starts, q_seg_ends = cu_seqlens_q[:-1], cu_seqlens_q[1:]

    xattn_q_ranges, xattn_k_ranges = [], []
    for i in range(batch_size):
        inner_xattn_q_ranges, inner_xattn_k_ranges = [], []
        for j in range(cp_shuffle_num):
            global_offset = i * total_seq_len * cp_shuffle_num + j * total_seq_len
            cp_boundaries = base_cp_boundaries + global_offset
            this_cp_start, this_cp_end = (cp_boundaries[cp_rank], cp_boundaries[cp_rank + 1])

            q_inter_starts = torch.maximum(this_cp_start, q_seg_starts)
            q_inter_ends = torch.minimum(this_cp_end, q_seg_ends)

            q_mask = q_inter_starts < q_inter_ends
            valid_q_starts = q_inter_starts[q_mask]
            valid_q_ends = q_inter_ends[q_mask]

            k_seg_starts, k_seg_ends = cu_seqlens_k[:-1], cu_seqlens_k[1:]
            valid_indices = torch.nonzero(q_mask, as_tuple=True)[0]

            valid_k_starts = k_seg_starts[valid_indices]
            valid_k_ends = k_seg_ends[valid_indices]

            part_xattn_q_rangs = torch.stack((valid_q_starts, valid_q_ends), dim=1)
            offset = part_xattn_q_rangs[:, 0].min()
            part_xattn_q_rangs = part_xattn_q_rangs - offset

            inner_xattn_q_ranges.append(part_xattn_q_rangs)
            inner_xattn_k_ranges.append(torch.stack((valid_k_starts, valid_k_ends), dim=1))
        inner_end_values = torch.tensor([ranges[-1, -1] for ranges in inner_xattn_q_ranges], dtype=torch.int32)
        inner_offsets = torch.cat(
            (torch.zeros(1, dtype=inner_end_values.dtype), torch.cumsum(inner_end_values[:-1], dim=0))
        )
        inner_xattn_q_ranges = [tensor + int(offset) for tensor, offset in zip(inner_xattn_q_ranges, inner_offsets)]
        xattn_q_ranges.append(torch.cat(inner_xattn_q_ranges, dim=0))
        xattn_k_ranges.append(torch.cat(inner_xattn_k_ranges, dim=0))

    end_values = torch.tensor([ranges[-1, -1].item() for ranges in xattn_q_ranges], dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=end_values.dtype), torch.cumsum(end_values[:-1], dim=0)))

    shifted_tensors = [tensor + int(offset) for tensor, offset in zip(xattn_q_ranges, offsets)]
    xattn_q_ranges_ts = torch.cat(shifted_tensors, dim=0)
    xattn_k_ranges_ts = torch.cat(xattn_k_ranges, dim=0)

    cu_seqlens_q = torch.unique(xattn_q_ranges_ts)
    cu_seqlens_k = torch.unique(xattn_k_ranges_ts)
    assert cu_seqlens_q.shape == cu_seqlens_k.shape, (
        f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, cu_seqlens_k.shape: {cu_seqlens_k.shape}, "
    )

    return PackedCrossAttnParams(
        q_ranges=xattn_q_ranges_ts,
        kv_ranges=xattn_k_ranges_ts,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=seq_len_cur_rank,
        max_seqlen_kv=cross_attn_params.max_seqlen_kv,
    )


def cp_ulysses_process(
    cp_size: int,
    x: torch.Tensor,
    condition_map: torch.Tensor,
    rope: torch.Tensor,
    xattn_mask_for_cuda_graph: Union[torch.Tensor, None],
    cross_attn_params: PackedCrossAttnParams,
):
    """Scatter tokens/RoPE/condition map and rewrite cross-attn ranges for Ulysses.

    Sequence is split as evenly as possible (remainder on the first ranks).
    RoPE travels with the tokens so positions stay aligned after the scatter.
    """
    seq_len, N, D = x.shape
    assert seq_len == rope.size(0), f"seq_len: {seq_len} != rope.size(0): {rope.size(0)}"
    assert condition_map.size(0) == seq_len, f"condition_map.size(0): {condition_map.size(0)} != seq_len: {seq_len}"

    # Part1: split for CP
    cp_split_sizes = [seq_len // cp_size] * cp_size
    for i in range(seq_len % cp_size):
        cp_split_sizes[i] += 1

    # Part2: scatter to CP
    x = scatter_to_context_parallel_region(x, cp_split_sizes)
    condition_map = scatter_to_context_parallel_region(condition_map, cp_split_sizes)
    rope = scatter_to_context_parallel_region(rope, cp_split_sizes)

    # Part3: update cross_attn cross_attn_params
    cross_attn_params = cp_update_cross_attn_qkv_range(cross_attn_params, N, cp_split_sizes, x.device)
    if xattn_mask_for_cuda_graph is not None:
        cross_attn_params = update_packed_seq_params_for_cuda_graph(cross_attn_params, xattn_mask_for_cuda_graph)

    return x, condition_map, rope, cp_split_sizes, cross_attn_params


def cp_shuffle_overlap_process(
    cp_size: int,
    x: torch.Tensor,
    condition_map: torch.Tensor,
    rope: torch.Tensor,
    xattn_mask_for_cuda_graph: Union[torch.Tensor, None],
    ardf_meta: dict,
    core_attn_params: PackedCoreAttnParams,
    cross_attn_params: PackedCrossAttnParams,
):
    """Scatter a shuffle-overlap layout and scale packed ranges for padding.

    When ``seq_len / denoising_range_num`` is not divisible by CP size,
    tokens are padded so each shuffle chunk shards evenly. Query ranges
    are scaled by the padded/unpadded ratio; KV ranges stay global.
    """
    seq_len, N, D = x.shape
    assert seq_len == rope.size(0), f"seq_len: {seq_len} != rope.size(0): {rope.size(0)}"
    assert condition_map.size(0) == seq_len, f"condition_map.size(0): {condition_map.size(0)} != seq_len: {seq_len}"
    cp_shuffle_num = ardf_meta["denoising_range_num"]

    # Part1: calculate cp_pad_size and cp_split_sizes
    cp_pad_size = 0
    if divide(seq_len, cp_shuffle_num) % cp_size != 0:
        cp_pad_size = (cp_size - divide(seq_len, cp_shuffle_num) % cp_size) * cp_shuffle_num
    cp_split_sizes = [(seq_len + cp_pad_size) // cp_size] * cp_size

    # Part2: scatter to CP
    x = scatter_to_context_parallel_region(x, cp_split_sizes, cp_shuffle_num, cp_pad_size)
    condition_map = scatter_to_context_parallel_region(condition_map, cp_split_sizes, cp_shuffle_num, cp_pad_size)
    rope = scatter_to_context_parallel_region(rope, cp_split_sizes, cp_shuffle_num, cp_pad_size)

    # Part3: update core_attn_params
    gcd = math.gcd(seq_len, seq_len + cp_pad_size)
    _sq = seq_len // gcd
    _psq = (seq_len + cp_pad_size) // gcd
    q_range = ardf_meta["q_range"] * _psq // _sq
    max_seqlen_q = ardf_meta["max_seqlen_q"] * _psq // _sq
    core_attn_params = PackedCoreAttnParams(
        q_range=q_range,
        k_range=ardf_meta["k_range"],
        np_q_range=q_range.cpu().numpy(),
        np_k_range=ardf_meta["k_range"].cpu().numpy(),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=ardf_meta["max_seqlen_k"],
    )

    # Part4: update cross_attn cross_attn_params
    cross_attn_params = cp_update_cross_attn_qkv_range(
        cross_attn_params, N, cp_split_sizes, x.device, cp_shuffle_num, cp_pad_size
    )
    if xattn_mask_for_cuda_graph is not None:
        cross_attn_params = update_packed_seq_params_for_cuda_graph(cross_attn_params, xattn_mask_for_cuda_graph)

    return x, condition_map, rope, cp_pad_size, cp_split_sizes, core_attn_params, cross_attn_params


def cp_pre_process(
    cp_size: int,
    cp_strategy: str,
    x: torch.Tensor,
    condition_map: torch.Tensor,
    rope: torch.Tensor,
    xattn_mask_for_cuda_graph: Union[torch.Tensor, None],
    ardf_meta: dict,
    core_attn_params: PackedCoreAttnParams,
    cross_attn_params: PackedCrossAttnParams,
):
    """
    This function is used to handle context parallel behavior,
    split input tensors into multiple parts and scatter them to different GPUs.

    Input:
        cp_strategy: str. cp_ulysses for hopper or newer, cp_shuffle_overlap for 4090 or older
        x: (S, N, D). torch.Tensor of inputs embedding (images or latent representations of images)
        condition_map: (N * S). torch.Tensor determine which condition to use for each token
        rope: (S, 96). torch.Tensor of rope
        xattn_mask_for_cuda_graph: (N * denoising_range_num, L, 1, 1). torch.Tensor of xattn mask for cuda graph, None means no cuda graph
        core_attn_params: PackedCoreAttnParams. Packed sequence parameters for core_atten
        cross_attn_params: PackedCrossAttnParams. Packed sequence parameters for cross_atten

    Output:
        x: (S', N, D). torch.Tensor of inputs embedding (images or latent representations of images)
        condition_map: (N * S'). torch.Tensor determine which condition to use for each token
        rope: (S', 96). torch.Tensor of rope
        cp_split_sizes: List[int]. Split sizes for each rank
        core_attn_params: PackedCoreAttnParams
        cross_attn_params: PackedCrossAttnParams
    """
    if cp_size == 1:
        return x, condition_map, rope, None, None, core_attn_params, cross_attn_params
    if cp_strategy == "cp_ulysses":
        (x, condition_map, rope, cp_split_sizes, cross_attn_params) = cp_ulysses_process(
            cp_size, x, condition_map, rope, xattn_mask_for_cuda_graph, cross_attn_params
        )
        return (x, condition_map, rope, 0, cp_split_sizes, core_attn_params, cross_attn_params)
    elif cp_strategy == "cp_shuffle_overlap":
        (
            x,
            condition_map,
            rope,
            cp_pad_size,
            cp_split_sizes,
            core_attn_params,
            cross_attn_params,
        ) = cp_shuffle_overlap_process(
            cp_size, x, condition_map, rope, xattn_mask_for_cuda_graph, ardf_meta, core_attn_params, cross_attn_params
        )
        return (x, condition_map, rope, cp_pad_size, cp_split_sizes, core_attn_params, cross_attn_params)
    else:
        raise ValueError(f"Invalid CP strategy: {cp_strategy}, expected cp_ulysses or cp_shuffle_overlap")


def cp_post_process(cp_size: int, cp_strategy: str, x: torch.Tensor, meta_args: ModelMetaArgs) -> torch.Tensor:
    """Gather context-parallel output back to the original sequence layout.

    Args:
        cp_size: Context-parallel world size. ``1`` returns ``x`` unchanged.
        cp_strategy: ``"cp_ulysses"`` or ``"cp_shuffle_overlap"``.
        x: This rank's local output tensor.
        meta_args: Split sizes and padding recorded by ``cp_pre_process``.

    Returns:
        Globally gathered tensor with shuffle-overlap padding removed.

    Raises:
        ValueError: ``cp_strategy`` is unknown.
    """
    if cp_size == 1:
        return x
    if cp_strategy == "cp_shuffle_overlap":
        x = gather_from_context_parallel_region(
            x, meta_args.cp_split_sizes, meta_args.denoising_range_num, meta_args.cp_pad_size
        )
    elif cp_strategy == "cp_ulysses":
        x = gather_from_context_parallel_region(x, meta_args.cp_split_sizes)
    else:
        raise ValueError(f"Invalid CP strategy: {cp_strategy}, expected cp_ulysses or cp_shuffle_overlap")
    return x


# ──────────────────────────────────────────────────────────────────────────
# Ulysses pipeline — all-to-all heads↔sequence, overlap comm with compute
# ──────────────────────────────────────────────────────────────────────────


def all_to_all_input_split(
    tensor: torch.Tensor, cp_split_sizes: List[int]
) -> Tuple[torch.Tensor, torch.distributed.Work]:
    """
    Scatter head_number and gather seq_len, for example:
    input: (seq_len, cp * hn, hd)
    output: (seq_len * cp, hn, hd)
    NOTE: seq_len of input maybe not equal, which depends on cp_split_sizes[mpu.get_cp_rank()]
    """
    cp_world_size = mpu.get_cp_world_size()
    if cp_world_size == 1:
        return tensor, FakeHandle()
    assert cp_split_sizes is not None
    _, hn, _ = tensor.shape
    if cp_world_size % hn == 0 and cp_world_size != hn:
        tensor = torch.repeat_interleave(tensor, repeats=divide(cp_world_size, hn), dim=1).contiguous()
    assert tensor.is_contiguous()
    input = _rearrange(tensor, "seq (cp hn) hd -> (cp seq) hn hd", cp=cp_world_size).contiguous()
    output = torch.empty([sum(cp_split_sizes), *input.shape[1:]], device=input.device, dtype=input.dtype)
    handle = torch.distributed.all_to_all_single(
        output, input, output_split_sizes=cp_split_sizes, group=mpu.get_cp_group(), async_op=True
    )
    return output, handle


def all_to_all_output_split(
    tensor: torch.Tensor, cp_split_sizes: List[int]
) -> Tuple[torch.Tensor, torch.distributed.Work]:
    """
    Scatter seq_len and gather head_number, for example:
    input: (seq_len * cp, hn, hd)
    output: (seq_len, cp * hn, hd)
    NOTE: seq_len of output maybe not equal, which depends on cp_split_sizes[mpu.get_cp_rank()]
    """
    cp_world_size = mpu.get_cp_world_size()
    if cp_world_size == 1:
        return tensor, FakeHandle()
    assert cp_split_sizes is not None
    assert tensor.is_contiguous()
    _, hn, _ = tensor.shape
    output = torch.empty(
        [cp_split_sizes[mpu.get_cp_rank()] * cp_world_size, *tensor.shape[1:]], device=tensor.device, dtype=tensor.dtype
    )
    handle = torch.distributed.all_to_all_single(
        output, tensor, input_split_sizes=cp_split_sizes, group=mpu.get_cp_group(), async_op=True
    )
    return output, handle


def fused_qkv_communication(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp_split_sizes: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-to-all Q, K, and V in one collective by concatenating heads.

    Repeats K/V heads when CP size is a multiple of KV heads (MQA/GQA)
    so every rank still receives a valid head shard. One fused launch
    cuts CPU-bound kernel overhead on short sequences.
    """
    cp_world_size = mpu.get_cp_world_size()
    if cp_world_size == 1:
        return q, k, v
    assert cp_split_sizes is not None
    _, k_head, _ = k.shape
    if cp_world_size % k_head == 0 and cp_world_size != k_head:
        k = torch.repeat_interleave(k, repeats=divide(cp_world_size, k_head), dim=1)
        v = torch.repeat_interleave(v, repeats=divide(cp_world_size, k_head), dim=1)

    q = _rearrange(q, "seq (cp hn) hd -> (cp seq) hn hd", cp=cp_world_size).contiguous()
    k = _rearrange(k, "seq (cp hn) hd -> (cp seq) hn hd", cp=cp_world_size).contiguous()
    v = _rearrange(v, "seq (cp hn) hd -> (cp seq) hn hd", cp=cp_world_size).contiguous()
    head_split_number = [q.shape[1], k.shape[1], v.shape[1]]
    qkv = torch.cat([q, k, v], dim=1).contiguous()

    qkv_output = torch.empty([sum(cp_split_sizes), *qkv.shape[1:]], device=qkv.device, dtype=qkv.dtype)
    torch.distributed.all_to_all_single(
        qkv_output, qkv, output_split_sizes=cp_split_sizes, group=mpu.get_cp_group(), async_op=False
    )
    q, k, v = torch.split(qkv_output, head_split_number, dim=1)
    return q, k, v


class UlyssesScheduler:
    """Overlap Ulysses all-to-all communication with Q/K/V and cross attention.

    Static entry points select separate, fused-KV, or fused-QKV communication
    schedules. Each returns self-attention output restored to model layout plus
    the concurrently computed cross-attention output.
    """

    def __init__(self):
        """No instance state; methods are static schedules."""
        pass

    @staticmethod
    def get_attn_and_xattn_with_comm_overlap(
        get_q_func: Callable,  # [seq hn hd]
        get_k_func: Callable,  # [seq hn hd]
        get_v_func: Callable,  # [seq hn hd]
        kv_cache_func: Callable,
        core_attn_func: Callable,
        cross_attn_func: Callable,
        overlap_degree: int,
        batch_size: int,
        cp_size: int,
        cp_split_sizes: List[int] = None,
    ):
        """
        Get Q, K, V with communication overlap.
        Input:
            get_q: Callable, function to get q, shape [b, sq, hn, hd]
            get_k: Callable, function to get k, shape [sq, b, hn, hd]
            get_v: Callable, function to get v, shape [sq, b, hn, hd]
        NOTE: Why follow such compute and comm order?
        1. v_compute
        2. k_compute(overlap with v_comm)
        3. q_compute(overlap with k_comm)
        4. kv_cache_func(overlap with q_comm)
        Follow the principle: We need to begin comm as soon as possible to hide the comm latency.
        The computation flops and commnunication order is:
        flops order: q_compute (larger hidden_size + layernorm) > k_compute (layernorm) > v_compute
        comm order: q_compute (larger hidden_size) > k_compute = v_compute
        """
        value = get_v_func()
        value, handle_v = all_to_all_input_split(value, cp_split_sizes)
        key = get_k_func()
        key, handle_k = all_to_all_input_split(key, cp_split_sizes)
        query = get_q_func()
        query, handle_q = all_to_all_input_split(query, cp_split_sizes)

        handle_v.wait()
        handle_k.wait()
        kv = torch.concat([key, value], dim=-1)

        key, value = kv_cache_func(kv)
        handle_q.wait()
        return UlyssesScheduler.get_attn_and_xattn_base(
            query, key, value, core_attn_func, cross_attn_func, overlap_degree, batch_size, cp_size, cp_split_sizes
        )

    @staticmethod
    def get_attn_and_xattn_with_fused_kv_comm(
        get_q_func: Callable,
        get_kv_func: Callable,
        kv_cache_func: Callable,
        core_attn_func: Callable,
        cross_attn_func: Callable,
        overlap_degree: int,
        batch_size: int,
        cp_size: int,
        cp_split_sizes: List[int] = None,
    ):
        """
        When seq_len is very small, CPU-bound issues are severe. By fusing kv communication,
        CPU operations and the number of kernel launches are reduced.
        """
        kv = get_kv_func()
        kv, handle_kv = all_to_all_input_split(kv, cp_split_sizes)
        query = get_q_func()
        query, handle_q = all_to_all_input_split(query, cp_split_sizes)
        handle_kv.wait()
        key, value = kv_cache_func(kv)
        handle_q.wait()
        return UlyssesScheduler.get_attn_and_xattn_base(
            query, key, value, core_attn_func, cross_attn_func, overlap_degree, batch_size, cp_size, cp_split_sizes
        )

    def get_attn_and_xattn_with_fused_qkv_comm(
        get_qkv_func: Callable,
        kv_cache_func: Callable,
        core_attn_func: Callable,
        cross_attn_func: Callable,
        overlap_degree: int,
        batch_size: int,
        cp_size: int,
        cp_split_sizes: List[int] = None,
    ):
        """
        By fusing the communication of q, k, and v together, further optimize CPU-bound issues.
        """
        q, k, v = get_qkv_func()
        q, k, v = fused_qkv_communication(q, k, v, cp_split_sizes)
        k, v = kv_cache_func(torch.cat([k, v], dim=-1))
        return UlyssesScheduler.get_attn_and_xattn_base(
            q, k, v, core_attn_func, cross_attn_func, overlap_degree, batch_size, cp_size, cp_split_sizes
        )

    @staticmethod
    def get_attn_and_xattn_base(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        core_attn_func: Callable,
        cross_attn_func: Callable,
        overlap_degree: int,
        batch_size: int,
        cp_size: int,
        cp_split_sizes: List[int] = None,
    ):
        """Run core attention in ``overlap_degree`` head chunks, then cross-attn.

        Query is split so each chunk's output all-to-all can overlap the next
        chunk's compute. Cross-attention runs after the last core kernel is
        launched, hiding the final restore collective. ``overlap_degree=-1``
        means one query-head group per KV head (GQA/MQA).
        """
        # Split Query, Key, Value into multiple parts
        # k/v may have different sequence length with q due to kv cache
        q_seq, q_head, q_hidden = query.shape
        kv_seq, kv_head, kv_hidden = key.shape
        if overlap_degree == -1:
            overlap_degree = q_head // kv_head
        else:
            assert overlap_degree <= q_head

        if overlap_degree == 1:
            query = [query]
        elif kv_head == 1:  # MQA
            query = query.chunk(overlap_degree, dim=1)
        else:  # GQA
            assert q_head % (overlap_degree * kv_head) == 0
            query = query.reshape(q_seq, kv_head, -1, q_hidden)
            query = query.chunk(overlap_degree, dim=2)
            query = [q.reshape(q_seq, -1, q_hidden) for q in query]

        # Compute Core Attention
        handle_attn = None
        core_attn_out = None
        core_attn_outs = []
        for i in range(overlap_degree):
            core_attn_out_new = core_attn_func(query[i], key, value)
            if handle_attn is not None:
                handle_attn.wait()
                core_attn_outs.append(core_attn_out)
            core_attn_out, handle_attn = all_to_all_output_split(core_attn_out_new, cp_split_sizes)

        xattn_out = cross_attn_func()
        handle_attn.wait()
        core_attn_outs.append(core_attn_out)
        core_attn_out = torch.cat(core_attn_outs, dim=1)

        core_attn_out = _rearrange(core_attn_out, "(cp sq b) hn hd -> (sq) b (cp hn hd)", cp=cp_size, b=batch_size)
        return core_attn_out, xattn_out


# ──────────────────────────────────────────────────────────────────────────
# CSO pipeline — shuffle-overlap all-to-all for older GPUs
# ──────────────────────────────────────────────────────────────────────────


def cso_communication(
    input: torch.Tensor, cp_world_size: int, cp_split_sizes: List[int], comm_type: str = None
) -> Tuple[torch.Tensor, torch.distributed.Work]:
    """Launch one context-shuffle-overlap all-to-all operation.

    Args:
        input: Local sequence/head tensor.
        cp_world_size: Context-parallel group size.
        cp_split_sizes: Per-rank split sizes.
        comm_type: ``"kv"`` additionally reshapes KV heads before exchange.

    Returns:
        Output buffer and asynchronous work handle. For one rank, the input and
        a completed fake handle are returned.
    """
    if cp_world_size == 1:
        return input, FakeHandle()
    assert cp_split_sizes is not None
    _, hn, _ = input.shape
    if comm_type == "kv":
        if cp_world_size % hn == 0 and cp_world_size != hn:
            input = torch.repeat_interleave(input, repeats=divide(cp_world_size, hn), dim=1)
        input = _rearrange(input, "spb (cp hn) hd -> (cp spb) hn hd", cp=cp_world_size).contiguous()
    output = torch.empty(input.shape, device=input.device, dtype=input.dtype)

    handle = torch.distributed.all_to_all_single(
        output, input, input_split_sizes=cp_split_sizes, group=mpu.get_cp_group(), async_op=True
    )

    return output, handle


class CSOHelper:
    """Pipeline chunked query exchange with attention computation.

    The helper splits queries into ``cp_shuffle_num`` chunks, starts the first
    asynchronous exchange, and rotates later query/output chunks so
    communication can overlap with ``fattn`` execution.
    """

    def __init__(self, cp_shuffle_num, cp_world_size, cp_split_sizes):
        """Record shuffle count and per-chunk split sizes (split sizes ÷ shuffle)."""
        self.cp_shuffle_num = cp_shuffle_num
        self.cp_world_size = cp_world_size
        self.cp_split_sizes = [divide(x, self.cp_shuffle_num) for x in cp_split_sizes]

    def split_query_for_overlap(self, query):
        """Reshape queries into shuffle chunks and start the first async exchange."""
        query = _rearrange(
            query, "(dn spb) (cp hn) hd -> (dn cp spb) hn hd", cp=self.cp_world_size, dn=self.cp_shuffle_num
        ).contiguous()
        querys = list(torch.chunk(query, self.cp_shuffle_num, dim=0))
        querys[0], handle_q = cso_communication(querys[0], self.cp_world_size, self.cp_split_sizes)
        return querys, handle_q

    def overlap(self, fattn, qs, k, v):
        """Rotate query/output all-to-alls around ``fattn`` so comm hides compute.

        When ``cp_shuffle_num>1``, later iterations pack the previous output
        with the next query so one collective restores output while fetching
        the next query. The last iteration issues a dedicated output exchange.
        """
        core_attn_outs = []
        o = None
        for i in range(self.cp_shuffle_num):
            if self.cp_shuffle_num == 1:
                q = qs[0]
            elif i == 0:
                q = qs[0]
                loop_var, loop_handle = cso_communication(qs[i + 1], self.cp_world_size, self.cp_split_sizes)
            else:
                loop_handle.wait()
                assert o is not None
                if loop_var.numel() == qs[0].numel():
                    q = loop_var
                else:
                    assert loop_var.numel() == qs[0].numel() * 2
                    q, ready_o = torch.chunk(loop_var, 2, dim=-1)
                    core_attn_outs.append(ready_o)
                loop_var = torch.concat([qs[i + 1], o], dim=-1) if i < self.cp_shuffle_num - 1 else o
                loop_var, loop_handle = cso_communication(loop_var, self.cp_world_size, self.cp_split_sizes)

            o = fattn(q, k, v, i)
            if i == self.cp_shuffle_num - 1:
                if i != 0:
                    loop_handle.wait()
                    assert loop_var.numel() == qs[0].numel()
                    core_attn_outs.append(loop_var)
                last_o, handle_attn = cso_communication(o, self.cp_world_size, self.cp_split_sizes)
                core_attn_outs.append(last_o)
        return core_attn_outs, handle_attn

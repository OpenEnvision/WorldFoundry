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

# from neophilia; Author: Qsh (qsh.zh27@gmail.com) Kaiwen Zheng (zkwthu@gmail.com)

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed import ProcessGroup

from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.attention import attention
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.kv_cache import AttnContext, KVCacheMode
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.rope import apply_rope


def post_all2all(local_seq_2_local_head, seq_world_size):
    def post_func(input):
        # b, s, n, h
        if local_seq_2_local_head:
            output = rearrange(input, "w bs seq h d -> bs (w seq) h d")
        else:
            output = rearrange(input, "w bs s h d -> bs s (w h) d", w=seq_world_size)

        return output

    return post_func


def single_all_to_all(input, local_seq_2_local_head, group, async_op=False):
    seq_world_size = dist.get_world_size(group)

    # b, s, n, h
    if local_seq_2_local_head:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert (
            num_total_head % seq_world_size == 0
        ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
        input_t = rearrange(input, "bs seq_len (w h) d -> w bs seq_len h d", w=seq_world_size, h=num_total_head // seq_world_size).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)
    else:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        input_t = rearrange(input, "bs (w s) h d -> w bs s h d", w=seq_world_size, s=global_seq_len // seq_world_size).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    res = post_all2all_fun(output)
    return res


def async_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    cp_size: int,
    cp_group: ProcessGroup,
    cp_stream: torch.cuda.Stream,
    local_seq_2_local_head: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    A2A communication for context parallelism. best used in communicate qkv
    Modified from Nvidia Transformer Engine.
    """
    a2a_inputs = [a2a_inputs] if not isinstance(a2a_inputs, list) else a2a_inputs
    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    a2a_post_fns = [None] * len(a2a_inputs)
    if local_seq_2_local_head:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True)
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(a2a_inputs[i], "bs seq_len (w h) d -> w bs seq_len h d", w=cp_size).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True)
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(a2a_inputs[i], "bs (w s) h d -> w bs s h d", w=cp_size).contiguous()
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: torch.Tensor, local_seq_2_local_head: bool) -> torch.Tensor:
        ctx.group = group
        res = single_all_to_all(input, local_seq_2_local_head, group, False)
        ctx.local_seq_2_local_head = local_seq_2_local_head
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, not ctx.local_seq_2_local_head), None)


class _SeqAllToAllQKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        q, k, v = async_a2a_communicate([q, k, v], cp_size, group, cp_stream, local_seq_2_local_head)
        return q, k, v

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q_grad, k_grad, v_grad = _SeqAllToAllQKV.apply(ctx.group, *grad_output, ctx.cp_size, ctx.cp_stream, not ctx.local_seq_2_local_head)
        return (None, q_grad, k_grad, v_grad, None, None, None)


class DistributedAttention(torch.nn.Module):
    """Wraps a local attention op with optional Ulysses-style CP (A2A) + optional KV cache."""

    def __init__(self, local_attention: Union[torch.nn.Module, Callable]) -> None:
        super().__init__()
        self.local_attn = local_attention
        self.pg = None
        self.stream = None

    def set_context_parallel_group(self, group, stream):
        self.pg = group
        self.stream = stream

    def _materialize_kv(self, key: torch.Tensor, value: torch.Tensor, attn_ctx: Optional[AttnContext]) -> Tuple[torch.Tensor, torch.Tensor]:
        if attn_ctx is None or attn_ctx.kv_cache is None:
            return key, value

        cache = attn_ctx.kv_cache
        if attn_ctx.mode == KVCacheMode.DISABLED:
            return key, value
        if attn_ctx.mode == KVCacheMode.APPEND:
            if attn_ctx.block_range is not None:
                # `committed_blocks` equals len(cache._cum_ends) for the plain cache;
                # an evicting cache reports blocks committed rather than retained.
                assert attn_ctx.block_range == cache.committed_blocks
            cache.append(key, value)
            return cache.get()
        if attn_ctx.mode == KVCacheMode.READONLY:
            if attn_ctx.fast_infer:
                prefix_end = cache.get_prefix_end(attn_ctx.block_range)
                if prefix_end == 0:
                    return key, value
                return cache.write_transient(key, value, prefix_end)
            k_cached, v_cached = cache.get(attn_ctx.block_range)
            if k_cached is None:
                return key, value
            return torch.cat([k_cached, key], dim=1), torch.cat([v_cached, value], dim=1)
        raise ValueError(f"Unknown KVCacheMode: {attn_ctx.mode}")

    def _collect_attention_stats(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_current: torch.Tensor,
        attn_ctx: Optional[AttnContext],
    ):
        if attn_ctx is None or attn_ctx.attn_observer is None:
            return
        cached_len = key.shape[1] - key_current.shape[1]
        if cached_len <= 0:
            return
        attn_ctx.attn_observer.observe(
            layer_idx=attn_ctx.layer_idx,
            q_block_idx=attn_ctx.q_block_idx,
            query=query.detach(),
            key=key.detach(),
            cached_len=cached_len,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, attn_ctx: Optional[AttnContext] = None, **kwargs
    ) -> torch.Tensor:
        """
        query/key/value expected layout (before CP):
          [B, S_local, H_total, D]
        If CP enabled, A2A converts Q/K/V to:
          [B, S_global (=cp*S_local), H_local (=H_total/cp), D]
        KV cache is stored in the SAME layout as local_attn sees.
        """

        pg_size = 1 if self.pg is None else dist.get_world_size(self.pg)
        if pg_size > 1:
            # A2A (Q/K/V): local_seq->global_seq, total_heads->local_heads
            query, key, value = _SeqAllToAllQKV.apply(self.pg, query, key, value, pg_size, self.stream, True)

        key_current = key
        rope = attn_ctx.rope if attn_ctx is not None else None

        if rope is not None and rope.cached_k_rotated and rope.current_key_freqs is not None:
            key = apply_rope(key, rope.current_key_freqs, fused=rope.use_fused)

        if rope is not None and rope.query_freqs is not None:
            query = apply_rope(query, rope.query_freqs, fused=rope.use_fused)

        if getattr(self.local_attn, "manages_kv_cache", False):
            if getattr(self.local_attn, "accepts_attn_ctx", False):
                out = self.local_attn(query, key, value, *args, attn_ctx=attn_ctx, **kwargs)
            else:
                out = self.local_attn(query, key, value, *args, **kwargs)
        else:
            key, value = self._materialize_kv(key, value, attn_ctx)
            if rope is not None and not rope.cached_k_rotated:
                key = apply_rope(key, rope.key_freqs, fused=rope.use_fused)

            self._collect_attention_stats(query, key, key_current, attn_ctx)
            out = self.local_attn(query, key, value, *args, attn_ctx=attn_ctx, **kwargs)

        if pg_size > 1:
            # A2A back: global_seq->local_seq, local_heads->total_heads
            out = _SeqAllToAll.apply(self.pg, out, False)
        return out


class MinimalA2AAttnOp(DistributedAttention):
    def __init__(self, local_attn=attention, *args, **kwargs):
        del args, kwargs
        super(MinimalA2AAttnOp, self).__init__(local_attn)

    def set_context_parallel_group(self, process_group, ranks, stream):
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs) -> torch.Tensor:
        results = super().forward(query, key, value, *args, **kwargs)
        return rearrange(results, "b ... h l -> b ... (h l)")

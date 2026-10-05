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

# from neophilia; Author: Qsh (qsh.zh27@gmail.com)
# MIT License

"""Context-parallel all-to-all attention wrappers for Gamma-World.

Splits the sequence dim across ranks, all-to-alls into a head-sharded
layout, runs local attention, then reverses.  Tensors are ``B S H D``.
"""

import os
from typing import Any, Callable, List, Union

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module

from worldfoundry.base_models.diffusion_model.models.networks.gamma_world.attention import attention
from worldfoundry.base_models.diffusion_model.models.networks.gamma_world.neighborhood_attention import NeighborhoodAttention


def post_all2all(local_seq_2_local_head, seq_world_size):
    """Build the inverse rearrange after an A2A: heads-local ↔ sequence-local."""

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
        assert num_total_head % seq_world_size == 0, (
            f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
        )
        input_t = rearrange(
            input, "bs seq_len (w h) d -> w bs seq_len h d", w=seq_world_size, h=num_total_head // seq_world_size
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)
    else:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        input_t = rearrange(
            input, "bs (w s) h d -> w bs s h d", w=seq_world_size, s=global_seq_len // seq_world_size
        ).contiguous()
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
    if len(a2a_inputs) > 1:
        first = a2a_inputs[0]
        compatible = all(
            value.shape == first.shape
            and value.dtype == first.dtype
            and value.device == first.device
            for value in a2a_inputs[1:]
        )
        try:
            max_bytes = max(
                int(float(os.getenv("WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB", "512") or "512") * 1024**2),
                0,
            )
        except ValueError:
            max_bytes = 512 * 1024**2
        total_bytes = sum(value.numel() * value.element_size() for value in a2a_inputs)
        if compatible and max_bytes > 0 and total_bytes <= max_bytes:
            batch_size = first.shape[0]
            packed = torch.cat(a2a_inputs, dim=0)
            exchanged = single_all_to_all(packed, local_seq_2_local_head, cp_group)
            return list(exchanged.split(batch_size, dim=0))

    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    a2a_post_fns = [None] * len(a2a_inputs)
    if local_seq_2_local_head:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(
                    a2a_inputs[i], "bs seq_len (w h) d -> w bs seq_len h d", w=cp_size
                ).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(a2a_inputs[i], "bs (w s) h d -> w bs s h d", w=cp_size).contiguous()
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(
        self,
        local_attention: Union[Module, Callable],
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.pg = None
        self.stream = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        if self.pg is None:
            return self.local_attn(query, key, value, *args, **kwargs)
        pg_size = dist.get_world_size(self.pg)
        if pg_size < 2:
            return self.local_attn(query, key, value, *args, **kwargs)

        query_layer, key_layer, value_layer = async_a2a_communicate(
            [query, key, value], pg_size, self.pg, self.stream, True
        )
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)

        output = single_all_to_all(context_layer, False, self.pg)
        return output

    def set_context_parallel_group(self, group, stream):
        self.pg = group
        self.stream = stream


class MinimalA2AAttnOp(DistributedAttention):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        super(MinimalA2AAttnOp, self).__init__(attention)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        results = super().forward(query, key, value, *args, **kwargs)
        return rearrange(results, "b ... h l -> b ... (h l)")


class NattenA2AAttnOp(MinimalA2AAttnOp):
    def __init__(self, *args, **kwargs):
        super(NattenA2AAttnOp, self).__init__(None)
        self.natten_op = NeighborhoodAttention(*args, **kwargs, base_attn_op=attention)
        self.local_attn = self.natten_op

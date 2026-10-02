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

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.config import (
    FSDP2ModelConfig as JobConfig,
)

# from torchtitan.config_manager import TORCH_DTYPE_MAP
# from torchtitan.logging import logger
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.parallel_dims import ParallelDims
from worldfoundry.core.distributed.runtime.logging import log as logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parallelize_qwen(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply inference tensor/data parallelism and optional compilation.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if job_config.experimental.enable_async_tensor_parallel and not job_config.parallel.compile:
            raise RuntimeError("Async TP requires parallel.compile")
        apply_tp(
            model,
            world_mesh["tp"],
            enable_float8=job_config.float8.enable_float8_linear,
            enable_async_tp=job_config.experimental.enable_async_tensor_parallel,
        )

    if job_config.parallel.compile:
        apply_compile(model)

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.parallel.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_float8: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,  # Output Dtensor
            ),
            "model.norm": SequenceParallel(),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears
    if enable_float8:
        # add a check here to enforce supported float8 all-gather configurations

        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.model.layers:
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(
                    Shard(1),  # hidden_states
                    None,  # attention_mask
                    None,  # position_ids
                    None,  # past_key_value
                    None,  # output_attentions
                    None,  # use_cache
                    None,  # cache_position
                    None,  # position_embeddings
                ),
                desired_input_layouts=(
                    Replicate(),
                    None,  # attention_mask
                    None,  # position_ids
                    None,  # past_key_value
                    None,  # output_attentions
                    None,  # use_cache
                    None,  # cache_position
                    None,  # position_embeddings),
                ),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }
        # map the name from llama to qwen
        names_mapping = {
            "attention_norm": "input_layernorm",
            "attention": "self_attn",
            "attention.wq": "self_attn.q_proj",
            "attention.wk": "self_attn.k_proj",
            "attention.wv": "self_attn.v_proj",
            "attention.wo": "self_attn.o_proj",
            "ffn_norm": "post_attention_layernorm",  # Norm after attention, before feed_forward
            "feed_forward": "mlp",
            "feed_forward.w1": "mlp.gate_proj",
            "feed_forward.w2": "mlp.down_proj",
            "feed_forward.w3": "mlp.up_proj",
        }
        new_layer_plan = {}
        for key, value in layer_plan.items():
            new_layer_plan[names_mapping[key]] = value
        del layer_plan
        layer_plan = new_layer_plan

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
    """
    if model.visual is not None:
        for layer_id, block in enumerate(model.visual.blocks):
            fully_shard(block, mesh=dp_mesh)

    for layer_id, transformer_block in enumerate(model.model.layers):
        fully_shard(
            transformer_block,
            mesh=dp_mesh,
        )
    fully_shard(model, mesh=dp_mesh)

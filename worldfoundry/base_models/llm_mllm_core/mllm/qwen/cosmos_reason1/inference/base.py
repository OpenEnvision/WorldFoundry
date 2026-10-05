"""Inference-only base for the bundled Reason1/Qwen text encoder."""

import os
from typing import Any, Dict, Optional

import torch
from torch._utils import _get_available_device_type, _get_device_module
from torch.nn.modules.module import _IncompatibleKeys

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.config import FSDP2ModelConfig
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.parallel_dims import ParallelDims
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.tokenizer import Processor
from worldfoundry.core.distributed.runtime import torch_process_group as distributed


class VLMBaseModel(torch.nn.Module):
    def __init__(self, model_config: FSDP2ModelConfig, tokenizer: Processor):
        super().__init__()
        original_precision = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, model_config.precision))
        self.tokenizer = tokenizer
        self.config = model_config
        self.precision = getattr(torch, model_config.precision)
        self.build_model(model_config)
        torch.set_default_dtype(original_precision)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        missing, unexpected = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict and (missing or unexpected):
            raise ValueError(f"Missing keys: {missing}\n\nUnexpected keys: {unexpected}")
        return _IncompatibleKeys(missing, unexpected)

    @torch.no_grad()
    def init_weights(self, buffer_device: Optional[torch.device] = None):
        if self.config.model_type == "qwen2_5":
            for module in self.model.modules():
                if hasattr(module, "rope_init_fn"):
                    buffer, module.attention_scaling = module.rope_init_fn(module.config, device=None)
                    module.register_buffer("inv_freq", buffer.to(torch.cuda.current_device()), persistent=False)
        else:
            self.model.init_weights(buffer_device)
        if self.vision_encoder is not None and not self.config.vision_encoder.startswith("siglip"):
            self.vision_encoder.init_weights(buffer_device)
        if self.mm_projector is not None:
            self.mm_projector.init_weights()

    @property
    def cp_mesh(self):
        return None

    @property
    def tp_mesh(self):
        return None

    def build_model(self, model_config):
        raise NotImplementedError

    def forward(self, tokens, data_batch=None, start_pos: int = 0) -> torch.Tensor:
        del tokens, data_batch, start_pos
        raise NotImplementedError


def init_mesh(model_config):
    parallel_dims = ParallelDims(
        dp_shard=model_config.parallel.data_parallel_shard_degree,
        dp_replicate=model_config.parallel.data_parallel_replicate_degree,
        cp=model_config.parallel.context_parallel_degree,
        tp=model_config.parallel.tensor_parallel_degree,
        pp=model_config.experimental.pipeline_parallel_degree,
        world_size=distributed.get_world_size(),
    )
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_type = _get_available_device_type() or "cuda"
    _get_device_module(device_type).set_device(torch.device(f"{device_type}:{local_rank}"))
    return parallel_dims.build_mesh(device_type=device_type), parallel_dims


__all__ = ["VLMBaseModel", "init_mesh"]

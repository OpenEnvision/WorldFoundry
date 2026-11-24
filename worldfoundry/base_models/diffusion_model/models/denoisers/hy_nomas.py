"""Hy-Nomas checkpoint mapping and native denoiser contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ...optimizations import OffloadPolicy, parse_quantization_policy
from ..networks.hy_nomas import HyNomasTransformer3DModel


def convert_hy_nomas_checkpoint(state: Mapping[str, Any]) -> dict[str, Any]:
    """Map the published split projections without altering tensor values."""
    result = {}
    for name, value in state.items():
        name = re.sub(r"^(double_blocks\.\d+)\.((?:img|txt)_attn_)", r"\1.attn.\2", name)
        name = re.sub(
            r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.(self_attn_(?:qkv|proj))",
            r"\1.attn.\2",
            name,
        )
        if name in result:
            raise ValueError(f"duplicate Hy-Nomas checkpoint parameter: {name}")
        result[name] = value
    return result


class HyNomasDenoiser:
    """Expose velocity prediction; request-owned caches arrive through conditioning."""

    def __init__(self, model: HyNomasTransformer3DModel, *, low_memory: bool = False) -> None:
        self.model = model
        self.low_memory = low_memory
        self.num_cache_layers = len(model.double_blocks)
        self.cache_head_dim = int(model.config.attention_head_dim)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        inputs = model_input
        values = dict(inputs.conditioning)
        values.pop("kv_cache_quantization", None)
        values.pop("max_cache_segments", None)
        dtype = next(self.model.parameters()).dtype
        sample = self.model(
            hidden_states=inputs.latents.to(dtype=dtype),
            timestep=inputs.timestep.reshape(1).expand(inputs.latents.shape[0]).float(),
            **values,
            return_dict=False,
        )[0]
        return DenoiserOutput(sample=sample)


def build_hy_nomas_denoiser(context: ComponentBuildContext) -> HyNomasDenoiser:
    """Load through the canonical loader, including native weight-only INT8."""
    options = dict(context.component_options)
    low_memory = bool(options.pop("low_memory", False))
    config = dict(cast(Mapping[str, Any], options.pop("config", {})))
    policy = replace(
        context.policy,
        offload=OffloadPolicy(),
        device="cpu" if low_memory else context.policy.device,
        quantization=(
            parse_quantization_policy(
                {
                    "mode": "int8",
                    "options": {
                        "keep_dense_fallback": False,
                        "use_kernel": False,
                        "group_size": 128,
                    },
                }
            )
            if low_memory
            else context.policy.quantization
        ),
    )
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HyNomasTransformer3DModel,
            config=config,
            state_dict_converter=convert_hy_nomas_checkpoint,
            layer_container="double_blocks",
        ),
        context.require_checkpoint(),
        policy,
    )
    for module in model.modules():
        processor = getattr(module, "processor", None)
        if processor is not None:
            processor._attention_backend = str(context.policy.attention.value)
    return HyNomasDenoiser(cast(HyNomasTransformer3DModel, model), low_memory=low_memory)

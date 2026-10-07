"""WorldPlay2's dual experts use the shared module loader and runtime policy."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.base_models.diffusion_model.contracts import DenoiserOutput
from worldfoundry.base_models.diffusion_model.loaders import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WAN22_I2V_A14B_CONFIG

from .modeling.model import WorldPlay2Model
from .scheduler import FixedPDD4Scheduler


class _WorldPlay2StateDict(Mapping):
    """Strip the released expert's model prefix without reading tensor bodies."""

    def __init__(self, state):
        self.state = state
        self.names = {name.removeprefix("model."): name for name in state}

    def __getitem__(self, name):
        return self.state[self.names[name]]

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def convert_worldplay2_state_dict(state, *, expert, fixed_pdd):
    converted = _WorldPlay2StateDict(state)
    if fixed_pdd:
        converted = FixedPDD4Scheduler().convert_state_dict(converted, expert=expert)
    return converted


class WorldPlay2Denoiser:
    def __init__(self, high, low, *, mode, dtype):
        self.models = {"high": high, "low": low}
        self.mode = mode
        self.dtype = dtype

    def new_cache(self):
        return {name: {branch: model.new_cache() for branch in ("positive", "negative")}
                for name, model in self.models.items()}

    def __call__(self, model_input):
        conditions = model_input.conditioning
        expert = str(conditions["expert"])
        model = self.models[expert]
        with torch.autocast(device_type=model_input.latents.device.type, dtype=self.dtype,
                            enabled=self.dtype in {torch.float16, torch.bfloat16}):
            output = model(
                model_input.latents, model_input.timestep.reshape(1), conditions["context"],
                y=conditions["condition_latents"], actions=conditions["actions"],
                memory=conditions.get("memory"), cache=conditions.get("cache"),
                chunk_start=int(conditions["chunk_start"]),
                **({"pdd_block_index": int(conditions["pdd_block_index"])} if self.mode == "few_step" else {}),
            )
        if self.mode == "few_step":
            displacement, endpoint = output.float().chunk(2, dim=1)
            return DenoiserOutput(displacement, {"endpoint_velocity": endpoint, "expert": expert})
        return DenoiserOutput(output.float(), {"expert": expert})

    def prepare_memory(self, *, memory_inputs, contexts, negative_context, caches):
        memories = {}
        for name, model in self.models.items():
            with torch.autocast(device_type=memory_inputs["hr_input"].device.type, dtype=self.dtype,
                                enabled=self.dtype in {torch.float16, torch.bfloat16}):
                memory = model(operation="compress", **memory_inputs)
                memories[name] = memory
                if caches is not None:
                    model(operation="prefill", context=contexts, memory=memory, cache=caches[name]["positive"])
                    if negative_context is not None:
                        model(operation="prefill", context=negative_context, memory=memory, cache=caches[name]["negative"])
        return memories


def build_worldplay2_denoiser(context):
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    mode = str(context.recipe_options["mode"])
    config = {**WAN22_I2V_A14B_CONFIG, "fixed_pdd": mode == "few_step"}
    config.update(dict(context.component_options.get("model_config", {})))
    models = {}
    for expert in ("high", "low"):
        models[expert] = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=WorldPlay2Model, config=config,
                state_dict_converter=lambda state, name=expert: convert_worldplay2_state_dict(
                    state, expert=name, fixed_pdd=mode == "few_step",
                ),
                vram_module_map={torch.nn.Linear: AutoWrappedLinear, torch.nn.Conv3d: AutoWrappedModule,
                                 torch.nn.Conv2d: AutoWrappedModule, torch.nn.LayerNorm: AutoWrappedModule,
                                 torch.nn.Embedding: AutoWrappedModule},
                layer_container="blocks",
            ),
            context.require_checkpoint(expert), context.policy,
        )
    return WorldPlay2Denoiser(models["high"], models["low"], mode=mode, dtype=context.policy.dtype)


__all__ = ["WorldPlay2Denoiser", "build_worldplay2_denoiser", "convert_worldplay2_state_dict"]

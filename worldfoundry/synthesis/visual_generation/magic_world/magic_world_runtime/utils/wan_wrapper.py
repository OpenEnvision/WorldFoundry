import os
from typing import List, Optional

import torch
from omegaconf import OmegaConf

from worldfoundry.base_models.diffusion_model.schedulers import (
    InferenceFlowMatchScheduler,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.resident import (
    WanVAEWrapper as _WanVAEWrapper,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.resident import (
    WanTextEncoder as _WanTextEncoder,
)
from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.magic_world.causal import CausalWanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.magic_world.model import WanModel
from worldfoundry.base_models.diffusion_model.loaders.wan_variant import (
    load_wan_transformer,
)


_WAN_ROOT = os.environ.get(
    "WORLDFOUNDRY_MAGICWORLD_WAN_ROOT",
    "checkpoints/Wan2.1-Fun-V1.1-1.3B-InP",
)
_MAGICWORLD_BASE_ROOT = os.environ.get(
    "WORLDFOUNDRY_MAGICWORLD_BASE_ROOT",
    "checkpoints/MagicWorld/MagicWorld-Base",
)
_WAN_CONFIG = os.environ.get(
    "WORLDFOUNDRY_MAGICWORLD_CONFIG",
    str(
        resolve_data_path(
            "models",
            "runtime",
            "configs",
            "video_x_fun",
            "wan2.1",
            "wan_civitai.yaml",
        )
    ),
)


class WanTextEncoder(_WanTextEncoder):
    def __init__(self) -> None:
        super().__init__(_WAN_ROOT)


class WanVAEWrapper(_WanVAEWrapper):
    def __init__(self) -> None:
        super().__init__(_WAN_ROOT)


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name=_MAGICWORLD_BASE_ROOT,
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        weight_dtype = torch.bfloat16
        config = OmegaConf.load(_WAN_CONFIG)
        transformer_subpath = config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')
        transformer_root = os.path.join(model_name, transformer_subpath)
        if not os.path.isdir(transformer_root):
            direct_config = os.path.join(model_name, "config.json")
            direct_weights = os.path.join(model_name, "diffusion_pytorch_model.safetensors")
            if os.path.isfile(direct_config) and os.path.isfile(direct_weights):
                transformer_root = model_name

        if is_causal:
            self.model = load_wan_transformer(
                CausalWanModel,
                transformer_root,
                additional_kwargs=OmegaConf.to_container(
                    config["transformer_additional_kwargs"], resolve=True
                ),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        else:
            self.model = load_wan_transformer(
                WanModel,
                transformer_root,
                additional_kwargs=OmegaConf.to_container(
                    config["transformer_additional_kwargs"], resolve=True
                ),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = InferenceFlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        clip_fea = conditional_dict["clip_fea"]
        y = conditional_dict["y"]
        y_camera = conditional_dict["y_camera"]
        y_history = conditional_dict["y_history"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                #### other condition ####
                y=y,
                y_camera=y_camera,
                clip_fea=clip_fea,
                y_history=y_history,
                ####
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                y=y,
                y_camera=y_camera,
                clip_fea=clip_fea,
                y_history=y_history,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self.scheduler.flow_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> InferenceFlowMatchScheduler:
        return self.scheduler

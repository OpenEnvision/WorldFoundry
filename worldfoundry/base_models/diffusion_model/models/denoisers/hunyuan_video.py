"""Denoiser adapters for original and 1.5 HunyuanVideo checkpoints.

Each class implements :class:`~...contracts.Denoiser`.  Factories load
weights through :class:`~...loaders.module.NativeModuleLoader`.  Required
conditioning keys (``text_states``, masks, pooled embeds) raise
:exc:`KeyError` / :exc:`TypeError` when missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config
from ...optimizations import AttentionBackend
from ..networks.hunyuan_video.h15.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from ..networks.hunyuan_video.i2v.model import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer
from ..networks.hunyuan_video.original import HunyuanVideoDiT
from .hunyuan_video_converter import HunyuanVideoDiTStateDictConverter


def _required_tensor(values: Mapping[str, object], key: str) -> torch.Tensor:
    try:
        value = values[key]
    except KeyError as error:
        raise KeyError(f"HunyuanVideo denoising requires conditioning {key!r}") from error
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"HunyuanVideo conditioning {key!r} must be a tensor")
    return value


def _batch_timestep(value: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    timestep = value.to(device=latents.device, dtype=latents.dtype).reshape(-1)
    if timestep.numel() == 1:
        timestep = timestep.expand(latents.shape[0])
    if timestep.numel() != latents.shape[0]:
        raise ValueError("HunyuanVideo timestep must be scalar or have one value per sample")
    return timestep


def _embedded_guidance(model_input: DenoiserInput, *, multiply: float = 1.0) -> torch.Tensor:
    scale = float(model_input.conditioning.get("embedded_guidance_scale", 6.0)) * multiply
    return torch.full(
        (model_input.latents.shape[0],),
        scale,
        device=model_input.latents.device,
        dtype=model_input.latents.dtype,
    )


class HunyuanVideoDenoiser:
    """Original HunyuanVideo T2V denoiser with embedded guidance."""

    def __init__(self, model: HunyuanVideoDiT) -> None:
        self.model = model
        self._rope: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        key = tuple(model_input.latents.shape[2:])
        freqs = self._rope.get(key)
        if freqs is None:
            freqs = self.model.prepare_freqs(model_input.latents)
            self._rope[key] = freqs
        cos, sin = (value.to(device=model_input.latents.device) for value in freqs)
        compute_dtype = next(self.model.parameters()).dtype
        with torch.autocast(
            device_type=model_input.latents.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            sample = self.model(
                x=model_input.latents,
                t=_batch_timestep(model_input.timestep, model_input.latents),
                prompt_emb=_required_tensor(model_input.conditioning, "text_states"),
                text_mask=_required_tensor(model_input.conditioning, "text_mask"),
                pooled_prompt_emb=_required_tensor(model_input.conditioning, "text_states_2"),
                freqs_cos=cos,
                freqs_sin=sin,
                guidance=_embedded_guidance(model_input),
            )
        return DenoiserOutput(sample=sample)


class HunyuanVideoI2VDenoiser:
    """Original token-replacement I2V denoiser with a frozen first latent frame."""

    def __init__(self, model: HYVideoDiffusionTransformer, *, rope_theta: float = 256.0) -> None:
        self.model = model
        self.rope_theta = float(rope_theta)
        self._rope: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def _freqs(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(latents.shape[2:])
        values = self._rope.get(key)
        if values is None:
            from worldfoundry.core.attention import get_nd_rotary_pos_embed

            sizes = tuple(size // patch for size, patch in zip(key, self.model.patch_size))
            values = get_nd_rotary_pos_embed(
                self.model.rope_dim_list,
                sizes,
                theta=self.rope_theta,
                use_real=True,
                theta_rescale_factor=1.0,
            )
            self._rope[key] = values
        return tuple(value.to(device=latents.device) for value in values)  # type: ignore[return-value]

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        cos, sin = self._freqs(model_input.latents)
        compute_dtype = next(self.model.parameters()).dtype
        with torch.autocast(
            device_type=model_input.latents.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            output = self.model(
                x=model_input.latents,
                t=_batch_timestep(model_input.timestep, model_input.latents),
                text_states=_required_tensor(model_input.conditioning, "text_states"),
                text_mask=_required_tensor(model_input.conditioning, "text_mask"),
                text_states_2=_required_tensor(model_input.conditioning, "text_states_2"),
                freqs_cos=cos,
                freqs_sin=sin,
                guidance=_embedded_guidance(model_input, multiply=1000.0),
                return_dict=False,
            )
        sample = output.clone()
        sample[:, :, :1] = 0
        return DenoiserOutput(sample=sample)


class HunyuanVideo15Denoiser:
    """HunyuanVideo 1.5 denoiser using native concat-condition inputs."""

    def __init__(self, model: HunyuanVideo_1_5_DiffusionTransformer, *, image_to_video: bool) -> None:
        self.model = model
        self.image_to_video = bool(image_to_video)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        condition = _required_tensor(model_input.conditioning, "condition_latents").to(model_input.latents)
        if condition.shape[0] != model_input.latents.shape[0] or condition.shape[2:] != model_input.latents.shape[2:]:
            raise ValueError("HunyuanVideo 1.5 condition latent geometry must match the noise trajectory")
        hidden_states = torch.cat((model_input.latents, condition), dim=1)
        vision = model_input.conditioning.get("vision_states")
        if vision is None:
            vision = torch.zeros(
                hidden_states.shape[0], 729, 1152,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
        if not isinstance(vision, torch.Tensor):
            raise TypeError("HunyuanVideo 1.5 vision_states must be a tensor")
        extra_kwargs = {
            "byt5_text_states": _required_tensor(model_input.conditioning, "byt5_text_states"),
            "byt5_text_mask": _required_tensor(model_input.conditioning, "byt5_text_mask"),
        }
        timestep = _batch_timestep(model_input.timestep, model_input.latents)
        timestep_r = None
        if bool(self.model.config.use_meanflow):
            timestep_r = _batch_timestep(model_input.next_timestep, model_input.latents)
        compute_dtype = next(self.model.parameters()).dtype
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            output, _ = self.model(
                hidden_states=hidden_states,
                timestep=timestep,
                timestep_r=timestep_r,
                text_states=_required_tensor(model_input.conditioning, "text_states"),
                text_states_2=model_input.conditioning.get("text_states_2"),
                encoder_attention_mask=_required_tensor(model_input.conditioning, "text_mask"),
                vision_states=vision.to(device=hidden_states.device, dtype=hidden_states.dtype),
                guidance=_embedded_guidance(model_input, multiply=1000.0),
                mask_type="i2v" if self.image_to_video else "t2v",
                extra_kwargs=extra_kwargs,
                return_dict=False,
            )
        return DenoiserOutput(sample=output)


def _original_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return HunyuanVideoDiTStateDictConverter().from_civitai(state_dict)


def build_hunyuan_video_denoiser(context: ComponentBuildContext) -> HunyuanVideoDenoiser:
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HunyuanVideoDiT,
            state_dict_converter=_original_state_dict,
            layer_container="double_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, HunyuanVideoDiT):
        raise TypeError(f"expected HunyuanVideoDiT, got {type(model).__name__}")
    return HunyuanVideoDenoiser(model)


def build_hunyuan_video_i2v_denoiser(context: ComponentBuildContext) -> HunyuanVideoI2VDenoiser:
    args = SimpleNamespace(
        text_states_dim=4096,
        text_states_dim_2=768,
        i2v_condition_type="token_replace",
        gradient_checkpoint=False,
        gradient_checkpoint_layers=-1,
    )
    config = dict(HUNYUAN_VIDEO_CONFIG["HYVideo-T/2-cfgdistill"])
    config.update({"args": args, "in_channels": 16, "out_channels": 16})
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HYVideoDiffusionTransformer,
            config=config,
            layer_container="double_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, HYVideoDiffusionTransformer):
        raise TypeError(f"expected HYVideoDiffusionTransformer, got {type(model).__name__}")
    return HunyuanVideoI2VDenoiser(model)


def _h15_config(checkpoint, relative_path: str, attention: AttentionBackend) -> Mapping[str, object]:
    config = dict(checkpoint_json_config(checkpoint, relative_path))
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    config["attn_mode"] = "flash" if attention is AttentionBackend.FLASH else "torch"
    return config


def build_hunyuan_video15_denoiser(context: ComponentBuildContext) -> HunyuanVideo15Denoiser:
    config_path = str(context.component_options.get("config_path", "transformer/480p_t2v_distilled/config.json"))
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HunyuanVideo_1_5_DiffusionTransformer,
            config_resolver=lambda checkpoint: _h15_config(checkpoint, config_path, context.policy.attention),
            layer_container="double_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, HunyuanVideo_1_5_DiffusionTransformer):
        raise TypeError(f"expected HunyuanVideo 1.5 transformer, got {type(model).__name__}")
    return HunyuanVideo15Denoiser(
        model,
        image_to_video=bool(context.component_options.get("image_to_video", False)),
    )


__all__ = [
    "HunyuanVideo15Denoiser",
    "HunyuanVideoDenoiser",
    "HunyuanVideoI2VDenoiser",
    "build_hunyuan_video15_denoiser",
    "build_hunyuan_video_denoiser",
    "build_hunyuan_video_i2v_denoiser",
]

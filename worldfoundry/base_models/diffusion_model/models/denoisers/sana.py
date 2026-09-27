"""Native Sana denoiser roles.

The Sana family has several checkpoint graphs, but it does not own an
execution framework.  This module keeps the differences in immutable graph
configuration and exposes every graph through the shared :class:`Denoiser`
contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.sana.normalization import RMSNorm


def _base_image_config(
    *,
    input_size: int,
    hidden_size: int,
    depth: int,
    num_heads: int,
    qk_norm: bool = False,
    cross_norm: bool = False,
) -> dict[str, object]:
    return {
        "input_size": input_size,
        "patch_size": 1,
        "in_channels": 32,
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": 2.5,
        "class_dropout_prob": 0.1,
        "pred_sigma": False,
        "caption_channels": 2304,
        "model_max_length": 300,
        "qk_norm": qk_norm,
        "y_norm": True,
        "y_norm_scale_factor": 0.01,
        "attn_type": "linear",
        "ffn_type": "glumbconv",
        "mlp_acts": ("silu", "silu", None),
        # Official Sana 2K/4K configs enable sinusoidal position embeddings.
        # The 512/1024px image configs keep them disabled.
        "use_pe": input_size >= 64,
        "pe_interpolation": 2.0 if input_size >= 128 else 1.0,
        "linear_head_dim": 32,
        "cross_norm": cross_norm,
    }


def sana_image_config(
    *,
    input_size: int,
    parameter_scale: str,
    sana_1p5: bool = False,
) -> dict[str, object]:
    """Return the original checkpoint-compatible image graph config."""

    scale = str(parameter_scale).strip().lower()
    if scale in {"600m", "0.6b"}:
        hidden_size, depth, num_heads = 1152, 28, 16
    elif scale in {"1600m", "1.6b", "2b"}:
        hidden_size, depth, num_heads = 2240, 20, 20
    elif scale in {"4800m", "4.8b"}:
        hidden_size, depth, num_heads = 2240, 60, 20
    else:
        raise ValueError(f"unsupported Sana image parameter scale: {parameter_scale!r}")
    return _base_image_config(
        input_size=input_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        qk_norm=sana_1p5,
        cross_norm=sana_1p5,
    )


def sana_sprint_config(*, input_size: int, parameter_scale: str) -> dict[str, object]:
    config = sana_image_config(
        input_size=input_size,
        parameter_scale=parameter_scale,
        sana_1p5=True,
    )
    config.update(
        class_dropout_prob=0.0,
        cross_attn_type="vanilla",
        logvar=True,
        cfg_embed=True,
        cfg_embed_scale=0.1,
        timestep_norm_scale_factor=1000.0,
    )
    return config


def sana_sprint_teacher_config(*, input_size: int, parameter_scale: str) -> dict[str, object]:
    """Return the frozen SANA-Sprint teacher graph used by SCM-LADD."""

    config = sana_sprint_config(input_size=input_size, parameter_scale=parameter_scale)
    config.update(logvar=False, cfg_embed=False, cross_attn_type="flash")
    return config


def sana_controlnet_config(*, input_size: int, parameter_scale: str) -> dict[str, object]:
    config = sana_image_config(input_size=input_size, parameter_scale=parameter_scale)
    config["copy_blocks_num"] = 7
    return config


def sana_video_config(*, resolution: str, longlive: bool = False) -> dict[str, object]:
    selected = str(resolution).strip().lower()
    if selected == "480p":
        input_size = 60
        in_channels = 16
        patch_size = (1, 2, 2)
    elif selected == "720p":
        input_size = 22
        in_channels = 128
        patch_size = (1, 1, 1)
    else:
        raise ValueError(f"unsupported Sana video resolution: {resolution!r}")
    return {
        "input_size": input_size,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "hidden_size": 2240,
        "depth": 20,
        "num_heads": 20,
        "mlp_ratio": 3.0,
        "class_dropout_prob": 0.1,
        "pred_sigma": False,
        "caption_channels": 2304,
        "model_max_length": 300,
        "qk_norm": True,
        "y_norm": True,
        "y_norm_scale_factor": 0.01,
        "attn_type": "cachedcausal" if longlive else "LiteLAReLURope",
        "ffn_type": "CachedGLUMBConvTemp" if longlive else "GLUMBConvTemp",
        "mlp_acts": ("silu", "silu", None),
        "use_pe": True,
        "pos_embed_type": "casual_wan_rope" if longlive else "wan_rope",
        "linear_head_dim": 112,
        "cross_norm": True,
        "t_kernel_size": 3,
    }


def sana_streaming_config(*, autoregressive: bool) -> dict[str, object]:
    config = sana_video_config(resolution="720p")
    config.update(
        additional_inchannels=128,
        attn_type=("V2VStateCachedBiGDNAttention" if autoregressive else "V2VBiGDNAttention"),
        softmax_ratio=0.25,
        softmax_attn_type=(
            "V2VAfterRoPEGatedSoftmaxAttention" if autoregressive else "V2VGatedSoftmaxAttention"
        ),
        ffn_type=("CachedGLUMBConvTemp" if autoregressive else "GLUMBConvTemp"),
        pos_embed_type=("casual_wan_rope" if autoregressive else "wan_rope"),
    )
    return config


def sana_world_config() -> dict[str, object]:
    """Return the checkpoint-exact SANA-WM bidirectional CamCtrl graph."""

    return {
        "input_size": 22,
        "patch_size": (1, 1, 1),
        "in_channels": 128,
        "hidden_size": 2240,
        "depth": 20,
        "num_heads": 20,
        "mlp_ratio": 3.0,
        "class_dropout_prob": 0.0,
        "pred_sigma": False,
        "caption_channels": 2304,
        "model_max_length": 300,
        "qk_norm": True,
        "y_norm": True,
        "y_norm_scale_factor": 0.01,
        "attn_type": "BidirectionalGDNTriton",
        "ffn_type": "GLUMBConvTemp",
        "mlp_acts": ("silu", "silu", None),
        "use_pe": True,
        "pos_embed_type": "wan_rope",
        "linear_head_dim": 112,
        "cross_norm": True,
        "t_kernel_size": 3,
        "camctrl_type": "BidirectionalGDNUCPESinglePathLiteLABothTriton",
        "cam_attn_compress": 1,
        "init_cam_from_base": True,
        "chunk_split_strategy": "first_chunk_plus_one",
        "conv_kernel_size": 4,
        "k_conv_only": True,
        "softmax_every_n": 4,
        "use_chunk_plucker_post_attn": True,
        "chunk_plucker_channels": 48,
        "chunk_plucker_post_attn_blocks": 20,
        "use_fp32_attention": True,
    }


def sana_world_streaming_config() -> dict[str, object]:
    """Return the chunk-causal Stage-1 graph used by SANA-WM streaming."""

    config = sana_world_config()
    config.update(
        chunk_size=3,
        chunk_split_strategy="first_chunk_plus_one",
    )
    return config


def _unwrap_sana_state_dict(value: Mapping[str, object]) -> Mapping[str, object]:
    state: Mapping[str, object] = value
    wrappers = {"state_dict", "generator", "module", "model_state"}
    while len(state) == 1:
        key = next(iter(state))
        nested = state[key]
        if key not in wrappers or not isinstance(nested, Mapping):
            break
        state = nested
    return state


def _sana_state_dict_converter(
    *,
    input_size: int,
    patch_size: int | tuple[int, int, int],
    hidden_size: int,
):
    """Normalize official wrappers and the unused persistent position buffer."""

    spatial_patch = patch_size[-1] if isinstance(patch_size, tuple) else patch_size
    # ``PatchEmbed`` historically divides by one spatial patch dimension rather
    # than its area.  Preserve that shape because official Sana checkpoints do.
    position_tokens = input_size * input_size // int(spatial_patch)
    expected_position_shape = (1, position_tokens, hidden_size)

    def convert(value: Mapping[str, object]) -> Mapping[str, object]:
        state = _unwrap_sana_state_dict(value)
        converted: dict[str, object] = {}
        for source, tensor in state.items():
            target = str(source)
            for prefix in ("module.", "model."):
                if target.startswith(prefix):
                    target = target.removeprefix(prefix)
            converted[target] = tensor
        position = converted.get("pos_embed")
        if not isinstance(position, torch.Tensor) or tuple(position.shape) != expected_position_shape:
            reference = next((item for item in converted.values() if isinstance(item, torch.Tensor)), None)
            dtype = reference.dtype if isinstance(reference, torch.Tensor) else torch.float32
            converted["pos_embed"] = torch.zeros(expected_position_shape, dtype=dtype)
        return converted

    return convert


def _sana_streaming_state_dict_converter(
    *,
    input_size: int,
    patch_size: int | tuple[int, int, int],
    hidden_size: int,
):
    """Unwrap the streaming ``model.pt`` payload before canonical key cleanup."""

    convert = _sana_state_dict_converter(
        input_size=input_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
    )

    def streaming_convert(value: Mapping[str, object]) -> Mapping[str, object]:
        state: Mapping[str, object] = value
        for wrapper in ("generator", "state_dict"):
            nested = state.get(wrapper)
            if isinstance(nested, Mapping):
                state = nested
        return convert(state)

    return streaming_convert


class SanaDenoiser:
    """Adapt one Sana checkpoint graph to the framework denoiser contract."""

    _DATA_INFO_KEYS = frozenset(
        {
            "aspect_ratio",
            "cfg_scale",
            "condition_frame_info",
            "control_signal",
            "image_embeds",
            "image_vae_embeds",
            "img_hw",
        }
    )

    def __init__(self, model: nn.Module, *, output_scale: float = 1.0) -> None:
        self.model = model
        self.output_scale = float(output_scale)

    def streaming_cache_layout(self) -> tuple[bool, ...]:
        """Describe cache semantics without exposing Sana blocks to the runner."""

        blocks = getattr(self.model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("Sana streaming graph must expose a ModuleList named 'blocks'")
        return tuple(
            getattr(getattr(block, "attn", None), "fixed_rope_cache_type", None) == "state"
            for block in blocks
        )

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        return self.forward_with_options(model_input)

    def forward_with_options(
        self,
        model_input: DenoiserInput,
        *,
        return_log_variance: bool = False,
        apply_output_scale: bool = True,
    ) -> DenoiserOutput:
        """Execute SANA while exposing the outputs needed by native training.

        Inference keeps the component's checkpoint-specific output scale.  SCM
        training requests the raw TrigFlow velocity because its objective owns
        the ``sigma_data`` scaling, and can request the learned log variance.
        """

        if not isinstance(return_log_variance, bool):
            raise TypeError("return_log_variance must be bool")
        if not isinstance(apply_output_scale, bool):
            raise TypeError("apply_output_scale must be bool")
        context = model_input.conditioning.get("context")
        mask = model_input.conditioning.get("context_mask")
        if not isinstance(context, torch.Tensor):
            raise TypeError("Sana denoising requires a tensor 'context' condition")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Sana denoising requires a tensor 'context_mask' condition")

        latents = model_input.latents
        timestep = model_input.timestep.to(device=latents.device, dtype=torch.float32)
        temporal_shape = (latents.shape[0], 1, latents.shape[2]) if latents.ndim == 5 else None
        if temporal_shape is not None and tuple(timestep.shape) == temporal_shape:
            timestep = timestep.clone()
        else:
            timestep = timestep.reshape(-1)
            if timestep.numel() == 1:
                timestep = timestep.expand(latents.shape[0])
            if timestep.numel() != latents.shape[0]:
                raise ValueError(
                    "Sana timestep must be scalar, contain one value per sample, "
                    "or be [B,1,T] for temporal prefix recomputation"
                )
        frame_info = model_input.conditioning.get("condition_frame_info")
        if frame_info and latents.ndim == 5:
            if not isinstance(frame_info, Mapping):
                raise TypeError("Sana condition_frame_info must map frame indices to timestep weights")
            frame_timestep = (
                timestep
                if timestep.ndim == 3
                else timestep[:, None, None].expand(-1, 1, latents.shape[2]).clone()
            )
            for raw_index, raw_weight in frame_info.items():
                index = int(raw_index)
                if not 0 <= index < latents.shape[2]:
                    raise ValueError(f"Sana conditioned frame index {index} is out of range")
                frame_timestep[:, :, index] *= float(raw_weight)
            timestep = frame_timestep

        values = model_input.conditioning
        data_info = {key: values[key] for key in self._DATA_INFO_KEYS if key in values}
        data_info.setdefault(
            "img_hw",
            torch.tensor(
                [[latents.shape[-2], latents.shape[-1]]],
                device=latents.device,
                dtype=latents.dtype,
            ).expand(latents.shape[0], -1),
        )
        data_info.setdefault(
            "aspect_ratio",
            torch.full(
                (latents.shape[0], 1),
                float(latents.shape[-2]) / float(latents.shape[-1]),
                device=latents.device,
                dtype=latents.dtype,
            ),
        )
        data_info.setdefault(
            "cfg_scale",
            torch.ones(latents.shape[0], device=latents.device, dtype=latents.dtype),
        )
        data_info = {
            key: item.to(device=latents.device) if isinstance(item, torch.Tensor) else item
            for key, item in data_info.items()
        }

        reserved = {"context", "context_mask", *self._DATA_INFO_KEYS}
        kwargs: dict[str, Any] = {
            key: item.to(device=latents.device) if isinstance(item, torch.Tensor) else item
            for key, item in values.items()
            if key not in reserved
        }
        if return_log_variance:
            kwargs["return_logvar"] = True
        sample = self.model(
            latents,
            timestep,
            context.to(device=latents.device, dtype=latents.dtype),
            mask=mask.to(device=latents.device),
            data_info=data_info,
            **kwargs,
        )
        extras: dict[str, object] = {}
        if isinstance(sample, tuple):
            if len(sample) != 2:
                raise ValueError(f"Sana graph returned an unsupported tuple of length {len(sample)}")
            sample, auxiliary = sample
            if return_log_variance:
                if not isinstance(auxiliary, torch.Tensor):
                    raise TypeError("Sana SCM graph must return tensor log variance")
                extras["log_variance"] = auxiliary
            else:
                extras["kv_cache"] = auxiliary
        elif return_log_variance:
            raise ValueError("Sana SCM graph did not return its learned log variance")
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"Sana graph returned {type(sample).__name__}, expected Tensor")
        sample = sample[:, : latents.shape[1]]
        slices = (slice(None), slice(None), *(slice(0, size) for size in latents.shape[2:]))
        sample = sample[slices]
        if apply_output_scale:
            sample = sample * self.output_scale
        return DenoiserOutput(sample=sample, extras=extras)


def _module_map() -> Mapping[type[nn.Module], type[nn.Module]]:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    return {
        nn.Linear: AutoWrappedLinear,
        nn.Conv1d: AutoWrappedModule,
        nn.Conv2d: AutoWrappedModule,
        nn.Conv3d: AutoWrappedModule,
        nn.LayerNorm: AutoWrappedModule,
        RMSNorm: AutoWrappedModule,
    }


def _prepare_sana_execution_tensors(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Place unwrapped direct tensors without materializing wrapped descendants."""

    from worldfoundry.core.vram import AutoTorchModule, move_direct_tensors_to_device

    def visit(module: nn.Module) -> None:
        module._worldfoundry_execution_dtype = dtype
        if isinstance(module, AutoTorchModule):
            return
        move_direct_tensors_to_device(module, device=device, dtype=dtype)
        for child in module.children():
            visit(child)

    visit(model)


def _build(
    context: ComponentBuildContext,
    *,
    module_class: type[nn.Module],
    config: Mapping[str, object],
    output_scale: float = 1.0,
    state_dict_converter=None,
) -> SanaDenoiser:
    converter = state_dict_converter or _sana_state_dict_converter(
        input_size=int(config["input_size"]),
        patch_size=config["patch_size"],  # type: ignore[arg-type]
        hidden_size=int(config["hidden_size"]),
    )
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            config=config,
            state_dict_converter=converter,
            vram_module_map=_module_map(),
            layer_container="blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    _prepare_sana_execution_tensors(
        model,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )
    return SanaDenoiser(model, output_scale=output_scale)


def build_sana_image_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    from ..networks.sana.sana_multi_scale import SanaMS

    options = context.component_options
    config = sana_image_config(
        input_size=int(options.get("input_size", 32)),
        parameter_scale=str(options.get("parameter_scale", "1600M")),
        sana_1p5=bool(options.get("sana_1p5", False)),
    )
    return _build(context, module_class=SanaMS, config=config)


def build_sana_sprint_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    from ..networks.sana.sana_multi_scale import SanaMSCM

    config = sana_sprint_config(
        input_size=int(context.component_options.get("input_size", 32)),
        parameter_scale=str(context.component_options.get("parameter_scale", "1600M")),
    )
    return _build(context, module_class=SanaMSCM, config=config, output_scale=0.5)


def build_sana_sprint_teacher_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    """Build the separately checkpointed frozen SCM teacher graph."""

    from ..networks.sana.sana_multi_scale import SanaMSCM

    config = sana_sprint_teacher_config(
        input_size=int(context.component_options.get("input_size", 32)),
        parameter_scale=str(context.component_options.get("parameter_scale", "1600M")),
    )
    return _build(context, module_class=SanaMSCM, config=config, output_scale=0.5)


def build_sana_controlnet_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    from ..networks.sana.sana_multi_scale_controlnet import SanaMSControlNet

    config = sana_controlnet_config(
        input_size=int(context.component_options.get("input_size", 32)),
        parameter_scale=str(context.component_options.get("parameter_scale", "1600M")),
    )
    return _build(context, module_class=SanaMSControlNet, config=config)


def build_sana_video_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    from ..networks.sana.sana_multi_scale_video import SanaMSVideo

    longlive = bool(context.component_options.get("longlive", False))
    config = sana_video_config(
        resolution=str(context.component_options.get("resolution", "480p")),
        longlive=longlive,
    )
    denoiser = _build(context, module_class=SanaMSVideo, config=config)
    if longlive:
        denoiser.model.set_fp32_attention(True)
    return denoiser


def build_sana_streaming_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    from ..networks.sana.sana_multi_scale_video_v2v import SanaMSVideoV2V

    config = sana_streaming_config(
        autoregressive=bool(context.component_options.get("autoregressive", False))
    )
    return _build(context, module_class=SanaMSVideoV2V, config=config)


def build_sana_world_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    from ..networks.sana.sana_multi_scale_video_camctrl import SanaMSVideoCamCtrl

    return _build(context, module_class=SanaMSVideoCamCtrl, config=sana_world_config())


def build_sana_world_streaming_denoiser(context: ComponentBuildContext) -> SanaDenoiser:
    """Load the public streaming ``model.pt`` onto the chunk-causal CamCtrl graph."""

    from ..networks.sana.sana_multi_scale_video_camctrl import SanaMSVideoCamCtrl

    config = sana_world_streaming_config()
    converter = _sana_streaming_state_dict_converter(
        input_size=int(config["input_size"]),
        patch_size=config["patch_size"],  # type: ignore[arg-type]
        hidden_size=int(config["hidden_size"]),
    )
    return _build(
        context,
        module_class=SanaMSVideoCamCtrl,
        config=config,
        state_dict_converter=converter,
    )


__all__ = [
    "SanaDenoiser",
    "build_sana_controlnet_denoiser",
    "build_sana_image_denoiser",
    "build_sana_sprint_denoiser",
    "build_sana_sprint_teacher_denoiser",
    "build_sana_streaming_denoiser",
    "build_sana_video_denoiser",
    "build_sana_world_denoiser",
    "build_sana_world_streaming_denoiser",
    "sana_controlnet_config",
    "sana_image_config",
    "sana_sprint_config",
    "sana_sprint_teacher_config",
    "sana_streaming_config",
    "sana_video_config",
    "sana_world_config",
    "sana_world_streaming_config",
]

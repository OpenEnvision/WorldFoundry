"""Checkpoint-compatible Gamma-World denoisers for the native role system.

Implements :class:`~...contracts.Denoiser` for bidirectional / causal /
few-step variants.  Sampling and KV-cache lifecycle stay in the runner;
this module only maps ``DenoiserInput`` onto the DiT and exposes cache
allocation helpers.  Weights load through
:class:`~...loaders.module.NativeModuleLoader`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ....components import ComponentBuildContext
from ....contracts import DenoiserInput, DenoiserOutput
from ....loaders import ModuleLoadSpec, NativeModuleLoader


_BASE_NETWORK_CONFIG: dict[str, object] = {
    "max_img_h": 240,
    "max_img_w": 240,
    "max_frames": 128,
    "in_channels": 16,
    "out_channels": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "model_channels": 2048,
    "num_blocks": 28,
    "num_heads": 16,
    "concat_padding_mask": True,
    "pos_emb_cls": "rope3d",
    "pos_emb_learnable": True,
    "pos_emb_interpolation": "crop",
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "extra_per_block_abs_pos_emb": False,
    "use_crossattn_projection": True,
    "crossattn_proj_in_channels": 100352,
    "crossattn_emb_channels": 1024,
    "use_wan_fp32_strategy": True,
}

_ACTION_NETWORK_CONFIG: dict[str, object] = {
    "rope_enable_fps_modulation": False,
    "rope_h_extrapolation_ratio": 2.0,
    "rope_w_extrapolation_ratio": 2.0,
    "rope_t_extrapolation_ratio": 1.0,
    "timestep_scale": 0.001,
    "enable_action_control": True,
    "action_keyboard_dim": 23,
    "action_camera_dim": 2,
    "action_use_camera": True,
    "action_embed_dim": 256,
    "action_temporal_downsample": 4,
    "use_multi_agent_rope": True,
    "multi_agent_rope_num_agents": 2,
    "multi_agent_rope_agent_id_offset": 0,
    "multi_agent_rope_simplex_pool_size": 4,
    "multi_agent_rope_agent_encoding": "simplex",
    "multi_agent_rope_agent_scale": 1.0,
    "multi_agent_rope_share_action_encoder": True,
}


def gamma_world_network_config(variant: str) -> dict[str, object]:
    """Return the released architecture without importing Hydra configuration."""

    key = str(variant).strip().lower().replace("-", "_")
    if key not in {"bidirectional", "causal", "causal_few_step"}:
        raise ValueError(f"unknown Gamma-World variant: {variant!r}")
    config = {**_BASE_NETWORK_CONFIG, **_ACTION_NETWORK_CONFIG}
    if key in {"causal", "causal_few_step"}:
        config.update({"use_sparse_hub": True, "z_num": 8})
    if key == "causal_few_step":
        config.update({"local_attn_size": 24, "sink_size": 0})
    return config


def convert_gamma_world_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Normalize released and trainer-prefixed Gamma safetensors layouts."""

    converted: dict[str, object] = {}
    for source_key, value in state_dict.items():
        key = source_key
        for prefix in ("model.net.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.endswith("_extra_state") or key.startswith("accum_"):
            continue
        converted[key] = value
    return converted


@dataclass(frozen=True, slots=True)
class GammaCacheSpec:
    """Architecture-dependent sizes used by the framework-owned window runner."""

    num_layers: int
    num_heads: int
    head_dim: int
    local_frames: int
    sparse_hub: bool
    z_tokens: int


class GammaWorldDenoiser:
    """One-step Gamma denoiser; sampling and cache lifecycle stay in runners."""

    def __init__(self, net: torch.nn.Module, *, variant: str) -> None:
        self.net = net
        self.variant = str(variant).strip().lower().replace("-", "_")
        self.block_size = 48 if self.variant == "bidirectional" else 3
        self.denoising_timesteps = (1000, 750, 500, 250)
        self.context_timestep = 128
        if hasattr(self.net, "num_frame_per_block"):
            self.net.num_frame_per_block = self.block_size
        if hasattr(self.net, "disable_context_parallel"):
            self.net.disable_context_parallel()

    @property
    def is_bidirectional(self) -> bool:
        return self.variant == "bidirectional"

    @property
    def is_distilled(self) -> bool:
        return self.variant == "causal_few_step"

    def cache_spec(self, latent_frames_per_view: int) -> GammaCacheSpec:
        local_attention = int(getattr(self.net, "local_attn_size", -1))
        # Global attention retains every earlier latent frame.  A 24-frame
        # cache only fits the local-attention variant and overflows halfway
        # through the released 48-latent-frame causal rollout.
        local_frames = local_attention if local_attention > 0 else int(latent_frames_per_view)
        heads = int(getattr(self.net, "num_heads"))
        channels = int(getattr(self.net, "model_channels"))
        return GammaCacheSpec(
            num_layers=int(getattr(self.net, "num_layers", getattr(self.net, "num_blocks"))),
            num_heads=heads,
            head_dim=channels // heads,
            local_frames=local_frames,
            sparse_hub=bool(getattr(self.net, "use_sparse_hub", False)),
            z_tokens=int(getattr(self.net, "z_num", 0)),
        )

    def create_kv_cache(
        self,
        *,
        batch_size: int,
        n_views: int,
        latent_frames_per_view: int,
        frame_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict[str, torch.Tensor]]:
        spec = self.cache_spec(latent_frames_per_view)
        if spec.sparse_hub:
            player_tokens = frame_sequence_length * spec.local_frames
            z_tokens = spec.z_tokens * spec.local_frames

            def one() -> dict[str, torch.Tensor]:
                return {
                    "k_players": torch.zeros(
                        batch_size, n_views, player_tokens, spec.num_heads, spec.head_dim,
                        device=device, dtype=dtype,
                    ),
                    "v_players": torch.zeros(
                        batch_size, n_views, player_tokens, spec.num_heads, spec.head_dim,
                        device=device, dtype=dtype,
                    ),
                    "k_z": torch.zeros(
                        batch_size, z_tokens, spec.num_heads, spec.head_dim,
                        device=device, dtype=dtype,
                    ),
                    "v_z": torch.zeros(
                        batch_size, z_tokens, spec.num_heads, spec.head_dim,
                        device=device, dtype=dtype,
                    ),
                    "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                    "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                    "z_local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                }
        else:
            sequence_tokens = frame_sequence_length * spec.local_frames * n_views

            def one() -> dict[str, torch.Tensor]:
                return {
                    "k": torch.zeros(
                        batch_size, sequence_tokens, spec.num_heads, spec.head_dim,
                        device=device, dtype=dtype,
                    ),
                    "v": torch.zeros(
                        batch_size, sequence_tokens, spec.num_heads, spec.head_dim,
                        device=device, dtype=dtype,
                    ),
                    "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                    "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                }
        return [one() for _ in range(spec.num_layers)]

    @staticmethod
    def _as_timestep(value: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            return value.expand(latents.shape[0], latents.shape[2])
        if value.ndim == 1:
            if value.shape[0] == latents.shape[0]:
                return value[:, None].expand(-1, latents.shape[2])
            if value.numel() == 1:
                return value.reshape(1, 1).expand(latents.shape[0], latents.shape[2])
        return value

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        from ...encoders.structured_conditioning import DataType

        condition = dict(model_input.conditioning)
        latents = model_input.latents
        parameter = next(self.net.parameters())
        model_dtype = parameter.dtype
        model_device = parameter.device

        gt_frames = condition.get("gt_frames")
        mask = condition.get("condition_video_input_mask_B_C_T_H_W")
        use_video_condition = bool(condition.get("use_video_condition", True))
        if isinstance(gt_frames, torch.Tensor) and isinstance(mask, torch.Tensor):
            gt_frames = gt_frames.to(device=latents.device, dtype=latents.dtype)
            mask = mask.to(device=latents.device, dtype=latents.dtype)
            if not use_video_condition:
                gt_frames = torch.zeros_like(gt_frames)
            expanded_mask = mask.expand(-1, latents.shape[1], -1, -1, -1)
            denoiser_input = gt_frames * expanded_mask + latents * (1.0 - expanded_mask)
        else:
            expanded_mask = None
            denoiser_input = latents

        timesteps = self._as_timestep(model_input.timestep, latents)
        context = condition.get("crossattn_emb")
        if not isinstance(context, torch.Tensor):
            raise TypeError("Gamma denoiser requires a crossattn_emb tensor")
        fps = condition.get("fps")
        if not isinstance(fps, torch.Tensor):
            fps = torch.full((latents.shape[0],), float(fps or 16.0), device=model_device)

        kwargs: dict[str, object] = {
            "x_B_C_T_H_W": denoiser_input.to(device=model_device, dtype=model_dtype),
            "timesteps_B_T": timesteps.to(device=model_device),
            "crossattn_emb": context.to(device=model_device, dtype=model_dtype),
            "condition_video_input_mask_B_C_T_H_W": (
                mask.to(device=model_device, dtype=model_dtype)
                if isinstance(mask, torch.Tensor)
                else torch.zeros(
                    latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4],
                    device=model_device, dtype=model_dtype,
                )
            ),
            "fps": fps.to(device=model_device),
            "padding_mask": condition.get("padding_mask"),
            "data_type": DataType.VIDEO,
            "view_indices_B_T": condition.get("view_indices_B_T"),
            "action_inputs": condition.get("action_inputs"),
        }
        if not self.is_bidirectional:
            cache = condition.get("kv_cache")
            if not isinstance(cache, list):
                raise TypeError("causal Gamma denoising requires a framework-managed kv_cache")
            kwargs.update(
                {
                    "kv_cache": cache,
                    "crossattn_cache": condition.get("crossattn_cache"),
                    "current_start": int(condition.get("current_start", 0)),
                    "current_end": int(condition.get("current_end", 0)),
                    "start_frame_for_rope": int(condition.get("start_frame_for_rope", 0)),
                }
            )

        for key in ("padding_mask", "view_indices_B_T"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device=model_device)

        output = self.net(**kwargs).float()
        if self.is_distilled:
            # The official self-forcing runtime warps the configured
            # [1000, 750, 500, 250] schedule before calling the network.  The
            # resulting model timestep is already sigma * 1000, so applying
            # the flow shift again here would double-shift the x0 conversion.
            sigma = (timesteps.float() / 1000.0).to(output.device)[:, None, :, None, None]
            sample = denoiser_input.float() - sigma * output
            if expanded_mask is not None:
                sample = gt_frames.float() * expanded_mask.float() + sample * (1.0 - expanded_mask.float())
            return DenoiserOutput(sample=sample, extras={"flow": output})

        noise = condition.get("block_noise", condition.get("initial_noise"))
        if expanded_mask is not None and isinstance(noise, torch.Tensor):
            clean_velocity = noise.to(output).float() - gt_frames.to(output).float()
            output = clean_velocity * expanded_mask.float() + output * (1.0 - expanded_mask.float())
        return DenoiserOutput(sample=output)


def _build_gamma_world_denoiser(
    context: ComponentBuildContext,
    *,
    variant: str,
) -> GammaWorldDenoiser:
    from ...networks.gamma_world.causal import CosmosCausalDiT
    from ...networks.gamma_world.multiview_dit import MinimalV1LVGDiT

    module_class = MinimalV1LVGDiT if variant == "bidirectional" else CosmosCausalDiT
    net = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            config=gamma_world_network_config(variant),
            state_dict_converter=convert_gamma_world_state_dict,
            layer_container="blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    return GammaWorldDenoiser(net, variant=variant)


def build_gamma_world_bidirectional_denoiser(
    context: ComponentBuildContext,
) -> GammaWorldDenoiser:
    return _build_gamma_world_denoiser(context, variant="bidirectional")


def build_gamma_world_causal_denoiser(context: ComponentBuildContext) -> GammaWorldDenoiser:
    return _build_gamma_world_denoiser(context, variant="causal")


def build_gamma_world_causal_few_step_denoiser(
    context: ComponentBuildContext,
) -> GammaWorldDenoiser:
    return _build_gamma_world_denoiser(context, variant="causal_few_step")


__all__ = [
    "GammaCacheSpec",
    "GammaWorldDenoiser",
    "build_gamma_world_bidirectional_denoiser",
    "build_gamma_world_causal_denoiser",
    "build_gamma_world_causal_few_step_denoiser",
    "convert_gamma_world_state_dict",
    "gamma_world_network_config",
]

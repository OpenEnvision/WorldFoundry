"""Small inference-only helpers shared by the Evoke CLI and pipeline."""

from __future__ import annotations

import math
import os
from typing import Literal, Optional

import torch
import torch.nn.functional as F


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def apply_schedule_shift(
    sigmas,
    noise,
    sigmas_two=None,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    exp_max: float = 7.0,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    mu: float | None = None,
    return_mu: bool = False,
):
    if mu is None:
        image_seq_len = (noise.shape[-1] * noise.shape[-2] * noise.shape[-3]) // 4
        mu = _calculate_shift(
            image_seq_len,
            base_seq_len,
            max_seq_len,
            base_shift,
            max_shift,
        )
        if time_shift_type == "exponential":
            mu = math.exp(min(mu, math.log(exp_max)))

    def _shift(values):
        return (values * mu) / (1 + (mu - 1) * values)

    shifted = _shift(sigmas)
    if sigmas_two is not None:
        result = (shifted, _shift(sigmas_two))
        return (*result, mu) if return_mu else result
    return (shifted, mu) if return_mu else shifted


def add_noise(original_samples, noise, timestep, sigmas, timesteps):
    sigmas = sigmas.to(noise.device)
    timesteps = timesteps.to(noise.device)
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


def convert_flow_pred_to_x0(flow_pred, xt, timestep, sigmas, timesteps):
    original_dtype = flow_pred.dtype
    device = flow_pred.device
    flow_pred, xt, sigmas, timesteps = (value.double().to(device) for value in (flow_pred, xt, sigmas, timesteps))
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    return (xt - sigma_t * flow_pred).to(original_dtype)


class AdaptiveAntiDrifting:
    def __init__(
        self,
        rho_mu: float = 0.9,
        rho_sigma: float = 0.9,
        delta_mu: float = 0.15,
        delta_sigma: float = 0.15,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.rho_mu = rho_mu
        self.rho_sigma = rho_sigma
        self.delta_mu = delta_mu
        self.delta_sigma = delta_sigma
        self.device = device
        self.dtype = dtype
        self.reset()

    @staticmethod
    def compute_latent_statistics(latent_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return latent_chunk.mean(dim=[2, 3, 4]), latent_chunk.var(dim=[2, 3, 4])

    def update_global_statistics(self, current_mean: torch.Tensor, current_var: torch.Tensor) -> None:
        if not self.is_initialized:
            self.global_mean = current_mean.clone()
            self.global_var = current_var.clone()
            self.is_initialized = True
            return
        self.global_mean = self.rho_mu * self.global_mean + (1 - self.rho_mu) * current_mean
        self.global_var = self.rho_sigma * self.global_var + (1 - self.rho_sigma) * current_var

    def detect_drift(self, current_mean: torch.Tensor, current_var: torch.Tensor) -> bool:
        if not self.is_initialized:
            return False
        mean_drift = torch.norm(current_mean - self.global_mean, p=2, dim=-1).mean().item()
        var_drift = torch.norm(current_var - self.global_var, p=2, dim=-1).mean().item()
        return mean_drift > self.delta_mu and var_drift > self.delta_sigma

    @staticmethod
    def apply_frame_aware_corruption(
        history_latents: torch.Tensor,
        corruption_strength: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(history_latents, generator=generator, device=history_latents.device)
        return history_latents + corruption_strength * noise

    def reset(self) -> None:
        self.global_mean = None
        self.global_var = None
        self.is_initialized = False


def vigeo_opts_from_cfg(cfg: dict) -> dict:
    keys = (
        "mode",
        "chunk_size",
        "scale_mode",
        "anchor_windows",
        "cache_keep_frames",
        "total_budget",
        "intr_source",
        "conf_transform",
        "num_tokens",
        "scale_value",
        "depth_median_target",
    )
    return {key: cfg.get(f"vigeo_{key}") for key in keys}


def geo_resize_visibility_to_latent(
    visibility_mask_pix: torch.Tensor,
    num_lat_per_chunk: int,
    height_latent: int,
    width_latent: int,
    vae_t_stride: int = 4,
    patch_hw: tuple[int, int] = (2, 2),
    cov_thresh: float = 0.5,
) -> torch.Tensor:
    sample_ids = torch.arange(num_lat_per_chunk, device=visibility_mask_pix.device) * int(vae_t_stride)
    sample_ids = sample_ids.clamp(max=visibility_mask_pix.shape[2] - 1)
    sampled = visibility_mask_pix.index_select(2, sample_ids)
    patch_height, patch_width = max(1, int(patch_hw[0])), max(1, int(patch_hw[1]))
    grid_height = max(1, int(height_latent) // patch_height)
    grid_width = max(1, int(width_latent) // patch_width)
    patch_cov = F.adaptive_avg_pool3d(sampled, (num_lat_per_chunk, grid_height, grid_width))
    patch_bin = (patch_cov > float(cov_thresh)).to(sampled.dtype)
    return F.interpolate(patch_bin, size=(num_lat_per_chunk, height_latent, width_latent), mode="nearest")


def _load_prefixed(module, state_dict: dict, prefix: str) -> set[str]:
    keys = [key for key in state_dict if key.startswith(prefix)]
    if not keys or module is None:
        return set()
    module.load_state_dict({key.removeprefix(prefix): state_dict[key] for key in keys}, strict=False)
    return set(keys)


def load_extra_components(args, model, checkpoint_path) -> None:
    """Load inference-relevant modules from an optional partial checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"expected a state dict in {checkpoint_path}")

    loaded: set[str] = set()
    for name in ("patch_short", "patch_mid", "patch_long"):
        loaded.update(_load_prefixed(getattr(model, name, None), state_dict, f"{name}."))

    for block_index, block in enumerate(getattr(model, "blocks", ())):
        attention = getattr(block, "attn1", None)
        if attention is None:
            continue
        for name in ("q_loras", "k_loras", "v_loras"):
            loaded.update(
                _load_prefixed(
                    getattr(attention, name, None),
                    state_dict,
                    f"blocks.{block_index}.attn1.{name}.",
                )
            )
        history_key = f"blocks.{block_index}.attn1.history_key_scale"
        if history_key in state_dict and hasattr(attention, "history_key_scale"):
            attention.history_key_scale.data.copy_(state_dict[history_key].to(attention.history_key_scale.device))
            loaded.add(history_key)

    warp_keys = [key for key in state_dict if key.startswith("warp_residual_mlp.")]
    if warp_keys:
        first_weight = state_dict.get("warp_residual_mlp.0.weight")
        if first_weight is not None and getattr(model, "warp_residual_mlp", None) is None:
            hidden_dim, inner_dim = first_weight.shape
            model.warp_residual_mlp = torch.nn.Sequential(
                torch.nn.Linear(int(inner_dim), int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hidden_dim), int(inner_dim)),
            ).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        loaded.update(_load_prefixed(getattr(model, "warp_residual_mlp", None), state_dict, "warp_residual_mlp."))

    model_config = getattr(args, "model_config", None)
    camera_config = getattr(model_config, "camera_control", None)
    if camera_config is not None and getattr(camera_config, "enabled", False):
        camera_path = os.path.join(os.path.dirname(os.fspath(checkpoint_path)), "camera_ctrl.safetensors")
        if os.path.exists(camera_path):
            from worldfoundry.base_models.diffusion_model.models.networks.evoke.camera_control import load_camera_ctrl_weights

            load_camera_ctrl_weights(
                model,
                camera_path,
                num_layers=model.config.num_layers,
                cam_ctrl_layers=camera_config.cam_ctrl_layers,
                strict=camera_config.strict_camera_ckpt,
            )

    print(f"Loaded {len(loaded)} inference component parameters from {checkpoint_path}")


__all__ = [
    "AdaptiveAntiDrifting",
    "add_noise",
    "apply_schedule_shift",
    "convert_flow_pred_to_x0",
    "geo_resize_visibility_to_latent",
    "load_extra_components",
    "vigeo_opts_from_cfg",
]

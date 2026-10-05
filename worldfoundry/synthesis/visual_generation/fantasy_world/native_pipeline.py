"""Native FantasyWorld model construction on the shared Wan inference infra."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.loaders import (
    WanInferenceComponents,
    load_wan_conditioning_components,
    load_wan_inference_components,
    load_wan_transformer_checkpoint,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    WAN21_I2V_14B_CONFIG,
    WAN22_I2V_A14B_CONFIG,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.fantasy_world import (
    FantasyWorldCameraCondition,
    FantasyWorldFusionModel,
)
from worldfoundry.base_models.diffusion_model.runners.wan_staged import WanStagedPipeline
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.variants.fantasy_world.models.vggt import (
    VGGT,
)
from worldfoundry.core.model_loading import (
    load_state_dict,
    merge_flattened_path_lora_,
)


LOGGER = logging.getLogger(__name__)

WAN21_OVERLAY_GROUPS = {
    "IRGBlock": 1650,
    "camera_condition": 22,
    "pipe": 695,
    "vggt": 765,
}
WAN22_OVERLAY_GROUPS = {
    "IRGBlock": 1440,
    "pipe": 453,
    "vggt": 765,
}


@dataclass(slots=True)
class FantasyWorldWan22Models:
    """High/low denoisers sharing one explicit conditioning pipeline."""

    high: FantasyWorldFusionModel
    low: FantasyWorldFusionModel


def _build_vggt(*, torch_dtype: torch.dtype) -> VGGT:
    return VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=False,
        DPT_patch_size=16,
    ).to(dtype=torch_dtype)


def load_fantasy_world_overlay(
    model: FantasyWorldFusionModel,
    checkpoint_path: str | Path,
    *,
    expected_groups: dict[str, int],
    label: str,
) -> None:
    """Fail closed on the released composite schema, then restore its tensors."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FantasyWorld {label} checkpoint not found: {path}")
    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    actual_groups = dict(Counter(key.split(".", 1)[0] for key in state_dict))
    if actual_groups != expected_groups:
        raise RuntimeError(
            f"FantasyWorld {label} checkpoint schema mismatch: "
            f"expected={expected_groups}, actual={actual_groups}"
        )

    model_state = model.state_dict()
    unexpected = sorted(set(state_dict) - set(model_state))
    mismatched = sorted(
        key
        for key in set(state_dict) & set(model_state)
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    )
    if unexpected or mismatched:
        raise RuntimeError(
            f"FantasyWorld {label} native graph does not match its released checkpoint: "
            f"unexpected={unexpected[:8]}, shape_mismatch={mismatched[:8]}"
        )
    messages = model.load_state_dict(state_dict, strict=False)
    if messages.unexpected_keys:
        raise RuntimeError(
            f"FantasyWorld {label} restore produced unexpected keys: "
            f"{messages.unexpected_keys[:8]}"
        )
    LOGGER.info(
        "restored FantasyWorld %s overlay: %d tensors; %d shared base tensors retained",
        label,
        len(state_dict),
        len(messages.missing_keys),
    )


def build_fantasy_world_wan21_model(
    *,
    base_model_root: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> FantasyWorldFusionModel:
    """Construct Wan2.1 FantasyWorld from explicit shared model roles."""

    components = load_wan_inference_components(
        base_model_root,
        image_conditioned=True,
        torch_dtype=torch_dtype,
        device="cpu",
        transformer_config=WAN21_I2V_14B_CONFIG,
    )
    pipe = WanStagedPipeline.from_components(
        components,
        device=device,
        torch_dtype=torch_dtype,
    )
    camera_condition = FantasyWorldCameraCondition(
        pipe.dit,
        pose_in_dim=768,
        plucker_fea_dim=2048,
        pose_inject_method="adaln",
        use_info="plucker",
        processor_layers=25,
    ).to(dtype=torch_dtype)
    model = FantasyWorldFusionModel(
        pipe=pipe,
        vggt=_build_vggt(torch_dtype=torch_dtype),
        camera_condition=camera_condition,
        start_index=16,
        fusion_blocks=range(24),
    )
    load_fantasy_world_overlay(
        model,
        checkpoint_path,
        expected_groups=WAN21_OVERLAY_GROUPS,
        label="Wan2.1",
    )
    return model


def _merge_reward_lora(
    dit: torch.nn.Module,
    lora_path: str | Path,
    *,
    scale: float = 0.55,
) -> None:
    state_dict = load_state_dict(
        str(Path(lora_path).expanduser().resolve()),
        torch_dtype=torch.float32,
        device="cpu",
    )
    expected = sum(key.endswith(".lora_up.weight") for key in state_dict)
    updated = merge_flattened_path_lora_(
        dit,
        state_dict,
        prefix="lora_unet__",
        scale=scale,
    )
    if updated != expected:
        raise RuntimeError(
            f"FantasyWorld reward LoRA matched {updated}/{expected} native Wan modules"
        )


def _build_wan22_dit(
    checkpoint_root: str | Path,
    *,
    lora_path: str | Path,
    torch_dtype: torch.dtype,
) -> torch.nn.Module:
    dit = load_wan_transformer_checkpoint(
        checkpoint_root,
        torch_dtype=torch_dtype,
        device="cpu",
        transformer_config=WAN22_I2V_A14B_CONFIG,
        strict=True,
    )
    dit.enable_control_adapter(in_dim=24)
    _merge_reward_lora(dit, lora_path)
    return dit


def build_fantasy_world_wan22_models(
    *,
    base_model_root: str | Path,
    high_lora_path: str | Path,
    low_lora_path: str | Path,
    high_checkpoint_path: str | Path,
    low_checkpoint_path: str | Path,
    high_device: str | torch.device = "cpu",
    low_device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> FantasyWorldWan22Models:
    """Construct native high/low Wan2.2 FantasyWorld denoisers."""

    root = Path(base_model_root).expanduser().resolve()
    conditioning = load_wan_conditioning_components(
        root,
        image_conditioned=False,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    high_dit = _build_wan22_dit(
        root / "high_noise_model",
        lora_path=high_lora_path,
        torch_dtype=torch_dtype,
    )
    low_dit = _build_wan22_dit(
        root / "low_noise_model",
        lora_path=low_lora_path,
        torch_dtype=torch_dtype,
    )
    high_components = WanInferenceComponents(
        dit=high_dit,
        text_encoder=conditioning.text_encoder,
        vae=conditioning.vae,
        tokenizer_path=conditioning.tokenizer_path,
        image_encoder=None,
    )
    high_pipe = WanStagedPipeline.from_components(
        high_components,
        device=high_device,
        torch_dtype=torch_dtype,
    )
    low_pipe = WanStagedPipeline(
        device=low_device,
        torch_dtype=torch_dtype,
        tokenizer_path=None,
    )
    low_pipe.dit = low_dit
    low_pipe.model_names = ["dit"]

    high_model = FantasyWorldFusionModel(
        pipe=high_pipe,
        vggt=_build_vggt(torch_dtype=torch_dtype),
        start_index=16,
        fusion_blocks=range(24),
    )
    low_model = FantasyWorldFusionModel(
        pipe=low_pipe,
        vggt=_build_vggt(torch_dtype=torch_dtype),
        start_index=16,
        fusion_blocks=range(24),
    )
    load_fantasy_world_overlay(
        high_model,
        high_checkpoint_path,
        expected_groups=WAN22_OVERLAY_GROUPS,
        label="Wan2.2 HIGH",
    )
    load_fantasy_world_overlay(
        low_model,
        low_checkpoint_path,
        expected_groups=WAN22_OVERLAY_GROUPS,
        label="Wan2.2 LOW",
    )
    return FantasyWorldWan22Models(high=high_model, low=low_model)


__all__ = [
    "FantasyWorldWan22Models",
    "WAN21_OVERLAY_GROUPS",
    "WAN22_OVERLAY_GROUPS",
    "build_fantasy_world_wan21_model",
    "build_fantasy_world_wan22_models",
    "load_fantasy_world_overlay",
]

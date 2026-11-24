"""Helper functions for event-centric pipeline.

This module contains utility functions for:
- Checkpoint loading and model creation
- LiveWorld pipeline initialization
- Geometry and video loading
"""

from __future__ import annotations

import os

import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file

from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pipeline_unified_backbone import (
    UnifiedBackbonePipeline,
)
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.wrapper import (
    BidirectionalWanWrapperSP,
    WanTextEncoder,
    WanVAEWrapper,
)

from .logger import logger

# =============================================================================
# Project path utilities
# =============================================================================


def get_project_root() -> str:
    """Return the runtime root for relative config lookup."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def resolve_train_config(config_path: str):
    """
    Load a training config and merge its default_config_path when provided.

    This lets inference reuse the same train yaml used by scripts/train.py.
    """
    cfg = OmegaConf.load(config_path)
    default_config_path = cfg.get("default_config_path", None)
    if not default_config_path:
        return cfg

    # Resolve relative default path against observer config dir first,
    # then fallback to project root.
    if not os.path.isabs(default_config_path):
        candidate_from_cfg_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(config_path)), default_config_path)
        )
        if os.path.exists(candidate_from_cfg_dir):
            default_config_path = candidate_from_cfg_dir
        else:
            default_config_path = os.path.normpath(os.path.join(get_project_root(), default_config_path))

    if not os.path.exists(default_config_path):
        logger.warning(
            f"default_config_path not found ({default_config_path}); loading observer config without defaults merge."
        )
        return cfg

    default_cfg = OmegaConf.load(default_config_path)
    return OmegaConf.merge(default_cfg, cfg)


# =============================================================================
# Checkpoint loading utilities
# =============================================================================


def find_checkpoint_file(ckpt_path: str) -> str:
    """Resolve a checkpoint path that can be a file or directory."""
    if os.path.isfile(ckpt_path):
        return ckpt_path

    if os.path.isdir(ckpt_path):
        candidates = ["model.pt", "model.safetensors", "checkpoint.pt", "checkpoint.safetensors"]
        for candidate in candidates:
            full_path = os.path.join(ckpt_path, candidate)
            if os.path.exists(full_path):
                return full_path

        for ext in (".pt", ".safetensors"):
            files = [f for f in os.listdir(ckpt_path) if f.endswith(ext)]
            if files:
                return os.path.join(ckpt_path, files[0])

    raise FileNotFoundError(f"Could not find checkpoint file in: {ckpt_path}")


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> dict:
    """Load a checkpoint (pt or safetensors)."""
    ckpt_file = find_checkpoint_file(ckpt_path)
    logger.info(f"Loading checkpoint: {os.path.basename(ckpt_file)}")

    if ckpt_file.endswith(".safetensors"):
        return load_file(ckpt_file, device=device)

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        if "generator" in checkpoint:
            return checkpoint["generator"]
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


# =============================================================================
# Model creation utilities
# =============================================================================


def create_generator(
    config,
    state_dict: dict,
    device: torch.device,
    dtype: torch.dtype,
    state_source: str | None = None,
):
    """Create the LiveWorld generator model and load weights."""
    generator = BidirectionalWanWrapperSP(config=config, role="generator")
    state_source = state_source or "<unknown>"

    def _extract_sp_keys(src_state_dict: dict) -> dict:
        sp_keys = {}
        for k, v in src_state_dict.items():
            if "sp_" not in k.lower() or "lora_" in k:
                continue
            clean_key = k.replace("._fsdp_wrapped_module", "")
            for prefix in ("base_model.model.model.", "model.model.", "base_model.model.", "model."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break
            sp_keys[clean_key] = v
        return sp_keys

    def _load_sp_weights(sp_state_dict: dict, source_name: str) -> int:
        if not sp_state_dict:
            return 0
        missing, unexpected = generator.model.model.load_state_dict(sp_state_dict, strict=False)
        sp_missing = [k for k in missing if "sp_" in k.lower()]
        if sp_missing:
            logger.warning(f"State Adapter missing keys ({source_name}): {sp_missing[:5]}")
        if unexpected:
            logger.warning(f"State Adapter unexpected keys ({source_name}): {unexpected[:5]}")
        logger.info(f"Loaded {len(sp_state_dict)} State Adapter parameters from {source_name}")
        return len(sp_state_dict)

    # LoRA handling (same as infer.py).
    use_lora = getattr(config, "use_lora", False)
    if use_lora:
        lora_config_dict = dict(config.lora_config)
        original_targets = lora_config_dict.get("target_modules", ["q", "k", "v", "o"])

        backbone_modules = []
        for name, _ in generator.named_modules():
            if "sp_" in name.lower():
                continue
            for target in original_targets:
                if name.endswith(f".{target}"):
                    backbone_modules.append(name)
                    break

        if backbone_modules:
            lora_config_dict["target_modules"] = backbone_modules
            logger.info(f"LoRA applied to {len(backbone_modules)} backbone modules")

        lora_config = LoraConfig(**lora_config_dict)
        generator = get_peft_model(generator, lora_config)
        for _, cfg in generator.peft_config.items():
            cfg.inference_mode = True
        logger.info(f"Loading LoRA module weights from: {state_source}")

        # state_dict from a PEFT model already has adapter name "default" in
        # LoRA keys (lora_A.default.weight).  set_peft_model_state_dict inserts
        # the adapter name again, so we must strip ".default." first.
        lora_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" not in k:
                continue
            clean_key = k.replace("._fsdp_wrapped_module", "")
            while clean_key.startswith("module."):
                clean_key = clean_key[len("module.") :]
            while clean_key.startswith("model."):
                clean_key = clean_key[len("model.") :]
            clean_key = clean_key.replace(".default.", ".")
            lora_state_dict[clean_key] = v
        result = set_peft_model_state_dict(generator, lora_state_dict)
        if result.missing_keys:
            logger.warning(f"LoRA missing keys: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            logger.warning(f"LoRA unexpected keys: {result.unexpected_keys[:5]}")
        logger.info(f"Loaded {len(lora_state_dict)} LoRA parameters from: {state_source}")

        # Load State Adapter weights separately (not handled by PEFT).
        # Priority:
        # 1) Explicit config path (sp_model_path)
        # 2) Current inference checkpoint (if it carries State Adapter keys)
        loaded_sp = 0
        sp_ckpt_path = getattr(config, "sp_model_path", None)
        if sp_ckpt_path:
            sp_ckpt_file = find_checkpoint_file(sp_ckpt_path)
            logger.info(f"Loading State Adapter module weights from: {sp_ckpt_file}")
            sp_state_dict = load_checkpoint(sp_ckpt_file, device="cpu")
            loaded_sp = _load_sp_weights(
                _extract_sp_keys(sp_state_dict),
                f"State Adapter checkpoint ({sp_ckpt_file})",
            )

        if loaded_sp == 0:
            logger.info(
                "State Adapter explicit checkpoint did not provide weights; "
                f"loading from state checkpoint: {state_source}"
            )
            loaded_sp = _load_sp_weights(_extract_sp_keys(state_dict), f"state checkpoint ({state_source})")

        if loaded_sp == 0:
            raise RuntimeError(
                "No State Adapter parameters were loaded. "
                "State Adapter blocks remain random and generation quality will collapse."
            )
    else:
        logger.info(f"Loading backbone/module weights from state checkpoint: {state_source}")
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k
            for prefix in ("model.model.", "model.", "module."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break
            cleaned_state_dict[clean_key] = v
        generator.model.load_state_dict(cleaned_state_dict, strict=False)
        logger.info(f"Loaded {len(cleaned_state_dict)} parameters into generator model from: {state_source}")

    generator.to(device=device, dtype=dtype)
    generator.eval()
    return generator


def load_unified_backbone_pipeline(
    observer_cfg: dict,
    device: torch.device,
    cpu_offload: bool,
) -> UnifiedBackbonePipeline:
    """Load UnifiedBackbonePipeline with weights and config.

    Args:
        observer_cfg: Observer section of the event-centric config.
        device: Target CUDA device.
        cpu_offload: Controlled by monitor_centric config runtime.cpu_offload.observer.
    """
    train_config = resolve_train_config(observer_cfg["config"])

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Forward observer-level overrides that the pipeline reads from config.
    # These let the system config (e.g. few-step) override train-config defaults.
    _OBSERVER_TO_TRAIN_CONFIG_KEYS = ["denoising_step_list", "timestep_shift"]
    for key in _OBSERVER_TO_TRAIN_CONFIG_KEYS:
        if key in observer_cfg:
            OmegaConf.update(train_config, key, observer_cfg[key])

    # Support split observer checkpoints:
    # - backbone_path: backbone-only weights (assigned to train_config generator_backbone_ckpt_path)
    # - lora_path: LoRA checkpoint used to populate PEFT adapters
    # - sp_path: State Adapter checkpoint used by create_generator() priority load
    # Keep backward compatibility with legacy single observer.ckpt_path.
    backbone_path = observer_cfg.get("backbone_path")
    lora_path = observer_cfg.get("lora_path")
    sp_path = observer_cfg.get("sp_path")
    legacy_ckpt_path = observer_cfg.get("ckpt_path")

    if backbone_path:
        train_config.generator_backbone_ckpt_path = backbone_path
    if sp_path:
        train_config.sp_model_path = sp_path

    use_lora = bool(getattr(train_config, "use_lora", False))
    if use_lora:
        state_ckpt_path = lora_path or legacy_ckpt_path
        if not state_ckpt_path:
            raise KeyError("observer must provide `lora_path` (or legacy `ckpt_path`) when use_lora=true")
    else:
        state_ckpt_path = legacy_ckpt_path or backbone_path
        if not state_ckpt_path:
            raise KeyError("observer must provide `ckpt_path` or `backbone_path` when use_lora=false")

    resolved_backbone_file = None
    effective_backbone_path = getattr(train_config, "generator_backbone_ckpt_path", None)
    if effective_backbone_path:
        resolved_backbone_file = find_checkpoint_file(effective_backbone_path)

    resolved_sp_file = None
    effective_sp_path = sp_path or getattr(train_config, "sp_model_path", None)
    if effective_sp_path:
        resolved_sp_file = find_checkpoint_file(effective_sp_path)

    state_ckpt_file = find_checkpoint_file(state_ckpt_path)

    logger.info("Observer module load plan:")
    logger.info(
        f"  backbone -> {resolved_backbone_file or effective_backbone_path or '<official wan_model_name only>'}"
    )
    logger.info(f"  lora     -> {state_ckpt_file if use_lora else '<disabled by config>'}")
    logger.info(f"  state_proj -> {resolved_sp_file or effective_sp_path or '<state checkpoint fallback>'}")
    logger.info(f"  state    -> {state_ckpt_file}")
    logger.info(
        "  base model init -> "
        f"wan_model_name={getattr(train_config, 'wan_model_name', None)}, "
        f"load_official_backbone={getattr(train_config, 'load_official_backbone', None)}"
    )

    state_dict = load_checkpoint(state_ckpt_file, device="cpu")

    init_device = torch.device("cpu") if cpu_offload else device

    generator = create_generator(
        train_config,
        state_dict,
        init_device,
        dtype,
        state_source=state_ckpt_file,
    )

    vae = WanVAEWrapper(model_name=train_config.wan_model_name)
    if cpu_offload:
        vae.to(dtype=dtype)
    else:
        vae.to(device=device, dtype=dtype)
    vae.eval()

    text_encoder = WanTextEncoder(model_name=train_config.wan_model_name)
    text_encoder.eval()

    pipeline = UnifiedBackbonePipeline(
        config=train_config,
        device=device,
        generator=generator,
        vae=vae,
        text_encoder=text_encoder,
        dtype=dtype,
    )
    return pipeline


# =============================================================================
# Resolution parsing utilities
# =============================================================================


def parse_target_resolution(observer_cfg: dict) -> tuple[int, int]:
    """Parse target resolution (H, W) from observer config.

    Reads the image_or_video_shape from the training config and computes
    pixel resolution by multiplying latent dimensions by 8.

    Args:
        observer_cfg: Observer configuration dict containing 'config' path.

    Returns:
        Tuple of (height, width) in pixels.
    """
    train_config = resolve_train_config(observer_cfg["config"])

    # image_or_video_shape: [batch, frames, channels, height, width]
    # The last two dimensions are latent height and width
    shape = train_config.image_or_video_shape
    latent_h = shape[3]  # Height in latent space
    latent_w = shape[4]  # Width in latent space

    # Convert to pixel resolution (latent * 8)
    pixel_h = latent_h * 8
    pixel_w = latent_w * 8

    logger.info(f"Target resolution from config: {pixel_h}x{pixel_w} (latent: {latent_h}x{latent_w})")
    return (pixel_h, pixel_w)


# =============================================================================
# Geometry and video loading utilities
# =============================================================================


def load_geometry_poses(geometry_path: str) -> np.ndarray:
    """Load camera poses from geometry.npz file."""
    data = np.load(geometry_path)
    if "poses_c2w" in data:
        return data["poses_c2w"].astype(np.float32)
    if "poses" in data:
        return data["poses"].astype(np.float32)
    if "c2w" in data:
        return data["c2w"].astype(np.float32)
    raise KeyError(f"No poses_c2w/poses/c2w found in geometry: {geometry_path}")


def load_frame_from_image(image_path: str, target_size: tuple[int, int] | None = None) -> Image.Image:
    """Load the first frame from an image file.

    Args:
        image_path: Path to image file (png, jpg, etc.)
        target_size: Optional (width, height) to resize to.
    """
    img = Image.open(image_path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return img

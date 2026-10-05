# pylint: disable=R0913,R0914,C0103
"""In-tree wrapper around the VideoCrafter runtime."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal, Optional

from .components import load_videocrafter_components


def _config_root() -> Path:
    """Helper function to config root.

    Returns:
        The return value.
    """
    return Path(str(files("worldfoundry.data").joinpath("models", "runtime", "configs", "videocrafter")))


def resolve_runtime_config(config: str) -> Path:
    """
    Resolve a VideoCrafter inference config to the packaged in-tree runtime.

    Args:
        config: Absolute path or package-relative config path from the model registry.
    """
    config_root = _config_root()
    candidate = Path(config).expanduser()
    parts = candidate.parts
    if ("runtime" in parts and "configs" in parts) and "videocrafter" in parts:
        runtime_index = parts.index("videocrafter")
        return (config_root / Path(*parts[runtime_index + 1 :])).resolve()

    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved != config_root and config_root not in resolved.parents:
            raise ValueError("VideoCrafter config must live under data/models/runtime/configs/videocrafter.")
        return resolved

    packaged = candidate.resolve()
    if packaged.is_file():
        return packaged

    return (config_root / candidate.name).resolve()


def _clear_open_clip_attn_mask(model) -> None:
    """Helper function to clear open clip attn mask.

    Args:
        model: The model.

    Returns:
        The return value.
    """
    cond_stage = getattr(model, "cond_stage_model", None)
    clip_model = getattr(cond_stage, "model", None)
    if clip_model is not None and hasattr(clip_model, "attn_mask"):
        clip_model.attn_mask = None


def resolve_runtime_checkpoint(ckpt_path: str | Path, model_name: str) -> Path:
    """Resolve legacy VideoCrafter filenames to staged Hub repository layouts."""

    requested = Path(ckpt_path).expanduser()
    if requested.is_file():
        return requested
    checkpoint_root = requested.parent.parent
    identity = f"{model_name} {requested.name}".lower()
    if "videocrafter2" in identity or "t2v_512_v2" in identity:
        candidate = checkpoint_root / "VideoCrafter--VideoCrafter2" / "model.ckpt"
    elif "image2video" in identity or "i2v" in identity:
        candidate = checkpoint_root / "VideoCrafter--Image2Video-512" / "model.ckpt"
    elif "text2video" in identity or "t2v_1024" in identity:
        candidate = checkpoint_root / "VideoCrafter--Text2Video-1024" / "model.ckpt"
    else:
        return requested
    return candidate if candidate.is_file() else requested


def _normalize_lvdm_config(model_config) -> None:
    params = model_config.get("params")
    if not params:
        return
    unet_config = params.get("unet_config")
    if not unet_config:
        return
    unet_params = unet_config.get("params")
    if not unet_params:
        return

    aliases = {
        "fps_cond": "fs_condition",
        "use_image_attention": "image_cross_attention",
    }
    for source_key, target_key in aliases.items():
        if source_key not in unet_params:
            continue
        value = unet_params.pop(source_key)
        if target_key not in unet_params:
            unet_params[target_key] = value


class VideoCrafter:
    """Video crafter implementation."""
    def __init__(
        self,
        model_name: str,
        config: str,
        ckpt_path: str,
        height: int,
        width: int,
        generation_type: Literal["t2v", "i2v"],
        frames: int = -1,
        fps: int = 8,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 12.0,
        seed: int = 123,
        device: Optional[str] = None,
    ):
        """
        Initialize VideoCrafter from packaged runtime code and external weights.

        Args:
            model_name: Registry id for the VideoCrafter model variant.
            config: In-tree inference config path.
            ckpt_path: External checkpoint asset path.
            height: Output frame height.
            width: Output frame width.
            generation_type: Text-to-video or image-to-video runtime mode.
            frames: Optional frame override, using model default when negative.
            fps: Conditioning frames per second.
            n_samples: Number of variants per prompt.
            ddim_steps: DDIM sampling step count.
            ddim_eta: DDIM eta value.
            unconditional_guidance_scale: Classifier-free guidance scale.
            seed: Sampling seed.
        """
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Error: image size [h,w] should be multiples of 16!")

        self.model_name = model_name
        self.config = str(resolve_runtime_config(config))
        self.ckpt_path = str(resolve_runtime_checkpoint(ckpt_path, model_name))
        self.height = height
        self.width = width
        self.frames = frames
        self.fps = fps
        self.generation_type = generation_type
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.seed = seed
        self.device = device
        self.model = None

        components = load_videocrafter_components()

        from omegaconf import OmegaConf

        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model_config = OmegaConf.load(self.config).pop("model", OmegaConf.create())
        _normalize_lvdm_config(model_config)
        model = components.instantiate_from_config(model_config)
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(target_device)
        if not Path(self.ckpt_path).is_file():
            raise FileNotFoundError(f"{self.ckpt_path} not found")

        self.model = components.load_model_checkpoint(model, self.ckpt_path).float().to(target_device)
        self.model.eval()

    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        """
        Generate video frames through the packaged VideoCrafter implementation.

        Args:
            prompt: Text prompt for video synthesis.
            image_path: Optional image conditioning path for image-to-video variants.
        """
        if self.model is None:
            raise RuntimeError("VideoCrafter runtime is not initialized.")

        components = load_videocrafter_components()
        import torch

        h, w = self.height // 8, self.width // 8
        frames = self.model.temporal_length if self.frames < 0 else self.frames
        channels = self.model.channels

        noise_shape = [1, channels, frames, h, w]
        fps = torch.tensor([self.fps]).to(self.model.device).long()

        prompts = [prompt]
        text_emb = self.model.get_learned_conditioning(prompts)

        if self.generation_type == "t2v":
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif self.generation_type == "i2v":
            cond_images = components.load_image_batch([image_path], (self.height, self.width))
            cond_images = cond_images.to(self.model.device)
            img_emb = self.model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        batch_samples = components.batch_ddim_sampling(
            self.model,
            cond,
            noise_shape,
            self.n_samples,
            self.ddim_steps,
            self.ddim_eta,
            self.unconditional_guidance_scale,
        )

        batch_samples = batch_samples.detach().squeeze().cpu()
        batch_samples = torch.clamp(batch_samples.float(), -1.0, 1.0)
        batch_samples = (batch_samples + 1.0) / 2.0
        return batch_samples.permute(1, 0, 2, 3)


__all__ = [
    "VideoCrafter",
    "resolve_runtime_checkpoint",
    "resolve_runtime_config",
]

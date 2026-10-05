"""Native MultiWorld inference on shared Wan model, VAE, and scheduler roles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.loaders import load_wan_vae_checkpoint
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WAN22_TI2V_5B_CONFIG
from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
    load_wan_vggt_environment_encoder,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import (
    MultiWorldWanModel,
)
from worldfoundry.base_models.diffusion_model.runners.staged import StagedDiffusionPipeline
from worldfoundry.core.model_loading import load_model, load_state_dict
from worldfoundry.core.nn import FlowMatchScheduler


def load_multiworld_config(path: str | Path) -> dict[str, Any]:
    """Read the inference-only fields from a released MultiWorld YAML config."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MultiWorld config not found: {resolved}")
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"MultiWorld config must contain a mapping: {resolved}")
    return value


def _native_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    simulator = config.get("simulator_config")
    if not isinstance(simulator, Mapping):
        raise KeyError("MultiWorld config is missing simulator_config")
    dit_config = simulator.get("dit_config")
    if not isinstance(dit_config, Mapping):
        raise KeyError("MultiWorld config is missing simulator_config.dit_config")
    params = dit_config.get("params")
    if not isinstance(params, Mapping):
        raise KeyError("MultiWorld config is missing simulator_config.dit_config.params")
    action = params.get("action_encoder_config")
    if not isinstance(action, Mapping):
        raise KeyError("MultiWorld config is missing action_encoder_config")
    action_params = action.get("params", action)
    if not isinstance(action_params, Mapping):
        raise TypeError("MultiWorld action_encoder_config.params must be a mapping")
    action_config = dict(action_params)
    action_config.pop("target", None)
    action_config["output_dim"] = WAN22_TI2V_5B_CONFIG["dim"]
    action_pe = dict(action_config.get("action_pe_config") or {})
    action_pe["dim"] = WAN22_TI2V_5B_CONFIG["dim"]
    action_config["action_pe_config"] = action_pe

    return {
        **WAN22_TI2V_5B_CONFIG,
        "seperated_timestep": bool(params.get("seperated_timestep", True)),
        "fuse_vae_embedding_in_latents": bool(
            params.get("fuse_vae_embedding_in_latents", True)
        ),
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "action_injection": str(params.get("action_injection", "bidi_cross_attention")),
        "action_encoder_config": action_config,
        "has_context_input": bool(params.get("has_context_input", True)),
    }


def _split_multiworld_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model_state: dict[str, torch.Tensor] = {}
    connector_state: dict[str, torch.Tensor] = {}
    for source, value in state_dict.items():
        name = source
        if name.startswith("pipe.env_encoder.connector."):
            connector_state[name.removeprefix("pipe.env_encoder.connector.")] = value
            continue
        if name.startswith("pipe.dit."):
            name = name.removeprefix("pipe.dit.")
        elif name.startswith("dit."):
            name = name.removeprefix("dit.")
        model_state[name] = value
    if not connector_state:
        raise KeyError("MultiWorld checkpoint does not contain the VGGT connector")
    return model_state, connector_state


def _to_device_tree(value: Any, device: str | torch.device) -> Any:
    if isinstance(value, Mapping):
        return {name: _to_device_tree(item, device) for name, item in value.items()}
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    dtype = torch.bfloat16 if tensor.is_floating_point() else torch.long
    return tensor.to(device=device, dtype=dtype)


class MultiWorldWanPipeline(StagedDiffusionPipeline):
    """ItTakesTwo sampling policy over explicit native model roles."""

    def __init__(
        self,
        *,
        dit: MultiWorldWanModel,
        vae: torch.nn.Module,
        env_encoder: torch.nn.Module,
        config: Mapping[str, Any],
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=vae.upsampling_factor * dit.patch_size[1],
            width_division_factor=vae.upsampling_factor * dit.patch_size[2],
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.dit = dit
        self.vae = vae
        self.env_encoder = env_encoder
        self.config = dict(config)
        self.scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.model_names = ["dit", "vae", "env_encoder"]

    def enable_vram_management(self, num_persistent_param_in_dit=None) -> None:
        del num_persistent_param_in_dit
        self.enable_cpu_offload()

    def _encode_first_frame(
        self,
        image: Image.Image,
        *,
        height: int,
        width: int,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        video = self.preprocess_image(image, pattern="C H W").unsqueeze(1)
        return self.vae.encode(
            [video],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def __call__(
        self,
        *,
        input_image: Image.Image,
        action: Mapping[str, Any],
        env_obv: torch.Tensor,
        seed: int | None = 0,
        rand_device: str | torch.device = "cpu",
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 35,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd=tqdm,
        show_progress: bool = True,
    ) -> list[Image.Image]:
        if input_image is None:
            raise ValueError("MultiWorld requires an input image")
        if not isinstance(action, Mapping):
            raise TypeError("MultiWorld actions must be a mapping")
        if env_obv is None:
            raise ValueError("MultiWorld requires an environment observation")
        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        latent_frames = (num_frames - 1) // 4 + 1
        latent_height = height // self.vae.upsampling_factor
        latent_width = width // self.vae.upsampling_factor
        latents = self.generate_noise(
            (1, self.vae.z_dim, latent_frames, latent_height, latent_width),
            seed=seed,
            rand_device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        self.load_models_to_device(["vae"])
        first_frame = self._encode_first_frame(
            input_image,
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, :1] = first_frame

        self.load_models_to_device(["env_encoder"])
        env_obv = _to_device_tree(env_obv, self.device)
        env_context = self.env_encoder(env_obv)

        self.load_models_to_device(["dit"])
        action = _to_device_tree(action, self.device)
        action_embeds = self.dit.action_encoder(action)
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        spatial_tokens = (latent_height // self.dit.patch_size[1]) * (
            latent_width // self.dit.patch_size[2]
        )
        for timestep in progress_bar_cmd(self.scheduler.timesteps, disable=not show_progress):
            frame_timestep = torch.full(
                (1, latent_frames),
                float(timestep),
                device=self.device,
                dtype=self.torch_dtype,
            )
            frame_timestep[:, 0] = 0
            token_timestep = frame_timestep.repeat_interleave(spatial_tokens, dim=1)
            prediction = self.dit(
                x=latents,
                timestep=token_timestep,
                context=None,
                action_embeds=action_embeds,
                env_context=env_context,
            )
            latents = self.scheduler.step(prediction, timestep, latents)
            latents[:, :, :1] = first_frame

        self.load_models_to_device(["vae"])
        decoded = self.vae.decode(
            latents.to(dtype=next(self.vae.parameters()).dtype),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self.load_models_to_device([])
        return self.vae_output_to_video(decoded)


def load_multiworld_pipeline(
    *,
    base_model_root: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    vggt_root: str | Path | None = None,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> MultiWorldWanPipeline:
    """Load released MultiWorld roles without a DiffSynth/model-manager backend."""

    config = load_multiworld_config(config_path)
    checkpoint = load_state_dict(
        str(Path(checkpoint_path).expanduser().resolve()),
        torch_dtype=torch_dtype,
        device="cpu",
    )
    model_state, connector_state = _split_multiworld_checkpoint(checkpoint)
    del checkpoint
    dit = load_model(
        MultiWorldWanModel,
        path=None,
        config=_native_model_config(config),
        state_dict=model_state,
        torch_dtype=torch_dtype,
        device="cpu",
        strict=True,
    )
    del model_state
    env_config = config.get("simulator_config", {}).get("env_encoder_config", {}).get("params", {})
    env_encoder = load_wan_vggt_environment_encoder(
        connector_state,
        checkpoint_root=vggt_root,
        input_dim=int(env_config.get("input_dim", 2048)),
        output_dim=int(env_config.get("output_dim", WAN22_TI2V_5B_CONFIG["dim"])),
        torch_dtype=torch_dtype,
        device="cpu",
    )
    del connector_state
    vae = load_wan_vae_checkpoint(
        base_model_root,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    pipeline = MultiWorldWanPipeline(
        dit=dit,
        vae=vae,
        env_encoder=env_encoder,
        config=config,
        device=device,
        torch_dtype=torch_dtype,
    )
    if str(device) != "cpu":
        pipeline.enable_cpu_offload()
    return pipeline


__all__ = [
    "MultiWorldWanPipeline",
    "load_multiworld_config",
    "load_multiworld_pipeline",
]

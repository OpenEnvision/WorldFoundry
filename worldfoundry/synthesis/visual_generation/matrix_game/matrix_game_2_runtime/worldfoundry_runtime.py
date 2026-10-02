from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file

from worldfoundry.core.utils.tensors.torch import set_seed_everywhere as set_seed
from worldfoundry.core.media.artifacts import process_game_control_video as process_video
from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path
from worldfoundry.evaluation.utils import worldfoundry_data_path


MATRIX_GAME_2_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "matrix_game_2")
MODE_CHECKPOINTS = {
    "universal": ("base_distilled_model", "base_distill.safetensors"),
    "gta_drive": ("gta_distilled_model", "gta_keyboard2dim.safetensors"),
    "templerun": ("templerun_distilled_model", "templerun_7dim_onlykey.safetensors"),
}
MODE_CONFIGS = {
    "universal": "inference_yaml/inference_universal.yaml",
    "gta_drive": "inference_yaml/inference_gta_drive.yaml",
    "templerun": "inference_yaml/inference_templerun.yaml",
}


def _hf_snapshot_dirs(root: Path) -> list[Path]:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def _matrix_game2_roots(primary: str | Path | None = None) -> list[Path]:
    raw: list[Path] = []
    if primary not in {None, ""}:
        raw.append(Path(str(primary)).expanduser())
    raw.extend(
        [
            checkpoint_root_path("Matrix-Game-2.0"),
            hfd_root_path("Skywork--Matrix-Game-2.0"),
            hfd_root_path("custom--Matrix-Game-2.0"),
            checkpoint_root_path("huggingface", "hub", "models--Skywork--Matrix-Game-2.0"),
            checkpoint_root_path("hf_home", "hub", "models--Skywork--Matrix-Game-2.0"),
        ]
    )

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in raw:
        if root.is_file():
            root = root.parent
        for candidate in [root, *_hf_snapshot_dirs(root)]:
            key = str(candidate)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
    return candidates


def _matrix_game2_layout_complete(root: Path, mode: str) -> bool:
    rel_dir, filename = MODE_CHECKPOINTS[mode]
    required = [
        root / "Wan2.1_VAE.pth",
        root / "xlm-roberta-large",
        root / rel_dir / filename,
    ]
    return all(path.exists() for path in required)


def _resolve_model_root(path_value: str | Path, mode: str) -> str:
    for candidate in _matrix_game2_roots(path_value):
        if _matrix_game2_layout_complete(candidate, mode):
            return str(candidate.resolve())
    for candidate in _matrix_game2_roots(path_value):
        if candidate.is_dir():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "Matrix-Game-2 requires a local checkpoint directory. "
        f"Checked: {[str(path) for path in _matrix_game2_roots(path_value)]}"
    )


def _enable_torch_compile() -> bool:
    return os.environ.get("WORLDFOUNDRY_ENABLE_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}


class MatrixGame2Runtime:
    def __init__(
        self,
        pipeline,
        vae,
        weight_dtype: torch.dtype | None = None,
        mode: str = "universal",
        device: str = "cuda",
    ):
        """
        the mode including "gta_drive", "templerun", "universal"
        """
        self.pipeline = pipeline
        self.vae = vae
        self.weight_dtype = weight_dtype or torch.bfloat16
        self.device = device
        self.mode = mode

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        mode: str = "universal",
        device=None,
        weight_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "MatrixGame2Runtime":
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        if mode not in ["universal", "gta_drive", "templerun"]:
            raise NotImplementedError("mode should be one of ['universal', 'gta_drive', 'templerun']")
        weight_dtype = weight_dtype or torch.bfloat16
        config_path = str(MATRIX_GAME_2_CONFIG_ROOT / MODE_CONFIGS[mode])

        config = OmegaConf.load(config_path)
        model_config = config["model_kwargs"]["model_config"]
        model_config_path = (
            os.fspath(model_config)
            if os.path.isabs(os.fspath(model_config))
            else os.fspath(MATRIX_GAME_2_CONFIG_ROOT / os.fspath(model_config))
        )
        if os.path.isdir(model_config_path):
            model_config_path = os.path.join(model_config_path, "config.yaml")
        if model_config_path.endswith((".yaml", ".yml")):
            config["model_kwargs"]["model_config"] = OmegaConf.to_container(
                OmegaConf.load(model_config_path),
                resolve=True,
            )
        else:
            config["model_kwargs"]["model_config"] = model_config_path

        model_root = _resolve_model_root(pretrained_model_path, mode)

        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
            VAEDecoderWrapper,
        )
        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.extension_modules.wanx_vae.wanx_vae import (
            get_wanx_vae_wrapper,
        )
        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline import (
            CausalInferencePipeline,
        )
        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.wan_wrapper import (
            WanDiffusionWrapper,
        )

        generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(model_root, "Wan2.1_VAE.pth"),
            map_location="cpu",
            weights_only=True,
        )
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if "decoder." in key or "conv2" in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        if _enable_torch_compile():
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=current_vae_decoder)

        resolved_checkpoint_path = cls._resolve_checkpoint_path(
            model_root=model_root,
            mode=mode,
            checkpoint_path=checkpoint_path,
        )
        print(f"Loading Pretrained Model from {resolved_checkpoint_path}...")
        state_dict = load_file(resolved_checkpoint_path)
        pipeline.generator.load_state_dict(state_dict)

        pipeline = pipeline.to(device=device, dtype=weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(model_root, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, weight_dtype)

        return cls(pipeline=pipeline, vae=vae, weight_dtype=weight_dtype, mode=mode, device=device)

    @staticmethod
    def _resolve_checkpoint_path(model_root: str, mode: str, checkpoint_path: str | None = None) -> str:
        if checkpoint_path is None:
            rel_dir, filename = MODE_CHECKPOINTS[mode]
            resolved = os.path.join(model_root, rel_dir, filename)
        elif os.path.isabs(checkpoint_path):
            resolved = checkpoint_path
        else:
            resolved = os.path.join(model_root, checkpoint_path)

        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Matrix-Game-2 checkpoint not found for mode='{mode}': {resolved}")
        return resolved

    @torch.no_grad()
    def predict(
        self,
        cond_concat,
        visual_context,
        operator_condition,
        num_output_frames,
        operation_visualization=True,
        seed=None,
    ):
        if seed is not None:
            # MG2's denoising loop also samples fresh noise inside the pipeline,
            # so we need to seed the global RNG chain, not only the initial tensor.
            set_seed(int(seed))
        sampled_noise = torch.randn(
            [1, 16, num_output_frames, cond_concat.size(-2), cond_concat.size(-1)],
            device=self.device,
            dtype=self.weight_dtype,
        )

        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
        }
        if "mouse_condition" in operator_condition:
            mouse_condition = operator_condition["mouse_condition"].unsqueeze(0).to(
                device=self.device,
                dtype=self.weight_dtype,
            )
            conditional_dict["mouse_cond"] = mouse_condition
        if "keyboard_condition" not in operator_condition:
            raise ValueError("keyboard_condition must be provided in operator_condition")
        keyboard_condition = operator_condition["keyboard_condition"].unsqueeze(0).to(
            device=self.device,
            dtype=self.weight_dtype,
        )
        conditional_dict["keyboard_cond"] = keyboard_condition

        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=self.mode,
                profile=False,
            )

        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)

        mouse_icon = None
        if self.mode != "templerun":
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy(),
            )
        else:
            config = keyboard_condition[0].float().cpu().numpy()
        output_video = process_video(
            video.astype(np.uint8),
            config,
            mouse_icon,
            mouse_scale=0.1,
            process_icon=operation_visualization,
            mode=self.mode,
        )
        return output_video


__all__ = [
    "MATRIX_GAME_2_CONFIG_ROOT",
    "MODE_CHECKPOINTS",
    "MODE_CONFIGS",
    "MatrixGame2Runtime",
    "process_video",
]

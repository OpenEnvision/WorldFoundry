"""WorldFoundry runtime for Stable Video Infinity 2.0."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

SVI_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)


def normalize_svi_lora_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the training-pipeline prefix used by the released SVI LoRA.

    The official ``ModelManager.load_lora_v2`` retries SVI checkpoints after
    stripping ``pipe.dit.`` because the adapter is merged into the bare Wan
    DiT.  Keeping the normalization explicit here also lets the runtime verify
    that every released LoRA pair found a native DiT module.
    """

    prefix = "pipe.dit."
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix(prefix)
        if normalized_key in normalized:
            raise ValueError(f"SVI LoRA key collision after removing {prefix!r}: {normalized_key!r}")
        normalized[normalized_key] = value
    return normalized


def normalize_prompt_stream(prompt: str | Sequence[str]) -> list[str]:
    """Normalize one prompt or a per-segment prompt stream."""

    if isinstance(prompt, str):
        prompts = [prompt]
    elif isinstance(prompt, Sequence):
        prompts = [str(item) for item in prompt]
    else:
        raise TypeError("SVI prompt must be a string or a sequence of strings.")
    prompts = [item.strip() for item in prompts]
    if not prompts or any(not item for item in prompts):
        raise ValueError("SVI prompts must be non-empty strings.")
    return prompts


def calculate_dimensions(image: Any, *, max_width: int) -> tuple[int, int]:
    """Match the official aspect-preserving, multiple-of-16 resize rule."""

    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"SVI received an invalid image size: {image.size!r}.")
    width = min(original_width, int(max_width))
    height = int(original_height * width / original_width)
    width = max(16, width // 16 * 16)
    height = max(16, height // 16 * 16)
    return height, width


class StableVideoInfinityRuntime:
    """Load SVI's LoRA over Wan2.1-I2V-14B and generate linked clips."""

    def __init__(
        self,
        wan_model_dir: str,
        svi_lora_path: str,
        *,
        model_name: str = "stable-video-infinity",
        device: str = "cuda",
        dtype: str = "bfloat16",
        fps: int = 24,
        num_clips: int = 999,
        num_frames: int = 81,
        num_motion_frames: int = 5,
        num_inference_steps: int = 50,
        cfg_scale_text: float = 5.0,
        sigma_shift: float = 5.0,
        lora_alpha: float = 1.0,
        ref_pad_cfg: bool = False,
        ref_pad_num: int = -1,
        prompt_repeat_times: int = 2,
        use_first_prompt_only: bool = False,
        repeat_first_clip: bool = False,
        prompt_prefix: str | None = None,
        base_seed: int | None = 0,
        seed_stride: int = 42,
        max_width: int = 832,
        height: int | None = None,
        width: int | None = None,
        tiled: bool = False,
        tile_size: Sequence[int] = (30, 52),
        tile_stride: Sequence[int] = (15, 26),
        enable_vram_management: bool = True,
        num_persistent_param_in_dit: int = 6_000_000_000,
        use_usp: bool = False,
        negative_prompt: str = SVI_NEGATIVE_PROMPT,
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)
        self.dtype_name = str(dtype)
        self.fps = int(fps)
        self.num_clips = int(num_clips)
        self.num_frames = int(num_frames)
        self.num_motion_frames = int(num_motion_frames)
        self.num_inference_steps = int(num_inference_steps)
        self.cfg_scale_text = float(cfg_scale_text)
        self.sigma_shift = float(sigma_shift)
        self.lora_alpha = float(lora_alpha)
        self.ref_pad_cfg = bool(ref_pad_cfg)
        self.ref_pad_num = int(ref_pad_num)
        self.prompt_repeat_times = int(prompt_repeat_times)
        self.use_first_prompt_only = bool(use_first_prompt_only)
        self.repeat_first_clip = bool(repeat_first_clip)
        normalized_prefix = str(prompt_prefix).strip() if prompt_prefix else ""
        self.prompt_prefix = None if not normalized_prefix or normalized_prefix.lower() == "none" else normalized_prefix
        self.base_seed = None if base_seed is None else int(base_seed)
        self.seed_stride = int(seed_stride)
        self.max_width = int(max_width)
        self.height = None if height is None else int(height)
        self.width = None if width is None else int(width)
        self.tiled = bool(tiled)
        self.tile_size = tuple(int(item) for item in tile_size)
        self.tile_stride = tuple(int(item) for item in tile_stride)
        self.negative_prompt = str(negative_prompt)

        self._validate_options()
        self.wan_model_dir = Path(wan_model_dir).expanduser().resolve()
        self.svi_lora_path = Path(svi_lora_path).expanduser().resolve()
        model_files, lora_files = self._resolve_weight_files()
        self.pipeline = self._load_pipeline(
            model_files=model_files,
            lora_files=lora_files,
            enable_vram_management=bool(enable_vram_management),
            num_persistent_param_in_dit=int(num_persistent_param_in_dit),
            use_usp=bool(use_usp),
        )

    def _validate_options(self) -> None:
        if self.fps <= 0:
            raise ValueError("SVI fps must be positive.")
        if self.num_clips <= 0:
            raise ValueError("SVI num_clips must be positive.")
        if self.num_frames <= 0:
            raise ValueError("SVI num_frames must be positive.")
        if not 1 <= self.num_motion_frames <= self.num_frames:
            raise ValueError("SVI num_motion_frames must be within the generated segment.")
        if self.num_inference_steps <= 0:
            raise ValueError("SVI num_inference_steps must be positive.")
        if self.prompt_repeat_times <= 0:
            raise ValueError("SVI prompt_repeat_times must be positive.")
        if self.ref_pad_num < -1:
            raise ValueError("SVI ref_pad_num must be -1, 0, or a positive integer.")
        if self.max_width < 16:
            raise ValueError("SVI max_width must be at least 16 pixels.")
        if (self.height is None) != (self.width is None):
            raise ValueError("SVI height and width must be provided together.")
        if self.height is not None and (self.height <= 0 or self.width <= 0):
            raise ValueError("SVI height and width must be positive.")
        if len(self.tile_size) != 2 or len(self.tile_stride) != 2:
            raise ValueError("SVI tile_size and tile_stride must each contain two integers.")
        if any(value <= 0 for value in (*self.tile_size, *self.tile_stride)):
            raise ValueError("SVI tile_size and tile_stride values must be positive.")

    def _resolve_weight_files(self) -> tuple[dict[str, Any], list[Path]]:
        root = self.wan_model_dir
        if not root.is_dir():
            raise FileNotFoundError(f"SVI Wan model directory not found: {root}")

        image_encoder = root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        text_encoder = root / "models_t5_umt5-xxl-enc-bf16.pth"
        vae = root / "Wan2.1_VAE.pth"
        tokenizer = root / "google" / "umt5-xxl"
        missing = [path for path in (image_encoder, text_encoder, vae) if not path.is_file()]
        if not tokenizer.is_dir():
            missing.append(tokenizer)
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"SVI is missing Wan2.1 component(s): {joined}")

        dit_files = sorted(root.glob("diffusion_pytorch_model-*.safetensors"))
        single_dit = root / "diffusion_pytorch_model.safetensors"
        if not dit_files and single_dit.is_file():
            dit_files = [single_dit]
        if not dit_files:
            raise FileNotFoundError(
                f"SVI Wan DiT shards were not found under {root}; expected diffusion_pytorch_model*.safetensors."
            )

        if self.svi_lora_path.is_file():
            lora_files = [self.svi_lora_path]
        elif self.svi_lora_path.is_dir():
            lora_files = sorted(self.svi_lora_path.glob("*.safetensors"))
        else:
            raise FileNotFoundError(f"SVI LoRA path not found: {self.svi_lora_path}")
        if not lora_files:
            raise FileNotFoundError(f"No SVI .safetensors files found in {self.svi_lora_path}")

        return {
            "image_encoder": image_encoder,
            "dit": dit_files,
            "text_encoder": text_encoder,
            "vae": vae,
            "tokenizer": tokenizer,
        }, lora_files

    def _load_pipeline(
        self,
        *,
        model_files: dict[str, Any],
        lora_files: list[Path],
        enable_vram_management: bool,
        num_persistent_param_in_dit: int,
        use_usp: bool,
    ) -> Any:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Stable Video Infinity requires CUDA, but CUDA is unavailable.")
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            torch_dtype = dtype_by_name[self.dtype_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported SVI dtype: {self.dtype_name!r}") from exc

        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import (
            WanVideoVAE,
            WanVideoVAEStateDictConverter,
        )
        from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
            WAN21_I2V_14B_CONFIG,
            convert_diffusers_wan_transformer_state_dict,
        )
        from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
            WanImageEncoder,
            WanImageEncoderStateDictConverter,
            WanTextEncoder,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
        from worldfoundry.core.model_loading import GeneralLoRALoader, load_model, load_state_dict

        from .official_pipeline import SVIVideoPipeline

        image_encoder = load_model(
            WanImageEncoder,
            str(model_files["image_encoder"]),
            torch_dtype=torch.float32,
            device="cpu",
            state_dict_converter=WanImageEncoderStateDictConverter().from_civitai,
        )
        dit = load_model(
            WanModel,
            [str(path) for path in model_files["dit"]],
            config=WAN21_I2V_14B_CONFIG,
            torch_dtype=torch_dtype,
            device="cpu",
            # ``load_model`` accepts a mapping-only converter.  The converter
            # object's public API additionally returns inferred config as a
            # tuple for model-manager callers, which is not valid here.
            state_dict_converter=convert_diffusers_wan_transformer_state_dict,
        )
        text_encoder = load_model(
            WanTextEncoder,
            str(model_files["text_encoder"]),
            torch_dtype=torch_dtype,
            device="cpu",
        )
        vae = load_model(
            WanVideoVAE,
            str(model_files["vae"]),
            torch_dtype=torch_dtype,
            device="cpu",
            state_dict_converter=WanVideoVAEStateDictConverter().from_civitai,
        )
        # Match the official SVI ``LoRAFromCivitai`` merge exactly: released
        # LoRA tensors are FP32, A/B products are evaluated as FP16 on CUDA,
        # and only the finished delta is cast into the BF16 Wan base weight.
        # Loading and multiplying the adapters as BF16 measurably changes the
        # 50-step denoising trajectory.
        lora_loader = GeneralLoRALoader(device=self.device, torch_dtype=torch.float16)
        for lora_path in lora_files:
            lora_state_dict = normalize_svi_lora_state_dict(
                load_state_dict(str(lora_path), torch_dtype=torch.float16, device="cpu")
            )
            expected_pairs = lora_loader.get_name_dict(lora_state_dict)
            if not expected_pairs:
                raise RuntimeError(f"SVI LoRA contains no supported A/B or up/down pairs: {lora_path}")
            updated = lora_loader.load(dit, lora_state_dict, alpha=self.lora_alpha)
            if updated != len(expected_pairs):
                module_names = {name for name, module in dit.named_modules() if hasattr(module, "weight")}
                missing = sorted(set(expected_pairs).difference(module_names))
                preview = ", ".join(missing[:5])
                raise RuntimeError(
                    f"SVI merged {updated}/{len(expected_pairs)} LoRA tensors from {lora_path}; "
                    f"unmatched Wan DiT modules: {preview or 'unknown'}"
                )

        pipeline = SVIVideoPipeline.from_components(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            dit=dit,
            vae=vae,
            tokenizer_path=str(model_files["tokenizer"]),
            torch_dtype=torch_dtype,
            device=self.device,
            use_usp=use_usp,
        )
        required = {
            "Wan image encoder": pipeline.image_encoder,
            "Wan text encoder": pipeline.text_encoder,
            "Wan DiT": pipeline.dit,
            "Wan VAE": pipeline.vae,
        }
        unavailable = [name for name, value in required.items() if value is None]
        if unavailable:
            raise RuntimeError("SVI failed to load: " + ", ".join(unavailable))

        if enable_vram_management:
            pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
        else:
            pipeline.to(self.device)
        pipeline.eval()
        return pipeline

    def _prompt_for_clip(self, prompts: list[str], clip_index: int) -> str:
        if self.use_first_prompt_only or len(prompts) == 1:
            value = prompts[0]
        else:
            prompt_index = (clip_index // self.prompt_repeat_times) % len(prompts)
            value = prompts[prompt_index]
        if self.prompt_prefix:
            return f"{self.prompt_prefix}, {value}"
        return value

    def _clip_count(self, prompts: list[str]) -> int:
        if self.use_first_prompt_only:
            return self.num_clips
        return min(self.num_clips, len(prompts) * self.prompt_repeat_times)

    def generate_video(
        self,
        prompt: str | Sequence[str],
        image_path: str | None = None,
    ) -> list[Any]:
        """Generate and concatenate the autoregressive SVI clip sequence."""

        if image_path is None:
            raise ValueError("Stable Video Infinity requires an input image.")
        from PIL import Image

        reference = Image.open(image_path).convert("RGB")
        if self.height is None:
            height, width = calculate_dimensions(reference, max_width=self.max_width)
        else:
            height = max(16, self.height // 16 * 16)
            width = max(16, self.width // 16 * 16)
        reference = reference.resize((width, height))

        prompts = normalize_prompt_stream(prompt)
        clip_count = self._clip_count(prompts)
        generated: list[Any] = []
        continuation: list[Any] = [reference] * self.num_motion_frames if self.repeat_first_clip else [reference]

        for clip_index in range(clip_count):
            seed = None
            if self.base_seed is not None:
                seed = self.base_seed + clip_index * self.seed_stride
            segment = self.pipeline(
                prompt=self._prompt_for_clip(prompts, clip_index),
                negative_prompt=self.negative_prompt,
                input_image=continuation,
                random_ref_frame=reference,
                ref_pad_cfg=self.ref_pad_cfg,
                ref_pad_num=self.ref_pad_num,
                seed=seed,
                height=height,
                width=width,
                num_frames=self.num_frames,
                cfg_scale=self.cfg_scale_text,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            )
            if len(segment) < self.num_motion_frames:
                raise RuntimeError(
                    f"SVI generated {len(segment)} frame(s), fewer than the "
                    f"{self.num_motion_frames} continuation frames requested."
                )
            continuation = list(segment[-self.num_motion_frames :])
            if clip_index + 1 < clip_count:
                generated.extend(segment[: -self.num_motion_frames])
            else:
                generated.extend(segment)
        return generated


__all__ = [
    "SVI_NEGATIVE_PROMPT",
    "StableVideoInfinityRuntime",
    "calculate_dimensions",
    "normalize_svi_lora_state_dict",
    "normalize_prompt_stream",
]

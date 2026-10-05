"""Resident inference runtime for ABot-World."""

from __future__ import annotations

import gc
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.abot_world import (
    ABotWorldModel,
)
from worldfoundry.base_models.diffusion_model.schedulers import (
    InferenceFlowMatchScheduler,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.tae_22 import (
    TAEW22StreamingDecoder,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import (
    T5EncoderModel,
)
from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8
from worldfoundry.core.kernels.capabilities import kernel_device_profile
from worldfoundry.core.media.processing.image_utils import load_pil_image


ACTION_KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
REFERENCE_SLOTS = ("head", "left", "right", "front", "back")


def _normalize_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).strip().lower().removeprefix("torch.")
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported ABot-World dtype: {dtype!r}") from exc


class ABotWorldInference:
    """Inference-only causal rollout with a resident DiT and KV cache."""

    required_checkpoint_paths = (
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
        "taew2_2.pth",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
    )

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
        height: int = 704,
        width: int = 1280,
        num_frame_per_block: int = 3,
        local_attn_size: int = 21,
        ref_num_slots: int = 5,
        ref_resolution: int = 512,
        denoising_steps: Sequence[int] = (1000, 750, 500, 250),
        timestep_shift: float = 5.0,
        context_noise: int = 0,
        use_relative_rope: bool = True,
        use_fp8: bool = True,
        offload_conditioners: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self._validate_checkpoint()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ABot-World inference requires a CUDA device")
        self.dtype = _normalize_dtype(dtype)
        self.height = int(height)
        self.width = int(width)
        self.num_frame_per_block = int(num_frame_per_block)
        self.local_attn_size = int(local_attn_size)
        self.ref_num_slots = int(ref_num_slots)
        self.ref_resolution = int(ref_resolution)
        self.context_noise = int(context_noise)
        self.use_relative_rope = bool(use_relative_rope)
        self.offload_conditioners = bool(offload_conditioners)
        if self.height % 32 or self.width % 32:
            raise ValueError("ABot-World height and width must be divisible by 32")
        if self.ref_resolution % 32:
            raise ValueError("ABot-World reference resolution must be divisible by 32")
        if self.num_frame_per_block != 3:
            raise ValueError("The released ABot-World checkpoint uses three latent frames per block")
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive for resident ABot-World inference")
        if not 1 <= self.ref_num_slots <= len(REFERENCE_SLOTS):
            raise ValueError("ref_num_slots must be between one and five")
        if len(denoising_steps) == 0:
            raise ValueError("denoising_steps must not be empty")

        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=str(
                self.checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
            ),
            tokenizer_path=str(self.checkpoint_dir / "google" / "umt5-xxl"),
        )
        self.decoder = (
            TAEW22StreamingDecoder(self.checkpoint_dir / "taew2_2.pth")
            .eval()
            .requires_grad_(False)
            .to(device=self.device, dtype=self.dtype)
        )
        self.decoder_compute_dtype = torch.float16

        logging.info("Loading ABot-World causal DiT from %s", self.checkpoint_dir)
        self.model = ABotWorldModel.from_pretrained(
            str(self.checkpoint_dir),
            torch_dtype=self.dtype,
            model_type="ci2v",
            local_attn_size=self.local_attn_size,
            num_frame_per_block=self.num_frame_per_block,
            use_relative_rope=self.use_relative_rope,
            downscale_factor_control_adapter=16,
            low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
        self.fp8_enabled = False
        if use_fp8:
            profile = kernel_device_profile(self.device)
            if profile.supports_fp8 and callable(getattr(torch, "_scaled_mm", None)):
                replaced = replace_linear_with_float8(
                    self.model,
                    min_features=1024,
                    keep_dense_fallback=False,
                )
                self.fp8_enabled = replaced > 0
                logging.info("ABot-World uses %d in-tree FP8 linear layers", replaced)
            else:
                warnings.warn(
                    "FP8 was requested but is unavailable on this device; using dense weights",
                    stacklevel=2,
                )
        self.model.to(self.device)

        self.scheduler = InferenceFlowMatchScheduler(
            shift=float(timestep_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000)
        warped = torch.cat(
            (self.scheduler.timesteps.cpu(), torch.tensor([0.0], dtype=torch.float32))
        )
        indices = 1000 - torch.as_tensor(tuple(denoising_steps), dtype=torch.long)
        if (indices < 0).any() or (indices >= warped.numel()).any():
            raise ValueError("denoising_steps must be in the range [0, 1000]")
        self.denoising_steps = warped[indices]

        self._prompt_cache: dict[str, list[torch.Tensor]] = {}
        self._configured = False
        self._kv_cache: list[dict[str, Any]] = []
        self._cross_attention_cache: list[dict[str, Any]] = []
        self._prompt_context: list[torch.Tensor] = []
        self._first_frame_latent: torch.Tensor | None = None
        self._reference_latents: torch.Tensor | None = None
        self._reference_mask: torch.Tensor | None = None
        self._generator: torch.Generator | None = None
        self.current_start_frame = 0

    def _validate_checkpoint(self) -> None:
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(f"ABot-World checkpoint directory not found: {self.checkpoint_dir}")
        missing = [
            relative
            for relative in self.required_checkpoint_paths
            if not (self.checkpoint_dir / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "ABot-World checkpoint is incomplete; missing: " + ", ".join(missing)
            )

    @staticmethod
    def _empty_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def _conditioning_phase(self):
        offload = self.offload_conditioners and self.device.type == "cuda"
        if offload:
            self.model.to("cpu")
            self._empty_cuda_cache()
        try:
            yield
        finally:
            if offload:
                self.model.to(self.device)
                self._empty_cuda_cache()

    def _encode_prompt(self, prompt: str) -> list[torch.Tensor]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            return [value.to(device=self.device, dtype=self.dtype) for value in cached]
        model = self.text_encoder.model
        model.to(self.device)
        try:
            context = self.text_encoder([prompt], self.device)
            cached = [value.detach().cpu() for value in context]
            self._prompt_cache[prompt] = cached
            return [value.to(device=self.device, dtype=self.dtype) for value in cached]
        finally:
            model.to("cpu")
            self._empty_cuda_cache()

    @staticmethod
    def _image_tensor(
        image: Image.Image,
        size: tuple[int, int],
        *,
        fit: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if fit:
            image = ImageOps.fit(
                image.convert("RGB"),
                size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        else:
            image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        array = np.array(image, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype).div_(127.5).sub_(1.0).unsqueeze(2)

    def _reference_images(self, references: Any) -> dict[int, Image.Image]:
        if references is None:
            return {}
        if isinstance(references, Mapping):
            normalized = {str(name).strip().lower(): value for name, value in references.items()}
            allowed = set(REFERENCE_SLOTS[: self.ref_num_slots])
            unknown = set(normalized).difference(allowed)
            if unknown:
                raise ValueError(
                    "Unknown ABot-World reference slots: " + ", ".join(sorted(unknown))
                )
            return {
                index: load_pil_image(normalized[name], first_sequence_item=False)
                for index, name in enumerate(REFERENCE_SLOTS[: self.ref_num_slots])
                if name in normalized and normalized[name] is not None
            }
        if torch.is_tensor(references) or isinstance(
            references,
            (str, Path, Image.Image, np.ndarray),
        ):
            references = [references]
        if not isinstance(references, Sequence):
            raise TypeError("reference_images must be a mapping or sequence")
        if len(references) > self.ref_num_slots:
            raise ValueError(f"ABot-World accepts at most {self.ref_num_slots} reference images")
        return {
            index: load_pil_image(value, first_sequence_item=False)
            for index, value in enumerate(references)
            if value is not None
        }

    def _encode_images(
        self,
        first_image: Image.Image,
        reference_images: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import (
            Wan2_2_VAE,
        )

        encoder = Wan2_2_VAE(
            vae_pth=str(self.checkpoint_dir / "Wan2.2_VAE.pth"),
            z_dim=48,
            dtype=self.dtype,
            device="cpu",
        )
        encoder.model.to(device=self.device, dtype=self.dtype)
        encoder.scale = [value.to(device=self.device, dtype=self.dtype) for value in encoder.scale]

        def encode(image: Image.Image, size: tuple[int, int], *, fit: bool) -> torch.Tensor:
            pixels = self._image_tensor(
                image,
                size,
                fit=fit,
                device=self.device,
                dtype=self.dtype,
            )
            latent = encoder.encode(pixels)
            if not latent:
                raise RuntimeError("Wan2.2 VAE did not produce an image latent")
            return latent[0].permute(1, 0, 2, 3).unsqueeze(0).to(self.dtype)

        try:
            first = encode(first_image, (self.width, self.height), fit=True)
            reference_height = self.ref_resolution // 16
            references = torch.zeros(
                1,
                self.ref_num_slots,
                48,
                1,
                reference_height,
                reference_height,
                device=self.device,
                dtype=self.dtype,
            )
            mask = torch.zeros(
                1,
                self.ref_num_slots,
                device=self.device,
                dtype=torch.float32,
            )
            for index, image in self._reference_images(reference_images).items():
                latent = encode(
                    image,
                    (self.ref_resolution, self.ref_resolution),
                    fit=False,
                )
                references[:, index] = latent.permute(0, 2, 1, 3, 4)
                mask[:, index] = 1
            return first, references, mask
        finally:
            del encoder
            gc.collect()
            self._empty_cuda_cache()

    def _initialize_caches(self, reference_token_len: int) -> None:
        latent_height = self.height // 16
        latent_width = self.width // 16
        frame_sequence_length = (latent_height // 2) * (latent_width // 2)
        video_tokens = self.local_attn_size * frame_sequence_length
        cache_shape = (
            1,
            reference_token_len + video_tokens,
            self.model.num_heads,
            self.model.dim // self.model.num_heads,
        )
        key_name = "k_raw" if self.use_relative_rope else "k"
        self._kv_cache = [
            {
                key_name: torch.empty(cache_shape, device=self.device, dtype=self.dtype),
                "v": torch.empty(cache_shape, device=self.device, dtype=self.dtype),
                "global_end_index": torch.zeros(1, device=self.device, dtype=torch.long),
                "local_end_index": torch.zeros(1, device=self.device, dtype=torch.long),
            }
            for _ in range(self.model.num_layers)
        ]
        self._cross_attention_cache = [
            {"is_init": False} for _ in range(self.model.num_layers)
        ]
        self._frame_sequence_length = frame_sequence_length

    @torch.no_grad()
    def configure(
        self,
        image: Any,
        *,
        prompt: str = "",
        reference_images: Any = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        if self._configured:
            self.reset()
        first_image = load_pil_image(image)
        with self._conditioning_phase():
            self._prompt_context = self._encode_prompt(str(prompt or ""))
            (
                self._first_frame_latent,
                self._reference_latents,
                self._reference_mask,
            ) = self._encode_images(first_image, reference_images)
        reference_token_len = (
            self.ref_num_slots
            * (self._reference_latents.shape[3] // self.model.patch_size[0])
            * (self._reference_latents.shape[4] // self.model.patch_size[1])
            * (self._reference_latents.shape[5] // self.model.patch_size[2])
        )
        self._initialize_caches(int(reference_token_len))
        self.decoder.reset()
        self.current_start_frame = 0
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._configured = True
        return {
            "height": self.height,
            "width": self.width,
            "latent_frames_per_block": self.num_frame_per_block,
            "first_output_frames": 9,
            "output_frames_per_block": 12,
            "fp8": self.fp8_enabled,
        }

    def _action_tensor(self, action: Any) -> torch.Tensor:
        if torch.is_tensor(action):
            value = action.detach().to(device=self.device, dtype=self.dtype)
            if value.ndim == 1 and value.numel() in {8, 32}:
                value = value.view(1, -1, 1, 1, 1)
            if value.ndim != 5 or value.shape[1] not in {8, 32}:
                raise ValueError("action tensor must contain 8 or 32 channels")
            expected = (1, self.num_frame_per_block, self.height, self.width)
            actual = (value.shape[0], value.shape[2], value.shape[3], value.shape[4])
            if any(got not in {1, want} for got, want in zip(actual, expected)):
                raise ValueError(
                    "action tensor batch/time/height/width must be singleton or "
                    f"match {expected}, got {actual}"
                )
            if value.shape[1] == 8:
                value = value.repeat_interleave(4, dim=1)
            return value.expand(
                1,
                32,
                self.num_frame_per_block,
                self.height,
                self.width,
            )
        if action is None:
            active: set[str] = set()
        elif isinstance(action, Mapping):
            active = {
                str(key).strip().upper()
                for key, enabled in action.items()
                if enabled
            }
        elif isinstance(action, str):
            normalized = action.strip().upper()
            for separator in (",", "+", "|", "/"):
                normalized = normalized.replace(separator, " ")
            tokens = normalized.split()
            if len(tokens) == 1 and set(tokens[0]).issubset(ACTION_KEYS):
                active = set(tokens[0])
            else:
                active = set(tokens)
        else:
            active = {str(key).strip().upper() for key in action}
        unknown = active.difference(ACTION_KEYS)
        if unknown:
            raise ValueError("Unknown ABot-World actions: " + ", ".join(sorted(unknown)))
        bits = torch.tensor(
            [float(key in active) for key in ACTION_KEYS],
            device=self.device,
            dtype=self.dtype,
        ).view(1, 8, 1, 1, 1)
        return bits.repeat_interleave(4, dim=1).expand(
            1,
            32,
            self.num_frame_per_block,
            self.height,
            self.width,
        )

    def _flow_to_x0(
        self,
        flow: torch.Tensor,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.scheduler.flow_to_x0(
            flow.flatten(0, 1),
            noisy.flatten(0, 1),
            timestep.flatten(),
        ).unflatten(0, flow.shape[:2])

    def _model_forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            noisy.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=self._prompt_context,
            seq_len=None,
            kv_cache=self._kv_cache,
            crossattn_cache=self._cross_attention_cache,
            current_start=self.current_start_frame * self._frame_sequence_length,
            act_context=action,
            act_context_scale=1.0,
            ref_latents=self._reference_latents,
            ref_mask=self._reference_mask,
        )
        return output.permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def step(self, action: Any = None) -> torch.Tensor:
        """Generate the next 9/12 RGB frames as a CPU ``[T,C,H,W]`` tensor."""

        if not self._configured or self._generator is None:
            raise RuntimeError("Call configure() before ABot-World step()")
        latent_shape = (
            1,
            self.num_frame_per_block,
            48,
            self.height // 16,
            self.width // 16,
        )
        action_tensor = self._action_tensor(action)
        noisy = torch.randn(
            latent_shape,
            generator=self._generator,
            device=self.device,
            dtype=self.dtype,
        )
        first_block = self.current_start_frame == 0
        if first_block:
            noisy[:, :1] = self._first_frame_latent
        for index, current_timestep in enumerate(self.denoising_steps):
            timestep = torch.full(
                (1, self.num_frame_per_block),
                float(current_timestep),
                device=self.device,
                dtype=torch.float32,
            )
            if first_block:
                timestep[:, 0] = 0
            flow = self._model_forward(noisy, timestep, action_tensor)
            denoised = self._flow_to_x0(flow, noisy, timestep)
            if index + 1 < len(self.denoising_steps):
                next_timestep = self.denoising_steps[index + 1]
                next_noise = torch.randn(
                    denoised.shape,
                    generator=self._generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                noisy = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    next_noise.flatten(0, 1),
                    torch.full(
                        (self.num_frame_per_block,),
                        float(next_timestep),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ).unflatten(0, denoised.shape[:2])
            else:
                noisy = denoised
            if first_block:
                noisy[:, :1] = self._first_frame_latent

        context_timestep = torch.full(
            (1, self.num_frame_per_block),
            float(self.context_noise),
            device=self.device,
            dtype=torch.float32,
        )
        if first_block:
            context_timestep[:, 0] = 0
            noisy[:, :1] = self._first_frame_latent
        self._model_forward(noisy, context_timestep, action_tensor)
        self.current_start_frame += self.num_frame_per_block

        with torch.autocast(device_type="cuda", dtype=self.decoder_compute_dtype):
            frames = self.decoder.decode(noisy)
        return frames[0].float().cpu()

    @staticmethod
    def _action_for_block(actions: Any, index: int) -> Any:
        if (
            isinstance(actions, Sequence)
            and not isinstance(actions, (str, bytes, bytearray))
            and actions
            and any(
                isinstance(item, (Mapping, Sequence, torch.Tensor))
                and not isinstance(item, (str, bytes, bytearray))
                for item in actions
            )
        ):
            return actions[min(index, len(actions) - 1)]
        return actions

    @torch.no_grad()
    def generate(
        self,
        image: Any,
        *,
        prompt: str = "",
        actions: Any = None,
        reference_images: Any = None,
        seed: int = 42,
        num_frames: int | None = 57,
        num_blocks: int | None = None,
    ) -> torch.Tensor:
        """Generate a complete CPU ``[T,C,H,W]`` video tensor in ``[0,1]``."""

        if num_frames is not None and int(num_frames) <= 0:
            raise ValueError("num_frames must be positive")
        if num_blocks is not None and int(num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if num_blocks is None:
            target_frames = 57 if num_frames is None else int(num_frames)
            num_blocks = max(1, math.ceil((target_frames + 3) / 12))
        self.configure(
            image,
            prompt=prompt,
            reference_images=reference_images,
            seed=seed,
        )
        blocks = [
            self.step(self._action_for_block(actions, index))
            for index in range(int(num_blocks))
        ]
        video = torch.cat(blocks, dim=0)
        return video if num_frames is None else video[: int(num_frames)]

    def reset(self) -> None:
        self._kv_cache.clear()
        self._cross_attention_cache.clear()
        self._prompt_context.clear()
        self._first_frame_latent = None
        self._reference_latents = None
        self._reference_mask = None
        self._generator = None
        self.current_start_frame = 0
        self.decoder.reset()
        self._configured = False
        self._empty_cuda_cache()


__all__ = ["ABotWorldInference", "ACTION_KEYS", "REFERENCE_SLOTS"]

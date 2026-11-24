"""CogVideoX runtime for WorldFoundry visual synthesis."""

from __future__ import annotations

from typing import Literal

_RESOLUTION_MAP = {
    "cogvideox-2b": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox1.5-5b-i2v": (768, 1360),
}


class CogVideoXOfficialRuntime:
    """Diffusers-backed CogVideoX runtime used by WorldFoundry synthesis wrappers."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        generation_type: Literal["t2v", "i2v", "v2v"],
        lora_path: str | None = None,
        lora_rank: int = 128,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        num_frames: int = 49,
        dtype: str | None = None,
        seed: int = 42,
        device: str = "cuda",
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str | None = None,
        max_sequence_length: int | None = None,
        scheduler: str = "dpm",
    ) -> None:
        self.model_name = model_name
        self.generation_type = generation_type
        self.num_videos_per_prompt = num_videos_per_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.num_frames = num_frames
        self.device = device
        self.negative_prompt = negative_prompt
        self.max_sequence_length = max_sequence_length
        self.use_dynamic_cfg = True
        self.height, self.width = self._resolve_resolution(model_name, model_path, height, width)

        import torch
        from diffusers import (
            CogVideoXDDIMScheduler,
            CogVideoXDPMScheduler,
            CogVideoXImageToVideoPipeline,
            CogVideoXPipeline,
            CogVideoXVideoToVideoPipeline,
        )

        pipeline_types = {
            "i2v": CogVideoXImageToVideoPipeline,
            "t2v": CogVideoXPipeline,
            "v2v": CogVideoXVideoToVideoPipeline,
        }
        if dtype is None:
            dtype = self._default_dtype(model_name, model_path)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.pipe = pipeline_types[self.generation_type].from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )

        if lora_path:
            self.pipe.load_lora_weights(
                lora_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="test_1",
            )
            self.pipe.fuse_lora(lora_scale=1 / lora_rank)

        scheduler = scheduler.lower()
        if scheduler == "dpm":
            self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            self.use_dynamic_cfg = True
        elif scheduler == "ddim":
            self.use_dynamic_cfg = False
            self.pipe.scheduler = CogVideoXDDIMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )
        else:
            raise ValueError("CogVideoX scheduler must be either 'dpm' or 'ddim'.")

        self.pipe.to(self.device)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    @staticmethod
    def _normalize_model_key(model_name: str, model_path: str) -> str:
        return f"{model_name} {model_path}".replace("_", "-").lower()

    @classmethod
    def _default_dtype(cls, model_name: str, model_path: str) -> str:
        model_key = cls._normalize_model_key(model_name, model_path)
        return "float16" if "2b" in model_key else "bfloat16"

    @classmethod
    def _resolve_resolution(
        cls,
        model_name: str,
        model_path: str,
        height: int | None,
        width: int | None,
    ) -> tuple[int, int]:
        model_key = cls._normalize_model_key(model_name, model_path)
        default = (480, 720)
        for key, resolution in _RESOLUTION_MAP.items():
            if key in model_key:
                default = resolution
                break
        resolved_height = default[0] if height is None else height
        resolved_width = default[1] if width is None else width
        return resolved_height, resolved_width

    def generate_video(
        self,
        prompt: str,
        image_path: str | None,
    ):
        import torch
        from diffusers.utils import load_image, load_video

        prompt_kwargs = {
            "prompt": prompt,
            "height": self.height,
            "width": self.width,
        }
        if self.generation_type == "i2v":
            prompt_kwargs["image"] = load_image(image=image_path)
        elif self.generation_type == "v2v":
            prompt_kwargs["video"] = load_video(video=image_path)

        call_kwargs = {
            **prompt_kwargs,
            "num_videos_per_prompt": self.num_videos_per_prompt,
            "num_inference_steps": self.num_inference_steps,
            "use_dynamic_cfg": self.use_dynamic_cfg,
            "guidance_scale": self.guidance_scale,
            "generator": torch.Generator().manual_seed(self.seed),
        }
        if self.generation_type != "v2v":
            call_kwargs["num_frames"] = self.num_frames
        if self.negative_prompt:
            call_kwargs["negative_prompt"] = self.negative_prompt
        if self.max_sequence_length is not None:
            call_kwargs["max_sequence_length"] = self.max_sequence_length

        generated_out = self.pipe(**call_kwargs)

        return generated_out.frames[0]


CogVideoX = CogVideoXOfficialRuntime


__all__ = ["CogVideoX", "CogVideoXOfficialRuntime"]

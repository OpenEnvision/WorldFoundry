from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

from .echo_infinity_runtime.pipeline import CausalInferencePipeline


def _dtype_from_name(name: str | None) -> torch.dtype:
    if name in {None, "bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported Echo-Infinity dtype: {name!r}")


class EchoInfinity:
    """In-tree Echo-Infinity text-to-video runtime."""

    def __init__(
        self,
        model_name: str,
        wan_root: str,
        generator_ckpt: str,
        wan_model_name: str = "Wan2.1-T2V-1.3B",
        lora_ckpt: Optional[str] = None,
        adapter: Optional[dict[str, Any]] = None,
        use_ema: bool = False,
        seed: int = 0,
        num_samples: int = 1,
        num_output_frames: int = 21,
        denoising_step_list: Optional[list[int]] = None,
        warp_denoising_step: bool = True,
        num_frame_per_block: int = 3,
        context_noise: int = 0,
        global_sink: bool = True,
        model_kwargs: Optional[dict[str, Any]] = None,
        memory_kwargs: Optional[dict[str, Any]] = None,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        low_memory: Optional[bool] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Echo-Infinity requires CUDA for generation, but CUDA is not available.")
        self.device = torch.device(device)
        self.dtype = _dtype_from_name(dtype)
        self.model_name = model_name
        self.seed = int(seed)
        self.num_samples = int(num_samples)
        self.num_output_frames = int(num_output_frames)
        self.low_memory = bool(low_memory) if low_memory is not None else False

        wan_root_path = Path(wan_root).expanduser().resolve()
        model_root = wan_root_path / wan_model_name
        generator_path = Path(generator_ckpt).expanduser().resolve()
        if not model_root.exists():
            raise FileNotFoundError(f"Echo-Infinity Wan model root not found: {model_root}")
        if not generator_path.is_file():
            raise FileNotFoundError(f"Echo-Infinity generator checkpoint not found: {generator_path}")

        model_kwargs = dict(model_kwargs or {})
        model_kwargs.setdefault("model_name", wan_model_name)
        model_kwargs.setdefault("wan_root", str(wan_root_path))

        args = SimpleNamespace(
            denoising_step_list=list(denoising_step_list or [1000, 750, 500, 250]),
            warp_denoising_step=bool(warp_denoising_step),
            model_kwargs=OmegaConf.create(model_kwargs),
            memory_kwargs=OmegaConf.create(memory_kwargs or {}),
            num_frame_per_block=int(num_frame_per_block),
            context_noise=int(context_noise),
            global_sink=bool(global_sink),
        )

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.set_grad_enabled(False)

        self.pipeline = CausalInferencePipeline(args, device=self.device)
        self._load_generator_checkpoint(str(generator_path), use_ema=bool(use_ema))
        if adapter:
            self._load_lora(adapter=adapter, lora_ckpt=lora_ckpt)

        self.pipeline = self.pipeline.to(dtype=self.dtype)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        self.pipeline.eval()

    def _load_generator_checkpoint(self, checkpoint_path: str, use_ema: bool) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            if use_ema and "generator_ema" in state_dict:
                raw_state = state_dict["generator_ema"]
                if "generator" in state_dict:
                    encoder_keys = {
                        key: value
                        for key, value in state_dict["generator"].items()
                        if "query_memory_encoder" in key
                    }
                    if encoder_keys:
                        raw_state = dict(raw_state)
                        raw_state.update(encoder_keys)
            else:
                raw_state = state_dict.get("generator", state_dict.get("generator_ema"))
        elif "model" in state_dict:
            raw_state = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {checkpoint_path}")

        cleaned = {key.replace("_fsdp_wrapped_module.", ""): value for key, value in raw_state.items()}
        missing, unexpected = self.pipeline.generator.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[Echo-Infinity] {len(missing)} missing parameters: {missing[:8]} ...")
        if unexpected:
            print(f"[Echo-Infinity] {len(unexpected)} unexpected parameters: {unexpected[:8]} ...")

    def _load_lora(self, adapter: dict[str, Any], lora_ckpt: Optional[str]) -> None:
        from .echo_infinity_runtime.utils.lora_utils import configure_lora_for_model
        import peft

        adapter_cfg = OmegaConf.create(adapter)
        self.pipeline.generator.model = configure_lora_for_model(
            self.pipeline.generator.model,
            model_name="generator",
            lora_config=adapter_cfg,
            is_main_process=True,
        )
        if not lora_ckpt:
            return
        lora_path = Path(lora_ckpt).expanduser().resolve()
        if not lora_path.is_file():
            raise FileNotFoundError(f"Echo-Infinity LoRA checkpoint not found: {lora_path}")
        checkpoint = torch.load(str(lora_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "generator_lora" in checkpoint:
            peft.set_peft_model_state_dict(self.pipeline.generator.model, checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(self.pipeline.generator.model, checkpoint)
        if isinstance(checkpoint, dict) and "query_memory_encoder" in checkpoint:
            inner = self.pipeline.generator.model
            if getattr(inner, "query_memory_encoder", None) is not None:
                inner.query_memory_encoder.load_state_dict(checkpoint["query_memory_encoder"], strict=False)

    @torch.no_grad()
    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        if image_path is not None:
            raise ValueError("Echo-Infinity is a text-to-video runtime; image conditioning is not supported.")
        prompts = [prompt] * self.num_samples
        noise_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        noise = torch.randn(
            [self.num_samples, self.num_output_frames, 16, 60, 104],
            device=self.device,
            dtype=self.dtype,
            generator=noise_generator,
        )
        video = self.pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            return_latents=False,
            low_memory=self.low_memory,
            profile=False,
        )
        if hasattr(self.pipeline.vae.model, "clear_cache"):
            self.pipeline.vae.model.clear_cache()
        return video[0].detach().cpu()


__all__ = ["EchoInfinity"]

"""Inference-only CogACT policy assembled from in-tree components."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from worldfoundry.core.action_normalization import (
    select_modality_statistics,
    unnormalize_action_values,
)
from worldfoundry.core.media.processing.image_utils import load_pil_image

from .action import ActionModel


class CogACT(nn.Module):
    """CogACT's Prismatic VLM plus diffusion action head.

    The class contains only inference behavior.  It deliberately computes the
    final cognition hidden state directly instead of calling text generation:
    the released policy consumes that state before the generated token, so
    materializing vocabulary logits and a KV cache is unnecessary.
    """

    EMPTY_TOKEN_ID = 29871
    COGNITION_TOKEN_ID = 2

    def __init__(
        self,
        *,
        vlm: nn.Module,
        processor: Any,
        action_model: ActionModel,
        norm_stats: Mapping[str, Any],
        device: str,
        vlm_dtype: torch.dtype,
        attention_implementation: str,
        checkpoint_path: str,
        load_seconds: float,
        compile_action_model: bool = False,
    ) -> None:
        super().__init__()
        self.vlm = vlm
        self.processor = processor
        self.action_model = action_model
        self.norm_stats = dict(norm_stats)
        self.inference_device = torch.device(device)
        self.vlm_dtype = vlm_dtype
        self.attention_implementation = attention_implementation
        self.checkpoint_path = checkpoint_path
        self.load_seconds = float(load_seconds)
        self.last_inference_metadata: dict[str, Any] = {}

        self._action_forward = self.action_model.net.forward
        self._action_forward_with_cfg = self.action_model.net.forward_with_cfg
        self.action_compiled = False
        if compile_action_model:
            from worldfoundry.runtime.compile_cache import CompilePolicy, compile_callable_cached

            policy = CompilePolicy(mode="reduce-overhead", fullgraph=False, dynamic=False)
            self._action_forward = compile_callable_cached(
                self._action_forward,
                policy=policy,
                namespace="cogact-action",
            )
            self._action_forward_with_cfg = compile_callable_cached(
                self._action_forward_with_cfg,
                policy=policy,
                namespace="cogact-action-cfg",
            )
            self.action_compiled = True

    def _synchronize(self) -> None:
        if self.inference_device.type == "cuda":
            torch.cuda.synchronize(self.inference_device)

    def _seed_context(self, seed: int | None) -> contextlib.AbstractContextManager[Any]:
        if seed is None:
            return contextlib.nullcontext()
        devices: list[int] = []
        if self.inference_device.type == "cuda":
            index = self.inference_device.index
            devices = [torch.cuda.current_device() if index is None else index]
        return torch.random.fork_rng(devices=devices)

    def _cognition_features(self, image: Any, instruction: str) -> tuple[torch.Tensor, str, float]:
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        preprocess_started = time.perf_counter()
        inputs = self.processor(
            prompt,
            load_pil_image(image),
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.inference_device)
        suffix = torch.tensor(
            [[self.EMPTY_TOKEN_ID, self.COGNITION_TOKEN_ID]],
            dtype=input_ids.dtype,
            device=self.inference_device,
        )
        input_ids = torch.cat((input_ids, suffix), dim=1)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.inference_device)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 2),
                        dtype=attention_mask.dtype,
                        device=self.inference_device,
                    ),
                ),
                dim=1,
            )
        pixel_values = inputs["pixel_values"].to(
            device=self.inference_device,
            dtype=self.vlm_dtype,
        )
        preprocess_seconds = time.perf_counter() - preprocess_started

        input_embeddings = self.vlm.get_input_embeddings()(input_ids)
        projected_patches = self.vlm.projector(self.vlm.vision_backbone(pixel_values))
        multimodal_embeddings, multimodal_attention_mask = self.vlm._build_multimodal_attention(
            input_embeddings,
            projected_patches,
            attention_mask,
        )
        output = self.vlm.language_model.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        cognition = output.last_hidden_state[:, -1, :]
        if tuple(cognition.shape) != (1, 4096):
            raise RuntimeError(f"CogACT cognition feature has unexpected shape {tuple(cognition.shape)}")
        return cognition, prompt, preprocess_seconds

    def _sample_actions(
        self,
        cognition: torch.Tensor,
        *,
        cfg_scale: float,
        use_ddim: bool,
        num_ddim_steps: int,
    ) -> torch.Tensor:
        action_dtype = next(self.action_model.net.parameters()).dtype
        cognition = cognition.unsqueeze(1).to(dtype=action_dtype)
        batch_size = cognition.shape[0]
        noise = torch.randn(
            batch_size,
            self.action_model.future_action_window_size + 1,
            self.action_model.in_channels,
            device=self.inference_device,
            dtype=action_dtype,
        )

        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat((noise, noise), dim=0)
            unconditional = self.action_model.net.z_embedder.uncondition.unsqueeze(0).expand(batch_size, 1, -1)
            model_kwargs = {"z": torch.cat((cognition, unconditional), dim=0), "cfg_scale": cfg_scale}
            sample_fn = self._action_forward_with_cfg
        else:
            model_kwargs = {"z": cognition}
            sample_fn = self._action_forward

        if use_ddim:
            if num_ddim_steps <= 0:
                raise ValueError("CogACT num_ddim_steps must be positive")
            if (
                self.action_model.ddim_diffusion is None
                or self.action_model.ddim_diffusion.num_timesteps != num_ddim_steps
            ):
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=self.inference_device,
                eta=0.0,
            )
        else:
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=self.inference_device,
            )
        if using_cfg:
            samples = samples.chunk(2, dim=0)[0]
        return samples[0]

    @torch.inference_mode()
    def predict_action(
        self,
        image: Any,
        instruction: str,
        *,
        unnorm_key: str | None = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not instruction.strip():
            raise ValueError("CogACT requires a non-empty instruction")
        total_started = time.perf_counter()
        if self.inference_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.inference_device)

        with self._seed_context(seed):
            if seed is not None:
                torch.manual_seed(int(seed))
                if self.inference_device.type == "cuda":
                    torch.cuda.manual_seed(int(seed))

            self._synchronize()
            vlm_started = time.perf_counter()
            with torch.autocast(
                device_type=self.inference_device.type,
                dtype=self.vlm_dtype,
                enabled=self.inference_device.type == "cuda" and self.vlm_dtype != torch.float32,
            ):
                cognition, prompt, preprocess_seconds = self._cognition_features(image, instruction)
            self._synchronize()
            vlm_seconds = time.perf_counter() - vlm_started - preprocess_seconds

            diffusion_started = time.perf_counter()
            samples = self._sample_actions(
                cognition,
                cfg_scale=float(cfg_scale),
                use_ddim=bool(use_ddim),
                num_ddim_steps=int(num_ddim_steps),
            )
            self._synchronize()
            diffusion_seconds = time.perf_counter() - diffusion_started

        normalized_actions = np.clip(samples.float().cpu().numpy(), -1.0, 1.0)
        if normalized_actions.shape[-1] != 7:
            raise RuntimeError(f"CogACT action head returned shape {normalized_actions.shape}; expected (*, 7)")
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0.0, 1.0)
        selected_key, action_stats = select_modality_statistics(
            self.norm_stats,
            modality="action",
            key=unnorm_key,
        )
        actions = unnormalize_action_values(normalized_actions, action_stats, mode="q99")

        metadata: dict[str, Any] = {
            "checkpoint_path": self.checkpoint_path,
            "load_seconds": round(self.load_seconds, 3),
            "preprocess_seconds": round(preprocess_seconds, 6),
            "vlm_seconds": round(max(vlm_seconds, 0.0), 6),
            "diffusion_seconds": round(diffusion_seconds, 6),
            "inference_seconds": round(time.perf_counter() - total_started, 6),
            "attention_implementation": self.attention_implementation,
            "vlm_dtype": str(self.vlm_dtype).removeprefix("torch."),
            "action_dtype": str(next(self.action_model.net.parameters()).dtype).removeprefix("torch."),
            "action_compiled": self.action_compiled,
            "unnorm_key": selected_key,
            "prompt": prompt,
            "cfg_scale": float(cfg_scale),
            "use_ddim": bool(use_ddim),
            "num_ddim_steps": int(num_ddim_steps),
            "seed": seed,
        }
        if self.inference_device.type == "cuda":
            metadata["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(self.inference_device))
            metadata["allocated_memory_bytes"] = int(torch.cuda.memory_allocated(self.inference_device))
        self.last_inference_metadata = metadata
        return actions, normalized_actions

    def get_action_stats(self, unnorm_key: str | None = None) -> Mapping[str, Any]:
        return select_modality_statistics(self.norm_stats, modality="action", key=unnorm_key)[1]

    def get_action_dim(self, unnorm_key: str | None = None) -> int:
        return len(self.get_action_stats(unnorm_key)["q01"])


__all__ = ["CogACT"]

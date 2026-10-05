# Adapted from seedleap/zing-world-model (Apache-2.0).
from __future__ import annotations

from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.loaders.wan_interactive import WanInteractiveTextEncoder as WanTextEncoder
from worldfoundry.base_models.diffusion_model.loaders.wan_interactive import WanInteractiveVAE as WanVAE
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.zing import WanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.zing.config import ZingConfig

from .processor import InferenceRequest
from .scheduler import DmdScheduler


class InferencePipeline:
    def __init__(self, config: ZingConfig, pretrained_dir: str | Path, checkpoint: str | Path, device: str = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.config = config
        self.device = torch.device(device)
        pretrained_path = Path(pretrained_dir)
        for name in ("text_encoder", "tokenizer", "vae"):
            if not (pretrained_path / name).is_dir():
                raise ValueError(f"pretrained directory is missing {name}/")
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.suffix != ".pt" or not checkpoint_path.is_file():
            raise ValueError("checkpoint must be an existing .pt file")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
            raise ValueError("checkpoint must contain a bare state dict")
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("checkpoint state dict values must all be tensors")
        with torch.device("meta"):
            generator = WanModel(config)
        generator.load_state_dict(state, strict=True, assign=True)
        del state
        self.generator = generator.eval().requires_grad_(False).to(device=self.device, dtype=torch.bfloat16)
        self.text_encoder = WanTextEncoder(pretrained_path, config.text_encoder.max_length)
        self.text_encoder.eval().requires_grad_(False).to(device="cpu", dtype=torch.bfloat16)
        self.vae = WanVAE(pretrained_path)
        self.vae.eval().requires_grad_(False).to(device="cpu", dtype=torch.bfloat16)

    def encode_reference(self, frames: torch.Tensor) -> torch.Tensor:
        self.vae.to(self.device)
        try:
            return self.vae.encode(frames).cpu()
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

    @staticmethod
    def _known(latents: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        known = (~mask).view(mask.shape[0], mask.shape[1], 1, 1, 1)
        return torch.where(known, clean, latents)

    def _model_flow(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: tuple[torch.Tensor, torch.Tensor],
        cache,
        cache_mode: str | None,
        action: torch.Tensor | None,
        prompt_switch: bool,
    ) -> torch.Tensor:
        value = latents.permute(0, 2, 1, 3, 4)
        output = self.generator(
            value,
            timestep,
            context[0],
            context[1],
            cache,
            cache_mode,
            action=action,
            prompt_switch=prompt_switch,
        )
        return output.permute(0, 2, 1, 3, 4)

    @torch.inference_mode()
    def generate(self, request: InferenceRequest) -> torch.Tensor:
        self.text_encoder.to(self.device)
        contexts = [self.text_encoder.encode([prompt]) for prompt in request.prompts]
        self.text_encoder.to("cpu")
        clean = request.clean_latents.to(device=self.device, dtype=torch.bfloat16)
        mask = request.label_mask.to(device=self.device, dtype=torch.bool)
        action = None if request.action is None else request.action.to(self.device)
        latents = self._known(torch.randn_like(clean), clean, mask)
        output = clean.clone()
        lengths = [int(value) for value in request.prompt_lengths.tolist()]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cache = self.generator.make_kv_cache()
        segment = 0
        last_end = request.chunk_spans[-1][1]
        for start, end in request.chunk_spans:
            previous_segment = segment
            while segment + 1 < len(contexts) and start >= boundaries[segment + 1]:
                segment += 1
            prompt_switch = segment != previous_segment
            context = (
                contexts[segment][0].to(device=self.device, dtype=torch.bfloat16),
                contexts[segment][1].to(device=self.device),
            )
            current = latents[:, start:end]
            current_clean = clean[:, start:end]
            current_mask = mask[:, start:end]
            current_action = None if action is None else action[:, start:end]
            keep_cache = end < last_end
            if keep_cache:
                patch = self.config.generator.patch_size
                token_count = (
                    (end - start) // patch[0] * (current.shape[3] // patch[1]) * (current.shape[4] // patch[2])
                )
                cache.reserve(token_count)
            if not bool(current_mask.any()):
                if keep_cache:
                    zero = torch.zeros((current.shape[0], end - start), device=self.device, dtype=torch.float32)
                    self._model_flow(current_clean, zero, context, cache, "final", current_action, prompt_switch)
                output[:, start:end] = current_clean
                continue
            scheduler = DmdScheduler(self.config.inference).to(self.device)
            cache_mode = "active" if keep_cache else None
            for step_index, timestep in enumerate(scheduler.timesteps):
                step_switch = prompt_switch and step_index == 0
                step_time = timestep * torch.ones(
                    (current.shape[0], end - start), device=self.device, dtype=torch.float32
                )
                step_time = torch.where(current_mask, step_time, 0)
                current = self._known(current, current_clean, current_mask)
                prediction = self._model_flow(
                    current, step_time, context, cache, cache_mode, current_action, step_switch
                )
                current, _ = scheduler.step(prediction, current)
                current = self._known(current, current_clean, current_mask)
            output[:, start:end] = current
            if keep_cache:
                zero = torch.zeros((current.shape[0], end - start), device=self.device, dtype=torch.float32)
                self._model_flow(current, zero, context, cache, "final", current_action, False)
        del contexts, cache, latents, clean, mask, action
        torch.cuda.empty_cache()
        self.vae.to(self.device)
        try:
            video = self.vae.decode(output)
            return (video * 0.5 + 0.5).clamp(0, 1).cpu()
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

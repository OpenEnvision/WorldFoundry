"""LTX-2 latent refiner component for SANA-WM streaming."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest, ModalityState
from ...loaders import NativeCheckpointResolver
from ...optimizations import OffloadMode


@contextmanager
def _masked_ltx2_blocks(transformer: torch.nn.Module, mask: torch.Tensor):
    """Pass the frozen-prefix mask through Diffusers releases without a top-level mask argument."""

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or not len(blocks):
        raise TypeError("LTX-2 refiner has no transformer blocks to receive the prefix mask")
    handles = []

    def supply_mask(module, args, kwargs):
        kwargs["self_attention_mask"] = mask
        return args, kwargs

    try:
        for block in blocks:
            if "self_attention_mask" not in inspect.signature(block.forward).parameters:
                raise TypeError("LTX-2 refiner block cannot receive the prefix mask")
            handles.append(block.register_forward_pre_hook(supply_mask, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack BCTHW latents into LTX-2's frame-major token sequence."""

    return latents.permute(0, 2, 3, 4, 1).flatten(1, 3)


def _unpack_latents(
    tokens: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    batch = tokens.shape[0]
    return tokens.reshape(batch, frames, height, width, -1).permute(0, 4, 1, 2, 3)


class SanaWMLTX2RefinerProcessor:
    """Refine active Stage-1 frames against a frozen rolling context prefix."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        sigmas: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0),
        offload_after_refine: bool = False,
    ) -> None:
        self.transformer = transformer
        self.device = torch.device(device)
        self.dtype = dtype
        self.sigmas = tuple(float(value) for value in sigmas)
        self.offload_after_refine = bool(offload_after_refine)
        if len(self.sigmas) < 2 or self.sigmas[-1] != 0.0:
            raise ValueError("SANA-WM refiner sigmas must contain steps and end at zero")
        if any(left <= right for left, right in zip(self.sigmas, self.sigmas[1:])):
            raise ValueError("SANA-WM refiner sigmas must be strictly descending")
        config = getattr(transformer, "config", None)
        if int(getattr(config, "patch_size", 1)) != 1 or int(
            getattr(config, "patch_size_t", 1)
        ) != 1:
            raise ValueError("SANA-WM streaming refiner requires unit spatial/temporal patches")

    def process(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, torch.Tensor]:
        """Expose the generic LatentProcessor contract for assembler validation."""

        del request
        return {name: state.latent for name, state in states.items()}

    @staticmethod
    def _self_attention_mask(
        *,
        batch_size: int,
        total_tokens: int,
        context_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(
            batch_size,
            total_tokens,
            total_tokens,
            device=device,
            dtype=torch.bool,
        )
        mask[:, :context_tokens, context_tokens:] = False
        return mask

    def _velocity(
        self,
        *,
        context: torch.Tensor,
        active: torch.Tensor,
        sigma: float,
        conditioning: Mapping[str, object],
        fps: float,
    ) -> torch.Tensor:
        context_tokens = _pack_latents(context)
        active_tokens = _pack_latents(active)
        video_tokens = torch.cat((context_tokens, active_tokens), dim=1)
        batch = video_tokens.shape[0]
        raw_timestep = torch.zeros(
            batch,
            video_tokens.shape[1],
            device=self.device,
            dtype=torch.float32,
        )
        raw_timestep[:, context_tokens.shape[1] :] = float(sigma)
        multiplier = float(
            getattr(self.transformer.config, "timestep_scale_multiplier", 1000.0)
        )
        model_timestep = raw_timestep * multiplier

        video_context = conditioning.get("refiner_video_context")
        audio_context = conditioning.get("refiner_audio_context")
        context_mask = conditioning.get("refiner_context_mask")
        if not isinstance(video_context, torch.Tensor):
            raise TypeError("SANA-WM refiner requires refiner_video_context")
        if not isinstance(audio_context, torch.Tensor):
            raise TypeError("SANA-WM refiner requires refiner_audio_context")
        if not isinstance(context_mask, torch.Tensor):
            raise TypeError("SANA-WM refiner requires refiner_context_mask")

        audio_channels = int(getattr(self.transformer.config, "audio_in_channels", 128))
        audio_tokens = torch.zeros(
            batch,
            1,
            audio_channels,
            device=self.device,
            dtype=self.dtype,
        )
        audio_timestep = torch.zeros(batch, 1, device=self.device, dtype=torch.float32)
        mask = self._self_attention_mask(
            batch_size=batch,
            total_tokens=video_tokens.shape[1],
            context_tokens=context_tokens.shape[1],
            device=self.device,
        )
        top_level_mask = "video_self_attention_mask" in inspect.signature(self.transformer.forward).parameters
        mask_scope = nullcontext() if top_level_mask else _masked_ltx2_blocks(self.transformer, mask)
        with mask_scope:
            output = self.transformer(
                hidden_states=video_tokens,
                audio_hidden_states=audio_tokens,
                encoder_hidden_states=video_context.to(device=self.device, dtype=self.dtype),
                audio_encoder_hidden_states=audio_context.to(device=self.device, dtype=self.dtype),
                timestep=model_timestep,
                audio_timestep=audio_timestep,
                encoder_attention_mask=context_mask.to(device=self.device),
                audio_encoder_attention_mask=context_mask.to(device=self.device),
                num_frames=int(context.shape[2] + active.shape[2]),
                height=int(active.shape[3]),
                width=int(active.shape[4]),
                fps=float(fps),
                audio_num_frames=1,
                isolate_modalities=True,
                return_dict=False,
                **({"video_self_attention_mask": mask} if top_level_mask else {}),
            )
        if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("LTX-2 refiner transformer must return a video tensor tuple")
        return output[0][:, context_tokens.shape[1] :].to(dtype=self.dtype)

    @torch.inference_mode()
    def refine_active_latents(
        self,
        *,
        context_latents: torch.Tensor,
        active_latents: torch.Tensor,
        conditioning: Mapping[str, object],
        fps: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if context_latents.ndim != 5 or active_latents.ndim != 5:
            raise ValueError("SANA-WM refiner latents must be BCTHW tensors")
        if context_latents.shape[:2] != active_latents.shape[:2] or (
            context_latents.shape[-2:] != active_latents.shape[-2:]
        ):
            raise ValueError("SANA-WM refiner context and active geometry must match")
        if context_latents.shape[2] <= 0 or active_latents.shape[2] <= 0:
            raise ValueError("SANA-WM refiner requires non-empty context and active frames")
        try:
            self.transformer.to(device=self.device, dtype=self.dtype)
            context = context_latents.to(device=self.device, dtype=self.dtype).contiguous()
            active = active_latents.to(device=self.device, dtype=self.dtype).contiguous()
            noise = torch.randn(
                active.shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
            noisy = (1.0 - self.sigmas[0]) * active + self.sigmas[0] * noise
            for sigma, following in zip(self.sigmas, self.sigmas[1:]):
                velocity = self._velocity(
                    context=context,
                    active=noisy,
                    sigma=sigma,
                    conditioning=conditioning,
                    fps=fps,
                )
                noisy_tokens = _pack_latents(noisy).float()
                next_tokens = noisy_tokens + (
                    float(following) - float(sigma)
                ) * velocity.float()
                noisy = _unpack_latents(
                    next_tokens.to(dtype=self.dtype),
                    frames=active.shape[2],
                    height=active.shape[3],
                    width=active.shape[4],
                )
            return noisy.to(device=active_latents.device, dtype=active_latents.dtype)
        finally:
            if self.offload_after_refine:
                self.transformer.to(device="cpu")
                if self.device.type == "cuda" and torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()


def build_sana_wm_ltx2_refiner_processor(
    context: ComponentBuildContext,
) -> SanaWMLTX2RefinerProcessor:
    """Load the pinned Diffusers LTX-2 transformer component locally."""

    from diffusers import LTX2VideoTransformer3DModel

    checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("weights"))
    transformer_root = checkpoint.directory("refiner_diffusers/transformer")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        str(transformer_root),
        torch_dtype=context.policy.dtype,
        local_files_only=True,
    ).eval()
    staged_offload = bool(
        context.policy.device.type != "cpu"
        and context.component_options.get(
            "offload",
            context.policy.offload.mode is not OffloadMode.NONE,
        )
    )
    if not staged_offload:
        transformer.to(device=context.policy.device, dtype=context.policy.dtype)
    return SanaWMLTX2RefinerProcessor(
        transformer,
        device=context.policy.device,
        dtype=context.policy.dtype,
        sigmas=tuple(
            float(value)
            for value in context.component_options.get(
                "sigmas",
                (0.909375, 0.725, 0.421875, 0.0),
            )
        ),
        offload_after_refine=staged_offload,
    )


__all__ = [
    "SanaWMLTX2RefinerProcessor",
    "build_sana_wm_ltx2_refiner_processor",
]

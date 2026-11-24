"""Autoregressive causal rollout for joint video and audio generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldfoundry.base_models.diffusion_model.models.representations.ltx.echo_wm import EchoWMModality as Modality

from .cache import configure_bounded_caches
from .causal_wrapper import CausalModelWrapper
from .scheduling import (
    DEFAULT_CAUSAL_TIMESTEPS,
    causal_audio_blocks,
    causal_audio_frames,
    causal_video_blocks,
    resolve_causal_sigmas,
)

Block = tuple[int, int]


def _modality(
    latent: torch.Tensor,
    positions: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    *,
    context_mask: torch.Tensor | None,
) -> Modality:
    return Modality(
        latent=latent,
        sigma=torch.ones(latent.shape[0], device=latent.device, dtype=latent.dtype),
        timesteps=torch.full(latent.shape[:2], sigma, device=latent.device, dtype=latent.dtype),
        positions=positions,
        context=context,
        context_mask=context_mask,
    )


def _random_like(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _advance_sample(
    denoised: torch.Tensor,
    next_sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return (1 - next_sigma) * denoised + next_sigma * _random_like(denoised, generator)


@dataclass(frozen=True)
class _BlockForward:
    """Bind rollout-wide context and expose a compact per-block model call."""

    wrapper: CausalModelWrapper
    caches: list[dict]
    video_positions: torch.Tensor
    audio_positions: torch.Tensor
    video_context: torch.Tensor
    audio_context: torch.Tensor
    context_mask: torch.Tensor | None
    action_cond: dict[str, torch.Tensor]

    @classmethod
    def create(
        cls,
        *,
        wrapper: CausalModelWrapper,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        video_positions: torch.Tensor,
        audio_positions: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        context_mask: torch.Tensor | None,
        action_cond: dict[str, torch.Tensor],
    ) -> _BlockForward:
        caches = wrapper.init_caches(
            batch_size=clean_video.shape[0],
            video_frames=video_positions.shape[2] // wrapper.patches_per_frame,
            audio_frames=clean_audio.shape[1],
            text_seq_len=video_context.shape[1],
            device=clean_video.device,
            dtype=clean_video.dtype,
        )
        configure_bounded_caches(
            wrapper,
            caches,
            video_positions,
            audio_positions,
            action_cond,
            clean_video.dtype,
        )
        return cls(
            wrapper=wrapper,
            caches=caches,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_context=video_context,
            audio_context=audio_context,
            context_mask=context_mask,
            action_cond=action_cond,
        )

    def __call__(
        self,
        video_latent: torch.Tensor,
        video_block: Block,
        video_sigma: float,
        audio_latent: torch.Tensor | None = None,
        audio_block: Block = (0, 0),
        audio_sigma: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video_start, video_end = video_block
        audio_start, audio_end = audio_block
        patches_per_frame = self.wrapper.patches_per_frame
        sliced_action = {
            key: value[:, video_start:video_end] if key in {"ucpe_viewmats", "ucpe_Ks"} else value
            for key, value in self.action_cond.items()
        }
        video = _modality(
            video_latent,
            self.video_positions[:, :, video_start * patches_per_frame : video_end * patches_per_frame],
            self.video_context,
            video_sigma,
            context_mask=self.context_mask,
        )
        audio = None
        if audio_latent is not None:
            audio = _modality(
                audio_latent,
                self.audio_positions[:, :, audio_start:audio_end],
                self.audio_context,
                video_sigma if audio_sigma is None else audio_sigma,
                context_mask=self.context_mask,
            )
        return self.wrapper(
            video,
            audio,
            sliced_action,
            self.caches,
            video_start,
            audio_start,
        )


@dataclass
class _RolloutBuffers:
    """Noise sources and output tensors shared by all rollout blocks."""

    initial_video: torch.Tensor
    initial_audio: torch.Tensor
    video_output: torch.Tensor
    audio_output: torch.Tensor
    clean_image: torch.Tensor

    @classmethod
    def create(
        cls,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        patches_per_frame: int,
        generator: torch.Generator,
    ) -> _RolloutBuffers:
        clean_image = clean_video[:, :patches_per_frame]
        video_output = torch.zeros_like(clean_video)
        video_output[:, :patches_per_frame] = clean_image
        return cls(
            initial_video=_random_like(clean_video, generator),
            initial_audio=_random_like(clean_audio, generator),
            video_output=video_output,
            audio_output=torch.zeros_like(clean_audio),
            clean_image=clean_image,
        )


def _denoise_audio_prefix(
    forward: _BlockForward,
    clean_image: torch.Tensor,
    initial_audio: torch.Tensor,
    video_block: Block,
    audio_block: Block,
    sigmas: list[float],
    generator: torch.Generator,
) -> torch.Tensor:
    audio_start, audio_end = audio_block
    audio_sample = initial_audio[:, audio_start:audio_end]
    for step, sigma in enumerate(sigmas):
        _, denoised_audio = forward(
            clean_image,
            video_block,
            0.0,
            audio_sample,
            audio_block,
            sigma,
        )
        if denoised_audio is None:
            raise RuntimeError("causal AV model returned no audio")
        audio_sample = (
            denoised_audio if step == len(sigmas) - 1 else _advance_sample(denoised_audio, sigmas[step + 1], generator)
        )
    return audio_sample


def _denoise_av_block(
    forward: _BlockForward,
    initial_video: torch.Tensor,
    initial_audio: torch.Tensor,
    video_block: Block,
    audio_block: Block,
    sigmas: list[float],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_start, video_end = video_block
    audio_start, audio_end = audio_block
    patches_per_frame = forward.wrapper.patches_per_frame
    video_sample = initial_video[:, video_start * patches_per_frame : video_end * patches_per_frame]
    audio_sample = initial_audio[:, audio_start:audio_end]

    for step, sigma in enumerate(sigmas):
        denoised_video, denoised_audio = forward(
            video_sample,
            video_block,
            sigma,
            audio_sample,
            audio_block,
            sigma,
        )
        if denoised_audio is None:
            raise RuntimeError("causal AV model returned no audio")
        if step == len(sigmas) - 1:
            video_sample, audio_sample = denoised_video, denoised_audio
        else:
            next_sigma = sigmas[step + 1]
            video_sample = _advance_sample(denoised_video, next_sigma, generator)
            audio_sample = _advance_sample(denoised_audio, next_sigma, generator)
    return video_sample, audio_sample


def _generate_audio_prefix(
    forward: _BlockForward,
    buffers: _RolloutBuffers,
    video_block: Block,
    audio_block: Block,
    sigmas: list[float],
    generator: torch.Generator,
) -> None:
    """Commit the image sink, generate its audio prefix, then refresh caches."""
    forward(buffers.clean_image, video_block, 0.0)
    audio_prefix = _denoise_audio_prefix(
        forward,
        buffers.clean_image,
        buffers.initial_audio,
        video_block,
        audio_block,
        sigmas,
        generator,
    )
    audio_start, audio_end = audio_block
    buffers.audio_output[:, audio_start:audio_end] = audio_prefix
    forward(buffers.clean_image, video_block, 0.0, audio_prefix, audio_block, 0.0)


def _generate_av_blocks(
    forward: _BlockForward,
    buffers: _RolloutBuffers,
    video_blocks: list[Block],
    audio_blocks: list[Block],
    sigmas: list[float],
    generator: torch.Generator,
) -> None:
    """Generate, store, and cache all blocks after the image sink."""
    patches_per_frame = forward.wrapper.patches_per_frame
    for video_block, audio_block in zip(video_blocks[1:], audio_blocks[1:], strict=True):
        video_sample, audio_sample = _denoise_av_block(
            forward,
            buffers.initial_video,
            buffers.initial_audio,
            video_block,
            audio_block,
            sigmas,
            generator,
        )
        video_start, video_end = video_block
        audio_start, audio_end = audio_block
        buffers.video_output[:, video_start * patches_per_frame : video_end * patches_per_frame] = video_sample
        buffers.audio_output[:, audio_start:audio_end] = audio_sample
        forward(video_sample, video_block, 0.0, audio_sample, audio_block, 0.0)


@torch.no_grad()
def causal_rollout(
    *,
    wrapper: CausalModelWrapper,
    clean_video: torch.Tensor,
    clean_audio: torch.Tensor,
    video_positions: torch.Tensor,
    audio_positions: torch.Tensor,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    context_mask: torch.Tensor | None,
    action_cond: dict[str, torch.Tensor],
    seed: int,
    timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all audio-video blocks and refresh caches with clean outputs."""
    patches_per_frame = wrapper.patches_per_frame
    video_frames = video_positions.shape[2] // patches_per_frame
    chunk_size = wrapper.cache.video_chunk_size
    video_blocks = causal_video_blocks(video_frames, chunk_size)
    audio_blocks = causal_audio_blocks(video_frames, chunk_size)
    if clean_audio.shape[1] != causal_audio_frames(video_frames, chunk_size):
        raise ValueError("audio latent length does not match the causal AV block layout")

    sigmas = resolve_causal_sigmas(timesteps)
    generator = torch.Generator(device=clean_video.device).manual_seed(seed)
    buffers = _RolloutBuffers.create(
        clean_video,
        clean_audio,
        patches_per_frame,
        generator,
    )
    forward = _BlockForward.create(
        wrapper=wrapper,
        clean_video=clean_video,
        clean_audio=clean_audio,
        video_positions=video_positions,
        audio_positions=audio_positions,
        video_context=video_context,
        audio_context=audio_context,
        context_mask=context_mask,
        action_cond=action_cond,
    )

    _generate_audio_prefix(
        forward,
        buffers,
        video_blocks[0],
        audio_blocks[0],
        sigmas,
        generator,
    )
    _generate_av_blocks(forward, buffers, video_blocks, audio_blocks, sigmas, generator)
    return buffers.video_output, buffers.audio_output

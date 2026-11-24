"""Framework-owned windowed autoregressive diffusion execution.

Implements the ``autoregressive-window`` strategy runner at the bottom of:

    recipe.execution.strategy == "autoregressive-window"
        → build_autoregressive_window_strategy
        → AutoregressiveWindowRunner.run

``run`` steps one :class:`AutoregressiveStream` to completion and decodes the
assembled latents once. Realtime callers use ``open_stream`` directly and
decode each committed block themselves.

Inputs: WindowedDenoiser plus canonical RunnerComponents, prediction_mode
(flow / distilled-x0), fixed_timesteps, optional first_block_timesteps, and
context_timestep.
Output: DiffusionOutput, or one BCTHW latent block per ``AutoregressiveStream.step``;
metadata includes block_size / n_views / prediction_mode.
Must not: let a model package own KV-cache windows or bypass
WindowedDenoiser.create_kv_cache; on the distilled-x0 path, do not shift
context_timestep a second time (see the cache-commit comment).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch
from einops import rearrange
from torch import Tensor

from ..contracts import (
    Conditioning,
    Denoiser,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    EncodedLatentInitializer,
    LatentInitialization,
    SchedulerStep,
)
from ..extensions import DiffusionRunContext
from ..schedulers.wan import add_flow_noise, shift_flow_sigmas
from .base import NativeDiffusionRunner


@runtime_checkable
class WindowedDenoiser(Denoiser, Protocol):
    """Extra architecture surface required by the window runner: block_size, variant, KV-cache creation.

    Attention implementations stay on the model component; the framework owns window
    slicing and cache lifetime. AutoregressiveWindowRunner construction raises
    :exc:`TypeError` if this protocol is not satisfied.
    """

    block_size: int
    variant: str

    def create_kv_cache(
        self,
        *,
        batch_size: int,
        n_views: int,
        latent_frames_per_view: int,
        frame_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any: ...


class AutoregressiveWindowRunner(NativeDiffusionRunner):
    """Own block traversal, CFG, solver updates, and KV-cache lifetime.

    ``prediction_mode`` may be only ``flow`` or ``distilled-x0``; the latter requires fixed_timesteps.
    ``renoise_dtype`` (distilled-x0) draws the re-noise in that dtype and mixes it with the
    clean block as the official FlowMatchScheduler.add_noise does: a block with batch x
    frames == 1 computes in ``renoise_dtype`` (bf16 for frame-wise bf16 students), a larger
    block computes in float32; see :meth:`_renoise_in_dtype`.
    ``first_block_timesteps`` (distilled-x0 only) replaces fixed_timesteps for the first
    noisy block of a rollout; later blocks keep fixed_timesteps, whose length the request's
    ``num_inference_steps`` must match.
    Construction: non-WindowedDenoiser → :exc:`TypeError`; illegal mode, missing
    fixed_timesteps, or first_block_timesteps outside distilled-x0 → :exc:`ValueError`.
    """

    def __init__(
        self,
        *,
        prediction_mode: str = "flow",
        fixed_timesteps: Sequence[int] = (),
        first_block_timesteps: Sequence[int] = (),
        context_timestep: int = 0,
        renoise_dtype: torch.dtype | None = None,
        guidance_mode: str = "standard",
        **kwargs: object,
    ) -> None:
        super().__init__(guidance_mode=guidance_mode, **kwargs)
        if not isinstance(self.components.denoiser, WindowedDenoiser):
            raise TypeError("autoregressive-window requires a WindowedDenoiser component")
        self.windowed_denoiser = self.components.denoiser
        self.prediction_mode = str(prediction_mode).strip().lower().replace("_", "-")
        if self.prediction_mode not in {"flow", "distilled-x0"}:
            raise ValueError(f"unsupported window prediction mode: {prediction_mode!r}")
        self.fixed_timesteps = tuple(int(value) for value in fixed_timesteps)
        self.first_block_timesteps = tuple(int(value) for value in first_block_timesteps)
        self.context_timestep = int(context_timestep)
        self.renoise_dtype = renoise_dtype
        if self.prediction_mode == "distilled-x0" and not self.fixed_timesteps:
            raise ValueError("distilled-x0 execution requires fixed_timesteps")
        if self.first_block_timesteps and self.prediction_mode != "distilled-x0":
            raise ValueError("first_block_timesteps requires distilled-x0 execution")
        self._active_stream: AutoregressiveStream | None = None

    def _prepare_context(
        self,
        request: DiffusionRequest,
    ) -> tuple[DiffusionRunContext, Tensor, Mapping[str, object]]:
        """Encode conditions, run extensions, initialize latents. Same semantics as the standard runner, without entering the sampling loop."""
        generator = self._generator(request.sampling.seed)
        conditioning = self.components.conditioner.encode(
            request,
            device=self.device,
            dtype=self.dtype,
        )
        if not isinstance(conditioning, Conditioning):
            raise TypeError("conditioner.encode must return Conditioning")
        context = DiffusionRunContext(
            request=request,
            components=self.components,
            conditioning=conditioning,
            generator=generator,
        )
        for extension in self.extensions:
            extension.on_run_start(context)
        for extension in self.extensions:
            context.conditioning = extension.prepare_conditioning(context, context.conditioning)
            if not isinstance(context.conditioning, Conditioning):
                raise TypeError(f"{extension.extension_id}.prepare_conditioning must return Conditioning")

        initializer = self.components.latent_initializer
        encoder = self.components.latent_encoder
        if encoder is not None and isinstance(initializer, EncodedLatentInitializer):
            initialized = initializer.initialize_with_encoder(
                request,
                latent_encoder=encoder,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            initialized = initializer.initialize(
                request,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
        artifacts: Mapping[str, object] = {}
        if isinstance(initialized, LatentInitialization):
            overlap = sorted(set(context.conditioning.shared) & set(initialized.conditioning))
            if overlap:
                raise ValueError(f"initializer conditions overlap conditioner values: {overlap}")
            context.conditioning = Conditioning(
                positive=context.conditioning.positive,
                negative=context.conditioning.negative,
                shared={**context.conditioning.shared, **initialized.conditioning},
            )
            latents = initialized.latents
            artifacts = initialized.artifacts
        else:
            latents = initialized
        if not isinstance(latents, Tensor):
            raise TypeError("latent initializer must return a tensor or LatentInitialization")
        if not self._is_runtime_device(latents.device):
            raise ValueError(f"latent initializer returned {latents.device}, expected {self.device}")
        return context, latents, artifacts

    @staticmethod
    def _slice_views(
        tensor: Tensor,
        *,
        n_views: int,
        frames_per_view: int,
        start: int,
        end: int,
    ) -> Tensor:
        """Fold as ``(V, T)`` then slice the time window. Return unchanged if lengths do not match, so shared conditions are not mis-sliced."""
        if tensor.ndim == 5:
            if tensor.shape[2] != n_views * frames_per_view:
                return tensor
            value = rearrange(tensor, "B C (V T) H W -> B C V T H W", V=n_views)
            return rearrange(value[:, :, :, start:end], "B C V T H W -> B C (V T) H W")
        if tensor.ndim == 2:
            if tensor.shape[1] != n_views * frames_per_view:
                return tensor
            value = rearrange(tensor, "B (V T) -> B V T", V=n_views)
            return rearrange(value[:, :, start:end], "B V T -> B (V T)")
        return tensor

    def _window_conditioning(
        self,
        values: Mapping[str, object],
        *,
        n_views: int,
        frames_per_view: int,
        start: int,
        end: int,
        cache: object,
        frame_sequence_length: int,
        block_noise: Tensor,
    ) -> dict[str, object]:
        """Slice temporal tensors onto the current block and inject kv_cache / RoPE bounds / block_noise."""
        result = dict(values)
        for key in (
            "gt_frames",
            "condition_video_input_mask_B_C_T_H_W",
            "view_indices_B_T",
            "initial_noise",
            "control_latents",
        ):
            value = result.get(key)
            if isinstance(value, Tensor):
                result[key] = self._slice_views(
                    value,
                    n_views=n_views,
                    frames_per_view=frames_per_view,
                    start=start,
                    end=end,
                )
        prefix_frames = cast(int, values.get("kv_prefix_frames", 0))
        result.update(
            {
                "kv_cache": cache,
                "current_start": (start + prefix_frames) * frame_sequence_length * n_views,
                "current_end": (end + prefix_frames) * frame_sequence_length * n_views,
                "start_frame_for_rope": start + prefix_frames,
                "block_noise": block_noise,
            }
        )
        return result

    def _predict_window(
        self,
        context: DiffusionRunContext,
        latents: Tensor,
        *,
        positive: Mapping[str, object],
        negative: Mapping[str, object] | None,
    ) -> DenoiserOutput:
        """One in-window prediction. distilled-x0 skips CFG; flow uses classic ``ε_neg + s*(ε_pos-ε_neg)``."""
        positive_output = self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch="positive",
            conditioning=positive,
        )
        if negative is None or self.prediction_mode == "distilled-x0":
            return positive_output
        negative_output = self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch="negative",
            conditioning=negative,
        )
        scale = context.request.sampling.guidance_scale
        if self.guidance_mode == "positive":
            guided = positive_output.sample + scale * (positive_output.sample - negative_output.sample)
        else:
            guided = negative_output.sample + scale * (positive_output.sample - negative_output.sample)
        return DenoiserOutput(
            sample=guided,
            extras={"positive": positive_output.extras, "negative": negative_output.extras},
        )

    def _use_negative_branch(self, context: DiffusionRunContext) -> bool:
        """Apply the same skip rules as the canonical runner's CFG lifecycle."""

        if self.prediction_mode != "flow" or not context.conditioning.negative:
            return False
        scale = context.request.sampling.guidance_scale
        if self.guidance_mode == "standard":
            return scale != 1.0
        return scale != 0.0

    @staticmethod
    def _noise_like(value: Tensor, generator: torch.Generator) -> Tensor:
        return torch.randn(
            value.shape,
            generator=generator,
            device=value.device,
            dtype=value.dtype,
        )

    def _run_flow_window(
        self,
        context: DiffusionRunContext,
        block_noise: Tensor,
        *,
        positive: Mapping[str, object],
        negative: Mapping[str, object] | None,
    ) -> Tensor:
        schedule = tuple(
            self.components.scheduler.schedule(
                context.request.sampling,
                device=self.device,
                dtype=self.dtype,
            )
        )
        if len(schedule) != context.request.sampling.num_inference_steps:
            raise ValueError("scheduler returned an unexpected number of steps")
        latents = block_noise.clone()
        for step in schedule:
            context.step = step
            model_input = self.components.scheduler.scale_model_input(latents, step)
            prediction = self._predict_window(
                context,
                model_input,
                positive=positive,
                negative=negative,
            )
            latents = self.components.scheduler.step(
                prediction.sample,
                step,
                latents,
                generator=context.generator,
            )
            for extension in self.extensions:
                latents = extension.after_step(context, latents)
        return latents

    def _run_distilled_window(
        self,
        context: DiffusionRunContext,
        block_noise: Tensor,
        *,
        positive: Mapping[str, object],
        timesteps: Sequence[int],
    ) -> Tensor:
        if context.request.sampling.num_inference_steps != len(self.fixed_timesteps):
            raise ValueError(f"this distilled checkpoint requires {len(self.fixed_timesteps)} inference steps")
        noisy = block_noise.clone()
        shift = float(context.request.sampling.scheduler_options.get("shift", 5.0))
        for index, timestep in enumerate(timesteps):
            next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else 0
            # Gamma's released few-step configuration names the unwarped
            # training indices [1000, 750, 500, 250].  Its official
            # BaseModel.warp_denoising_step path converts those indices to
            # the shifted FlowMatch scheduler timeline before invoking the
            # network (approximately [1000, 937.5, 833.33, 625] for shift=5).
            # Keep recipes expressed in the released schedule while matching
            # the exact model-time semantics here.
            model_timestep = (
                shift_flow_sigmas(
                    torch.tensor(timestep / 1000.0, device=self.device),
                    shift,
                )
                * 1000.0
            )
            next_model_timestep = (
                shift_flow_sigmas(
                    torch.tensor(next_timestep / 1000.0, device=self.device),
                    shift,
                )
                * 1000.0
                if next_timestep
                else torch.tensor(0.0, device=self.device)
            )
            context.step = SchedulerStep(
                index=index,
                timestep=model_timestep,
                next_timestep=next_model_timestep,
            )
            prediction = self._predict_window(
                context,
                noisy,
                positive=positive,
                negative=None,
            )
            clean = prediction.sample
            if next_timestep:
                sigma = shift_flow_sigmas(
                    torch.tensor(next_timestep / 1000.0, device=clean.device),
                    shift,
                )
                if self.renoise_dtype is None:
                    noisy = add_flow_noise(clean, self._noise_like(clean, context.generator), sigma).to(
                        block_noise.dtype
                    )
                else:
                    reference = clean.to(self.renoise_dtype)
                    noise = self._noise_like(reference, context.generator)
                    noisy = self._renoise_in_dtype(reference, noise, sigma).to(block_noise.dtype)
        return clean

    @staticmethod
    def _renoise_in_dtype(reference: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
        """Re-noise ``renoise_dtype`` operands with the official FlowMatchScheduler.add_noise promotion.

        The official scheduler keeps a float32 sigma that is 0-dim when the block's
        timestep tensor has one element (batch x block frames == 1) and per-frame
        otherwise. A 0-dim float32 sigma leaves a bf16 block in bf16 arithmetic; a
        per-frame sigma promotes it to float32. The result is cast to ``noise``'s dtype.
        """
        batch, _, frames = reference.shape[:3]
        weight = sigma.to(device=reference.device, dtype=torch.float32)
        if batch * frames == 1:
            weight = weight.reshape(())
        else:
            weight = weight.reshape(1, 1, 1, 1, 1).expand(batch, 1, frames, 1, 1)
        return ((1 - weight) * reference + weight * noise).type_as(noise)

    @torch.no_grad()
    def open_stream(self, request: DiffusionRequest, *, unbounded: bool = False) -> AutoregressiveStream:
        """Open a resident rollout for ``request``; each :meth:`AutoregressiveStream.step` commits one block.

        Opening encodes conditions, runs extension ``on_run_start`` /
        ``prepare_conditioning``, draws the full-length initial noise through the
        latent initializer, creates the KV caches, and commits KV-only prefixes or
        the first-frame latent. One stream may be open per runner: opening a second
        one, or calling :meth:`run`, while a stream is open raises :exc:`RuntimeError`.

        An ``unbounded`` stream keeps stepping past the request's frames until it is
        closed. Those later blocks draw fresh noise from the rollout generator and
        are returned without being buffered, so they need a denoiser whose KV cache
        rolls within a fixed size.
        """
        if self._active_stream is not None:
            raise RuntimeError("an autoregressive stream is already open on this runner")
        stream = AutoregressiveStream(self, request, unbounded=unbounded)
        self._active_stream = stream
        return stream

    @torch.no_grad()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        """Step one stream to completion, then decode the assembled latents once."""
        stream = self.open_stream(request)
        try:
            while not stream.done:
                stream.step()
            context = stream._context
            state = stream._state
            assert state is not None
            latents = rearrange(state.output_by_view, "B C V T H W -> B C (V T) H W")
            sample = self.components.decoder.decode(latents, request)
            output_frames = stream._output_num_frames
            if output_frames is not None and not request.inputs.get("return_latent", False):
                if sample.ndim != 5 or not 0 < int(output_frames) <= sample.shape[2]:
                    raise ValueError("output_num_frames must fit the decoded BCTHW video")
                sample = sample[:, :, : int(output_frames)]
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
            result = DiffusionOutput(
                sample=sample,
                latents=latents,
                artifacts=stream._artifacts,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "guidance_scale": request.sampling.guidance_scale,
                    "execution_strategy": "autoregressive-window",
                    "prediction_mode": self.prediction_mode,
                    "block_size": stream._block_size,
                    "n_views": stream._n_views,
                    "first_frame_conditioned": stream.first_frame_latent is not None,
                    "kv_prefix_frames": stream._prefix_frames,
                },
            )
        except BaseException as error:
            stream._finish(error)
            raise
        stream.close()
        return result


@dataclass(slots=True)
class _RolloutState:
    """Tensors and caches a stream holds between blocks; dropped on close."""

    noise_by_view: Tensor
    output_by_view: Tensor
    positive_cache: Any
    negative_cache: Any
    positive_base: dict[str, object]
    negative_base: dict[str, object] | None


class AutoregressiveStream:
    """One resident ``autoregressive-window`` rollout, advanced one latent block per :meth:`step`.

    Created by :meth:`AutoregressiveWindowRunner.open_stream`. The stream owns the
    run context, generator, KV caches, full-length initial noise and latent buffer,
    so stepping to completion reproduces :meth:`AutoregressiveWindowRunner.run`
    bitwise, including one ``on_diffusion_complete`` once every block is
    committed. Decoding stays with the caller (for example
    ``WanVideoDecoder.new_stream_state()`` / ``decode_chunk()``).

    An unbounded stream never finishes: past the request's frames it steps on with
    fresh per-block noise until :meth:`close`, and never runs ``on_diffusion_complete``.

    :meth:`close` releases the caches and runs extension ``on_run_end`` once; it is
    idempotent. A failure inside opening or :meth:`step` runs ``on_run_error``
    instead, releases the caches and leaves the stream closed, so the runner can
    open the next stream.
    """

    def __init__(
        self,
        runner: AutoregressiveWindowRunner,
        request: DiffusionRequest,
        *,
        unbounded: bool = False,
    ) -> None:
        self._runner = runner
        self._unbounded = unbounded
        self._closed = False
        self._completed_blocks = 0
        self._last_block_frames: tuple[int, int] | None = None
        context: DiffusionRunContext | None = None
        try:
            context, initial_noise, artifacts = runner._prepare_context(request)
            n_views = int(context.conditioning.shared.get("n_views", 1))
            frames_per_view = int(
                context.conditioning.shared.get(
                    "latent_frames_per_view",
                    initial_noise.shape[2] // n_views,
                )
            )
            block_size = int(runner.windowed_denoiser.block_size)
            first_frame_latent = context.conditioning.shared.get("first_frame_latent")
            if first_frame_latent is not None:
                if not isinstance(first_frame_latent, Tensor) or n_views != 1:
                    raise ValueError("causal first-frame conditioning requires one tensor view")
                if first_frame_latent.shape != initial_noise[:, :, :1].shape:
                    raise ValueError("causal first-frame latent must match one noise frame")
            if first_frame_latent is None and frames_per_view % block_size:
                raise ValueError(
                    f"latent frames per view ({frames_per_view}) must be divisible by block_size={block_size}"
                )
            prefix = context.conditioning.shared.get("kv_prefix_latents")
            prefix_frames = 0
            if prefix is not None:
                if not isinstance(prefix, Tensor) or prefix.ndim != 5 or n_views != 1:
                    raise ValueError("KV-only prefixes require a BCTHW tensor and one view")
                if prefix.shape[:2] != initial_noise.shape[:2] or prefix.shape[-2:] != initial_noise.shape[-2:]:
                    raise ValueError("KV-only prefix geometry must match the generated latents")
                if first_frame_latent is not None:
                    raise ValueError("KV-only and output first-frame prefixes cannot be combined")
                prefix_frames = prefix.shape[2]
            frame_sequence_length = int(initial_noise.shape[-2] * initial_noise.shape[-1] // 4)
            cache_kwargs = {
                "batch_size": initial_noise.shape[0],
                "n_views": n_views,
                "latent_frames_per_view": frames_per_view + prefix_frames,
                "frame_sequence_length": frame_sequence_length,
                "device": initial_noise.device,
                "dtype": runner.dtype,
            }
            positive_cache = runner.windowed_denoiser.create_kv_cache(**cache_kwargs)
            use_negative = runner._use_negative_branch(context)
            negative_cache = runner.windowed_denoiser.create_kv_cache(**cache_kwargs) if use_negative else None

            output = torch.zeros_like(initial_noise)
            noise_by_view = rearrange(initial_noise, "B C (V T) H W -> B C V T H W", V=n_views)
            output_by_view = rearrange(output, "B C (V T) H W -> B C V T H W", V=n_views)
            shared = dict(context.conditioning.shared)
            shared["kv_prefix_frames"] = prefix_frames
            positive_base = {**shared, **context.conditioning.positive}
            negative_base = {**shared, **context.conditioning.negative} if use_negative else None

            if prefix is not None:
                context.step = SchedulerStep(
                    index=0,
                    timestep=torch.tensor(0.0, device=runner.device),
                    next_timestep=torch.tensor(0.0, device=runner.device),
                )
                for branch, base, cache in (
                    ("positive-cache-commit", positive_base, positive_cache),
                    ("negative-cache-commit", negative_base, negative_cache),
                ):
                    if base is None:
                        continue
                    conditioning = runner._window_conditioning(
                        base,
                        n_views=1,
                        frames_per_view=frames_per_view,
                        start=0,
                        end=prefix_frames,
                        cache=cache,
                        frame_sequence_length=frame_sequence_length,
                        block_noise=prefix,
                    )
                    conditioning.update(
                        current_start=0, current_end=prefix_frames * frame_sequence_length, start_frame_for_rope=0
                    )
                    runner._call_denoiser_with_conditioning(
                        context,
                        latents=prefix,
                        branch=branch,
                        conditioning=conditioning,
                    )

            start_frame = 0
            if first_frame_latent is not None:
                # FastVideo I2V encodes the source image as a clean latent,
                # commits it to both experts at timestep zero, then rolls out
                # only the remaining noisy frames.
                output_by_view[:, :, :, :1] = first_frame_latent.unsqueeze(2)
                context.step = SchedulerStep(
                    index=max(request.sampling.num_inference_steps - 1, 0),
                    timestep=torch.tensor(0.0, device=runner.device),
                    next_timestep=torch.tensor(0.0, device=runner.device),
                )
                positive_seed = runner._window_conditioning(
                    positive_base,
                    n_views=1,
                    frames_per_view=frames_per_view,
                    start=0,
                    end=1,
                    cache=positive_cache,
                    frame_sequence_length=frame_sequence_length,
                    block_noise=first_frame_latent,
                )
                runner._call_denoiser_with_conditioning(
                    context,
                    latents=first_frame_latent,
                    branch="positive-cache-commit",
                    conditioning=positive_seed,
                )
                if negative_base is not None and negative_cache is not None:
                    negative_seed = runner._window_conditioning(
                        negative_base,
                        n_views=1,
                        frames_per_view=frames_per_view,
                        start=0,
                        end=1,
                        cache=negative_cache,
                        frame_sequence_length=frame_sequence_length,
                        block_noise=first_frame_latent,
                    )
                    runner._call_denoiser_with_conditioning(
                        context,
                        latents=first_frame_latent,
                        branch="negative-cache-commit",
                        conditioning=negative_seed,
                    )
                start_frame = 1
            if start_frame >= frames_per_view:
                # A first-frame-only request has no noisy blocks; the committed
                # first frame is already the complete rollout.
                runner._notify_diffusion_complete(
                    context,
                    rearrange(output_by_view, "B C V T H W -> B C (V T) H W"),
                )
        except BaseException as error:
            try:
                if context is not None:
                    for extension in reversed(runner.extensions):
                        extension.on_run_error(context, error)
            finally:
                runner._end_denoiser_request(context, error=error)
            raise

        self._context = context
        self._artifacts = artifacts
        self._n_views = n_views
        self._frames_per_view = frames_per_view
        self._block_size = block_size
        self._prefix_frames = prefix_frames
        self._first_frame_latent = (
            None
            if first_frame_latent is None
            else rearrange(output_by_view[:, :, :, :1], "B C V T H W -> B C (V T) H W").clone()
        )
        self._frame_sequence_length = frame_sequence_length
        self._output_num_frames = cast(int | None, shared.get("output_num_frames"))
        self._block_starts = tuple(range(start_frame, frames_per_view, block_size))
        self._state: _RolloutState | None = _RolloutState(
            noise_by_view=noise_by_view,
            output_by_view=output_by_view,
            positive_cache=positive_cache,
            negative_cache=negative_cache,
            positive_base=positive_base,
            negative_base=negative_base,
        )

    @property
    def total_blocks(self) -> int | None:
        """Number of noisy blocks in the rollout, excluding prefix and first-frame commits; ``None`` if unbounded."""
        return None if self._unbounded else len(self._block_starts)

    @property
    def completed_blocks(self) -> int:
        """Number of blocks denoised and committed so far."""
        return self._completed_blocks

    @property
    def done(self) -> bool:
        """Whether every block has been committed; an unbounded stream is never done."""
        return not self._unbounded and self._completed_blocks == len(self._block_starts)

    @property
    def last_block_frames(self) -> tuple[int, int] | None:
        """Per-view latent frame range ``[start, end)`` of the latest :meth:`step`, or ``None`` before it."""
        return self._last_block_frames

    @property
    def first_frame_latent(self) -> Tensor | None:
        """Committed frame-0 latent ``B C 1 H W`` of a first-frame request, or ``None``.

        :meth:`step` starts after this frame, so streaming decoders take it from here.
        """
        return self._first_frame_latent

    @torch.no_grad()
    def step(self) -> Tensor:
        """Denoise the next block, commit it to the KV caches, and return its latents.

        The result is a ``B C (V T) H W`` slice of the stream's latent buffer in the
        layout of ``DiffusionOutput.latents``, covering :attr:`last_block_frames` of
        every view. The step that commits the last block also runs extension
        ``on_diffusion_complete`` with the full latent buffer. Blocks of an unbounded
        stream past the request's frames are returned as they are, without a buffer.
        Stepping a closed
        or finished stream raises :exc:`RuntimeError`;
        any other failure closes the stream through ``on_run_error`` and re-raises.
        """
        state = self._state
        if state is None:
            raise RuntimeError("autoregressive stream is closed")
        if self.done:
            raise RuntimeError("autoregressive stream has no blocks left")
        runner = self._runner
        context = self._context
        n_views = self._n_views
        frames_per_view = self._frames_per_view
        frame_sequence_length = self._frame_sequence_length
        buffered = self._completed_blocks < len(self._block_starts)
        if buffered:
            start = self._block_starts[self._completed_blocks]
            end = min(start + self._block_size, frames_per_view)
        else:
            start = frames_per_view + (self._completed_blocks - len(self._block_starts)) * self._block_size
            end = start + self._block_size
        try:
            if buffered:
                block_noise = rearrange(
                    state.noise_by_view[:, :, :, start:end],
                    "B C V T H W -> B C (V T) H W",
                )
            else:
                batch, channels, views, _, height, width = state.noise_by_view.shape
                block_noise = rearrange(
                    torch.randn(
                        (batch, channels, views, end - start, height, width),
                        generator=context.generator,
                        device=state.noise_by_view.device,
                        dtype=state.noise_by_view.dtype,
                    ),
                    "B C V T H W -> B C (V T) H W",
                )
            positive = runner._window_conditioning(
                state.positive_base,
                n_views=n_views,
                frames_per_view=frames_per_view,
                start=start,
                end=end,
                cache=state.positive_cache,
                frame_sequence_length=frame_sequence_length,
                block_noise=block_noise,
            )
            negative = None
            if state.negative_base is not None and state.negative_cache is not None:
                negative = runner._window_conditioning(
                    state.negative_base,
                    n_views=n_views,
                    frames_per_view=frames_per_view,
                    start=start,
                    end=end,
                    cache=state.negative_cache,
                    frame_sequence_length=frame_sequence_length,
                    block_noise=block_noise,
                )

            if runner.prediction_mode == "flow":
                block = runner._run_flow_window(
                    context,
                    block_noise,
                    positive=positive,
                    negative=negative,
                )
            else:
                block = runner._run_distilled_window(
                    context,
                    block_noise,
                    positive=positive,
                    timesteps=(
                        runner.first_block_timesteps
                        if self._completed_blocks == 0 and runner.first_block_timesteps
                        else runner.fixed_timesteps
                    ),
                )
            if buffered:
                state.output_by_view[:, :, :, start:end] = rearrange(block, "B C (V T) H W -> B C V T H W", V=n_views)

            commit = block
            commit_timestep = runner.context_timestep if runner.prediction_mode == "distilled-x0" else 0
            if commit_timestep > 0:
                # ``context_timestep`` is already expressed on the
                # model's shifted timeline.  Gamma's official runtime
                # passes 128 directly to both FlowMatchScheduler.add_noise
                # and the denoiser, which resolves to sigma=0.128.  Only
                # the released *denoising* indices above need the
                # raw-to-shifted conversion; shifting the context value a
                # second time injects sigma~=0.423 noise into every cache
                # commit and causes progressive block corruption.
                sigma = torch.tensor(
                    commit_timestep / 1000.0,
                    device=commit.device,
                )
                commit = add_flow_noise(
                    commit,
                    runner._noise_like(commit, context.generator),
                    sigma,
                )
            context.step = SchedulerStep(
                index=max(context.request.sampling.num_inference_steps - 1, 0),
                timestep=torch.tensor(commit_timestep, device=runner.device),
                next_timestep=torch.tensor(0, device=runner.device),
            )
            runner._call_denoiser_with_conditioning(
                context,
                latents=commit,
                branch="positive-cache-commit",
                conditioning=positive,
            )
            if negative is not None:
                runner._call_denoiser_with_conditioning(
                    context,
                    latents=commit,
                    branch="negative-cache-commit",
                    conditioning=negative,
                )
            self._completed_blocks += 1
            self._last_block_frames = (start, end)
            if self.done:
                runner._notify_diffusion_complete(
                    context,
                    rearrange(state.output_by_view, "B C V T H W -> B C (V T) H W"),
                )
        except BaseException as error:
            self._finish(error)
            raise
        if not buffered:
            return block
        return rearrange(state.output_by_view[:, :, :, start:end], "B C V T H W -> B C (V T) H W")

    def close(self) -> None:
        """Release the caches and buffers, then run extension ``on_run_end``. Idempotent."""
        self._finish(None)

    def _finish(self, error: BaseException | None) -> None:
        """Close once: drop rollout state, run end or error hooks, end the denoiser request."""
        if self._closed:
            return
        self._closed = True
        self._state = None
        runner = self._runner
        if runner._active_stream is self:
            runner._active_stream = None
        context = self._context
        try:
            if error is None:
                try:
                    for extension in reversed(runner.extensions):
                        extension.on_run_end(context)
                except BaseException as end_error:
                    error = end_error
                    for extension in reversed(runner.extensions):
                        extension.on_run_error(context, end_error)
                    raise
            else:
                for extension in reversed(runner.extensions):
                    extension.on_run_error(context, error)
        finally:
            runner._end_denoiser_request(context, error=error)


__all__ = ["AutoregressiveStream", "AutoregressiveWindowRunner", "WindowedDenoiser"]

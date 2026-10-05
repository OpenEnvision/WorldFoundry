"""Framework-owned windowed autoregressive diffusion execution.

Implements the ``autoregressive-window`` strategy runner at the bottom of:

    recipe.execution.strategy == "autoregressive-window"
        → build_autoregressive_window_strategy
        → AutoregressiveWindowRunner.run

Inputs: WindowedDenoiser plus canonical RunnerComponents, prediction_mode
(flow / distilled-x0), fixed_timesteps, and context_timestep.
Output: DiffusionOutput; metadata includes block_size / n_views / prediction_mode.
Must not: let a model package own KV-cache windows or bypass
WindowedDenoiser.create_kv_cache; on the distilled-x0 path, do not shift
context_timestep a second time (see the cache-commit comment).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

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
    Construction: non-WindowedDenoiser → :exc:`TypeError`; illegal mode or missing
    fixed_timesteps → :exc:`ValueError`.
    """

    def __init__(
        self,
        *,
        prediction_mode: str = "flow",
        fixed_timesteps: Sequence[int] = (),
        context_timestep: int = 0,
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
        self.context_timestep = int(context_timestep)
        if self.prediction_mode == "distilled-x0" and not self.fixed_timesteps:
            raise ValueError("distilled-x0 execution requires fixed_timesteps")

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
        result.update(
            {
                "kv_cache": cache,
                "current_start": start * frame_sequence_length * n_views,
                "current_end": end * frame_sequence_length * n_views,
                "start_frame_for_rope": start,
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
            guided = positive_output.sample + scale * (
                positive_output.sample - negative_output.sample
            )
        else:
            guided = negative_output.sample + scale * (
                positive_output.sample - negative_output.sample
            )
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
    ) -> Tensor:
        if context.request.sampling.num_inference_steps != len(self.fixed_timesteps):
            raise ValueError(
                f"this distilled checkpoint requires {len(self.fixed_timesteps)} inference steps"
            )
        noisy = block_noise.clone()
        shift = float(context.request.sampling.scheduler_options.get("shift", 5.0))
        for index, timestep in enumerate(self.fixed_timesteps):
            next_timestep = self.fixed_timesteps[index + 1] if index + 1 < len(self.fixed_timesteps) else 0
            # Gamma's released few-step configuration names the unwarped
            # training indices [1000, 750, 500, 250].  Its official
            # BaseModel.warp_denoising_step path converts those indices to
            # the shifted FlowMatch scheduler timeline before invoking the
            # network (approximately [1000, 937.5, 833.33, 625] for shift=5).
            # Keep recipes expressed in the released schedule while matching
            # the exact model-time semantics here.
            model_timestep = shift_flow_sigmas(
                torch.tensor(timestep / 1000.0, device=self.device),
                shift,
            ) * 1000.0
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
                noisy = add_flow_noise(clean, self._noise_like(clean, context.generator), sigma)
        return clean

    @torch.no_grad()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        context: DiffusionRunContext | None = None
        run_error: BaseException | None = None
        try:
            context, initial_noise, artifacts = self._prepare_context(request)
            n_views = int(context.conditioning.shared.get("n_views", 1))
            frames_per_view = int(
                context.conditioning.shared.get(
                    "latent_frames_per_view",
                    initial_noise.shape[2] // n_views,
                )
            )
            block_size = int(self.windowed_denoiser.block_size)
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
            frame_sequence_length = int(initial_noise.shape[-2] * initial_noise.shape[-1] // 4)
            cache_kwargs = {
                "batch_size": initial_noise.shape[0],
                "n_views": n_views,
                "latent_frames_per_view": frames_per_view,
                "frame_sequence_length": frame_sequence_length,
                "device": initial_noise.device,
                "dtype": self.dtype,
            }
            positive_cache = self.windowed_denoiser.create_kv_cache(**cache_kwargs)
            use_negative = self._use_negative_branch(context)
            negative_cache = self.windowed_denoiser.create_kv_cache(**cache_kwargs) if use_negative else None

            output = torch.zeros_like(initial_noise)
            noise_by_view = rearrange(initial_noise, "B C (V T) H W -> B C V T H W", V=n_views)
            output_by_view = rearrange(output, "B C (V T) H W -> B C V T H W", V=n_views)
            shared = dict(context.conditioning.shared)
            positive_base = {**shared, **context.conditioning.positive}
            negative_base = (
                {**shared, **context.conditioning.negative} if use_negative else None
            )

            start_frame = 0
            if first_frame_latent is not None:
                # FastVideo I2V encodes the source image as a clean latent,
                # commits it to both experts at timestep zero, then rolls out
                # only the remaining noisy frames.
                output_by_view[:, :, :, :1] = first_frame_latent.unsqueeze(2)
                context.step = SchedulerStep(
                    index=max(request.sampling.num_inference_steps - 1, 0),
                    timestep=torch.tensor(0.0, device=self.device),
                    next_timestep=torch.tensor(0.0, device=self.device),
                )
                positive_seed = self._window_conditioning(
                    positive_base,
                    n_views=1,
                    frames_per_view=frames_per_view,
                    start=0,
                    end=1,
                    cache=positive_cache,
                    frame_sequence_length=frame_sequence_length,
                    block_noise=first_frame_latent,
                )
                self._call_denoiser_with_conditioning(
                    context,
                    latents=first_frame_latent,
                    branch="positive-cache-commit",
                    conditioning=positive_seed,
                )
                if negative_base is not None and negative_cache is not None:
                    negative_seed = self._window_conditioning(
                        negative_base,
                        n_views=1,
                        frames_per_view=frames_per_view,
                        start=0,
                        end=1,
                        cache=negative_cache,
                        frame_sequence_length=frame_sequence_length,
                        block_noise=first_frame_latent,
                    )
                    self._call_denoiser_with_conditioning(
                        context,
                        latents=first_frame_latent,
                        branch="negative-cache-commit",
                        conditioning=negative_seed,
                    )
                start_frame = 1

            for start in range(start_frame, frames_per_view, block_size):
                end = min(start + block_size, frames_per_view)
                block_noise = rearrange(
                    noise_by_view[:, :, :, start:end],
                    "B C V T H W -> B C (V T) H W",
                )
                positive = self._window_conditioning(
                    positive_base,
                    n_views=n_views,
                    frames_per_view=frames_per_view,
                    start=start,
                    end=end,
                    cache=positive_cache,
                    frame_sequence_length=frame_sequence_length,
                    block_noise=block_noise,
                )
                negative = None
                if negative_base is not None and negative_cache is not None:
                    negative = self._window_conditioning(
                        negative_base,
                        n_views=n_views,
                        frames_per_view=frames_per_view,
                        start=start,
                        end=end,
                        cache=negative_cache,
                        frame_sequence_length=frame_sequence_length,
                        block_noise=block_noise,
                    )

                if self.prediction_mode == "flow":
                    block = self._run_flow_window(
                        context,
                        block_noise,
                        positive=positive,
                        negative=negative,
                    )
                else:
                    block = self._run_distilled_window(
                        context,
                        block_noise,
                        positive=positive,
                    )
                output_by_view[:, :, :, start:end] = rearrange(
                    block, "B C (V T) H W -> B C V T H W", V=n_views
                )

                commit = block
                commit_timestep = self.context_timestep if self.prediction_mode == "distilled-x0" else 0
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
                        self._noise_like(commit, context.generator),
                        sigma,
                    )
                context.step = SchedulerStep(
                    index=max(request.sampling.num_inference_steps - 1, 0),
                    timestep=torch.tensor(commit_timestep, device=self.device),
                    next_timestep=torch.tensor(0, device=self.device),
                )
                self._call_denoiser_with_conditioning(
                    context,
                    latents=commit,
                    branch="positive-cache-commit",
                    conditioning=positive,
                )
                if negative is not None:
                    self._call_denoiser_with_conditioning(
                        context,
                        latents=commit,
                        branch="negative-cache-commit",
                        conditioning=negative,
                    )

            latents = rearrange(output_by_view, "B C V T H W -> B C (V T) H W")
            self._notify_diffusion_complete(context, latents)
            sample = self.components.decoder.decode(latents, request)
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
            result = DiffusionOutput(
                sample=sample,
                latents=latents,
                artifacts=artifacts,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "guidance_scale": request.sampling.guidance_scale,
                    "execution_strategy": "autoregressive-window",
                    "prediction_mode": self.prediction_mode,
                    "block_size": block_size,
                    "n_views": n_views,
                    "first_frame_conditioned": first_frame_latent is not None,
                },
            )
            for extension in reversed(self.extensions):
                extension.on_run_end(context)
            return result
        except BaseException as error:
            run_error = error
            if context is not None:
                for extension in reversed(self.extensions):
                    extension.on_run_error(context, error)
            raise
        finally:
            self._end_denoiser_request(context, error=run_error)


__all__ = ["AutoregressiveWindowRunner", "WindowedDenoiser"]

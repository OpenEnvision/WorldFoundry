"""Framework-owned prefix-recomputation diffusion execution.

The ``prefix-recompute`` strategy is used by chunk-causal models whose
released inference path recomputes a bounded clean prefix for every active
chunk instead of retaining model-owned KV state.  The runner owns temporal
window selection, clean-context timesteps, scheduler updates, and optional
active-chunk refinement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ..contracts import (
    ConditionEncoder,
    Conditioning,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    EncodedLatentInitializer,
    LatentInitialization,
    SchedulerStep,
)
from ..extensions import DiffusionRunContext
from ..models.networks.sana.chunking import chunk_index_from_chunk_size
from .base import NativeDiffusionRunner


@dataclass(frozen=True, slots=True)
class PrefixRecomputeWindow:
    """Absolute frame indices plus the local active span for one model call."""

    frame_indices: tuple[int, ...]
    active_start: int
    active_end: int


@runtime_checkable
class PrefixRefiner(Protocol):
    """Optional active-chunk refiner surface consumed by this runner."""

    def refine_active_latents(
        self,
        *,
        context_latents: Tensor,
        active_latents: Tensor,
        conditioning: Mapping[str, object],
        fps: float,
        generator: torch.Generator,
    ) -> Tensor:
        """Refine only ``active_latents`` while keeping context frozen."""


def prefix_chunk_boundaries(total_frames: int, chunk_size: int) -> tuple[int, ...]:
    """Return ``(0, chunk_size + 1, ...)`` boundaries for a sink-plus-chunks rollout."""

    if total_frames <= 1:
        raise ValueError("prefix recomputation requires more than one latent frame")
    if chunk_size <= 0:
        raise ValueError("prefix recomputation chunk_size must be positive")
    active_frames = total_frames - 1
    if active_frames % chunk_size:
        raise ValueError(
            "active latent frames must divide prefix chunk_size: "
            f"active={active_frames}, chunk_size={chunk_size}"
        )
    boundaries = [0, chunk_size + 1]
    while boundaries[-1] < total_frames:
        boundaries.append(boundaries[-1] + chunk_size)
    return tuple(boundaries)


def prefix_recompute_window(
    *,
    start: int,
    end: int,
    history_frames: int,
    sink_frames: int = 1,
) -> PrefixRecomputeWindow:
    """Build a sink + recent-history + active window using absolute frame indices."""

    if start < 0 or end <= start:
        raise ValueError(f"invalid prefix chunk bounds: start={start}, end={end}")
    if history_frames < 0 or sink_frames < 0:
        raise ValueError("prefix history_frames and sink_frames must be non-negative")
    if start == 0:
        if sink_frames >= end:
            raise ValueError("the first prefix chunk must contain active frames after the sink")
        return PrefixRecomputeWindow(
            frame_indices=tuple(range(end)),
            active_start=sink_frames,
            active_end=end,
        )

    retained_sink = min(sink_frames, start)
    history_start = max(retained_sink, start - history_frames)
    frame_indices = tuple(range(retained_sink)) + tuple(range(history_start, end))
    active_frames = end - start
    active_start = len(frame_indices) - active_frames
    return PrefixRecomputeWindow(
        frame_indices=frame_indices,
        active_start=active_start,
        active_end=len(frame_indices),
    )


def _condition_frame_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def remap_condition_frame_info(
    value: object,
    frame_indices: tuple[int, ...],
) -> dict[object, object]:
    """Map absolute conditioned-frame keys onto one non-contiguous local window."""

    if not isinstance(value, Mapping):
        return {}
    absolute_to_local = {frame: local for local, frame in enumerate(frame_indices)}
    result: dict[object, object] = {}
    for raw_frame, weight in value.items():
        absolute = _condition_frame_index(raw_frame)
        if absolute is None:
            continue
        local = absolute_to_local.get(absolute)
        if local is not None:
            result[local] = weight
    return result


def slice_prefix_tensor(
    value: Tensor,
    *,
    frame_indices: tuple[int, ...],
    total_frames: int,
) -> Tensor:
    """Slice BCTHW or BT... temporal conditioning with non-contiguous indices."""

    index = torch.tensor(frame_indices, device=value.device, dtype=torch.long)
    if value.ndim == 5 and value.shape[2] >= total_frames:
        return value.index_select(2, index).contiguous()
    if value.ndim >= 3 and value.shape[1] >= total_frames:
        return value.index_select(1, index).contiguous()
    return value


def slice_prefix_conditioning(
    values: Mapping[str, object],
    *,
    frame_indices: tuple[int, ...],
    active_start: int,
    active_end: int,
    total_frames: int,
    chunk_size: int,
) -> dict[str, object]:
    """Slice camera/Pluecker tensors and rebuild local frame/chunk metadata."""

    result: dict[str, object] = {}
    for key, value in values.items():
        if key in {"camera_cache", "rotary_emb", "chunk_plucker_emb"}:
            continue
        if key == "condition_frame_info":
            result[key] = remap_condition_frame_info(value, frame_indices)
        elif key == "data_info" and isinstance(value, Mapping):
            data_info = dict(value)
            data_info["condition_frame_info"] = remap_condition_frame_info(
                value.get("condition_frame_info"),
                frame_indices,
            )
            result[key] = data_info
        elif isinstance(value, Tensor):
            result[key] = slice_prefix_tensor(
                value,
                frame_indices=frame_indices,
                total_frames=total_frames,
            )
        elif isinstance(value, Mapping):
            result[key] = dict(value)
        else:
            result[key] = value

    # Every retained history frame is clean context, even if the initializer
    # only named the anchor in its absolute condition_frame_info mapping.
    frame_info = dict(result.get("condition_frame_info", {}))
    frame_info.update(
        {local: 0.0 for local in range(len(frame_indices)) if not active_start <= local < active_end}
    )
    result["condition_frame_info"] = frame_info
    result["chunk_index"] = chunk_index_from_chunk_size(
        len(frame_indices),
        chunk_size,
        strategy="first_chunk_plus_one",
    )
    return result


class PrefixRecomputeRunner(NativeDiffusionRunner):
    """Denoise active chunks against a bounded, clean, recomputed prefix."""

    def __init__(
        self,
        *,
        chunk_size: int = 3,
        history_frames: int = 6,
        sink_frames: int = 1,
        refiner: PrefixRefiner | None = None,
        refiner_conditioner: ConditionEncoder | None = None,
        refiner_max_frames: int = 11,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.chunk_size = int(chunk_size)
        self.history_frames = int(history_frames)
        self.sink_frames = int(sink_frames)
        self.refiner_max_frames = int(refiner_max_frames)
        if self.chunk_size <= 0:
            raise ValueError("prefix-recompute chunk_size must be positive")
        if self.history_frames < 0 or self.sink_frames <= 0:
            raise ValueError("prefix-recompute requires non-negative history and a positive sink")
        if self.refiner_max_frames <= self.chunk_size:
            raise ValueError("refiner_max_frames must leave room for clean context")
        if (refiner is None) != (refiner_conditioner is None):
            raise ValueError("prefix-recompute refiner and refiner_conditioner must be bound together")
        if refiner is not None and not isinstance(refiner, PrefixRefiner):
            raise TypeError("prefix-recompute refiner does not implement refine_active_latents")
        self.refiner = refiner
        self.refiner_conditioner = refiner_conditioner

    def _prepare_context(
        self,
        request: DiffusionRequest,
    ) -> tuple[DiffusionRunContext, Tensor, Mapping[str, object]]:
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
                raise TypeError(
                    f"{extension.extension_id}.prepare_conditioning must return Conditioning"
                )

        initializer = self.components.latent_initializer
        encoder = self.components.latent_encoder
        if encoder is None or not isinstance(initializer, EncodedLatentInitializer):
            raise TypeError("prefix-recompute requires an encoded latent initializer")
        initialized = initializer.initialize_with_encoder(
            request,
            latent_encoder=encoder,
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
        if latents.ndim != 5:
            raise ValueError("prefix-recompute requires BCTHW latents")
        if not self._is_runtime_device(latents.device):
            raise ValueError(f"latent initializer returned {latents.device}, expected {self.device}")
        return context, latents, artifacts

    @staticmethod
    def _temporal_timestep(
        value: Tensor,
        *,
        batch_size: int,
        window_frames: int,
        active_start: int,
        active_end: int,
        device: torch.device,
    ) -> Tensor:
        current = value.to(device=device, dtype=torch.float32)
        if current.numel() == 1:
            current = current.reshape(1, 1, 1).expand(batch_size, 1, active_end - active_start)
        elif current.ndim == 1 and current.numel() == batch_size:
            current = current.reshape(batch_size, 1, 1).expand(
                batch_size,
                1,
                active_end - active_start,
            )
        elif current.shape != (batch_size, 1, active_end - active_start):
            raise ValueError(
                "prefix scheduler timestep must be scalar, [B], or [B,1,active_T]"
            )
        result = torch.zeros(
            batch_size,
            1,
            window_frames,
            device=device,
            dtype=torch.float32,
        )
        result[:, :, active_start:active_end] = current
        return result

    def _use_negative_branch(self, context: DiffusionRunContext) -> bool:
        if not context.conditioning.negative:
            return False
        scale = context.request.sampling.guidance_scale
        return scale != (1.0 if self.guidance_mode == "standard" else 0.0)

    def _predict_window(
        self,
        context: DiffusionRunContext,
        latents: Tensor,
        *,
        positive: Mapping[str, object],
        negative: Mapping[str, object] | None,
    ) -> DenoiserOutput:
        positive_output = self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch="positive",
            conditioning=positive,
        )
        if negative is None:
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

    @staticmethod
    def _refiner_context(
        latents: Tensor,
        *,
        active_start: int,
        context_frames: int,
        sink_frames: int,
    ) -> Tensor:
        if active_start <= sink_frames or context_frames <= sink_frames:
            return latents[:, :, :sink_frames].contiguous()
        history_budget = context_frames - sink_frames
        history_start = max(sink_frames, active_start - history_budget)
        return torch.cat(
            (
                latents[:, :, :sink_frames],
                latents[:, :, history_start:active_start],
            ),
            dim=2,
        ).contiguous()

    @torch.no_grad()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        context: DiffusionRunContext | None = None
        run_error: BaseException | None = None
        try:
            context, initial_noise, artifacts = self._prepare_context(request)
            total_frames = int(initial_noise.shape[2])
            boundaries = prefix_chunk_boundaries(total_frames, self.chunk_size)
            schedule = tuple(
                self.components.scheduler.schedule(
                    request.sampling,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            if len(schedule) != request.sampling.num_inference_steps:
                raise ValueError("scheduler returned an unexpected number of steps")
            for expected_index, step in enumerate(schedule):
                if step.index != expected_index:
                    raise ValueError("scheduler step indices must be contiguous and zero-based")

            use_negative = self._use_negative_branch(context)
            shared = dict(context.conditioning.shared)
            positive_base = {**shared, **context.conditioning.positive}
            negative_base = (
                {**shared, **context.conditioning.negative} if use_negative else None
            )
            refiner_conditioning: Mapping[str, object] | None = None
            refiner_generator: torch.Generator | None = None
            if self.refiner_conditioner is not None:
                encoded = self.refiner_conditioner.encode(
                    request,
                    device=self.device,
                    dtype=self.dtype,
                )
                if not isinstance(encoded, Conditioning):
                    raise TypeError("refiner_conditioner.encode must return Conditioning")
                refiner_conditioning = {**encoded.shared, **encoded.positive}
                refiner_seed = int(request.inputs.get("refiner_seed", request.sampling.seed))
                refiner_generator = self._generator(refiner_seed)

            stage1_state = initial_noise.clone()
            refined_state = initial_noise.clone()
            anchor = initial_noise[:, :, : self.sink_frames].clone()
            for chunk_number, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                absolute_active_start = self.sink_frames if start == 0 else start
                window = prefix_recompute_window(
                    start=start,
                    end=end,
                    history_frames=self.history_frames,
                    sink_frames=self.sink_frames,
                )
                index = torch.tensor(
                    window.frame_indices,
                    device=stage1_state.device,
                    dtype=torch.long,
                )
                positive = slice_prefix_conditioning(
                    positive_base,
                    frame_indices=window.frame_indices,
                    active_start=window.active_start,
                    active_end=window.active_end,
                    total_frames=total_frames,
                    chunk_size=self.chunk_size,
                )
                negative = (
                    slice_prefix_conditioning(
                        negative_base,
                        frame_indices=window.frame_indices,
                        active_start=window.active_start,
                        active_end=window.active_end,
                        total_frames=total_frames,
                        chunk_size=self.chunk_size,
                    )
                    if negative_base is not None
                    else None
                )
                active = stage1_state[:, :, absolute_active_start:end].clone()
                for step in schedule:
                    model_latents = stage1_state.index_select(2, index).clone()
                    scaled_active = self.components.scheduler.scale_model_input(active, step)
                    model_latents[:, :, window.active_start : window.active_end] = scaled_active
                    model_step = SchedulerStep(
                        index=step.index,
                        timestep=self._temporal_timestep(
                            step.timestep,
                            batch_size=active.shape[0],
                            window_frames=len(window.frame_indices),
                            active_start=window.active_start,
                            active_end=window.active_end,
                            device=active.device,
                        ),
                        next_timestep=self._temporal_timestep(
                            step.next_timestep,
                            batch_size=active.shape[0],
                            window_frames=len(window.frame_indices),
                            active_start=window.active_start,
                            active_end=window.active_end,
                            device=active.device,
                        ),
                    )
                    context.step = model_step
                    prediction = self._predict_window(
                        context,
                        model_latents,
                        positive=positive,
                        negative=negative,
                    )
                    active_prediction = prediction.sample[
                        :, :, window.active_start : window.active_end
                    ]
                    context.step = step
                    active = self.components.scheduler.step(
                        active_prediction,
                        step,
                        active,
                        generator=context.generator,
                    )
                    if not isinstance(active, Tensor):
                        raise TypeError("scheduler.step must return a tensor")
                    for extension in self.extensions:
                        active = extension.after_step(context, active)
                        if not isinstance(active, Tensor):
                            raise TypeError(
                                f"{extension.extension_id}.after_step must return a tensor"
                            )
                    stage1_state[:, :, absolute_active_start:end] = active
                    stage1_state[:, :, : self.sink_frames] = anchor

                refined_active = active
                if self.refiner is not None:
                    if refiner_conditioning is None or refiner_generator is None:
                        raise RuntimeError("prefix refiner conditioning was not initialized")
                    context_budget = max(
                        self.sink_frames,
                        self.refiner_max_frames - int(active.shape[2]),
                    )
                    refiner_context = self._refiner_context(
                        refined_state,
                        active_start=absolute_active_start,
                        context_frames=context_budget,
                        sink_frames=self.sink_frames,
                    )
                    refined_active = self.refiner.refine_active_latents(
                        context_latents=refiner_context,
                        active_latents=active,
                        conditioning=refiner_conditioning,
                        fps=float(request.inputs.get("fps", 16)),
                        generator=refiner_generator,
                    )
                    if not isinstance(refined_active, Tensor):
                        raise TypeError("prefix refiner must return a tensor")
                    if refined_active.shape != active.shape:
                        raise ValueError(
                            "prefix refiner output shape must match active latents: "
                            f"{tuple(refined_active.shape)} != {tuple(active.shape)}"
                        )
                refined_state[:, :, absolute_active_start:end] = refined_active
                refined_state[:, :, : self.sink_frames] = anchor

            self._notify_diffusion_complete(context, refined_state)
            sample = self.components.decoder.decode(refined_state, request)
            if not isinstance(sample, Tensor):
                raise TypeError("decoder.decode must return a tensor")
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
                if not isinstance(sample, Tensor):
                    raise TypeError(
                        f"{extension.extension_id}.after_decode must return a tensor"
                    )
            output = DiffusionOutput(
                sample=sample,
                latents=refined_state,
                artifacts=artifacts,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "guidance_scale": request.sampling.guidance_scale,
                    "execution_strategy": "prefix-recompute",
                    "chunk_size": self.chunk_size,
                    "history_frames": self.history_frames,
                    "sink_frames": self.sink_frames,
                    "num_chunks": len(boundaries) - 1,
                    "refiner": self.refiner is not None,
                },
            )
            for extension in reversed(self.extensions):
                extension.on_run_end(context)
            return output
        except BaseException as error:
            run_error = error
            if context is not None:
                for extension in reversed(self.extensions):
                    extension.on_run_error(context, error)
            raise
        finally:
            self._end_denoiser_request(context, error=run_error)


__all__ = [
    "PrefixRecomputeRunner",
    "PrefixRecomputeWindow",
    "PrefixRefiner",
    "prefix_chunk_boundaries",
    "prefix_recompute_window",
    "remap_condition_frame_info",
    "slice_prefix_conditioning",
    "slice_prefix_tensor",
]

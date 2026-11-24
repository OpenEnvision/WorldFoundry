"""Framework-owned joint-modality, multi-stage diffusion execution.

Implements ``joint-multistage``: recipe bindings declare ``scheduler-*`` roles;
the assembler hands constructed components here. This runner does not use the
standard NativeDiffusionRunner loop.

Inputs: MultiStageComponents, stage_steps matching the scheduler count.
Output: DiffusionOutput whose artifacts come from decode_modalities (must include ``video``).
Must not: skip the latent processor between stages, or accept a request whose
num_inference_steps does not equal sum(stage_steps).

Opt-in component hooks: a conditioner with ``resolve_request(request, conditioning)``
may update the request after prompt encoding (for example a predicted frame
count); a scheduler with ``step_state(denoised, step, state, *, generator)``
returns the next latent including its own conditioning-mask projection; a decoder
with ``uses_run_generator = True`` receives the run generator.  The execution
option ``step_noise_seed_offset`` gives scheduler steps a per-stage generator
seeded ``seed + offset`` instead of the run generator.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from worldfoundry.core.execution.deadline import check_deadline

from ..contracts import (
    ConditionEncoder,
    Conditioning,
    DiffusionOutput,
    DiffusionRequest,
    DiffusionScheduler,
    LatentProcessor,
    ModalityState,
    MultiModalDenoiserInput,
    MultiModalDenoiserOutput,
    SamplingConfig,
)


class MultiStageLatentInitializer:
    """Runtime surface required by the generic multi-stage runner.

    ``initialize`` is a convenience that starts stage 0. Concrete recipes must
    implement :meth:`initialize_stage` (this base raises NotImplementedError).
    """

    def initialize(self, request, *, generator, device, dtype):
        return self.initialize_stage(
            request,
            stage_index=0,
            previous_latents={},
            noise_scale=1.0,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def initialize_stage(
        self,
        request: DiffusionRequest,
        *,
        stage_index: int,
        previous_latents: Mapping[str, Tensor],
        noise_scale: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, ModalityState]:
        raise NotImplementedError


class MultiModalLatentDecoder:
    """Runtime surface for decoders that emit more than one artifact.

    ``decode`` returns only the ``video`` artifact for DiffusionOutput.sample.
    :meth:`decode_modalities` must be implemented by the recipe decoder.  A
    decoder that sets ``uses_run_generator`` continues the run's noise generator
    and receives it as the ``generator`` keyword.
    """

    uses_run_generator: bool = False

    def decode(self, latents, request):
        return self.decode_modalities(latents, request)["video"]

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MultiStageComponents:
    """Bound roles for JointMultiStageDiffusionRunner. ``schedulers`` is one per stage."""

    denoiser: object
    conditioner: ConditionEncoder
    latent_initializer: MultiStageLatentInitializer
    schedulers: tuple[DiffusionScheduler, ...]
    processor: LatentProcessor | None
    decoder: MultiModalLatentDecoder
    stage_denoisers: tuple[object, ...] = ()


class JointMultiStageDiffusionRunner:
    """Run declaratively scheduled stages over a mapping of ModalityState values.

    Construction requires one scheduler per stage_steps entry; each stage_steps
    value must be positive. CFG is classic ``ε_neg + s*(ε_pos-ε_neg)`` applied
    independently per modality. After each intra-stage step, denoise_mask blends
    the updated latent with clean_latent so frozen regions are not updated.
    """

    def __init__(
        self,
        *,
        model_id: str,
        components: MultiStageComponents,
        stage_steps: Sequence[int],
        device: str | torch.device,
        dtype: torch.dtype,
        step_noise_seed_offset: int | None = None,
    ) -> None:
        if len(components.schedulers) != len(stage_steps):
            raise ValueError("joint multi-stage execution requires one scheduler per stage")
        if any(int(steps) <= 0 for steps in stage_steps):
            raise ValueError("joint multi-stage stage_steps must be positive")
        if components.stage_denoisers and len(components.stage_denoisers) != len(stage_steps):
            raise ValueError("joint multi-stage execution requires one denoiser per stage")
        self.model_id = str(model_id)
        self.components = components
        self.stage_steps = tuple(int(steps) for steps in stage_steps)
        self.device = torch.device(device)
        self.dtype = dtype
        self.step_noise_seed_offset = None if step_noise_seed_offset is None else int(step_noise_seed_offset)

    def _generator(self, seed: int) -> torch.Generator:
        try:
            generator = torch.Generator(device=self.device)
        except (RuntimeError, TypeError):
            generator = torch.Generator(device=self.device.type)
        return generator.manual_seed(seed)

    @staticmethod
    def _conditioning(
        conditioning: Conditioning,
        branch: str = "positive",
        *,
        request_id: str | None = None,
    ) -> Mapping[str, object]:
        values = dict(conditioning.shared)
        values.update(conditioning.positive if branch == "positive" else conditioning.negative)
        if request_id is not None:
            existing = values.get("request_id")
            if not (isinstance(existing, str) and existing.strip()):
                values["request_id"] = request_id
        return values

    def _end_denoiser_requests(
        self,
        request_id: str,
        *,
        error: BaseException | None,
    ) -> None:
        """Finalize request-owned denoiser caches without masking a run failure."""
        seen: set[int] = set()
        denoisers = [self.components.denoiser, *self.components.stage_denoisers]
        for denoiser in denoisers:
            if denoiser is None or id(denoiser) in seen:
                continue
            seen.add(id(denoiser))
            end_fn = getattr(denoiser, "end_request", None)
            if not callable(end_fn):
                end_fn = getattr(denoiser, "end_feature_cache_request", None)
            if not callable(end_fn):
                continue
            try:
                end_fn(request_id, error=error)
            except BaseException as cleanup_error:
                if error is None:
                    raise
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(f"denoiser request cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")

    def _predict(
        self,
        *,
        states: Mapping[str, ModalityState],
        conditioning: Conditioning,
        timestep: Tensor,
        step_index: int,
        total_steps: int,
        guidance_scale: float,
        stage_index: int = 0,
        request_id: str | None = None,
    ) -> MultiModalDenoiserOutput:
        """Per-modality CFG. At scale==1 or without negative, return the positive branch only."""

        denoiser = (
            self.components.stage_denoisers[stage_index]
            if self.components.stage_denoisers
            else self.components.denoiser
        )

        def call(branch: str) -> MultiModalDenoiserOutput:
            output = denoiser(
                MultiModalDenoiserInput(
                    modalities=states,
                    timestep=timestep,
                    conditioning=self._conditioning(conditioning, branch, request_id=request_id),
                    step_index=step_index,
                    total_steps=total_steps,
                    branch=branch,
                )
            )
            if not isinstance(output, MultiModalDenoiserOutput):
                raise TypeError("joint denoiser must return MultiModalDenoiserOutput")
            return output

        positive = call("positive")
        if guidance_scale == 1.0 or not conditioning.negative:
            return positive
        negative = call("negative")
        if set(positive.samples) != set(negative.samples):
            raise ValueError("positive and negative denoiser modalities must match")
        return MultiModalDenoiserOutput(
            samples={
                name: negative.samples[name] + guidance_scale * (positive.samples[name] - negative.samples[name])
                for name in positive.samples
            },
            extras={"positive": positive.extras, "negative": negative.extras},
        )

    @staticmethod
    def _sampling(request: DiffusionRequest, steps: int) -> SamplingConfig:
        return SamplingConfig(
            num_inference_steps=steps,
            guidance_scale=request.sampling.guidance_scale,
            seed=request.sampling.seed,
            scheduler_options=request.sampling.scheduler_options,
        )

    @staticmethod
    def _validate_states(states: Mapping[str, ModalityState]) -> dict[str, ModalityState]:
        result = dict(states)
        if not result:
            raise ValueError("multi-stage initializer returned no modalities")
        if not all(isinstance(value, ModalityState) for value in result.values()):
            raise TypeError("multi-stage initializer values must be ModalityState instances")
        return result

    @torch.no_grad()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        expected_steps = sum(self.stage_steps)
        if request.sampling.num_inference_steps != expected_steps:
            raise ValueError(
                "joint multi-stage request step count must match the recipe stages: "
                f"{request.sampling.num_inference_steps} != {expected_steps}"
            )
        deadline_value = request.inputs.get("deadline_monotonic")
        deadline = float(deadline_value) if deadline_value is not None else None
        check_deadline(deadline, stage="prompt conditioning")
        generator = self._generator(request.sampling.seed)
        conditioning = self.components.conditioner.encode(
            request,
            device=self.device,
            dtype=self.dtype,
        )
        if not isinstance(conditioning, Conditioning):
            raise TypeError("conditioner.encode must return Conditioning")
        check_deadline(deadline, stage="conditioned request resolution")
        resolve_request = getattr(self.components.conditioner, "resolve_request", None)
        if callable(resolve_request):
            request = resolve_request(request, conditioning)
            if not isinstance(request, DiffusionRequest):
                raise TypeError("conditioner.resolve_request must return DiffusionRequest")

        raw_request_id = request.inputs.get("request_id")
        if isinstance(raw_request_id, str) and raw_request_id.strip():
            request_id = raw_request_id.strip()
        else:
            request_id = f"{self.model_id}-{uuid.uuid4().hex}"

        previous_latents: Mapping[str, Tensor] = {}
        states: dict[str, ModalityState] = {}
        run_error: BaseException | None = None
        try:
            for stage_index, (scheduler, step_count) in enumerate(
                zip(self.components.schedulers, self.stage_steps, strict=True)
            ):
                check_deadline(deadline, stage=f"stage {stage_index + 1} initialization")
                schedule = tuple(
                    scheduler.schedule(
                        self._sampling(request, step_count),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                if len(schedule) != step_count:
                    raise ValueError(
                        f"stage {stage_index} scheduler returned {len(schedule)} steps; expected {step_count}"
                    )
                noise_scale = float(schedule[0].timestep.item())
                states = self._validate_states(
                    self.components.latent_initializer.initialize_stage(
                        request,
                        stage_index=stage_index,
                        previous_latents=previous_latents,
                        noise_scale=noise_scale,
                        generator=generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                modality_schedulers = {name: copy.deepcopy(scheduler) for name in states}
                step_generator = (
                    generator
                    if self.step_noise_seed_offset is None
                    else self._generator(request.sampling.seed + self.step_noise_seed_offset)
                )

                for expected_index, step in enumerate(schedule):
                    check_deadline(deadline, stage=f"stage {stage_index + 1} step {expected_index + 1}")
                    if step.index != expected_index:
                        raise ValueError("scheduler step indices must be contiguous and zero-based")
                    model_states = {
                        name: state.with_updates(latent=modality_schedulers[name].scale_model_input(state.latent, step))
                        for name, state in states.items()
                    }
                    prediction = self._predict(
                        states=model_states,
                        conditioning=conditioning,
                        timestep=step.timestep,
                        step_index=step.index,
                        total_steps=step_count,
                        guidance_scale=request.sampling.guidance_scale,
                        stage_index=stage_index,
                        request_id=request_id,
                    )
                    if set(prediction.samples) != set(states):
                        raise ValueError("joint denoiser output modalities must match initialized states")

                    next_states: dict[str, ModalityState] = {}
                    for name, state in states.items():
                        denoised = prediction.samples[name]
                        if denoised.shape != state.latent.shape:
                            raise ValueError(f"{name} denoiser output shape does not match its latent state")
                        step_state = getattr(modality_schedulers[name], "step_state", None)
                        if callable(step_state):
                            # The scheduler owns the conditioning-mask projection of this update.
                            latent = step_state(denoised, step, state, generator=step_generator)
                        else:
                            latent = modality_schedulers[name].step(
                                denoised,
                                step,
                                state.latent,
                                generator=step_generator,
                            )
                            # A scheduler may consume velocity, noise, or clean-sample predictions.
                            # Anchor conditioning in latent space after the solver update.
                            latent = latent * state.denoise_mask + state.clean_latent * (1 - state.denoise_mask)
                        next_states[name] = state.with_updates(latent=latent)
                    states = next_states

                if stage_index + 1 < len(self.stage_steps):
                    if self.components.processor is None:
                        raise ValueError("multi-stage execution requires a latent processor between stages")
                    check_deadline(deadline, stage=f"stage {stage_index + 1} latent processing")
                    previous_latents = self.components.processor.process(states, request)
        except BaseException as exc:
            run_error = exc
            raise
        finally:
            self._end_denoiser_requests(request_id, error=run_error)

        check_deadline(deadline, stage="media decoding")
        decoder = self.components.decoder
        if getattr(decoder, "uses_run_generator", False):
            artifacts = dict(decoder.decode_modalities(states, request, generator=generator))
        else:
            artifacts = dict(decoder.decode_modalities(states, request))
        check_deadline(deadline, stage="decoded output delivery")
        try:
            sample = artifacts["video"]
        except KeyError as error:
            raise KeyError("multi-modal decoder must return a 'video' artifact") from error
        if not isinstance(sample, Tensor):
            raise TypeError("the primary 'video' artifact must be a tensor")
        final_latents = states["video"].latent if "video" in states else next(iter(states.values())).latent
        return DiffusionOutput(
            sample=sample,
            latents=final_latents,
            artifacts=artifacts,
            metadata={
                "model_id": self.model_id,
                "seed": request.sampling.seed,
                "stage_steps": self.stage_steps,
                "execution_strategy": "joint-multistage",
            },
        )


__all__ = [
    "JointMultiStageDiffusionRunner",
    "MultiModalLatentDecoder",
    "MultiStageComponents",
    "MultiStageLatentInitializer",
]

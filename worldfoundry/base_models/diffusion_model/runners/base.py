"""Canonical diffusion runner with an explicit framework-owned sampling loop.

This is the downstream standard runner, instantiated by ``standard`` /
``frozen-context`` / ``masked-latent`` (and similar) strategies:

    recipe.execution.strategy
        → ExecutionStrategyRegistry.build_* → NativeDiffusionRunner (or a CFG variant)
        → run(DiffusionRequest) → DiffusionOutput

Inputs: RunnerComponents (five canonical roles + optional LatentEncoder),
RuntimePolicy device/dtype, DiffusionExtension values, and guidance_mode.
Output: DiffusionOutput (pixel sample, final latents, artifacts, metadata).
Must not let a model package replace the whole loop to swap loaders, or put a
factory for this class on a recipe. Subclasses should override only :meth:`predict`
to change the CFG formula while keeping the same extension lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor

from worldfoundry.core.observability.nvtx import nvtx_range

from ..contracts import (
    ConditionEncoder,
    Conditioning,
    Denoiser,
    DenoiserInput,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    DiffusionScheduler,
    EncodedLatentInitializer,
    FinalDenoiseScheduler,
    LatentDecoder,
    LatentEncoder,
    LatentInitialization,
    LatentInitializer,
)
from ..extensions import DiffusionExtension, DiffusionRunContext


@dataclass(frozen=True, slots=True)
class RunnerComponents:
    """The five canonical components required by the standard runner, plus optional LatentEncoder.

    Fields map onto contracts protocols: Denoiser / ConditionEncoder / LatentInitializer /
    DiffusionScheduler / LatentDecoder. ``latent_encoder`` is used by
    :meth:`NativeDiffusionRunner.run` only when the initializer implements EncodedLatentInitializer.
    """

    denoiser: Denoiser
    conditioner: ConditionEncoder
    latent_initializer: LatentInitializer
    scheduler: DiffusionScheduler
    decoder: LatentDecoder
    latent_encoder: LatentEncoder | None = None


class NativeDiffusionRunner:
    """Framework-owned denoising loop composed from native component contracts.

    Construction fails with :exc:`ValueError` for an empty model_id, a guidance_mode
    other than ``standard`` / ``positive``, or duplicate extension_id values.
    Neighbors: the conditioner yields Conditioning; the scheduler yields SchedulerStep;
    extension hooks must return the contracted types or raise :exc:`TypeError`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        components: RunnerComponents,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        extensions: Iterable[DiffusionExtension] = (),
        guidance_mode: str = "standard",
        cfg_parallel_degree: int = 1,
        cfg_gate_step: float = 1.0,
    ) -> None:
        self.model_id = str(model_id)
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        self.components = components
        self.device = torch.device(device)
        self.dtype = dtype
        self.extensions = tuple(extensions)
        self.guidance_mode = str(guidance_mode).strip().lower().replace("_", "-")
        if self.guidance_mode not in {"standard", "positive"}:
            raise ValueError(f"unsupported classifier-free guidance mode: {guidance_mode!r}")
        self.cfg_parallel_degree = int(cfg_parallel_degree)
        self._cfg_parallel_local_branch_calls = {"positive": 0, "negative": 0}
        self._cfg_parallel_collective_calls = 0
        if self.cfg_parallel_degree not in {1, 2}:
            raise ValueError("cfg_parallel_degree must be 1 or 2")
        self.cfg_gate_step = float(cfg_gate_step)
        if not 0.0 <= self.cfg_gate_step <= 1.0:
            raise ValueError("cfg_gate_step must be in [0.0, 1.0]")
        if self.cfg_gate_step < 1.0 and self.cfg_parallel_degree != 1:
            raise ValueError(
                "cfg_gate_step is not composable with cfg_parallel yet; "
                "the cached-delta step must prove that the negative branch was skipped"
            )
        if self.cfg_parallel_degree == 2:
            from worldfoundry.core.distributed.sequence_parallel.runtime import (
                get_cfg_parallel_world_size,
            )

            if get_cfg_parallel_world_size() != 2:
                raise RuntimeError("cfg_parallel=2 requires an active two-rank CFG group")
        self._cfg_delta_cache: Tensor | None = None
        self._cfg_gate_total_steps = 0
        self._cfg_gate_index = 0
        self._cfg_gate_delta_refreshes = 0
        self._cfg_gate_cached_delta_steps = 0
        self._cfg_gate_positive_branch_calls = 0
        self._cfg_gate_negative_branch_calls = 0
        extension_ids = [extension.extension_id for extension in self.extensions]
        if len(extension_ids) != len(set(extension_ids)):
            raise ValueError("extension_id values must be unique within one runner")

    def _generator(self, seed: int) -> torch.Generator:
        """Build a seeded Generator. Fall back to ``device.type`` when a device-bound generator is unsupported."""
        try:
            generator = torch.Generator(device=self.device)
        except (RuntimeError, TypeError):
            generator = torch.Generator(device=self.device.type)
        return generator.manual_seed(seed)

    def _is_runtime_device(self, device: torch.device) -> bool:
        """Accept the active device when the configured index is implicit (e.g. ``cuda``)."""

        if device.type != self.device.type:
            return False
        return self.device.index is None or device.index == self.device.index

    def _end_denoiser_request(
        self,
        context: DiffusionRunContext | None,
        *,
        error: BaseException | None,
    ) -> None:
        """Finalize request-owned denoiser state without masking a run failure.

        Custom execution loops subclass :class:`NativeDiffusionRunner` but do
        not call :meth:`run`.  Keeping cleanup in one helper lets every loop
        release feature-cache tensors and freeze its receipt from ``finally``.
        """

        if context is None:
            return
        end_request = getattr(self.components.denoiser, "end_request", None)
        if not callable(end_request):
            return
        try:
            end_request(context.request_id, error=error)
        except BaseException as cleanup_error:
            if error is None:
                raise
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "denoiser request cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    @staticmethod
    def _branch_conditioning(
        conditioning: Conditioning,
        branch: str,
    ) -> Mapping[str, object]:
        """Merge shared + positive/negative into one branch's denoiser condition mapping."""
        values: dict[str, object] = dict(conditioning.shared)
        values.update(conditioning.positive if branch == "positive" else conditioning.negative)
        return values

    def _call_denoiser(
        self,
        context: DiffusionRunContext,
        *,
        latents: Tensor,
        branch: str,
    ) -> DenoiserOutput:
        """Call the denoiser with conditions taken from context.conditioning for ``branch``."""
        return self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch=branch,
            conditioning=self._branch_conditioning(context.conditioning, branch),
        )

    def _call_denoiser_with_conditioning(
        self,
        context: DiffusionRunContext,
        *,
        latents: Tensor,
        branch: str,
        conditioning: Mapping[str, object],
    ) -> DenoiserOutput:
        """Evaluate one CFG branch with an explicitly composed condition mapping.

        ``context.step`` must already be selected, else :exc:`RuntimeError`.
        before_denoiser / after_denoiser must return DenoiserInput / DenoiserOutput.
        The output sample shape must match the input latents, else :exc:`ValueError`.
        """

        step = context.step
        if step is None:
            raise RuntimeError("denoiser called before a scheduler step was selected")
        model_input = DenoiserInput(
            latents=latents,
            timestep=step.timestep,
            next_timestep=step.next_timestep,
            conditioning=conditioning,
            step_index=step.index,
            total_steps=context.total_steps or context.request.sampling.num_inference_steps,
            branch=branch,
            request_id=context.request_id,
        )
        for extension in self.extensions:
            model_input = extension.before_denoiser(context, model_input)
            if not isinstance(model_input, DenoiserInput):
                raise TypeError(
                    f"{extension.extension_id}.before_denoiser must return "
                    f"DenoiserInput, got {type(model_input).__name__}"
                )
        with nvtx_range(f"worldfoundry.denoiser.{branch}"):
            model_output = self.components.denoiser(model_input)
        if not isinstance(model_output, DenoiserOutput):
            raise TypeError(f"denoiser must return DenoiserOutput, got {type(model_output).__name__}")
        for extension in reversed(self.extensions):
            model_output = extension.after_denoiser(context, model_output)
            if not isinstance(model_output, DenoiserOutput):
                raise TypeError(
                    f"{extension.extension_id}.after_denoiser must return "
                    f"DenoiserOutput, got {type(model_output).__name__}"
                )
        if model_output.sample.shape != latents.shape:
            raise ValueError(
                "denoiser output shape must match latent shape: "
                f"{tuple(model_output.sample.shape)} != {tuple(latents.shape)}"
            )
        return model_output

    def _parallel_cfg_pair(
        self,
        context: DiffusionRunContext,
        latents: Tensor,
    ) -> tuple[DenoiserOutput, DenoiserOutput]:
        """Evaluate positive/negative branches on separate CFG replicas."""

        import torch.distributed as dist

        from worldfoundry.core.distributed.sequence_parallel.runtime import (
            get_cfg_parallel_group,
            get_cfg_parallel_rank,
        )

        group = get_cfg_parallel_group()
        if group is None or not dist.is_initialized():
            raise RuntimeError("CFG parallel group disappeared during inference")
        cfg_rank = get_cfg_parallel_rank()
        branch = "positive" if cfg_rank == 0 else "negative"
        self._cfg_parallel_local_branch_calls[branch] += 1
        conditioning = dict(self._branch_conditioning(context.conditioning, branch))
        conditioning["_worldfoundry_cfg_parallel_request"] = True
        local = self._call_denoiser_with_conditioning(
            context,
            latents=latents,
            branch=branch,
            conditioning=conditioning,
        )
        samples = [torch.empty_like(local.sample) for _ in range(2)]
        dist.all_gather(samples, local.sample.contiguous(), group=group)
        self._cfg_parallel_collective_calls += 1
        return (
            DenoiserOutput(
                sample=samples[0],
                extras={"cfg_parallel_rank": 0, "local": cfg_rank == 0},
            ),
            DenoiserOutput(
                sample=samples[1],
                extras={"cfg_parallel_rank": 1, "local": cfg_rank == 1},
            ),
        )

    def _parallel_optimization_report(
        self,
        *,
        branch_calls_before: Mapping[str, int] | None = None,
        collectives_before: int = 0,
    ) -> dict[str, object]:
        """Report requested and exercised CFG parallelism for one request window."""

        before = branch_calls_before or {"positive": 0, "negative": 0}
        branch_calls = {
            name: self._cfg_parallel_local_branch_calls[name] - int(before.get(name, 0))
            for name in ("positive", "negative")
        }
        collective_calls = self._cfg_parallel_collective_calls - int(collectives_before)
        requested = self.cfg_parallel_degree == 2
        effective = (
            "branch-per-rank"
            if collective_calls > 0
            else "exact-local"
            if not requested
            else "exact-local (two-branch CFG was not exercised)"
        )
        fallbacks = (
            []
            if not requested or collective_calls > 0
            else [
                "cfg_parallel: requested degree 2 but this request did not exercise "
                "two-branch classifier-free guidance"
            ]
        )
        return {
            "requested": {
                "cfg_parallel": self.cfg_parallel_degree if requested else False
            },
            "effective": {"cfg_parallel": effective},
            "fallbacks": fallbacks,
            "quality_tier": "exact",
            "runtime": {
                "cfg_parallel_local_branch_calls": branch_calls,
                "cfg_parallel_collective_calls": collective_calls,
            },
        }

    def _reset_cfg_gate_request(self, total_steps: int) -> None:
        """Clear stale-unconditional state at the start of every request."""

        self._cfg_delta_cache = None
        self._cfg_gate_total_steps = int(total_steps)
        self._cfg_gate_index = int(self._cfg_gate_total_steps * self.cfg_gate_step)
        self._cfg_gate_delta_refreshes = 0
        self._cfg_gate_cached_delta_steps = 0
        self._cfg_gate_positive_branch_calls = 0
        self._cfg_gate_negative_branch_calls = 0

    def cfg_gate_optimization_report(self) -> dict[str, object]:
        """Return request-window proof for FastVideo-style stale-uncond reuse.

        Merely configuring a fraction is not effective.  A request proves the
        acceleration only after at least one guided step executes the positive
        branch without executing the negative branch and consumes a delta that
        was refreshed by an earlier dense pair.
        """

        requested = self.cfg_gate_step < 1.0
        effective = requested and self._cfg_gate_cached_delta_steps > 0
        fallbacks = []
        if requested and not effective:
            fallbacks.append(
                "cfg_delta_cache: requested but this request did not execute a "
                "positive-only guided step with a cached cond-uncond delta"
            )
        return {
            "requested": {
                "cfg_delta_cache": self.cfg_gate_step if requested else False,
            },
            "effective": {
                "cfg_delta_cache": (
                    "stale-uncond-delta-reuse"
                    if effective
                    else "not-requested"
                    if not requested
                    else "requested-not-exercised"
                ),
            },
            "fallbacks": fallbacks,
            "quality_tier": "approximate" if requested else "exact",
            "runtime": {
                "cfg_delta_cache": {
                    "gate_fraction": self.cfg_gate_step,
                    "total_steps": self._cfg_gate_total_steps,
                    "gate_step_index": self._cfg_gate_index,
                    "delta_refreshes": self._cfg_gate_delta_refreshes,
                    "cached_delta_steps": self._cfg_gate_cached_delta_steps,
                    "positive_branch_calls": self._cfg_gate_positive_branch_calls,
                    "negative_branch_calls": self._cfg_gate_negative_branch_calls,
                }
            },
        }

    def _guided_output(
        self,
        *,
        positive: DenoiserOutput,
        negative: DenoiserOutput,
        scale: float,
    ) -> DenoiserOutput:
        """Combine one freshly evaluated positive/negative CFG pair."""

        delta = positive.sample - negative.sample
        if self.cfg_gate_step < 1.0:
            self._cfg_delta_cache = delta.detach()
            self._cfg_gate_delta_refreshes += 1
        if self.guidance_mode == "positive":
            guided = positive.sample + scale * delta
        else:
            guided = negative.sample + scale * delta
        return DenoiserOutput(
            sample=guided,
            extras={"positive": positive.extras, "negative": negative.extras},
        )

    def predict(
        self,
        context: DiffusionRunContext,
        model_latents: Tensor,
    ) -> DenoiserOutput:
        """Predict one guided update.

        Families with embedded guidance or multi-stage denoisers can override this
        method while keeping the same lifecycle and extension contracts.
        CFG formulas are documented on the branches below.
        """

        scale = context.request.sampling.guidance_scale
        if not context.conditioning.negative:
            return self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
        # Standard CFG: uncond + s*(cond-uncond). At s==1 this equals positive, so skip the negative pass.
        if self.guidance_mode == "standard" and scale == 1.0:
            return self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
        # Positive mode: cond + s*(cond-uncond). At s==0 this equals positive.
        if self.guidance_mode == "positive" and scale == 0.0:
            return self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
        if self.cfg_parallel_degree == 2:
            positive, negative = self._parallel_cfg_pair(context, model_latents)
        else:
            positive = self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
            self._cfg_gate_positive_branch_calls += 1
            step = context.step
            if step is None:
                raise RuntimeError("CFG prediction requested before scheduler step selection")
            use_cached_delta = (
                self.cfg_gate_step < 1.0
                and int(step.index) >= self._cfg_gate_index
                and self._cfg_delta_cache is not None
            )
            if use_cached_delta:
                self._cfg_gate_cached_delta_steps += 1
                delta = self._cfg_delta_cache.to(
                    device=positive.sample.device,
                    dtype=positive.sample.dtype,
                )
                if self.guidance_mode == "positive":
                    guided = positive.sample + scale * delta
                else:
                    # neg + s*(pos-neg) == pos + (s-1)*(pos-neg).
                    guided = positive.sample + (scale - 1.0) * delta
                return DenoiserOutput(
                    sample=guided,
                    extras={
                        "positive": positive.extras,
                        "negative": {"cached_delta": True},
                        "cfg_delta_cache": {
                            "gate_step_index": self._cfg_gate_index,
                            "source": "previous-dense-pair",
                        },
                    },
                )
            negative = self._call_denoiser(
                context,
                latents=model_latents,
                branch="negative",
            )
            self._cfg_gate_negative_branch_calls += 1
        return self._guided_output(positive=positive, negative=negative, scale=scale)

    @torch.no_grad()
    @nvtx_range("worldfoundry.run")
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        """Execute one native diffusion request: encode → init → schedule → CFG steps → decode.

        On failure, extension.on_run_error is called in reverse order, then the error is re-raised.
        Typical failures: wrong return type from conditioner / decoder / scheduler.step →
        :exc:`TypeError`; unexpected step count or non-contiguous step.index → :exc:`ValueError`;
        latent device mismatch → :exc:`ValueError`.
        """

        cfg_branch_calls_before = dict(self._cfg_parallel_local_branch_calls)
        cfg_collectives_before = self._cfg_parallel_collective_calls
        self._reset_cfg_gate_request(request.sampling.num_inference_steps)
        generator = self._generator(request.sampling.seed)
        with nvtx_range("worldfoundry.encode"):
            conditioning = self.components.conditioner.encode(
                request,
                device=self.device,
                dtype=self.dtype,
            )
        if not isinstance(conditioning, Conditioning):
            raise TypeError(f"conditioner.encode must return Conditioning, got {type(conditioning).__name__}")
        context = DiffusionRunContext(
            request=request,
            components=self.components,
            conditioning=conditioning,
            generator=generator,
        )

        run_error: BaseException | None = None
        try:
            for extension in self.extensions:
                extension.on_run_start(context)
            for extension in self.extensions:
                context.conditioning = extension.prepare_conditioning(
                    context,
                    context.conditioning,
                )
                if not isinstance(context.conditioning, Conditioning):
                    raise TypeError(
                        f"{extension.extension_id}.prepare_conditioning must return "
                        f"Conditioning, got {type(context.conditioning).__name__}"
                    )

            # Use encode-then-initialize only when both an encoder and EncodedLatentInitializer
            # are present; otherwise fall back to plain initialize so an optional codec role
            # cannot be misused.
            if self.components.latent_encoder is not None and isinstance(
                self.components.latent_initializer, EncodedLatentInitializer
            ):
                initialization = self.components.latent_initializer.initialize_with_encoder(
                    request,
                    latent_encoder=self.components.latent_encoder,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                initialization = self.components.latent_initializer.initialize(
                    request,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            initialization_artifacts: Mapping[str, object] = {}
            if isinstance(initialization, LatentInitialization):
                overlap = sorted(set(context.conditioning.shared) & set(initialization.conditioning))
                if overlap:
                    raise ValueError(f"latent initialization conditions overlap existing shared values: {overlap}")
                shared = dict(context.conditioning.shared)
                shared.update(initialization.conditioning)
                context.conditioning = Conditioning(
                    positive=context.conditioning.positive,
                    negative=context.conditioning.negative,
                    shared=shared,
                )
                latents = initialization.latents
                initialization_artifacts = initialization.artifacts
            else:
                latents = initialization
            if not isinstance(latents, Tensor):
                raise TypeError("latent_initializer.initialize must return a tensor or LatentInitialization")
            if not self._is_runtime_device(latents.device):
                raise ValueError(f"latent initializer returned {latents.device}, expected {self.device}")

            schedule = tuple(
                self.components.scheduler.schedule(
                    request.sampling,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            expected_step_count = getattr(self.components.scheduler, "expected_step_count", None)
            expected_steps = (
                expected_step_count(request.sampling)
                if callable(expected_step_count)
                else request.sampling.num_inference_steps
            )
            if len(schedule) != expected_steps:
                raise ValueError(
                    "scheduler returned an unexpected number of steps: "
                    f"{len(schedule)} != {expected_steps}"
                )
            if expected_steps != request.sampling.num_inference_steps:
                self._reset_cfg_gate_request(expected_steps)
            context.total_steps = len(schedule)
            for expected_index, step in enumerate(schedule):
                if step.index != expected_index:
                    raise ValueError(
                        "scheduler step indices must be contiguous and zero-based: "
                        f"got {step.index} at position {expected_index}"
                    )

            for step in schedule:
                with nvtx_range("worldfoundry.denoise_step"):
                    context.step = step
                    model_latents = self.components.scheduler.scale_model_input(
                        latents,
                        step,
                    )
                    prediction = self.predict(context, model_latents)
                    latents = self.components.scheduler.step(
                        prediction.sample,
                        step,
                        latents,
                        generator=generator,
                    )
                    if not isinstance(latents, Tensor):
                        raise TypeError("scheduler.step must return a tensor")
                    for extension in self.extensions:
                        latents = extension.after_step(context, latents)
                        if not isinstance(latents, Tensor):
                            raise TypeError(
                                f"{extension.extension_id}.after_step must return a tensor, got {type(latents).__name__}"
                            )

            # Final denoise: some schedulers (FinalDenoiseScheduler) request one extra
            # clean prediction after the regular steps. Use predict.sample as the final
            # latents and skip scheduler.step so no extra noise is mixed in.
            final_denoise = False
            if isinstance(self.components.scheduler, FinalDenoiseScheduler):
                final_step = self.components.scheduler.final_denoise_step()
                if final_step is not None:
                    final_denoise = True
                    context.step = final_step
                    model_latents = self.components.scheduler.scale_model_input(
                        latents,
                        final_step,
                    )
                    prediction = self.predict(context, model_latents)
                    latents = prediction.sample
                    for extension in self.extensions:
                        latents = extension.after_step(context, latents)
                        if not isinstance(latents, Tensor):
                            raise TypeError(
                                f"{extension.extension_id}.after_step must return a tensor, "
                                f"got {type(latents).__name__}"
                            )

            with nvtx_range("worldfoundry.decode"):
                sample = self.components.decoder.decode(latents, request)
            if not isinstance(sample, Tensor):
                raise TypeError("decoder.decode must return a tensor")
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
                if not isinstance(sample, Tensor):
                    raise TypeError(
                        f"{extension.extension_id}.after_decode must return a tensor, got {type(sample).__name__}"
                    )
            output = DiffusionOutput(
                sample=sample,
                latents=latents,
                artifacts=initialization_artifacts,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "num_inference_steps": request.sampling.num_inference_steps,
                    "model_evaluations": len(schedule),
                    "guidance_scale": request.sampling.guidance_scale,
                    "guidance_mode": self.guidance_mode,
                    "cfg_parallel_degree": self.cfg_parallel_degree,
                    "parallel_optimization_report": self._parallel_optimization_report(
                        branch_calls_before=cfg_branch_calls_before,
                        collectives_before=cfg_collectives_before,
                    ),
                    "cfg_gate_optimization_report": self.cfg_gate_optimization_report(),
                    "final_denoise": final_denoise,
                    "extensions": [extension.extension_id for extension in self.extensions],
                },
            )
            for extension in reversed(self.extensions):
                extension.on_run_end(context)
            return output
        except BaseException as error:
            run_error = error
            for extension in reversed(self.extensions):
                extension.on_run_error(context, error)
            raise
        finally:
            self._end_denoiser_request(context, error=run_error)


class DualConditionGuidanceRunner(NativeDiffusionRunner):
    """Framework-owned three-branch CFG for text plus a droppable secondary condition (e.g. image).

    Branches: positive (full text + secondary), negative (no text + secondary),
    unconditional (no text and ``drop_secondary_condition=True``).
    Extra forwards are skipped when ``text_scale==1`` or there is no negative condition.
    """

    def __init__(
        self,
        *,
        secondary_guidance_scale: float = 1.0,
        secondary_guidance_input: str = "secondary_guidance_scale",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.secondary_guidance_scale = float(secondary_guidance_scale)
        self.secondary_guidance_input = str(secondary_guidance_input)

    def predict(
        self,
        context: DiffusionRunContext,
        model_latents: Tensor,
    ) -> DenoiserOutput:
        positive = self._call_denoiser(context, latents=model_latents, branch="positive")
        text_scale = context.request.sampling.guidance_scale
        if text_scale == 1.0 or not context.conditioning.negative:
            return positive

        negative = self._call_denoiser(context, latents=model_latents, branch="negative")
        # Third branch: drop the secondary condition on top of negative to get a true uncond prediction.
        unconditional_values = dict(self._branch_conditioning(context.conditioning, "negative"))
        unconditional_values["drop_secondary_condition"] = True
        unconditional = self._call_denoiser_with_conditioning(
            context,
            latents=model_latents,
            branch="unconditional",
            conditioning=unconditional_values,
        )
        secondary_scale = float(
            context.request.inputs.get(
                self.secondary_guidance_input,
                self.secondary_guidance_scale,
            )
        )
        # ε = ε_uncond + s_sec*(ε_neg - ε_uncond) + s_text*(ε_pos - ε_neg)
        guided = (
            unconditional.sample
            + secondary_scale * (negative.sample - unconditional.sample)
            + text_scale * (positive.sample - negative.sample)
        )
        return DenoiserOutput(
            sample=guided,
            extras={
                "positive": positive.extras,
                "negative": negative.extras,
                "unconditional": unconditional.extras,
            },
        )


class Wan22DualExpertGuidanceRunner(NativeDiffusionRunner):
    """Apply Wan2.2 A14B's released low/high expert CFG schedules.

    ``boundary_ratio`` must be in (0, 1); ``num_train_timesteps`` must be positive;
    both guidance scales must be non-negative. Otherwise construction raises :exc:`ValueError`.
    The live scale is chosen by comparing timestep / num_train_timesteps against the boundary.
    Request inputs may override high_noise_guidance_scale / low_noise_guidance_scale.
    """

    def __init__(
        self,
        *,
        boundary_ratio: float,
        low_noise_guidance_scale: float,
        high_noise_guidance_scale: float,
        num_train_timesteps: int = 1000,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.boundary_ratio = float(boundary_ratio)
        self.low_noise_guidance_scale = float(low_noise_guidance_scale)
        self.high_noise_guidance_scale = float(high_noise_guidance_scale)
        self.num_train_timesteps = int(num_train_timesteps)
        if not 0.0 < self.boundary_ratio < 1.0:
            raise ValueError("Wan2.2 A14B boundary_ratio must be in (0, 1)")
        if self.num_train_timesteps <= 0:
            raise ValueError("Wan2.2 A14B num_train_timesteps must be positive")
        if min(self.low_noise_guidance_scale, self.high_noise_guidance_scale) < 0:
            raise ValueError("Wan2.2 A14B guidance scales must be non-negative")

    def _guidance_scale(self, context: DiffusionRunContext) -> float:
        """Normalize the current timestep to a [0, 1] sigma, then pick the expert scale via boundary_ratio."""
        if context.step is None:
            raise RuntimeError("Wan2.2 A14B guidance requested before scheduler step selection")
        sigma = float(context.step.timestep.detach().float().reshape(-1)[0].cpu())
        sigma /= float(self.num_train_timesteps)
        if not torch.isfinite(torch.tensor(sigma)):
            raise ValueError("Wan2.2 A14B scheduler timestep must be finite")
        if sigma >= self.boundary_ratio:
            return float(
                context.request.inputs.get(
                    "high_noise_guidance_scale",
                    self.high_noise_guidance_scale,
                )
            )
        return float(
            context.request.inputs.get(
                "low_noise_guidance_scale",
                self.low_noise_guidance_scale,
            )
        )

    def predict(
        self,
        context: DiffusionRunContext,
        model_latents: Tensor,
    ) -> DenoiserOutput:
        if not context.conditioning.negative:
            return self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
        scale = self._guidance_scale(context)
        if scale == 1.0:
            return self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
        if self.cfg_parallel_degree == 2:
            positive, negative = self._parallel_cfg_pair(context, model_latents)
        else:
            positive = self._call_denoiser(
                context,
                latents=model_latents,
                branch="positive",
            )
            negative = self._call_denoiser(
                context,
                latents=model_latents,
                branch="negative",
            )
        return DenoiserOutput(
            sample=negative.sample + scale * (positive.sample - negative.sample),
            extras={
                "positive": positive.extras,
                "negative": negative.extras,
                "guidance_scale": scale,
            },
        )


__all__ = [
    "DualConditionGuidanceRunner",
    "NativeDiffusionRunner",
    "RunnerComponents",
    "Wan22DualExpertGuidanceRunner",
]

"""Typed contracts shared by canonical diffusion models and runners.

These dataclasses and Protocols are the only boundary between the runner and
model components.

Design rules
------------
- **Immutable.** Requests, conditions, and step state are frozen dataclasses.
  Extensions and models must not smuggle implicit state by mutating a shared
  dict in place.
- **Explicit.** Images, camera trajectories, actions, and memory references
  live on :class:`DiffusionRequest.inputs`, not in process-wide config.
- **Protocols, not base classes.** A denoiser / encoder / scheduler only has
  to satisfy the Protocol; it does not inherit a framework type.

Data flow for one ``runner.run``::

    DiffusionRequest
        → ConditionEncoder.encode        → Conditioning
        → LatentInitializer.initialize   → Tensor | LatentInitialization
        → DiffusionScheduler.schedule    → Sequence[SchedulerStep]
        → Denoiser(DenoiserInput) / step → DenoiserOutput, then updated latents
        → LatentDecoder.decode           → DiffusionOutput
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor


def _frozen_mapping(value: Mapping | None) -> Mapping:
    """Copy a mapping into a read-only ``MappingProxyType`` so callers cannot mutate shared state."""
    return MappingProxyType(dict(value or {}))


def _string_tuple(value: str | Sequence[str], *, field_name: str) -> tuple[str, ...]:
    """Normalize a single string or a sequence of strings into a non-empty ``tuple[str, ...]``."""
    items = (value,) if isinstance(value, str) else tuple(str(item) for item in value)
    if not items:
        raise ValueError(f"{field_name} cannot be empty")
    return items


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Model-independent settings for one denoising run.

    Attributes:
        num_inference_steps: Requested inference steps. Schedulers normally
            return exactly this many updates; model-specific interval-based
            samplers may declare one extra evaluation point.
        guidance_scale: Classifier-free guidance strength.  In ``standard``
            mode, ``1.0`` skips the negative branch.  In ``positive`` mode,
            ``0.0`` skips it.
        seed: Seed for the ``torch.Generator`` that produces reproducible noise.
        scheduler_options: Extra scheduler knobs (shift, sigma range, …).
            Frozen after construction.
    """

    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 0
    scheduler_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        # Frozen dataclasses can only assign normalized fields via object.__setattr__.
        object.__setattr__(
            self,
            "scheduler_options",
            _frozen_mapping(self.scheduler_options),
        )


@dataclass(frozen=True, slots=True)
class DiffusionRequest:
    """Normalized request consumed by the native runner.

    Model-specific conditions such as images, camera trajectories, actions, or
    memory references live in ``inputs``.  They remain explicit values rather
    than process-wide mutable configuration.  ``metadata`` is audit-only and
    is not used in the numerical loop.

    Attributes:
        prompt: One or more positive texts.
        negative_prompt: Optional negative texts.  A single item is broadcast
            across the batch.
        height / width / num_frames: Pixel-space geometry used by the
            initializer and decoder to infer latent shapes.
        sampling: Step count, CFG, and seed.
        inputs: Family-specific tensors or paths (first frame, reference
            video, camera, …).
        metadata: Caller audit fields copied into :class:`DiffusionOutput`.
    """

    prompt: str | Sequence[str]
    negative_prompt: str | Sequence[str] | None = None
    height: int = 512
    width: int = 512
    num_frames: int = 1
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    inputs: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prompts = _string_tuple(self.prompt, field_name="prompt")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.negative_prompt is not None:
            negative = _string_tuple(
                self.negative_prompt,
                field_name="negative_prompt",
            )
            # A single negative prompt is shared by the whole batch; otherwise
            # it must align one-to-one with the positive prompts.
            if len(negative) not in (1, len(prompts)):
                raise ValueError("negative_prompt must contain one item or match the prompt batch")
        object.__setattr__(self, "inputs", _frozen_mapping(self.inputs))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def prompts(self) -> tuple[str, ...]:
        """Normalized positive prompt tuple."""
        return _string_tuple(self.prompt, field_name="prompt")

    @property
    def negative_prompts(self) -> tuple[str, ...] | None:
        """Normalized negative prompts; a singleton is broadcast to the batch."""
        if self.negative_prompt is None:
            return None
        values = _string_tuple(self.negative_prompt, field_name="negative_prompt")
        if len(values) == 1 and len(self.prompts) > 1:
            return values * len(self.prompts)
        return values

    @property
    def batch_size(self) -> int:
        """Batch size equals the number of positive prompts."""
        return len(self.prompts)


@dataclass(frozen=True, slots=True)
class Conditioning:
    """Positive, negative, and branch-independent denoiser conditions.

    Before each denoiser call the runner merges ``shared`` with the active
    branch: ``{**shared, **positive}`` or ``{**shared, **negative}``.
    An empty ``negative`` mapping means CFG is off.
    """

    positive: Mapping[str, object]
    negative: Mapping[str, object] = field(default_factory=dict)
    shared: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive", _frozen_mapping(self.positive))
        object.__setattr__(self, "negative", _frozen_mapping(self.negative))
        object.__setattr__(self, "shared", _frozen_mapping(self.shared))


@dataclass(frozen=True, slots=True)
class LatentInitialization:
    """Initial noise plus run-local conditions produced during initialization.

    Image/video encoders often belong to the latent codec rather than the text
    conditioner.  Returning their results here keeps that state explicit and
    lets the framework merge ``conditioning`` into immutable
    :class:`Conditioning.shared` before the first denoiser call.
    ``artifacts`` are copied into the final :class:`DiffusionOutput`
    (for example encoded reference frames).
    """

    latents: Tensor
    conditioning: Mapping[str, object] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.latents, Tensor):
            raise TypeError("LatentInitialization.latents must be a tensor")
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning))
        object.__setattr__(self, "artifacts", _frozen_mapping(self.artifacts))


@dataclass(frozen=True, slots=True)
class SchedulerStep:
    """One immutable point in a scheduler-owned denoising trajectory.

    ``index`` must be contiguous and zero-based; the runner checks this.
    ``timestep`` / ``next_timestep`` are the noise levels for this interval
    (discrete ``t`` or continuous sigma).  The scheduler owns the semantics;
    the denoiser only consumes the tensors.
    """

    index: int
    timestep: Tensor
    next_timestep: Tensor


@dataclass(frozen=True, slots=True)
class DenoiserInput:
    """Complete input for one conditional denoiser evaluation.

    ``request_id`` owns one inference lifecycle and ``branch`` distinguishes
    its CFG arms (``positive`` / ``negative`` / ``unconditional``). Stateful
    caches such as TeaCache key by both values so interleaved requests cannot
    exchange residuals. ``with_updates`` is the immutable replacement used by
    ``before_denoiser`` extensions.
    """

    latents: Tensor
    timestep: Tensor
    next_timestep: Tensor
    conditioning: Mapping[str, object]
    step_index: int
    total_steps: int
    branch: str = "positive"
    request_id: str | None = None

    def with_updates(self, **changes: object) -> "DenoiserInput":
        """Return a new instance with selected fields replaced (frozen dataclass)."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class DenoiserOutput:
    """Noise, velocity, or flow prediction returned by a denoiser.

    ``sample`` must match the input latent shape; the runner checks this.
    ``extras`` holds non-primary tensors such as attention maps.
    """

    sample: Tensor
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _frozen_mapping(self.extras))

    def with_updates(self, **changes: object) -> "DenoiserOutput":
        """Immutable update used by ``after_denoiser`` extensions."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class DiffusionOutput:
    """Decoded output plus final latent and reproducibility metadata.

    ``sample`` is the decoder output.  ``latents`` is the trajectory at the
    end of denoising, kept for a second decode or for debugging.
    ``metadata`` records ``model_id``, seed, step count, and installed
    extensions.
    """

    sample: Tensor
    latents: Tensor
    artifacts: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _frozen_mapping(self.artifacts))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def with_metadata(self, **values: object) -> "DiffusionOutput":
        """Merge extra metadata and return a new instance."""
        metadata = dict(self.metadata)
        metadata.update(values)
        return replace(self, metadata=metadata)


@runtime_checkable
class Denoiser(Protocol):
    """A trainable network that predicts one denoising update (noise / v / flow)."""

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        """Evaluate one positive, negative, or unconditional branch."""


@runtime_checkable
class ConditionEncoder(Protocol):
    """Encode prompts and model-specific inputs into :class:`Conditioning`."""

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return positive / negative / shared conditions for one request."""


@runtime_checkable
class LatentInitializer(Protocol):
    """Create initial noise or image/video-conditioned latents."""

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | LatentInitialization:
        """Return the initial latent tensor, or a :class:`LatentInitialization`."""


@runtime_checkable
class EncodedLatentInitializer(Protocol):
    """Initializer that consumes a recipe-bound shared :class:`LatentEncoder`.

    When the recipe also binds ``latent_encoder`` and the initializer implements
    this protocol, the runner calls ``initialize_with_encoder`` so each
    initializer does not load its own VAE.
    """

    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: "LatentEncoder",
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | LatentInitialization:
        """Encode request media and create the initial latent trajectory."""


@runtime_checkable
class LatentEncoder(Protocol):
    """Encode pixels into a model's latent representation (usually a VAE encoder)."""

    def encode(self, images: Tensor) -> Tensor:
        """Encode a normalized image or video batch."""


@runtime_checkable
class LatentDecoder(Protocol):
    """Decode final latents into pixels or another generated representation."""

    def decode(self, latents: Tensor, request: DiffusionRequest) -> Tensor:
        """Decode a completed latent trajectory.  ``request`` supplies geometry."""


@runtime_checkable
class DiffusionScheduler(Protocol):
    """Own the immutable denoising schedule and numerical update rule.

    ``schedule`` must not mutate global state.  Each run builds a fresh step
    list from :class:`SamplingConfig`.
    """

    def schedule(
        self,
        sampling: SamplingConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Sequence[SchedulerStep]:
        """Build a schedule whose length equals ``num_inference_steps``."""

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
        """Scale latents before a denoiser evaluation (for example DDPM ``1/sqrt(alpha)``)."""

    def step(
        self,
        model_output: Tensor,
        step: SchedulerStep,
        latents: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Advance the latent trajectory by one numerical step."""


@runtime_checkable
class FinalDenoiseScheduler(Protocol):
    """Optional scheduler capability for a terminal clean prediction.

    Some native samplers advance through every interval in their numerical
    schedule and then evaluate the denoiser once more at the last non-zero
    noise level, treating that prediction as the final latent.  Keeping that
    request on the scheduler lets the shared runner implement the lifecycle
    without a model-name branch.
    """

    def final_denoise_step(self) -> SchedulerStep | None:
        """Return the terminal prediction point, or ``None`` when unused."""


@dataclass(frozen=True, slots=True)
class ModalityState:
    """One modality's latent, immutable conditioning mask, and positions.

    ``denoise_mask`` marks the region updated on this step.  ``clean_latent``
    is the frozen / conditioned reference; a masked-latent strategy projects
    frozen regions back after every step.
    """

    latent: Tensor
    denoise_mask: Tensor
    positions: Tensor
    clean_latent: Tensor
    attention_mask: Tensor | None = None

    def with_updates(self, **changes: object) -> "ModalityState":
        """Immutable field replacement."""
        return replace(self, **changes)

    def clone(self) -> "ModalityState":
        """Deep-copy every tensor so stages do not share storage."""
        return ModalityState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
        )


@dataclass(frozen=True, slots=True)
class MultiModalDenoiserInput:
    """Joint state passed to a denoiser that couples several modalities."""

    modalities: Mapping[str, ModalityState]
    timestep: Tensor
    conditioning: Mapping[str, object]
    step_index: int
    total_steps: int
    branch: str = "positive"

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", _frozen_mapping(self.modalities))
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning))


@dataclass(frozen=True, slots=True)
class MultiModalDenoiserOutput:
    """Denoised predictions keyed by modality name."""

    samples: Mapping[str, Tensor]
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _frozen_mapping(self.samples))
        object.__setattr__(self, "extras", _frozen_mapping(self.extras))


@runtime_checkable
class LatentProcessor(Protocol):
    """Transform completed modality states between framework-owned stages.

    The joint-multistage runner calls this before the next stage so each
    modality is packed into the dense latents that stage's initializer needs.
    """

    def process(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, Tensor]:
        """Return dense modality latents used to initialize the next stage."""


__all__ = [
    "ConditionEncoder",
    "Conditioning",
    "Denoiser",
    "DenoiserInput",
    "DenoiserOutput",
    "DiffusionOutput",
    "DiffusionRequest",
    "DiffusionScheduler",
    "EncodedLatentInitializer",
    "FinalDenoiseScheduler",
    "LatentDecoder",
    "LatentEncoder",
    "LatentInitialization",
    "LatentInitializer",
    "LatentProcessor",
    "ModalityState",
    "MultiModalDenoiserInput",
    "MultiModalDenoiserOutput",
    "SamplingConfig",
    "SchedulerStep",
]

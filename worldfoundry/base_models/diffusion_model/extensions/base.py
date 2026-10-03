"""Composable research extensions for the canonical diffusion loop.

The runner owns the sampling loop in :mod:`..contracts` types.  An extension
is a per-instance hook object: it must return the same contract types it
receives (``Conditioning``, ``DenoiserInput``, ``DenoiserOutput``, or a latent
tensor).  A wrong return type is a :exc:`TypeError` at the runner, not here.

Hooks are no-ops by default so a recipe can install a mix of research
extensions without every class implementing every seam.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    SchedulerStep,
)

if TYPE_CHECKING:
    from ..runners import RunnerComponents


@dataclass(slots=True)
class DiffusionRunContext:
    """Per-run state visible to explicitly installed extensions.

    Attributes:
        request: Frozen :class:`~..contracts.DiffusionRequest` for this run.
        components: The five runner roles (denoiser, conditioner, …).
        conditioning: Frozen :class:`~..contracts.Conditioning` after
            ``prepare_conditioning``.
        generator: Seeded ``torch.Generator`` shared with the scheduler.
        state: Mutable per-run bag owned by extensions (not by the recipe).
        step: Current :class:`~..contracts.SchedulerStep`, or ``None``
            outside the denoise loop.
        final_latents: Borrowed final tensor, available only while completion
            hooks run. Observe it without in-place writes; use a readiness
            event before consuming it from a different CUDA stream.
    """

    request: DiffusionRequest
    components: "RunnerComponents"
    conditioning: Conditioning
    generator: torch.Generator
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: dict[str, object] = field(default_factory=dict)
    step: SchedulerStep | None = None
    total_steps: int | None = None
    final_latents: Tensor | None = None


class DiffusionExtension:
    """No-op base class for memory, control, and research interventions.

    Extensions are installed on one runner instance.  They do not mutate class
    definitions or process-wide registries, so multiple research variants can
    coexist safely in the same Python process.
    """

    extension_id = "extension"

    def on_run_start(self, context: DiffusionRunContext) -> None:
        """Initialize extension-owned state.

        Called once after conditioning is encoded and before the first
        scheduler step.  Failures here abort the run (no sample is produced).
        """

    def prepare_conditioning(
        self,
        context: DiffusionRunContext,
        conditioning: Conditioning,
    ) -> Conditioning:
        """Add or transform conditions once before sampling.

        Must return a :class:`~..contracts.Conditioning`.  The runner raises
        :exc:`TypeError` on any other type.
        """

        return conditioning

    def before_denoiser(
        self,
        context: DiffusionRunContext,
        model_input: DenoiserInput,
    ) -> DenoiserInput:
        """Inject step-local conditions immediately before the network call.

        Runs once per CFG branch.  Must return a :class:`~..contracts.DenoiserInput`.
        """

        return model_input

    def after_denoiser(
        self,
        context: DiffusionRunContext,
        model_output: DenoiserOutput,
    ) -> DenoiserOutput:
        """Observe or transform one branch prediction.

        Must return a :class:`~..contracts.DenoiserOutput`.
        """

        return model_output

    def after_step(
        self,
        context: DiffusionRunContext,
        latents: Tensor,
    ) -> Tensor:
        """Observe or transform latents after a scheduler update.

        Must return a tensor broadcastable as the next latent state.
        Frozen-mask / frozen-context extensions typically fail here with
        :exc:`TypeError` / :exc:`ValueError` when required conditions are
        missing or shapes do not match.
        """

        return latents

    def on_diffusion_complete(self, context: DiffusionRunContext) -> None:
        """Observe final latents after all updates/cache commits, before decoding.

        This is a host-side submission boundary, including an optional final
        denoise pass. CUDA work is not synchronized; a consumer on another
        stream must record and wait on its own readiness event. This hook does
        not transfer ownership of latent/KV state or start another request.
        ``context.final_latents`` exposes the complete decoder input during
        this callback. Standard, chunked, AR-window and prefix-recompute
        runners invoke it once per request, after any final refinement.
        """

    def after_decode(
        self,
        context: DiffusionRunContext,
        sample: Tensor,
    ) -> Tensor:
        """Observe or transform the decoded result.

        Must return a tensor; the runner stores it as ``DiffusionOutput.sample``.
        """

        return sample

    def on_run_end(self, context: DiffusionRunContext) -> None:
        """Release extension-owned resources after a successful run."""

    def on_run_error(
        self,
        context: DiffusionRunContext,
        error: BaseException,
    ) -> None:
        """Release extension-owned resources after a failed run.

        Must not raise; the original ``error`` is re-raised by the runner.
        """


__all__ = ["DiffusionExtension", "DiffusionRunContext"]

"""Drive the approximate-attention dense-boundary schedule during denoising.

The opt-in sparse-attention lane (:mod:`...optimizations.approximate_attention`)
keeps the first/last ``dense_steps`` denoise steps dense — those set global
structure, where sparsity hurts quality most. That schedule is a per-step
counter on the installed state; nothing advances it unless the denoise loop
tells it which step it is on. This extension is that bridge: it resets the
counter at run start and advances it before each denoiser call, so the
dense-boundary window actually takes effect during real generation.

It is a no-op unless the denoiser's model carries
``_worldfoundry_approximate_attention`` (attached by the loader when the user
opted in via ``options["approximate_attention"]``), so it is safe to install
unconditionally.
"""

from __future__ import annotations

from ..contracts import DenoiserInput
from .base import DiffusionExtension, DiffusionRunContext


class ApproximateAttentionStepExtension(DiffusionExtension):
    """Advance the approximate-attention step schedule each denoise step."""

    extension_id = "approximate-attention-step"

    def _state(self, context: DiffusionRunContext):
        model = getattr(context.components.denoiser, "model", None)
        return getattr(model, "_worldfoundry_approximate_attention", None)

    def on_run_start(self, context: DiffusionRunContext) -> None:
        state = self._state(context)
        if state is not None:
            from ..optimizations.approximate_attention import reset_approximate_attention

            reset_approximate_attention(state)

    def before_denoiser(
        self,
        context: DiffusionRunContext,
        model_input: DenoiserInput,
    ) -> DenoiserInput:
        state = self._state(context)
        if state is not None and context.step is not None:
            from ..optimizations.approximate_attention import advance_approximate_step

            advance_approximate_step(
                state,
                context.step.index,
                context.request.sampling.num_inference_steps,
            )
        return model_input

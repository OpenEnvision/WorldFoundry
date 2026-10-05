"""AlayaWorld's LTX-specific dynamic-shape compilation adapter.

Generic module compilation and cache policy live in ``worldfoundry.core``;
this file remains with AlayaWorld because it marks that product's exact LTX
argument layout and patches its model forward contract.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import (
    BatchedPerturbationConfig,
    PerturbationType,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import (
    BlockPerturbationsProcessor,
    TransformerArgs,
)

# Defaults applied inside the patched forward. Overriding via CompilationConfig
# replaces these wholesale; it does not merge.
_DEFAULT_INDUCTOR_CONFIG: dict[str, Any] = {"unsafe_skip_cache_dynamic_shape_guards": True}
_DEFAULT_DYNAMO_CONFIG: dict[str, Any] = {"inline_inbuilt_nn_modules": True, "cache_size_limit": 256}


def _supported_config_options(config_module: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Drop options unavailable in the installed PyTorch release.

    AlayaWorld's upstream compile defaults target a newer Inductor.  PyTorch's
    config patcher raises ``KeyError`` for unknown keys, so version portability
    requires feature detection rather than assuming every optimization exists.
    """

    known = getattr(config_module, "_config", None)
    if isinstance(known, dict):
        return {key: value for key, value in options.items() if key in known}
    return {
        key: value
        for key, value in options.items()
        if hasattr(config_module, key)
    }


@dataclass(frozen=True)
class CompilationConfig:
    """``torch.compile`` configuration for transformer blocks. ``None`` keeps eager."""

    mode: str | None = None
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool | None = None
    inductor_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_INDUCTOR_CONFIG))
    dynamo_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_DYNAMO_CONFIG))


class _SeqDynamicMarkingProcessor:
    """Marks the per-block seq dim dynamic, then delegates to an inner processor.
    Installed by ``compile_transformer`` so the per-block compile artifact stays
    shape-polymorphic. Wraps whatever ``block_input_processor`` was already on
    the model -- callers that customised the processor keep their customisation;
    only the seq-dim marking is layered on top. Lives outside the compiled
    region, so ``mark_dynamic`` runs in eager mode on the tensors that are
    about to cross into the trace.
    """

    def __init__(self, inner: BlockPerturbationsProcessor, *, enabled: bool = True) -> None:
        self.inner = inner
        self.enabled = bool(enabled)

    def __call__(
        self,
        args: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
        cross_attn_type: PerturbationType,
    ) -> TransformerArgs:
        # ``torch.compile(dynamic=False)`` requires every input dimension to be
        # specialised.  Calling ``mark_dynamic`` in that mode creates mutually
        # exclusive guards and fails before the first block executes.  Keep the
        # wrapper installed so custom processors retain the same call path, but
        # only add dynamic-shape annotations when the compile policy permits it.
        if not self.enabled:
            return self.inner(args, perturbations, block_idx, self_attn_type, cross_attn_type)

        # Positional embeddings are second-from-last regardless of rope type:
        # split rope is (B, H, T, D//2) -- dim -2 == 2; interleaved rope is (B, T, D)
        # -- dim -2 == 1. Both work via the negative index.
        torch._dynamo.mark_dynamic(args.x, 1)
        cos, sin = args.positional_embeddings
        torch._dynamo.mark_dynamic(cos, cos.ndim - 2)
        torch._dynamo.mark_dynamic(sin, sin.ndim - 2)
        if args.cross_positional_embeddings is not None:
            cross_cos, cross_sin = args.cross_positional_embeddings
            torch._dynamo.mark_dynamic(cross_cos, cross_cos.ndim - 2)
            torch._dynamo.mark_dynamic(cross_sin, cross_sin.ndim - 2)
        if args.self_attention_mask is not None:
            # Dense form is (B, 1, T, T); key-padding form (from the SP wrapper)
            # is (B, 1, 1, T) -- leave the size-1 query dim static so Dynamo
            # keeps the broadcast.
            if args.self_attention_mask.shape[2] > 1:
                torch._dynamo.mark_dynamic(args.self_attention_mask, 2)
            torch._dynamo.mark_dynamic(args.self_attention_mask, 3)
        if args.context_mask is not None:
            torch._dynamo.mark_dynamic(args.context_mask, 2)
        # `timesteps` / `embedded_timestep` are per-token when conditioning sets a
        # per-position denoise mask, in which case their dim 1 equals the seq length
        # and must vary with it. When they're a single timestep broadcast across the
        # sequence (dim 1 == 1), leaving them static lets Dynamo keep the size-1
        # broadcast.
        if args.timesteps.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.timesteps, 1)
        if args.embedded_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.embedded_timestep, 1)
        # `cross_scale_shift_timestep` is the cross-attn AdaLN scale/shift input
        # derived from the own-modality per-token timesteps (denoise_mask * sigma),
        # so its dim 1 equals the seq length when conditioning is per-token.
        # `cross_gate_timestep` is the cross-modality sigma scalar -- dim 1 is 1
        # and broadcasts, leave it static. Same guard pattern as `timesteps`.
        if args.cross_scale_shift_timestep is not None and args.cross_scale_shift_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.cross_scale_shift_timestep, 1)
        return self.inner(args, perturbations, block_idx, self_attn_type, cross_attn_type)


def compile_transformer(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Compile each transformer block via ``torch.compile`` with the given settings.
    The patched forward emits ``torch.compiler.cudagraph_mark_step_begin()`` once
    per step. Under CUDA-graph-enabling modes (``"reduce-overhead"`` /
    ``"max-autotune"``) this overrides Dynamo's per-invocation auto-mark
    heuristic, which would otherwise fire once per compiled block call (48 per
    forward) and treat each block call as a fresh iteration. Under other modes
    the mark is a no-op (decrements an unread counter).
    """
    model.transformer_blocks = torch.nn.ModuleList(
        torch.compile(m, mode=config.mode, backend=config.backend, fullgraph=config.fullgraph, dynamic=config.dynamic)
        for m in model.transformer_blocks
    )
    model.block_input_processor = _SeqDynamicMarkingProcessor(
        inner=model.block_input_processor,
        enabled=config.dynamic is not False,
    )

    def patched_dynamo_forward(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        inductor_options = _supported_config_options(
            torch._inductor.config, config.inductor_config
        )
        dynamo_options = _supported_config_options(
            torch._dynamo.config, config.dynamo_config
        )
        with (
            torch._inductor.config.patch(**inductor_options),
            torch._dynamo.config.patch(**dynamo_options),  # type: ignore[attr-defined]
        ):
            return model.forward_without_compilation(*args, **kwargs)

    model.forward_without_compilation = model.forward
    model.forward = patched_dynamo_forward
    return model

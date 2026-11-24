"""Builder ops for swapping attention backends on a meta model before load."""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.core.attention.backends.model_backends import AttentionCallable, MaskedAttentionCallable
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.attention_policy import (
    AttentionFunction,
    MaskedAttentionFunction,
)

from .module_ops import ModuleOps


def set_attention_module_op(
    attention: AttentionFunction | AttentionCallable | None = None,
    masked_attention: MaskedAttentionFunction | MaskedAttentionCallable | None = None,
) -> ModuleOps:
    """Build a ``ModuleOps`` that overrides the attention callables on every
    ``Attention`` submodule of a model. Applied via ``create_meta_model`` so
    the meta model is mutated before weight loading. Matcher returns False
    for models with no ``Attention`` submodules, so the op is a no-op there.
    Either or both slots may be supplied; *None* leaves that slot untouched.
    """
    fn = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
    masked_fn = (
        masked_attention.to_callable() if isinstance(masked_attention, MaskedAttentionFunction) else masked_attention
    )

    def matcher(model: torch.nn.Module) -> bool:
        return any(isinstance(m, Attention) for m in model.modules())

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        for module in model.modules():
            if isinstance(module, Attention):
                if fn is not None:
                    module.attention_function = fn
                if masked_fn is not None:
                    module.masked_attention_function = masked_fn
        return model

    return ModuleOps(name="set_attention_backend", matcher=matcher, mutator=mutator)

"""mira.world_model: the action-conditioned latent diffusion world model.

Public API:
    ActionConfig — the key vocabulary and source/target fps of an action stream
    ActionTensors — batched, time-indexed keyboard/mouse actions consumed by the action encoder
    stack_action_tensors — concatenate per-sample ActionTensors along the batch dimension
    LatentWorldModelConfig — architecture/training config of the world model
    WorldModelInferenceConfig — sampling knobs for the autoregressive rollout
    LatentWorldModel — the frozen-codec + diffusion-transformer world model
    MultiWrapperWorldModel — tiles n_players clips into one frame and wraps a LatentWorldModel
    MultiWrapperWorldModelConfig — player count + inner world-model config of the wrapper
    InferenceOutputs — the outputs of an autoregressive rollout
    DiffusionTransformer — the action-conditioned flow-matching transformer over codec latents
    ActionEncoder — embeds keyboard/mouse actions into per-latent-frame conditioning tokens
    build_inference_schedule — the tau integration grid for the denoiser

``LatentWorldModel`` / ``InferenceOutputs`` are imported lazily (PEP 562): they pull in the codec,
which itself depends on this package (via ``data.batch`` -> ``actions_config``), so eagerly importing
them here would create an import cycle. Accessing them as attributes resolves them on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .actions_config import ActionConfig, ActionTensors, stack_action_tensors
from .config import LatentWorldModelConfig, WorldModelInferenceConfig
from .diffusion_transformer import DiffusionTransformer
from .layers.action_encoder import ActionEncoder
from .schedule import build_inference_schedule

if TYPE_CHECKING:
    from .latent_world_model import InferenceOutputs, LatentWorldModel
    from .multi_wrapper_world_model import MultiWrapperWorldModel, MultiWrapperWorldModelConfig

__all__ = [
    "ActionConfig",
    "ActionEncoder",
    "ActionTensors",
    "DiffusionTransformer",
    "InferenceOutputs",
    "LatentWorldModel",
    "LatentWorldModelConfig",
    "MultiWrapperWorldModel",
    "MultiWrapperWorldModelConfig",
    "WorldModelInferenceConfig",
    "build_inference_schedule",
    "stack_action_tensors",
]

_LAZY = {"LatentWorldModel", "InferenceOutputs"}
_LAZY_MULTI = {"MultiWrapperWorldModel", "MultiWrapperWorldModelConfig"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import latent_world_model

        return getattr(latent_world_model, name)
    if name in _LAZY_MULTI:
        from . import multi_wrapper_world_model

        return getattr(multi_wrapper_world_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

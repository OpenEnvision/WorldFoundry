import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.networks.step_video.normalization import (
    get_activation as _get_activation,
)

_ELU = nn.ELU()


def get_activation(act_fn: str) -> nn.Module:
    """Extend the shared activation registry with EasyAnimate's ELU option."""
    if act_fn.lower() == "elu":
        return _ELU
    return _get_activation(act_fn)

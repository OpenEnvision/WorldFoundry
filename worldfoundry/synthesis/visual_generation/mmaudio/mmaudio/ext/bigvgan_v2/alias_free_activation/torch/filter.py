# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio.alias_free import kaiser_sinc_filter1d


from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio.alias_free import LowPassFilter1d

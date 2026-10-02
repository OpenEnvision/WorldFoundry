from typing import Dict

import torch
import torch.nn as nn
from vwm.util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super(Denoiser, self).__init__()
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
            self,
            network: nn.Module,
            noised_input: torch.Tensor,
            sigma: torch.Tensor,
            cond: Dict
    ):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, noised_input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return (network(noised_input * c_in, c_noise, cond) * c_out + noised_input * c_skip)

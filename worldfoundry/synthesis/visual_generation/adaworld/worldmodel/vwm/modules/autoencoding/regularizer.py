from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn

from ...modules.distributions.distributions import DiagonalGaussianDistribution


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super(AbstractRegularizer, self).__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError



class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super(DiagonalGaussianRegularizer, self).__init__()
        self.sample = sample


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z, {}

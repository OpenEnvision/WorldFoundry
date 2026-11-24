import torch
from torch import nn

from ...modules.distributions.distributions import DiagonalGaussianDistribution


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample() if self.sample else posterior.mode()
        return z, {}

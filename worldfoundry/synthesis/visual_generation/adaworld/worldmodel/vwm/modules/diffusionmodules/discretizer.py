from abc import abstractmethod

import numpy as np
import torch
from vwm.util import append_zero


def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False, dtype=int)[::-1]


class Discretization:
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class EDMDiscretization(Discretization):
    def __init__(self, sigma_min=0.002, sigma_max=700.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class EDMShiftDiscretization(Discretization):
    def __init__(self, sigma_min=0.002, sigma_max=700.0, rho=7.0, shifting_factor=17.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.shifting_factor = shifting_factor

    def get_sigmas(self, n, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        ramp = (self.shifting_factor * ramp) / (1 + (self.shifting_factor - 1) * ramp)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas

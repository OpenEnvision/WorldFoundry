"""EDM log-normal SDE used by Cosmos Predict1 residual samplers.

:class:`EDMSDE` draws training / solver sigmas from a log-normal
whose ``(p_mean, p_std)`` match Karras EDM, then clamps to
``[sigma_min, sigma_max]``.  :meth:`EDMSDE.marginal_prob` is the
identity perturbation ``(x0, sigma)``; subclasses may override.

``sample_t`` currently allocates on CUDA.  This module does not
implement the residual ODE stepper; see :mod:`.runge_kutta` and
:mod:`.res_sampler`.
"""

from statistics import NormalDist

import numpy as np
import torch


class EDMSDE:
    """Edmsde implementation."""

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
    ):
        """Init.

        Args:
            p_mean: The p mean.
            p_std: The p std.
            sigma_max: The sigma max.
            sigma_min: The sigma min.
        """
        self.gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample t.

        Args:
            batch_size: The batch size.

        Returns:
            The return value.
        """
        cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]

        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        return torch.exp(log_sigma)

    def marginal_prob(self, x0: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This is trivial in the base class, but may be used by derived classes in a more interesting way"""
        return x0, sigma

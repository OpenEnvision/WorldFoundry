"""Framework-independent latent distribution and VAE output types.

Diagonal Gaussian / Dirac plus named output tuples so a VAE encode
path does not depend on diffusers. Sampling stays in one place for
KL and deterministic eval.

Not this module:
    2D decoder blocks live in :mod:`worldfoundry.core.nn.latent.vae2d`.
    Scheduler noise calendars live in
    :mod:`worldfoundry.core.nn.diffusion.schedulers`.

Public surface:

- :class:`AbstractDistribution` / :class:`DiracDistribution` /
  :class:`DiagonalGaussianDistribution` — sample / mode / KL.
- :class:`GaussianDistribution` / :class:`IdentityDistribution` —
  ``nn.Module`` wrappers for tokenizer graphs.
- :func:`normal_kl` — broadcastable closed-form KL.
- :class:`AutoencoderKLOutput` / :class:`DecoderOutput`.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch


# ──────────────────────────────────────────────────────────────────────────
# Abstract + Dirac — deterministic tokenize / eval without a posterior
# ──────────────────────────────────────────────────────────────────────────


class AbstractDistribution:
    """Minimal distribution protocol retained by checkpoint-shaped VAE modules."""

    def sample(self):
        """Draw a latent; subclasses must implement this."""

        raise NotImplementedError

    def mode(self):
        """Return the most likely latent; subclasses must implement this."""

        raise NotImplementedError


class DiracDistribution(AbstractDistribution):
    """Deterministic distribution returning one stored value."""

    def __init__(self, value: torch.Tensor) -> None:
        """Hold ``value`` as both sample and mode (no noise at eval)."""

        self.value = value

    def sample(self) -> torch.Tensor:
        """Return the stored tensor unchanged."""

        return self.value

    def mode(self) -> torch.Tensor:
        """Same as :meth:`sample` — a Dirac has a single support point."""

        return self.value


# ──────────────────────────────────────────────────────────────────────────
# Module wrappers — tokenizer graphs that emit (sample, (mean, logvar))
# ──────────────────────────────────────────────────────────────────────────


class IdentityDistribution(torch.nn.Module):
    """Pass deterministic tokenizer parameters through unchanged."""

    def forward(self, parameters: torch.Tensor):
        """Return ``(parameters, (0, 0))`` so KL-shaped call sites stay valid."""

        zero = parameters.new_zeros(1)
        return parameters, (zero, zero)


class GaussianDistribution(torch.nn.Module):
    """Sample a diagonal Gaussian from concatenated mean/log-variance parameters."""

    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0) -> None:
        """Clamp log-variance so ``exp(0.5·logvar)`` cannot overflow in half."""

        super().__init__()
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def forward(self, parameters: torch.Tensor):
        """Split channels into mean/logvar, sample, and return ``(z, (μ, logσ²))``.

        ``parameters`` is ``[B, 2C, ...]`` along dim=1 (VAE encoder layout).
        """

        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return sample, (mean, logvar)


# ──────────────────────────────────────────────────────────────────────────
# Diagonal Gaussian posterior — KL / NLL used by native VAEs
# ──────────────────────────────────────────────────────────────────────────


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior used by native variational autoencoders."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        """Split ``parameters`` on dim=1 and clamp log-variance to ``[-30, 20]``.

        ``deterministic=True`` zeros std/var so :meth:`sample` equals the mean
        (eval / teacher-forcing encode).
        """

        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(
        self,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Reparameterize ``μ + σ ⊙ ε``; ``noise`` is cast to the posterior device/dtype."""

        if noise is None:
            noise = torch.randn(
                self.mean.shape,
                generator=generator,
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
        else:
            noise = noise.to(device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        """Return the mean (MAP of a diagonal Gaussian)."""

        return self.mean

    def kl(self, other: "DiagonalGaussianDistribution | None" = None) -> torch.Tensor:
        """Batch KL; ``other is None`` is KL to standard normal.

        Reduction sums every non-batch axis. Deterministic posteriors return 0.
        """

        if self.deterministic:
            return torch.zeros((), device=self.mean.device, dtype=self.mean.dtype)
        dimensions = tuple(range(1, self.mean.ndim))
        if other is None:
            value = 0.5 * (self.mean.square() + self.var - 1.0 - self.logvar)
        else:
            value = 0.5 * (
                (self.mean - other.mean).square() / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )
        return value.sum(dim=dimensions)

    def nll(self, sample: torch.Tensor, dims: tuple[int, ...] | list[int] | None = None) -> torch.Tensor:
        """Return diagonal-Gaussian negative log likelihood."""

        if self.deterministic:
            return torch.zeros((), device=self.mean.device, dtype=self.mean.dtype)
        dimensions = tuple(dims) if dims is not None else tuple(range(1, self.mean.ndim))
        value = math.log(2.0 * math.pi) + self.logvar + (sample - self.mean).square() / self.var
        return 0.5 * value.sum(dim=dimensions)


def normal_kl(
    mean1: torch.Tensor | float,
    logvar1: torch.Tensor | float,
    mean2: torch.Tensor | float,
    logvar2: torch.Tensor | float,
) -> torch.Tensor:
    """Compute element-wise KL divergence between broadcastable Gaussians."""

    tensor = next(
        (value for value in (mean1, logvar1, mean2, logvar2) if isinstance(value, torch.Tensor)),
        None,
    )
    if tensor is None:
        raise TypeError("normal_kl requires at least one tensor argument")
    first_logvar = logvar1 if isinstance(logvar1, torch.Tensor) else torch.as_tensor(logvar1, device=tensor.device)
    second_logvar = logvar2 if isinstance(logvar2, torch.Tensor) else torch.as_tensor(logvar2, device=tensor.device)
    return 0.5 * (
        -1.0
        + second_logvar
        - first_logvar
        + torch.exp(first_logvar - second_logvar)
        + (mean1 - mean2) ** 2 * torch.exp(-second_logvar)
    )


# ──────────────────────────────────────────────────────────────────────────
# Named encode / decode outputs — avoid depending on diffusers dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class AutoencoderKLOutput:
    """Encoded VAE posterior."""

    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    """Decoded video tensor and optional posterior."""

    sample: torch.Tensor
    posterior: DiagonalGaussianDistribution | None = None


__all__ = [
    "AbstractDistribution",
    "AutoencoderKLOutput",
    "DecoderOutput",
    "DiagonalGaussianDistribution",
    "DiracDistribution",
    "GaussianDistribution",
    "IdentityDistribution",
    "normal_kl",
]

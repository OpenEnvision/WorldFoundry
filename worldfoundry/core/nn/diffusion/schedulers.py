"""Reusable diffusion scheduler primitives.

Noise calendars, sigma shift, and x0/noise/velocity conversions live here so
each DiT does not invent its own schedule. Two families:

- :class:`SchedulerInterface` is the classic DDPM-style contract
  (``alphas_cumprod`` plus ``add_noise``). The conversion helpers run in
  float64 so half-precision timesteps do not accumulate drift when a sampler
  switches prediction type (x0 ↔ noise ↔ velocity).
- :class:`FlowMatchScheduler` is the flow-matching / rectified-flow schedule
  used by bundled video runtimes. ``shift`` (and optional exponential shift)
  concentrates steps near sigma=0, which is where video DiTs are most
  sensitive. ``step`` is an Euler integrator on ``sample + v * Δσ``.

Callers should treat this module as the single source of σ; do not re-derive
linspaces in model code.

Not this module:
    Velocity → sample without a schedule lives in
    :func:`~worldfoundry.core.nn.transformer.dit.velocity_to_denoised`.
    Timestep *embeddings* live in :mod:`worldfoundry.core.nn.diffusion.timestep`.

Public surface:

- :class:`SchedulerInterface` — DDPM ``alphas_cumprod`` + prediction converters.
- :class:`FlowMatchScheduler` — shifted sigma grid and Euler ``step``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch


# ──────────────────────────────────────────────────────────────────────────
# DDPM-style contract — convert x0 / noise / velocity in float64
# ──────────────────────────────────────────────────────────────────────────


class SchedulerInterface(ABC):
    """Base interface for diffusion noise schedules."""

    alphas_cumprod: torch.Tensor

    @abstractmethod
    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """Run the forward corruption process."""

    def convert_x0_to_noise(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a clean-data prediction to a noise prediction."""

        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(
            lambda x: x.double().to(x0.device),
            [x0, xt, self.alphas_cumprod],
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t**0.5 * x0) / beta_prod_t**0.5
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self,
        noise: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a noise prediction to a clean-data prediction."""

        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(
            lambda x: x.double().to(noise.device),
            [noise, xt, self.alphas_cumprod],
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t**0.5 * noise) / alpha_prod_t**0.5
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self,
        velocity: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a velocity prediction to a clean-data prediction."""

        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device),
            [velocity, xt, self.alphas_cumprod],
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = alpha_prod_t**0.5 * xt - beta_prod_t**0.5 * velocity
        return x0_pred.to(original_dtype)


# ──────────────────────────────────────────────────────────────────────────
# Flow-matching / rectified-flow — shifted σ and Euler integration
# ──────────────────────────────────────────────────────────────────────────


class FlowMatchScheduler:
    """Flow-matching scheduler shared by bundled video runtimes.

    Why shift exists: a linear sigma grid spends too many steps at high noise
    for video. ``shift * σ / (1 + (shift-1)σ)`` (or the exponential form)
    stretches the low-noise end. ``denoising_strength < 1`` starts from an
    intermediate sigma for img2vid / inpaint. ``extra_one_step`` drops the
    terminal sample so the last model call is still a real denoise step.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ) -> None:
        """Store shift knobs and build the default inference ``sigmas`` grid.

        ``sigma_min`` defaults to ``0.003/1.002`` (SD3 / Flux convention).
        ``set_timesteps`` is called here so a constructed scheduler is usable.
        """

        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
        dynamic_shift_len: int | None = None,
        exponential_shift_mu: float | None = None,
    ) -> None:
        """Rebuild ``sigmas`` / ``timesteps`` for a new step count or shift."""
        if shift is not None:
            self.shift = float(shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = exponential_shift_mu
            if mu is None and dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            if mu is None:
                mu = self.exponential_shift_mu
            if mu is None:
                raise ValueError("exponential_shift requires a shift mu or dynamic_shift_len")
            exp_mu = math.exp(float(mu))
            self.sigmas = exp_mu / (exp_mu + (1 / self.sigmas - 1))
        else:
            # Stretch low-noise steps: shift * σ / (1 + (shift-1)σ).
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_sigma = 1 - self.sigmas
            scale = one_minus_sigma[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - one_minus_sigma / scale
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
        self.training = bool(training)

    def _indices(self, timestep, *, device: torch.device) -> tuple[torch.Tensor, bool]:
        """Nearest-neighbor lookup of ``timestep`` on the stored calendar.

        Returns the index tensor and whether the input was a scalar so
        :meth:`_broadcast_sigma` can keep a 0-dim σ for unbatched samples.
        """

        timesteps = self.timesteps.to(device)
        value = torch.as_tensor(timestep, device=device, dtype=timesteps.dtype)
        scalar = value.ndim == 0 or value.numel() == 1
        value = value.reshape(-1)
        indices = torch.argmin((timesteps.unsqueeze(0) - value.unsqueeze(1)).abs(), dim=1)
        return indices, scalar

    @staticmethod
    def _broadcast_sigma(sigma: torch.Tensor, sample: torch.Tensor, scalar: bool) -> torch.Tensor:
        """Keep a 0-dim σ for scalars; otherwise ``[B, 1, 1, ...]`` against ``sample``."""

        if scalar:
            return sigma.reshape(())
        return sigma.reshape(-1, *((1,) * (sample.ndim - 1)))

    def step(self, model_output, timestep, sample, to_final: bool = False):
        """One Euler step: ``sample + velocity * (σ_next - σ)``.

        ``to_final`` forces the terminal sigma even mid-schedule (last-step
        snap). Inverse/reversed calendars use terminal ``1.0`` instead of
        ``0.0`` so the integrator still walks toward clean data.
        """
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id, scalar = self._indices(timestep, device=model_output.device)
        sigma = self._broadcast_sigma(self.sigmas[timestep_id], sample, scalar)
        terminal = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        next_id = torch.clamp(timestep_id + 1, max=len(self.timesteps) - 1)
        next_sigma = self.sigmas[next_id]
        terminal_mask = timestep_id + 1 >= len(self.timesteps)
        if to_final:
            terminal_mask = torch.ones_like(terminal_mask, dtype=torch.bool)
        next_sigma = torch.where(terminal_mask, torch.full_like(next_sigma, terminal), next_sigma)
        sigma_ = self._broadcast_sigma(next_sigma, sample, scalar)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, stabilized_sample):
        """Recover a velocity that maps ``sample`` to ``stabilized_sample``."""

        self.sigmas = self.sigmas.to(sample.device)
        self.timesteps = self.timesteps.to(sample.device)
        timestep_id, scalar = self._indices(timestep, device=sample.device)
        sigma = self._broadcast_sigma(self.sigmas[timestep_id], sample, scalar)
        return (sample - stabilized_sample) / sigma

    def add_noise(self, original_samples, noise, timestep):
        """Run the forward corruption process."""

        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id, scalar = self._indices(timestep, device=noise.device)
        sigma = self._broadcast_sigma(self.sigmas[timestep_id], noise, scalar)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        """Flow-matching target ``noise - sample`` (velocity toward noise)."""
        del timestep
        return noise - sample

    def training_weight(self, timestep):
        """Per-timestep loss weight from the Gaussian bump built in training mode."""
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id, _ = self._indices(timestep, device=timestep.device)
        return self.linear_timesteps_weights[timestep_id]

    @staticmethod
    def calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        """Interpolate exponential-shift μ from sequence length (Flux-style)."""
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        return image_seq_len * slope + intercept


__all__ = ["FlowMatchScheduler", "SchedulerInterface"]

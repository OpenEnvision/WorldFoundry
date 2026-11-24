# Official LTX-2 Scheduler — copied from ltx_core/components/schedulers.py
# with ltx_core dependency removed and training utilities preserved.
import math

import torch

from worldfoundry.base_models.diffusion_model.schedulers.ltx_sigma import BetaScheduler, LinearQuadraticScheduler

from ..configs.ltx2_config import ltx2_shared_cfg

__all__ = ["DISTILLED_SIGMA_VALUES", "BetaScheduler", "LTX2Scheduler", "LinearQuadraticScheduler"]

DISTILLED_SIGMA_VALUES = ltx2_shared_cfg.distilled_sigma_values

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler:
    """
    Default scheduler for LTX-2 diffusion sampling.
    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.
    Copied verbatim from official ltx_core/components/schedulers.py.
    """

    # Keep class-level anchors for backward compat with training code
    BASE_SHIFT_ANCHOR = BASE_SHIFT_ANCHOR
    MAX_SHIFT_ANCHOR = MAX_SHIFT_ANCHOR

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.base_shift = base_shift
        self.stretch = stretch
        self.terminal = terminal
        self._current_shift = shift

        # Compute default sigmas for training add_noise
        self.sigmas = self._compute_sigmas_with_stretch(num_train_timesteps, shift)
        self.timesteps = self.sigmas * num_train_timesteps

    # ── Official execute() — verbatim copy ──────────────────────────────
    def execute(
        self,
        steps: int,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        default_number_of_tokens: int = MAX_SHIFT_ANCHOR,
        **_kwargs,
    ) -> torch.FloatTensor:
        tokens = math.prod(latent.shape[2:]) if latent is not None else default_number_of_tokens
        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)

    # ── Training utilities ──────────────────────────────────────────────
    def _compute_sigmas_with_stretch(self, num_steps: int, shift: float) -> torch.Tensor:
        """Compute sigma schedule with shift + stretch, matching execute() behavior.

        Same logic as execute(): independent of the latent token count, using the given shift.
        1. Flux time shift: sigma = exp(shift) / (exp(shift) + (1/t - 1))
        2. Stretch so the smallest non-zero sigma equals the terminal value
        """
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        sigmas = torch.where(
            sigmas != 0,
            math.exp(shift) / (math.exp(shift) + (1 / sigmas - 1)),
            torch.zeros_like(sigmas),
        )
        if self.stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - self.terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched
        return sigmas.to(torch.float32)

    def compute_shift(self, num_tokens: int) -> float:
        """Compute dynamic shift based on number of tokens."""
        x1, x2 = BASE_SHIFT_ANCHOR, MAX_SHIFT_ANCHOR
        mm = (self.shift - self.base_shift) / (x2 - x1)
        return num_tokens * mm + self.base_shift - mm * x1

    def set_timesteps(self, num_inference_steps: int, latent: torch.Tensor | None = None):
        if latent is not None:
            self._current_shift = self.compute_shift(math.prod(latent.shape[2:]))
        self.sigmas = self._compute_sigmas_with_stretch(num_inference_steps, self._current_shift)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def get_distilled_sigmas(self) -> torch.Tensor:
        return torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32)

    def _lookup_sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Look up sigma. The table already has shift and stretch applied at init/set_timesteps,
        so this is a nearest-neighbour lookup."""
        self.sigmas = self.sigmas.to(timesteps.device)
        self.timesteps = self.timesteps.to(timesteps.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timesteps.float().unsqueeze(1)).abs(), dim=1)
        return self.sigmas[timestep_id]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        num_tokens: int | None = None,
    ) -> torch.Tensor:
        """Look up the shifted and stretched sigma and build the noisy sample."""
        sigmas = self._lookup_sigma(timesteps.flatten()).reshape(timesteps.shape)
        while sigmas.dim() < original_samples.dim():
            sigmas = sigmas.unsqueeze(-1)
        sigmas = sigmas.to(original_samples.device, dtype=original_samples.dtype)
        return (1 - sigmas) * original_samples + sigmas * noise

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigmas = self._lookup_sigma(timestep.flatten()).reshape(timestep.shape)
        while sigmas.dim() < x0.dim():
            sigmas = sigmas.unsqueeze(-1)
        sigmas = sigmas.to(x0.device, dtype=x0.dtype)
        return (xt - (1 - sigmas) * x0) / sigmas.clamp(min=1e-8)

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigmas = self._lookup_sigma(timestep.flatten()).reshape(timestep.shape)
        while sigmas.dim() < noise.dim():
            sigmas = sigmas.unsqueeze(-1)
        sigmas = sigmas.to(noise.device, dtype=noise.dtype)
        return (xt - sigmas * noise) / (1 - sigmas).clamp(min=1e-8)

    def convert_velocity_to_x0(self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.convert_noise_to_x0(velocity, xt, timestep)

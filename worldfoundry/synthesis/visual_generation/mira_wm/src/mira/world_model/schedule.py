"""Sampling schedules and the symlog action normalisation used by the world model.

``symlog_normalize`` squashes raw mouse deltas into a bounded range before the action encoder
embeds them; ``build_inference_schedule`` produces the ``tau`` integration grid for the flow-matching
denoiser at inference time.
"""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Signed logarithm ``sign(x) * log(1 + |x|)``."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symlog_normalize(value: torch.Tensor, scale: float, max_value: float) -> torch.Tensor:
    """Normalize a tensor with a symmetric log transform.

    Args:
        value: Tensor to normalize.
        scale: Linear multiplier applied before the symlog transform; controls the squashing
            strength.
        max_value: Expected maximum absolute value (before scaling); sets the normalization range so
            the result of ``value == max_value`` maps to 1.

    Returns:
        The symlog-normalized tensor, with ``value == max_value`` mapping to 1.
    """
    max_value_t = torch.tensor(max_value, device=value.device)

    value = symlog(scale * value)
    norm_constant = symlog(scale * max_value_t)

    result = value / norm_constant

    return result


def linear_quadratic_schedule(
    n_steps: int,
    device: torch.device,
    threshold_noise: float = 0.1,
    n_linear_steps: int | None = None,
) -> torch.Tensor:
    """Linear-then-quadratic ``tau`` schedule (linear near 0, quadratic toward 1)."""
    if n_linear_steps is None:
        n_linear_steps = n_steps // 2
    if n_steps < 2:
        return torch.tensor([0.0, 1.0], device=device)
    n_quadratic_steps = n_steps - n_linear_steps

    linear_timesteps = torch.linspace(0, threshold_noise, n_linear_steps + 1, device=device)
    start_value = torch.sqrt(linear_timesteps[-1])
    quadratic_timesteps = torch.linspace(start_value, 1.0, n_quadratic_steps + 1, device=device) ** 2

    timesteps = torch.cat([linear_timesteps[:-1], quadratic_timesteps])
    return timesteps


def linear_schedule(n_steps: int, device: torch.device) -> torch.Tensor:
    """Simple uniformly-spaced sampling schedule from 0 to 1 (no threshold / knee).

    Returns n_steps + 1 evenly-spaced points, i.e. n_steps integration deltas of equal size.
    Empirically matches or beats the tuned linear_quadratic schedule on Frechet DINO while having no
    hyperparameters.
    """
    if n_steps < 1:
        return torch.tensor([0.0, 1.0], device=device)
    return torch.linspace(0, 1.0, n_steps + 1, device=device)


def build_inference_schedule(
    n_steps: int, device: torch.device, schedule_type: str = "linear_quadratic"
) -> torch.Tensor:
    """Dispatch to a sampling schedule by name.

    Returns timesteps in ``[0, 1]`` with ``n_steps`` integration deltas (``n_steps + 1`` points),
    ending at 1.0.
    """
    if schedule_type == "linear":
        return linear_schedule(n_steps, device)
    elif schedule_type == "linear_quadratic":
        return linear_quadratic_schedule(n_steps, device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type!r}")

from __future__ import annotations

import torch
from torch import Tensor


def flow_match_timesteps(
    num_steps: int,
    scheduler_scale: float,
    device: torch.device | str,
) -> Tensor:
    """Rescaled linear timestep schedule used by Kandinsky 6.

    Returns (num_steps + 1,) tensor from ~1.0 to 0.0.
    scheduler_scale=1 → uniform; scheduler_scale>1 → front-loaded.
    """
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return scheduler_scale * t / (1.0 + (scheduler_scale - 1.0) * t)

from pathlib import Path
from typing import Any

import torch

from worldfoundry.core.nn.blocks.normalization import (
    rms_norm_tensor as rms_norm,  # noqa: F401 - Public positional-epsilon RMS export.
)


def check_config_value(config: dict, key: str, expected: Any) -> None:
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def find_matching_file(root_path: str, pattern: str) -> Path:
    """
    Recursively search for files matching a glob pattern and return the first match.
    """
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return matches[0]

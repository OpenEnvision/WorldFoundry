"""Misc utilities.

Note on `randn_*_fp32_cast` helpers below: empirically, sampling N(0, 1)
directly in bf16 quantizes values to ~256 discrete levels per unit
interval and degrades both inference quality (verified with
WanVideoReCamMasterPipeline) and inference-time distributional
fidelity. The helpers sample in fp32 and cast to the requested dtype.
A module-level flag (`set_fp32_noise(False)`) restores the original
bf16-direct sampling for ablation / config-toggle.
"""

import random
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


@contextmanager
def nvtx_range(name, enabled=True):
    """Context manager for NVTX profiling ranges (used with NVIDIA Nsight Systems).

    When ``enabled=False`` the context manager is a pure no-op so there is zero
    overhead in normal (non-profiling) runs.

    Example usage::

        with nvtx_range("vae_decode", profile):
            video = vae.decode(latents)

    Args:
        name: Label that appears in the Nsight Systems timeline.
        enabled: Whether to actually push/pop the NVTX range.
    """
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


# use_fp32_noise in config
_USE_FP32_NOISE = True  # default; flip via set_fp32_noise(False)


def set_fp32_noise(enabled: bool) -> None:
    """Toggle whether `randn_fp32_cast` / `randn_like_fp32_cast` sample in
    fp32 first (True, default) or fall through to native sampling at the
    requested dtype (False — recovers original bf16 randn behavior)."""
    global _USE_FP32_NOISE
    _USE_FP32_NOISE = bool(enabled)


def randn_fp32_cast(shape, *, device, dtype, generator=None):
    """N(0, 1) sampled in fp32 on `device` then cast to `dtype` when the
    global fp32-noise flag is on (default).

    When the flag is off, equivalent to `torch.randn(shape, device=device,
    dtype=dtype, generator=generator)` — i.e., the original bf16-native
    behavior is preserved.
    """
    if not _USE_FP32_NOISE:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    n = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return n.to(dtype=dtype) if dtype != torch.float32 else n


def randn_like_fp32_cast(x, *, generator=None):
    """`torch.randn_like(x)` routed through `randn_fp32_cast` (so it honors
    the same fp32-or-native flag)."""
    return randn_fp32_cast(x.shape, device=x.device, dtype=x.dtype, generator=generator)

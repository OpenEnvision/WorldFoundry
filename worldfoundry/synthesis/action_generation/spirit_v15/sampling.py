# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team. Modified by WorldFoundry for inference-only use.
# ==============================================================================

import torch


def sample_noise(shape, device, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=dtype, device=device)

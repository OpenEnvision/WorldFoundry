"""Reference self-forcing flow sampling shared by causal video adapters."""

import torch


def reference_flow_schedule(*, indices, shift, num_train_timesteps=1000, sigma_max=0.999, device=None):
    """FP32 warped sigma table and integer network timesteps, without BF16 rounding."""
    sigmas = torch.linspace(sigma_max, 0.0, num_train_timesteps + 1, dtype=torch.float32)[:-1]
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    selected = sigmas[list(indices)].to(device=device)
    return (selected * num_train_timesteps).to(torch.int64), selected


def reference_flow_to_clean(noisy, flow, sigma):
    """Keep reference arithmetic, casting clean state at the network boundary."""
    return (noisy - sigma * flow).to(flow.dtype)

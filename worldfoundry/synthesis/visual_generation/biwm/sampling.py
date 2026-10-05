"""BiWM flow inference; shared transformer and causal VAE stay in BaseModel."""
from __future__ import annotations

import math
import torch


def shifted_sigmas(values, shift):
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("sigma_shift must be finite and positive")
    values = list(values)
    if (not values or values[0] != 1 or values[-1] != 0
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or any(a <= b for a, b in zip(values, values[1:]))):
        raise ValueError("Sigmas must strictly decrease from 1 to 0")
    return [shift * s / (1 + (shift - 1) * s) for s in values]


def history_indices(total, max_frames, sink_frames, device):
    if total <= max_frames:
        return torch.arange(total, device=device)
    sink = min(sink_frames, max_frames)
    return torch.cat([torch.arange(sink, device=device),
                      torch.arange(total - (max_frames - sink), total, device=device)])


def velocity(model, x, sigma, context, labels, history=0):
    return model(x, torch.tensor([sigma * 1000], device=x.device), context,
                 action_labels=labels, cond_latent_frames=history)


@torch.no_grad()
def sample_stage2(model, context, labels, *, latent_shape, num_chunks,
                  chunk_size=4, max_chunks=5, sink_chunks=1, prefix=None,
                  sigmas=(1., .75, .5, .25), sigma_shift=5., generator=None):
    if (num_chunks < 1 or chunk_size < 1 or max_chunks < 1 or sink_chunks < 0):
        raise ValueError("Invalid chunk or history window sizes")
    schedule = shifted_sigmas([*sigmas, 0.], sigma_shift)
    device, dtype = context.device, context.dtype
    clean = prefix
    channels, height, width = latent_shape
    for _ in range(num_chunks):
        start = 0 if clean is None else clean.shape[2]
        indices = history_indices(start, (max_chunks - 1) * chunk_size,
                                  sink_chunks * chunk_size, device)
        n_history = indices.numel()
        history = clean[:, :, indices] if n_history else None
        current = torch.arange(start, start + chunk_size, device=device)
        window_labels = labels[:, torch.cat([indices, current])]
        x = torch.randn((1, channels, chunk_size, height, width), device=device,
                        dtype=dtype, generator=generator)
        for i, sigma in enumerate(schedule[:-1]):
            full = torch.cat([history, x], dim=2) if n_history else x
            v = velocity(model, full, sigma, context, window_labels, n_history)[:, :, n_history:]
            x0 = (x.float() - sigma * v.float()).to(dtype)
            if i < len(schedule) - 2:
                noise = torch.randn(x0.shape, device=device, dtype=dtype, generator=generator)
                x = ((1 - schedule[i + 1]) * x0 + schedule[i + 1] * noise).to(dtype)
        clean = x0 if clean is None else torch.cat([clean, x0], dim=2)
    return clean


@torch.no_grad()
def sample_stage1(model, context, labels, *, latent_shape, num_latent_frames,
                  negative_context, num_steps=50, guidance_scale=5.,
                  sigma_shift=5., generator=None):
    if num_steps < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("Invalid stage-1 steps or guidance")
    shifted_sigmas([1., 0.], sigma_shift)  # Validate without altering upstream's <=1 policy.
    schedule = torch.linspace(1, 0, num_steps + 1, device=context.device)
    if sigma_shift > 1:
        schedule = sigma_shift * schedule / (1 + (sigma_shift - 1) * schedule)
    schedule = schedule.tolist()
    channels, height, width = latent_shape
    x = torch.randn((1, channels, num_latent_frames, height, width),
                    device=context.device, dtype=context.dtype, generator=generator)
    for sigma, next_sigma in zip(schedule, schedule[1:]):
        v = velocity(model, x, sigma, context, labels)
        if guidance_scale > 1:
            uncond = velocity(model, x, sigma, negative_context, None)
            v = uncond + guidance_scale * (v - uncond)
        x = (x.float() + v.float() * (next_sigma - sigma)).to(context.dtype)
    return x

# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hansheng Chen.
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3:
#   lakonlab/models/diffusions/schedulers/flow_euler_ode.py
#   lakonlab/pipelines/pipeline_minimax_h3.py
"""pi-H3 FlowEulerODE sigma schedule and exact first-order ODE update."""

from __future__ import annotations

import math

import numpy as np
import torch

PI_H3_MAX_RAW_T: dict[int, float] = {2: 0.93, 4: 0.96}
PI_H3_DEFAULT_VIDEO_SHIFT: float = 12.0
PI_H3_DEFAULT_AUDIO_SHIFT: float = 3.0
PI_H3_DEFAULT_FINAL_STEP_SIZE_SCALE: float = 0.6


def pi_h3_flow_euler_sigmas(
    num_steps: int,
    *,
    shift: float,
    max_raw_t: float | None = None,
    min_raw_t: float = 0.0,
    final_step_size_scale: float = PI_H3_DEFAULT_FINAL_STEP_SIZE_SCALE,
) -> list[float]:
    """Compute LakonLab's ``FlowEulerODEScheduler`` sigma schedule with terminal ``0.0``.

    Matches ``FlowEulerODEScheduler.set_timesteps`` in float32:

    1. ``raw_end = (max_raw_t - min_raw_t) * final_step_size_scale / (num_steps - 1 + final_step_size_scale) + min_raw_t``
    2. ``raw = np.linspace(max_raw_t, raw_end, num_steps, dtype=np.float32)``
    3. ``sigmas = shift * raw / (1 + (shift - 1) * raw)``
    4. Append terminal ``0.0`` (returning ``num_steps + 1`` entries).
    """

    if isinstance(num_steps, bool) or int(num_steps) <= 0:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}.")
    steps = int(num_steps)
    if max_raw_t is None:
        if steps not in PI_H3_MAX_RAW_T:
            raise ValueError(
                f"pi-H3 default max_raw_t is defined for {sorted(PI_H3_MAX_RAW_T)} steps; "
                f"got {steps} without an explicit max_raw_t."
            )
        resolved_max_raw_t = PI_H3_MAX_RAW_T[steps]
    else:
        resolved_max_raw_t = float(max_raw_t)

    resolved_min_raw_t = float(min_raw_t)
    resolved_shift = float(shift)
    resolved_scale = float(final_step_size_scale)
    if not math.isfinite(resolved_shift) or resolved_shift <= 0.0:
        raise ValueError(f"shift must be finite and positive, got {shift!r}.")
    if not (0.0 <= resolved_min_raw_t < resolved_max_raw_t <= 1.0):
        raise ValueError(
            f"Expected 0.0 <= min_raw_t < max_raw_t <= 1.0, got min_raw_t={resolved_min_raw_t}, "
            f"max_raw_t={resolved_max_raw_t}."
        )
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError(f"final_step_size_scale must be finite and positive, got {final_step_size_scale!r}.")

    raw_end = (resolved_max_raw_t - resolved_min_raw_t) * resolved_scale / (
        steps - 1 + resolved_scale
    ) + resolved_min_raw_t
    raw_sigmas = np.linspace(resolved_max_raw_t, raw_end, steps, dtype=np.float32)
    sigmas = torch.from_numpy(raw_sigmas)
    sigmas = resolved_shift * sigmas / (1.0 + (resolved_shift - 1.0) * sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)])
    return [float(value) for value in sigmas.tolist()]


def pi_h3_flow_euler_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
    prediction_type: str = "v",
    use_fp64: bool = False,
) -> torch.Tensor:
    """Advance one first-order FlowEulerODE step matching LakonLab.

    Sign convention:
      * ``prediction_type="v"``: ``model_output`` is the raw MiniMax-H3 transformer
        velocity ``v`` (as returned by :class:`PiH3DiTModel` and canonical H3).
        LakonLab negates it to ``u = -v`` before stepping along ``dt = sigma_next - sigma``,
        giving ``sample + (-v) * (sigma_next - sigma) = sample + v * (sigma - sigma_next)``.
      * ``prediction_type="u"``: ``model_output`` is already LakonLab's negated
        ``noise_pred = -v``, giving ``sample + u * (sigma_next - sigma)``.
      * ``prediction_type="x0"``: ``model_output`` is clean ``x0``, giving
        ``derivative = (sample - x0) / sigma`` and ``sample + derivative * (sigma_next - sigma)``.
    """

    if prediction_type not in ("v", "u", "x0"):
        raise ValueError(f"prediction_type must be one of ('v', 'u', 'x0'), got {prediction_type!r}.")
    if sample.shape != model_output.shape:
        raise ValueError(f"sample and model_output shapes must match, got {sample.shape} vs {model_output.shape}.")
    sigma_val = float(sigma)
    sigma_next_val = float(sigma_next)
    if not (math.isfinite(sigma_val) and math.isfinite(sigma_next_val)) or sigma_val < 0.0 or sigma_next_val < 0.0:
        raise ValueError(f"sigma and sigma_next must be finite and non-negative, got {sigma}, {sigma_next}.")
    if sigma_val == 0.0:
        return sample

    ori_dtype = sample.dtype
    solver_dtype = torch.float64 if use_fp64 else torch.float32
    x = sample.to(solver_dtype)
    out = model_output.to(solver_dtype)
    sigma_t = x.new_tensor(sigma_val, dtype=solver_dtype)
    sigma_next_t = x.new_tensor(sigma_next_val, dtype=solver_dtype)

    if prediction_type == "v":
        derivative = -out
    elif prediction_type == "u":
        derivative = out
    else:
        derivative = (x - out) / sigma_t

    dt = sigma_next_t - sigma_t
    prev_sample = x + derivative * dt
    return prev_sample.to(ori_dtype)


__all__ = [
    "PI_H3_DEFAULT_AUDIO_SHIFT",
    "PI_H3_DEFAULT_FINAL_STEP_SIZE_SCALE",
    "PI_H3_DEFAULT_VIDEO_SHIFT",
    "PI_H3_MAX_RAW_T",
    "pi_h3_flow_euler_sigmas",
    "pi_h3_flow_euler_step",
]

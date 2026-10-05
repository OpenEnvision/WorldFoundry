"""MiniMax H3 rectified-flow Euler-ancestral (eta=0) numerics.

Ported from SGLang's Apache-2.0
``scheduling_minimax_h3_euler_ancestral`` and the frame/latent + sigma helpers
in the H3 ``time_request`` stage. H3 drives video and audio with *independent*
sigma schedules inside a single coupled denoise loop, so these are exposed as
pure functions (plus a thin stateless adapter) rather than the framework
``DiffusionScheduler`` protocol — the bespoke H3 pipeline calls them directly.

Contract, kept identical to the reference:

* rectified flow, ``sigma_t = 1 - t``; the model predicts velocity ``v`` and
  ``x0 = xt + (1 - t) * v``.
* Euler eta=0 update ``out = ratio * state + (1 - ratio) * denoised`` where
  ``ratio = sigma_next / sigma_curr`` computed in fp32 for reduced-precision
  states, returned in the state dtype. ``sigma_curr == 0`` returns the state.
* the sigma schedule is a fixed ``[1, 0]`` linspace over ``num_steps``,
  time-shifted by ``shift_scale * s / (1 + (shift_scale - 1) * s)``, deduped,
  with a terminal ``0`` appended (video shift 12, audio shift 3).
"""

from __future__ import annotations

import math
from typing import Any

import torch


# --------------------------------------------------------------------------- #
# Validation helpers.
# --------------------------------------------------------------------------- #
def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_unit_timestep(timestep: torch.Tensor, name: str) -> None:
    if not isinstance(timestep, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(timestep):
        raise ValueError(f"{name} must be a floating point tensor")
    _require_finite_tensor(timestep, name)
    if bool(((timestep < 0) | (timestep > 1)).any().item()):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def _validate_timestep_sigma_pair(timestep: torch.Tensor, sigma_curr: float, name: str) -> float:
    _validate_unit_timestep(timestep, f"{name}_timestep")
    sigma = _validate_sigma(sigma_curr, f"{name}_sigma_curr")
    expected = 1.0 - timestep.detach().to(dtype=torch.float32)
    actual = torch.full_like(expected, sigma)
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{name}_sigma_curr must equal 1 - {name}_timestep")
    return sigma


# --------------------------------------------------------------------------- #
# Rectified-flow velocity -> x0 and the Euler eta=0 update.
# --------------------------------------------------------------------------- #
def _minimax_h3_rf_v_to_x0(xt: torch.Tensor, v: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    return xt + sigma_t * v


def minimax_h3_rf_v_to_x0(xt: torch.Tensor, v: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Convert a velocity prediction to a clean ``x0`` estimate."""

    if xt.shape != v.shape:
        raise ValueError(f"xt and v shapes must match, got {xt.shape} vs {v.shape}")
    if not torch.is_floating_point(xt):
        raise ValueError("xt must be a floating point tensor")
    if not torch.is_floating_point(v):
        raise ValueError("v must be a floating point tensor")
    _require_finite_tensor(xt, "xt")
    _require_finite_tensor(v, "v")
    _validate_unit_timestep(timestep, "timestep")
    x0 = _minimax_h3_rf_v_to_x0(xt, v, timestep)
    _require_finite_tensor(x0, "x0")
    return x0


def _minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
    sigma_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    if sigma_curr == 0.0:
        return state
    compute_dtype = torch.float32
    if state.dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = state.dtype
    if sigma_ratio is None:
        sigma_curr_t = state.new_tensor(sigma_curr, dtype=compute_dtype)
        sigma_next_t = state.new_tensor(sigma_next, dtype=compute_dtype)
        ratio = sigma_next_t / sigma_curr_t
    else:
        ratio = sigma_ratio.to(device=state.device, dtype=compute_dtype)
    out = ratio * state.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
    return out.to(dtype=state.dtype)


def minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """Advance one rectified-flow Euler eta=0 step."""

    if state.shape != denoised.shape:
        raise ValueError(f"state and denoised shapes must match, got {state.shape} vs {denoised.shape}")
    if not torch.is_floating_point(state):
        raise ValueError("state must be a floating point tensor")
    if not torch.is_floating_point(denoised):
        raise ValueError("denoised must be a floating point tensor")
    _require_finite_tensor(state, "state")
    _require_finite_tensor(denoised, "denoised")
    sigma_curr = _validate_sigma(sigma_curr, "sigma_curr")
    sigma_next = _validate_sigma(sigma_next, "sigma_next")
    if sigma_curr == 0.0 and sigma_next != 0.0:
        raise ValueError("sigma_next must be 0 when sigma_curr is 0")
    out = _minimax_h3_euler_eta0_step(state, denoised, sigma_curr=sigma_curr, sigma_next=sigma_next)
    _require_finite_tensor(out, "euler_eta0_step output")
    return out


# --------------------------------------------------------------------------- #
# Frame / latent-T arithmetic (17n+5 frames, 5n+2 video latent T, 40 Hz audio).
# --------------------------------------------------------------------------- #
def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""

    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    """Video latent temporal length for an aligned ``frame_count`` (5n+2)."""

    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    """Inverse of :func:`minimax_h3_video_latent_t`."""

    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    """Audio latent temporal length at the 40 Hz latent boundary."""

    return int(round(float(duration_seconds) * 40.0))


# --------------------------------------------------------------------------- #
# Time-shifted sigma schedule.
# --------------------------------------------------------------------------- #
def minimax_h3_time_shift_sigmas(*, num_steps: int = 50, shift_scale: float = 6.0) -> list[float]:
    """Return the time-shifted rectified-flow sigma schedule with terminal 0."""

    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    base = torch.linspace(1.0, 0.0, int(num_steps), device="cpu", dtype=torch.float32)
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]


# --------------------------------------------------------------------------- #
# Stateless adapter mirroring the reference step_denoising surface.
# --------------------------------------------------------------------------- #
class MiniMaxH3EulerAncestralEta0Scheduler:
    """Stateless coupled video+audio rectified-flow Euler eta=0 stepper."""

    def __init__(self, **config: Any) -> None:
        if config:
            raise ValueError(f"{type(self).__name__} does not accept config fields: {sorted(config)}")

    def set_shift(self, _flow_shift: float) -> None:
        """No-op; per-modality shift is baked into the sigma schedule."""

    def step_denoising(
        self,
        *,
        input_visual_latent: torch.Tensor,
        input_audio_latent: torch.Tensor,
        timestep: torch.Tensor,
        noise_pred_visual: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        video_timestep: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        video_sigma_curr: float | None = None,
        video_sigma_next: float | None = None,
        audio_sigma_curr: float | None = None,
        audio_sigma_next: float | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_timestep = timestep if video_timestep is None else video_timestep
        audio_timestep = timestep if audio_timestep is None else audio_timestep
        visual_sigma_curr = sigma_curr if video_sigma_curr is None else video_sigma_curr
        visual_sigma_next = sigma_next if video_sigma_next is None else video_sigma_next
        audio_sigma_curr = sigma_curr if audio_sigma_curr is None else audio_sigma_curr
        audio_sigma_next = sigma_next if audio_sigma_next is None else audio_sigma_next
        visual_sigma_curr = _validate_timestep_sigma_pair(visual_timestep, visual_sigma_curr, "video")
        audio_sigma_curr = _validate_timestep_sigma_pair(audio_timestep, audio_sigma_curr, "audio")

        denoised_visual = minimax_h3_rf_v_to_x0(input_visual_latent, noise_pred_visual, visual_timestep)
        denoised_audio = minimax_h3_rf_v_to_x0(input_audio_latent, noise_pred_audio, audio_timestep)
        return {
            "output_visual_latent": minimax_h3_euler_eta0_step(
                input_visual_latent,
                denoised_visual,
                sigma_curr=visual_sigma_curr,
                sigma_next=visual_sigma_next,
            ),
            "output_audio_latent": minimax_h3_euler_eta0_step(
                input_audio_latent,
                denoised_audio,
                sigma_curr=audio_sigma_curr,
                sigma_next=audio_sigma_next,
            ),
        }


def build_minimax_h3_euler_ancestral_scheduler() -> MiniMaxH3EulerAncestralEta0Scheduler:
    """Construct the stateless MiniMax H3 coupled scheduler."""

    return MiniMaxH3EulerAncestralEta0Scheduler()


__all__ = [
    "MiniMaxH3EulerAncestralEta0Scheduler",
    "build_minimax_h3_euler_ancestral_scheduler",
    "minimax_h3_align_frame_count",
    "minimax_h3_audio_latent_t",
    "minimax_h3_euler_eta0_step",
    "minimax_h3_frame_count_from_video_latent_t",
    "minimax_h3_rf_v_to_x0",
    "minimax_h3_time_shift_sigmas",
    "minimax_h3_video_latent_t",
]

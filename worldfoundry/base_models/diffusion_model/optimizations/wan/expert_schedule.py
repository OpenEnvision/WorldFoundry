"""Immutable global-to-local schedule for resident Wan2.2 expert caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _scalar_timestep(value: Any, *, allow_batch: bool = False) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError("expert schedule requires finite scalar tensor timesteps")
    values = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(values).all()) or (
        values.numel() != 1 and (not allow_batch or not bool((values == values[0]).all()))
    ):
        raise ValueError("expert schedule requires finite scalar tensor timesteps")
    return float(values[0])


@dataclass(frozen=True, slots=True)
class ExpertScheduleStep:
    global_index: int
    timestep: float
    next_timestep: float
    expert: str
    local_index: int
    local_count: int


@dataclass(frozen=True, slots=True)
class WanExpertSchedule:
    """Snapshot the whole schedule before deriving either expert's boundaries."""

    steps: tuple[ExpertScheduleStep, ...]

    @classmethod
    def build(cls, schedule, *, boundary_ratio: float, num_train_timesteps: int):
        raw = tuple(schedule)
        if not raw:
            raise ValueError("expert schedule cannot be empty")
        values = []
        for index, step in enumerate(raw):
            if type(step.index) is not int or step.index != index:
                raise ValueError("expert schedule indices must be contiguous and zero-based")
            timestep = _scalar_timestep(step.timestep)
            next_timestep = _scalar_timestep(step.next_timestep)
            if next_timestep > timestep:
                raise ValueError("expert schedule must decrease monotonically")
            high = bool(torch.tensor(timestep, dtype=torch.float32) / num_train_timesteps >= boundary_ratio)
            values.append((timestep, next_timestep, "high-noise" if high else "low-noise"))
        for previous, current in zip(values, values[1:]):
            if previous[0] < current[0] or previous[1] != current[0]:
                raise ValueError("expert schedule must decrease monotonically with matching next timesteps")
            if previous[2] == "low-noise" and current[2] == "high-noise":
                raise ValueError("expert schedule permits only one high-to-low transition")
        counts = {expert: sum(row[2] == expert for row in values) for expert in ("high-noise", "low-noise")}
        seen = {expert: 0 for expert in counts}
        steps = []
        for index, (timestep, next_timestep, expert) in enumerate(values):
            steps.append(ExpertScheduleStep(index, timestep, next_timestep, expert, seen[expert], counts[expert]))
            seen[expert] += 1
        return cls(tuple(steps))

    def select(self, model_input) -> ExpertScheduleStep:
        index = model_input.step_index
        if (
            type(index) is not int
            or not 0 <= index < len(self.steps)
            or type(model_input.total_steps) is not int
            or model_input.total_steps != len(self.steps)
        ):
            raise ValueError("denoiser input must match the prepared expert schedule")
        step = self.steps[index]
        if (
            _scalar_timestep(model_input.timestep, allow_batch=True) != step.timestep
            or _scalar_timestep(model_input.next_timestep, allow_batch=True) != step.next_timestep
        ):
            raise ValueError("denoiser timesteps changed after expert schedule preparation")
        return step

    def receipt(self) -> dict[str, object]:
        return {
            "global_steps": len(self.steps),
            "phases": [
                {
                    "expert": step.expert,
                    "global_start": step.global_index,
                    "global_end": step.global_index + step.local_count - 1,
                    "local_steps": step.local_count,
                }
                for step in self.steps
                if step.local_index == 0
            ],
        }

# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Map explicit trajectory phases to video frames using the reference protocol."""

from dataclasses import dataclass

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

INVERSE_ACTION = {
    "yaw_left": "yaw_right",
    "yaw_right": "yaw_left",
    "pitch_up": "pitch_down",
    "pitch_down": "pitch_up",
    "move_left": "move_right",
    "move_right": "move_left",
    "move_forward": "move_backward",
    "move_backward": "move_forward",
}


@dataclass(frozen=True)
class PhaseWindow:
    phase_id: int
    action: str
    axis: str
    start: int
    end: int

    @property
    def indices(self) -> range:
        return range(self.start, self.end + 1)


@dataclass(frozen=True)
class RevisitSlot:
    slot_id: int
    outbound_phase: int
    return_phase: int
    axis: str


@dataclass(frozen=True)
class EnvironmentProtocol:
    condition_id: str
    total_frames: int
    actions: tuple[str, ...]
    phase_transitions: tuple[int, ...]
    tail_hold_transitions: int
    phases: tuple[PhaseWindow, ...]
    slots: tuple[RevisitSlot, ...]


def protocol_fields(value: dict) -> dict:
    """Read the three scoring fields; do not infer timing from a directory name."""
    if not isinstance(value, dict):
        raise DataInvalidError("Environment trajectory must be a JSON object")
    actions = value.get("actions")
    transitions = value.get("phase_transitions")
    tail = value.get("tail_hold_transitions", 0)
    if (
        not isinstance(actions, list)
        or not actions
        or len(actions) % 2
        or any(not isinstance(action, str) or action not in INVERSE_ACTION for action in actions)
    ):
        raise DataInvalidError("trajectory.actions must contain supported outbound/return action pairs")
    if any(INVERSE_ACTION[actions[i]] != actions[i + 1] for i in range(0, len(actions), 2)):
        raise DataInvalidError("each Environment action pair must be inverse")
    if (
        not isinstance(transitions, list)
        or len(transitions) != len(actions)
        or any(type(count) is not int or count <= 0 for count in transitions)
    ):
        raise DataInvalidError("trajectory.phase_transitions must contain one positive integer per action")
    if type(tail) is not int or tail < 0:
        raise DataInvalidError("trajectory.tail_hold_transitions must be a non-negative integer")
    return {"actions": list(actions), "phase_transitions": list(transitions), "tail_hold_transitions": tail}


def protocol_for_sample(sample: dict, *, total_frames: int) -> EnvironmentProtocol:
    fields = protocol_fields((sample.get("annotations") or {}).get("environment_protocol"))
    if total_frames < 3:
        raise DataInvalidError("Environment video requires at least three frames")
    actions = tuple(fields["actions"])
    transitions = tuple(fields["phase_transitions"])
    tail = fields["tail_hold_transitions"]
    source_last = sum(transitions) + tail
    phases = []
    cursor = 0
    for index, (action, count) in enumerate(zip(actions, transitions)):
        start = int(round(cursor * (total_frames - 1) / source_last))
        cursor += count
        end = int(round(cursor * (total_frames - 1) / source_last))
        if end <= start:
            raise DataInvalidError(f"phase {index} collapses in decoded frame space")
        if action in ("move_left", "move_right"):
            axis = "lateral"
        elif action in ("move_forward", "move_backward"):
            axis = "longitudinal"
        else:
            axis = action.split("_", 1)[0]
        phases.append(PhaseWindow(index, action, axis, start, end))
    slots = tuple(RevisitSlot(i // 2 + 1, i, i + 1, phases[i].axis) for i in range(0, len(phases), 2))
    return EnvironmentProtocol(
        str(sample.get("condition_id") or ""), total_frames, actions, transitions, tail, tuple(phases), slots
    )

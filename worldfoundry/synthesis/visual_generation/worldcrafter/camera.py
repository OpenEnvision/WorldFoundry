# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Camera actions in a right/down/forward coordinate system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from functools import partial

import numpy as np

from worldfoundry.core.io.filesystem.file_utils import file_sha256


CHUNK_FRAMES = 33
FPS = 16
MAX_TRANSLATION = 5.0
ACTION_FIELDS = {
    "forward": ("forward", 1),
    "backward": ("forward", -1),
    "left": ("right", -1),
    "right": ("right", 1),
    "up": ("up", 1),
    "down": ("up", -1),
    "yaw_left": ("yaw", -1),
    "yaw_right": ("yaw", 1),
    "pitch_up": ("pitch", 1),
    "pitch_down": ("pitch", -1),
}
ALIASES = {
    "f": "forward", "b": "backward", "l": "left", "r": "right",
    "yl": "yaw_left", "yr": "yaw_right", "pu": "pitch_up", "pd": "pitch_down",
}


@dataclass(frozen=True)
class Action:
    forward: float = 0.0
    right: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    speed: float = 1.0
    up: float = 0.0

    def validate(self):
        values = (self.forward, self.right, self.up, self.yaw, self.pitch)
        if not all(math.isfinite(v) for v in (*values, self.speed)):
            raise ValueError("Control values must be finite")
        if sum(v != 0 for v in values) > 1:
            raise ValueError("Only one movement or rotation may be active per chunk")

    def normalized(self):
        """Apply the interactive controls' slider limits."""
        self.validate()
        return Action(
            forward=max(-1.0, min(1.0, self.forward)),
            right=max(-1.0, min(1.0, self.right)),
            up=max(-1.0, min(1.0, self.up)),
            yaw=max(-30.0, min(30.0, self.yaw)),
            pitch=max(-30.0, min(30.0, self.pitch)),
            speed=max(0.1, min(MAX_TRANSLATION, self.speed)),
        )

    def json(self):
        return asdict(self)


def parse_event(event: str) -> tuple[str, float]:
    match = re.fullmatch(r"([a-z_]+)([0-9]+(?:\.[0-9]+)?)", event.lower())
    if match is None:
        raise ValueError(f"Invalid action {event!r}; use e.g. forward1 or yaw_left30")
    name, value = match.groups()
    name = ALIASES.get(name, name)
    if name not in ACTION_FIELDS:
        raise ValueError(f"Unknown action {name!r}; choose from {', '.join(ACTION_FIELDS)}")
    amount = float(value)
    if not math.isfinite(amount):
        raise ValueError(f"Action amount must be finite: {event}")
    if ACTION_FIELDS[name][0] in {"forward", "right", "up"} and amount > MAX_TRANSLATION:
        raise ValueError(f"{event}: translation must not exceed {MAX_TRANSLATION:g} per chunk")
    return name, amount


def parse_actions(text: str) -> list[str]:
    """Expand space/comma-separated actions, xN repetitions, and # comments."""
    text = re.sub(r"#[^\n]*", "", text)
    text = re.sub(r"\s*&\s*", "&", text)
    events = []
    for token in re.split(r"[\s,]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(.+?)(?:x([1-9][0-9]*))?", token.lower())
        event, repeat = match.groups()
        if re.fullmatch(r"reverse(?:_frames)?[1-9][0-9]*", event):
            canonical = event
        else:
            parts = []
            axes = set()
            for component in event.split("&"):
                name, _ = parse_event(component)
                field = ACTION_FIELDS[name][0]
                if field in axes:
                    raise ValueError(f"An action may use each axis only once: {event}")
                axes.add(field)
                value = re.search(r"[0-9].*", component).group()
                parts.append(name + value)
            canonical = "&".join(parts)
        events.extend([canonical] * int(repeat or 1))
    if not events:
        raise ValueError("Provide at least one camera action")
    return events


def parse_trajectory(text: str) -> tuple[list[str], dict[str, str]]:
    """Read actions and optional @dtype, @sampling, and @last_frame headers."""
    choices = {
        "dtype": {"float32", "float64"},
        "sampling": {"linear", "smooth_turns"},
        "last_frame": {"exclude", "include"},
    }
    options, lines = {}, []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("@"):
            fields = line[1:].split()
            if len(fields) != 2 or fields[0] not in choices or fields[1] not in choices[fields[0]]:
                raise ValueError(f"Invalid trajectory setting: {line}")
            if lines:
                raise ValueError("Trajectory settings must precede the actions")
            options[fields[0]] = fields[1]
        elif line:
            lines.append(line)
    return parse_actions("\n".join(lines)), options


def count_chunks(events: list[str]) -> int:
    return sum(
        int(re.search(r"[0-9]+$", event).group()) if event.startswith("reverse") else 1
        for event in events
    )


def action_from_event(event: str) -> Action:
    name, amount = parse_event(event)
    field, sign = ACTION_FIELDS[name]
    if field in {"yaw", "pitch"}:
        return Action(**{field: sign * amount})
    return Action(**{field: sign}, speed=amount)


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=np.float64,
    )


def horizontal_direction(rotation: np.ndarray, forward: bool) -> np.ndarray:
    axis = rotation[:, 2 if forward else 0].copy()
    axis[1] = 0.0
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        # Keep a horizontal heading when looking straight up or down.
        other = rotation[:, 0 if forward else 2]
        axis = np.array([-other[2], 0.0, other[0]])
        if not forward:
            axis = -axis
        norm = np.linalg.norm(axis)
    return axis / norm


def sample_chunk(
    start: np.ndarray, action: Action, *, fractions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one action; the logical endpoint starts the next chunk."""
    action.validate()
    alpha = np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES if fractions is None else fractions
    poses = np.repeat(start[None], len(alpha), axis=0)
    end = start.copy()
    if action.yaw or action.pitch:
        def rotated(fraction):
            if action.yaw:
                return rotation_y(action.yaw * fraction) @ start[:3, :3]
            return start[:3, :3] @ rotation_x(action.pitch * fraction)

        for index, fraction in enumerate(alpha):
            poses[index, :3, :3] = rotated(fraction)
        end[:3, :3] = rotated(1.0)
    else:
        if action.up:
            direction = np.array([0.0, -action.up, 0.0])
        elif action.forward:
            direction = action.forward * horizontal_direction(start[:3, :3], True)
        else:
            direction = action.right * horizontal_direction(start[:3, :3], False)
        delta = action.speed * direction
        if np.linalg.norm(delta) > MAX_TRANSLATION + 1e-12:
            raise ValueError(f"Translation must not exceed {MAX_TRANSLATION:g} per chunk")
        poses[:, :3, 3] = start[:3, 3] + alpha[:, None] * delta
        end[:3, 3] = start[:3, 3] + delta
    return poses, end


def _sample_event(start: np.ndarray, event: str, fractions: np.ndarray) -> np.ndarray:
    components = event.split("&")
    if len(components) == 1:
        return sample_chunk(start, action_from_event(event), fractions=fractions)[0]
    poses = np.repeat(start[None], len(fractions), axis=0)
    delta = np.zeros(3)
    for component in components:
        action = action_from_event(component)
        if action.yaw:
            for index, fraction in enumerate(fractions):
                poses[index, :3, :3] = rotation_y(action.yaw * fraction) @ poses[index, :3, :3]
        elif action.pitch:
            for index, fraction in enumerate(fractions):
                poses[index, :3, :3] = poses[index, :3, :3] @ rotation_x(action.pitch * fraction)
        else:
            _, end = sample_chunk(start, action)
            delta += end[:3, 3] - start[:3, 3]
    if np.linalg.norm(delta) > MAX_TRANSLATION + 1e-12:
        raise ValueError(f"{event}: combined translation must not exceed {MAX_TRANSLATION:g} per chunk")
    poses[:, :3, 3] = start[:3, 3] + fractions[:, None] * delta
    return poses


def _sample_curve(start, event, tangent_start, tangent_end, times):
    fractions = times
    if tangent_start is not None:
        fractions = (
            -2 * times**3 + 3 * times**2
            + (times**3 - 2 * times**2 + times) * tangent_start
            + (times**3 - times**2) * tangent_end
        )
    return _sample_event(start, event, fractions)


def _reverse_curve(curve, times):
    return curve(1.0 - times)


def _reverse_sampled_curve(curve, last_time, times):
    return curve(last_time * (1.0 - times))


def build_trajectory(
    events: list[str], *, dtype: str = "float64", sampling: str = "linear",
    last_frame: str = "exclude",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not events:
        raise ValueError("Provide at least one camera action")
    world = np.eye(4, dtype=np.float64)
    chunks, records, curves, sample_times = [], [], [], []

    def append(event, curve, times, poses=None):
        nonlocal world
        start, end = curve(np.array([0.0, 1.0]))
        chunks.append(curve(times) if poses is None else poses)
        records.append({
            "chunk_index": len(records), "event": event,
            "logical_start_c2w": start.tolist(), "logical_end_c2w": end.tolist(),
        })
        curves.append(curve)
        sample_times.append(times)
        world = end

    for index, event in enumerate(events):
        if event.startswith("reverse"):
            count = int(re.search(r"[0-9]+$", event).group())
            if count > len(chunks):
                raise ValueError(f"{event} needs {count} preceding chunks; only {len(chunks)} exist")
            indices = list(range(len(chunks) - 1, len(chunks) - count - 1, -1))
            for source in indices:
                if event.startswith("reverse_frames"):
                    curve = partial(_reverse_sampled_curve, curves[source], sample_times[source][-1])
                    times = np.linspace(0.0, 1.0, CHUNK_FRAMES)
                    append(event, curve, times, chunks[source][::-1].copy())
                else:
                    curve = partial(_reverse_curve, curves[source])
                    include_end = last_frame == "include" and index == len(events) - 1 and source == indices[-1]
                    times = (
                        np.linspace(0.0, 1.0, CHUNK_FRAMES) if include_end
                        else np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES
                    )
                    poses = None
                    original = records[source]["event"]
                    if sampling == "linear" and "&" not in original and not original.startswith("reverse"):
                        action = action_from_event(original)
                        if not action.yaw and not action.pitch and sample_times[source][-1] < 1.0 and not include_end:
                            # Reuse linear translation samples without another interpolation roundoff.
                            endpoint = np.array(records[source]["logical_end_c2w"])
                            poses = np.concatenate([endpoint[None], chunks[source][1:][::-1]])
                    append(event, curve, times, poses)
            continue
        smooth = sampling == "smooth_turns"
        entering = index > 0 and events[index - 1] == event
        leaving = index + 1 < len(events) and events[index + 1] == event
        include_end = (smooth and not leaving) or (last_frame == "include" and index == len(events) - 1)
        times = (
            np.linspace(0.0, 1.0, CHUNK_FRAMES) if include_end
            else np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES
        )
        curve = partial(
            _sample_curve, world.copy(), event,
            float(entering) if smooth else None, float(leaving),
        )
        append(event, curve, times)
    return np.concatenate(chunks).astype(dtype), records


def save_trajectory(
    directory: Path, camera: np.ndarray, records: list[dict[str, object]], *, fps: int = FPS,
    events: list[str] | None = None, options: dict[str, str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    camera_path = directory / "camera.npy"
    np.save(camera_path, camera[:, :3, :4])
    events = events if events is not None else [record["event"] for record in records]
    headers = [f"@{key} {value}" for key, value in (options or {}).items()]
    (directory / "actions.txt").write_text("\n".join(headers + events) + "\n", encoding="utf-8")
    manifest = {
        "format": "worldcrafter_camera_trajectory_v1",
        "fps": fps,
        "chunk_frames": CHUNK_FRAMES,
        "num_chunks": len(records),
        "num_frames": len(camera),
        "motion_sequence": events,
        "options": options or {},
        "camera": camera_path.name,
        "camera_semantics": "global metric c2w; x right, y down, z forward",
        "sha256": {"camera": file_sha256(camera_path)},
        "chunks": records,
    }
    (directory / "trajectory.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return camera_path

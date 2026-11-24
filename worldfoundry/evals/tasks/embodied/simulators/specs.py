"""Action and observation interface specs for embodied simulators.

The simulator wrappers declare what they consume and produce through
``get_action_spec()`` and ``get_observation_spec()``.  Policy runtimes can use
these specs to catch common convention mismatches before a full rollout, such
as Euler-vs-axis-angle rotation, inverted gripper signs, or missing robot state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DimSpec:
    """One component of an embodied action or observation contract."""

    name: str
    dims: int
    format: str
    range: tuple[float, float] | None = None
    accepts: frozenset[str] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Coerce raw initialization parameters into final, immutable and validated formats."""
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "dims", int(self.dims))
        object.__setattr__(self, "format", str(self.format))
        if self.range is not None:
            lo, hi = self.range
            object.__setattr__(self, "range", (float(lo), float(hi)))
        if self.accepts is not None and not isinstance(self.accepts, frozenset):
            object.__setattr__(self, "accepts", frozenset(str(item) for item in self.accepts))

    def validate(self, value: Any) -> list[str]:
        """Return validation errors for ``value`` against this component spec.

        Args:
            value: The numeric array-like value to validate.

        Returns:
            A list of string descriptions detailing validation mismatches.
        """

        errors: list[str] = []
        if self.dims <= 0:
            return errors
        try:
            flat = np.asarray(value, dtype=np.float64).flatten()
        except (TypeError, ValueError) as exc:
            return [f"{self.name}: cannot coerce value to numeric array ({exc})"]
        if len(flat) < self.dims:
            errors.append(f"{self.name}: expected at least {self.dims}D, got {len(flat)}D")
        if self.range is not None:
            lo, hi = self.range
            chunk = flat[: self.dims]
            if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
                errors.append(f"{self.name}: contains NaN/Inf")
            elif np.any(chunk < lo - 0.01) or np.any(chunk > hi + 0.01):
                errors.append(f"{self.name}: values outside [{lo}, {hi}]")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert this DimSpec instance into a plain serializable dictionary.

        Returns:
            A dictionary containing serialized DimSpec configuration fields.
        """
        payload: dict[str, Any] = {"name": self.name, "dims": self.dims, "format": self.format}
        if self.range is not None:
            payload["range"] = [self.range[0], self.range[1]]
        if self.accepts is not None:
            payload["accepts"] = sorted(self.accepts)
        if self.description:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DimSpec:
        """Instantiate a DimSpec instance from a serializable dictionary mapping.

        Args:
            payload: A dictionary mapping of initialization fields.

        Returns:
            A new DimSpec instance.
        """
        return cls(
            name=str(payload["name"]),
            dims=int(payload["dims"]),
            format=str(payload["format"]),
            range=tuple(payload["range"]) if payload.get("range") is not None else None,
            accepts=frozenset(str(item) for item in payload["accepts"]) if payload.get("accepts") else None,
            description=str(payload.get("description") or ""),
        )

    def is_compatible(self, consumer: DimSpec) -> tuple[bool, str]:
        """Check whether this producer spec can be safely consumed by ``consumer``.

        Args:
            consumer: The DimSpec target representing the consumer side.

        Returns:
            A tuple (compatible_bool, failure_reason_str).
        """

        if consumer.accepts is not None:
            if self.format not in consumer.accepts:
                return False, f"{self.name}: {self.format} not in accepts {set(consumer.accepts)}"
            return True, ""
        if self.format != consumer.format:
            return False, f"{self.name}: {self.format} vs {consumer.format}"
        if self.dims != consumer.dims and self.dims > 0 and consumer.dims > 0:
            return False, f"{self.name}: {self.dims}D vs {consumer.dims}D"
        return True, ""


def _coerce_spec_map(specs: Mapping[str, DimSpec | Mapping[str, Any]]) -> dict[str, DimSpec]:
    """Helper to safely coerce a map of string keys to DimSpec instances or raw dictionary specs.

    Args:
        specs: Input map representing dimensions.

    Returns:
        A mapping of string keys to DimSpec instances.
    """
    result: dict[str, DimSpec] = {}
    for key, value in specs.items():
        result[str(key)] = value if isinstance(value, DimSpec) else DimSpec.from_dict(value)
    return result


def check_specs(
    producer_action: Mapping[str, DimSpec | Mapping[str, Any]],
    consumer_action: Mapping[str, DimSpec | Mapping[str, Any]],
    producer_observation: Mapping[str, DimSpec | Mapping[str, Any]],
    consumer_observation: Mapping[str, DimSpec | Mapping[str, Any]],
) -> list[str]:
    """Compare policy and simulator specs and return mismatch descriptions."""

    server_action = _coerce_spec_map(producer_action)
    bench_action = _coerce_spec_map(consumer_action)
    bench_obs = _coerce_spec_map(producer_observation)
    server_obs = _coerce_spec_map(consumer_observation)
    warnings: list[str] = []

    if server_action and bench_action and not (server_action.keys() & bench_action.keys()):
        warnings.append("action: no overlapping keys between policy and simulator specs")
    for key in bench_action:
        if key not in server_action and server_action:
            warnings.append(f"action [{key}]: simulator expects it but policy does not declare it")
    for key in server_action.keys() & bench_action.keys():
        ok, reason = server_action[key].is_compatible(bench_action[key])
        if not ok:
            warnings.append(f"action [{key}]: {reason}")

    for key in server_obs:
        if key not in bench_obs:
            warnings.append(f"observation [{key}]: policy expects it but simulator does not provide it")
    for key in server_obs.keys() & bench_obs.keys():
        ok, reason = bench_obs[key].is_compatible(server_obs[key])
        if not ok:
            warnings.append(f"observation [{key}]: {reason}")
    return warnings


# Position
POSITION_DELTA = DimSpec("position", 3, "delta_xyz", (-1, 1))
POSITION_ABSOLUTE = DimSpec("position", 3, "absolute_xyz")

# Rotation
ROTATION_EULER = DimSpec("rotation", 3, "euler_xyz", (-3.15, 3.15))
ROTATION_AA = DimSpec("rotation", 3, "axis_angle", (-3.15, 3.15))
ROTATION_QUAT = DimSpec("rotation", 4, "quaternion_xyzw", (-1, 1))
ROTATION_ROT6D_INTERLEAVED = DimSpec("rotation", 6, "rot6d_interleaved")
ROTATION_EULER_ACCEPTS_AA = DimSpec(
    "rotation",
    3,
    "euler_xyz",
    (-3.15, 3.15),
    accepts=frozenset({"euler_xyz", "axis_angle"}),
)

# Gripper
GRIPPER_CLOSE_POS = DimSpec("gripper", 1, "binary_close_positive", (-1, 1))
GRIPPER_CLOSE_NEG = DimSpec("gripper", 1, "binary_close_negative", (-1, 1))
GRIPPER_01 = DimSpec("gripper", 1, "continuous_01", (0, 1))
GRIPPER_RAW = DimSpec("gripper", 1, "raw")

# Observation
IMAGE_RGB = DimSpec("image", 0, "rgb_hwc_uint8")
STATE_EEF_POS_QUAT_GRIP = DimSpec("state", 8, "eef_pos3_quat4_gripper1")
STATE_EEF_POS_AA_GRIP = DimSpec("state", 8, "eef_pos3_axisangle3_gripper2")
STATE_EEF_POS_EULER_GRIP = DimSpec("state", 8, "eef_pos3_euler3_gripper2")
STATE_ROT6D_PROPRIO_20D = DimSpec("state", 20, "rot6d_interleaved_proprio_20d")
STATE_JOINT = DimSpec("state", 0, "joint_positions")
LANGUAGE = DimSpec("language", 0, "language")
RAW = DimSpec("raw", 0, "raw")


__all__ = [
    "GRIPPER_01",
    "GRIPPER_CLOSE_NEG",
    "GRIPPER_CLOSE_POS",
    "GRIPPER_RAW",
    "IMAGE_RGB",
    "LANGUAGE",
    "POSITION_ABSOLUTE",
    "POSITION_DELTA",
    "RAW",
    "ROTATION_AA",
    "ROTATION_EULER",
    "ROTATION_EULER_ACCEPTS_AA",
    "ROTATION_QUAT",
    "ROTATION_ROT6D_INTERLEAVED",
    "STATE_EEF_POS_AA_GRIP",
    "STATE_EEF_POS_EULER_GRIP",
    "STATE_EEF_POS_QUAT_GRIP",
    "STATE_JOINT",
    "STATE_ROT6D_PROPRIO_20D",
    "DimSpec",
    "check_specs",
]

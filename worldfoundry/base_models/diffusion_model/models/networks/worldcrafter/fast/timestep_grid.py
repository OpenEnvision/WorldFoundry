# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Checkpoint timestep tables for Fast inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

DMD_TIMESTEP_CONTRACT_FILENAME = "dmd_timestep_contract.json"
DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION = 3
_LOCKED_LATENT_SHAPE = (16, 9, 48, 80)

_LOCKED_ROLLOUT_STEPS = (2, 2, 2)

_LOCKED_NORMAL_TIMESTEPS = (
    (998.5342, 833.9636),
    (742.8216, 547.1926),
    (385.4137, 253.9905),
)

_LOCKED_EMPTY_HISTORY_TIMESTEPS = (
    (998.5342, 902.2183, 833.9636, 783.0660),
    (742.8216, 640.0038, 547.1926, 462.9951),
    (385.4137, 328.6249, 253.9905, 151.5308),
)


def resolve_dmd_contract_path(checkpoint_path: str | Path) -> Path:
    """Return the canonical sidecar location for a checkpoint/run directory."""

    path = Path(checkpoint_path)
    if path.name == DMD_TIMESTEP_CONTRACT_FILENAME:
        return path
    if path.is_dir():
        return path / DMD_TIMESTEP_CONTRACT_FILENAME
    if path.suffix:
        return path.parent / DMD_TIMESTEP_CONTRACT_FILENAME
    return path / DMD_TIMESTEP_CONTRACT_FILENAME


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Invalid {name} keys: missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True)
class DmdTimestepStep:
    """One student query and the flow interpolation coefficients around it."""

    model_timestep: float
    current_sigma: float
    next_sigma: float

    def to_dict(self) -> dict[str, float]:
        return {
            "model_timestep": self.model_timestep,
            "current_sigma": self.current_sigma,
            "next_sigma": self.next_sigma,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DmdTimestepStep":
        _require_exact_keys(
            payload, {"model_timestep", "current_sigma", "next_sigma"}, "DMD step"
        )
        return cls(
            model_timestep=float(payload["model_timestep"]),
            current_sigma=float(payload["current_sigma"]),
            next_sigma=float(payload["next_sigma"]),
        )


@dataclass(frozen=True)
class DmdSchedulerConfig:
    """The fixed stage/band math from which the serialized student table came."""

    num_train_timesteps: int
    stages: int
    stage_range: tuple[float, ...]
    gamma: float
    shift: float
    version: str
    use_dynamic_shifting: bool = True
    time_shift_type: str = "linear"
    base_seq_len: int = 256
    max_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15

    def validate(self) -> None:
        if self.num_train_timesteps != 1000:
            raise ValueError(
                f"DMD requires 1000 scheduler timesteps, got {self.num_train_timesteps}"
            )
        if self.stages != 3:
            raise ValueError(
                f"Direct DMD requires three pyramid stages, got {self.stages}"
            )
        expected_range = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
        if len(self.stage_range) != len(expected_range) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(self.stage_range, expected_range)
        ):
            raise ValueError(f"Unsupported DMD stage range: {self.stage_range}")
        if not math.isclose(self.gamma, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unsupported DMD stage gamma: {self.gamma}")
        if not math.isclose(self.shift, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unsupported DMD stage shift: {self.shift}")
        if self.version != "v1":
            raise ValueError(f"Unsupported DMD stage scheduler version: {self.version}")
        if not self.use_dynamic_shifting or self.time_shift_type != "linear":
            raise ValueError(
                "Direct DMD student schedules require linear resolution-dependent shifting"
            )
        if (self.base_seq_len, self.max_seq_len) != (256, 4096):
            raise ValueError("Direct DMD student shift sequence-length anchors changed")
        if not math.isclose(self.base_shift, 0.5) or not math.isclose(
            self.max_shift, 1.15
        ):
            raise ValueError("Direct DMD student shift endpoints changed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "stages": self.stages,
            "stage_range": list(self.stage_range),
            "gamma": self.gamma,
            "shift": self.shift,
            "version": self.version,
            "use_dynamic_shifting": self.use_dynamic_shifting,
            "time_shift_type": self.time_shift_type,
            "base_seq_len": self.base_seq_len,
            "max_seq_len": self.max_seq_len,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DmdSchedulerConfig":
        expected = {
            "num_train_timesteps",
            "stages",
            "stage_range",
            "gamma",
            "shift",
            "version",
            "use_dynamic_shifting",
            "time_shift_type",
            "base_seq_len",
            "max_seq_len",
            "base_shift",
            "max_shift",
        }
        _require_exact_keys(payload, expected, "student scheduler provenance")
        provenance = cls(
            num_train_timesteps=int(payload["num_train_timesteps"]),
            stages=int(payload["stages"]),
            stage_range=tuple(float(value) for value in payload["stage_range"]),
            gamma=float(payload["gamma"]),
            shift=float(payload["shift"]),
            version=str(payload["version"]),
            use_dynamic_shifting=bool(payload["use_dynamic_shifting"]),
            time_shift_type=str(payload["time_shift_type"]),
            base_seq_len=int(payload["base_seq_len"]),
            max_seq_len=int(payload["max_seq_len"]),
            base_shift=float(payload["base_shift"]),
            max_shift=float(payload["max_shift"]),
        )
        provenance.validate()
        return provenance


@dataclass(frozen=True)
class DmdTimestepContract:
    schema_version: int
    latent_shape: tuple[int, int, int, int]
    rollout_steps_per_stage: tuple[int, int, int]
    amplify_empty_history: bool
    normal_stages: tuple[tuple[DmdTimestepStep, ...], ...]
    empty_history_stages: tuple[tuple[DmdTimestepStep, ...], ...]
    student_scheduler: DmdSchedulerConfig
    fingerprint: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.schema_version not in (2, DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported DMD timestep contract schema {self.schema_version}; "
                f"expected 2 or {DMD_TIMESTEP_CONTRACT_SCHEMA_VERSION}"
            )
        if len(self.latent_shape) != 4 or any(
            value <= 0 for value in self.latent_shape
        ):
            raise ValueError(f"Invalid DMD latent shape: {self.latent_shape}")
        if self.rollout_steps_per_stage != _LOCKED_ROLLOUT_STEPS:
            raise ValueError(
                f"Unsupported DMD rollout step counts {self.rollout_steps_per_stage}; "
                f"expected {_LOCKED_ROLLOUT_STEPS}"
            )
        self.student_scheduler.validate()
        if len(self.normal_stages) != 3 or len(self.empty_history_stages) != 3:
            raise ValueError(
                "DMD student schedule must contain exactly three pyramid stages"
            )
        expected_empty_multiplier = 2 if self.amplify_empty_history else 1
        for stage_index, (normal, empty) in enumerate(
            zip(self.normal_stages, self.empty_history_stages)
        ):
            expected_normal = self.rollout_steps_per_stage[stage_index]
            if len(normal) != expected_normal:
                raise ValueError(
                    f"DMD normal stage {stage_index} contains {len(normal)} steps, expected {expected_normal}"
                )
            if len(empty) != expected_normal * expected_empty_multiplier:
                raise ValueError(
                    f"DMD empty-history stage {stage_index} contains {len(empty)} steps, expected "
                    f"{expected_normal * expected_empty_multiplier}"
                )
            self._validate_stage(stage_index, normal, "normal")
            self._validate_stage(stage_index, empty, "empty_history")
        if (
            not self.amplify_empty_history
            and self.empty_history_stages != self.normal_stages
        ):
            raise ValueError(
                "Unamplified DMD contracts must use the normal table for empty history"
            )
        if self.latent_shape == _LOCKED_LATENT_SHAPE:
            self._validate_locked_timesteps(
                self.normal_stages, _LOCKED_NORMAL_TIMESTEPS, "normal"
            )
            if self.amplify_empty_history:
                self._validate_locked_timesteps(
                    self.empty_history_stages,
                    _LOCKED_EMPTY_HISTORY_TIMESTEPS,
                    "empty_history",
                )

    @staticmethod
    def _validate_stage(
        stage_index: int, steps: Sequence[DmdTimestepStep], name: str
    ) -> None:
        previous_timestep = float("inf")
        for index, step in enumerate(steps):
            values = (step.model_timestep, step.current_sigma, step.next_sigma)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Non-finite value in {name} stage {stage_index}, step {index}: {values}"
                )
            if not 0.0 <= step.next_sigma < step.current_sigma <= 1.0:
                raise ValueError(
                    f"Invalid sigma transition in {name} stage {stage_index}, step {index}: {values}"
                )
            if step.model_timestep >= previous_timestep:
                raise ValueError(
                    f"Timesteps are not strictly descending in {name} stage {stage_index}"
                )
            previous_timestep = step.model_timestep
            if index + 1 < len(steps) and not math.isclose(
                step.next_sigma,
                steps[index + 1].current_sigma,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"Broken x0 re-noise transition in {name} stage {stage_index}, step {index}"
                )
        if steps[-1].next_sigma != 0.0:
            raise ValueError(
                f"The final {name} transition in stage {stage_index} must end at clean x0"
            )

    @staticmethod
    def _validate_locked_timesteps(
        stages: Sequence[Sequence[DmdTimestepStep]],
        expected: Sequence[Sequence[float]],
        name: str,
    ) -> None:
        for stage_index, (actual_stage, expected_stage) in enumerate(
            zip(stages, expected)
        ):
            actual = [step.model_timestep for step in actual_stage]
            if len(actual) != len(expected_stage) or any(
                not math.isclose(value, target, rel_tol=0.0, abs_tol=1e-3)
                for value, target in zip(actual, expected_stage)
            ):
                raise ValueError(
                    f"Locked 384x640x9 DMD {name} schedule changed at stage {stage_index}: "
                    f"expected {list(expected_stage)}, got {actual}"
                )

    def stage(
        self, stage_index: int, *, empty_history: bool = False
    ) -> tuple[DmdTimestepStep, ...]:
        if stage_index not in range(3):
            raise IndexError(f"DMD pyramid stage must be 0, 1 or 2, got {stage_index}")
        return (self.empty_history_stages if empty_history else self.normal_stages)[
            stage_index
        ]

    def stage_tensors(
        self,
        stage_index: int,
        *,
        empty_history: bool = False,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = self.stage(stage_index, empty_history=empty_history)
        return {
            "model_timestep": torch.tensor(
                [step.model_timestep for step in steps],
                device=device,
                dtype=torch.float32,
            ),
            "current_sigma": torch.tensor(
                [step.current_sigma for step in steps],
                device=device,
                dtype=torch.float32,
            ),
            "next_sigma": torch.tensor(
                [step.next_sigma for step in steps], device=device, dtype=torch.float32
            ),
        }

    @staticmethod
    def student_condition(
        model_timestep: float | DmdTimestepStep | torch.Tensor,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(model_timestep, DmdTimestepStep):
            model_timestep = model_timestep.model_timestep
        condition = torch.as_tensor(model_timestep, dtype=torch.float32, device=device)
        if condition.ndim == 0 or condition.numel() == 1:
            return condition.reshape(1).expand(batch_size)
        if condition.shape != (batch_size,):
            raise ValueError(
                f"Student timestep condition must be scalar or shape ({batch_size},), got {tuple(condition.shape)}"
            )
        return condition

    @staticmethod
    def renoise_x0(
        x0: torch.Tensor,
        noise: torch.Tensor,
        next_sigma: float | DmdTimestepStep | torch.Tensor,
    ) -> torch.Tensor:
        """Apply the shared flow transition ``(1-sigma)*x0 + sigma*noise``."""

        if x0.shape != noise.shape:
            raise ValueError(
                f"x0/noise shapes differ: {tuple(x0.shape)} vs {tuple(noise.shape)}"
            )
        if isinstance(next_sigma, DmdTimestepStep):
            next_sigma = next_sigma.next_sigma
        sigma = torch.as_tensor(next_sigma, device=x0.device, dtype=torch.float32)
        if sigma.ndim == 0:
            pass
        elif sigma.ndim == 1 and sigma.shape[0] in (1, x0.shape[0]):
            sigma = sigma.reshape(sigma.shape[0], *([1] * (x0.ndim - 1)))
        else:
            raise ValueError(
                f"next_sigma must be scalar or one value per batch item; got shape {tuple(sigma.shape)}"
            )
        # The contract owns the numerical transition as well as its coefficients:
        # every caller keeps the rollout state in FP32 between model queries and
        # casts only the transformer's input. Returning ``x0.dtype`` here would
        # silently make a BF16 caller follow a different trajectory.
        return (1.0 - sigma) * x0.to(torch.float32) + sigma * noise.to(torch.float32)

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        expected_latent_shape=None,
        expected_fingerprint: str | None = None,
    ) -> "DmdTimestepContract":
        path = Path(path)
        source = (
            path if path.suffix.lower() == ".json" else resolve_dmd_contract_path(path)
        )
        document = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "fingerprint" not in document:
            raise ValueError(f"Missing DMD contract fingerprint: {source}")
        claimed = document.pop("fingerprint")
        actual = hashlib.sha256(_canonical_json(document)).hexdigest()
        if claimed != actual or (
            expected_fingerprint is not None and actual != expected_fingerprint
        ):
            raise ValueError(f"DMD contract fingerprint mismatch: {source}")
        # Hash the complete original document, including metadata unused by inference.
        student = document["student"]
        if student["condition_dtype"] != "float32":
            raise ValueError("DMD timestep conditions must use float32")
        contract = cls(
            schema_version=int(document["schema_version"]),
            latent_shape=tuple(int(x) for x in document["latent_shape"]),
            rollout_steps_per_stage=tuple(
                int(x) for x in student["rollout_steps_per_stage"]
            ),
            amplify_empty_history=bool(student["amplify_empty_history"]),
            normal_stages=tuple(
                tuple(DmdTimestepStep.from_dict(x) for x in stage)
                for stage in student["normal_stages"]
            ),
            empty_history_stages=tuple(
                tuple(DmdTimestepStep.from_dict(x) for x in stage)
                for stage in student["empty_history_stages"]
            ),
            student_scheduler=DmdSchedulerConfig.from_dict(
                student["scheduler_provenance"]
            ),
            fingerprint=actual,
        )
        if expected_latent_shape is not None and contract.latent_shape != tuple(
            expected_latent_shape
        ):
            raise ValueError(f"DMD latent shape mismatch: {contract.latent_shape}")
        return contract

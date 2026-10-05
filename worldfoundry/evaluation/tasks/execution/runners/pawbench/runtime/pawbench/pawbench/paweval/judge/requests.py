"""Judge request records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

AxisName = Literal["outcome_readout", "trustworthiness_audit"]

@dataclass(frozen=True)
class RowIdentity:
    sample_id: str
    scene_id: str
    model_or_lane: str
    repeat_index: int

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        model_or_lane: str | None = None,
        repeat_index: int | None = None,
    ) -> "RowIdentity":
        value = repeat_index if repeat_index is not None else row.get("repeat_index", row.get("repeat"))
        if not isinstance(value, int):
            raise ValueError("repeat_index must be an integer")
        return cls(
            sample_id=str(row.get("sample_id") or ""),
            scene_id=str(row.get("scene_id") or ""),
            model_or_lane=str(model_or_lane if model_or_lane is not None else row.get("model_or_lane") or ""),
            repeat_index=value,
        )

    def sort_key(self, axis: AxisName | str = "") -> tuple[str, str, str, int, str]:
        return (
            self.sample_id,
            self.scene_id,
            self.model_or_lane,
            self.repeat_index,
            str(axis),
        )


@dataclass(frozen=True)
class JudgeRequest:
    row_identity: RowIdentity
    axis: AxisName
    prompt: str
    model: str | None = None
    request_payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return self.row_identity.sample_id

    @property
    def scene_id(self) -> str:
        return self.row_identity.scene_id

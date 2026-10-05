"""Contracts for model generation requests and results.

GenerationRequest describes the input needed to run one benchmark sample through
a model, while GenerationResult records the model output artifacts, status,
timing, and metadata. Helper functions normalize runner status values and
restore nested ArtifactRef objects after JSON deserialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from worldfoundry.evaluation.api.json_contract import JsonContract, copy_mapping

from .artifacts import ArtifactRef, coerce_artifact_refs, restore_artifact_refs

GENERATION_REQUEST_SCHEMA_VERSION = "worldfoundry-generation-request"
GENERATION_RESULT_SCHEMA_VERSION = "worldfoundry-generation-result"
GENERATION_SUCCESS_STATUSES = frozenset({"succeeded", "success", "ok", "completed", "done"})
GENERATION_UNKNOWN_STATUS = "unknown"


def normalize_generation_status(status: Any) -> str:
    """Return the canonical lowercase generation status text.

    Missing/blank statuses normalize to ``"unknown"`` (fail-closed): an
    incomplete result row written by an external runner must not be treated
    as a scoreable success. Callers that own an in-process success signal
    (e.g. a pipeline returning normally) should set the status explicitly.

    Args:
        status: Raw status value from a generation runner or result row.
    """
    text = str(status or "").strip().lower()
    return text or GENERATION_UNKNOWN_STATUS


def is_generation_status_successful(status: Any) -> bool:
    """Return whether a status represents a materialized generation artifact.

    Args:
        status: Raw status value to classify.
    """
    return normalize_generation_status(status) in GENERATION_SUCCESS_STATUSES


def is_generation_result_successful(result: "GenerationResult") -> bool:
    """Return whether a result may be scored as a completed generation.

    Args:
        result: Generation result emitted by an runner or loaded from disk.
    """
    return is_generation_status_successful(result.status) and result.error in (None, "", False)


@dataclass(frozen=True, init=False)
class GenerationRequest(JsonContract):
    """Model generation input for one benchmark sample."""

    sample_id: str
    task_name: str
    split: str = "default"
    request_id: str | None = None
    inputs: Mapping[str, Any] = field(default_factory=dict)
    controls: Mapping[str, Any] = field(default_factory=dict)
    generation_kwargs: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    cache_policy: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GENERATION_REQUEST_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        sample_id: str,
        task_name: str | None = None,
        *,
        task_id: str | None = None,
        split: str = "default",
        request_id: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        cache_policy: Mapping[str, Any] | None = None,
        schema_version: str = GENERATION_REQUEST_SCHEMA_VERSION,
    ) -> None:
        resolved_task_name = task_name if task_name is not None else task_id
        if not resolved_task_name:
            raise ValueError("GenerationRequest requires task_name or task_id.")
        if schema_version != GENERATION_REQUEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported GenerationRequest schema_version: {schema_version}")
        object.__setattr__(self, "sample_id", str(sample_id))
        object.__setattr__(self, "task_name", str(resolved_task_name))
        object.__setattr__(self, "split", str(split))
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "inputs", restore_artifact_refs(copy_mapping(inputs)))
        object.__setattr__(self, "controls", restore_artifact_refs(copy_mapping(controls)))
        object.__setattr__(self, "generation_kwargs", copy_mapping(generation_kwargs))
        object.__setattr__(self, "output_schema", copy_mapping(output_schema))
        object.__setattr__(self, "cache_policy", copy_mapping(cache_policy))
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def task_id(self) -> str:
        return self.task_name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationRequest":
        return cls(
            sample_id=str(data["sample_id"]),
            task_name=data.get("task_name"),
            task_id=data.get("task_id"),
            split=data.get("split", "default"),
            request_id=data.get("request_id"),
            inputs=data.get("inputs"),
            controls=data.get("controls"),
            generation_kwargs=data.get("generation_kwargs"),
            output_schema=data.get("output_schema"),
            cache_policy=data.get("cache_policy"),
            schema_version=data.get("schema_version", GENERATION_REQUEST_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class GenerationResult(JsonContract):
    """Model generation output for one benchmark sample."""

    sample_id: str
    request_id: str | None = None
    model_id: str = ""
    artifacts: Mapping[str, ArtifactRef] = field(default_factory=dict)
    status: str = "succeeded"
    error: str | None = None
    timings: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GENERATION_RESULT_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_RESULT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported GenerationResult schema_version: {self.schema_version}")
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "artifacts", coerce_artifact_refs(self.artifacts))
        object.__setattr__(self, "timings", copy_mapping(self.timings))
        object.__setattr__(self, "metadata", copy_mapping(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationResult":
        # Rows serialized by to_dict always carry a status; a row without one
        # is incomplete external data and must not default to success.
        return cls(
            sample_id=str(data["sample_id"]),
            request_id=data.get("request_id"),
            model_id=data.get("model_id", ""),
            artifacts=data.get("artifacts"),
            status=data.get("status") or GENERATION_UNKNOWN_STATUS,
            error=data.get("error"),
            timings=data.get("timings"),
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", GENERATION_RESULT_SCHEMA_VERSION),
        )

"""Materialization of dataset samples into standard GenerationRequests.

This module maps raw, unstructured dataset samples (from JSON, YAML, manifests)
into standardized, strongly-typed GenerationRequest objects containing input features,
control actions, expected output definitions, and caching directives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.datasets import (
    DatasetManifest,
    load_dataset_manifest,
    read_dataset_samples,
    resolve_dataset_samples_path,
)

# Canonical schema version tracking materialized output lists
MATERIALIZED_REQUESTS_SCHEMA_VERSION = "worldfoundry-materialized-requests"

# Default task-specific control keys extracted during materialization
DEFAULT_CONTROL_KEYS = (
    "action",
    "actions",
    "camera",
    "camera_path",
    "control_sequence",
    "controls",
    "physics_case",
    "trajectory",
)


@dataclass(frozen=True)
class MaterializedRequests:
    """Structure encapsulating a fully compiled list of GenerationRequests ready for execution.

    Attributes:
        schema_version: Standard schema identification string.
        task_type: Task grouping name of requests inside this batch.
        benchmark_name: Name of the benchmark corresponding to these tasks.
        split: The dataset evaluation split (e.g. "default", "validation").
        requests: The underlying flat list of standard GenerationRequests.
    """
    schema_version: str
    task_type: str
    benchmark_name: str
    split: str
    requests: tuple[GenerationRequest, ...]

    @property
    def sample_count(self) -> int:
        """Returns total count of requests parsed."""
        return len(self.requests)

    def to_dict(self) -> dict[str, Any]:
        """Converts the compiled materialization summary into a plain serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "benchmark_name": self.benchmark_name,
            "split": self.split,
            "sample_count": len(self.requests),
            "requests": [request.to_dict() for request in self.requests],
        }


def _sample_id(sample: Mapping[str, Any], index: int) -> str:
    """Resolves or defaults the sample identifier from dictionary properties or index counters."""
    return str(sample.get("sample_id", sample.get("id", f"sample-{index:04d}")))


def _merge_mapping(target: dict[str, Any], value: Any) -> None:
    """Updates target mapping safely if input value is a valid dictionary."""
    if isinstance(value, Mapping):
        target.update({str(key): item for key, item in value.items()})


def _inputs_for_sample(sample: Mapping[str, Any], input_keys: Sequence[str]) -> dict[str, Any]:
    """Compiles input characteristics for a sample, locating standard text prompt and media fields."""
    inputs: dict[str, Any] = {}
    _merge_mapping(inputs, sample.get("initial_context"))
    for key in input_keys:
        if key in sample:
            inputs[str(key)] = sample[key]
    if not inputs:
        # Fallback fields scanning for basic multimodality contexts
        for key in ("prompt", "generation_text", "image", "ref_image", "video", "input_video"):
            if key in sample:
                inputs[key] = sample[key]
    references = sample.get("references")
    if isinstance(references, Mapping):
        inputs["references"] = dict(references)
    return inputs


def _controls_for_sample(sample: Mapping[str, Any], control_keys: Sequence[str]) -> dict[str, Any]:
    """Resolves control signals, robot actions, or camera trajectory values from the sample."""
    controls: dict[str, Any] = {}
    _merge_mapping(controls, sample.get("controls"))
    for key in control_keys:
        if key == "controls":
            continue
        if key in sample:
            controls[str(key)] = sample[key]
    return controls


def _output_schema_for_sample(sample: Mapping[str, Any], output_keys: Sequence[str]) -> dict[str, Any]:
    """Establishes expected output schemas and reference values for the sample evaluation."""
    output_schema = {str(key): {"kind": str(key)} for key in output_keys}
    expected_outputs = sample.get("expected_outputs")
    if isinstance(expected_outputs, Mapping):
        for key, value in expected_outputs.items():
            if isinstance(value, Mapping):
                output_schema[str(key)] = dict(value)
            else:
                output_schema[str(key)] = {"expected": value}
    return output_schema


def materialize_generation_requests(
    samples: Sequence[Mapping[str, Any]],
    *,
    task_name: str,
    split: str = "default",
    input_keys: Sequence[str] = (),
    output_keys: Sequence[str] = ("generated_video",),
    control_keys: Sequence[str] = DEFAULT_CONTROL_KEYS,
    generation_defaults: Mapping[str, Any] | None = None,
    cache_policy: Mapping[str, Any] | None = None,
) -> tuple[GenerationRequest, ...]:
    """Constructs standardized immutable `GenerationRequest` instances from polymorphic dictionaries.

    Provides a clean, uniform translation layer bridging unstructured JSON datasets to the 
    rigid, schema-enforced inputs consumed by active model runners.

    Args:
        samples: Raw list of sample dictionaries.
        task_name: High-level task description name.
        split: Selected split identifier (e.g. "validation", "test").
        input_keys: Explicit input field names to extract from raw samples.
        output_keys: Expected model output media assets (default is video generation).
        control_keys: Target keys defining mechanical or action controls (e.g., trajectories).
        generation_defaults: Default hyperparameters passed down to generation_kwargs.
        cache_policy: Instructions dictating cache rules.

    Returns:
        A tuple of materialized GenerationRequests.
    """
    requests: list[GenerationRequest] = []
    for index, sample in enumerate(samples):
        sample_id = _sample_id(sample, index)
        generation_kwargs = dict(generation_defaults or {})
        _merge_mapping(generation_kwargs, sample.get("generation_kwargs"))
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name=task_name,
                split=split,
                request_id=f"{task_name}:{sample_id}",
                inputs=_inputs_for_sample(sample, input_keys),
                controls=_controls_for_sample(sample, control_keys),
                generation_kwargs=generation_kwargs,
                output_schema=_output_schema_for_sample(sample, output_keys),
                cache_policy=cache_policy or {},
            )
        )
    return tuple(requests)


def materialize_requests_from_benchmark(
    benchmark: Any,
    dataset_root: str | Path,
    *,
    limit: int | None = None,
    split: str = "default",
    cache_policy: Mapping[str, Any] | None = None,
) -> MaterializedRequests:
    """Invokes custom benchmark-specific class loaders to materialize evaluation requests.

    Args:
        benchmark: Loaded benchmark adapter conforming to evaluation loader protocols.
        dataset_root: Root directory where dataset files are stored on disk.
        limit: Optional maximum number of samples to read.
        split: Target dataset partition.
        cache_policy: Caching directives.

    Returns:
        A compiled MaterializedRequests container object.
    """
    if getattr(benchmark, "source_kind", None) == "benchmark_zoo":
        raise ValueError(
            "Benchmark Zoo entries do not materialize samples through "
            "materialize_requests_from_benchmark. Use `worldfoundry-eval run "
            "--benchmark <id> --model <id>` for benchmark-zoo runs, or "
            "`worldfoundry-eval task materialize` with an explicit filesystem "
            "task YAML and dataset manifest."
        )
    if not callable(getattr(benchmark, "load_samples", None)):
        raise TypeError("benchmark must provide load_samples(dataset_root, limit=...)")
    task_cfg, samples = benchmark.load_samples(dataset_root, limit=limit)
    generation_defaults = getattr(task_cfg, "generation_defaults", None)
    requests = materialize_generation_requests(
        samples,
        task_name=str(benchmark.task_type),
        split=split,
        input_keys=tuple(getattr(benchmark, "input_keys", ())),
        output_keys=tuple(getattr(benchmark, "output_keys", ("generated_video",))),
        generation_defaults=generation_defaults if isinstance(generation_defaults, Mapping) else {},
        cache_policy=cache_policy,
    )
    return MaterializedRequests(
        schema_version=MATERIALIZED_REQUESTS_SCHEMA_VERSION,
        task_type=str(benchmark.task_type),
        benchmark_name=str(benchmark.benchmark_name),
        split=split,
        requests=requests,
    )


def materialize_requests_from_dataset_manifest(
    manifest: str | Path | DatasetManifest | Mapping[str, Any],
    *,
    task_name: str,
    split: str | None = None,
    input_keys: Sequence[str] = (),
    output_keys: Sequence[str] = ("generated_video",),
    control_keys: Sequence[str] = DEFAULT_CONTROL_KEYS,
    generation_defaults: Mapping[str, Any] | None = None,
    cache_policy: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> MaterializedRequests:
    """Reads a dataset manifest file to directly compile matching evaluation GenerationRequests.

    Args:
        manifest: DatasetManifest class, raw dictionary, or file path to load.
        task_name: The task name applied to all output requests.
        split: Optional override for the dataset split.
        input_keys: Standard keys captured from dataset samples to be treated as inputs.
        output_keys: Expected targets mapping to the output schema.
        control_keys: Sequence of keys describing control parameters.
        generation_defaults: Default hyperparameters passed down to the model.
        cache_policy: Caching directives.
        limit: Max number of samples to parse.

    Returns:
        A compiled, frozen MaterializedRequests instance.
    """
    manifest_path = Path(manifest) if isinstance(manifest, (str, Path)) else None
    dataset_manifest = (
        load_dataset_manifest(manifest_path)
        if manifest_path is not None
        else manifest
        if isinstance(manifest, DatasetManifest)
        else DatasetManifest.from_dict(manifest)
    )
    samples_path = resolve_dataset_samples_path(dataset_manifest, manifest_path=manifest_path)
    samples = read_dataset_samples(samples_path)
    if limit is not None:
        samples = samples[: int(limit)]
    request_split = split or dataset_manifest.split
    requests = materialize_generation_requests(
        samples,
        task_name=task_name,
        split=request_split,
        input_keys=input_keys,
        output_keys=output_keys,
        control_keys=control_keys,
        generation_defaults=generation_defaults,
        cache_policy=cache_policy,
    )
    return MaterializedRequests(
        schema_version=MATERIALIZED_REQUESTS_SCHEMA_VERSION,
        task_type=task_name,
        benchmark_name=dataset_manifest.dataset_id,
        split=request_split,
        requests=requests,
    )


# Expose alias for compatibility
materialize_requests = materialize_generation_requests


__all__ = [
    "DEFAULT_CONTROL_KEYS",
    "MATERIALIZED_REQUESTS_SCHEMA_VERSION",
    "MaterializedRequests",
    "materialize_generation_requests",
    "materialize_requests",
    "materialize_requests_from_benchmark",
    "materialize_requests_from_dataset_manifest",
]

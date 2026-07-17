"""Core execution layer interfaces, protocols, and data schemas for WorldFoundry.

This module defines the primary data transfer objects (DTOs) and structural protocols
governing how benchmarks are loaded, datasets are materialized, samples are iterated over,
and results/stages of official benchmark executions are tracked, normalized, and collected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


# Generic type alias for JSON-compatible values
JsonValue = Any


def _plain(value: JsonValue) -> JsonValue:
    """Recursively converts Path objects and collection containers into plain JSON-serializable values.

    Args:
        value: Any data structure containing potential non-serializable objects like Path.

    Returns:
        A copy of the input structure with nested Paths converted to string representations.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class BenchmarkSample:
    """Represents a single evaluation item or prompt-response sample in a benchmark.

    Attributes:
        sample_id: Unique identifier for this benchmark sample.
        inputs: Input features, text prompts, or condition parameters mapped by string keys.
        expected_outputs: Target answers, reference values, or ground-truth data mapped by string keys.
        metadata: Additional contextual metadata associated with this sample.
    """
    sample_id: str
    inputs: Mapping[str, JsonValue] = field(default_factory=dict)
    expected_outputs: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serializes the benchmark sample to a plain JSON-compatible dictionary."""
        return _plain(asdict(self))


@dataclass(frozen=True)
class DatasetMaterializationPlan:
    """Defines the download, retrieval, and preparation steps for benchmark dataset assets.

    Provides a declarative blueprint that execution runners can inspect to download required
    data bundles, execute preparation bash/shell commands, or authenticate before starting evaluation.

    Attributes:
        benchmark_id: Canonical identifier of the target benchmark.
        dataset_ids: Tuple of dataset identifiers or HuggingFace repo/split ids required.
        commands: Nested shell command tuples executed in sequence during dataset setup.
        expected_paths: List of relative or absolute file paths expected to exist post-materialization.
        requires_auth: Flag indicating if dataset retrieval needs credentials/login tokens.
        notes: Helpful diagnostic explanations or instructions regarding setup prerequisites.
    """
    benchmark_id: str
    dataset_ids: tuple[str, ...] = ()
    commands: tuple[tuple[str, ...], ...] = ()
    expected_paths: tuple[str, ...] = ()
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        """Serializes the materialization plan to a plain JSON-compatible dictionary."""
        return _plain(asdict(self))


@dataclass(frozen=True)
class OfficialRunResult:
    """The canonical scorecard and outcome payload resulting from a benchmark evaluation run.

    Provides pointers to standard WorldFoundry output directories, scorecard files, and key
    verification states demonstrating if the run was official and meets compliance.

    Attributes:
        benchmark_id: Canonical identifier of the executed benchmark.
        output_dir: Path to the root output directory of this run.
        scorecard_path: Path to the generated scorecard summary JSON.
        raw_results_path: Optional path to raw, un-normalized model outputs or evaluation logs.
        official_benchmark_verified: Indicates if execution conformed fully to upstream protocol standards.
        integration_evidence: Indicates if execution included verification evidence such as environment details.
        artifacts: Dictionary mapping unique asset names to their generated file parameters or metadata.
        metadata: Arbitrary execution or system metrics reported by the runner.
    """
    benchmark_id: str
    output_dir: Path
    scorecard_path: Path
    raw_results_path: Path | None = None
    official_benchmark_verified: bool = False
    integration_evidence: bool = False
    artifacts: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def full_official_ok(self) -> bool:
        """Determines if the result has successfully passed both official verification and evidence checks."""
        return self.official_benchmark_verified and self.integration_evidence

    @property
    def execution_ok(self) -> bool:
        """Return whether the official runner produced valid integration evidence."""
        return self.integration_evidence

    @property
    def operation_ok(self) -> bool:
        """Return whether the operation selected by ``metadata.mode`` succeeded.

        Official execution remains evidence-gated.  Result-import modes may
        succeed without becoming integration evidence when their normalizer
        produced usable metrics.  Contract mode is likewise an internal
        operation outcome, not an official benchmark claim.
        """
        if self.execution_ok:
            return True
        mode = str(self.metadata.get("mode") or "")
        if mode in {"normalizer", "official-validation"}:
            return (
                self.metadata.get("normalization_ok") is True
                and self.metadata.get("returncode") in {None, 0}
            )
        if mode == "contract":
            return self.metadata.get("contract_only") is True
        return False

    @property
    def ok(self) -> bool:
        """Return whether execution succeeded, independently of full-suite eligibility."""
        return self.operation_ok

    def to_dict(self) -> dict[str, JsonValue]:
        """Converts the official run result into a plain JSON-compatible dictionary, including properties."""
        payload = _plain(asdict(self))
        payload["execution_ok"] = self.execution_ok
        payload["operation_ok"] = self.operation_ok
        payload["full_official_ok"] = self.full_official_ok
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class OfficialRunStage:
    """Tracks state and accumulated artifacts at a specific phase of the official evaluation lifecycle.

    Used by OfficialBenchmarkRunner instances to propagate execution state across lifecycle stages
    such as preparation, running, collection, and normalization.

    Attributes:
        benchmark_id: Canonical identifier of the target benchmark.
        stage: Current phase name (e.g., 'prepared', 'run_complete', 'collected', 'normalized').
        output_dir: Base workspace or directory containing intermediate outputs for this stage.
        status: Status string representing the stage's health (default is "ok").
        artifacts: Mapping of keys to file references or structural logs produced in this stage.
        data: Internal operational state or properties transferred between runner pipeline steps.
        metadata: Additional environment or runner telemetry captured during this phase.
    """
    benchmark_id: str
    stage: str
    output_dir: Path
    status: str = "ok"
    artifacts: Mapping[str, JsonValue] = field(default_factory=dict)
    data: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serializes the official run stage to a plain JSON-compatible dictionary."""
        return _plain(asdict(self))


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Protocol establishing the interface required for any system benchmark integration.

    Implementations are responsible for loading baseline manifests, defining setup dependencies,
    streaming task inputs, and executing evaluation procedures.
    """
    benchmark_id: str

    def load_manifest(self) -> Mapping[str, JsonValue]:
        """Return the upstream or WorldFoundry manifest metadata used by this runner."""

    def materialization_plan(self) -> DatasetMaterializationPlan:
        """Return the dataset/download plan needed before evaluation can run."""

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        """Yield benchmark samples or generated-artifact records in canonical form."""

    def evaluate(self, *, output_dir: str | Path, **kwargs: JsonValue) -> OfficialRunResult:
        """Run or normalize the benchmark and return canonical scorecard paths."""


@runtime_checkable
class OfficialBenchmarkRunner(BenchmarkRunner, Protocol):
    """Lifecycle surface for runners that wrap or normalize official benchmark runtimes.

    Extends basic BenchmarkRunner by providing fine-grained step-by-step methods allowing
    orchestrators to isolate environment preparation, actual system execution, downstream collection,
    and standard scorecard normalization.
    """

    def prepare(self, *, output_dir: str | Path, **kwargs: JsonValue) -> OfficialRunStage:
        """Create the run workspace and resolve runner-specific inputs."""

    def run(self, prepared: OfficialRunStage) -> OfficialRunStage:
        """Run the official benchmark runtime, or record an explicit no-op for dry/contract modes."""

    def collect(self, run_result: OfficialRunStage) -> OfficialRunStage:
        """Collect upstream outputs and generated artifacts for normalization."""

    def normalize(self, collected: OfficialRunStage) -> OfficialRunResult:
        """Write normalized WorldFoundry scorecard artifacts."""

    def report_metadata(self) -> Mapping[str, JsonValue]:
        """Return stable runner metadata suitable for scorecards and audit reports."""

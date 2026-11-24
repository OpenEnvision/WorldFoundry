"""Benchmark-zoo manifest runner: prepare → run → collect → normalize.

Maps checked-in benchmark-zoo YAML entries onto :class:`ManifestBenchmarkRunner`
instances that drive official subprocesses or contract fixtures and emit
``scorecard.json``.

Layout:

* :mod:`.benchmark_commands` — command, environment, and results-path resolution for subprocess runs.
* :mod:`.benchmark_normalization` — specialized, embodied, and generic official-result normalizers.
* :class:`ManifestBenchmarkRunner` — per-benchmark lifecycle orchestrator.
* :class:`BenchmarkRunnerRegistry` lookup and ``run_benchmark_execution``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import run_logged_subprocess
from worldfoundry.core.io.formats.serialization import write_json
from worldfoundry.core.observability.logging_setup import get_logger, log_context
from worldfoundry.core.observability.time import utc_now_iso
from worldfoundry.evals.reporting import inspect_scorecard_runtime_flags
from worldfoundry.evals.tasks.catalog.dispatch import (
    artifact_official_run_benchmarks,
)
from worldfoundry.evals.tasks.execution.framework.result_normalizer import OfficialResultsNormalizer

from ...catalog.schema import BenchmarkZooEntry, load_entries
from ...catalog.zoo_registry import BenchmarkZooRegistry, UnknownBenchmarkZooKeyError, load_benchmark_zoo_registry
from ...contracts.external import get_external_benchmark_contract
from ...execution.framework.scoring_registry import (
    contract_evaluator_kind,
    has_benchmark_contract_evaluator,
    write_benchmark_contract_evaluation,
)
from ..runners._benchmark_metrics import evaluate_external_metric, list_external_metric_evaluators
from .benchmark_commands import (
    DEFAULT_MANIFEST_PATH,
    REPO_ROOT,
    SCORECARD_SCHEMA_VERSION,
    BenchmarkExecutionUnavailableError,
    JsonValue,
    _apply_benchmark_data_root_env,
    _artifact_path,
    _command_for_mode,
    _command_kind_for_mode,
    _command_to_json,
    _ensure_repo_on_pythonpath,
    _env_mapping,
    _expected_artifact_checks,
    _requires_external_runtime,
    _resolve_python_command,
    _runner_runtime_spec,
    _runtime_spec_env,
    _subprocess_command,
    _timeout_seconds,
    _workspace_official_run_command,
    _write_runtime_report,
)
from .benchmark_normalization import (
    _dataset_commands,
    _expected_paths,
    _metric_raw_row,
    _metric_scorecard_entry,
    _run_embodied_result_normalizer,
    _run_specialized_result_normalizer,
    _scorecard_normalization_available,
)
from .interfaces import (
    BENCHMARK_RUN_OFFICIAL_MODES as _OFFICIAL_MODES,
)
from .interfaces import (
    BenchmarkSample,
    DatasetMaterializationPlan,
    OfficialRunResult,
    OfficialRunStage,
    normalize_benchmark_run_mode,
)


def _emitted_scorecard_status(output_dir: Path) -> str | None:
    """Return the import/normalize status an official command wrote into its own scorecard, if any."""
    emitted_scorecard_path = output_dir / "scorecard.json"
    if not emitted_scorecard_path.is_file():
        return None
    try:
        emitted_scorecard = json.loads(emitted_scorecard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        emitted_scorecard = {}
    emitted_run = emitted_scorecard.get("run") if isinstance(emitted_scorecard, Mapping) else {}
    emitted_status = emitted_run.get("status") if isinstance(emitted_run, Mapping) else None
    if emitted_status in {"official_results_imported", "official_results_normalized"}:
        return str(emitted_status)
    return None


def _official_run_blocker(
    *,
    mode: str,
    runtime_spec: Mapping[str, Any],
    results_path: Any,
) -> tuple[str, str] | None:
    """Return the ``(status, error)`` when the results path or runtime spec rules out an official run."""
    if results_path and runtime_spec.get("results_path_source") == "official_results_path":
        return "missing_official_results_path", f"official results path does not exist: {results_path}"
    if mode == "normalizer":
        if results_path:
            return "missing_official_results_path", f"official results path does not exist: {results_path}"
        return (
            "missing_official_results_path",
            "--mode normalizer requires --official-results-path or a configured results path",
        )
    if runtime_spec.get("kind") == "external_official_repo" and not runtime_spec.get("repo_url"):
        return "missing_official_runtime_spec", "missing repo_url for external_official_repo runtime"
    return None


@dataclass(frozen=True)
class ManifestBenchmarkRunner:
    """Per-benchmark orchestrator backed by one benchmark-zoo manifest entry.

    Execution flow:

    * ``prepare`` — resolve ``mode``, runtime spec, and output workspace.
    * ``run`` — subprocess official command, import existing results, or contract fixture.
    * ``collect`` — harvest generated files and artifact checks.
    * ``normalize`` — specialized, embodied, generic, or contract scorecard paths.
    * ``evaluate`` — full ``prepare → run → collect → normalize`` chain.
    """

    entry: BenchmarkZooEntry
    manifest_path: Path | None = None

    @property
    def benchmark_id(self) -> str:
        """Canonical benchmark identifier."""
        return self.entry.benchmark_id

    def load_manifest(self) -> Mapping[str, JsonValue]:
        """Serialize manifest entry (plus ``manifest_path`` when set)."""
        payload = self.entry.to_dict()
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
        return payload

    def materialization_plan(self) -> DatasetMaterializationPlan:
        """Build dataset download plan from manifest refs."""
        notes = [
            *self.entry.notes,
            *(f"blocked: {blocker}" for blocker in self.entry.blockers),
        ]
        if self.entry.dataset.not_applicable and self.entry.dataset.reason:
            notes.append(f"dataset not applicable: {self.entry.dataset.reason}")
        elif not self.entry.hf_dataset_ids and self.entry.dataset.reason:
            notes.append(f"dataset acquisition blocked: {self.entry.dataset.reason}")
        return DatasetMaterializationPlan(
            benchmark_id=self.benchmark_id,
            dataset_ids=self.entry.hf_dataset_ids,
            commands=_dataset_commands(self.entry),
            expected_paths=_expected_paths(self.entry),
            requires_auth=self.entry.requires_auth,
            notes=tuple(notes),
        )

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        """Yield a contract placeholder sample for integration tracking."""
        contract = get_external_benchmark_contract(self.benchmark_id)
        yield BenchmarkSample(
            sample_id=f"{self.benchmark_id}:manifest",
            inputs=dict.fromkeys(contract.input_keys),
            expected_outputs=dict.fromkeys(contract.output_keys),
            metadata={
                "benchmark_id": self.benchmark_id,
                "integration_status": self.entry.integration_status,
                "verification_status": self.entry.verification_status,
                "contract_only": self.entry.integration_status != "integrated",
            },
        )

    def report_metadata(self) -> Mapping[str, JsonValue]:
        """Runner capability metadata for scorecards and audits."""
        contract = get_external_benchmark_contract(self.benchmark_id)
        metadata: dict[str, JsonValue] = {
            "benchmark_id": self.benchmark_id,
            "runner": "benchmark_zoo_manifest_runner",
            "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
            "manifest_integration_status": self.entry.integration_status,
            "manifest_verification_status": self.entry.verification_status,
            "requires_upstream_runtime": _requires_external_runtime(
                default=contract.requires_upstream_runtime,
                runtime_spec=self.entry.runner_runtime,
            ),
        }
        if self.entry.runner_target:
            metadata["runner_target"] = self.entry.runner_target
        if self.entry.runner_runtime:
            metadata["runner_runtime"] = dict(self.entry.runner_runtime)
        if self.entry.install_profile:
            metadata["install_profile"] = self.entry.install_profile
        if self.manifest_path is not None:
            metadata["manifest_path"] = str(self.manifest_path)
        return metadata

    def prepare(
        self,
        *,
        output_dir: str | Path,
        mode: str = "contract",
        generated_artifact_dir: str | Path | None = None,
        **kwargs: JsonValue,
    ) -> OfficialRunStage:
        """Resolve runtime spec and create output workspace for ``mode``."""
        resolved_mode = normalize_benchmark_run_mode(mode)

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        plan = self.materialization_plan()
        command = _command_for_mode(self.entry, resolved_mode)
        command_kind = _command_kind_for_mode(resolved_mode) if resolved_mode in _OFFICIAL_MODES else None
        runtime_spec = _runner_runtime_spec(self.entry, benchmark_id=self.benchmark_id, kwargs=kwargs)
        input_path = runtime_spec.get("results_path")
        input_path = Path(str(input_path)).resolve() if input_path else None
        reports = {
            root / name
            for name in (
                "scorecard.json",
                "summary.json",
                "report.md",
                "raw_metric_table.jsonl",
                "runner_runtime_report.json",
            )
        }
        reports.update(Path(item["resolved_path"]) for item in _expected_artifact_checks(self.entry, root))
        for path in reports:
            if path.resolve().is_relative_to(root.resolve()) and path.resolve() != input_path and path.is_file():
                path.unlink()
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="prepare",
            output_dir=root,
            status="ready",
            data={
                "mode": resolved_mode,
                "contract_only": resolved_mode == "contract",
                "command_kind": command_kind,
                "command": _command_to_json(command),
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "kwargs": dict(kwargs),
                "runtime": runtime_spec,
            },
            metadata={
                **self.report_metadata(),
                "materialization_plan": plan.to_dict(),
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
            },
        )

    def run(self, prepared: OfficialRunStage) -> OfficialRunStage:
        """Run official subprocess, import results, or emit contract fixture."""
        data = dict(prepared.data)
        mode = str(data.get("mode", "contract"))
        if mode in _OFFICIAL_MODES:
            return self._run_official(prepared)
        if mode != "contract":
            raise ValueError(f"unsupported benchmark run mode: {mode}")

        run_status = "contract_fixture"
        data.update(
            {
                "run_status": run_status,
                "evidence_level": "contract_fixture_only",
                "official_benchmark_verified": False,
                "integration_evidence": False,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=run_status,
            artifacts=prepared.artifacts,
            data=data,
            metadata=prepared.metadata,
        )

    def _failed_official_run_result(
        self,
        prepared: OfficialRunStage,
        *,
        status: str,
        error: str,
        data: dict[str, JsonValue],
        metadata: Mapping[str, JsonValue],
    ) -> OfficialRunStage:
        """Build a failed ``run`` stage with ``error`` populated."""
        data.update(
            {
                "run_status": status,
                "official_benchmark_verified": False,
                "integration_evidence": False,
                "error": error,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=status,
            artifacts=prepared.artifacts,
            data=data,
            metadata={**metadata, "run_status": status},
        )

    def _run_official(self, prepared: OfficialRunStage) -> OfficialRunStage:
        """Execute official subprocess and capture stdout/stderr logs."""
        data = dict(prepared.data)
        mode = str(data.get("mode", "official-validation"))
        command_kind = str(data.get("command_kind") or _command_kind_for_mode(mode))
        command = _command_for_mode(self.entry, mode)
        kwargs = data.get("kwargs")
        kwargs = kwargs if isinstance(kwargs, Mapping) else {}
        if mode == "official-run":
            workspace_command = _workspace_official_run_command(
                self.benchmark_id,
                output_dir=prepared.output_dir,
                generated_artifact_dir=data.get("generated_artifact_dir"),
                kwargs=kwargs,
            )
            if workspace_command is not None:
                command = workspace_command
        runtime_spec = data.get("runtime")
        runtime_spec = (
            dict(runtime_spec)
            if isinstance(runtime_spec, Mapping)
            else _runner_runtime_spec(self.entry, benchmark_id=self.benchmark_id)
        )
        data["runtime"] = runtime_spec
        metadata = {
            **prepared.metadata,
            "runtime": runtime_spec,
            "runner_runtime_spec": runtime_spec,
        }
        results_path = runtime_spec.get("results_path")
        results_path_source = runtime_spec.get("results_path_source")
        implicit_data_root_result = mode == "official-run" and results_path_source == "benchmark_data_root"
        if results_path and Path(str(results_path)).exists() and not implicit_data_root_result:
            return self._imported_results_stage(
                prepared, status="official_results_import", command_kind=command_kind, data=data, metadata=metadata
            )
        if implicit_data_root_result and results_path:
            # Dataset roots commonly contain annotations or stale results from
            # earlier runs.  They are inputs, never authority to skip a fresh
            # official-run subprocess or replace its newly written scorecard
            # during normalization.
            data["ignored_preexisting_results_path"] = str(results_path)
            metadata["ignored_preexisting_results_path"] = str(results_path)
            runtime_spec["ignored_preexisting_results_path"] = str(results_path)
            runtime_spec["ignored_results_path_source"] = str(results_path_source)
            runtime_spec["results_path"] = None
            runtime_spec["results_path_source"] = None
        score_dir = kwargs.get("score_dir") or os.environ.get("WORLDFOUNDRY_CAMERABENCH_SCORE_DIR")
        if self.benchmark_id == "camerabench" and mode in {"official-validation", "normalizer"} and score_dir:
            score_dir_path = Path(str(score_dir))
            if score_dir_path.is_dir():
                return self._imported_results_stage(
                    prepared,
                    status="official_score_dir_import",
                    command_kind=command_kind,
                    data=data,
                    metadata=metadata,
                )
        blocker = _official_run_blocker(mode=mode, runtime_spec=runtime_spec, results_path=results_path)
        if blocker is not None:
            status, error = blocker
            return self._failed_official_run_result(prepared, status=status, error=error, data=data, metadata=metadata)
        if command is None:
            return self._failed_official_run_result(
                prepared,
                status="missing_official_command",
                error=f"missing {command_kind}_command",
                data=data,
                metadata=metadata,
            )

        workdir = Path(str(kwargs.get("workdir"))) if kwargs.get("workdir") else REPO_ROOT
        env = self._official_command_env(
            output_dir=prepared.output_dir,
            command_kind=command_kind,
            kwargs=kwargs,
            runtime_spec=runtime_spec,
            generated_artifact_dir=data.get("generated_artifact_dir"),
        )

        resolved_command = _resolve_python_command(command)
        stdout_path = prepared.output_dir / f"{command_kind}_stdout.log"
        stderr_path = prepared.output_dir / f"{command_kind}_stderr.log"
        timeout_seconds = _timeout_seconds(kwargs.get("timeout_seconds", kwargs.get("timeout")))
        start = time.monotonic()
        try:
            completed = run_logged_subprocess(
                _subprocess_command(resolved_command),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cwd=workdir,
                env=env,
                timeout=timeout_seconds,
                shell=isinstance(resolved_command, str),
            )
            returncode: int | None = completed.returncode
            status = "succeeded" if completed.returncode == 0 else "failed"
            if completed.returncode == 0:
                status = _emitted_scorecard_status(prepared.output_dir) or status
            error = None if completed.returncode == 0 else f"official command exited with code {completed.returncode}"
        except subprocess.TimeoutExpired:
            returncode = None
            status = "timeout"
            error = f"official command timed out after {timeout_seconds} seconds"
        except OSError as exc:
            returncode = None
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            status = "failed_to_launch"
            error = str(exc)
        duration_seconds = time.monotonic() - start

        data.update(
            {
                "run_status": status,
                "returncode": returncode,
                "duration_seconds": duration_seconds,
                "command": _command_to_json(resolved_command),
                "command_kind": command_kind,
                "workdir": str(workdir),
                "runtime": runtime_spec,
                "stdout_path": str(stdout_path.resolve()),
                "stderr_path": str(stderr_path.resolve()),
                "error": error,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=status,
            artifacts=prepared.artifacts,
            data=data,
            metadata={**metadata, "run_status": status, "returncode": returncode},
        )

    def _imported_results_stage(
        self,
        prepared: OfficialRunStage,
        *,
        status: str,
        command_kind: str,
        data: dict[str, Any],
        metadata: Mapping[str, Any],
    ) -> OfficialRunStage:
        """Record a run stage that imports existing official results instead of launching a command."""
        data.update(
            {
                "run_status": status,
                "returncode": 0,
                "duration_seconds": 0.0,
                "command": None,
                "command_kind": command_kind,
                "workdir": str(REPO_ROOT),
                "stdout_path": None,
                "stderr_path": None,
                "expected_artifact_checks": [],
                "scorecard_runtime_flags": {
                    "official_benchmark_verified": False,
                    "integration_evidence": False,
                },
                "official_benchmark_verified": False,
                "integration_evidence": False,
                "error": None,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=status,
            artifacts=prepared.artifacts,
            data=data,
            metadata={**metadata, "run_status": status},
        )

    def _official_command_env(
        self,
        *,
        output_dir: Path,
        command_kind: str,
        kwargs: Mapping[str, Any],
        runtime_spec: Mapping[str, Any],
        generated_artifact_dir: Any,
    ) -> dict[str, str]:
        """Build the environment for the official benchmark subprocess."""
        env = os.environ.copy()
        env.update(
            {
                "WORLDFOUNDRY_BENCHMARK_ID": self.benchmark_id,
                "WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR": str(output_dir),
                "WORLDFOUNDRY_BENCHMARK_COMMAND_KIND": command_kind,
            }
        )
        _ensure_repo_on_pythonpath(env)
        env.setdefault("WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable)
        if generated_artifact_dir is not None:
            env["WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"] = str(generated_artifact_dir)
        if kwargs.get("limit") not in (None, ""):
            env["WORLDFOUNDRY_BENCHMARK_LIMIT"] = str(kwargs["limit"])
        if kwargs.get("model_id") not in (None, ""):
            env["WORLDFOUNDRY_MODEL_ID"] = str(kwargs["model_id"])
        if kwargs.get("split") not in (None, ""):
            env["WORLDFOUNDRY_BENCHMARK_SPLIT"] = str(kwargs["split"])
        env.update(
            _runtime_spec_env(
                runtime_spec,
                benchmark_id=self.benchmark_id,
                generated_artifact_dir=generated_artifact_dir,
            )
        )
        _apply_benchmark_data_root_env(env, self.benchmark_id, kwargs)
        if kwargs.get("prompt_manifest"):
            env["WORLDFOUNDRY_PROMPT_MANIFEST"] = str(kwargs["prompt_manifest"])
            if self.benchmark_id == "phyfps-bench-gen":
                env["WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST"] = str(kwargs["prompt_manifest"])
        env.update(_env_mapping(kwargs.get("env_overrides", kwargs.get("env"))))
        return env

    def collect(self, run_result: OfficialRunStage) -> OfficialRunStage:
        """Harvest output files and expected-artifact checks after ``run``."""
        data = dict(run_result.data)
        generated_files: list[str] = []
        generated_artifact_dir = data.get("generated_artifact_dir")
        if generated_artifact_dir is not None:
            artifact_root = Path(str(generated_artifact_dir))
            if artifact_root.exists():
                generated_files = [str(path) for path in sorted(artifact_root.rglob("*")) if path.is_file()]

        data["generated_files"] = generated_files
        data["generated_file_count"] = len(generated_files)
        data["output_files"] = [str(path) for path in sorted(run_result.output_dir.rglob("*")) if path.is_file()]
        data["expected_artifact_checks"] = _expected_artifact_checks(self.entry, run_result.output_dir)
        data["scorecard_runtime_flags"] = (
            {}
            if data.get("run_status") == "official_results_import"
            else inspect_scorecard_runtime_flags(run_result.output_dir / "scorecard.json")
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="collect",
            output_dir=run_result.output_dir,
            status="collected",
            artifacts=run_result.artifacts,
            data=data,
            metadata=run_result.metadata,
        )

    def normalize(self, collected: OfficialRunStage) -> OfficialRunResult:
        """Normalize collected outputs into ``scorecard.json`` (official or contract)."""
        data = dict(collected.data)
        mode = str(data.get("mode", "contract"))
        mode = normalize_benchmark_run_mode(mode)
        if mode in _OFFICIAL_MODES:
            return self._normalize_official(collected, mode=mode)

        root = collected.output_dir
        contract = get_external_benchmark_contract(self.benchmark_id)
        generated_files = [str(path) for path in data.get("generated_files") or ()]
        generated_artifact_dir = data.get("generated_artifact_dir")
        contract_only = mode == "contract"
        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}

        if contract_only and has_benchmark_contract_evaluator(self.benchmark_id):
            evaluator_kwargs = dict(extra_metadata)
            result = write_benchmark_contract_evaluation(
                benchmark_id=self.benchmark_id,
                display_name=self.entry.name or contract.display_name,
                official_metric_ids=contract.metric_ids,
                output_dir=root,
                generated_artifact_dir=None if generated_artifact_dir is None else str(generated_artifact_dir),
                manifest=self.load_manifest(),
                runner="benchmark_zoo_manifest_runner",
                mode=mode,
                **evaluator_kwargs,
            )
            return OfficialRunResult(
                benchmark_id=self.benchmark_id,
                output_dir=root,
                scorecard_path=root / "scorecard.json",
                raw_results_path=root / "raw_metric_table.jsonl",
                official_benchmark_verified=False,
                integration_evidence=False,
                artifacts=result["artifacts"],
                metadata={
                    "mode": mode,
                    "contract_only": contract_only,
                    "manifest_integration_status": self.entry.integration_status,
                    "manifest_verification_status": self.entry.verification_status,
                    "requires_upstream_runtime": contract.requires_upstream_runtime,
                    "evaluator": contract_evaluator_kind(self.benchmark_id),
                    **extra_metadata,
                },
            )

        generated_artifact_manifest: JsonValue = {
            "generated_files": generated_files,
        }
        if generated_artifact_dir is not None:
            generated_artifact_manifest["generated_artifact_dir"] = str(generated_artifact_dir)
        task_metadata: dict[str, JsonValue] = {
            "benchmark_id": self.benchmark_id,
            "expected_output_keys": list(contract.output_keys),
            "metric_ids": list(contract.metric_ids),
        }
        if self.entry.expected_artifacts:
            task_metadata["required_artifacts"] = [
                item if isinstance(item, str) else item.get("path", item.get("uri"))
                for item in self.entry.expected_artifacts
                if isinstance(item, (str, Mapping))
            ]

        metric_results = [
            evaluate_external_metric(
                self.benchmark_id,
                metric_id,
                generated_artifact_manifest=generated_artifact_manifest,
                task_metadata=task_metadata,
                reference={"contract": contract.to_dict(), "manifest": self.load_manifest()},
                sample_id=f"{self.benchmark_id}:contract",
                artifact_base_dir=generated_artifact_dir,
            ).to_dict()
            for metric_id in contract.metric_ids
        ]

        raw_rows = [_metric_raw_row(result) for result in metric_results]
        per_metric = {str(result["metric_id"]): _metric_scorecard_entry(result) for result in metric_results}
        blocked_count = sum(
            1
            for result in metric_results
            if result.get("skip_reason") in {"judge_api_required", "judge_required", "external_runtime_required"}
        )
        failed_count = sum(
            1
            for result in metric_results
            if result.get("valid") is not True
            and result.get("skip_reason") not in {"judge_api_required", "judge_required", "external_runtime_required"}
        )
        available_count = sum(1 for result in raw_rows if result.get("available") is True)

        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "run": {
                "status": "contract_fixture",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_manifest_runner",
                "mode": mode,
            },
            "benchmark": {
                "benchmark_id": self.benchmark_id,
                "name": self.entry.name or contract.display_name,
                "contract_only": contract_only,
                "evidence_level": "contract_fixture_only",
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
            },
            "dataset": {
                "hf_dataset_ids": list(self.entry.hf_dataset_ids),
                "requires_auth": self.entry.requires_auth,
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "generated_file_count": len(generated_files),
            },
            "metrics": {
                "per_metric": per_metric,
                "summary": {
                    "sample_count": len(generated_files),
                    "successful_samples": len(generated_files),
                    "failed_samples": 0,
                    "available_metric_count": available_count,
                    "blocked_metric_count": blocked_count,
                    "failed_metric_count": failed_count,
                },
            },
            "evaluation": {
                "available": available_count > 0,
                "kind": "external_metric_registry",
                "evidence_level": "contract_fixture_only",
                "skip_count": len(contract.metric_ids) - available_count,
                "blocked_count": blocked_count,
                "failed_count": failed_count,
            },
            "artifacts": {
                "scorecard": _artifact_path(root, "scorecard.json"),
                "benchmark_contract": _artifact_path(root, "benchmark_contract.json"),
                "raw_metric_table": _artifact_path(root, "raw_metric_table.jsonl"),
            },
        }
        benchmark_contract = contract.to_dict()
        benchmark_contract["manifest"] = self.load_manifest()
        benchmark_contract["generated_files"] = generated_files
        benchmark_contract["metric_evaluators"] = [
            entry.to_dict()
            for entry in list_external_metric_evaluators(self.benchmark_id)
            if entry.benchmark_id == self.benchmark_id
        ]

        write_json(root / "scorecard.json", scorecard)
        write_json(root / "benchmark_contract.json", benchmark_contract)
        (root / "raw_metric_table.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in raw_rows),
            encoding="utf-8",
        )

        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=root / "scorecard.json",
            raw_results_path=root / "raw_metric_table.jsonl",
            official_benchmark_verified=False,
            integration_evidence=False,
            artifacts=scorecard["artifacts"],
            metadata={
                "mode": mode,
                "contract_only": contract_only,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                **extra_metadata,
            },
        )

    def _normalize_generic_official_results(
        self,
        *,
        root: Path,
        contract: Any,
        data: Mapping[str, JsonValue],
        mode: str,
        results_path: Path,
        runtime_spec: Mapping[str, JsonValue],
        runtime_report_path: Path,
        generated_artifact_dir: JsonValue,
        generated_files: list[str],
    ) -> OfficialRunResult:
        """Normalize official results via :class:`OfficialResultsNormalizer`."""
        normalization = OfficialResultsNormalizer.from_benchmark_entry(self.entry).normalize_file(str(results_path))
        raw_metric_path = root / "raw_metric_table.jsonl"
        raw_metric_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalization.raw_metric_rows()
            ),
            encoding="utf-8",
        )
        per_metric = normalization.scorecard_metrics()
        available_count = sum(1 for item in per_metric.values() if item.get("available") is True)
        blocked_count = len(per_metric) - available_count
        scorecard_path = root / "scorecard.json"
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": available_count > 0,
            "run": {
                "status": data.get("run_status", "normalizer_only"),
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_manifest_runner",
                "mode": mode,
                "command": data.get("command"),
                "returncode": data.get("returncode"),
                "error": data.get("error"),
            },
            "benchmark": {
                "benchmark_id": self.benchmark_id,
                "name": self.entry.name or contract.display_name,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
            },
            "dataset": {
                "hf_dataset_ids": list(self.entry.hf_dataset_ids),
                "requires_auth": self.entry.requires_auth,
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "generated_file_count": len(generated_files),
                "official_results_path": str(results_path),
                "official_result_record_count": len(normalization.records),
            },
            "metrics": {
                "leaderboard": {
                    metric_id: entry["normalized_score"]
                    for metric_id, entry in per_metric.items()
                    if entry.get("available") is True
                },
                "per_metric": per_metric,
                "summary": {
                    "sample_count": len(normalization.records),
                    "successful_samples": len(normalization.records),
                    "failed_samples": 0,
                    "available_metric_count": available_count,
                    "blocked_metric_count": blocked_count,
                },
            },
            "evaluation": {
                "available": available_count > 0,
                "kind": "official_results_normalizer",
                "evidence_level": "normalizer_only",
                "num_results": len(normalization.metric_results),
                "skip_count": blocked_count,
                "blocked_count": blocked_count,
            },
            "runtime": dict(runtime_spec),
            "artifacts": {
                "scorecard": _artifact_path(root, "scorecard.json"),
                "runner_runtime_report": str(runtime_report_path.resolve()),
                "raw_metric_table": str(raw_metric_path.resolve()),
                "stdout": data.get("stdout_path"),
                "stderr": data.get("stderr_path"),
            },
        }
        write_json(scorecard_path, scorecard)
        artifacts = {
            "scorecard": str(scorecard_path.resolve()),
            "runner_runtime_report": str(runtime_report_path.resolve()),
            "raw_metric_table": str(raw_metric_path.resolve()),
        }
        for key, value in (("stdout", data.get("stdout_path")), ("stderr", data.get("stderr_path"))):
            if value:
                artifacts[key] = str(value)
        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=raw_metric_path,
            official_benchmark_verified=False,
            integration_evidence=False,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                "run_status": data.get("run_status"),
                "returncode": data.get("returncode"),
                "runtime": dict(runtime_spec),
                "runner_runtime_spec": dict(runtime_spec),
                **extra_metadata,
                "normalization_ok": available_count > 0,
            },
        )

    def _specialized_normalization_result(
        self,
        specialized: Mapping[str, Any],
        *,
        root: Path,
        contract: Any,
        mode: str,
        runtime_spec: dict[str, Any],
        runtime_report: dict[str, Any],
        runtime_report_path: Path,
        extra_metadata: dict[str, Any],
    ) -> OfficialRunResult:
        """Finish a run whose results a benchmark-specific normalizer scored."""
        scorecard_path = root / "scorecard.json"
        runtime_report["command"] = specialized["command"]
        runtime_report["returncode"] = specialized["returncode"]
        runtime_report["duration_seconds"] = specialized["duration_seconds"]
        runtime_report["stdout_path"] = specialized["stdout_path"]
        runtime_report["stderr_path"] = specialized["stderr_path"]
        runtime_report["run_status"] = (
            "official_results_normalized" if specialized["ok"] else "official_results_missing_scores"
        )
        scorecard = specialized.get("scorecard")
        scorecard = dict(scorecard) if isinstance(scorecard, Mapping) else {}
        artifacts = dict(scorecard.get("artifacts") or {})
        artifacts["runner_runtime_report"] = str(runtime_report_path.resolve())
        scorecard["runtime"] = runtime_spec
        scorecard["artifacts"] = artifacts
        write_json(scorecard_path, scorecard)
        expected_artifact_checks = _expected_artifact_checks(self.entry, root)
        refreshed_flags = inspect_scorecard_runtime_flags(scorecard_path)
        scorecard_runtime_flags = {
            "official_benchmark_verified": refreshed_flags.get("official_benchmark_verified") is True,
            "integration_evidence": refreshed_flags.get("integration_evidence") is True,
            **refreshed_flags,
        }
        artifacts_ok = all(item.get("ok") is True for item in expected_artifact_checks)
        command_ok = specialized["returncode"] == 0
        runtime_report.update(
            {
                "error": specialized.get("error"),
                "expected_artifact_checks": expected_artifact_checks,
                "scorecard_runtime_flags": scorecard_runtime_flags,
                "official_benchmark_verified": (
                    command_ok and artifacts_ok and scorecard_runtime_flags.get("official_benchmark_verified") is True
                ),
                "integration_evidence": (
                    command_ok and artifacts_ok and scorecard_runtime_flags.get("integration_evidence") is True
                ),
            }
        )
        _write_runtime_report(root, runtime_report)
        raw_metric_path = root / "raw_metric_table.jsonl"
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=raw_metric_path if raw_metric_path.is_file() else None,
            official_benchmark_verified=scorecard.get("official_benchmark_verified") is True,
            integration_evidence=scorecard.get("integration_evidence") is True,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "normalizer_only": scorecard.get("normalizer_only") is not False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": _requires_external_runtime(
                    default=contract.requires_upstream_runtime,
                    runtime_spec=runtime_spec,
                ),
                "run_status": runtime_report["run_status"],
                "returncode": specialized["returncode"],
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
                "specialized_normalizer": True,
                "specialized_normalizer_error": specialized.get("error"),
                **extra_metadata,
                "normalization_ok": specialized.get("ok") is True,
            },
        )

    def _embodied_normalization_result(
        self,
        embodied: Mapping[str, Any],
        *,
        root: Path,
        contract: Any,
        mode: str,
        runtime_spec: dict[str, Any],
        runtime_report: dict[str, Any],
        extra_metadata: dict[str, Any],
    ) -> OfficialRunResult:
        """Finish a run whose results the embodied-benchmark normalizer scored."""
        scorecard_path = root / "scorecard.json"
        runtime_report["command"] = embodied["command"]
        runtime_report["returncode"] = embodied["returncode"]
        runtime_report["duration_seconds"] = embodied["duration_seconds"]
        runtime_report["stdout_path"] = embodied["stdout_path"]
        runtime_report["stderr_path"] = embodied["stderr_path"]
        runtime_report["run_status"] = (
            "official_results_normalized" if embodied["ok"] else "official_results_missing_scores"
        )
        runtime_report_path = _write_runtime_report(root, runtime_report)
        scorecard = embodied.get("scorecard")
        scorecard = dict(scorecard) if isinstance(scorecard, Mapping) else {}
        artifacts = dict(scorecard.get("artifacts") or {})
        artifacts["runner_runtime_report"] = str(runtime_report_path.resolve())
        scorecard["runtime"] = runtime_spec
        scorecard["artifacts"] = artifacts
        write_json(scorecard_path, scorecard)
        raw_metric_path = root / "raw_metric_table.jsonl"
        raw_results_path = root / "raw_results.jsonl"
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=(
                raw_metric_path
                if raw_metric_path.is_file()
                else raw_results_path
                if raw_results_path.is_file()
                else None
            ),
            official_benchmark_verified=scorecard.get("official_benchmark_verified") is True,
            integration_evidence=scorecard.get("integration_evidence") is True,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "normalizer_only": True,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                "run_status": runtime_report["run_status"],
                "returncode": embodied["returncode"],
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
                "embodied_normalizer": True,
                "embodied_normalizer_error": embodied.get("error"),
                **extra_metadata,
                "normalization_ok": embodied.get("ok") is True,
            },
        )

    def _contract_evaluation_result(
        self,
        *,
        root: Path,
        contract: Any,
        data: Mapping[str, Any],
        mode: str,
        results_path: Any,
        runtime_spec: dict[str, Any],
        runtime_report_path: Path,
        generated_artifact_dir: Any,
        extra_metadata: dict[str, Any],
    ) -> OfficialRunResult:
        """Score imported official results with the benchmark's contract evaluator."""
        evaluator_kwargs = dict(extra_metadata)
        evaluator_kwargs.pop("official_results_path", None)
        evaluator_kwargs.pop("judge_results_path", None)
        result = write_benchmark_contract_evaluation(
            benchmark_id=self.benchmark_id,
            display_name=self.entry.name or contract.display_name,
            official_metric_ids=contract.metric_ids,
            output_dir=root,
            generated_artifact_dir=None if generated_artifact_dir is None else str(generated_artifact_dir),
            manifest=self.load_manifest(),
            runner="benchmark_zoo_manifest_runner",
            mode=mode,
            official_results_path=str(results_path),
            **evaluator_kwargs,
        )
        scorecard_path = root / "scorecard.json"
        raw_metric_path = root / "raw_metric_table.jsonl"
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        artifacts = dict(result.get("artifacts") or {})
        artifacts["runner_runtime_report"] = str(runtime_report_path.resolve())
        scorecard["runtime"] = runtime_spec
        scorecard["artifacts"] = {**dict(scorecard.get("artifacts") or {}), **artifacts}
        write_json(scorecard_path, scorecard)
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=raw_metric_path,
            official_benchmark_verified=False,
            integration_evidence=False,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "normalizer_only": True,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                "run_status": data.get("run_status"),
                "returncode": data.get("returncode"),
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
                **extra_metadata,
                "normalization_ok": _scorecard_normalization_available(scorecard),
            },
        )

    def _write_missing_runtime_scorecard(
        self,
        *,
        root: Path,
        contract: Any,
        data: Mapping[str, Any],
        mode: str,
        runtime_spec: dict[str, Any],
        runtime_report_path: Path,
        generated_artifact_dir: Any,
        generated_files: list[str],
    ) -> None:
        """Write an unavailable scorecard when the official runtime produced none."""
        scorecard_path = root / "scorecard.json"
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "run": {
                "status": data.get("run_status", "failed"),
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_manifest_runner",
                "mode": mode,
                "command": data.get("command"),
                "returncode": data.get("returncode"),
                "error": data.get("error") or "official runtime did not write scorecard.json",
            },
            "benchmark": {
                "benchmark_id": self.benchmark_id,
                "name": self.entry.name or contract.display_name,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
            },
            "dataset": {
                "hf_dataset_ids": list(self.entry.hf_dataset_ids),
                "requires_auth": self.entry.requires_auth,
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "generated_file_count": len(generated_files),
            },
            "metrics": {
                "per_metric": {
                    metric_id: {
                        "available": False,
                        "reason": "official_runtime_scorecard_missing",
                    }
                    for metric_id in contract.metric_ids
                },
                "summary": {
                    "sample_count": len(generated_files),
                    "successful_samples": 0,
                    "failed_samples": len(generated_files),
                },
            },
            "evaluation": {
                "available": False,
                "kind": mode,
                "skip_count": len(contract.metric_ids),
            },
            "runtime": runtime_spec,
            "artifacts": {
                "scorecard": _artifact_path(root, "scorecard.json"),
                "runner_runtime_report": str(runtime_report_path.resolve()),
                "stdout": data.get("stdout_path"),
                "stderr": data.get("stderr_path"),
            },
        }
        write_json(scorecard_path, scorecard)

    def _normalize_official(self, collected: OfficialRunStage, *, mode: str) -> OfficialRunResult:
        """Route official normalization across specialized, embodied, and generic paths."""
        data = dict(collected.data)
        root = collected.output_dir
        contract = get_external_benchmark_contract(self.benchmark_id)
        generated_files = [str(path) for path in data.get("generated_files") or ()]
        generated_artifact_dir = data.get("generated_artifact_dir")
        scorecard_path = root / "scorecard.json"
        scorecard_flags = data.get("scorecard_runtime_flags")
        scorecard_flags = scorecard_flags if isinstance(scorecard_flags, Mapping) else {}
        scorecard_runtime_flags = {
            "official_benchmark_verified": scorecard_flags.get("official_benchmark_verified") is True,
            "integration_evidence": scorecard_flags.get("integration_evidence") is True,
            **dict(scorecard_flags),
        }
        runtime_spec = data.get("runtime")
        runtime_spec = (
            dict(runtime_spec)
            if isinstance(runtime_spec, Mapping)
            else _runner_runtime_spec(self.entry, benchmark_id=self.benchmark_id)
        )
        expected_artifact_checks = [
            dict(item) for item in data.get("expected_artifact_checks") or () if isinstance(item, Mapping)
        ]
        artifacts_ok = all(item.get("ok") is True for item in expected_artifact_checks)
        command_ok = data.get("returncode") == 0
        official_benchmark_verified = (
            command_ok and artifacts_ok and scorecard_flags.get("official_benchmark_verified") is True
        )
        integration_evidence = command_ok and artifacts_ok and scorecard_flags.get("integration_evidence") is True
        runtime_report = {
            "schema_version": "worldfoundry-official-benchmark-runtime-report",
            "benchmark_id": self.benchmark_id,
            "mode": mode,
            "command_kind": data.get("command_kind"),
            "command": data.get("command"),
            "returncode": data.get("returncode"),
            "run_status": data.get("run_status"),
            "duration_seconds": data.get("duration_seconds"),
            "workdir": data.get("workdir"),
            "runtime": runtime_spec,
            "stdout_path": data.get("stdout_path"),
            "stderr_path": data.get("stderr_path"),
            "error": data.get("error"),
            "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
            "generated_file_count": len(generated_files),
            "expected_artifact_checks": expected_artifact_checks,
            "scorecard_runtime_flags": scorecard_runtime_flags,
            "official_benchmark_verified": official_benchmark_verified,
            "integration_evidence": integration_evidence,
            "manifest": self.load_manifest(),
        }
        runtime_report_path = _write_runtime_report(root, runtime_report)

        results_path = runtime_spec.get("results_path")
        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        score_dir = extra_metadata.get("score_dir") if self.benchmark_id == "camerabench" else None
        has_score_dir = score_dir not in (None, "") and Path(str(score_dir)).is_dir()
        has_results_path = bool(
            "results_path" in runtime_spec
            and runtime_spec["results_path"]
            and Path(str(runtime_spec["results_path"])).exists()
        )
        if has_results_path or has_score_dir:
            specialized = _run_specialized_result_normalizer(
                benchmark_id=self.benchmark_id,
                output_dir=root,
                results_path=Path(str(results_path)) if has_results_path else None,
                runtime_spec=runtime_spec,
                generated_artifact_dir=generated_artifact_dir,
                run_official=(
                    mode == "official-run"
                    and generated_artifact_dir is not None
                    and self.benchmark_id in artifact_official_run_benchmarks()
                ),
                kwargs=extra_metadata,
            )
            if specialized is not None:
                return self._specialized_normalization_result(
                    specialized,
                    root=root,
                    contract=contract,
                    mode=mode,
                    runtime_spec=runtime_spec,
                    runtime_report=runtime_report,
                    runtime_report_path=runtime_report_path,
                    extra_metadata=extra_metadata,
                )
            embodied = _run_embodied_result_normalizer(
                benchmark_id=self.benchmark_id,
                output_dir=root,
                results_path=Path(str(results_path)),
                runtime_spec=runtime_spec,
                kwargs=extra_metadata,
            )
            if embodied is not None:
                return self._embodied_normalization_result(
                    embodied,
                    root=root,
                    contract=contract,
                    mode=mode,
                    runtime_spec=runtime_spec,
                    runtime_report=runtime_report,
                    extra_metadata=extra_metadata,
                )
            if has_benchmark_contract_evaluator(self.benchmark_id):
                return self._contract_evaluation_result(
                    root=root,
                    contract=contract,
                    data=data,
                    mode=mode,
                    results_path=results_path,
                    runtime_spec=runtime_spec,
                    runtime_report_path=runtime_report_path,
                    generated_artifact_dir=generated_artifact_dir,
                    extra_metadata=extra_metadata,
                )
            return self._normalize_generic_official_results(
                root=root,
                contract=contract,
                data=data,
                mode=mode,
                results_path=Path(str(results_path)),
                runtime_spec=runtime_spec,
                runtime_report_path=runtime_report_path,
                generated_artifact_dir=generated_artifact_dir,
                generated_files=generated_files,
            )

        if not scorecard_path.is_file():
            self._write_missing_runtime_scorecard(
                root=root,
                contract=contract,
                data=data,
                mode=mode,
                runtime_spec=runtime_spec,
                runtime_report_path=runtime_report_path,
                generated_artifact_dir=generated_artifact_dir,
                generated_files=generated_files,
            )
        artifacts = {
            "scorecard": str(scorecard_path.resolve()),
            "runner_runtime_report": str(runtime_report_path.resolve()),
        }
        for key, value in (("stdout", data.get("stdout_path")), ("stderr", data.get("stderr_path"))):
            if value:
                artifacts[key] = str(value)

        normalizer_only = scorecard_runtime_flags.get("validation_normalizer_only") is True or (
            scorecard_runtime_flags.get("normalization_ok") is True
            and not official_benchmark_verified
            and not integration_evidence
        )
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=runtime_report_path,
            official_benchmark_verified=official_benchmark_verified,
            integration_evidence=integration_evidence,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                "run_status": data.get("run_status"),
                "returncode": data.get("returncode"),
                "normalizer_only": normalizer_only,
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
                **extra_metadata,
                "normalization_ok": scorecard_runtime_flags.get("normalization_ok") is True,
            },
        )

    def evaluate(
        self,
        *,
        output_dir: str | Path,
        mode: str = "contract",
        generated_artifact_dir: str | Path | None = None,
        **kwargs: JsonValue,
    ) -> OfficialRunResult:
        """Run ``prepare → run → collect → normalize``."""
        prepared = self.prepare(
            output_dir=output_dir,
            mode=mode,
            generated_artifact_dir=generated_artifact_dir,
            **kwargs,
        )
        run_result = self.run(prepared)
        collected = self.collect(run_result)
        return self.normalize(collected)


class BenchmarkRunnerRegistry:
    """Lookup table of benchmark-zoo entries with contract runner surfaces."""

    def __init__(
        self,
        entries: Iterable[BenchmarkZooEntry | Mapping[str, JsonValue]],
        *,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.zoo = BenchmarkZooRegistry(entries)
        self.manifest_path = None if manifest_path is None else Path(manifest_path)

    def __contains__(self, key: object) -> bool:
        return key in self.zoo

    def __len__(self) -> int:
        return len(self.zoo)

    def __iter__(self) -> Iterator[BenchmarkZooEntry]:
        return iter(self.list_entries())

    def list_entries(self) -> list[BenchmarkZooEntry]:
        """Return sorted registered benchmark entries."""
        return self.zoo.list()

    def by_integration_status(self, status: str) -> list[BenchmarkZooEntry]:
        """Filter entries by ``integration_status``."""
        return self.zoo.by_integration_status(status)

    def integrated(self) -> list[BenchmarkZooEntry]:
        """Return entries with ``integration_status == integrated``."""
        return self.by_integration_status("integrated")

    def planned(self) -> list[BenchmarkZooEntry]:
        """Return entries with ``integration_status == planned``."""
        return self.by_integration_status("planned")

    def blocked(self) -> list[BenchmarkZooEntry]:
        """Return entries with ``integration_status == blocked``."""
        return self.by_integration_status("blocked")

    def has_contract_runner(self, benchmark_id: str) -> bool:
        """Return True when an external benchmark contract exists."""
        entry = self.zoo.get(benchmark_id)
        try:
            get_external_benchmark_contract(entry.benchmark_id)
        except KeyError:
            return False
        return True

    def has_official_runner(self, benchmark_id: str) -> bool:
        """Return True when manifest marks verified official integration."""
        entry = self.zoo.get(benchmark_id)
        return (
            entry.integration_status == "integrated"
            and entry.verification_status == "verified"
            and entry.official_benchmark_verified
            and entry.integration_evidence
        )

    def has_runner(self, benchmark_id: str) -> bool:
        """Return True when a contract runner is available."""
        return self.has_contract_runner(benchmark_id)

    def get_runner(self, benchmark_id: str) -> ManifestBenchmarkRunner:
        """Return :class:`ManifestBenchmarkRunner` for ``benchmark_id``."""
        try:
            entry = self.zoo.get(benchmark_id)
        except UnknownBenchmarkZooKeyError:
            raise
        if not self.has_runner(benchmark_id):
            raise BenchmarkExecutionUnavailableError(
                f"{entry.benchmark_id!r} has no registered contract runner; "
                f"manifest status is {entry.integration_status}/{entry.verification_status}"
            )
        return ManifestBenchmarkRunner(entry=entry, manifest_path=self.manifest_path)


def build_benchmark_runner_registry(path: str | Path = DEFAULT_MANIFEST_PATH) -> BenchmarkRunnerRegistry:
    """Load benchmark-zoo manifests and build a :class:`BenchmarkRunnerRegistry`."""
    manifest_path = Path(path)
    entries = (
        load_benchmark_zoo_registry(manifest_path).list() if manifest_path.is_dir() else load_entries(manifest_path)
    )
    return BenchmarkRunnerRegistry(entries, manifest_path=manifest_path)


def run_benchmark_execution(
    benchmark_id: str,
    *,
    output_dir: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    mode: str = "contract",
    generated_artifact_dir: str | Path | None = None,
    **kwargs: JsonValue,
) -> OfficialRunResult:
    """Run full benchmark lifecycle for ``benchmark_id``."""
    with log_context(benchmark_id=benchmark_id, phase="benchmark.execution"):
        logger = get_logger(__name__)
        logger.event(
            "INFO",
            "benchmark.execution.started",
            "Benchmark execution started",
            output_dir=str(output_dir),
            mode=mode,
        )
        try:
            registry = build_benchmark_runner_registry(manifest_path)
            runner = registry.get_runner(benchmark_id)
            result = runner.evaluate(
                output_dir=output_dir,
                mode=mode,
                generated_artifact_dir=generated_artifact_dir,
                **kwargs,
            )
        except Exception:
            logger.event(
                "ERROR",
                "benchmark.execution.failed",
                "Benchmark execution failed",
                exc_info=True,
                output_dir=str(output_dir),
                mode=mode,
            )
            raise
        logger.event(
            "INFO" if result.ok else "ERROR",
            "benchmark.execution.finished",
            "Benchmark execution finished",
            status="ok" if result.ok else "failed",
            output_dir=str(result.output_dir),
        )
        return result

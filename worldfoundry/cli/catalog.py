# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Model, benchmark, and runtime-profile catalog rows shared by the TUI and the MCP server."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.cli.inference_variants import (
    INFER_GROUP_TASKS,
    INFER_MODEL_FAMILY_GROUPS,
    INFER_MODEL_VARIANTS,
    SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID,
    _dedupe_text,
    _infer_group_from_workload,
    _script_infer_family_id,
    _studio_inference_spec,
    _tui_model_id_for_studio_entry,
    _unique_studio_entries,
    _workload_from_template,
    infer_variant_options,
    is_script_infer_model_id,
)
from worldfoundry.evals.utils import (
    BENCHMARK_ZOO_DIR,
    MODEL_RUNTIME_ENVIRONMENTS_ROOT,
    MODEL_RUNTIME_PROFILES_ROOT,
    MODEL_ZOO_DIR,
    load_manifest,
    load_manifest_collection,
)
from worldfoundry.runtime.env import resolve_cache_dir

# ── Default directories ──────────────────────────────────────────
DEFAULT_MODEL_ZOO_DIR = MODEL_ZOO_DIR


DEFAULT_BENCHMARK_ZOO_DIR = BENCHMARK_ZOO_DIR


DEFAULT_RUNTIME_PROFILES_PATH = MODEL_RUNTIME_PROFILES_ROOT


DEFAULT_CONDA_ENVS_PATH = MODEL_RUNTIME_ENVIRONMENTS_ROOT


DEFAULT_INFERENCE_EVIDENCE_PATH = MODEL_ZOO_DIR.parent / "validation" / "inference_evidence.yaml"


@dataclass(frozen=True)
class ModelCatalogRow:
    """Row representation of a model zoo entry for the TUI catalog tables.

    Attributes:
        model_id: Unique model family identifier.
        name: Human-readable display name.
        tasks: Supported task labels (e.g. ``"text-to-video"``).
        provider: Optional model provider / source label.
        source_status: Repository availability status.
        integration_status: Integration readiness level.
        runner_status: Runner verification status.
        runner_kind: Runner type — ``"infer_script"`` or ``"studio_runtime"``.
        runnable_variants: Variant IDs that have runnable infer scripts.
        runtime_profile: Optional runtime profile identifier.
        min_vram_gb: Minimum VRAM requirement in GB, or ``None``.
        hf_repo_ids: HuggingFace repo identifiers for weights.
        official_repo_url: Official repository URL, or ``None``.
        requires_auth: Whether HuggingFace gated access is required.
        notes: Free-form notes tuple.
    """

    model_id: str
    name: str
    tasks: tuple[str, ...]
    provider: str | None
    source_status: str
    integration_status: str
    runner_status: str
    runner_kind: str
    runnable_variants: tuple[str, ...]
    runtime_profile: str | None
    min_vram_gb: float | None
    hf_repo_ids: tuple[str, ...]
    official_repo_url: str | None
    requires_auth: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "tasks": list(self.tasks),
            "provider": self.provider,
            "source_status": self.source_status,
            "integration_status": self.integration_status,
            "runner_status": self.runner_status,
            "runner_kind": self.runner_kind,
            "runnable_variants": list(self.runnable_variants),
            "runtime_profile": self.runtime_profile,
            "min_vram_gb": self.min_vram_gb,
            "hf_repo_ids": list(self.hf_repo_ids),
            "official_repo_url": self.official_repo_url,
            "requires_auth": self.requires_auth,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkCatalogRow:
    """Row representation of a benchmark zoo entry for the TUI catalog tables.

    Attributes:
        benchmark_id: Unique benchmark identifier.
        name: Human-readable display name.
        domains: Evaluation domain labels (e.g. ``"dynamic_photorealistic"``).
        modalities: Supported modality labels.
        tags: Additional tagging metadata.
        source_status: Dataset availability status.
        integration_status: Integration readiness level.
        verification_status: Verification / correctness status.
        maturity: Maturity level label.
        official_benchmark_verified: Whether the official benchmark was verified.
        integration_evidence: Whether integration evidence is available.
        leaderboard_valid: Whether the benchmark supports leaderboard submission.
        runner_target: Target runner identifier, or ``None``.
        install_profile: Conda / pip install profile, or ``None``.
        metrics: Metric identifiers supported by this benchmark.
        hf_dataset_ids: HuggingFace dataset identifiers.
        official_repo_url: Official repository URL, or ``None``.
        paper_url: Paper URL, or ``None``.
        requires_auth: Whether gated dataset access is required.
        notes: Free-form notes tuple.
    """

    benchmark_id: str
    name: str
    domains: tuple[str, ...]
    modalities: tuple[str, ...]
    tags: tuple[str, ...]
    source_status: str
    integration_status: str
    verification_status: str
    maturity: str
    official_benchmark_verified: bool
    integration_evidence: bool
    leaderboard_valid: bool
    runner_target: str | None
    install_profile: str | None
    metrics: tuple[str, ...]
    hf_dataset_ids: tuple[str, ...]
    official_repo_url: str | None
    paper_url: str | None
    requires_auth: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "domains": list(self.domains),
            "modalities": list(self.modalities),
            "tags": list(self.tags),
            "source_status": self.source_status,
            "integration_status": self.integration_status,
            "verification_status": self.verification_status,
            "maturity": self.maturity,
            "official_benchmark_verified": self.official_benchmark_verified,
            "integration_evidence": self.integration_evidence,
            "leaderboard_valid": self.leaderboard_valid,
            "runner_target": self.runner_target,
            "install_profile": self.install_profile,
            "metrics": list(self.metrics),
            "hf_dataset_ids": list(self.hf_dataset_ids),
            "official_repo_url": self.official_repo_url,
            "paper_url": self.paper_url,
            "requires_auth": self.requires_auth,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RuntimeProfileRow:
    """Row representation of a runtime profile entry for the TUI catalog.

    Attributes:
        profile_id: Profile identifier (typically matches a model ID).
        task_family: Task family the profile covers, or ``None``.
        artifact_kind: Kind of artifact produced, or ``None``.
        backend_stage: Backend stage label.
        integration_status: Integration readiness level.
        runtime_status: Runtime / execution readiness level.
        conda_env_name: Conda environment name, or ``None``.
        conda_env_prefix: Conda environment prefix path, or ``None``.
        conda_driver_status: Conda driver availability status, or ``None``.
        conda_env_exists: Whether the Conda environment directory exists.
        conda_python_exists: Whether ``python`` exists inside the Conda environment.
        validation_imports: Import paths validated for the environment.
        notes: Free-form notes tuple.
    """

    profile_id: str
    task_family: str | None
    artifact_kind: str | None
    backend_stage: str
    integration_status: str
    runtime_status: str
    conda_env_name: str | None
    conda_env_prefix: str | None
    conda_driver_status: str | None
    conda_env_exists: bool
    conda_python_exists: bool
    validation_imports: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "task_family": self.task_family,
            "artifact_kind": self.artifact_kind,
            "backend_stage": self.backend_stage,
            "integration_status": self.integration_status,
            "runtime_status": self.runtime_status,
            "conda_env_name": self.conda_env_name,
            "conda_env_prefix": self.conda_env_prefix,
            "conda_driver_status": self.conda_driver_status,
            "conda_env_exists": self.conda_env_exists,
            "conda_python_exists": self.conda_python_exists,
            "validation_imports": list(self.validation_imports),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Catalog:
    """Aggregated catalog container for all TUI data.

    Attributes:
        models: Ordered model catalog rows.
        benchmarks: Ordered benchmark catalog rows.
        runtime_profiles: Ordered runtime profile rows.
        conda_status: Mapping of Conda availability diagnostics, or ``None``.
    """

    models: tuple[ModelCatalogRow, ...]
    benchmarks: tuple[BenchmarkCatalogRow, ...]
    runtime_profiles: tuple[RuntimeProfileRow, ...] = ()
    conda_status: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "models": [row.to_dict() for row in self.models],
            "benchmarks": [row.to_dict() for row in self.benchmarks],
            "runtime_profiles": [row.to_dict() for row in self.runtime_profiles],
            "conda_status": dict(self.conda_status or {}),
            "summary": {
                "models": len(self.models),
                "benchmarks": len(self.benchmarks),
                "runtime_profiles": len(self.runtime_profiles),
                "runnable_models": sum(1 for row in self.models if row.runner_kind == "runnable_runner"),
                "integrated_benchmarks": sum(1 for row in self.benchmarks if row.integration_status == "integrated"),
                "leaderboard_ready_benchmarks": sum(1 for row in self.benchmarks if row.leaderboard_valid),
                "conda_envs_existing": int((self.conda_status or {}).get("envs_existing", 0)),
            },
        }


def _model_tasks(entry) -> tuple[str, ...]:
    """Collect deduplicated task labels from a model entry and its variants."""
    variant_tasks = [variant.task for variant in entry.variants if variant.task]
    return _dedupe_text((*entry.tasks, *variant_tasks))


def _runnable_variant_ids(entry) -> tuple[str, ...]:
    """Return variant IDs that have runnable runner entries."""
    return _dedupe_text(variant.variant_id for variant in entry.variants if variant.is_runnable_runner_entry)


def _model_runtime_profile(entry) -> str | None:
    """Return the first non-empty runtime profile identifier from an entry or its variants."""
    if entry.runtime_profile:
        return entry.runtime_profile
    for variant in entry.variants:
        if variant.runtime_profile:
            return variant.runtime_profile
    return None


def _model_min_vram(entry) -> float | None:
    """Return the minimum VRAM requirement across the entry and its variants, or ``None``."""
    values = [entry.min_vram_gb, *(variant.min_vram_gb for variant in entry.variants)]
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return min(concrete)


def _model_to_row(entry) -> ModelCatalogRow:
    """Convert a model zoo registry entry to a :class:`ModelCatalogRow`."""
    return ModelCatalogRow(
        model_id=entry.model_id,
        name=entry.name or entry.model_id,
        tasks=_model_tasks(entry),
        provider=entry.provider,
        source_status=entry.source_status,
        integration_status=entry.integration_status,
        runner_status=entry.verification_status,
        runner_kind=entry.runner_entry_kind,
        runnable_variants=_runnable_variant_ids(entry),
        runtime_profile=_model_runtime_profile(entry),
        min_vram_gb=_model_min_vram(entry),
        hf_repo_ids=entry.hf_repo_ids,
        official_repo_url=entry.official_repo_url,
        requires_auth=entry.requires_auth,
        notes=entry.notes,
    )


def _normalised_evidence_id(value: object) -> str:
    """Normalize model identifiers for evidence matching across punctuation aliases."""
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


@lru_cache(maxsize=1)
def _verified_inference_evidence_ids() -> frozenset[str]:
    """Load model IDs backed by recorded real-inference evidence."""
    try:
        payload = load_manifest(DEFAULT_INFERENCE_EVIDENCE_PATH)
    except (OSError, ValueError):
        return frozenset()
    verified = {
        _normalised_evidence_id(model_id)
        for model_id in payload.get("verified_methods") or ()
        if _normalised_evidence_id(model_id)
    }
    for model_id, item in (payload.get("methods") or {}).items():
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "").strip().casefold()
        if status not in {"verified", "pass", "passed", "complete", "completed"}:
            continue
        verified.add(_normalised_evidence_id(model_id))
        verified.update(
            _normalised_evidence_id(alias) for alias in item.get("aliases") or () if _normalised_evidence_id(alias)
        )
    return frozenset(verified)


def _inference_runner_status(
    model_id: str,
    variants: Sequence[str] = (),
    aliases: Sequence[str] = (),
) -> str:
    """Distinguish recorded real inference from a structurally valid contract."""
    verified = _verified_inference_evidence_ids()
    if any(_normalised_evidence_id(value) in verified for value in (model_id, *aliases)):
        return "verified"
    variant_matches = [_normalised_evidence_id(variant) in verified for variant in variants]
    if variant_matches and all(variant_matches):
        return "verified"
    if any(variant_matches):
        return "partially_verified"
    return "contract_ready"


def _infer_virtual_model_row(model_id: str, group: str) -> ModelCatalogRow:
    """Build a :class:`ModelCatalogRow` for a script-infer model not present in the zoo registry."""
    return ModelCatalogRow(
        model_id=model_id,
        name=model_id,
        tasks=INFER_GROUP_TASKS.get(group, (group,)),
        provider=None,
        source_status="in_tree",
        integration_status="integrated",
        runner_status=_inference_runner_status(model_id, INFER_MODEL_VARIANTS.get(model_id, ())),
        runner_kind="infer_script",
        runnable_variants=INFER_MODEL_VARIANTS.get(model_id, ()),
        runtime_profile=None,
        min_vram_gb=None,
        hf_repo_ids=(),
        official_repo_url=None,
        requires_auth=False,
        notes=(f"scripts/inference/run_infer.sh --category {group}",),
    )


def _merge_script_infer_row(existing: ModelCatalogRow | None, script_row: ModelCatalogRow) -> ModelCatalogRow:
    """Merge a script-infer row into an existing model row, overlaying infer-specific metadata."""
    if existing is None:
        return script_row
    return replace(
        existing,
        integration_status="integrated",
        runner_status=script_row.runner_status,
        tasks=_dedupe_text((*script_row.tasks, *existing.tasks)),
        runnable_variants=script_row.runnable_variants,
        runtime_profile=existing.runtime_profile or script_row.runtime_profile,
        min_vram_gb=existing.min_vram_gb if existing.min_vram_gb is not None else script_row.min_vram_gb,
        hf_repo_ids=_dedupe_text((*existing.hf_repo_ids, *script_row.hf_repo_ids)),
        notes=_dedupe_text((*existing.notes, *script_row.notes)),
    )


def _studio_infer_model_row(entry: Any) -> ModelCatalogRow:
    """Build a :class:`ModelCatalogRow` from a studio catalog entry with runtime metadata."""
    from worldfoundry.studio.inference.catalog import template_id_hint

    template_id = template_id_hint(entry)
    workload = _workload_from_template(template_id)
    group = _infer_group_from_workload(workload)
    spec = _studio_inference_spec(entry.model_id)
    task_names = tuple(task.label or task.task_id for task in spec.tasks) if spec is not None else ()
    variants = tuple(variant.variant_id for variant in spec.variants) if spec is not None else ("default",)
    return ModelCatalogRow(
        model_id=entry.model_id,
        name=entry.display_name or entry.model_id,
        tasks=_dedupe_text((*INFER_GROUP_TASKS.get(group, (group,)), *task_names)),
        provider=None,
        source_status="in_tree",
        integration_status="integrated",
        runner_status=_inference_runner_status(
            entry.model_id,
            variants,
            tuple(getattr(entry, "aliases", ()) or ()),
        ),
        runner_kind="studio_runtime",
        runnable_variants=variants,
        runtime_profile=None,
        min_vram_gb=None,
        hf_repo_ids=(entry.default_model_ref,) if entry.default_model_ref and "/" in entry.default_model_ref else (),
        official_repo_url=None,
        requires_auth=False,
        notes=("studio_runtime", "worldfoundry.studio.workspace_job infer"),
    )


def _row_has_studio_runtime(row: ModelCatalogRow) -> bool:
    """Return whether *row* carries a ``studio_runtime`` runner kind or note."""
    return row.runner_kind == "studio_runtime" or any("studio_runtime" in note for note in row.notes)


def is_infer_model_row(row: ModelCatalogRow) -> bool:
    """Return whether *row* supports inference (script-based or studio-runtime)."""
    script_family = _script_infer_family_id(row.model_id)
    if script_family in INFER_MODEL_VARIANTS:
        tui_family = SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID.get(script_family, script_family)
        return row.model_id == tui_family and bool(infer_variant_options(row.model_id))
    return _row_has_studio_runtime(row)


def _merge_studio_infer_row(existing: ModelCatalogRow | None, studio_row: ModelCatalogRow) -> ModelCatalogRow:
    """Merge a studio-runtime row into an existing model row, enriching metadata where applicable."""
    if existing is None:
        return studio_row
    if is_script_infer_model_id(existing.model_id):
        return replace(
            existing,
            integration_status="integrated",
            runner_status=existing.runner_status,
            tasks=_dedupe_text((*existing.tasks, *studio_row.tasks)),
            hf_repo_ids=_dedupe_text((*existing.hf_repo_ids, *studio_row.hf_repo_ids)),
            notes=_dedupe_text((*existing.notes, "workspace_job infer available")),
        )
    return replace(
        existing,
        integration_status="integrated",
        runner_status=studio_row.runner_status,
        runner_kind="studio_runtime",
        tasks=_dedupe_text((*existing.tasks, *studio_row.tasks)),
        runnable_variants=studio_row.runnable_variants,
        hf_repo_ids=_dedupe_text((*existing.hf_repo_ids, *studio_row.hf_repo_ids)),
        notes=_dedupe_text((*existing.notes, *studio_row.notes)),
    )


def _benchmark_to_row(entry) -> BenchmarkCatalogRow:
    """Convert a benchmark zoo registry entry to a :class:`BenchmarkCatalogRow`."""
    return BenchmarkCatalogRow(
        benchmark_id=entry.benchmark_id,
        name=entry.name or entry.benchmark_id,
        domains=entry.domains,
        modalities=entry.modalities,
        tags=entry.tags,
        source_status=entry.source_status,
        integration_status=entry.integration_status,
        verification_status=entry.verification_status,
        maturity=entry.maturity or "planned",
        official_benchmark_verified=entry.official_benchmark_verified,
        integration_evidence=entry.integration_evidence,
        leaderboard_valid=entry.leaderboard_valid,
        runner_target=entry.runner_target,
        install_profile=entry.install_profile,
        metrics=tuple(metric.metric_id for metric in entry.metrics),
        hf_dataset_ids=entry.hf_dataset_ids,
        official_repo_url=entry.official_repo_url,
        paper_url=entry.paper_url,
        requires_auth=entry.requires_auth,
        notes=entry.notes,
    )


def _read_manifest_mapping(path: Path, *, item_key: str) -> dict[str, Any]:
    """Read a YAML manifest file or collection directory into a mapping.

    Raises:
        TypeError: When the parsed payload is not a ``dict``.
    """
    if not path.exists():
        return {}
    payload = load_manifest_collection(path, item_key=item_key) if path.is_dir() else load_manifest(path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return payload


def _runtime_profile_path(runtime_profile_dir: str | Path | None) -> Path:
    """Resolve the runtime profile YAML path from an override directory or the default location."""
    if runtime_profile_dir is None:
        return DEFAULT_RUNTIME_PROFILES_PATH
    path = Path(runtime_profile_dir)
    if path.is_file() or (path.is_dir() and (path / "_manifest.yaml").is_file()):
        return path
    return path / DEFAULT_RUNTIME_PROFILES_PATH.name


def _conda_env_path(runtime_profile_dir: str | Path | None) -> Path:
    """Resolve the Conda env YAML path from an override directory or the default location."""
    if runtime_profile_dir is None:
        return DEFAULT_CONDA_ENVS_PATH
    path = Path(runtime_profile_dir)
    if path.is_file() or (path.is_dir() and (path / "_manifest.yaml").is_file()):
        return path
    return path / DEFAULT_CONDA_ENVS_PATH.name


def _conda_env_root(payload: Mapping[str, Any], env_root: str | Path | None = None) -> Path:
    """Resolve the Conda envs root directory from manifest defaults, env overrides, or the fallback cache."""
    configured = None if env_root is None else str(env_root)
    defaults = payload.get("defaults")
    if configured in {None, ""} and isinstance(defaults, Mapping):
        configured = str(defaults.get("env_root") or "")
    resolved = _expand_configured_path(configured)
    return resolved if resolved is not None else resolve_cache_dir() / "conda_envs"


def _expand_configured_path(value: object | None) -> Path | None:
    """Expand environment variables and ``~`` in a configured path string, returning ``None`` for unresolved values."""
    if value is None:
        return None
    text = os.path.expandvars(str(value)).strip()
    if not text or "$" in text:
        return None
    return Path(text).expanduser()


def _conda_specs_by_model(
    path: Path,
    *,
    conda_envs_root: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Any]]:
    """Load Conda env specs per model from the manifest, computing existence and Python checks.

    Returns:
        A tuple of ``(specs_by_model_id, env_root, conda_status_mapping)``.
    """
    payload = _read_manifest_mapping(path, item_key="envs")
    env_root = _conda_env_root(payload, conda_envs_root)
    specs: dict[str, dict[str, Any]] = {}
    for item in payload.get("envs") or ():
        if not isinstance(item, Mapping) or not item.get("model_id"):
            continue
        spec = dict(item)
        env_name = str(spec.get("env_name") or spec["model_id"])
        spec_env_root = _expand_configured_path(spec.get("env_root")) or env_root
        env_prefix = spec_env_root / env_name
        spec["env_prefix"] = str(env_prefix)
        spec["env_exists"] = env_prefix.exists()
        spec["python_exists"] = (env_prefix / "bin" / "python").is_file() or (env_prefix / "python.exe").is_file()
        specs[str(spec["model_id"])] = spec

    conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    conda_status: dict[str, Any] = {
        "manifest_path": str(path),
        "manifest_found": path.exists(),
        "conda_executable": conda_executable,
        "conda_available": conda_executable is not None,
        "active_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "active_prefix": os.environ.get("CONDA_PREFIX"),
        "env_root": str(env_root),
        "env_specs": len(specs),
        "envs_existing": sum(1 for spec in specs.values() if spec.get("env_exists") is True),
        "envs_with_python": sum(1 for spec in specs.values() if spec.get("python_exists") is True),
    }
    return specs, env_root, conda_status


def _profile_to_row(item: Mapping[str, Any], conda_specs: Mapping[str, Mapping[str, Any]]) -> RuntimeProfileRow | None:
    """Convert a runtime profile manifest item to a :class:`RuntimeProfileRow`, merging Conda spec data."""
    profile_id = item.get("id") or item.get("model_id")
    if not profile_id:
        return None
    spec = conda_specs.get(str(profile_id), {})
    # Schema v2 profiles keep status fields under ``execution``.
    execution = item.get("execution")
    status = {**(execution if isinstance(execution, Mapping) else {}), **item}
    return RuntimeProfileRow(
        profile_id=str(profile_id),
        task_family=None if item.get("task_family") is None else str(item.get("task_family")),
        artifact_kind=None if item.get("artifact_kind") is None else str(item.get("artifact_kind")),
        backend_stage=str(status.get("backend_stage") or "unknown"),
        integration_status=str(status.get("integration_status") or "unknown"),
        runtime_status=str(status.get("runtime_status") or "unknown"),
        conda_env_name=None if spec.get("env_name") is None else str(spec.get("env_name")),
        conda_env_prefix=None if spec.get("env_prefix") is None else str(spec.get("env_prefix")),
        conda_driver_status=None if spec.get("driver_status") is None else str(spec.get("driver_status")),
        conda_env_exists=spec.get("env_exists") is True,
        conda_python_exists=spec.get("python_exists") is True,
        validation_imports=_dedupe_text(str(value) for value in spec.get("validation_imports") or ()),
        notes=_dedupe_text(str(value) for value in item.get("notes") or ()),
    )


def load_runtime_profile_rows(
    *,
    runtime_profile_dir: str | Path | None = None,
    conda_envs_root: str | Path | None = None,
) -> tuple[tuple[RuntimeProfileRow, ...], dict[str, object]]:
    """Load runtime profile rows and Conda status diagnostics from YAML manifests.

    Args:
        runtime_profile_dir: Override for the runtime profile directory.
        conda_envs_root: Override for the Conda envs root directory.

    Returns:
        A tuple of (sorted runtime profile rows, Conda status mapping).
    """
    profiles_payload = _read_manifest_mapping(_runtime_profile_path(runtime_profile_dir), item_key="profiles")
    conda_specs, _env_root, conda_status = _conda_specs_by_model(
        _conda_env_path(runtime_profile_dir),
        conda_envs_root=conda_envs_root,
    )
    rows = [
        row
        for item in profiles_payload.get("profiles") or ()
        if isinstance(item, Mapping)
        for row in (_profile_to_row(item, conda_specs),)
        if row is not None
    ]
    return tuple(sorted(rows, key=lambda row: row.profile_id)), conda_status


def load_catalog(
    *,
    model_manifest_dir: str | Path | None = None,
    benchmark_manifest_dir: str | Path | None = None,
    runtime_profile_dir: str | Path | None = None,
    conda_envs_root: str | Path | None = None,
) -> Catalog:
    """Load and merge all model, benchmark, and runtime-profile catalog data.

    Merges registry entries with script-infer and studio-runtime rows to produce
    a unified :class:`Catalog` ready for TUI rendering.

    Args:
        model_manifest_dir: Override for the model zoo manifest directory.
        benchmark_manifest_dir: Override for the benchmark zoo manifest directory.
        runtime_profile_dir: Override for the runtime profile directory.
        conda_envs_root: Override for the Conda envs root directory.

    Returns:
        A fully populated :class:`Catalog`.
    """
    from worldfoundry.evals.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
    from worldfoundry.zoo.catalog import load_model_zoo_registry

    # ── Load registries ──
    model_registry = load_model_zoo_registry(model_manifest_dir or DEFAULT_MODEL_ZOO_DIR)
    benchmark_registry = load_benchmark_zoo_registry(benchmark_manifest_dir or DEFAULT_BENCHMARK_ZOO_DIR)
    runtime_profiles, conda_status = load_runtime_profile_rows(
        runtime_profile_dir=runtime_profile_dir,
        conda_envs_root=conda_envs_root,
    )
    # ── Build model rows and merge script-infer / studio-runtime entries ──
    model_rows = {row.model_id: row for row in (_model_to_row(entry) for entry in model_registry)}
    # ── Merge script-infer virtual rows ──
    for model_id, group in INFER_MODEL_FAMILY_GROUPS.items():
        script_row = _infer_virtual_model_row(model_id, group)
        tui_model_id = SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID.get(model_id, model_id)
        script_row = replace(script_row, model_id=tui_model_id)
        model_rows[tui_model_id] = _merge_script_infer_row(model_rows.get(tui_model_id), script_row)
    # ── Merge studio-runtime rows ──
    for entry in _unique_studio_entries():
        tui_model_id = _tui_model_id_for_studio_entry(entry)
        studio_row = replace(_studio_infer_model_row(entry), model_id=tui_model_id)
        model_rows[tui_model_id] = _merge_studio_infer_row(model_rows.get(tui_model_id), studio_row)
    # ── Sort and build the final catalog ──
    models = tuple(sorted(model_rows.values(), key=lambda row: row.model_id))
    benchmarks = tuple(
        sorted((_benchmark_to_row(entry) for entry in benchmark_registry), key=lambda row: row.benchmark_id)
    )
    return Catalog(
        models=models,
        benchmarks=benchmarks,
        runtime_profiles=runtime_profiles,
        conda_status=conda_status,
    )

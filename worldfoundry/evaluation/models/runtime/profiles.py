"""Runtime profiles: model metadata, catalog entries, and command-template synthesis.

Defines :class:`RuntimeProfile` for declaring model identity, task family, source
repositories, and checkpoints, plus :class:`RuntimeProfileSynthesis` for planning or
executing model inference via command-line templates.  Loaders merge model-catalog
manifests with per-model runtime profiles into a unified ``dict[str, RuntimeProfile]``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml

from worldfoundry.runtime.env import resolve_hfd_root
from worldfoundry.runtime.cuda_tiers import SUPPORTED_CUDA_TIERS, unified_env_exists
from worldfoundry.evaluation.models.catalog.schema import iter_model_zoo_payloads
from worldfoundry.evaluation.utils import DATA_ROOT, REPO_ROOT
from worldfoundry.evaluation.utils import load_manifest, load_manifest_collection, manifest_paths
from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from worldfoundry.runtime.conda import (
    load_runtime_conda_env_specs_with_overrides,
    resolve_conda_env_context,
    resolve_conda_executable,
)


# ── Project root discovery ────────────────────────────────────


def project_root() -> Path:
    """Resolve the workspace or project root path by searching for ``pyproject.toml``."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # NOTE: Fallback to an assumed ancestor depth if no pyproject.toml is found.
    return current.parents[5]


# ── Default paths and constants ───────────────────────────────

# Primary model catalog root for runtime-profile synthesis.
DEFAULT_CATALOG_MANIFEST = DATA_ROOT / "models" / "catalog"
DEFAULT_CATALOG_MANIFESTS = (DEFAULT_CATALOG_MANIFEST,)
# Canonical per-model runtime profile YAML files (schema v2 and legacy single-file profiles).
DEFAULT_RUNTIME_PROFILES_ROOT = DATA_ROOT / "models" / "runtime" / "profiles"
DEFAULT_TARGET_PROFILE_MANIFESTS = (DEFAULT_RUNTIME_PROFILES_ROOT,)
# Local cache root for cloned acquisition targets.
DEFAULT_ACQUISITION_ROOT = REPO_ROOT / "cache" / "generative_taxonomy"
# Shared Hugging Face cache root resolved from ``WORLDFOUNDRY_HFD_ROOT``.
DEFAULT_SHARED_HFD_ROOT = resolve_hfd_root()
# Stable-Diffusion 1.5 checkpoint path, overridable via ``WORLDFOUNDRY_SD15_ROOT``.
DEFAULT_SD15_ROOT = Path(
    os.environ.get(
        "WORLDFOUNDRY_SD15_ROOT",
        str(DEFAULT_SHARED_HFD_ROOT / "stable-diffusion-v1-5--stable-diffusion-v1-5"),
    )
)
# Default conditioning data directory for runtime commands.
DEFAULT_COND_DIR = str(DATA_ROOT / "test_cases")


# ── String helpers and coercion utilities ────────────────────


def _safe_name(value: str) -> str:
    """Format and return a filesystem-safe name string from a value."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "item"


def repo_slug(url: str) -> str:
    """Compute a clean, filesystem-safe folder name for a repository URL.

    Handles ``git@``, ``https://``, and bare path URLs, producing a
    ``<org>--<repo>`` style slug (e.g. ``facebook--dinov2``).
    """
    text = str(url).rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("git@") and ":" in text:
        path = text.split(":", 1)[1]
    else:
        parsed = urlparse(text)
        path = parsed.path if parsed.scheme else text
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) >= 2:
        return f"{_safe_name(parts[-2])}--{_safe_name(parts[-1])}"
    return _safe_name(parts[-1] if parts else text)


def _runtime_source_repo(source: Mapping[str, Any]) -> dict[str, Any]:
    """Retrieve runtime source repository dictionary, dropping the ``local_dir`` key."""
    item = dict(source)
    # NOTE: local_dir is environment-specific and must not leak into serialized profiles.
    item.pop("local_dir", None)
    return item


def _iter_source_mappings(value: Any, *, string_key: str) -> tuple[Mapping[str, Any], ...]:
    """Coerce various string or dictionary mapping structures into a tuple of mappings."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, str):
        return ({string_key: value},)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                items.append(item)
            elif isinstance(item, str):
                items.append({string_key: item})
        return tuple(items)
    return ()


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerce any scalar value or sequence into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _tuple_of_mapping(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Coerce any sequence or single mapping to a tuple of dictionaries."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(key in value for key in ("id", "repo_id", "url", "local_dir", "kind", "role")):
            return (dict(value),)
        items: list[Mapping[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                mapped = dict(item)
                mapped.setdefault("id", str(key))
                items.append(mapped)
        return tuple(items)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    return ()


def _profile_notes(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract and compile notes or evidence notes from profile data mapping."""
    evidence = data.get("evidence") if isinstance(data.get("evidence"), Mapping) else {}
    return _tuple_of_str(data.get("notes") or evidence.get("notes"))


def _hf_dir_for_repo(repo_id: str, *, hf_models_root: Path, acquisition_root: Path) -> str:
    """Generate the local Hugging Face model directory path for a repo ID."""
    del acquisition_root
    dirname = repo_id.replace("/", "--")
    return str(hf_models_root / dirname)


def _is_hf_model_source(source: Mapping[str, Any]) -> bool:
    """Determine if a source repository is an official Hugging Face model source."""
    source_type = str(source.get("type") or source.get("repo_type") or "model").strip().lower()
    return source_type in {"model", "checkpoint", "weights"}


def _primary_source_context(source_repos: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Extract context properties from the primary source repository entry."""
    source = dict(source_repos[0]) if source_repos else {}
    source_dir = str(source.get("local_dir") or "")
    subdir = str(source.get("subdir") or "")
    source_workdir = str(Path(source_dir) / subdir) if source_dir and subdir else source_dir
    return {
        "source_repo_url": str(source.get("url") or ""),
        "source_repo_revision": str(source.get("revision") or source.get("head_sha") or source.get("confirmed_ref") or ""),
        "source_repo_dir": source_dir,
        "source_repo_subdir": subdir,
        "source_repo_workdir": source_workdir,
    }


def _task_family(groups: Sequence[str]) -> str:
    """Deduce task family based on a sequence of group tags."""
    group_set = {str(item).lower() for item in groups}
    if group_set & {"action_chunking", "act"}:
        return "action_chunking_policy"
    if group_set & {"action_diffusion", "diffusion_policy", "visuomotor"}:
        return "visuomotor_policy"
    if "vla" in group_set or "policy" in group_set or "humanoid" in group_set:
        return "vla_policy"
    if "va" in group_set or "vam" in group_set or "latent_action" in group_set:
        return "visual_action_model"
    if "wam" in group_set:
        return "world_action_model"
    if "3d" in group_set:
        return "three_dimension"
    if "4d" in group_set:
        return "four_dimension"
    if "world" in group_set:
        return "world_model"
    return "video_generation"


def _artifact_kind(task_family: str) -> str:
    """Determine artifact kind based on the resolved task family."""
    if task_family == "vla_policy":
        return "action_trace"
    if task_family == "visual_action_model":
        return "action_tokens"
    if task_family == "world_action_model":
        return "generated_world"
    if task_family in {"action_chunking_policy", "visuomotor_policy"}:
        return "action_trace"
    if task_family == "three_dimension":
        return "generated_3d_asset"
    if task_family == "four_dimension":
        return "generated_4d_scene"
    if task_family == "world_model":
        return "generated_world"
    return "generated_video"


def _artifact_filename(artifact_kind: str) -> str:
    """Determine artifact filename extension/convention based on the artifact kind."""
    if artifact_kind == "action_trace":
        return "action_trace.json"
    if artifact_kind == "action_tokens":
        return "action_tokens.json"
    if artifact_kind == "world_state":
        return "world_state.json"
    if artifact_kind == "session_trace":
        return "session_trace.json"
    if artifact_kind == "generated_3d_asset":
        return "artifact.glb"
    if artifact_kind == "generated_4d_scene":
        return "scene.json"
    if artifact_kind == "generated_world":
        return "world.mp4"
    return "video.mp4"


# ── Core data model ──────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeProfile:
    """Model runtime profile detailing tasks, repos, checkpoints, and execution options.

    Attributes:
        model_id: Unique identifier for the model (e.g. ``"cogvideox-5b"``).
        display_name: Human-readable model name used in reports and UI.
        task_family: Broad task category such as ``"video_generation"``
            or ``"vla_policy"``.
        groups: Tag set for filtering and sub-categorization.
        source_repos: Tuple of source-repository mappings (``url``, ``revision``, …).
        checkpoints: Tuple of checkpoint / weight-file mappings.
        input_schema: Expected input specification (required/optional keys).
        artifact_kind: Output artifact type, e.g. ``"generated_video"``.
        artifact_filename: Default output filename, e.g. ``"video.mp4"``.
        command_template: Shell command template parts used for execution.
        conda_env: Resolved conda environment specification dict.
        backend_stage: Integration maturity label (``"profile_only"`` etc.).
        runtime_status: Runtime readiness status.
        integration_status: Integration tracking status.
        execution: Additional execution-time configuration mapping.
        output: Output-related configuration mapping.
        notes: Free-text notes and blocker descriptions.
        schema_version: Schema version marker; only ``2`` is currently supported.
    """

    model_id: str
    display_name: str
    task_family: str
    groups: tuple[str, ...] = ()
    source_repos: tuple[Mapping[str, Any], ...] = ()
    checkpoints: tuple[Mapping[str, Any], ...] = ()
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    artifact_kind: str = "generated_video"
    artifact_filename: str = "video.mp4"
    command_template: tuple[str, ...] = ()
    conda_env: Mapping[str, Any] = field(default_factory=dict)
    backend_stage: str = "profile_only"
    runtime_status: str = "profiled"
    integration_status: str = "planned"
    execution: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    schema_version: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RuntimeProfile":
        """Build a :class:`RuntimeProfile` from a raw YAML mapping.

        Supports multiple legacy key names (e.g. ``"id"`` → ``"model_id"``) and
        infers ``task_family``, ``artifact_kind``, and ``artifact_filename``
        when not explicitly provided.

        Args:
            data: Raw mapping loaded from a YAML profile manifest.

        Returns:
            A validated :class:`RuntimeProfile` instance.
        """
        artifact = data.get("artifact") if isinstance(data.get("artifact"), Mapping) else {}
        inputs = data.get("inputs") if isinstance(data.get("inputs"), Mapping) else {}
        execution = data.get("execution") if isinstance(data.get("execution"), Mapping) else {}
        output = data.get("output") if isinstance(data.get("output"), Mapping) else {}
        integration = data.get("integration") if isinstance(data.get("integration"), Mapping) else {}
        profile_id = str(data.get("profile_id") or data.get("id") or data.get("model_id") or "")
        model_id = str(data.get("model_id") or data.get("id") or profile_id)
        input_schema = data.get("input_schema") if isinstance(data.get("input_schema"), Mapping) else {}
        # NOTE: Derive input_schema from legacy ``inputs`` mapping when not explicit.
        if not input_schema and inputs:
            input_schema = {
                "required": list(_tuple_of_str(inputs.get("required"))),
                "optional": list(_tuple_of_str(inputs.get("optional"))),
            }
        task_family = str(data.get("task_family") or execution.get("task_family") or "video_generation")
        artifact_kind = str(artifact.get("kind") or data.get("artifact_kind") or _artifact_kind(task_family))
        profile = cls(
            model_id=model_id,
            display_name=str(data.get("display_name") or data.get("name") or model_id),
            task_family=task_family,
            schema_version=_schema_version(data.get("schema_version")),
            groups=_tuple_of_str(data.get("groups")),
            source_repos=tuple(_runtime_source_repo(source) for source in _tuple_of_mapping(data.get("source_repos"))),
            checkpoints=_tuple_of_mapping(data.get("checkpoints")),
            input_schema=dict(input_schema),
            artifact_kind=artifact_kind,
            artifact_filename=str(
                artifact.get("filename") or data.get("artifact_filename") or _artifact_filename(artifact_kind)
            ),
            command_template=_tuple_of_str(execution.get("command_template") or data.get("command_template")),
            conda_env=dict(data.get("conda_env")) if isinstance(data.get("conda_env"), Mapping) else {},
            backend_stage=str(execution.get("backend_stage") or data.get("backend_stage") or "profile_only"),
            runtime_status=str(execution.get("runtime_status") or data.get("runtime_status") or "profiled"),
            integration_status=str(
                integration.get("status") or execution.get("integration_status") or data.get("integration_status") or "planned"
            ),
            execution=dict(execution),
            output=dict(output),
            notes=_profile_notes(data),
        )
        # Preserve the original profile_id inside execution if it differs from model_id.
        if profile_id and profile_id != profile.model_id:
            profile = replace(
                profile,
                execution={
                    **dict(profile.execution),
                    "profile_id": profile_id,
                },
            )
        profile.validate()
        return profile

    def validate(self) -> None:
        """Raise ``ValueError`` if the profile is missing required fields or has an unsupported schema."""
        if self.schema_version is not None and self.schema_version != 2:
            raise ValueError(f"runtime profile {self.model_id!r} uses unsupported schema_version {self.schema_version!r}.")
        if not self.model_id:
            raise ValueError("runtime profile requires model_id.")
        if not self.display_name:
            raise ValueError(f"runtime profile {self.model_id!r} requires display_name.")
        if not self.task_family:
            raise ValueError(f"runtime profile {self.model_id!r} requires task_family.")
        if not self.artifact_kind:
            raise ValueError(f"runtime profile {self.model_id!r} requires artifact_kind.")
        if not self.artifact_filename:
            raise ValueError(f"runtime profile {self.model_id!r} requires artifact_filename.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a plain dictionary suitable for JSON/YAML serialization."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task_family": self.task_family,
            "groups": list(self.groups),
            "source_repos": [dict(item) for item in self.source_repos],
            "checkpoints": [dict(item) for item in self.checkpoints],
            "input_schema": dict(self.input_schema),
            "artifact_kind": self.artifact_kind,
            "artifact_filename": self.artifact_filename,
            "command_template": list(self.command_template),
            "conda_env": dict(self.conda_env),
            "backend_stage": self.backend_stage,
            "runtime_status": self.runtime_status,
            "integration_status": self.integration_status,
            "execution": dict(self.execution),
            "output": dict(self.output),
            "notes": list(self.notes),
        }


# ── Schema version helper ─────────────────────────────────────


def _schema_version(value: Any) -> int | None:
    """Coerce any schema version value to int or return ``None``."""
    if value in (None, ""):
        return None
    return int(value)


# ── Profile override loaders ──────────────────────────────────


def _load_profile_override_mapping(path: Path) -> dict[str, Any]:
    """Load and parse profile override mapping dictionary from the specified path."""
    if not path.exists():
        return {}
    payload = load_manifest_collection(path, item_key="profiles")
    return payload if isinstance(payload, dict) else {}


def _target_profile_paths(root: str | Path | Sequence[str | Path] | None = None) -> tuple[Path, ...]:
    """Retrieve all profile YAML paths recursively from one or multiple root directories."""
    if root is None:
        roots = DEFAULT_TARGET_PROFILE_MANIFESTS
    elif isinstance(root, (str, Path)):
        roots = (Path(root),)
    else:
        roots = tuple(Path(item) for item in root)

    paths: list[Path] = []
    for path in roots:
        paths.extend(_target_profile_paths_for_root(path))
    return tuple(dict.fromkeys(paths))


def _target_profile_paths_for_root(root: Path) -> tuple[Path, ...]:
    """Retrieve sorted list of all YAML target profile file paths under a single root path."""
    path = Path(root)
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    return tuple(sorted(item for item in path.rglob("*.y*ml") if item.is_file()))


def _iter_target_profile_mappings(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load and iterate over raw target profiles defined in a YAML manifest file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"runtime profile file must contain a mapping: {path}")
    entries = payload.get("profiles") or payload.get("runtime_profiles")
    if entries is None:
        return (payload,) if payload.get("model_id") or payload.get("profile_id") or payload.get("id") else ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise TypeError(f"runtime profile collection must be a list: {path}")
    return tuple(item for item in entries if isinstance(item, Mapping))


def load_runtime_profile_manifest(path: str | Path) -> RuntimeProfile:
    """Load a single RuntimeProfile from a YAML manifest path."""
    entries = _iter_target_profile_mappings(Path(path))
    if len(entries) != 1:
        raise ValueError(f"expected one runtime profile in {path}, found {len(entries)}")
    return RuntimeProfile.from_mapping(entries[0])


def load_runtime_profile_manifests(root: str | Path | None = None) -> tuple[RuntimeProfile, ...]:
    """Load and parse all target RuntimeProfile manifests under the given root."""
    profiles: list[RuntimeProfile] = []
    for path in _target_profile_paths(root):
        profiles.extend(RuntimeProfile.from_mapping(data) for data in _iter_target_profile_mappings(path))
    return tuple(profiles)


# ── Catalog target helpers ────────────────────────────────────


def _manifest_paths(value: str | Path | Sequence[str | Path] | None) -> tuple[Path, ...]:
    """Resolve and coerce multiple catalog or legacy acquisition manifest paths."""
    if value is None:
        return DEFAULT_CATALOG_MANIFESTS
    if isinstance(value, (str, Path)):
        return (Path(value),)
    return tuple(Path(item) for item in value)


def _is_catalog_manifest_root(path: Path) -> bool:
    """Return True when ``path`` points at the model catalog tree."""
    normalized = path.resolve()
    return normalized.name == "catalog" or normalized == DEFAULT_CATALOG_MANIFEST.resolve()


def _catalog_payload_to_runtime_target(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert a catalog YAML payload into the legacy runtime-target shape."""
    model_id = str(entry.get("id") or entry.get("model_id") or "")
    official_sources = entry.get("official_sources") if isinstance(entry.get("official_sources"), Mapping) else {}
    sources = entry.get("sources") if isinstance(entry.get("sources"), Mapping) else {}
    if not official_sources and sources:
        official_sources = dict(sources)

    source = entry.get("source") if isinstance(entry.get("source"), Mapping) else {}
    github = official_sources.get("github")
    if isinstance(github, str):
        official_sources = {**dict(official_sources), "github": {"url": github}}
    elif not github and isinstance(sources.get("github"), Mapping):
        official_sources = {**dict(official_sources), "github": dict(sources["github"])}
    elif not github and source.get("official_repo_url"):
        official_sources = {
            **dict(official_sources),
            "github": {"url": str(source["official_repo_url"])},
        }

    huggingface_sources = None
    for key in ("huggingface_models", "huggingface", "hf_models", "models"):
        value = official_sources.get(key)
        if value:
            huggingface_sources = value
            break
    if huggingface_sources is None:
        for key in ("huggingface", "huggingface_models", "hf_models", "models"):
            value = sources.get(key)
            if value:
                huggingface_sources = value
                break
    if huggingface_sources is not None and not any(
        key in official_sources for key in ("huggingface_models", "huggingface", "hf_models", "models")
    ):
        if isinstance(huggingface_sources, Mapping):
            huggingface_sources = [dict(huggingface_sources)]
        official_sources = {
            **dict(official_sources),
            "huggingface_models": [
                dict(ref) if isinstance(ref, Mapping) else {"repo_id": str(ref)}
                for ref in huggingface_sources
            ],
        }

    checkpoint_refs = entry.get("checkpoint_refs") or ()
    checkpoints = entry.get("checkpoints")
    if not checkpoint_refs and isinstance(checkpoints, Mapping):
        primary = checkpoints.get("primary")
        if isinstance(primary, Mapping):
            checkpoint_refs = (primary,)
        repos = checkpoints.get("repos")
        if isinstance(repos, list):
            checkpoint_refs = tuple(repos)
    if checkpoint_refs and not any(
        key in official_sources for key in ("huggingface_models", "huggingface", "hf_models", "models")
    ):
        official_sources = {
            **dict(official_sources),
            "huggingface_models": [
                dict(ref) if isinstance(ref, Mapping) else {"repo_id": str(ref)}
                for ref in checkpoint_refs
            ],
        }

    capabilities = entry.get("capabilities") if isinstance(entry.get("capabilities"), Mapping) else {}
    groups = [str(item) for item in entry.get("tasks") or entry.get("groups") or ()]
    if not groups and capabilities.get("modalities"):
        groups = [str(item) for item in capabilities.get("modalities") or ()]
    catalog_category = str(entry.get("catalog_category") or "").strip()
    if catalog_category and catalog_category not in groups:
        groups.append(catalog_category)
    integration = entry.get("integration") if isinstance(entry.get("integration"), Mapping) else {}
    return {
        "id": model_id,
        "taxonomy_name": str(entry.get("name") or entry.get("taxonomy_name") or model_id),
        "groups": groups,
        "official_sources": official_sources,
        "integration": integration,
        "kind": "model",
    }


def _iter_catalog_runtime_targets(root: Path) -> tuple[Mapping[str, Any], ...]:
    """Iterate model-catalog entries that can seed runtime profiles."""
    targets: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths(root):
        if manifest_path.name in {"_manifest.yaml", "_DEPRECATED.yaml"}:
            continue
        payload = load_manifest(manifest_path)
        for item in iter_model_zoo_payloads(payload):
            if not isinstance(item, Mapping):
                continue
            target = _catalog_payload_to_runtime_target(item)
            target_id = str(target.get("id") or "")
            if not target_id or target_id in seen:
                continue
            targets.append(target)
            seen.add(target_id)
    return tuple(targets)


def _is_runtime_target(target: Mapping[str, Any]) -> bool:
    """Filter out non-model / non-runtime acquisition targets like benchmarks or frameworks."""
    target_id = str(target.get("id") or "")
    if not target_id:
        return False
    kind = str(target.get("kind") or "model").lower()
    if kind in {"benchmark", "benchmark_runtime", "dataset", "framework", "simulator"}:
        return False
    return True


def _iter_runtime_targets(paths: Sequence[Path]) -> tuple[Mapping[str, Any], ...]:
    """Iterate, parse, and de-duplicate runtime targets from catalog manifests."""
    targets: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not _is_catalog_manifest_root(path):
            warnings.warn(
                f"runtime target manifest {path} is not under data/models/catalog; skipping.",
                DeprecationWarning,
                stacklevel=3,
            )
            continue
        for target in _iter_catalog_runtime_targets(path):
            target_id = str(target["id"])
            if target_id in seen:
                continue
            targets.append(target)
            seen.add(target_id)
    return tuple(targets)


def _profile_override_mapping(profile_path: str | Path | None) -> dict[str, Mapping[str, Any]]:
    """Load optional profile override manifests when explicitly requested."""
    if profile_path is None:
        return {}
    resolved = Path(profile_path)
    override_payload = _load_profile_override_mapping(resolved)
    return {
        str(item.get("id")): item
        for item in override_payload.get("profiles", [])
        if isinstance(item, Mapping) and item.get("id")
    }


def _uses_catalog_manifest(paths: Sequence[Path]) -> bool:
    return any(_is_catalog_manifest_root(path) for path in paths)


def _profile_from_target(
    target: Mapping[str, Any],
    *,
    override: Mapping[str, Any],
    acquisition_root: Path,
    hf_models_root: Path,
    conda_env: Mapping[str, Any] | None = None,
) -> RuntimeProfile:
    """Build a :class:`RuntimeProfile` instance from an acquisition target, merging manual overrides.

    Args:
        target: Raw acquisition target mapping from the catalog manifest.
        override: Per-model override mapping that supplements or replaces target fields.
        acquisition_root: Local directory where cloned repos are cached.
        hf_models_root: Shared Hugging Face model cache root.
        conda_env: Optional pre-resolved conda environment spec to attach.

    Returns:
        A :class:`RuntimeProfile` instance (not yet validated).
    """
    model_id = str(target["id"])
    groups = tuple(str(item) for item in target.get("groups") or ())
    task_family = str(override.get("task_family") or _task_family(groups))
    artifact_kind = str(override.get("artifact_kind") or _artifact_kind(task_family))
    official_sources = target.get("official_sources") if isinstance(target.get("official_sources"), Mapping) else {}

    source_repos: list[dict[str, Any]] = []
    for source in _iter_source_mappings(official_sources.get("github"), string_key="url"):
        source_repos.append(_runtime_source_repo(source))

    checkpoints: list[dict[str, Any]] = []
    hf_sources: list[Mapping[str, Any]] = []
    for key in ("huggingface_models", "huggingface", "hf_models", "models"):
        hf_sources.extend(_iter_source_mappings(official_sources.get(key), string_key="repo_id"))
    for source in hf_sources:
        if not source.get("repo_id") or not _is_hf_model_source(source):
            continue
        item = dict(source)
        item.setdefault(
            "local_dir",
            _hf_dir_for_repo(str(item["repo_id"]), hf_models_root=hf_models_root, acquisition_root=acquisition_root),
        )
        checkpoints.append(item)
    for source in override.get("checkpoints") or ():
        if not isinstance(source, Mapping):
            continue
        checkpoints.append(dict(source))

    integration = target.get("integration") if isinstance(target.get("integration"), Mapping) else {}
    notes = list(override.get("notes") or ())
    notes.extend(str(item) for item in integration.get("blocked_reasons") or ())
    command_template_items = ()

    return RuntimeProfile(
        model_id=model_id,
        display_name=str(override.get("display_name") or target.get("taxonomy_name") or model_id),
        task_family=task_family,
        groups=groups,
        source_repos=tuple(source_repos),
        checkpoints=tuple(checkpoints),
        input_schema=dict(override.get("input_schema") or {}),
        artifact_kind=artifact_kind,
        artifact_filename=str(override.get("artifact_filename") or _artifact_filename(artifact_kind)),
        command_template=tuple(str(item) for item in command_template_items),
        conda_env=dict(conda_env or {}),
        backend_stage=str(override.get("backend_stage") or "profile_only"),
        runtime_status=str(override.get("runtime_status") or "profile_only_needs_in_tree_runtime_port"),
        integration_status=str(override.get("integration_status") or integration.get("status") or "planned"),
        notes=tuple(notes),
    )


# ── Public profile loaders ────────────────────────────────────


def _path_or_sequence_cache_key(value: str | Path | Sequence[str | Path] | None) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(Path(value))
    return tuple(str(Path(item)) for item in value)


def _runtime_profile_env_cache_key() -> tuple[tuple[str, str], ...]:
    keys = (
        "WORLDFOUNDRY_CONDA_ENV_ROOT",
        "WORLDFOUNDRY_CONDA_ENVS_ROOT",
        "WORLDFOUNDRY_USE_UNIFIED_ENV",
        "WORLDFOUNDRY_UNIFIED_ENV_PREFIX",
        "WORLDFOUNDRY_CUDA_PROFILE",
        "WORLDFOUNDRY_CUDA_TIER",
        "WORLDFOUNDRY_DETECTED_DRIVER_CUDA",
        "WORLDFOUNDRY_HFD_ROOT",
        "WORLDFOUNDRY_CKPT_DIR",
        "WORLDFOUNDRY_HOME",
    )
    values = [(key, os.environ.get(key, "")) for key in keys]
    values.extend((f"unified_env_exists:{tier}", str(unified_env_exists(tier))) for tier in SUPPORTED_CUDA_TIERS)
    return tuple(values)


def _load_runtime_profiles_uncached(
    manifest_path: str | Path | Sequence[str | Path] | None = None,
    profile_path: str | Path | None = None,
    *,
    target_profile_path: str | Path | Sequence[str | Path] | None = None,
    conda_env_path: str | Path | None = None,
    acquisition_root: str | Path | None = None,
    hf_models_root: str | Path | None = None,
    check_conda_env_exists: bool = True,
) -> dict[str, RuntimeProfile]:
    """Load, merge, and aggregate runtime profiles from the model catalog and profile tree.

    Builds base profiles from ``data/models/catalog``, optionally applies
    explicit override manifests when ``profile_path`` is provided, then overlays
    ``data/models/runtime/profiles`` entries.
    """
    manifest_paths = _manifest_paths(manifest_path)
    overrides = _profile_override_mapping(profile_path)
    root = Path(acquisition_root or DEFAULT_ACQUISITION_ROOT)
    hfd_root = Path(hf_models_root or DEFAULT_SHARED_HFD_ROOT)
    conda_env_specs = load_runtime_conda_env_specs_with_overrides(
        conda_env_path,
        env_root=os.environ.get("WORLDFOUNDRY_CONDA_ENV_ROOT"),
    )
    profiles: dict[str, RuntimeProfile] = {}
    runtime_targets = list(_iter_runtime_targets(manifest_paths))
    for target in runtime_targets:
        target_id = str(target["id"])
        env_spec = conda_env_specs.get(target_id)
        profile = _profile_from_target(
            target,
            override=overrides.get(target_id, {}),
            acquisition_root=root,
            hf_models_root=hfd_root,
            conda_env=env_spec.to_dict(check_exists=check_conda_env_exists) if env_spec is not None else {},
        )
        profiles[profile.model_id] = profile
    resolved_target_profile_path = (
        DEFAULT_RUNTIME_PROFILES_ROOT if target_profile_path is None else target_profile_path
    )
    for target_profile in load_runtime_profile_manifests(resolved_target_profile_path):
        existing_profile = profiles.get(target_profile.model_id)
        if existing_profile is not None:
            fallback_fields: dict[str, Any] = {}
            if (
                target_profile.display_name == target_profile.model_id
                and existing_profile.display_name
                and existing_profile.display_name != existing_profile.model_id
            ):
                fallback_fields["display_name"] = existing_profile.display_name
            if not target_profile.groups and existing_profile.groups:
                fallback_fields["groups"] = existing_profile.groups
            if not target_profile.source_repos and existing_profile.source_repos:
                fallback_fields["source_repos"] = existing_profile.source_repos
            if not target_profile.checkpoints and existing_profile.checkpoints:
                fallback_fields["checkpoints"] = existing_profile.checkpoints
            if fallback_fields:
                target_profile = replace(target_profile, **fallback_fields)
        if not target_profile.conda_env:
            env_key = str(target_profile.execution.get("environment") or target_profile.model_id)
            env_spec = conda_env_specs.get(env_key) or conda_env_specs.get(target_profile.model_id)
            if env_spec is not None:
                target_profile = replace(
                    target_profile,
                    conda_env=env_spec.to_dict(check_exists=check_conda_env_exists),
                )
        profiles[target_profile.model_id] = target_profile
    return profiles


@lru_cache(maxsize=16)
def _load_runtime_profiles_no_exists_cached(
    manifest_path_key: str | tuple[str, ...] | None,
    profile_path_key: str | None,
    target_profile_path_key: str | tuple[str, ...] | None,
    conda_env_path_key: str | None,
    acquisition_root_key: str | None,
    hf_models_root_key: str | None,
    env_cache_key: tuple[tuple[str, str], ...],
) -> dict[str, RuntimeProfile]:
    del env_cache_key
    return _load_runtime_profiles_uncached(
        manifest_path=manifest_path_key,
        profile_path=profile_path_key,
        target_profile_path=target_profile_path_key,
        conda_env_path=conda_env_path_key,
        acquisition_root=acquisition_root_key,
        hf_models_root=hf_models_root_key,
        check_conda_env_exists=False,
    )


def load_runtime_profiles(
    manifest_path: str | Path | Sequence[str | Path] | None = None,
    profile_path: str | Path | None = None,
    *,
    target_profile_path: str | Path | Sequence[str | Path] | None = None,
    conda_env_path: str | Path | None = None,
    acquisition_root: str | Path | None = None,
    hf_models_root: str | Path | None = None,
    check_conda_env_exists: bool = True,
) -> dict[str, RuntimeProfile]:
    """Load, merge, and aggregate runtime profiles from the model catalog and profile tree.

    Builds base profiles from ``data/models/catalog``, optionally applies
    explicit override manifests when ``profile_path`` is provided, then overlays
    ``data/models/runtime/profiles`` entries.
    """
    if check_conda_env_exists:
        return _load_runtime_profiles_uncached(
            manifest_path=manifest_path,
            profile_path=profile_path,
            target_profile_path=target_profile_path,
            conda_env_path=conda_env_path,
            acquisition_root=acquisition_root,
            hf_models_root=hf_models_root,
            check_conda_env_exists=True,
        )
    return dict(
        _load_runtime_profiles_no_exists_cached(
            _path_or_sequence_cache_key(manifest_path),
            None if profile_path is None else str(Path(profile_path)),
            _path_or_sequence_cache_key(target_profile_path),
            None if conda_env_path is None else str(Path(conda_env_path)),
            None if acquisition_root is None else str(Path(acquisition_root)),
            None if hf_models_root is None else str(Path(hf_models_root)),
            _runtime_profile_env_cache_key(),
        )
    )


def clear_runtime_profiles_cache() -> None:
    """Clear cached runtime profile manifests for no-exists loaders."""
    _load_runtime_profiles_no_exists_cached.cache_clear()


load_runtime_profiles.cache_clear = clear_runtime_profiles_cache  # type: ignore[attr-defined]


def load_runtime_profile(model_id: str, **kwargs: Any) -> RuntimeProfile:
    """Load a single :class:`RuntimeProfile` by model ID.

    Args:
        model_id: The unique identifier of the desired model profile.
        **kwargs: Forwarded to :func:`load_runtime_profiles`.

    Raises:
        KeyError: If ``model_id`` is not found among loaded profiles.
    """
    profiles = load_runtime_profiles(**kwargs)
    if model_id not in profiles:
        raise KeyError(f"Unknown runtime profile: {model_id}")
    return profiles[model_id]


def _coerce_path_input(value: Any, destination: Path, stem: str) -> str | None:
    """Coerce any PIL image, file path, or remote URL into a relative/absolute path string."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith(("http://", "https://", "s3://", "gs://", "hf://")):
            return text
        path = Path(text).expanduser()
        return str(path) if path.exists() else text
    save = getattr(value, "save", None)
    if callable(save):
        path = destination / f"{stem}.png"
        save(path)
        return str(path)
    return str(value)


def _json_safe(value: Any) -> Any:
    """Recursively convert custom/path objects so they are fully JSON-serializable."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


# ── Synthesis bridge ──────────────────────────────────────────


class RuntimeProfileSynthesis(BaseSynthesis):
    """Runtime-profile planning/execution bridge for model-specific shims.

    A profile is runnable only when it has an in-tree backend implementation or
    an integrated command template. Metadata-only and profile-only records are
    provenance/planning surfaces and must not be treated as full integrations.
    """

    MODEL_ID: str | None = None

    def __init__(
        self,
        profile: RuntimeProfile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the synthesis shim with a runtime profile, device, command template, and environment."""
        self.profile = profile
        self.model_id = profile.model_id
        self.model_name = profile.model_id
        self.generation_type = "t2v" if profile.task_family == "video_generation" else "runtime_profile"
        self.device = device
        self.command_template = tuple(command_template or profile.command_template)
        self.env = dict(env or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path=None,
        args=None,
        device=None,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        acquisition_root: str | Path | None = None,
        hf_models_root: str | Path | None = None,
        command_template: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "RuntimeProfileSynthesis":
        """Load the runtime profile and instantiate a RuntimeProfileSynthesis instance."""
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID or "")
        if not resolved_model_id:
            raise ValueError("RuntimeProfileSynthesis requires model_id/profile_id.")
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        resolved_template = command_template or options.get("command_template")
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=resolved_template,
            env=options.get("env"),
        )

    def _context(
        self,
        *,
        prompt: str,
        images: Any,
        video: Any,
        interactions: Sequence[str],
        output_path: str | Path | None,
        fps: int | None,
        run_dir: Path,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Assemble formatting variables and inputs for the runtime command template.

        Writes prompt, actions, and extra_inputs to files under ``run_dir``
        and resolves conda/python paths for command substitution.

        Args:
            prompt: Text prompt for the generation request.
            images: Image input(s) — path, URL, or PIL image.
            video: Video input(s) — path, URL, or file-like object.
            interactions: Sequence of action/interaction strings.
            output_path: Desired output file path; defaults to the profile's
                ``artifact_filename`` inside ``run_dir``.
            fps: Frames-per-second override; included as ``""`` when unset.
            run_dir: Temporary working directory for the current run.
            extra: Arbitrary extra parameters forwarded into the template context.

        Returns:
            A ``dict[str, str]`` mapping template variable names to their
            resolved string values.
        """
        # ── Write input files ────────────────────────────
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        actions_path = run_dir / "actions.json"
        actions_path.write_text(
            json.dumps(_json_safe(list(interactions)), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        extra_inputs_path = run_dir / "extra_inputs.json"
        extra_inputs_path.write_text(
            json.dumps(_json_safe(dict(extra)), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # ── Resolve media inputs ──────────────────────────
        image_path = _coerce_path_input(images, run_dir, "input_image")
        video_path = _coerce_path_input(video, run_dir, "input_video")
        resolved_output = Path(output_path) if output_path is not None else run_dir / self.profile.artifact_filename
        if not resolved_output.is_absolute():
            resolved_output = (Path.cwd() / resolved_output).resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        # ── Resolve conda / python paths ──────────────────
        primary_ckpt = dict(self.profile.checkpoints[0]) if self.profile.checkpoints else {}
        source_context = _primary_source_context(self.profile.source_repos)
        conda_env = resolve_conda_env_context(self.model_id)
        conda_python = str(extra.get("python_executable") or "")
        if not conda_python:
            conda_python = str(conda_env.get("python_executable") or "")
            if not Path(conda_python).is_file():
                # NOTE: Fall back to the current interpreter if the env doesn't exist.
                conda_python = sys.executable
        conda_torchrun = str(extra.get("torchrun_executable") or "")
        if not conda_torchrun:
            conda_torchrun = resolve_conda_executable(self.model_id, "torchrun") or resolve_existing_tool("torchrun")
        conda_env_prefix = str(conda_env.get("env_prefix") or "")
        # ── Build the full template context ──────────────
        context = {
            "python": conda_python,
            "torchrun": conda_torchrun,
            "model_id": self.model_id,
            "display_name": self.profile.display_name,
            "worldfoundry_root": str(REPO_ROOT),
            "repo_root": str(REPO_ROOT),
            "checkpoint_dir": primary_ckpt.get("local_dir", ""),
            "prompt": prompt,
            "prompt_path": str(prompt_path),
            "actions_path": str(actions_path),
            "extra_inputs_path": str(extra_inputs_path),
            "image_path": image_path or "",
            "video_path": video_path or "",
            "output_path": str(resolved_output),
            "output_dir": str(resolved_output.parent),
            "run_dir": str(run_dir),
            "device": self.device,
            "fps": "" if fps is None else str(fps),
            "parallel": str(extra.get("parallel", 4)),
            "tensor_parallel_degree": str(extra.get("tensor_parallel_degree", 2)),
            "tp_degree": str(extra.get("tp_degree", extra.get("tensor_parallel_degree", 2))),
            "ulysses_degree": str(extra.get("ulysses_degree", 2)),
            "vae_url": str(extra.get("vae_url", "127.0.0.1")),
            "caption_url": str(extra.get("caption_url", "127.0.0.1")),
            "infer_steps": str(extra.get("infer_steps", 50)),
            "cfg_scale": str(extra.get("cfg_scale", 9.0)),
            "time_shift": str(extra.get("time_shift", 13.0)),
            "class_id": str(extra.get("class_id", 207)),
            "batch_size": str(extra.get("batch_size", 1)),
            "seed": str(extra.get("seed", 1234)),
            "height": str(extra.get("height", 256)),
            "width": str(extra.get("width", 256)),
            "condtype": str(extra.get("condtype", "both")),
            "cond_dir": str(extra.get("cond_dir", DEFAULT_COND_DIR)),
            "nproc_per_node": str(extra.get("nproc_per_node", 1)),
            "master_port": str(extra.get("master_port", 25000)),
            "config": str(extra.get("config", "")),
            "ckpt_path": str(extra.get("ckpt_path", primary_ckpt.get("local_dir", ""))),
            "unnorm_key": str(extra.get("unnorm_key", "bridge_orig")),
            "attn_implementation": str(extra.get("attn_implementation", "eager")),
            "torch_dtype": str(extra.get("torch_dtype", "auto")),
            "sd15_path": str(extra.get("sd15_path", DEFAULT_SD15_ROOT)),
            "motion_module_ckpt": str(extra.get("motion_module_ckpt", "")),
            "pose_adaptor_ckpt": str(extra.get("pose_adaptor_ckpt", primary_ckpt.get("local_dir", ""))),
            "trajectory_file": str(extra.get("trajectory_file", "")),
            "conda_env_name": str(conda_env.get("env_name") or ""),
            "conda_env_prefix": conda_env_prefix,
            "conda_env_exists": str(bool(conda_env.get("exists"))).lower(),
            "conda_env_driver_status": str(conda_env.get("driver_status") or ""),
            "conda_env_cuda_profile": str(conda_env.get("cuda_profile") or ""),
            "driver_cuda": str(conda_env.get("driver_cuda") or ""),
            "source_repos_json": json.dumps([dict(item) for item in self.profile.source_repos], ensure_ascii=False),
            "checkpoints_json": json.dumps([dict(item) for item in self.profile.checkpoints], ensure_ascii=False),
            "conda_env_json": json.dumps(conda_env, ensure_ascii=False),
            "backend_stage": self.profile.backend_stage,
            **source_context,
        }
        # Overlay any extra string/numeric parameters into the context.
        for key, value in extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                context[str(key)] = "" if value is None else str(value)
        return context

    def _runtime_env(self, context: Mapping[str, Any]) -> dict[str, str]:
        """Establish environment variables overlaid with conda/cuda parameters."""
        env = os.environ.copy()
        env.update(self.env)
        env_prefix = str(context.get("conda_env_prefix") or "")
        if env_prefix:
            bin_dir = str(Path(env_prefix) / "bin")
            lib_dir = str(Path(env_prefix) / "lib")
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
            env["CONDA_PREFIX"] = env_prefix
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        return env

    def _format_command(self, context: Mapping[str, Any]) -> list[str]:
        """Format the profile command template with resolved execution context variables."""
        return [part.format(**context) for part in self.command_template]

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute or plan the generation request using command-line templates."""
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix=f"{_safe_name(self.model_id)}_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        context = self._context(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            run_dir=run_dir,
            extra=kwargs,
        )
        command = self._format_command(context) if self.command_template else []
        plan_path = run_dir / "runtime_profile_plan.json"
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "command": command,
        }
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if plan_only:
            artifact_path = plan_path
            if output_path is not None:
                requested_output = Path(output_path).expanduser().resolve()
                if requested_output.suffix.lower() == ".json":
                    requested_output.parent.mkdir(parents=True, exist_ok=True)
                    requested_output.write_text(
                        json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    artifact_path = requested_output
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": "runtime_profile_plan",
                "artifact_path": str(artifact_path),
                "run_dir": str(run_dir),
                "plan_path": str(artifact_path),
                "command": command,
                "runtime": "worldfoundry.runtime_profile.plan",
                "backend_quality": "plan",
                "profile": self.profile.to_dict(),
            }
        return {
            "status": "blocked",
            "model_id": self.model_id,
            "artifact_kind": "runtime_profile_plan",
            "artifact_path": str(plan_path),
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "command": command,
            "runtime": "worldfoundry.runtime_profile.vendor_blocked",
            "backend_quality": "vendor_blocked",
            "blocked_reason": "official runtime not yet vendored into WorldFoundry",
            "profile": self.profile.to_dict(),
        }


# ── Utility helpers ──────────────────────────────────────────


def available_runtime_profile_ids() -> tuple[str, ...]:
    """Retrieve a sorted list of all available runtime profile IDs."""
    return tuple(sorted(load_runtime_profiles()))


def resolve_existing_tool(name: str) -> str:
    """Resolve the absolute path of a command-line tool, falling back to its name."""
    resolved = shutil.which(name)
    return resolved or name

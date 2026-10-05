"""Model-catalog manifest loaders and conversion utilities.

Load YAML model-catalog files, parse them into :class:`ModelCatalogManifest`
instances, and convert those manifests (or :class:`ModelZooEntry` objects)
into the public :class:`WorldModelManifest` contract used by the evaluation
harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import yaml

from ...api import WorldModelManifest
from ...api.registry import ModelRegistry
from ...utils import DATA_ROOT
from .schema import ModelZooEntry, load_entries, select_default_variant


# ── Type aliases and constants ───────────────────────────────

SourceModelMap = Mapping[str, Any]
SourceFamilyMaps = Mapping[str, SourceModelMap]
DEFAULT_MODEL_CATALOG_ROOT = DATA_ROOT / "models" / "catalog"


# ── Coercion helpers ─────────────────────────────────────────

def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerce any scalar or iterable value to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    """Coerce a value to a dictionary, or return an empty dictionary."""
    return dict(value) if isinstance(value, Mapping) else {}


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Extract and return unique non-empty stripped strings in order."""
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _source_status(value: Mapping[str, Any]) -> str:
    """Resolve source availability status from an availability/source dictionary."""
    return str(value.get("source") or value.get("status") or value.get("source_status") or "unknown")


def _checkpoint_items(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Extract checkpoint configuration dictionaries from mapping attributes."""
    if not value:
        return ()
    if any(key in value for key in ("repo_id", "hf_repo_id", "path", "filename", "local_dir")):
        return (value,)
    items: list[Mapping[str, Any]] = []
    for key, item in value.items():
        if isinstance(item, Mapping):
            mapped = dict(item)
            mapped.setdefault("id", str(key))
            items.append(mapped)
    return tuple(items)


def _dedupe(values: Iterable[str | None]) -> tuple[str, ...]:
    """Filter out empty/duplicate strings from an iterable, preserving order."""
    results: list[str] = []
    for value in values:
        text = str(value).strip() if value else ""
        if text and text not in results:
            results.append(text)
    return tuple(results)


def _aliases_for_entry(entry: ModelZooEntry) -> tuple[str, ...]:
    """Generate all alternative alias strings for a model zoo entry."""
    return _dedupe(
        value
        for value in (
            *entry.aliases,
            entry.name,
            entry.hf_repo_id,
            *(variant.variant_id for variant in entry.variants),
        )
        if value and value != entry.model_id
    )


# ── Output artifact deduction ─────────────────────────────────

def _coerce_model_zoo_entry(value: ModelZooEntry | Mapping[str, Any]) -> ModelZooEntry:
    """Coerce a dictionary mapping or ModelZooEntry instance to a ModelZooEntry."""
    if isinstance(value, ModelZooEntry):
        return value
    if isinstance(value, Mapping):
        return ModelZooEntry.from_dict(value)
    raise TypeError(f"expected ModelZooEntry or mapping, got {type(value).__name__}")


def _output_artifacts_for_tasks(tasks: tuple[str, ...]) -> tuple[str, ...]:
    """Deduce expected output artifacts based on task definitions.

    Maps well-known task keywords (``"vla"``, ``"3d"``, ``"4d"``, etc.) to
    their canonical artifact types, falling back to ``"generated_artifact"``
    when no keyword matches.
    """
    outputs: list[str] = []
    normalized = " ".join(tasks).lower()
    if "vla" in normalized or "robot" in normalized or "policy" in normalized:
        outputs.extend(("actions", "action_trace", "rollout_video"))
    if "vam" in normalized or "latent_action" in normalized or "video_action" in normalized:
        outputs.extend(("action_tokens", "predicted_video", "plan_trace"))
    if "video" in normalized and not any(
        token in normalized for token in ("latent_action", "policy", "robot", "video_action", "vla", "wam")
    ):
        outputs.append("generated_video")
    if "3d" in normalized or "mesh" in normalized or "point" in normalized:
        outputs.append("generated_3d_asset")
    if "4d" in normalized:
        outputs.append("generated_4d_scene")
    if "world" in normalized and "generated_world" not in outputs:
        outputs.append("generated_world")
    if "wam" in normalized:
        outputs.extend(("world_state", "session_trace"))
    return _dedupe(outputs) or ("generated_artifact",)


def _output_artifacts_for_entry(entry: ModelZooEntry, capabilities: tuple[str, ...]) -> tuple[str, ...]:
    """Retrieve output artifacts for a model entry, fallback to task-based deduction."""
    if entry.output_artifacts:
        return entry.output_artifacts
    if entry.integration_status == "blocked" and entry.runner_entry_kind != "runnable_runner":
        return ("blocked_plan",)
    return _output_artifacts_for_tasks(capabilities)


def _provider_for_entry(entry: ModelZooEntry) -> str:
    """Determine the model provider from variant or source metadata."""
    if entry.provider:
        return entry.provider
    for variant in entry.variants:
        if variant.provider:
            return variant.provider
    if entry.source_status == "api":
        return "api"
    if entry.hf_repo_ids:
        return "huggingface"
    if entry.official_repo_url:
        return "official_repo"
    return entry.source_status


def _variant_metadata(variant: Any) -> dict[str, Any]:
    """Extract normalized dictionary metadata for a model variant."""
    data = variant.to_dict()
    data["verification_status"] = variant.verification_status
    data["runner_entry_kind"] = variant.runner_entry_kind
    data["runnable_runner"] = variant.is_runnable_runner_entry
    data["integration"] = {
        "status": variant.integration_status,
        "verification_status": variant.verification_status,
    }
    return data


# ── Manifest dataclass ────────────────────────────────────────

@dataclass(frozen=True)
class ModelCatalogManifest:
    """Target model-catalog manifest from ``data/models/catalog``.

    Attributes:
        model_id: Primary identifier for the model.
        name: Human-readable display name.
        family: Provider or organisation family.
        domain: Task domain (e.g. ``"world_generation"``).
        capabilities: Task-family and modality metadata.
        availability: Source availability and licensing info.
        sources: Official repos (GitHub, HuggingFace, etc.).
        checkpoints: HuggingFace checkpoint references.
        integration: Integration status and runtime profile.
        evidence: Supplementary evidence notes.
        aliases: Alternative identifiers for lookup.
        schema_version: Manifest schema version (default 2).
    """

    model_id: str
    name: str = ""
    family: str = ""
    domain: str = ""
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    availability: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Any] = field(default_factory=dict)
    checkpoints: Mapping[str, Any] = field(default_factory=dict)
    integration: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    schema_version: int | str = 2

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelCatalogManifest":
        """Instantiate a ModelCatalogManifest from a mapping dictionary representation."""
        model_id = str(data.get("model_id") or data.get("id") or "")
        manifest = cls(
            schema_version=data.get("schema_version", 2),
            model_id=model_id,
            name=str(data.get("name") or data.get("display_name") or data.get("taxonomy_name") or model_id),
            family=str(data.get("family") or data.get("provider") or ""),
            domain=str(data.get("domain") or data.get("category") or ""),
            capabilities=_mapping_or_empty(data.get("capabilities")),
            availability=_mapping_or_empty(data.get("availability") or data.get("source")),
            sources=_mapping_or_empty(data.get("sources") or data.get("official_sources")),
            checkpoints=_mapping_or_empty(data.get("checkpoints") or data.get("checkpoint")),
            integration=_mapping_or_empty(data.get("integration")),
            evidence=_mapping_or_empty(data.get("evidence")),
            aliases=_tuple_of_str(data.get("aliases") or data.get("alias")),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_model_zoo_entry(cls, entry: ModelZooEntry) -> "ModelCatalogManifest":
        """Build a ModelCatalogManifest from a ModelZooEntry instance."""
        github = entry.official_repo_url
        hf_items = [{"repo_id": repo_id} for repo_id in entry.hf_repo_ids]
        checkpoint_refs = {
            str(index): ref.to_dict()
            for index, ref in enumerate(entry.checkpoint_refs or (entry.checkpoint,), start=1)
            if ref.hf_repo_id
        }
        manifest = cls(
            model_id=entry.model_id,
            name=entry.name or entry.model_id,
            family=entry.provider or "",
            domain=entry.tasks[0] if entry.tasks else "",
            capabilities={"task_family": entry.tasks[0] if entry.tasks else "", "modalities": list(entry.output_artifacts)},
            availability={
                "source": entry.source.status,
                "license": entry.license,
                "requires_auth": entry.requires_auth,
            },
            sources={
                "github": {"url": github} if github else {},
                "huggingface": hf_items,
            },
            checkpoints=checkpoint_refs,
            integration={
                "status": entry.integration_status,
                "runtime_profile": entry.runtime_profile,
                "pipeline_target": entry.pipeline_target,
                "pipeline_binding": entry.pipeline_binding,
                "runner": entry.runner_target,
            },
            evidence={"notes": list(entry.notes)},
            aliases=_aliases_for_entry(entry),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Validate integrity and schema conformity of the manifest attributes."""
        if not self.model_id:
            raise ValueError("model catalog manifest requires model_id.")
        if self.integration:
            status = str(self.integration.get("status") or "")
            if status and status not in {"integrated", "planned", "blocked"}:
                raise ValueError(f"model catalog manifest {self.model_id!r} has invalid integration.status: {status!r}")

    @property
    def task_family(self) -> str:
        """Get the primary task family of this manifest."""
        return str(self.capabilities.get("task_family") or "")

    @property
    def modalities(self) -> tuple[str, ...]:
        """Get the expected output modalities of this manifest."""
        return _tuple_of_str(self.capabilities.get("modalities"))

    @property
    def source_status(self) -> str:
        """Get the availability source status of this manifest."""
        return _source_status(self.availability)

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest properties into a serializable dictionary mapping."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "name": self.name,
            "family": self.family,
            "domain": self.domain,
            "capabilities": dict(self.capabilities),
            "availability": dict(self.availability),
            "sources": dict(self.sources),
            "checkpoints": dict(self.checkpoints),
            "integration": dict(self.integration),
            "evidence": dict(self.evidence),
            "aliases": list(self.aliases),
        }

    def to_world_model_manifest(self) -> WorldModelManifest:
        """Convert this manifest to the public WorldModelManifest representation."""
        capabilities = _unique((self.task_family, *self.modalities, self.domain))
        checkpoints = [dict(item) for item in _checkpoint_items(self.checkpoints)]
        return WorldModelManifest(
            model_id=self.model_id,
            name=self.name,
            aliases=self.aliases,
            provider=self.source_status,
            capabilities=capabilities,
            supported_tasks=(self.task_family,) if self.task_family else (),
            required_artifacts=tuple(str(item.get("repo_id") or item.get("hf_repo_id") or item.get("id") or "") for item in checkpoints),
            tags=_unique((self.family, self.domain, str(self.integration.get("status") or ""))),
            metadata={
                "family": self.family,
                "domain": self.domain,
                "availability": dict(self.availability),
                "sources": dict(self.sources),
                "checkpoints": dict(self.checkpoints),
                "integration": dict(self.integration),
                "evidence": dict(self.evidence),
            },
        )


# ── Catalog loading ──────────────────────────────────────────

def _catalog_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """Retrieve all YAML catalog file paths from a root directory recursively."""
    path = Path(root) if root is not None else DEFAULT_MODEL_CATALOG_ROOT
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    return tuple(sorted(item for item in path.rglob("*.y*ml") if item.is_file()))


def _iter_catalog_mappings(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load and iterate over model catalog mappings defined in a YAML file.

    YAML parse errors are re-raised with the file path in the message.
    """
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"failed to parse model catalog file {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"model catalog file must contain a mapping: {path}")
    entries = payload.get("models") or payload.get("entries") or payload.get("manifests")
    if entries is None:
        return (payload,) if payload.get("model_id") or payload.get("id") else ()
    if not isinstance(entries, Iterable) or isinstance(entries, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"model catalog collection must be a list: {path}")
    return tuple(item for item in entries if isinstance(item, Mapping))


def load_model_catalog_manifest(path: str | Path) -> ModelCatalogManifest:
    """Load a single model catalog manifest from a file path."""
    entries = _iter_catalog_mappings(Path(path))
    if len(entries) != 1:
        raise ValueError(f"expected one model catalog manifest in {path}, found {len(entries)}")
    return catalog_manifest_from_mapping(entries[0])


def catalog_manifest_from_mapping(data: Mapping[str, Any]) -> ModelCatalogManifest:
    """Parse a mapping representation into a ModelCatalogManifest, supporting schema versions."""
    if str(data.get("schema_version") or "") == "2":
        return ModelCatalogManifest.from_mapping(data)
    return ModelCatalogManifest.from_model_zoo_entry(ModelZooEntry.from_dict(data))


def _catalog_mapping_priority(data: Mapping[str, Any]) -> int:
    """Evaluate loading priority based on the mapping schema version.

    Schema version 2 entries receive priority 1; all others receive 0,
    so that v2 manifests win during de-duplication.
    """
    return 1 if str(data.get("schema_version") or "") == "2" else 0


def load_model_catalog_manifests(root: str | Path | None = None) -> tuple[ModelCatalogManifest, ...]:
    """Load and de-duplicate all model catalog manifests from a root directory.

    When multiple YAML files define the same ``model_id``, the entry with the
    higher schema version priority wins.
    """
    manifests_by_id: dict[str, tuple[int, ModelCatalogManifest]] = {}
    for path in _catalog_paths(root):
        for data in _iter_catalog_mappings(path):
            try:
                manifest = catalog_manifest_from_mapping(data)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{path}: {exc}") from exc
            priority = _catalog_mapping_priority(data)
            existing = manifests_by_id.get(manifest.model_id)
            if existing is None or priority > existing[0]:
                manifests_by_id[manifest.model_id] = (priority, manifest)
    manifests = [item[1] for item in manifests_by_id.values()]
    return tuple(sorted(manifests, key=lambda item: item.model_id))


def load_world_model_manifests_from_catalog(root: str | Path | None = None) -> tuple[WorldModelManifest, ...]:
    """Retrieve public WorldModelManifest contracts parsed from catalog files."""
    return tuple(manifest.to_world_model_manifest() for manifest in load_model_catalog_manifests(root))


def model_zoo_entry_to_world_model_manifest(value: ModelZooEntry | Mapping[str, Any]) -> WorldModelManifest:
    """Convert one model-zoo entry into the public WorldFoundry model manifest contract.

    Builds a :class:`WorldModelManifest` with full metadata including variant
    details, pipeline routes, integration and verification status, and
    deduplicated checkpoint references.

    Args:
        value: A :class:`ModelZooEntry` instance or a raw mapping to coerce.
    """

    entry = _coerce_model_zoo_entry(value)
    variant_tasks = tuple(variant.task for variant in entry.variants)
    capabilities = _dedupe((*entry.tasks, *variant_tasks)) or ("world_generation",)
    default_variant = select_default_variant(entry, allow_runner_target_fallback=False)
    default_runner_target = default_variant.runner_target if default_variant is not None else (
        entry.runner_target if entry.integration_status == "integrated" else None
    )
    default_runtime_profile = default_variant.runtime_profile if default_variant is not None else (
        entry.runtime_profile if default_runner_target else None
    )
    default_pipeline_target = default_variant.pipeline_target if default_variant is not None else (
        entry.pipeline_target if default_runner_target else None
    )
    default_pipeline_binding = default_variant.pipeline_binding if default_variant is not None else (
        entry.pipeline_binding if default_runner_target else None
    )
    required_artifacts = _dedupe(entry.hf_repo_ids)
    metadata = {
        "provider": _provider_for_entry(entry),
        "source": entry.source.to_dict(),
        "checkpoint": entry.checkpoint.to_dict(),
        "checkpoint_refs": [item.to_dict() for item in entry.checkpoint_refs],
        "variants": [_variant_metadata(item) for item in entry.variants],
        "variant_ids": [item.variant_id for item in entry.variants],
        "integrated_variant_ids": [
            item.variant_id for item in entry.variants if item.integration_status == "integrated"
        ],
        "runnable_runner_variant_ids": [
            item.variant_id for item in entry.variants if item.is_runnable_runner_entry
        ],
        "hf_repo_ids": list(entry.hf_repo_ids),
        "integration_status": entry.integration_status,
        "verification_status": entry.verification_status,
        "integration": {
            "status": entry.integration_status,
            "verification_status": entry.verification_status,
        },
        "demo_parity": entry.demo_parity.to_dict(),
        "runner_parity": entry.runner_parity.to_dict(),
        "runner_entry_kind": entry.runner_entry_kind,
        "runnable_runner": entry.is_runnable_runner_entry,
        "requires_auth": entry.requires_auth,
        "install_profile": entry.install_profile,
        "runner_target": entry.runner_target,
        "pipeline_target": entry.pipeline_target,
        "pipeline_binding": entry.pipeline_binding,
        "runtime_profile": entry.runtime_profile,
        "default_variant_id": default_variant.variant_id if default_variant is not None else None,
        "default_runner_target": default_runner_target,
        "default_pipeline_target": default_pipeline_target,
        "default_pipeline_binding": default_pipeline_binding,
        "default_runtime_profile": default_runtime_profile,
        "default_integration_status": (
            default_variant.integration_status if default_variant is not None else entry.integration_status
        ),
        "default_verification_status": (
            default_variant.verification_status if default_variant is not None else entry.verification_status
        ),
        "default_min_vram_gb": (
            default_variant.min_vram_gb if default_variant is not None else entry.min_vram_gb
        ),
        "min_vram_gb": entry.min_vram_gb,
        "notes": entry.notes,
    }
    return WorldModelManifest(
        model_id=entry.model_id,
        name=entry.name or entry.model_id,
        aliases=_aliases_for_entry(entry),
        provider=_provider_for_entry(entry),
        capabilities=capabilities,
        supported_tasks=capabilities,
        required_artifacts=required_artifacts,
        output_artifacts=_output_artifacts_for_entry(entry, capabilities),
        tags=(entry.source_status, entry.integration_status),
        metadata=metadata,
    )


def model_zoo_entries_to_world_model_manifests(
    entries: Iterable[ModelZooEntry | Mapping[str, Any]],
) -> tuple[WorldModelManifest, ...]:
    """Convert an iterable of model zoo entries/mappings to a tuple of WorldModelManifest instances."""
    return tuple(model_zoo_entry_to_world_model_manifest(entry) for entry in entries)


def load_world_model_manifests(path: str | Path) -> tuple[WorldModelManifest, ...]:
    """Load model entries from a path and convert them to WorldModelManifest instances."""
    return model_zoo_entries_to_world_model_manifests(load_entries(path))


def write_world_model_manifests_json(entries_path: str | Path, output_path: str | Path) -> Path:
    """Load manifests from a source file, and write their serialized representations to a JSON file."""
    manifests = load_world_model_manifests(entries_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [manifest.to_dict() for manifest in manifests]
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


# ── Source-map manifest builders ──────────────────────────────

def model_manifests_from_source_maps(
    loaders_by_family: SourceFamilyMaps,
    infers_by_family: SourceFamilyMaps,
    *,
    provider: str | Mapping[str, str] = "source",
    capabilities_by_family: Mapping[str, Iterable[str] | str] | None = None,
) -> tuple[WorldModelManifest, ...]:
    """Build model manifests from caller-supplied source loader/infer maps.

    Callers pass the mappings they want to expose, and the returned manifests
    can be registered with the core registry.  This function intentionally
    does not import the source example mappings.

    Args:
        loaders_by_family: Mapping of family name → model ID → loader callable.
        infers_by_family: Mapping of family name → model ID → infer callable.
        provider: Provider string or per-family provider mapping.
        capabilities_by_family: Per-family capability list override.
    """

    manifests: list[WorldModelManifest] = []
    family_names = sorted(set(loaders_by_family) | set(infers_by_family))

    for family in family_names:
        loaders = loaders_by_family.get(family, {})
        infers = infers_by_family.get(family, {})
        capabilities = _capabilities_for_family(family, capabilities_by_family)
        family_provider = _provider_for_family(family, provider)

        for model_ids in _model_id_groups(loaders, infers):
            model_id = model_ids[0]
            aliases = tuple(model_ids[1:])
            has_loader = any(candidate in loaders for candidate in model_ids)
            has_infer = any(candidate in infers for candidate in model_ids)
            loader_ref = _first_object_reference(loaders, model_ids)
            infer_ref = _first_object_reference(infers, model_ids)

            metadata: dict[str, Any] = {
                "family": family,
                "capabilities": capabilities,
                "has_loader": has_loader,
                "has_infer": has_infer,
                "provider": family_provider,
                "source_model_id": model_id,
                "source_model_ids": tuple(model_ids),
                "aliases": aliases,
            }
            if loader_ref:
                metadata["loader"] = loader_ref
            if infer_ref:
                metadata["infer"] = infer_ref

            manifests.append(
                WorldModelManifest(
                    model_id=model_id,
                    name=model_id,
                    provider=family_provider,
                    capabilities=capabilities,
                    metadata=metadata,
                )
            )

    return tuple(manifests)


# ── Source-map helper functions ──────────────────────────────

def build_model_manifests(
    loaders_by_family: SourceFamilyMaps,
    infers_by_family: SourceFamilyMaps,
    *,
    provider: str | Mapping[str, str] = "source",
    capabilities_by_family: Mapping[str, Iterable[str] | str] | None = None,
) -> tuple[WorldModelManifest, ...]:
    """Build a tuple of model manifests from source loader and inference family maps."""
    return model_manifests_from_source_maps(
        loaders_by_family,
        infers_by_family,
        provider=provider,
        capabilities_by_family=capabilities_by_family,
    )


def build_model_manifest_registry(
    loaders_by_family: SourceFamilyMaps,
    infers_by_family: SourceFamilyMaps,
    *,
    provider: str | Mapping[str, str] = "source",
    capabilities_by_family: Mapping[str, Iterable[str] | str] | None = None,
) -> ModelRegistry:
    """Build a ModelRegistry from source loader and inference family mappings."""
    manifests = model_manifests_from_source_maps(
        loaders_by_family,
        infers_by_family,
        provider=provider,
        capabilities_by_family=capabilities_by_family,
    )
    return ModelRegistry(manifests)


def _model_id_groups(loaders: SourceModelMap, infers: SourceModelMap) -> tuple[tuple[str, ...], ...]:
    """Group model IDs sharing identical loader/inference python object references."""
    groups: dict[tuple[int | None, int | None], list[str]] = {}
    for model_id in sorted(set(loaders) | set(infers)):
        signature = (
            id(loaders[model_id]) if model_id in loaders else None,
            id(infers[model_id]) if model_id in infers else None,
        )
        groups.setdefault(signature, []).append(str(model_id))

    return tuple(tuple(model_ids) for model_ids in sorted(groups.values(), key=lambda values: values[0]))


def _capabilities_for_family(
    family: str,
    capabilities_by_family: Mapping[str, Iterable[str] | str] | None,
) -> tuple[str, ...]:
    """Retrieve capabilities defined for a family, falling back to family name."""
    if capabilities_by_family is None or family not in capabilities_by_family:
        return (str(family),)
    return _string_tuple(capabilities_by_family[family])


def _provider_for_family(family: str, provider: str | Mapping[str, str]) -> str:
    """Resolve the provider identifier for a given family."""
    if isinstance(provider, Mapping):
        return str(provider.get(family, "source"))
    return str(provider)


def _string_tuple(values: Iterable[str] | str) -> tuple[str, ...]:
    """Coerce scalar string or iterable strings to a tuple of strings."""
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _first_object_reference(items: SourceModelMap, model_ids: Iterable[str]) -> str:
    """Find the first matching model ID in items and return its object reference string."""
    for model_id in model_ids:
        if model_id in items:
            return _object_reference(items[model_id])
    return ""


def _object_reference(value: Any) -> str:
    """Generate a dotted object import target string for any given python object."""
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", "")
    if module and qualname:
        return f"{module}:{qualname}"

    name = getattr(value, "__name__", "")
    if name:
        return str(name)

    value_type = type(value)
    type_module = getattr(value_type, "__module__", "")
    type_qualname = getattr(value_type, "__qualname__", value_type.__name__)
    if type_module:
        return f"{type_module}:{type_qualname}"
    return str(type_qualname)


__all__ = [
    "DEFAULT_MODEL_CATALOG_ROOT",
    "ModelCatalogManifest",
    "SourceFamilyMaps",
    "SourceModelMap",
    "build_model_manifest_registry",
    "build_model_manifests",
    "catalog_manifest_from_mapping",
    "load_model_catalog_manifest",
    "load_model_catalog_manifests",
    "load_world_model_manifests",
    "load_world_model_manifests_from_catalog",
    "model_manifests_from_source_maps",
    "model_zoo_entries_to_world_model_manifests",
    "model_zoo_entry_to_world_model_manifest",
    "write_world_model_manifests_json",
]

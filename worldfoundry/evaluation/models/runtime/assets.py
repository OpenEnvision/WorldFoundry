"""Runtime asset manifests for model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.models.runtime.profiles import RuntimeProfile, load_runtime_profiles
from worldfoundry.evaluation.utils import DATA_ROOT
from worldfoundry.runtime.assets import expand_worldfoundry_path

from ._shared import iter_manifest_mappings, schema_version_or_none, tuple_of_str, yaml_manifest_paths

# Default directory containing runtime asset manifest YAML files.
DEFAULT_RUNTIME_ASSETS_ROOT = DATA_ROOT / "models" / "runtime" / "assets"


def _tuple_of_mapping(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Coerce any sequence or single mapping to a tuple of dictionaries."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    return ()


def _mapping_of_mapping(value: Any) -> Mapping[str, Mapping[str, Any]]:
    """Coerce any mapping input to a dictionary of string-to-dictionary mappings."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): dict(item)
        for key, item in value.items()
        if isinstance(item, Mapping)
    }


# Module-local alias for the shared runtime loader helper.
_tuple_of_str = tuple_of_str


def _expand_asset_path(value: Any) -> str:
    """Expand and resolve environment variables/user variables in worldfoundry asset path."""
    if value is None:
        return ""
    return str(expand_worldfoundry_path(value))


# ── Core data models ──────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeAsset:
    """One checkpoint, repository, dataset, or auxiliary runtime asset.

    Attributes:
        asset_id: Unique identifier for the asset.
        kind: Asset type, e.g. ``"checkpoint"`` or ``"component"``.
        uri: Download URI for the asset (direct URL).
        repo_id: Hugging Face ``repo_id`` for the asset.
        local_dir: Resolved local filesystem directory for the asset.
        local_candidates: Tuple of alternative local paths to search.
        role: Functional role of this asset within the model runtime.
        revision: Git revision or commit SHA for the asset.
        gated: Whether the asset requires gated access approval on Hugging Face.
        required: Whether the asset is mandatory for the runtime to function.
        metadata: Arbitrary metadata mapping attached to the asset.
        source: Provenance label — ``"target"`` for manifest-sourced,
            ``"runtime_profile"`` for profile-derived assets.
    """

    asset_id: str
    kind: str
    uri: str = ""
    repo_id: str = ""
    local_dir: str = ""
    local_candidates: tuple[str, ...] = ()
    role: str = ""
    revision: str = ""
    gated: bool | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "target"

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        default_id: str,
        default_kind: str = "checkpoint",
        source: str = "target",
    ) -> "RuntimeAsset":
        """Build a :class:`RuntimeAsset` from a raw YAML mapping.

        Supports multiple legacy key names (``"id"``, ``"name"``, ``"type"``)
        and resolves local paths via :func:`expand_worldfoundry_path`.

        Args:
            data: Raw mapping loaded from a YAML manifest.
            default_id: Fallback ``asset_id`` when none is provided.
            default_kind: Fallback ``kind`` when none is provided.
            source: Provenance label for the asset.

        Returns:
            A validated :class:`RuntimeAsset` instance.
        """
        metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        asset = cls(
            asset_id=str(data.get("asset_id") or data.get("id") or data.get("name") or default_id),
            kind=str(data.get("kind") or data.get("type") or default_kind),
            uri=str(data.get("uri") or data.get("url") or ""),
            repo_id=str(data.get("repo_id") or ""),
            local_dir=_expand_asset_path(data.get("local_dir")),
            local_candidates=tuple(_expand_asset_path(item) for item in _tuple_of_str(data.get("local_candidates"))),
            role=str(data.get("role") or ""),
            revision=str(data.get("revision") or data.get("head_sha") or data.get("confirmed_ref") or ""),
            gated=bool(data["gated"]) if "gated" in data else None,
            required=bool(data.get("required", True)),
            metadata=dict(metadata),
            source=source,
        )
        asset.validate()
        return asset

    def validate(self) -> None:
        """Raise ``ValueError`` if the asset is missing required ``asset_id`` or ``kind``."""
        if not self.asset_id:
            raise ValueError("runtime asset requires asset_id.")
        if not self.kind:
            raise ValueError(f"runtime asset {self.asset_id!r} requires kind.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the asset to a plain dictionary suitable for serialization."""
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "uri": self.uri,
            "repo_id": self.repo_id,
            "local_dir": self.local_dir,
            "local_candidates": list(self.local_candidates),
            "role": self.role,
            "revision": self.revision,
            "gated": self.gated,
            "required": self.required,
            "metadata": dict(self.metadata),
            "source": self.source,
        }


@dataclass(frozen=True)
class RuntimeAssetProfile:
    """Resolved runtime assets for one model.

    Attributes:
        asset_profile_id: Unique identifier for this asset profile.
        model_id: Model identifier that these assets belong to.
        assets: Tuple of resolved :class:`RuntimeAsset` instances.
        roots: Mapping of named root directories to their resolved paths.
        components: Nested mapping of component name → component configuration.
        source_repos: Tuple of source-repository mappings.
        notes: Free-text notes and blocker descriptions.
        metadata: Arbitrary metadata mapping for the profile.
        source: Provenance label — ``"target"`` or ``"runtime_profile"``.
        schema_version: Schema version marker; only ``2`` is currently supported.
    """

    asset_profile_id: str
    model_id: str
    assets: tuple[RuntimeAsset, ...] = ()
    roots: Mapping[str, str] = field(default_factory=dict)
    components: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source_repos: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "target"
    schema_version: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "target") -> "RuntimeAssetProfile":
        """Build a :class:`RuntimeAssetProfile` from a raw YAML mapping.

        Args:
            data: Raw mapping loaded from a YAML asset manifest.
            source: Provenance label for the profile.

        Returns:
            A validated :class:`RuntimeAssetProfile` instance.
        """
        profile_id = str(data.get("asset_profile_id") or data.get("id") or data.get("model_id") or "")
        model_id = str(data.get("model_id") or data.get("id") or profile_id)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        roots = {str(key): _expand_asset_path(value) for key, value in dict(data.get("roots") or {}).items()}
        assets = tuple(
            RuntimeAsset.from_mapping(item, default_id=f"{model_id}-{index}", source=source)
            for index, item in enumerate(_asset_mappings_from_profile(data), start=1)
        )
        profile = cls(
            asset_profile_id=profile_id,
            model_id=model_id,
            schema_version=_schema_version(data.get("schema_version")),
            assets=assets,
            roots=roots,
            components=_mapping_of_mapping(data.get("components")),
            source_repos=_tuple_of_mapping(data.get("source_repos")),
            notes=_tuple_of_str(data.get("notes")),
            metadata=dict(metadata),
            source=source,
        )
        profile.validate()
        return profile

    @classmethod
    def from_runtime_profile(cls, profile: RuntimeProfile, *, source: str = "runtime_profile") -> "RuntimeAssetProfile":
        """Derive a :class:`RuntimeAssetProfile` from a :class:`RuntimeProfile`'s checkpoints.

        Args:
            profile: Source :class:`RuntimeProfile` whose checkpoints are converted.
            source: Provenance label (defaults to ``"runtime_profile"``).

        Returns:
            A validated :class:`RuntimeAssetProfile` instance.
        """
        assets = tuple(
            RuntimeAsset.from_mapping(
                checkpoint,
                default_id=str(checkpoint.get("role") or checkpoint.get("repo_id") or f"{profile.model_id}-{index}"),
                default_kind="checkpoint",
                source=source,
            )
            for index, checkpoint in enumerate(profile.checkpoints, start=1)
            if isinstance(checkpoint, Mapping)
        )
        asset_profile = cls(
            asset_profile_id=profile.model_id,
            model_id=profile.model_id,
            assets=assets,
            roots={},
            components={},
            source_repos=tuple(dict(item) for item in profile.source_repos),
            notes=profile.notes,
            metadata={"task_family": profile.task_family, "artifact_kind": profile.artifact_kind},
            source=source,
        )
        asset_profile.validate()
        return asset_profile

    def validate(self) -> None:
        """Raise ``ValueError`` if the profile is missing required fields or uses an unsupported schema."""
        if self.schema_version is not None and self.schema_version != 2:
            raise ValueError(
                f"runtime asset profile {self.asset_profile_id!r} uses unsupported schema_version {self.schema_version!r}."
            )
        if not self.asset_profile_id:
            raise ValueError("runtime asset profile requires asset_profile_id.")
        if not self.model_id:
            raise ValueError(f"runtime asset profile {self.asset_profile_id!r} requires model_id.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a plain dictionary suitable for serialization."""
        return {
            "schema_version": self.schema_version,
            "asset_profile_id": self.asset_profile_id,
            "model_id": self.model_id,
            "assets": [asset.to_dict() for asset in self.assets],
            "roots": dict(self.roots),
            "components": {key: dict(value) for key, value in self.components.items()},
            "source_repos": [dict(item) for item in self.source_repos],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "source": self.source,
        }


# ── Schema version / manifest scanning (shared helpers) ───────

_schema_version = schema_version_or_none


def _manifest_paths(root: str | Path | None) -> tuple[Path, ...]:
    """Retrieve all YAML manifest paths found under a root directory recursively."""
    return yaml_manifest_paths(root, default_root=DEFAULT_RUNTIME_ASSETS_ROOT)


def _iter_asset_profile_mappings(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load and iterate over asset profiles defined under key tags in a YAML manifest file."""
    return iter_manifest_mappings(
        path,
        collection_keys=("asset_profiles", "assets_profiles", "profiles"),
        id_keys=("model_id", "asset_profile_id", "id"),
        kind="runtime asset",
    )


def _asset_mappings_from_profile(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Extract and compile raw asset configurations from checkpoints/components/assets in a profile."""
    explicit_assets = list(_tuple_of_mapping(data.get("assets")))
    checkpoints = data.get("checkpoints")
    if isinstance(checkpoints, Mapping):
        for key, item in checkpoints.items():
            if not isinstance(item, Mapping):
                continue
            mapped = dict(item)
            mapped.setdefault("id", str(key))
            mapped.setdefault("role", str(key))
            mapped.setdefault("kind", "checkpoint")
            explicit_assets.append(mapped)
    elif isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes, bytearray)):
        for index, item in enumerate(checkpoints, start=1):
            if not isinstance(item, Mapping):
                continue
            mapped = dict(item)
            mapped.setdefault("id", str(mapped.get("role") or f"checkpoint-{index}"))
            mapped.setdefault("kind", "checkpoint")
            explicit_assets.append(mapped)

    components = data.get("components")
    if isinstance(components, Mapping):
        for key, item in components.items():
            if not isinstance(item, Mapping):
                continue
            mapped = dict(item)
            mapped.setdefault("id", str(key))
            mapped.setdefault("role", str(key))
            mapped.setdefault("kind", "component")
            explicit_assets.append(mapped)
    return tuple(explicit_assets)


# ── Public loaders ────────────────────────────────────────────


def load_runtime_asset_profile(path: str | Path) -> RuntimeAssetProfile:
    """Load a single :class:`RuntimeAssetProfile` from a YAML file path.

    Args:
        path: Path to a YAML file containing exactly one asset profile.

    Raises:
        ValueError: If the file contains zero or more than one profile.
    """
    entries = _iter_asset_profile_mappings(Path(path))
    if len(entries) != 1:
        raise ValueError(f"expected one runtime asset profile in {path}, found {len(entries)}")
    return RuntimeAssetProfile.from_mapping(entries[0])


def load_runtime_asset_profiles(
    root: str | Path | None = None,
    *,
    runtime_profiles: Mapping[str, RuntimeProfile] | None = None,
    runtime_profile_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, RuntimeAssetProfile]:
    """Load and aggregate all :class:`RuntimeAssetProfile` entries under a directory root.

    First derives profiles from :class:`RuntimeProfile` checkpoints, then
    overlays any manifest-sourced profiles found under ``root``.

    Args:
        root: Directory root to scan for YAML asset manifests.
        runtime_profiles: Pre-loaded runtime profiles to derive asset profiles from.
        runtime_profile_kwargs: Forwarded to :func:`load_runtime_profiles`
            when ``runtime_profiles`` is ``None``.

    Returns:
        A ``dict`` keyed by ``model_id`` mapping to resolved :class:`RuntimeAssetProfile` instances.
    """
    source_profiles = runtime_profiles or load_runtime_profiles(**dict(runtime_profile_kwargs or {}))
    profiles = {
        model_id: RuntimeAssetProfile.from_runtime_profile(profile)
        for model_id, profile in source_profiles.items()
    }
    for path in _manifest_paths(root):
        for data in _iter_asset_profile_mappings(path):
            profile = RuntimeAssetProfile.from_mapping(data, source="target")
            profiles[profile.model_id] = profile
    return profiles


def load_runtime_asset_profile_by_id(model_id: str, **kwargs: Any) -> RuntimeAssetProfile:
    """Load a single :class:`RuntimeAssetProfile` by model/profile ID.

    Args:
        model_id: The unique identifier of the desired asset profile.
        **kwargs: Forwarded to :func:`load_runtime_asset_profiles`.

    Raises:
        KeyError: If ``model_id`` is not found among loaded profiles.
    """
    profiles = load_runtime_asset_profiles(**kwargs)
    if model_id not in profiles:
        raise KeyError(f"unknown runtime asset profile: {model_id}")
    return profiles[model_id]


resolve_runtime_asset_profile = load_runtime_asset_profile_by_id


__all__ = [
    "DEFAULT_RUNTIME_ASSETS_ROOT",
    "RuntimeAsset",
    "RuntimeAssetProfile",
    "load_runtime_asset_profile",
    "load_runtime_asset_profile_by_id",
    "load_runtime_asset_profiles",
    "resolve_runtime_asset_profile",
]

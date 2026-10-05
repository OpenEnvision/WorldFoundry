"""Data-backed alias helpers for model pipeline bindings.

Loads YAML alias definitions, builds :class:`PipelineAliasGroup` /
:class:`PipelineAliasRegistry` objects, and resolves alternative model
names to their canonical IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from worldfoundry.evaluation.utils import DATA_ROOT

HandlerT = TypeVar("HandlerT")

DEFAULT_PIPELINE_ALIASES_ROOT = DATA_ROOT / "models" / "bindings" / "aliases"

# ── Runtime video alias table ────────────────────────────────────────
# Each entry maps a canonical video pipeline ID to its accepted aliases.
RUNTIME_VIDEO_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("allegro_ti2v", ("allegro",)),
    ("cogvideox_2b_t2v", ("cogvideox-2b-t2v",)),
    ("cogvideox_5b_i2v", ("cogvideox-5b-i2v",)),
    ("cogvideox_5b_t2v", ("cogvideox-5b-t2v", "cogvideox", "cogvideox-5b", "cogvideox-t2v")),
    ("dynamicrafter_1024_i2v", ("dynamicrafter-1024-i2v",)),
    ("dynamicrafter_512_i2v", ("dynamicrafter-512-i2v",)),
    ("easyanimate_i2v", ("easyanimate-i2v", "easyanimate")),
    ("gen_3_i2v", ("gen-3-i2v", "gen3-i2v", "gen-3")),
    ("ltx_video_i2v", ("ltx-video-i2v", "ltx-video")),
    ("ltx2_i2v", ("ltx2-i2v", "ltx2", "ltx-2", "ltx-2-i2v")),
    ("ltx2_3_i2v", ("ltx2.3", "ltx-2.3", "ltx2.3-i2v", "ltx-2.3-i2v")),
    ("minimax_i2v", ("minimax-i2v",)),
    ("t2v_turbo_t2v", ("t2v-turbo-t2v", "t2v-turbo")),
    ("vchitect_2_t2v", ("vchitect-2-t2v", "vchitect2-t2v", "vchitect-2")),
    ("videocrafter1_i2v", ("videocrafter1-i2v",)),
    ("videocrafter1_t2v", ("videocrafter1-t2v",)),
    ("videocrafter2_t2v", ("videocrafter2-t2v",)),
    ("wan2.1_i2v", ("wan2.1-i2v", "wan2p1-i2v", "wan2-1-i2v")),
    ("wan2.1_t2v", ("wan2.1-t2v", "wan2p1-t2v", "wan2-1-t2v")),
)


# ── Alias group dataclass ────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineAliasGroup:
    """Group of alternative aliases mapping to a single canonical model ID.

    Attributes:
        canonical_id: The primary, authoritative model identifier.
        aliases: Alternative names that resolve to ``canonical_id``.
        domain: Logical domain or sub-directory the group belongs to (e.g. ``"video"``).
    """

    canonical_id: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    domain: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, domain: str = "") -> "PipelineAliasGroup":
        """Instantiate a PipelineAliasGroup from a mapping dictionary."""
        aliases = data.get("aliases") or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        return cls(
            canonical_id=str(data.get("canonical_id") or data.get("canonical") or data.get("model_id") or ""),
            aliases=tuple(str(alias) for alias in aliases if str(alias).strip()),
            domain=str(data.get("domain") or domain),
        )

    def validate(self) -> None:
        """Validate integrity and self-referencing check of the alias group."""
        if not self.canonical_id:
            raise ValueError("pipeline alias group requires canonical_id.")
        if any(_alias_key(alias) == _alias_key(self.canonical_id) for alias in self.aliases):
            raise ValueError(f"alias group {self.canonical_id!r} aliases itself.")


# ── Alias registry ──────────────────────────────────────────────────

class PipelineAliasRegistry:
    """Registry maintaining pipeline alias groups with lookups.

    Stores groups keyed by normalized canonical IDs and alias strings,
    enabling fast resolution of any alias to its canonical form.
    """

    def __init__(self, groups: tuple[PipelineAliasGroup, ...] = ()) -> None:
        """Initialize the PipelineAliasRegistry with a collection of groups."""
        self._frozen = False
        self._canonical: dict[str, PipelineAliasGroup] = {}
        self._aliases: dict[str, PipelineAliasGroup] = {}
        for group in groups:
            self.register(group)

    def register(self, group: PipelineAliasGroup) -> None:
        """Register a single PipelineAliasGroup and its aliases into the registry."""
        if self._frozen:
            raise RuntimeError(
                "cannot register on a cached PipelineAliasRegistry; "
                "construct a new instance or call clear_pipeline_alias_registry_cache()"
            )
        group.validate()
        canonical_key = _alias_key(group.canonical_id)
        existing = self._canonical.get(canonical_key)
        if existing is not None and existing != group:
            raise ValueError(f"duplicate pipeline alias canonical_id: {group.canonical_id!r}")
        self._canonical[canonical_key] = group
        for alias in group.aliases:
            alias_key = _alias_key(alias)
            existing_group = self._aliases.get(alias_key) or self._canonical.get(alias_key)
            if existing_group is not None and existing_group != group:
                raise ValueError(f"duplicate pipeline alias: {alias!r}")
            self._aliases[alias_key] = group

    def canonical_id(self, key: str) -> str:
        """Resolve any given alias or canonical key to its primary canonical ID."""
        normalized = _alias_key(key)
        if normalized in self._canonical:
            return self._canonical[normalized].canonical_id
        try:
            return self._aliases[normalized].canonical_id
        except KeyError as exc:
            raise KeyError(f"unknown pipeline alias: {key!r}") from exc

    def aliases_for(self, canonical_id: str) -> tuple[str, ...]:
        """Get all alternative aliases associated with a primary canonical ID."""
        return self._canonical[_alias_key(canonical_id)].aliases

    def as_alias_mapping(self) -> dict[str, str]:
        """Compile a direct mapping of all aliases back to their canonical IDs."""
        return {
            alias: group.canonical_id
            for group in self._canonical.values()
            for alias in group.aliases
        }

    def list(self) -> tuple[PipelineAliasGroup, ...]:
        """List all unique registered PipelineAliasGroups, sorted by domain and canonical ID."""
        return tuple(sorted(self._canonical.values(), key=lambda item: (item.domain, item.canonical_id)))


# ── Alias mapping helpers ────────────────────────────────────────────

def build_alias_mapping(handlers: Mapping[str, HandlerT], aliases: Mapping[str, str]) -> dict[str, HandlerT]:
    """Copy canonical handlers and attach aliases with explicit validation.

    Args:
        handlers: Mapping of canonical pipeline names to handler objects.
        aliases: Mapping of alias names to their canonical pipeline names.

    Raises:
        KeyError: If an alias references a canonical name absent from ``handlers``.
    """
    mapping = dict(handlers)
    for alias, canonical_name in aliases.items():
        if canonical_name not in mapping:
            raise KeyError(f"Alias {alias!r} points to unknown canonical pipeline {canonical_name!r}.")
        mapping[alias] = mapping[canonical_name]
    return mapping


def build_runtime_video_mapping(handlers: Mapping[str, HandlerT]) -> dict[str, HandlerT]:
    """Build runtime-video dispatch entries from canonical handlers.

    Selects only the video pipelines listed in :data:`RUNTIME_VIDEO_ALIASES`
    and expands them with all their aliases.
    """
    canonical_handlers = {canonical_name: handlers[canonical_name] for canonical_name, _ in RUNTIME_VIDEO_ALIASES}
    aliases = {
        alias: canonical_name
        for canonical_name, canonical_aliases in RUNTIME_VIDEO_ALIASES
        for alias in canonical_aliases
    }
    return build_alias_mapping(canonical_handlers, aliases)


# ── Private helpers ──────────────────────────────────────────────────

def _alias_key(value: str) -> str:
    """Normalize a pipeline alias string for lookups (strip + lowercase)."""
    return str(value).strip().lower()


def _validate_schema_version(payload: Mapping[str, Any], *, path: Path) -> None:
    """Check that the schema version defined in the alias mapping is supported.

    Only schema version 2 is accepted; missing or empty versions are ignored.
    """
    version = payload.get("schema_version")
    if version in (None, ""):
        return
    if int(version) != 2:
        raise ValueError(f"pipeline alias file uses unsupported schema_version {version!r}: {path}")


def _alias_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """Retrieve all YAML alias mapping file paths found under the root."""
    path = Path(root) if root is not None else DEFAULT_PIPELINE_ALIASES_ROOT
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    return tuple(sorted(item for item in path.rglob("*.y*ml") if item.is_file()))


def _groups_from_alias_mapping(data: Mapping[str, Any], *, domain: str) -> tuple[PipelineAliasGroup, ...]:
    """Invert an alias-to-canonical dictionary and build a list of :class:`PipelineAliasGroup`.

    Given a flat ``{alias: canonical}`` mapping, groups aliases by their
    ``canonical_id`` and returns one :class:`PipelineAliasGroup` per unique
    canonical ID.
    """
    by_canonical: dict[str, list[str]] = {}
    for alias, canonical in data.items():
        canonical_id = str(canonical)
        by_canonical.setdefault(canonical_id, []).append(str(alias))
    return tuple(
        PipelineAliasGroup(canonical_id=canonical_id, aliases=tuple(aliases), domain=domain)
        for canonical_id, aliases in by_canonical.items()
    )


def _groups_from_payload(payload: Mapping[str, Any], *, domain: str) -> tuple[PipelineAliasGroup, ...]:
    """Parse and assemble :class:`PipelineAliasGroups` from various structured alias payloads.

    Supports three payload formats:
      1. Explicit ``groups`` list of mapping dicts.
      2. An ``aliases`` dict with ``{alias: canonical}`` entries.
      3. A flat ``{canonical: [aliases]}`` mapping.

    Raises:
        TypeError: If the payload does not match any recognised format.
    """
    # Format 1: explicit groups list.
    if isinstance(payload.get("groups"), list):
        return tuple(
            PipelineAliasGroup.from_mapping(item, domain=domain)
            for item in payload["groups"]
            if isinstance(item, Mapping)
        )

    # Format 2: aliases dict (no canonical keys at top level).
    aliases = payload.get("aliases")
    if isinstance(aliases, Mapping) and not any(key in payload for key in ("canonical", "canonical_id", "model_id")):
        return _groups_from_alias_mapping(aliases, domain=domain)

    # Format 3: single group with canonical_id at top level.
    if any(key in payload for key in ("canonical", "canonical_id", "model_id")):
        return (PipelineAliasGroup.from_mapping(payload, domain=domain),)

    # Format 4: flat {canonical: [aliases]} mapping.
    if all(isinstance(value, (list, tuple, str)) for value in payload.values()):
        groups: list[PipelineAliasGroup] = []
        for canonical_id, aliases_value in payload.items():
            aliases_tuple = (aliases_value,) if isinstance(aliases_value, str) else tuple(aliases_value)
            groups.append(
                PipelineAliasGroup(
                    canonical_id=str(canonical_id),
                    aliases=tuple(str(alias) for alias in aliases_tuple),
                    domain=domain,
                )
            )
        return tuple(groups)

    raise TypeError("pipeline alias file must define groups, aliases, or canonical_id entries.")


# ── YAML loading ────────────────────────────────────────────────────

def load_pipeline_alias_groups(root: str | Path | None = None) -> tuple[PipelineAliasGroup, ...]:
    """Load and parse all :class:`PipelineAliasGroup` records from the aliases root.

    Walks all ``*.yml`` / ``*.yaml`` files under ``root``, validates schema
    versions, and returns one group per file section.
    """
    groups: list[PipelineAliasGroup] = []
    for path in _alias_paths(root):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"failed to parse pipeline alias file {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise TypeError(f"pipeline alias file must contain a mapping: {path}")
        _validate_schema_version(payload, path=path)
        try:
            # NOTE: domain is inferred from the parent directory name.
            groups.extend(_groups_from_payload(payload, domain=path.parent.name))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{path}: {exc}") from exc
    return tuple(groups)


@lru_cache(maxsize=32)
def _cached_pipeline_alias_registry(root_text: str) -> PipelineAliasRegistry:
    """Load and cache the :class:`PipelineAliasRegistry` instance.

    The ``root_text`` string key ensures different root paths produce
    independent cached registries.
    """
    root = Path(root_text) if root_text else DEFAULT_PIPELINE_ALIASES_ROOT
    registry = PipelineAliasRegistry(load_pipeline_alias_groups(root))
    registry._frozen = True
    return registry


def load_pipeline_alias_registry(root: str | Path | None = None) -> PipelineAliasRegistry:
    """Retrieve the cached global :class:`PipelineAliasRegistry`.

    Repeated calls with the same ``root`` return the same frozen instance
    (``register`` raises).  Call :func:`clear_pipeline_alias_registry_cache`
    after editing alias YAML inside a long-lived process.
    """
    root_text = "" if root is None else str(Path(root))
    return _cached_pipeline_alias_registry(root_text)


def clear_pipeline_alias_registry_cache() -> None:
    """Clear the cached PipelineAliasRegistry instances."""
    _cached_pipeline_alias_registry.cache_clear()


__all__ = [
    "DEFAULT_PIPELINE_ALIASES_ROOT",
    "PipelineAliasGroup",
    "PipelineAliasRegistry",
    "RUNTIME_VIDEO_ALIASES",
    "build_alias_mapping",
    "build_runtime_video_mapping",
    "clear_pipeline_alias_registry_cache",
    "load_pipeline_alias_groups",
    "load_pipeline_alias_registry",
]

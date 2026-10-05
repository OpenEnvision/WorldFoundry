"""Data-backed pipeline binding helpers.

Provides the :class:`PipelineBinding` dataclass, the
:class:`PipelineBindingRegistry` index, and helpers for loading,
resolving, and merging YAML-backed pipeline bindings with plugin
contributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from worldfoundry.evaluation.utils import DATA_ROOT

LOGGER = logging.getLogger(__name__)

# ── Constants & type aliases ───────────────────────────────────────────

DEFAULT_PIPELINE_BINDINGS_ROOT = DATA_ROOT / "models" / "bindings" / "pipelines"
#: Default filesystem root for built-in YAML pipeline bindings.

PipelineRoute = tuple[str, str, str]
#: ``(pipeline_target, binding_id, source)`` triple returned by route resolution.


@dataclass(frozen=True)
class PipelineBinding:
    """Resolved pipeline binding entry for a model id.

    Attributes:
        binding_id: Unique identifier for this binding entry.
        model_id: HuggingFace-style model identifier this binding targets.
        runner: Runner class dotted path that orchestrates invocation.
        pipeline_target: ``module:Class`` dotted path of the pipeline implementation.
        loading_method: Strategy used to instantiate the pipeline (default ``"from_pretrained"``).
        loading_signature: Parameter signature expected by the loading method (default ``"unified"``).
        invocation_mode: How the pipeline should be invoked (default ``"native_or_standard_invocation"``).
        aliases: Short alias names that also resolve to this binding.
        schema_version: YAML schema version; only version ``2`` is supported.
    """

    binding_id: str
    model_id: str
    runner: str
    pipeline_target: str
    loading_method: str = "from_pretrained"
    loading_signature: str = "unified"
    invocation_mode: str = "native_or_standard_invocation"
    aliases: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PipelineBinding":
        """Instantiate a PipelineBinding from a mapping dictionary."""
        pipeline = data.get("pipeline") if isinstance(data.get("pipeline"), Mapping) else {}
        loading = pipeline.get("loading") if isinstance(pipeline.get("loading"), Mapping) else {}
        invocation = pipeline.get("invocation") if isinstance(pipeline.get("invocation"), Mapping) else {}
        aliases = data.get("aliases") or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        return cls(
            binding_id=str(data.get("binding_id") or data.get("model_id") or ""),
            model_id=str(data.get("model_id") or ""),
            runner=str(data.get("runner") or "worldfoundry.pipeline"),
            pipeline_target=str(pipeline.get("target") or data.get("pipeline_target") or ""),
            schema_version=_schema_version(data.get("schema_version")),
            loading_method=str(loading.get("method") or "from_pretrained"),
            loading_signature=str(loading.get("signature") or "unified"),
            invocation_mode=str(invocation.get("mode") or "native_or_standard_invocation"),
            aliases=tuple(str(alias) for alias in aliases if str(alias).strip()),
        )

    def validate(self) -> None:
        """Validate the fields and structure of the pipeline binding."""
        if self.schema_version is not None and self.schema_version != 2:
            raise ValueError(
                f"pipeline binding {self.binding_id!r} uses unsupported schema_version {self.schema_version!r}."
            )
        if not self.binding_id:
            raise ValueError("pipeline binding requires binding_id.")
        if not self.model_id:
            raise ValueError(f"pipeline binding {self.binding_id!r} requires model_id.")
        if ":" not in self.pipeline_target:
            raise ValueError(
                f"pipeline binding {self.binding_id!r} requires pipeline.target in 'module:Class' form."
            )


# ── Registry ───────────────────────────────────────────────────────────


class PipelineBindingRegistry:
    """Validated index of data-backed pipeline bindings."""

    def __init__(self, bindings: tuple[PipelineBinding, ...] = ()) -> None:
        """Initialize the registry with a collection of PipelineBindings."""
        self._frozen = False
        self._bindings: dict[str, PipelineBinding] = {}
        self._aliases: dict[str, PipelineBinding] = {}
        for binding in bindings:
            self.register(binding)

    def register(self, binding: PipelineBinding) -> None:
        """Register a single PipelineBinding and its aliases into the registry."""
        if self._frozen:
            raise RuntimeError(
                "cannot register on a cached PipelineBindingRegistry; "
                "construct a new instance or call clear_pipeline_binding_registry_cache()"
            )
        binding.validate()
        for key in (binding.binding_id, binding.model_id):
            normalized = _binding_key(key)
            existing = self._bindings.get(normalized)
            if existing is not None and existing != binding:
                raise ValueError(f"duplicate pipeline binding id/model_id: {key!r}")
            self._bindings[normalized] = binding
        for alias in binding.aliases:
            normalized = _binding_key(alias)
            existing = self._aliases.get(normalized) or self._bindings.get(normalized)
            if existing is not None and existing != binding:
                raise ValueError(f"duplicate pipeline binding alias: {alias!r}")
            self._aliases[normalized] = binding

    def list(self) -> tuple[PipelineBinding, ...]:
        """List all unique registered PipelineBindings, sorted by model and binding ID."""
        return tuple(sorted(set(self._bindings.values()), key=lambda item: (item.model_id, item.binding_id)))

    def get(self, key: str) -> PipelineBinding:
        """Retrieve a registered PipelineBinding by binding ID, model ID, or alias."""
        normalized = _binding_key(key)
        try:
            return self._bindings[normalized]
        except KeyError:
            try:
                return self._aliases[normalized]
            except KeyError as exc:
                raise KeyError(f"unknown pipeline binding: {key!r}") from exc


# ── Utility helpers ────────────────────────────────────────────────────


def _binding_key(value: str) -> str:
    """Normalize a pipeline binding key for index lookup."""
    return str(value).strip().lower()


def first_text(*values: Any) -> str:
    """Return the first non-empty string among the given arguments."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def runtime_profile_id(value: Any, *, default: str = "") -> str:
    """Extract the clean runtime profile ID from a string, stripping any prefix."""
    text = first_text(value)
    if not text:
        return default
    prefix = "runtime-profile:"
    if text.startswith(prefix):
        return text.removeprefix(prefix) or default
    return text


def runtime_profile_execution_metadata(value: Any) -> dict[str, Any]:
    """Load a runtime profile and return its execution metadata.

    An unknown profile id is tolerated (logged, empty metadata returned) so
    routing can fall through to the next priority. Genuine loading failures
    (corrupt YAML, broken environment) propagate so they stay visible instead
    of silently degrading the route.
    """
    profile_id = runtime_profile_id(value)
    if not profile_id:
        return {}
    try:
        from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile

        profile = load_runtime_profile(profile_id, check_conda_env_exists=False)
    except KeyError:
        LOGGER.warning(
            "Runtime profile %r not found; continuing without profile metadata.",
            profile_id,
        )
        return {}
    except Exception as exc:
        LOGGER.warning("Failed to load runtime profile %r: %s", profile_id, exc)
        raise
    metadata: dict[str, Any] = {
        "runtime_profile": value,
        "resolved_runtime_profile": profile_id,
    }
    for key in ("pipeline_target", "pipeline_binding", "environment", "assets"):
        if profile.execution.get(key):
            metadata[key] = profile.execution[key]
    return metadata


def _unique_texts(*values: Any) -> tuple[str, ...]:
    """Get unique non-empty stripped string arguments as a tuple."""
    items: list[str] = []
    for value in values:
        text = first_text(value)
        if text and text not in items:
            items.append(text)
    return tuple(items)


# ── YAML loading & caching ────────────────────────────────────────────


def _schema_version(value: Any) -> int | None:
    """Coerce any schema version value to int or return None."""
    if value in (None, ""):
        return None
    return int(value)


def _binding_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """Retrieve all YAML binding files from a directory recursively."""
    path = Path(root) if root is not None else DEFAULT_PIPELINE_BINDINGS_ROOT
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    return tuple(sorted(item for item in path.rglob("*.y*ml") if item.is_file()))


def load_pipeline_binding(path: str | Path) -> PipelineBinding:
    """Load and validate a PipelineBinding from a YAML file.

    Parse and validation errors are re-raised with the file path in the
    message so one broken file inside the bindings tree is locatable.
    """
    resolved = Path(path)
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"failed to parse pipeline binding {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"pipeline binding file must contain a mapping: {resolved}")
    try:
        binding = PipelineBinding.from_mapping(payload)
        binding.validate()
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{resolved}: {exc}") from exc
    return binding


def load_pipeline_bindings(root: str | Path | None = None) -> tuple[PipelineBinding, ...]:
    """Load all PipelineBinding entries found under the bindings root."""
    return tuple(load_pipeline_binding(path) for path in _binding_paths(root))


@lru_cache(maxsize=32)
def _cached_pipeline_binding_registry(root_text: str) -> PipelineBindingRegistry:
    """Cache and load the PipelineBindingRegistry from a root directory string."""
    root = Path(root_text) if root_text else DEFAULT_PIPELINE_BINDINGS_ROOT
    registry = PipelineBindingRegistry(load_pipeline_bindings(root))
    registry._frozen = True
    return registry


def load_pipeline_binding_registry(root: str | Path | None = None) -> PipelineBindingRegistry:
    """Load and cache the PipelineBindingRegistry from the bindings root.

    The returned registry is a process-wide cached instance and is frozen —
    ``register`` raises.  Use :func:`merge_pipeline_binding_plugins` for
    per-call overlays.  Call :func:`clear_pipeline_binding_registry_cache`
    after editing binding YAML inside a long-lived process.
    """
    root_text = "" if root is None else str(Path(root))
    return _cached_pipeline_binding_registry(root_text)


def clear_pipeline_binding_registry_cache() -> None:
    """Clear the cached PipelineBindingRegistry instances."""
    _cached_pipeline_binding_registry.cache_clear()


# ── Merge & resolution ────────────────────────────────────────────────


def merge_pipeline_binding_plugins(
    registry: PipelineBindingRegistry,
    plugin_bindings: Mapping[str, PipelineBinding],
) -> PipelineBindingRegistry:
    """Return a per-call registry with built-ins first and plugins second."""

    merged = PipelineBindingRegistry(registry.list())
    for slug, binding in sorted(plugin_bindings.items(), key=lambda item: item[0]):
        try:
            merged.register(binding)
        except ValueError as exc:
            LOGGER.warning(
                "Skipping plugin pipeline binding %r from slug %r: %s",
                binding.binding_id,
                slug,
                exc,
            )
            continue
    return merged


def load_pipeline_binding_registry_with_plugins(
    root: str | Path | None = None,
    *,
    include_plugins: bool = True,
) -> PipelineBindingRegistry:
    """Load the PipelineBindingRegistry with both built-ins and discovered plugins."""
    registry = load_pipeline_binding_registry(root)
    if not include_plugins:
        return registry
    from .discovery import discover_pipeline_bindings

    return merge_pipeline_binding_plugins(registry, discover_pipeline_bindings())


def _default_alias_root_for_binding_root(root: str | Path | None) -> Path | None:
    """Determine the default alias folder root relative to a bindings root."""
    if root is None:
        return None
    path = Path(root)
    if path.name == "pipelines":
        return path.parent / "aliases"
    if path.parent.name == "pipelines":
        return path.parent.parent / "aliases"
    return None


def resolve_pipeline_binding(
    key: str,
    *,
    root: str | Path | None = None,
    alias_root: str | Path | None = None,
    include_plugins: bool = True,
) -> PipelineBinding:
    """Resolve a PipelineBinding by model ID, registry key, or alias."""
    registry = load_pipeline_binding_registry_with_plugins(root, include_plugins=include_plugins)
    try:
        return registry.get(key)
    except KeyError:
        from .aliases import load_pipeline_alias_registry

        resolved_alias_root = Path(alias_root) if alias_root is not None else _default_alias_root_for_binding_root(root)
        aliases = load_pipeline_alias_registry(resolved_alias_root)
        canonical_id = aliases.canonical_id(key)
        return registry.get(canonical_id)


def resolve_pipeline_route(
    *,
    model_id: str,
    pipeline_target: Any = None,
    pipeline_binding: Any = None,
    runtime_profile: Any = None,
    profile_metadata: Mapping[str, Any] | None = None,
    binding_root: str | Path | None = None,
    alias_root: str | Path | None = None,
    include_model_id_binding: bool = True,
    include_plugins: bool = True,
) -> PipelineRoute | None:
    """Resolve a pipeline route from the binding catalog without importing model code.

    Resolution priority (first match wins):

    1. **Direct ``pipeline_target``** — if a ``module:Class`` target is supplied
       explicitly, use it immediately.
    2. **Runtime-profile metadata** — fall back to ``pipeline_target`` embedded
       in a runtime profile's execution metadata.
    3. **Binding catalog lookup** — iterate over candidate keys (``pipeline_binding``
       > runtime-profile ``pipeline_binding`` > ``model_id``) and resolve each
       through :func:`resolve_pipeline_binding`, including aliases.

    Args:
        model_id: HuggingFace-style model identifier.
        pipeline_target: Explicit ``module:Class`` pipeline target override.
        pipeline_binding: Binding ID or model ID override.
        runtime_profile: Runtime-profile string or identifier.
        profile_metadata: Pre-resolved execution metadata dict.
        binding_root: Root directory for YAML binding files.
        alias_root: Root directory for alias YAML files.
        include_model_id_binding: Whether to add ``model_id`` as a candidate binding key.
        include_plugins: Whether to include discovered plugin bindings.

    Returns:
        A :data:`PipelineRoute` ``(pipeline_target, binding_id, source)`` triple,
        or ``None`` if no route could be resolved.
    """

    metadata = dict(profile_metadata or {})
    if not metadata and runtime_profile:
        metadata = runtime_profile_execution_metadata(runtime_profile)

    direct_target = first_text(pipeline_target)
    if direct_target:
        return direct_target, first_text(pipeline_binding), "pipeline_target"

    profile_target = first_text(metadata.get("pipeline_target"))
    if profile_target:
        return (
            profile_target,
            first_text(metadata.get("pipeline_binding"), pipeline_binding),
            "runtime_profile.pipeline_target",
        )

    binding_keys = list(_unique_texts(pipeline_binding, metadata.get("pipeline_binding")))
    if include_model_id_binding:
        model_binding_id = first_text(model_id)
        if model_binding_id and model_binding_id not in binding_keys:
            binding_keys.append(model_binding_id)

    for binding_id in binding_keys:
        try:
            binding = resolve_pipeline_binding(
                binding_id,
                root=binding_root,
                alias_root=alias_root,
                include_plugins=include_plugins,
            )
        except KeyError:
            continue
        if first_text(pipeline_binding) == binding_id:
            source = "pipeline_binding"
        elif first_text(metadata.get("pipeline_binding")) == binding_id:
            source = "runtime_profile.pipeline_binding"
        elif first_text(model_id) == binding_id:
            source = "model_id.pipeline_binding"
        else:
            source = "pipeline_binding"
        return binding.pipeline_target, binding.binding_id, source

    return None


__all__ = [
    "DEFAULT_PIPELINE_BINDINGS_ROOT",
    "PipelineRoute",
    "PipelineBinding",
    "PipelineBindingRegistry",
    "first_text",
    "clear_pipeline_binding_registry_cache",
    "load_pipeline_binding",
    "load_pipeline_binding_registry",
    "load_pipeline_binding_registry_with_plugins",
    "load_pipeline_bindings",
    "merge_pipeline_binding_plugins",
    "resolve_pipeline_binding",
    "resolve_pipeline_route",
    "runtime_profile_execution_metadata",
    "runtime_profile_id",
]

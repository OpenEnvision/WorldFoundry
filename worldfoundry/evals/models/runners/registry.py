"""Model runner registry: register and resolve world-model runners.

Builtin runtime runners register eagerly; arbitrary ``module:Class`` targets
defer import until :meth:`ModelRunnerRegistry.create`.

Sections:

* **Entry dataclass** — one registered runner target.
* **ModelRunnerRegistry** — keyed lookup, lazy import, runner instantiation.
* **Module helpers** — default snapshot and listing utilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from worldfoundry.evals.api import WorldModelConfig
from worldfoundry.evals.models.import_target import import_dotted_attr
from worldfoundry.evals.models.runners.builtins import BUILTIN_RUNTIME_RUNNERS


def _runner_key(value: str) -> str:
    """Normalize a runner registry name or key string for case-insensitive lookup."""
    return value.strip().lower().replace("_", "-")


def _is_module_target(value: str) -> bool:
    """Check if the given string represents a valid 'module:Class' target format."""
    module_name, separator, attr_name = value.partition(":")
    if not (separator and module_name.strip() and attr_name.strip()):
        return False
    module_parts = module_name.split(".")
    attr_parts = attr_name.split(".")
    return all(part.isidentifier() for part in module_parts) and all(part.isidentifier() for part in attr_parts)


# ---------------------------------------------------------------------------
# Registry entry dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRunnerRegistryEntry:
    """One registered runnable model runner target."""

    name: str
    runner_target: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    source: str = "custom"
    origin: str = ""
    runner_class: type | None = None

    @classmethod
    def from_mapping(cls, entry: Mapping[str, Any]) -> ModelRunnerRegistryEntry:
        """Build entry from a mapping with at least ``name``."""
        return cls(
            name=str(entry["name"]),
            runner_target=str(entry.get("runner_target", entry["name"])),
            aliases=tuple(str(item) for item in entry.get("aliases", ())),
            description=str(entry.get("description", "")),
            source=str(entry.get("source", "custom")),
            origin=str(entry.get("origin", "")),
            runner_class=entry.get("runner_class"),
        )

    def keys(self) -> tuple[str, ...]:
        """Return deduplicated lookup keys (name + aliases)."""
        return tuple(dict.fromkeys((self.name, *self.aliases)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to a JSON-friendly dict."""
        runner_class = None
        if self.runner_class is not None:
            runner_class = f"{self.runner_class.__module__}:{self.runner_class.__qualname__}"
        return {
            "name": self.name,
            "runner_target": self.runner_target,
            "aliases": list(self.aliases),
            "description": self.description,
            "source": self.source,
            "origin": self.origin,
            "runner_class": runner_class,
        }


class ModelRunnerRegistry:
    """Keyed registry with lazy import and :class:`WorldModelConfig` factory."""

    def __init__(
        self,
        entries: Sequence[ModelRunnerRegistryEntry | Mapping[str, Any]] = (),
        *,
        include_builtins: bool = True,
    ) -> None:
        """Initialize registry with optional entries and built-in runners."""
        self._entries: dict[str, ModelRunnerRegistryEntry] = {}
        self._by_key: dict[str, ModelRunnerRegistryEntry] = {}
        if include_builtins:
            for entry in BUILTIN_RUNTIME_RUNNERS:
                self.register(
                    ModelRunnerRegistryEntry(
                        name=entry.name,
                        # Lazy builtins carry a module:Class target so the
                        # class is only imported at resolve time.
                        runner_target=entry.runner_target or entry.name,
                        aliases=entry.aliases,
                        description=entry.description,
                        source="builtin",
                        origin="worldfoundry.evals.models.runners.builtins",
                        runner_class=entry.runner_class,
                    )
                )
        for entry in entries:
            self.register(entry)

    def _drop_entry(self, entry: ModelRunnerRegistryEntry) -> None:
        """Unregister a single entry and all of its associated alias keys."""
        for key, existing in tuple(self._by_key.items()):
            if existing is entry:
                self._by_key.pop(key, None)
        self._entries.pop(_runner_key(entry.name), None)

    def register(
        self,
        entry: ModelRunnerRegistryEntry | Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> ModelRunnerRegistryEntry:
        """Register entry, optionally replacing conflicting keys."""
        if not isinstance(entry, ModelRunnerRegistryEntry):
            entry = ModelRunnerRegistryEntry.from_mapping(entry)

        normalized_keys = tuple(_runner_key(key) for key in entry.keys())  # noqa: SIM118 - Entry keys are not dictionary keys.
        if not replace:
            for key in normalized_keys:
                if key in self._by_key:
                    raise ValueError(f"runner registry key already exists: {key!r}")
        else:
            replaced = {existing for key in normalized_keys if (existing := self._by_key.get(key)) is not None}
            if existing := self._entries.get(_runner_key(entry.name)):
                replaced.add(existing)
            for existing in replaced:
                self._drop_entry(existing)
        for key in normalized_keys:
            self._by_key[key] = entry
        self._entries[_runner_key(entry.name)] = entry
        return entry

    def register_runner(
        self,
        name: str,
        *,
        runner_target: str | None = None,
        runner_class: type | None = None,
        aliases: Sequence[str] = (),
        description: str = "",
        source: str = "custom",
        origin: str = "",
        replace: bool = False,
    ) -> ModelRunnerRegistryEntry:
        """Register a runner by name with optional target/class metadata."""
        if runner_target is None:
            runner_target = name if runner_class is None else f"{runner_class.__module__}:{runner_class.__qualname__}"
        return self.register(
            ModelRunnerRegistryEntry(
                name=name,
                runner_target=runner_target,
                aliases=tuple(str(item) for item in aliases),
                description=description,
                source=source,
                origin=origin,
                runner_class=runner_class,
            ),
            replace=replace,
        )

    def list(self) -> tuple[ModelRunnerRegistryEntry, ...]:
        """List all unique registered runner entries sorted by name."""
        return tuple(self._entries[key] for key in sorted(self._entries))

    def get(self, key: str) -> ModelRunnerRegistryEntry:
        """Retrieve a registered runner entry by name or alias."""
        try:
            return self._by_key[_runner_key(key)]
        except KeyError as exc:
            raise KeyError(f"unknown model runner registry key: {key}") from exc

    def resolve_key(self, key: str) -> ModelRunnerRegistryEntry:
        """Resolve a key to its primary runner registry entry, falling back to module target."""
        try:
            return self.get(key)
        except KeyError:
            if not _is_module_target(key):
                raise
            return ModelRunnerRegistryEntry(
                name=key,
                runner_target=key,
                source="module_target",
                origin=key,
            )

    def resolve_runner_class(self, entry_or_key: ModelRunnerRegistryEntry | str) -> Any:
        """Import and return runner class for an entry or key."""
        entry = self.resolve_key(entry_or_key) if isinstance(entry_or_key, str) else entry_or_key
        if entry.runner_class is not None:
            return entry.runner_class
        return import_dotted_attr(entry.runner_target)

    def create(self, config: WorldModelConfig | Mapping[str, Any]) -> Any:
        """Instantiate a runner from :class:`WorldModelConfig`."""
        model_config = config if isinstance(config, WorldModelConfig) else WorldModelConfig.from_dict(config)
        runner_obj = self.resolve_runner_class(model_config.runner)
        factory = getattr(runner_obj, "from_config", None)
        return factory(model_config) if callable(factory) else runner_obj(model_config)


# ---------------------------------------------------------------------------
# Module-level registry helpers
# ---------------------------------------------------------------------------

_BUILTIN_MODEL_RUNNER_REGISTRY = ModelRunnerRegistry()


def default_model_runner_registry() -> ModelRunnerRegistry:
    """Return a fresh registry of the builtin runtime runners."""
    return model_runner_registry_snapshot()


def builtin_model_runner_registry() -> ModelRunnerRegistry:
    """Return the core built-in model runner registry."""
    return _BUILTIN_MODEL_RUNNER_REGISTRY


def model_runner_registry_snapshot() -> ModelRunnerRegistry:
    """Return a fresh, mutable copy of the builtin model runner registry."""
    return ModelRunnerRegistry(entries=_BUILTIN_MODEL_RUNNER_REGISTRY.list(), include_builtins=False)


def list_model_runner_registry_entries() -> tuple[ModelRunnerRegistryEntry, ...]:
    """Return all registered model runner registry entries."""
    return default_model_runner_registry().list()

"""Runtime profile, environment, and asset manifests for models."""

from typing import Any

from .assets import (
    DEFAULT_RUNTIME_ASSETS_ROOT,
    RuntimeAsset,
    RuntimeAssetProfile,
    load_runtime_asset_profile,
    load_runtime_asset_profile_by_id,
    load_runtime_asset_profiles,
    resolve_runtime_asset_profile,
)
from .environments import (
    DEFAULT_RUNTIME_ENVIRONMENTS_ROOT,
    RuntimeEnvironmentProfile,
    load_runtime_environment_profile,
    load_runtime_environment_profile_by_id,
    load_runtime_environment_profiles,
    resolve_runtime_environment_profile,
)
from .profiles import (
    DEFAULT_CATALOG_MANIFEST,
    DEFAULT_RUNTIME_PROFILES_ROOT,
    RuntimeProfile,
    load_runtime_profile,
    load_runtime_profile_manifest,
    load_runtime_profile_manifests,
    load_runtime_profiles,
)

__all__ = [
    "DEFAULT_CATALOG_MANIFEST",
    "DEFAULT_RUNTIME_ASSETS_ROOT",
    "DEFAULT_RUNTIME_ENVIRONMENTS_ROOT",
    "DEFAULT_RUNTIME_PROFILES_ROOT",
    "RuntimeAsset",
    "RuntimeAssetProfile",
    "RuntimeEnvironmentProfile",
    "RuntimeProfile",
    "RuntimeProfileSynthesis",
    "load_runtime_asset_profile",
    "load_runtime_asset_profile_by_id",
    "load_runtime_asset_profiles",
    "load_runtime_environment_profile",
    "load_runtime_environment_profile_by_id",
    "load_runtime_environment_profiles",
    "load_runtime_profile",
    "load_runtime_profile_manifest",
    "load_runtime_profile_manifests",
    "load_runtime_profiles",
    "resolve_runtime_asset_profile",
    "resolve_runtime_environment_profile",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the synthesis bridge (pulls torch via BaseSynthesis)."""
    if name == "RuntimeProfileSynthesis":
        from .profile_synthesis import RuntimeProfileSynthesis

        globals()[name] = RuntimeProfileSynthesis
        return RuntimeProfileSynthesis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

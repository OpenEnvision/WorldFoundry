"""Adapter for the optional, separately distributed native-kernel package.

No optional package or DSO is imported at module definition time.  Runtime
code should inspect availability during plan resolution and call ``load`` at a
prewarm boundary, never from a compiled region or CUDA Graph capture.

Not this module:
    In-tree Triton and PyTorch fallbacks live in :mod:`.diffusion` /
    :mod:`.moe`. This adapter never substitutes for those paths.

Public surface:

- :class:`NativeProviderStatus` — serializable inspect/load state.
- :class:`NativeProviderUnavailable` — raised only when ``strict=True``.
- :func:`native_provider_status` / :func:`load_native_provider` — inspect or
  load; non-strict failures return a status instead of raising.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping


class NativeProviderUnavailable(RuntimeError):
    """Raised when strict resolution requires an unavailable native provider."""


@dataclass(frozen=True, slots=True)
class NativeProviderStatus:
    """Serializable optional-provider state."""

    state: str
    installed: bool
    manifest_valid: bool
    runtime_compatible: bool | None
    loaded: bool
    reason: str | None = None
    build_id: str | None = None
    operator_abi_version: int | None = None
    capabilities: tuple[str, ...] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """True only after a successful ``load`` with a compatible ABI."""

        return (
            self.state == "loaded"
            and self.installed
            and self.manifest_valid
            and self.runtime_compatible is True
            and self.loaded
        )

    @property
    def inspectable(self) -> bool:
        """True when a sidecar manifest can be read without loading the DSO."""

        return self.installed and self.manifest_valid

    def to_dict(self) -> dict[str, Any]:
        """Flatten status for manifests; ``capabilities`` becomes a list."""

        return {
            "available": self.available,
            "inspectable": self.inspectable,
            "state": self.state,
            "installed": self.installed,
            "manifest_valid": self.manifest_valid,
            "runtime_compatible": self.runtime_compatible,
            "loaded": self.loaded,
            "reason": self.reason,
            "build_id": self.build_id,
            "operator_abi_version": self.operator_abi_version,
            "capabilities": list(self.capabilities),
            "manifest": dict(self.manifest),
        }


# ──────────────────────────────────────────────────────────────────────────
# Import / inspect — ModuleNotFoundError for this package is "absent", not fatal
# ──────────────────────────────────────────────────────────────────────────


def _import_native_package() -> Any | None:
    """Import ``worldfoundry_native_kernels`` or return ``None`` if it is not installed.

    Failure: a missing *different* module still raises so broken dependencies
    are not mistaken for an optional extra.
    """

    try:
        return importlib.import_module("worldfoundry_native_kernels")
    except ModuleNotFoundError as exc:
        if exc.name == "worldfoundry_native_kernels":
            return None
        raise


def _status_from_manifest(
    package: Any,
    manifest: Mapping[str, Any],
    *,
    compatible: bool | None,
    state: str,
) -> NativeProviderStatus:
    """Build a status from a mapping manifest; malformed capability lists become empty."""

    capabilities = manifest.get("capabilities")
    normalized_capabilities = (
        tuple(str(value) for value in capabilities)
        if isinstance(capabilities, (tuple, list))
        else ()
    )
    operator_abi = manifest.get("operator_abi_version")
    return NativeProviderStatus(
        state=state,
        installed=True,
        manifest_valid=True,
        runtime_compatible=compatible,
        loaded=bool(package.is_loaded()),
        build_id=str(manifest.get("build_id")) if manifest.get("build_id") is not None else None,
        operator_abi_version=int(operator_abi) if operator_abi is not None else None,
        capabilities=normalized_capabilities,
        manifest=dict(manifest),
        )


# ──────────────────────────────────────────────────────────────────────────
# Public inspect / load — never import torch through the optional package here
# ──────────────────────────────────────────────────────────────────────────


def native_provider_status(*, load: bool = False, strict: bool = False) -> NativeProviderStatus:
    """Inspect or explicitly load the optional provider.

    ``load=False`` performs only sidecar/file validation and therefore does not
    import PyTorch through the optional package.  ``load=True`` performs exact
    runtime ABI checks before the package calls ``torch.ops.load_library``.
    """

    package: Any | None = None
    try:
        package = _import_native_package()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if strict:
            raise NativeProviderUnavailable(
                f"native provider import failed: {type(exc).__name__}: {exc}"
            ) from exc
        return NativeProviderStatus(
            state="import_failed",
            installed=False,
            manifest_valid=False,
            runtime_compatible=None,
            loaded=False,
            reason=f"{type(exc).__name__}: {exc}",
        )

    if package is None:
        reason = "worldfoundry-native-kernels is not installed"
        if strict:
            raise NativeProviderUnavailable(reason)
        return NativeProviderStatus(
            state="absent",
            installed=False,
            manifest_valid=False,
            runtime_compatible=None,
            loaded=False,
            reason=reason,
        )

    try:
        manifest = package.inspect()
        if not isinstance(manifest, Mapping):
            raise NativeProviderUnavailable("native package returned a non-mapping manifest")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if strict:
            raise NativeProviderUnavailable(
                f"native provider manifest is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        return NativeProviderStatus(
            state="manifest_invalid",
            installed=True,
            manifest_valid=False,
            runtime_compatible=None,
            loaded=False,
            reason=f"{type(exc).__name__}: {exc}",
        )

    if not load:
        return _status_from_manifest(package, manifest, compatible=None, state="inspectable")

    try:
        loaded_manifest = package.load()
        if not isinstance(loaded_manifest, Mapping):
            raise NativeProviderUnavailable("native package returned a non-mapping loaded manifest")
        return _status_from_manifest(package, loaded_manifest, compatible=True, state="loaded")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if strict:
            raise NativeProviderUnavailable(
                f"native provider load failed: {type(exc).__name__}: {exc}"
            ) from exc
        compatibility_error = getattr(package, "NativeKernelCompatibilityError", ())
        incompatible = isinstance(compatibility_error, type) and isinstance(exc, compatibility_error)
        return NativeProviderStatus(
            state="runtime_incompatible" if incompatible else "load_failed",
            installed=True,
            manifest_valid=True,
            runtime_compatible=False,
            loaded=bool(package.is_loaded()),
            reason=f"{type(exc).__name__}: {exc}",
            build_id=str(manifest.get("build_id")) if manifest.get("build_id") is not None else None,
            operator_abi_version=(
                int(manifest["operator_abi_version"])
                if manifest.get("operator_abi_version") is not None
                else None
            ),
            capabilities=tuple(str(value) for value in manifest.get("capabilities", ())),
            manifest=dict(manifest),
        )


def load_native_provider(*, strict: bool = False) -> NativeProviderStatus:
    """Load the provider at an explicit prewarm boundary."""

    return native_provider_status(load=True, strict=strict)


__all__ = [
    "NativeProviderStatus",
    "NativeProviderUnavailable",
    "load_native_provider",
    "native_provider_status",
]

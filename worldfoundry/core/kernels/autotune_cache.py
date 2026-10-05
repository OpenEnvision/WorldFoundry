"""Persistent, hardware-scoped cache for eager kernel selections.

Serving processes are commonly short lived while Triton compilation and kernel
cross-over decisions are stable for one software/hardware/workload contract.
This module persists only the winner selected by the numerical autotuner.  The
cache identity includes the concrete GPU profile, tensor signature, runtime
compiler fingerprint, and the complete in-tree kernel source fingerprint, so a
decision measured on one architecture or kernel revision cannot leak to
another.

Not this module:
    First-use timing and numerical comparison live in
    :class:`worldfoundry.core.kernels.registry.KernelRegistry`. Triton
    ``@autotune`` config pickles live in :mod:`.triton_autotune`.

Public surface:

- :func:`persistent_autotune_cache_enabled` — env gate (XC-9 plus legacy name).
- :func:`load_persistent_selection` / :func:`store_persistent_selection` —
  exact-key JSON records; any miss or I/O error is a no-op.
- :func:`invalidate_persistent_selection` — drop one winner after a later
  optional-kernel failure.
- :func:`persistent_selection_cache_info` — manifest-safe metadata.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_CACHE_SCHEMA = "worldfoundry-kernel-autotune-v1"
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled", "none"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})


# ──────────────────────────────────────────────────────────────────────────
# Env / identity — unknown tokens keep the default so a typo cannot disable
# inference; cache keys hash schema + source + runtime + device + signature
# ──────────────────────────────────────────────────────────────────────────


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean env flag; unrecognized values keep *default*."""

    configured = os.getenv(name)
    if configured is None or not configured.strip():
        return default
    value = configured.strip().casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def persistent_autotune_cache_enabled() -> bool:
    """Return whether successful numerical autotune decisions may be reused.

    Prefer ``WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED`` (XC-9). The older
    ``WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE`` boolean name remains a fallback so
    existing jobs keep working.
    """

    configured = os.getenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED")
    if configured is not None and configured.strip():
        return _env_flag("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED", default=True)
    return _env_flag("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE", default=True)


def _stable_value(value: Any) -> Any:
    """Convert a workload signature into a deterministic JSON value."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_stable_value(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_stable_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item) for item in value]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, Mapping):
        items = ((str(key), _stable_value(item)) for key, item in value.items())
        return {"mapping": sorted(items, key=lambda item: item[0])}
    return {
        "type": f"{type(value).__module__}:{type(value).__qualname__}",
        "repr": repr(value),
    }


@lru_cache(maxsize=1)
def _kernel_source_fingerprint() -> str:
    """Fingerprint every in-tree Python kernel implementation and dispatcher."""

    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            # An unreadable source tree must never make the exact PyTorch path
            # unavailable.  The file metadata still keeps this key local to the
            # current installation instead of pretending the source is absent.
            try:
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
            except OSError:
                digest.update(b"unreadable")
    return digest.hexdigest()


def _device_profile(tensor: Any) -> dict[str, Any]:
    """Describe performance-relevant properties without using a device UUID."""

    import torch

    device = tensor.device
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"device_type": str(device.type)}
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    properties = torch.cuda.get_device_properties(index)
    try:
        capability: tuple[int, int] | str = tuple(
            int(item) for item in torch.cuda.get_device_capability(index)
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        capability = "unknown"
    return {
        "device_type": "rocm" if getattr(torch.version, "hip", None) else "cuda",
        "name": str(getattr(properties, "name", "unknown")),
        "compute_capability": capability,
        "multiprocessors": int(getattr(properties, "multi_processor_count", 0)),
        "shared_memory_per_multiprocessor": int(
            getattr(properties, "shared_memory_per_multiprocessor", 0)
        ),
        "total_memory": int(getattr(properties, "total_memory", 0)),
        "warp_size": int(getattr(properties, "warp_size", 0)),
    }


@lru_cache(maxsize=1)
def _runtime_profile() -> dict[str, Any]:
    """Describe runtime components that can change generated kernel code."""

    import torch

    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except (AttributeError, RuntimeError):
        cudnn_version = None
    return {
        "python_cache_tag": getattr(sys.implementation, "cache_tag", None),
        "torch": str(getattr(torch, "__version__", "unknown")),
        "triton": triton_version,
        "cuda": getattr(torch.version, "cuda", None),
        "hip": getattr(torch.version, "hip", None),
        "cudnn": cudnn_version,
    }


def _selection_identity(
    *,
    op: str,
    candidate: str,
    signature: tuple[object, ...],
    tensor: Any,
) -> tuple[str, dict[str, Any]]:
    """Return ``(sha256, payload)`` for one op/candidate/signature/device.

    Device UUIDs are excluded so cloned nodes with the same SKU share a hit.
    Source and runtime fingerprints keep a newer kernel tree from reusing a
    winner measured against older Triton or in-tree Python.
    """

    payload = {
        "schema": _CACHE_SCHEMA,
        "source": _kernel_source_fingerprint(),
        "runtime": _runtime_profile(),
        "op": str(op),
        "candidate": str(candidate),
        "device": _device_profile(tensor),
        "signature": _stable_value(signature),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _cache_directory() -> Path | None:
    """Resolve the on-disk root, or ``None`` when the cache must stay unused.

    Failure: any environment or filesystem error returns ``None`` so the eager
    dispatcher remains operational. The directory is further scoped by a
    truncated source fingerprint to isolate kernel-tree revisions.
    """

    if not persistent_autotune_cache_enabled():
        return None
    configured = os.getenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_DIR", "").strip()
    try:
        if configured:
            root = Path(configured).expanduser()
        else:
            from worldfoundry.core.execution.compile_cache import configure_persistent_compile_cache

            root = configure_persistent_compile_cache(namespace="kernel-autotune").root / "kernel-autotune"
        directory = root / _kernel_source_fingerprint()[:20]
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except Exception:
        # This cache is an optimization. Environment- or filesystem-specific
        # setup failures must leave the exact eager dispatcher operational.
        return None


# ──────────────────────────────────────────────────────────────────────────
# Public load / store — atomic replace; schema or winner mismatch is a miss
# ──────────────────────────────────────────────────────────────────────────


def persistent_selection_cache_info() -> dict[str, object]:
    """Return manifest-safe persistent selection cache metadata."""

    directory = _cache_directory()
    return {
        "enabled": persistent_autotune_cache_enabled(),
        "directory": None if directory is None else str(directory),
        "source_fingerprint": _kernel_source_fingerprint(),
        "refresh": _env_flag("WORLDFOUNDRY_KERNEL_AUTOTUNE_REFRESH", default=False),
    }


def load_persistent_selection(
    *,
    op: str,
    candidate: str,
    signature: tuple[object, ...],
    tensor: Any,
) -> dict[str, Any] | None:
    """Load one exact persistent selection, or return ``None`` on any miss."""

    if _env_flag("WORLDFOUNDRY_KERNEL_AUTOTUNE_REFRESH", default=False):
        return None
    directory = _cache_directory()
    if directory is None:
        return None
    key, _identity = _selection_identity(
        op=op,
        candidate=candidate,
        signature=signature,
        tensor=tensor,
    )
    path = directory / f"{key}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _CACHE_SCHEMA or payload.get("key") != key:
        return None
    winner = payload.get("winner")
    if winner not in {"torch", candidate}:
        return None
    record = payload.get("record")
    return {
        "winner": winner,
        "record": dict(record) if isinstance(record, dict) else {},
        "path": str(path),
    }


def store_persistent_selection(
    *,
    op: str,
    candidate: str,
    signature: tuple[object, ...],
    tensor: Any,
    winner: str,
    record: Mapping[str, object],
) -> None:
    """Atomically retain one validated winner without affecting inference."""

    if winner not in {"torch", candidate}:
        return
    directory = _cache_directory()
    if directory is None:
        return
    key, identity = _selection_identity(
        op=op,
        candidate=candidate,
        signature=signature,
        tensor=tensor,
    )
    destination = directory / f"{key}.json"
    payload = {
        "schema": _CACHE_SCHEMA,
        "key": key,
        "identity": identity,
        "winner": winner,
        "record": dict(record),
        "written_at_unix": time.time(),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{key}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            handle.flush()
        os.replace(temporary, destination)
        temporary = None
    except (OSError, TypeError, ValueError):
        return
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def invalidate_persistent_selection(
    *,
    op: str,
    candidate: str,
    signature: tuple[object, ...],
    tensor: Any,
) -> None:
    """Remove an exact cached selection after an optional-kernel failure."""

    directory = _cache_directory()
    if directory is None:
        return
    key, _identity = _selection_identity(
        op=op,
        candidate=candidate,
        signature=signature,
        tensor=tensor,
    )
    try:
        (directory / f"{key}.json").unlink(missing_ok=True)
    except OSError:
        pass


__all__ = [
    "invalidate_persistent_selection",
    "load_persistent_selection",
    "persistent_autotune_cache_enabled",
    "persistent_selection_cache_info",
    "store_persistent_selection",
]

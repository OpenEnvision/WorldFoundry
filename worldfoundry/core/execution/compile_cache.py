"""Persistent, in-tree torch.compile cache and wrapper reuse.

``torch.compile`` is expensive to start and easy to mis-cache: two
callers that request different Inductor options must not share the
first wrapper attached to a long-lived module, and generated code from
sm80 vs sm90 (or a different Triton/cuDNN) must not collide.

This module owns that contract:

- :func:`configure_persistent_compile_cache` points Inductor, Triton,
  CUDA driver JIT, and ``TORCH_EXTENSIONS_DIR`` at
  ``WORLDFOUNDRY_CACHE_DIR/compile/<fingerprint>/...`` (or
  ``WORLDFOUNDRY_COMPILE_CACHE_DIR``). The fingerprint includes torch /
  Triton / cuDNN / Python cache tag and the *set* of visible GPUs, not
  ``LOCAL_RANK``, so a heterogeneous multi-GPU process cannot namespace
  everything under rank 0's device. Explicit cache env vars win; a
  read-only directory falls back to ``$TMP/worldfoundry-compile``.
- :func:`compile_module_cached` / :func:`compile_callable_cached` keep
  one wrapper per ``(owner, CompilePolicy, options)`` on
  ``_worldfoundry_compiled_variants``. Failed compile falls back to
  eager unless ``strict=True``, and the fallback is warned once per
  ``(target, exception type)``.

Torch is imported lazily so control-plane and CPU-only processes can
configure paths without loading an accelerator runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import logging
import os
import re
import sys
import tempfile
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from worldfoundry.core.io.paths import cache_root_path

logger = logging.getLogger(__name__)

# One warning per (target, exception type): silent eager fallback makes
# performance regressions impossible to attribute, but the fallback path can
# re-run on every call for the same target.
_FALLBACK_WARNED: set[tuple[str, str]] = set()
_FALLBACK_WARNED_LOCK = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────
# Variant identity — one wrapper per (owner, policy, options)
# ──────────────────────────────────────────────────────────────────────────


def _warn_compile_fallback(kind: str, target: Any, exc: Exception) -> None:
    """Log once that *target* silently fell back to eager execution."""
    if kind == "module":
        name = type(target).__name__
    else:
        name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None) or repr(target)
    key = (f"{kind}:{name}", type(exc).__name__)
    with _FALLBACK_WARNED_LOCK:
        if key in _FALLBACK_WARNED:
            return
        _FALLBACK_WARNED.add(key)
    logger.warning(
        "torch.compile unavailable for %s %r; falling back to eager execution (%s: %s)",
        kind,
        name,
        type(exc).__name__,
        exc,
    )


@dataclass(frozen=True, slots=True)
class CompilePolicy:
    """Stable options that identify one compiled module variant."""

    backend: str = "inductor"
    mode: str | None = "default"
    fullgraph: bool = False
    dynamic: bool | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class CompileCacheLayout:
    """Resolved persistent compiler cache directories."""

    root: Path
    inductor: Path
    triton: Path
    fingerprint: str
    cuda: Path | None = None
    torch_extensions: Path | None = None


_CONFIGURE_LOCK = threading.Lock()
_COMPILE_LOCK = threading.RLock()
_DEFAULT_CACHE_LAYOUT: CompileCacheLayout | None = None
_DEFAULT_CACHE_CONTEXT: tuple[str, str | None, str | None, str | None] | None = None


def _freeze_compile_option(value: Any) -> object:
    """Return a hashable identity for nested ``torch.compile`` options."""

    if isinstance(value, Mapping):
        entries = ((str(key), _freeze_compile_option(item)) for key, item in value.items())
        return ("mapping", tuple(sorted(entries, key=lambda entry: entry[0])))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ("sequence", tuple(_freeze_compile_option(item) for item in value))
    if isinstance(value, (set, frozenset)):
        items = tuple(sorted((_freeze_compile_option(item) for item in value), key=repr))
        return ("set", items)
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return ("value", value)


def _compile_variant_key(
    policy: CompilePolicy,
    options: Mapping[str, Any] | None,
) -> tuple[object, ...]:
    """Hashable identity so two callers with different Inductor options never share a wrapper."""
    return (
        policy.backend,
        policy.mode,
        policy.fullgraph,
        policy.dynamic,
        _freeze_compile_option(options or {}),
    )


def _callable_variant_owner(function: Any) -> tuple[Any, object | None]:
    """Return a stable cache owner for functions, modules, and bound methods."""

    owner = getattr(function, "__self__", None)
    implementation = getattr(function, "__func__", None)
    if owner is not None and implementation is not None:
        return owner, (
            "bound_method",
            getattr(implementation, "__module__", ""),
            getattr(implementation, "__qualname__", repr(implementation)),
        )
    return function, None


def _compile_target(
    compile_fn: Any,
    target: Any,
    *,
    policy: CompilePolicy,
    options: Mapping[str, Any] | None,
) -> Any:
    """Invoke ``torch.compile`` with only the policy fields that override torch defaults."""
    kwargs: dict[str, Any] = {
        "backend": policy.backend,
    }
    if policy.mode is not None:
        kwargs["mode"] = policy.mode
    # Preserve torch.compile's own defaults when the policy does not request
    # an override.  Besides keeping the wrapper portable across torch
    # versions, this matters for third-party compile shims that accept only
    # the options explicitly selected by the caller.
    if policy.fullgraph:
        kwargs["fullgraph"] = True
    if policy.dynamic is not None:
        kwargs["dynamic"] = policy.dynamic
    if options:
        # Copy the mapping so caller mutation cannot change compiler behaviour
        # after the variant cache key has been constructed.
        kwargs["options"] = dict(options)
    return compile_fn(target, **kwargs)


def _safe_token(value: Any) -> str:
    """Filesystem-safe cache token; long values are truncated and hashed so paths stay portable."""
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "unknown")).strip("-.")
    if not token:
        return "unknown"
    if len(token) <= 96:
        return token
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return f"{token[:79]}-{digest}"


def _torch_fingerprint() -> str:
    """Return a cache namespace that separates incompatible generated code."""

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "torch-unavailable"

    version = _safe_token(getattr(torch, "__version__", "unknown"))
    try:
        triton_version = _safe_token(importlib.metadata.version("triton"))
    except importlib.metadata.PackageNotFoundError:
        triton_version = "unavailable"
    python_tag = _safe_token(getattr(sys.implementation, "cache_tag", "python-unknown"))
    runtime = getattr(torch, "version", None)
    hip = getattr(runtime, "hip", None)
    cuda_version = getattr(runtime, "cuda", None)
    try:
        cudnn_version = getattr(getattr(torch, "backends", None), "cudnn", None)
        cudnn_version = cudnn_version.version() if cudnn_version is not None else None
    except Exception:
        cudnn_version = None
    accelerator = "cpu"
    try:
        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if hip and torch.cuda.is_available():
            devices = set()
            for device_index in range(device_count):
                properties = torch.cuda.get_device_properties(device_index)
                devices.add(
                    f"{getattr(properties, 'gcnArchName', 'unknown')}-"
                    f"{getattr(properties, 'name', 'gpu')}-"
                    f"cus{getattr(properties, 'multi_processor_count', 'unknown')}-"
                    f"smem{getattr(properties, 'shared_memory_per_multiprocessor', 'unknown')}"
                )
            accelerator = f"rocm-{hip}-" + "_and_".join(sorted(devices))
        elif torch.cuda.is_available():
            # The cache directory is process-global. Fingerprint the unique
            # visible hardware set instead of LOCAL_RANK so a heterogeneous
            # multi-GPU process cannot accidentally namespace all generated
            # code under whichever device happened to share its rank index.
            devices = set()
            for device_index in range(device_count):
                major, minor = torch.cuda.get_device_capability(device_index)
                properties = torch.cuda.get_device_properties(device_index)
                devices.add(
                    f"sm{major}{minor}-{getattr(properties, 'name', 'gpu')}-"
                    f"sms{getattr(properties, 'multi_processor_count', 'unknown')}-"
                    f"smem{getattr(properties, 'shared_memory_per_multiprocessor', 'unknown')}"
                )
            accelerator = f"cuda-{cuda_version}-" + "_and_".join(sorted(devices))
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            accelerator = "xpu"
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None:
            if torch.backends.mps.is_available():
                accelerator = "mps"
    except Exception:
        accelerator = "accelerator-unknown"
    return _safe_token(
        f"torch-{version}-triton-{triton_version}-cudnn-{cudnn_version or 'unknown'}-{python_tag}-{accelerator}"
    )


def _ensure_cache_directory(directory: Path, *, fingerprint: str, kind: str) -> Path:
    """Create a compiler cache directory with a writable local fallback."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
        # ``mkdir(exist_ok=True)`` says nothing about an existing read-only
        # directory. Exercise the same create/delete permission compiler caches
        # require before committing the process-global environment to it.
        with tempfile.NamedTemporaryFile(prefix=".worldfoundry-write-test-", dir=directory):
            pass
        return directory
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "worldfoundry-compile" / fingerprint / kind
        fallback.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            f"Cannot use compiler cache directory {directory}: {exc}. Falling back to {fallback}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return fallback


# ──────────────────────────────────────────────────────────────────────────
# Public cache + compile — persist Inductor/Triton, reuse wrappers
# ──────────────────────────────────────────────────────────────────────────


def configure_persistent_compile_cache(
    *,
    namespace: str = "default",
    environ: MutableMapping[str, str] | None = None,
) -> CompileCacheLayout:
    """Configure persistent Inductor and Triton caches once per process.

    Explicit cache environment values are respected. Otherwise Inductor,
    Triton, CUDA driver JIT, and torch C++/CUDA extension caches live below
    ``WORLDFOUNDRY_CACHE_DIR``. Generated-code caches are partitioned by the
    torch/runtime/accelerator fingerprint; the CUDA driver cache is safe to
    share because its own key includes device and code identities.
    """

    global _DEFAULT_CACHE_CONTEXT, _DEFAULT_CACHE_LAYOUT

    del namespace
    env = os.environ if environ is None else environ
    configured_root = env.get("WORLDFOUNDRY_COMPILE_CACHE_DIR", "").strip()
    cache_context = (
        configured_root,
        env.get("CUDA_VISIBLE_DEVICES"),
        env.get("NVIDIA_VISIBLE_DEVICES"),
        env.get("CUDA_DEVICE_ORDER"),
    )
    cache_default_environment = env is os.environ

    with _CONFIGURE_LOCK:
        if (
            cache_default_environment
            and _DEFAULT_CACHE_LAYOUT is not None
            and _DEFAULT_CACHE_CONTEXT == cache_context
            and env.get("CUDA_CACHE_PATH") == str(_DEFAULT_CACHE_LAYOUT.cuda)
            and env.get("TORCHINDUCTOR_CACHE_DIR") == str(_DEFAULT_CACHE_LAYOUT.inductor)
            and env.get("TRITON_CACHE_DIR") == str(_DEFAULT_CACHE_LAYOUT.triton)
            and env.get("TORCH_EXTENSIONS_DIR") == str(_DEFAULT_CACHE_LAYOUT.torch_extensions)
        ):
            return _DEFAULT_CACHE_LAYOUT

        base = Path(configured_root).expanduser() if configured_root else cache_root_path(env) / "compile"
        # Set the driver cache before querying GPU properties in
        # _torch_fingerprint; some runtimes initialize CUDA while collecting
        # those properties. This ordering applies to the process environment;
        # custom mappings remain side-effect-free test/configuration inputs.
        cuda = Path(env.get("CUDA_CACHE_PATH") or base / "cuda").expanduser()
        cuda = _ensure_cache_directory(cuda, fingerprint="cuda-driver", kind="cuda")
        env["CUDA_CACHE_PATH"] = str(cuda)
        env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))

        fingerprint = _torch_fingerprint()
        root = base / fingerprint
        inductor = Path(env.get("TORCHINDUCTOR_CACHE_DIR") or root / "inductor").expanduser()
        triton = Path(env.get("TRITON_CACHE_DIR") or root / "triton").expanduser()
        torch_extensions = Path(env.get("TORCH_EXTENSIONS_DIR") or root / "torch_extensions").expanduser()
        inductor = _ensure_cache_directory(inductor, fingerprint=fingerprint, kind="inductor")
        triton = _ensure_cache_directory(triton, fingerprint=fingerprint, kind="triton")
        torch_extensions = _ensure_cache_directory(
            torch_extensions,
            fingerprint=fingerprint,
            kind="torch_extensions",
        )
        # Assignment is intentional: when an explicitly configured path is not
        # writable, _ensure_cache_directory has selected a valid fallback.
        env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
        env["TRITON_CACHE_DIR"] = str(triton)
        env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions)
        env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        env.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")
        env.setdefault("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE", "0")

        layout = CompileCacheLayout(
            root=root,
            inductor=inductor,
            triton=triton,
            fingerprint=fingerprint,
            cuda=cuda,
            torch_extensions=torch_extensions,
        )
        if cache_default_environment:
            _DEFAULT_CACHE_LAYOUT = layout
            _DEFAULT_CACHE_CONTEXT = cache_context
        return layout


def compile_module_cached(
    module: Any,
    *,
    policy: CompilePolicy | None = None,
    namespace: str = "modules",
    options: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> Any:
    """Return one cached ``torch.compile`` wrapper per module and policy.

    Compiler options participate in the wrapper identity.  This prevents two
    callers that request different Inductor settings from accidentally sharing
    the first wrapper attached to a long-lived model instance.
    """

    selected = CompilePolicy() if policy is None else policy
    if not selected.enabled:
        return module
    if hasattr(module, "_orig_mod"):
        return module

    try:
        configure_persistent_compile_cache(namespace=namespace)
        torch = importlib.import_module("torch")
    except Exception as exc:
        if strict:
            raise
        _warn_compile_fallback("module", module, exc)
        return module
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return module

    key = _compile_variant_key(selected, options)
    with _COMPILE_LOCK:
        variants = getattr(module, "_worldfoundry_compiled_variants", None)
        if not isinstance(variants, dict):
            variants = {}
            try:
                setattr(module, "_worldfoundry_compiled_variants", variants)
            except Exception:
                variants = {}
        cached = variants.get(key)
        if cached is not None:
            return cached
        try:
            compiled = _compile_target(
                compile_fn,
                module,
                policy=selected,
                options=options,
            )
        except Exception as exc:
            if strict:
                raise
            _warn_compile_fallback("module", module, exc)
            return module
        variants[key] = compiled
        return compiled


def compile_callable_cached(
    function: Any,
    *,
    policy: CompilePolicy | None = None,
    namespace: str = "functions",
    options: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> Any:
    """Compile a callable once without selecting compilation implicitly."""

    selected = CompilePolicy() if policy is None else policy
    if not selected.enabled:
        return function
    try:
        configure_persistent_compile_cache(namespace=namespace)
        torch = importlib.import_module("torch")
    except Exception as exc:
        if strict:
            raise
        _warn_compile_fallback("callable", function, exc)
        return function
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return function

    cache_owner, callable_identity = _callable_variant_owner(function)
    key = (callable_identity, *_compile_variant_key(selected, options))
    with _COMPILE_LOCK:
        variants = getattr(cache_owner, "_worldfoundry_compiled_variants", None)
        if not isinstance(variants, dict):
            variants = {}
            try:
                setattr(cache_owner, "_worldfoundry_compiled_variants", variants)
            except Exception:
                # Some extension-backed callables and immutable owners cannot
                # carry attributes. Compilation remains usable for this call,
                # but those objects cannot retain a reusable wrapper.
                variants = {}
        cached = variants.get(key)
        if cached is not None:
            return cached
        try:
            compiled = _compile_target(
                compile_fn,
                function,
                policy=selected,
                options=options,
            )
        except Exception as exc:
            if strict:
                raise
            _warn_compile_fallback("callable", function, exc)
            return function
        variants[key] = compiled
        return compiled


__all__ = [
    "CompileCacheLayout",
    "CompilePolicy",
    "compile_callable_cached",
    "compile_module_cached",
    "configure_persistent_compile_cache",
]

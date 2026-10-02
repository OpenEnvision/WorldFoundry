"""Attention backend capability probes shared by inference dispatchers.

This module is responsible for safely, lightly, and side-effect-freely probing and
dispatching the optimal Attention computation backend in the current runtime environment.

Core Architectural Design:
1. Package Probing: Uses `importlib.machinery.PathFinder` to locate packages
   rather than physically importing them. Since many acceleration packages (e.g., `flash_attn`)
   might crash hard during import/initialization (e.g., due to missing shared libraries or
   mismatched CUDA runtimes), this probing mechanism ensures a graceful fallback to PyTorch
   native SDPA without crashing the process.
2. Two-Stage Validation: Separates physical availability (`available`) from runtime hardware
   compatibility (`usable`). For example, even if `flash_attn` is successfully installed, it
   will be marked unusable if the active GPU Compute Capability is < 8.0 (e.g., V100, T4).
3. Explicit Opt-In Providers: ``auto`` deliberately resolves to the in-tree PyTorch SDPA
   provider only (the "no-external-repo contract"); external packages such as flash-attn,
   SageAttention, and xFormers are used only when explicitly requested. When an explicitly
   requested dense provider is unusable, resolution degrades to PyTorch SDPA with a warning
   naming the request and reason.
4. Model-Specific Sparse Providers: VSA, FlexBlock, V-MoBA, SLA, and SageSLA need metadata
   that a generic Q/K/V call cannot reconstruct (for example a 3D grid, block mask, layer
   routing, or checkpoint-trained projection). They remain recognizable for capability
   reporting, but selecting one through the generic dispatcher fails fast instead of silently
   executing dense SDPA and falsely reporting the sparse provider as effective.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.machinery import PathFinder
from typing import Mapping

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Capability record — installed vs usable are separate bits on purpose
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttentionKernelCapability:
    """Runtime availability metadata for a single attention kernel family.

    Attributes:
        name: Canonical identifier of the attention backend.
        package: The underlying Python package or C++/CUDA extension.
        available: True if the module/package is physically installed in the environment.
        usable: True if the backend is physically runnable on the active hardware (e.g., CUDA capability).
        reason: Explains why the backend is unusable or unavailable, if applicable.
    """

    name: str
    package: str
    available: bool
    usable: bool
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Name aliases — colloquial strings collapse here; no package is imported
# ──────────────────────────────────────────────────────────────────────────

_AUTO = "auto"
_TORCH = "torch"
_FLASH_AUTO = "flash_attention"
_BACKEND_ALIASES: Mapping[str, str] = {
    "auto": _AUTO,
    "default": _AUTO,
    "torch": _TORCH,
    "torch_sdpa": _TORCH,
    "sdpa": _TORCH,
    "math": _TORCH,
    "efficient": _TORCH,
    "cudnn": _TORCH,
    "flash": _FLASH_AUTO,
    "flash_attn": _FLASH_AUTO,
    "flash_attention": _FLASH_AUTO,
    "flash_attention_2": "flash_attention_2",
    "flash2": "flash_attention_2",
    "flash_attn_2": "flash_attention_2",
    "flash_attn2": "flash_attention_2",
    "flash_attention_3": "flash_attention_3",
    "flash3": "flash_attention_3",
    "flash_attn_3": "flash_attention_3",
    "flash_attn3": "flash_attention_3",
    "flash_attention_4": "flash_attention_4",
    "flash4": "flash_attention_4",
    "flash_attn_4": "flash_attention_4",
    "flash_attn4": "flash_attention_4",
    "sage": "sage_attention",
    "sage_attn": "sage_attention",
    "sage_attention": "sage_attention",
    "sageattn": "sage_attention",
    "sage_attn_three": "sage_attention_3",
    "sage_attention_3": "sage_attention_3",
    "sage3": "sage_attention_3",
    "xformers": "xformers",
    "xformers_attention": "xformers",
    "video_sparse_attn": "video_sparse_attention",
    "video_sparse_attention": "video_sparse_attention",
    "flex_block_attn": "flex_block_attention",
    "flex_block_attention": "flex_block_attention",
    "vmoba": "vmoba_attention",
    "vmoba_attn": "vmoba_attention",
    "vmoba_attention": "vmoba_attention",
    "sla": "sla_attention",
    "sla_attn": "sla_attention",
    "sla_attention": "sla_attention",
    "sage_sla": "sage_sla_attention",
    "sage_sla_attn": "sage_sla_attention",
    "sage_sla_attention": "sage_sla_attention",
}
_DEFAULT_PRIORITY = (_TORCH,)
_REPORT_PRIORITY = (
    "flash_attention_4",
    "flash_attention_3",
    "flash_attention_2",
    "sage_attention",
    "xformers",
    _TORCH,
)
_EXPLICIT_PRIORITY = ("sage_attention_3",)
_EXPERIMENTAL_PRIORITY = (
    "flex_block_attention",
    "video_sparse_attention",
    "vmoba_attention",
    "sla_attention",
    "sage_sla_attention",
)
_VIDEO_SPARSE_KERNEL_IMPORT = "fast" + "video_kernel"

# These names describe model-specific attention *systems*, not drop-in Q/K/V
# kernels. Keeping the missing inputs explicit prevents a package-presence
# probe from being mistaken for an executable generic backend.
_MODEL_SPECIFIC_BACKEND_REQUIREMENTS: Mapping[str, str] = {
    "video_sparse_attention": (
        "a VideoSparseAttention metadata builder (3D token grid, tiled indices, "
        "variable block sizes, timestep/sparsity schedule, and compatible VSA/QAT weights)"
    ),
    "flex_block_attention": (
        "an explicit block mask plus query/key block sizes prepared by the model-specific "
        "sparse-attention planner"
    ),
    "vmoba_attention": (
        "per-layer V-MoBA routing metadata (layer index, patch resolution, chunk policy, "
        "top-k, and threshold configuration)"
    ),
    "sla_attention": (
        "SLA metadata and checkpoint-trained sparse/linear blending projections"
    ),
    "sage_sla_attention": (
        "SLA metadata, checkpoint-trained blending projections, and the spas_sage_attn "
        "quantized sparse kernel stack"
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Model-specific contract — fail fast; never degrade these to dense SDPA
# ──────────────────────────────────────────────────────────────────────────


class ModelSpecificAttentionBackendError(ValueError):
    """A model-specific sparse system was selected as generic Q/K/V attention."""


def require_generic_attention_backend(value: str | None) -> str:
    """Return a normalized generic provider or fail with its missing model contract.

    Package installation alone is insufficient for the sparse systems listed
    above. Calling them without their metadata would either be incorrect or
    force a dense fallback, so callers must install a model-specific
    processor/graph instead.
    """

    canonical = normalize_attention_backend(value)
    requirement = _MODEL_SPECIFIC_BACKEND_REQUIREMENTS.get(canonical)
    if requirement is not None:
        raise ModelSpecificAttentionBackendError(
            f"Attention backend {canonical!r} is model-specific and is not executable "
            f"by the generic Q/K/V dispatcher; it requires {requirement}. Install a "
            "checkpoint-compatible model processor that constructs this metadata."
        )
    return canonical


def attention_backend_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Read and resolve the canonical attention backend requested by the environment.

    Inspects `WORLDFOUNDRY_ATTENTION_IMPLEMENTATION` or `WORLDFOUNDRY_ATTENTION_BACKEND`,
    falling back to `"auto"` if neither is specified.
    """
    env = os.environ if environ is None else environ
    value = env.get("WORLDFOUNDRY_ATTENTION_IMPLEMENTATION") or env.get("WORLDFOUNDRY_ATTENTION_BACKEND") or _AUTO
    return normalize_attention_backend(value)


def normalize_attention_backend(value: str | None) -> str:
    """Normalize colloquial or varied attention backend names into standard keys.

    For example, maps 'flash-attn-2', 'flash2', or 'flash_attention_2' to 'flash_attention_2'
    while stripping and lowering input values to handle typos gracefully.
    """
    if value is None:
        return _AUTO
    key = str(value).strip().lower().replace("-", "_")
    if not key:
        return _AUTO
    canonical = _BACKEND_ALIASES.get(key)
    if canonical is None:
        allowed = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"Unknown attention backend {value!r}. Expected one of: {allowed}")
    return canonical


# ──────────────────────────────────────────────────────────────────────────
# Probes — find_spec only; results are cached per (capability, HIP, accel)
# ──────────────────────────────────────────────────────────────────────────


def probe_attention_backends(device: torch.device | str | int | None = None) -> dict[str, AttentionKernelCapability]:
    """Probe installed attention packages and hardware compatibility.

    Results are cached by runtime and compute capability, rather than globally.
    This keeps mixed A100/H100 nodes correct when the active tensor/device
    changes after module import.
    """

    capability = _cuda_compute_capability(device)
    hip = bool(getattr(torch.version, "hip", None))
    accelerator = _torch_cuda_accelerator_available(device)
    return _probe_attention_backends_cached(capability, hip, accelerator)


@lru_cache(maxsize=16)
def _probe_attention_backends_cached(
    capability: tuple[int, int] | None,
    hip: bool,
    accelerator: bool,
) -> dict[str, AttentionKernelCapability]:
    """Build the capability table for one (capability, HIP, accelerator) triple.

    Cached so mixed A100/H100 nodes stay correct when the active device
    changes after import. Sparse systems stay ``usable=False`` even when
    the package is present — package presence is not a generic Q/K/V
    contract.
    """
    nvidia_cuda = capability is not None and not hip
    flash_gpu = capability is not None and capability[0] in {8, 9}
    flash3_gpu = capability == (9, 0)
    # CuTeDSL FA4 forward kernels cover Ampere and newer NVIDIA targets. It is
    # intentionally explicit-only because first-use JIT and shape support vary
    # across pinned builds.
    flash4_gpu = capability is not None and capability[0] >= 8
    sage_gpu = capability is not None and capability[0] in {8, 9}
    # Keep this aligned with the architectures compiled by SageAttention3's
    # upstream Blackwell extension.  In particular, its current build does not
    # target SM103, so B300/GB300 must retain exact cuDNN/SDPA rather than being
    # accepted merely because they share the Blackwell generation name.
    sage3_gpu = capability in {(10, 0), (12, 0), (12, 1)}
    return {
        "flash_attention_4": _package_capability(
            name="flash_attention_4",
            package="flash_attn.cute",
            usable_if=flash4_gpu,
            unavailable_reason="flash-attn-4 (flash_attn.cute) is not installed",
            unusable_reason="FlashAttention 4 requires NVIDIA Ampere or newer",
        ),
        "flash_attention_3": _package_capability(
            name="flash_attention_3",
            package="flash_attn_interface",
            usable_if=flash3_gpu,
            unavailable_reason="flash_attn_interface is not installed",
            unusable_reason="FlashAttention 3 requires NVIDIA Hopper (SM90)",
        ),
        "flash_attention_2": _package_capability(
            name="flash_attention_2",
            package="flash_attn",
            usable_if=flash_gpu,
            unavailable_reason="flash_attn is not installed",
            unusable_reason="FlashAttention 2 requires a supported NVIDIA Ampere, Ada, or Hopper GPU",
        ),
        "sage_attention": _package_capability(
            name="sage_attention",
            package="sageattention",
            usable_if=sage_gpu,
            unavailable_reason="sageattention is not installed",
            unusable_reason="SageAttention requires a supported NVIDIA Ampere, Ada, or Hopper GPU",
        ),
        "sage_attention_3": _package_capability(
            name="sage_attention_3",
            package="sageattn3",
            usable_if=sage3_gpu,
            unavailable_reason="the in-tree sageattn3 extension is not built",
            unusable_reason="SageAttention 3 requires an explicitly supported NVIDIA Blackwell target",
        ),
        "xformers": _package_capability(
            name="xformers",
            package="xformers.ops",
            usable_if=accelerator,
            unavailable_reason="xformers is not installed",
            unusable_reason="xFormers attention requires a CUDA or ROCm accelerator build",
        ),
        "video_sparse_attention": _package_capability(
            name="video_sparse_attention",
            package="fastvideo_kernel",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            # Package presence proves only that kernels exist. The generic
            # dispatcher has no VSA metadata/weight graph, so it must never
            # advertise this model-specific system as executable.
            usable_if=False,
            unavailable_reason="video sparse attention kernel package is not installed",
            unusable_reason=(
                "Video sparse attention is package-available but requires a model-specific "
                "VSA metadata/QAT graph; it is not a generic Q/K/V backend"
                if nvidia_cuda
                else "Video sparse attention requires CUDA and a model-specific VSA metadata/QAT graph"
            ),
        ),
        "flex_block_attention": _package_capability(
            name="flex_block_attention",
            package="flex_block_attn",
            usable_if=False,
            unavailable_reason="flex_block_attn is not installed",
            unusable_reason=(
                "FlexBlockAttention requires a model-provided block mask and block geometry; "
                "it is not a generic Q/K/V backend"
                if nvidia_cuda
                else "FlexBlockAttention requires CUDA plus a model-provided block mask and block geometry"
            ),
        ),
        "vmoba_attention": _package_capability(
            name="vmoba_attention",
            package="fastvideo_kernel",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=False,
            unavailable_reason="V-MoBA attention kernel package is not installed",
            unusable_reason=(
                "V-MoBA requires model/layer routing metadata; it is not a generic Q/K/V backend"
                if nvidia_cuda
                else "V-MoBA requires CUDA and model/layer routing metadata"
            ),
        ),
        "sla_attention": _package_capability(
            name="sla_attention",
            package="fastvideo_kernel",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=False,
            unavailable_reason="sparse linear attention kernel package is not installed",
            unusable_reason=(
                "SLA requires model metadata and checkpoint-trained blending projections; "
                "it is not a generic Q/K/V backend"
                if nvidia_cuda
                else "SLA requires CUDA, model metadata, and checkpoint-trained blending projections"
            ),
        ),
        "sage_sla_attention": _package_capability(
            name="sage_sla_attention",
            package="spas_sage_attn",
            import_name="spas_sage_attn",
            usable_if=False,
            unavailable_reason="SageSLA attention kernel package is not installed",
            unusable_reason=(
                "SageSLA requires model metadata and checkpoint-trained blending projections; "
                "it is not a generic Q/K/V backend"
                if nvidia_cuda
                else "SageSLA requires CUDA, model metadata, and checkpoint-trained blending projections"
            ),
        ),
        _TORCH: AttentionKernelCapability(name=_TORCH, package="torch", available=True, usable=True),
    }


# Preserve the cache invalidation hook exposed by the previous cached public
# function; tests and long-running plugin hosts use it after environment changes.
setattr(probe_attention_backends, "cache_clear", _probe_attention_backends_cached.cache_clear)


_EXPLICIT_FALLBACK_WARNED: set[tuple[str, str, str]] = set()
_AUTO_FASTER_BACKENDS_LOGGED: set[tuple[str, ...]] = set()


def _warn_explicit_backend_fallback(requested: str, resolved: str, reason: str) -> None:
    """Log once per (requested, resolved, reason) when an explicit request degrades."""

    key = (requested, resolved, reason)
    if key in _EXPLICIT_FALLBACK_WARNED:
        return
    _EXPLICIT_FALLBACK_WARNED.add(key)
    logger.warning(
        "Attention backend %r was explicitly requested but is not usable; using %r instead. Reason: %s",
        requested,
        resolved,
        reason or "unknown",
    )


def _log_faster_backends_available(capabilities: Mapping[str, AttentionKernelCapability]) -> None:
    """Log once when auto mode keeps PyTorch SDPA while faster providers are usable."""

    usable_external = tuple(
        name for name in _REPORT_PRIORITY if name != _TORCH and name in capabilities and capabilities[name].usable
    )
    if not usable_external or usable_external in _AUTO_FASTER_BACKENDS_LOGGED:
        return
    _AUTO_FASTER_BACKENDS_LOGGED.add(usable_external)
    logger.info(
        "Attention backend 'auto' resolves to the in-tree PyTorch SDPA provider by design "
        "(no-external-repo contract). Usable external backends detected but not enabled: %s. "
        "Set WORLDFOUNDRY_ATTENTION_BACKEND=<name> to opt in explicitly.",
        ", ".join(usable_external),
    )


# ──────────────────────────────────────────────────────────────────────────
# Resolution — auto stays in-tree; explicit requests may degrade to SDPA
# ──────────────────────────────────────────────────────────────────────────


def resolve_attention_backend(
    preferred: str | None = None,
    device: torch.device | str | int | None = None,
    *,
    allow_model_specific: bool = False,
) -> str:
    """Resolve the requested backend, falling back to PyTorch SDPA when unusable.

    ``"auto"`` deliberately resolves to the in-tree PyTorch SDPA provider only:
    external FlashAttention/SageAttention/xFormers packages are explicit
    opt-ins (the no-external-repo contract). When usable external providers
    exist under ``auto``, an informational log points them out once.
    If an explicitly requested package is not usable on the active hardware,
    resolution degrades to ``"torch"`` SDPA and logs a warning with the
    requested backend, the effective backend, and the capability reason.

    Model-specific sparse systems fail by default. A caller that already owns
    their full metadata/checkpoint graph may pass ``allow_model_specific=True``;
    package and NVIDIA-device availability are still required, and there is no
    dense fallback for that explicit request.
    """
    requested = attention_backend_from_env() if preferred is None else normalize_attention_backend(preferred)
    if requested in _MODEL_SPECIFIC_BACKEND_REQUIREMENTS:
        if not allow_model_specific:
            # Even an installed sparse kernel is not callable without its
            # model/checkpoint metadata. Degrading that explicit request to
            # dense SDPA would make the audit record dishonest.
            require_generic_attention_backend(requested)
        capability = probe_attention_backends(device)[requested]
        if not capability.available:
            raise ModelSpecificAttentionBackendError(
                f"Attention backend {requested!r} was selected by a model-specific "
                f"attention graph, but its package {capability.package!r} is unavailable: "
                f"{capability.reason or 'package probe failed'}"
            )
        if _cuda_compute_capability(device) is None:
            raise ModelSpecificAttentionBackendError(
                f"Attention backend {requested!r} was selected by a model-specific "
                "attention graph, but it requires an NVIDIA CUDA device"
            )
        return requested
    capabilities = probe_attention_backends(device)
    if requested == _AUTO:
        for name in _DEFAULT_PRIORITY:
            if capabilities[name].usable:
                resolved = name
                break
        else:
            resolved = _TORCH
        if resolved == _TORCH:
            _log_faster_backends_available(capabilities)
        return resolved
    if requested == _FLASH_AUTO:
        for name in ("flash_attention_3", "flash_attention_2"):
            if capabilities[name].usable:
                return name
        reasons = "; ".join(
            f"{name}: {capabilities[name].reason or 'not usable'}"
            for name in ("flash_attention_3", "flash_attention_2")
        )
        _warn_explicit_backend_fallback(_FLASH_AUTO, _TORCH, reasons)
        return _TORCH
    capability = capabilities.get(requested)
    if capability is not None and capability.usable:
        return requested
    reason = capability.reason if capability is not None else "backend is not registered in the capability table"
    _warn_explicit_backend_fallback(requested, _TORCH, reason)
    return _TORCH


def resolve_transformers_attention_implementation(
    preferred: str | None = None,
    device: torch.device | str | int | None = None,
) -> str:
    """Resolve a backend name accepted by Transformers model configs.

    Transformers currently exposes portable ``eager``/``sdpa`` paths and the
    separately installed ``flash_attention_2`` provider.  WorldFoundry has a
    wider backend vocabulary (including FA3 and model-specific kernels), so a
    small adapter is needed before writing ``config._attn_implementation``.
    Unsupported or unavailable providers conservatively map to PyTorch SDPA.
    """

    requested = attention_backend_from_env() if preferred is None else str(preferred)
    normalized = requested.strip().lower().replace("-", "_")
    if normalized == "eager":
        return "eager"
    resolved = resolve_attention_backend(requested, device)
    return "flash_attention_2" if resolved == "flash_attention_2" else "sdpa"


def attention_backend_capability(
    name: str,
    device: torch.device | str | int | None = None,
) -> AttentionKernelCapability:
    """Retrieve runtime capability metadata for a single normalized backend."""
    canonical = normalize_attention_backend(name)
    if canonical in {_AUTO, _FLASH_AUTO}:
        canonical = resolve_attention_backend(canonical, device)
    return probe_attention_backends(device)[canonical]


def attention_backend_report() -> tuple[AttentionKernelCapability, ...]:
    """Retrieve capability status of all registered backends for display purposes.

    The ordering is a fixed display order (fastest candidates first), *not*
    the automatic dispatch chain: automatic mode uses the in-tree PyTorch
    SDPA provider only, and every other backend listed here requires an
    explicit opt-in via ``WORLDFOUNDRY_ATTENTION_BACKEND``/
    ``WORLDFOUNDRY_ATTENTION_IMPLEMENTATION``.
    """
    capabilities = probe_attention_backends()
    return tuple(
        capabilities[name]
        for name in (*_REPORT_PRIORITY, *_EXPLICIT_PRIORITY, *_EXPERIMENTAL_PRIORITY)
        if name in capabilities
    )


def gpu_supports_flash_attention(device: torch.device | str | int | None = None) -> bool:
    """Determine if the active CUDA device is architecturally capable of running FlashAttention.

    This in-tree FA2 path targets NVIDIA Ampere (SM8x), Ada (SM89), and
    Hopper (SM90). Blackwell kernels are resolved separately because SM100/103
    and SM120 are not binary-compatible feature targets.
    """
    capability = _cuda_compute_capability(device)
    if capability is None:
        return False
    # The in-tree FA2 dispatcher is validated for NVIDIA Ampere/Ada/Hopper.
    # Blackwell data-center (SM100/103) and client (SM120) kernels require
    # separate providers and must not inherit this compatibility decision.
    return capability[0] in {8, 9}


def gpu_supports_flash_attention_3(device: torch.device | str | int | None = None) -> bool:
    """Return whether the active device is a validated FA3 target.

    FA3 is a Hopper-specific provider in WorldFoundry.  Importability of the
    ``flash_attn_interface`` module alone is not sufficient: Ampere, Ada and
    Blackwell builds use different kernel targets.
    """

    return _cuda_compute_capability(device) == (9, 0)


# ──────────────────────────────────────────────────────────────────────────
# Hardware / package helpers — never import crash-prone extensions
# ──────────────────────────────────────────────────────────────────────────


def _cuda_compute_capability(
    device: torch.device | str | int | None = None,
) -> tuple[int, int] | None:
    """Return an NVIDIA CUDA capability without mistaking ROCm for CUDA."""

    if getattr(torch.version, "hip", None):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return int(major), int(minor)
    except Exception:
        return None


def _torch_cuda_accelerator_available(device: torch.device | str | int | None = None) -> bool:
    """Return whether *device* refers to PyTorch's CUDA/HIP accelerator API."""

    if not torch.cuda.is_available():
        return False
    if device is None or isinstance(device, int):
        return True
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, RuntimeError):
        return False


def _package_capability(
    *,
    name: str,
    package: str,
    import_name: str | None = None,
    usable_if: bool,
    unavailable_reason: str,
    unusable_reason: str,
) -> AttentionKernelCapability:
    """Underlying safe package prober.

    Walks package search paths without importing parent packages or extension modules.
    """
    parts = (import_name or package).split(".")
    spec = PathFinder.find_spec(parts[0])
    for index in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            spec = None
            break
        spec = PathFinder.find_spec(".".join(parts[:index + 1]), spec.submodule_search_locations)
    available = spec is not None
    if not available:
        return AttentionKernelCapability(
            name=name, package=package, available=False, usable=False, reason=unavailable_reason
        )
    if not usable_if:
        return AttentionKernelCapability(
            name=name, package=package, available=True, usable=False, reason=unusable_reason
        )
    return AttentionKernelCapability(name=name, package=package, available=True, usable=True)


__all__ = [
    "AttentionKernelCapability",
    "ModelSpecificAttentionBackendError",
    "attention_backend_capability",
    "attention_backend_from_env",
    "attention_backend_report",
    "gpu_supports_flash_attention",
    "normalize_attention_backend",
    "probe_attention_backends",
    "require_generic_attention_backend",
    "resolve_attention_backend",
    "resolve_transformers_attention_implementation",
]

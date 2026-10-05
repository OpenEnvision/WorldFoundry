"""Runtime inventory for acceleration families (TeaCache, Graph, FP8, …).

Responsibility: list what is *enabled and hardware-eligible* in this
process so a quality regression can be tied to a specific approximation.

This module is not a capability installer and does not enable anything.
Optional packages are never treated as integration. ``hardware_active``
is a probe (SM / ``_scaled_mm`` / CUDA version), not “the kernel ran”.

Public surface:
- :class:`AccelerationTechnology` — one family row for logs / diffs.
- :func:`acceleration_technology_report` — frozen tuple of in-tree rows.

:func:`acceleration_technology_report` lists what is *enabled*, not
what is installed. Use it in logs so a quality regression can be
tied to a specific approximation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from worldfoundry.core.acceleration.quantization.nvfp4 import _nvfp4_hardware_eligible
from worldfoundry.core.attention.sparse.piecewise import piecewise_attention_available
from worldfoundry.core.kernels import kernel_device_profile, kernel_dispatch_report


# ──────────────────────────────────────────────────────────────────────────
# Report row — JSON-safe fields for logs and regression diffs
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AccelerationTechnology:
    """One in-tree acceleration family: policy name, impl path, hardware bit."""

    family: str
    name: str
    implementation: str
    in_tree: bool
    approximate: bool
    hardware_active: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Flatten for JSON / log sinks without a custom encoder."""
        return asdict(self)


def _resolved_device(device: torch.device | str | int | None) -> torch.device:
    """Bind a report device without creating a CUDA context on an implicit GPU 0.

    A bare ``int`` is ``cuda:<n>``. ``None`` uses the current CUDA device
    only when CUDA is already available; otherwise CPU, so a CPU-only
    process does not call ``current_device()``.
    """
    if isinstance(device, int):
        return torch.device("cuda", device)
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(device)


# ──────────────────────────────────────────────────────────────────────────
# Inventory — enabled-in-tree rows, not optional-package advertisements
# ──────────────────────────────────────────────────────────────────────────


def acceleration_technology_report(
    device: torch.device | str | int | None = None,
) -> tuple[AccelerationTechnology, ...]:
    """Report real in-tree capabilities without treating optional packages as integration."""

    resolved = _resolved_device(device)
    profile = kernel_device_profile(resolved)
    kernels = kernel_dispatch_report().get("operators", {})
    fp8_active = (
        resolved.type == "cuda"
        and profile.supports_fp8
        and callable(getattr(torch, "_scaled_mm", None))
    )
    fp4_active = _nvfp4_hardware_eligible(resolved)
    pisa_active = piecewise_attention_available(resolved) if resolved.type == "cuda" else False
    fusion_names = ", ".join(sorted(str(name) for name in kernels))
    return (
        AccelerationTechnology(
            "cache",
            "fixed-step output cache",
            "worldfoundry.core.acceleration.FixedStepCache",
            True,
            True,
            True,
        ),
        AccelerationTechnology(
            "cache",
            "adaptive residual cache",
            "worldfoundry.core.acceleration.AdaptiveResidualCache",
            True,
            True,
            True,
        ),
        AccelerationTechnology(
            "token_pruning",
            "feature-norm token pruning",
            "worldfoundry.core.acceleration.TokenPruner",
            True,
            True,
            True,
        ),
        AccelerationTechnology(
            "sparse_attention",
            "PISA piecewise attention",
            "worldfoundry.core.attention.piecewise_attention",
            True,
            True,
            pisa_active,
            "TMA kernel requires SM90 or data-center Blackwell; other GPUs use exact SDPA"
            if not pisa_active
            else "",
        ),
        AccelerationTechnology(
            "quantization",
            "FP8 dynamic activation/weight linear",
            "worldfoundry.core.acceleration.Float8Linear",
            True,
            True,
            fp8_active,
            "native FP8 GEMM requires Ada/Hopper/Blackwell; dense fallback is retained"
            if not fp8_active
            else "",
        ),
        AccelerationTechnology(
            "quantization",
            "NVFP4 dynamic activation/weight linear",
            "worldfoundry.core.acceleration.NVFP4Linear",
            True,
            True,
            fp4_active,
            "native NVFP4 GEMM requires a known Blackwell target and CUDA >= 12.8; dense fallback is retained"
            if not fp4_active
            else "",
        ),
        AccelerationTechnology(
            "kernel_fusion",
            "DiT fused glue kernels",
            fusion_names or "PyTorch exact fallbacks",
            True,
            False,
            bool(kernels),
        ),
        AccelerationTechnology(
            "compilation",
            "persistent Inductor/Triton cache",
            "worldfoundry.core.execution.compile_cache",
            True,
            False,
            True,
        ),
    )


__all__ = ["AccelerationTechnology", "acceleration_technology_report"]

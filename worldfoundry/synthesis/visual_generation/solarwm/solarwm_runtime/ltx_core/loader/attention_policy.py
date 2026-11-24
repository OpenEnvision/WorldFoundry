# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""SolarWM attention backend selection; tensor callables come from core."""

import functools
import sys
from enum import Enum

import torch
from torch.nn.attention import SDPBackend

from worldfoundry.base_models.diffusion_model.optimizations.ltx.attention import MPSSdpaAttention
from worldfoundry.core.attention.backends.model_backends import (
    AttentionCallable,
    FlashAttention3,
    FlashAttention4,
    MaskedAttentionCallable,
    PytorchAttention,
    _resolve_sdpa_variant,
    _sdpa_full_priority,
)

flash_attn_interface = None
flash_attn_4_func = None
try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None
try:
    # FA4's cute path depends on a matching nvidia-cutlass-dsl; a newer cutlass
    # can raise AttributeError (e.g. missing cute.core.ThrMma) rather than
    # ImportError — treat any probe failure as "FA4 unavailable".
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
except Exception:
    flash_attn_4_func = None
try:
    # macOS only: routes SDPA to Apple's prebuilt MPSGraph attention kernel.
    from mps_sdpa import sdpa_opt as _mps_sdpa_opt
except ImportError:
    _mps_sdpa_opt = None


# --- Automatic selection -----------------------------------------------------
# AUTOMATIC inspects installed extras and the GPU arch and returns the fastest
# usable callable for each path. The selection runs once per process (cached).
# The unmasked and masked picks are independent: each calls its own helper and
# may end up on different backends (e.g. FA3 unmasked + SDPA masked on H100).


def _on_macos() -> bool:
    """True on macOS, where torch's native SDPA materializes the score matrix and
    AUTOMATIC routes to Apple's fused ``mps-sdpa`` kernel instead."""
    return sys.platform == "darwin"


def _mps_sdpa_available() -> bool:
    """True when the ``mps-sdpa`` package is importable. It is a platform-marked
    hard dependency on Apple Silicon, so this is always True there; it is only
    False on non-Apple-Silicon macs (e.g. Intel/CPU), where AUTOMATIC falls back
    to torch's SDPA (acceptable on CPU, which has no MPS memory wall)."""
    return _mps_sdpa_opt is not None


def _select_primary_attention() -> AttentionCallable:
    """Pick the fastest unmasked attention based on installed extras and GPU arch.
    Priority by arch:
    - Hopper (sm_90, H100): FA3 > FA4 > SDPA.
    - Datacenter Blackwell (sm_100, B200): FA4 > SDPA. FA4 is intentionally *not*
      picked on consumer Blackwell (sm_120) -- known regressions in newer
      FA4 betas; users who want it on sm_120 must opt in explicitly.
    - macOS (Apple Silicon / MPS): Apple's fused MPSGraph kernel via ``mps-sdpa``
      (a platform-marked hard dependency on Apple Silicon) -- it avoids the
      full-score-matrix memory wall on long video sequences. On a non-Apple-Silicon
      mac (Intel/CPU) it falls back to torch's SDPA.
    - Everywhere else (Ada, Ampere, CPU): SDPA with the full backend priority
      list -- torch's runtime dispatcher picks the best fit at call time.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major == 9:
            if flash_attn_interface is not None:
                return FlashAttention3()
            if flash_attn_4_func is not None:
                return FlashAttention4()
        if major == 10 and flash_attn_4_func is not None:
            return FlashAttention4()
    if _on_macos():
        return MPSSdpaAttention() if _mps_sdpa_available() else _sdpa_full_priority()
    return _sdpa_full_priority()


def _select_masked_attention() -> MaskedAttentionCallable:
    """Pick a mask-aware attention. On macOS, Apple's fused MPSGraph kernel via
    ``mps-sdpa`` (a hard dependency on Apple Silicon, else torch's SDPA on
    Intel/CPU macs); else SDPA with the full priority list (the dispatcher
    rejects FLASH automatically when a mask is present and walks past it --
    torch SDPA handles the additive mask directly)."""
    if _on_macos():
        return MPSSdpaAttention() if _mps_sdpa_available() else _sdpa_full_priority()
    return _sdpa_full_priority()


@functools.cache
def automatic_attention() -> AttentionCallable:
    """Cached AUTOMATIC pick for the unmasked path.
    Cached so every ``AttentionOps`` in the process shares one instance."""
    return _select_primary_attention()


@functools.cache
def automatic_masked_attention() -> MaskedAttentionCallable:
    """Cached AUTOMATIC pick for the masked path. See :func:`automatic_attention`."""
    return _select_masked_attention()


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_FLASH = "sdpa_flash"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    # Apple's fused MPSGraph SDPA via the `mps-sdpa` package (macOS/MPS only, a
    # platform-marked hard dependency on Apple Silicon). The AUTOMATIC default on
    # MPS; never materializes the score matrix.
    MPS_SDPA = "mps_sdpa"
    # Pick the fastest unmasked backend for the current GPU/extras combo; see
    # :func:`automatic_attention`. Default for :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks.
        Every non-AUTOMATIC variant raises :class:`RuntimeError` when the backend
        isn't usable on this machine -- missing package or SDPA backend rejected
        on this hardware (e.g. cuDNN under ``torch.use_deterministic_algorithms``).
        Opting in means "this kernel or fail loudly". ``AUTOMATIC`` returns the
        cached :func:`automatic_attention` instance so every build shares one callable.
        """
        match self:
            case AttentionFunction.AUTOMATIC:
                return automatic_attention()
            case AttentionFunction.PYTORCH:
                return PytorchAttention()
            case AttentionFunction.FLASH_ATTENTION_3:
                if flash_attn_interface is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 selected but `flash-attn-3` is not installed."
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 requires CUDA. Use PyTorch SDPA on CPU or MPS."
                    )
                return FlashAttention3()
            case AttentionFunction.FLASH_ATTENTION_4:
                if flash_attn_4_func is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 selected but `flash-attn-4` is not installed."
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 requires CUDA. Use PyTorch SDPA on CPU or MPS."
                    )
                return FlashAttention4()
            case AttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case AttentionFunction.MPS_SDPA:
                if _mps_sdpa_opt is None:
                    raise RuntimeError("AttentionFunction.MPS_SDPA selected but `mps-sdpa` is not installed.")
                return MPSSdpaAttention()
            case AttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "AttentionFunction.SDPA_CUDNN", with_mask=False
                )
            case AttentionFunction.SDPA_FLASH:
                return _resolve_sdpa_variant(
                    SDPBackend.FLASH_ATTENTION, "AttentionFunction.SDPA_FLASH", with_mask=False
                )
            case AttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "AttentionFunction.SDPA_EFFICIENT", with_mask=False
                )


class MaskedAttentionFunction(Enum):
    """Backends usable on the masked path. Mirrors :class:`AttentionFunction` minus
    the variants the torch SDPA dispatcher (or the wrapped kernel) rejects with a
    mask: ``SDPA_FLASH`` -- FLASH kernel cannot serve an additive ``attn_mask``;
    ``FLASH_ATTENTION_3``/``FLASH_ATTENTION_4`` -- neither has a mask kernel at all.
    Keeping them out makes "this backend cannot mask" a type error, not a runtime one."""

    PYTORCH = "pytorch"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    # Apple's fused MPSGraph SDPA via the `mps-sdpa` package (macOS/MPS only, a
    # platform-marked hard dependency on Apple Silicon); the AUTOMATIC default on
    # MPS. Mask-aware.
    MPS_SDPA = "mps_sdpa"
    # Pick the fastest mask-capable backend for the current extras combo; see
    # :func:`automatic_masked_attention`. Default for the masked slot of
    # :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> MaskedAttentionCallable:
        """Resolve to a concrete masked callable. Same backend classes as
        :meth:`AttentionFunction.to_callable`; the protocol returned just exposes
        the masked call signature.
        Non-AUTOMATIC variants raise :class:`RuntimeError` when the backend isn't
        usable for the masked path on this machine. SDPA probes run with
        ``with_mask=True`` so the capability check considers the protocol the
        caller will actually use."""
        match self:
            case MaskedAttentionFunction.AUTOMATIC:
                return automatic_masked_attention()
            case MaskedAttentionFunction.PYTORCH:
                return PytorchAttention()
            case MaskedAttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case MaskedAttentionFunction.MPS_SDPA:
                if _mps_sdpa_opt is None:
                    raise RuntimeError("MaskedAttentionFunction.MPS_SDPA selected but `mps-sdpa` is not installed.")
                return MPSSdpaAttention()
            case MaskedAttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "MaskedAttentionFunction.SDPA_CUDNN", with_mask=True
                )
            case MaskedAttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "MaskedAttentionFunction.SDPA_EFFICIENT", with_mask=True
                )

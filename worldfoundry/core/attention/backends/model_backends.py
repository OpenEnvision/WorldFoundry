"""Model-facing attention callables: SDPA wrappers, piecewise, and backend enums.

DiT blocks should depend on these Protocols / Enums rather than importing
FlashAttention or xFormers. The split is intentional:

- :class:`AttentionCallable` / :class:`MaskedAttentionCallable` — QKV
  signatures used by native transformer blocks.
- :class:`PytorchAttention` — in-tree SDPA with a recorded backend
  priority (cuDNN / Flash / Efficient / Math). ``auto`` stays here.
- :class:`XFormersAttention` / :class:`FlashAttention3` /
  :class:`FlashAttention4` — explicit opt-in adapters. They are never
  selected by ``automatic_attention()``.
- :class:`PiecewiseAttention` — in-tree PISA sparse path for long LTX
  self-attention; density/block size come from env or constructor.
- :class:`AttentionFunction` / :class:`MaskedAttentionFunction` —
  string-stable enums so a recipe can pin a backend without importing
  the adapter class.

:func:`_torch_default_sdpa_priority` reads torch's live priority so the
wrapper does not drift when PyTorch reorders cuDNN vs Flash.
"""

import functools
import importlib.util
import logging
import os
from enum import Enum
from typing import Protocol

import torch
from torch.nn.attention import SDPBackend

from worldfoundry.core.attention.backends.native import (
    flattened_multihead_attention,
    native_sdpa_priority,
)
from worldfoundry.core.attention.sparse.piecewise import piecewise_attention

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# SDPA helpers — snapshot torch's live priority; never hard-code the order
# ──────────────────────────────────────────────────────────────────────────


def _torch_default_sdpa_priority() -> list[SDPBackend]:
    """Fetch torch's current default SDPA priority order at runtime.
    Used as the default for ``PytorchAttention`` so the wrapper-always
    code path matches torch's native dispatch order without hard-coding it
    (which would drift if torch updates the default).
    ``torch._C._get_sdp_priority_order`` is a private API; we accept that
    risk because the project pins ``torch`` in the lockfile, so any
    rename/removal surfaces on a controlled torch bump rather than silently.
    """
    getter = getattr(torch._C, "_get_sdp_priority_order", None)
    if callable(getter):
        return [SDPBackend(p) for p in getter()]
    # torch 2.5 exposes ``sdpa_kernel`` but not the priority introspection
    # helper added later. Preserve its native CUDA preference and let the
    # context manager skip kernels unavailable on the current device.
    return [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]


def _module_available(name: str) -> bool:
    """Return whether ``name`` is importable without executing the package."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# ──────────────────────────────────────────────────────────────────────────
# Protocols + adapters — DiT blocks depend on these, not on Flash/xFormers
# ──────────────────────────────────────────────────────────────────────────


class AttentionCallable(Protocol):
    """Unmasked attention. Backends without a mask kernel (FA3/FA4) implement only
    this protocol; backends that support masks too (Pytorch/SDPA, xFormers) are
    structurally usable here and as :class:`MaskedAttentionCallable`."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
        """Unmasked QKV; ``heads`` unpacks the flattened last dimension."""
        ...


class MaskedAttentionCallable(Protocol):
    """Masked attention. Mask is required (not optional) -- the caller has already
    decided this is the masked path and chosen a backend that can serve it. Used
    by :class:`Attention` when its forward receives a non-None ``mask``."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked QKV; ``mask`` is required because this is already the masked path."""
        ...


class PytorchAttention(AttentionCallable):
    """Exact SDPA wrapper with an explicit backend priority list.

    ``priority=None`` snapshots torch's current default order at
    construction so later ``sdpa_kernel`` mutations do not change an
    already-built block. Also implements the masked protocol.
    """

    def __init__(self, priority: list[SDPBackend] | None = None) -> None:
        """Snapshot ``priority`` or torch's current default SDPA order."""
        # priority=None -> snapshot torch's default SDPA priority at construction.
        self._priority = priority if priority is not None else _torch_default_sdpa_priority()

    @property
    def label(self) -> str:
        """Human-readable identifier (used in the AUTOMATIC selection log).
        Encodes the SDPA priority list so a single-backend pin reads differently
        from the full-priority dispatcher walk."""
        return f"SDPA[{'>'.join(b.name for b in self._priority)}]"

    def __call__(
        self,         q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Serve both unmasked and masked protocols with the pinned SDPA order."""
        return flattened_multihead_attention(
            q,
            k,
            v,
            heads,
            attn_mask=mask,
            backends=self._priority,
        )


class XFormersAttention(AttentionCallable):
    """Explicit xFormers ``memory_efficient_attention`` adapter.

    Bool masks are converted to additive ``0 / -inf`` (xformers does not
    accept SDPA bool masks). The last mask dimension is padded to an
    8-aligned *storage* width, then sliced back to the logical key
    length. Not part of ``automatic_attention()``.
    """

    label = "xFormers"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run xFormers attention; bool masks become additive so they actually mask."""
        try:
            from xformers.ops import memory_efficient_attention
        except (ImportError, OSError) as exc:
            raise RuntimeError("XFormersAttention was selected but `xformers` is unavailable.") from exc

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            if mask.dtype == torch.bool:
                # CC-09: xformers' ``attn_bias`` is additive. Assigning a bool
                # mask into a float tensor would silently cast it to 1.0/0.0,
                # which barely biases the softmax instead of masking. Convert
                # to SDPA bool-mask semantics explicitly: True = attend (0.0),
                # False = masked out (-inf).
                additive = torch.zeros(mask.shape, dtype=q.dtype, device=mask.device)
                additive.masked_fill_(~mask, float("-inf"))
                mask = additive
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad the key dimension up to a multiple of 8 (0 if already aligned)
            pad = (-mask.shape[-1]) % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # xformers requires the last dimension's *storage* to be 8-aligned.
            # Slicing back to the logical key width keeps the values identical
            # while the underlying buffer (allocated with the padded width)
            # satisfies the alignment; the padding columns are never read.
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    """Explicit Hopper FlashAttention-3 adapter (no mask support)."""

    label = "FlashAttention3"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        """Run Hopper FA3; there is no mask kernel on this path."""
        try:
            import flash_attn_interface
        except (ImportError, OSError) as exc:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is unavailable.") from exc

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention4(AttentionCallable):
    """Explicit FlashAttention-4 (cute) adapter (no mask support)."""

    label = "FlashAttention4"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        """Run CuTeDSL FA4; there is no mask kernel on this path."""
        try:
            from flash_attn.cute import flash_attn_func as flash_attn_4_func
        except (ImportError, OSError) as exc:
            raise RuntimeError("FlashAttention4 was selected but `flash-attn-4` is unavailable.") from exc

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out, _ = flash_attn_4_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class PiecewiseAttention(AttentionCallable):
    """Explicit in-tree PISA adapter for long LTX self-attention."""

    label = "PISA[in-tree]"

    def __init__(self, density: float | None = None, block_size: int | None = None) -> None:
        """Read density/block size from env when the constructor leaves them unset."""
        self.density = float(os.getenv("WORLDFOUNDRY_PISA_DENSITY", "0.1")) if density is None else float(density)
        self.block_size = (
            int(os.getenv("WORLDFOUNDRY_PISA_BLOCK_SIZE", "64")) if block_size is None else int(block_size)
        )

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
        """Apply piecewise sparse attention; density/block size are constructor policy."""
        batch, _, hidden = q.shape
        head_dim = hidden // heads
        q_heads, k_heads, v_heads = (value.view(batch, -1, heads, head_dim).transpose(1, 2) for value in (q, k, v))
        out = piecewise_attention(
            q_heads,
            k_heads,
            v_heads,
            density=self.density,
            block_size=self.block_size,
        )
        return out.transpose(1, 2).reshape(batch, -1, hidden)


# ──────────────────────────────────────────────────────────────────────────
# Automatic selection — stay inside PyTorch; external providers stay explicit
# ──────────────────────────────────────────────────────────────────────────
# AUTOMATIC chooses an exact SDPA policy per GPU family. External providers
# remain explicit enum choices, so the default LTX path never depends on
# another repository package.


def _sdpa_can_use(backend: SDPBackend, *, with_mask: bool) -> bool:
    """Ask torch whether *backend* can run with the given mask shape.
    ``MATH`` is the universal SDPA fallback (pure PyTorch ops, no kernel
    requirements) so it returns True everywhere, CPU included. The other
    backends use ``torch.backends.cuda.can_use_*`` capability checks (no GPU
    compute, no synchronization) and are False without CUDA. The probe shapes
    are small but realistic enough to surface constraints (head dim, dtype)
    that the per-backend rules care about.
    """
    if backend is SDPBackend.MATH:
        return True
    if not torch.cuda.is_available():
        return False
    q = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16) if with_mask else None
    params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, False)
    if backend is SDPBackend.CUDNN_ATTENTION:
        return torch.backends.cuda.can_use_cudnn_attention(params, debug=False)
    if backend is SDPBackend.FLASH_ATTENTION:
        return torch.backends.cuda.can_use_flash_attention(params, debug=False)
    if backend is SDPBackend.EFFICIENT_ATTENTION:
        return torch.backends.cuda.can_use_efficient_attention(params, debug=False)
    return False


_SDPA_FULL_PRIORITY: tuple[SDPBackend, ...] = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def _sdpa_full_priority() -> PytorchAttention:
    """Hand SDPA the full backend priority order; let torch's dispatcher pick at call time.
    ``sdpa_kernel(_SDPA_FULL_PRIORITY, set_priority=True)`` enables all four
    backends and orders them; torch then walks the order at call time and picks
    the first backend whose ``can_use_*`` check passes for the actual
    shapes/dtype/mask. FLASH is rejected automatically when a mask is present;
    CUDNN may be rejected under deterministic mode; MATH is the universal
    fallback. Probing per-backend usability up front from generic probe shapes
    cannot anticipate the variety of real call sites (e.g. broadcast key-only
    masks, large head dim), so we defer the choice to the dispatcher.
    """
    return PytorchAttention(priority=list(_SDPA_FULL_PRIORITY))


def _native_pytorch_attention(
    *,
    with_mask: bool,
    device: torch.device | str | None = None,
) -> PytorchAttention:
    """Build a :class:`PytorchAttention` from :func:`native_sdpa_priority` names.

    An empty priority (ROCm / unknown capability) falls back to torch's
    live default order rather than inventing an architecture-specific list.
    """
    names = native_sdpa_priority(device, has_mask=with_mask)
    if not names:
        return PytorchAttention()
    mapping = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return PytorchAttention(priority=[mapping[name] for name in names])


class _DeviceAwarePytorchAttention(AttentionCallable):
    """Choose the in-tree exact SDPA policy from each call's tensor device."""

    def __init__(self, *, with_mask: bool) -> None:
        """Prebuild per-device SDPA policies so compiled forwards never probe CUDA."""
        self.with_mask = with_mask
        self.label = "SDPA[device-aware,masked]" if with_mask else "SDPA[device-aware]"
        # Build policies outside model forward.  In particular,
        # ``torch._C._get_sdp_priority_order`` and CUDA capability queries are
        # not Dynamo-traceable and must never execute inside a compiled graph.
        self._cpu_attention = _native_pytorch_attention(
            with_mask=with_mask,
            device=torch.device("cpu"),
        )
        self._fallback_attention = PytorchAttention()
        self._cuda_attention = {
            index: _native_pytorch_attention(
                with_mask=with_mask,
                device=torch.device("cuda", index),
            )
            for index in range(torch.cuda.device_count())
        }

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pick the policy for ``q``'s device; unknown CUDA indices use the fallback."""
        if q.device.type != "cuda":
            implementation = self._cpu_attention
        else:
            implementation = self._cuda_attention.get(
                q.device.index,
                self._fallback_attention,
            )
        return implementation(q, k, v, heads, mask)


def _select_primary_attention() -> AttentionCallable:
    """Use bundled device-aware exact SDPA; optional packages remain explicit."""

    return _DeviceAwarePytorchAttention(with_mask=False)


def _select_masked_attention() -> MaskedAttentionCallable:
    """Use bundled mask-aware SDPA with cuDNN preference on modern CUDA."""

    return _DeviceAwarePytorchAttention(with_mask=True)


@functools.cache
def automatic_attention() -> AttentionCallable:
    """Cached AUTOMATIC pick for the unmasked path. Logs the chosen label once
    per process."""
    fn = _select_primary_attention()
    logger.info("Automatic attention selected: %s", fn.label)
    return fn


@functools.cache
def automatic_masked_attention() -> MaskedAttentionCallable:
    """Cached AUTOMATIC pick for the masked path. Logs the chosen label once
    per process."""
    fn = _select_masked_attention()
    logger.info("Automatic masked attention selected: %s", fn.label)
    return fn


# ──────────────────────────────────────────────────────────────────────────
# String-stable enums — recipes pin a backend without importing the adapter
# ──────────────────────────────────────────────────────────────────────────


def _resolve_sdpa_variant(backend: SDPBackend, name: str, *, with_mask: bool) -> PytorchAttention:
    """Build a single-backend ``PytorchAttention`` pin, raising if the backend
    can't actually serve the call on this machine. Used by both
    :meth:`AttentionFunction.to_callable` and :meth:`MaskedAttentionFunction.to_callable`;
    ``with_mask`` differs between the two so the capability check considers
    the protocol the caller intends to use. Not used for ``MATH`` -- MATH is
    the universal fallback and would falsely fail the CUDA-only probe on CPU.
    """
    if not _sdpa_can_use(backend, with_mask=with_mask):
        raise RuntimeError(
            f"{name} selected but the SDPA {backend.name} backend is not usable on this machine "
            "(either no CUDA, the backend rejected the probe shapes, or "
            "torch.use_deterministic_algorithms(True) excluded it)."
        )
    return PytorchAttention(priority=[backend])


class AttentionFunction(Enum):
    """String-stable unmasked attention backend pin for native DiT blocks.

    ``AUTOMATIC`` stays inside PyTorch SDPA. Every other member is an
    explicit opt-in and raises if the package or SDPA backend is unusable.
    """

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_FLASH = "sdpa_flash"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    PIECEWISE = "piecewise"
    # Pick the fastest unmasked backend for the current GPU/extras combo; see
    # :func:`automatic_attention`. Default for :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> AttentionCallable:  # noqa: PLR0911
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks.
        Every non-AUTOMATIC variant raises :class:`RuntimeError` when the backend
        isn't usable on this machine -- missing package or SDPA backend rejected
        on this hardware (e.g. cuDNN under ``torch.use_deterministic_algorithms``).
        Opting in means "this kernel or fail loudly". ``AUTOMATIC`` returns the
        cached :func:`automatic_attention` instance so the once-per-process log
        fires only on the first resolution.
        """
        match self:
            case AttentionFunction.AUTOMATIC:
                return automatic_attention()
            case AttentionFunction.PYTORCH:
                return PytorchAttention()
            case AttentionFunction.XFORMERS:
                if not _module_available("xformers"):
                    raise RuntimeError("AttentionFunction.XFORMERS selected but `xformers` is not installed.")
                return XFormersAttention()
            case AttentionFunction.FLASH_ATTENTION_3:
                if not _module_available("flash_attn_interface"):
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 selected but `flash-attn-3` is not installed."
                    )
                return FlashAttention3()
            case AttentionFunction.FLASH_ATTENTION_4:
                # CC-10: probe the FA4-specific submodule; ``flash_attn`` alone
                # is satisfied by a FlashAttention 2 install whose first
                # forward would then fail late inside FlashAttention4.
                if not _module_available("flash_attn") or not _module_available("flash_attn.cute"):
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 selected but the FlashAttention 4 build "
                        "(`flash_attn.cute`) is not installed."
                    )
                return FlashAttention4()
            case AttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case AttentionFunction.PIECEWISE:
                return PiecewiseAttention()
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
    """String-stable masked attention backend pin (SDPA / xFormers only).

    FlashAttention 3/4 are omitted because they have no mask kernel.
    ``AUTOMATIC`` uses device-aware SDPA with a cuDNN preference.
    """

    """Backends usable on the masked path. Mirrors :class:`AttentionFunction` minus
    the variants the torch SDPA dispatcher (or the wrapped kernel) rejects with a
    mask: ``SDPA_FLASH`` -- FLASH kernel cannot serve an additive ``attn_mask``;
    ``FLASH_ATTENTION_3``/``FLASH_ATTENTION_4`` -- neither has a mask kernel at all.
    Keeping them out makes "this backend cannot mask" a type error, not a runtime one."""

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
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
            case MaskedAttentionFunction.XFORMERS:
                if not _module_available("xformers"):
                    raise RuntimeError("MaskedAttentionFunction.XFORMERS selected but `xformers` is not installed.")
                return XFormersAttention()
            case MaskedAttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case MaskedAttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "MaskedAttentionFunction.SDPA_CUDNN", with_mask=True
                )
            case MaskedAttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "MaskedAttentionFunction.SDPA_EFFICIENT", with_mask=True
                )

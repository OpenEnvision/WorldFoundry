"""In-tree accelerator kernels with capability probing, registry dispatch, and portable PyTorch fallbacks.

The contract is: use a Triton / architecture kernel when it is usable, otherwise
a mathematically equivalent PyTorch path. Callers should depend on this package's
public functions (``group_norm_silu``, ``qk_rmsnorm_rope``, ``routed_swiglu_moe``,
…) and not import ``triton_*`` files directly.

- ``capabilities``: ``KernelDeviceProfile`` for the current device (CUDA/HIP, SM,
  whether Triton is allowed). This decides *whether* a fused kernel may run.
- ``registry`` / ``diffusion``: cache the chosen implementation per op name.
  Unsupported shapes or failures fall back to PyTorch.
  ``kernel_dispatch_report`` / ``clear_kernel_dispatch_cache`` reset that cache
  after a device change or a hot-reloaded extension in a long-lived process.
- ``native_provider``: optional separately-distributed native DSO, never imported
  at package definition time.
- ``triton_*``: concrete kernels. Those files should carry a module note plus
  entry-function docs — do not copy autotune details into business code.
- ``moe`` / ``quantized_gemm``: dispatch entries for routed SwiGLU MoE and
  quantized GEMM.

Not this package:
    Attention backends live in :mod:`worldfoundry.core.attention`. Persistent
    compile-cache roots live in :mod:`worldfoundry.core.execution.compile_cache`.
    NVFP4 / weight-only quantization lives in
    :mod:`worldfoundry.core.acceleration`.

Public surface:

- Diffusion / AdaLN / RoPE ops re-exported from :mod:`.diffusion`.
- :func:`routed_swiglu_moe` / :func:`routed_swiglu_moe_pytorch` from :mod:`.moe`.
- :func:`routed_swiglu_moe_triton` is resolved lazily so CPU-only imports survive.
- :class:`KernelDeviceProfile` / :func:`kernel_device_profile` from :mod:`.capabilities`.

Triton implementations import lazily (see ``routed_swiglu_moe_triton``) so a
CPU-only environment does not fail on ``import``.
"""

# ──────────────────────────────────────────────────────────────────────────
# Eager public re-exports — Triton MoE stays lazy so CPU imports stay cheap
# ──────────────────────────────────────────────────────────────────────────

from worldfoundry.core.kernels.capabilities import KernelDeviceProfile, kernel_device_profile
from worldfoundry.core.kernels.diffusion import (
    clear_kernel_dispatch_cache,
    group_norm_silu,
    hidden_qk_rmsnorm_rope_3d,
    kernel_dispatch_report,
    layer_norm_scale_shift,
    qk_rmsnorm_rope,
    residual_gate_add,
    residual_gate_add_,
    rms_norm_scale_shift,
    scale_shift,
    silu_and_mul,
    silu_mul,
)
from worldfoundry.core.kernels.moe import routed_swiglu_moe, routed_swiglu_moe_pytorch


def __getattr__(name: str):
    """Resolve optional Triton-only names without importing Triton at package load.

    Failure: unknown names raise :class:`AttributeError`. A missing Triton
    installation surfaces only when ``routed_swiglu_moe_triton`` is first
    requested, not when this package is imported.
    """

    if name == "routed_swiglu_moe_triton":
        from worldfoundry.core.kernels.triton_moe import routed_swiglu_moe_triton

        return routed_swiglu_moe_triton
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "clear_kernel_dispatch_cache",
    "group_norm_silu",
    "hidden_qk_rmsnorm_rope_3d",
    "KernelDeviceProfile",
    "kernel_device_profile",
    "kernel_dispatch_report",
    "layer_norm_scale_shift",
    "qk_rmsnorm_rope",
    "residual_gate_add",
    "residual_gate_add_",
    "rms_norm_scale_shift",
    "scale_shift",
    "routed_swiglu_moe_pytorch",
    "routed_swiglu_moe",
    "routed_swiglu_moe_triton",
    "silu_and_mul",
    "silu_mul",
]

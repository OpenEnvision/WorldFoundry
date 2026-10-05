"""Triton-fused GDN operator surface for SANA video / V2V / CamCtrl.

This package does not own the Python GDN math (that lives in
``sana_gdn_blocks`` / ``sana_v2v_attn_blocks``).  It prepares QKV + RoPE
tables and dispatches to:

* :mod:`fused_gdn` — bidirectional fused scan + RoPE / inv-RMS helpers
* :mod:`fused_gdn_chunkwise` — chunk-parallel Phase A/B/C pipeline (default
  when ``USE_CHUNKWISE_GDN=1``)
* :mod:`fused_cam_gdn` — camera-branch UCPE prep fused with the same scan

``_prepare_fused_gdn_inputs`` is the shared projection/normalize step used
by the V2V attention blocks.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_gdn import _precompute_inv_rms, fused_bidi_merge, prepare_rope_tables
from .fused_gdn_chunkwise import fused_bidi_stateful_chunkwise_shared_phase_a


def _resolve_gdn_variant() -> str:
    """Pick the V2V GDN forward path from environment configuration."""
    return "chunkwise" if os.environ.get("USE_CHUNKWISE_GDN", "1") == "1" else "pytorch"


@dataclass(frozen=True)
class _FusedGDNPrep:
    """Normalized tensors and dimensions consumed by fused V2V GDN kernels."""

    B: int
    N: int
    C: int
    T: int
    H_s: int
    W_s: int
    S: int
    H: int
    D: int
    dtype_orig: torch.dtype
    qkv: torch.Tensor
    beta_p: torch.Tensor
    decay: torch.Tensor
    k_scale: float
    q_nw: torch.Tensor
    k_nw: torch.Tensor


def _execution_weight_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return materialized parameters for both generic and linear VRAM wrappers."""

    computation = getattr(module, "computation", None)
    if callable(computation):
        materialized = computation()
        if isinstance(materialized, tuple):
            weight, bias = materialized
            return weight, bias
        module = materialized
    return module.weight, getattr(module, "bias", None)


def _prepare_fused_gdn_inputs(self, x: torch.Tensor, HW) -> _FusedGDNPrep:
    """Project and normalize the common fused V2V GDN inputs."""
    B, N, C = x.shape
    T, H_s, W_s = HW
    S = H_s * W_s
    H, D = self.heads, self.dim

    # The fused path consumes weights directly instead of calling q/k/v.  Use
    # each VRAM wrapper's materialized computation module; ``wrapper.weight``
    # is intentionally the CPU/offloaded source parameter.
    q_w, q_b = _execution_weight_bias(self.q)
    k_w, k_b = _execution_weight_bias(self.k)
    v_w, v_b = _execution_weight_bias(self.v)
    q_w = q_w.squeeze(-1)
    k_w = k_w.squeeze(-1)
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
    biases = (q_b, k_b, v_b)
    qkv_b = None
    if any(bias is not None for bias in biases):
        qkv_b = torch.cat(
            [
                bias
                if bias is not None
                else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
                for bias, weight in zip(biases, (q_w, k_w, v_w))
            ]
        )
    qkv = F.linear(x, qkv_w, qkv_b).reshape(B, N, 3, H, D)

    beta, decay = self._compute_frame_gates(x, HW)
    beta_p = beta.permute(0, 3, 1, 2).contiguous()
    k_scale = (D**-0.5) * (S**-0.5)

    if not isinstance(self.q_norm, nn.Identity):
        q_nw = _execution_weight_bias(self.q_norm)[0].to(device=x.device, dtype=torch.float32)
        k_nw = _execution_weight_bias(self.k_norm)[0].to(device=x.device, dtype=torch.float32)
    else:
        q_nw = torch.ones(C, device=x.device, dtype=torch.float32)
        k_nw = torch.ones(C, device=x.device, dtype=torch.float32)

    return _FusedGDNPrep(
        B=B,
        N=N,
        C=C,
        T=T,
        H_s=H_s,
        W_s=W_s,
        S=S,
        H=H,
        D=D,
        dtype_orig=x.dtype,
        qkv=qkv,
        beta_p=beta_p,
        decay=decay,
        k_scale=k_scale,
        q_nw=q_nw,
        k_nw=k_nw,
    )


__all__ = [
    "_precompute_inv_rms",
    "_execution_weight_bias",
    "_prepare_fused_gdn_inputs",
    "_resolve_gdn_variant",
    "fused_bidi_merge",
    "fused_bidi_stateful_chunkwise_shared_phase_a",
    "prepare_rope_tables",
]

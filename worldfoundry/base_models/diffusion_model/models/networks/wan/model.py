"""Native packed-tensor Wan 2.1 / 2.2 diffusion transformer.

This is the WorldFoundry-owned graph used by most native Wan recipes.
It is checkpoint-compatible with official Wan DiT weights but uses a
single ``B C F H W`` latent (not the official list-of-videos packing in
:mod:`.reference_21` / :mod:`.reference_22`).

Architecture
------------
- 3D conv patch embed, UMT5 text projection, sinusoidal timestep + AdaLN.
- Optional CLIP image tokens (Wan 2.1 I2V), VAE condition concat, FPS
  sample-info, and :class:`SimpleAdapter` camera residual.
- Each :class:`DiTBlock` is self-attn (3D RoPE) → text/image cross-attn
  → FFN, all gated by six timestep modulation channels.
- Attention processors are swappable so research variants (TeaCache,
  action, camera, linear attn) can replace QKV policy without forking
  the stem.

Variant hooks (override on subclasses; default is a no-op)
----------------------------------------------------------
``prepare_condition_context``, ``prepare_token_sequence``,
``block_forward_kwargs``, ``after_transformer_block``,
``finalize_token_sequence``.  Optional ``memory_adapter`` can rewrite
tokens before the block loop (Echo-Infinity / memory recipes).

VACE, causal attention, and linear attention are *not* in this file;
they live in :mod:`.vace` and :mod:`.variants`.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.core.attention import (
    apply_complex_rotary_embedding as rope_apply,
)
from worldfoundry.core.attention import (
    complex_rotary_frequencies_3d as precompute_freqs_cis_3d,
)
from worldfoundry.core.attention import (
    packed_sequence_attention as flash_attention,
)
from worldfoundry.core.kernels import (
    hidden_qk_rmsnorm_rope_3d,
    layer_norm_scale_shift,
    residual_gate_add,
    residual_gate_add_,
)
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.core.nn import RMSNorm, sinusoidal_embedding_1d
from worldfoundry.core.nn import scale_shift as modulate  # noqa: F401
from worldfoundry.core.nn.checkpointing.gradient import gradient_checkpoint_forward

from .adapter import SimpleAdapter

_SELF_ATTENTION_INTERNAL_KWARGS = (
    "_worldfoundry_attention_context",
    "_worldfoundry_rope_precision",
    "_worldfoundry_rope_grid",
    "_worldfoundry_rope_table",
    "_worldfoundry_sparse_grid",
)


class AttentionModule(nn.Module):
    """Thin wrapper around packed-sequence flash attention.

    ``compatibility_mode`` forces exact PyTorch SDPA.  Training binds that
    flag on every instance via :meth:`WanModel.set_attention_compatibility_mode`
    because the shared dispatcher snapshots its backend at import time.
    """

    def __init__(self, num_heads):
        """Store head count used by the packed-sequence attention kernel.

        Args:
            num_heads: Number of attention heads (must divide the hidden size).
        """
        super().__init__()
        self.num_heads = num_heads
        self.compatibility_mode = False
        self.attention_backend: str | None = None

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind a pipeline-scoped backend without changing process globals."""

        self.attention_backend = None if backend is None else str(backend)

    def forward(self, q, k, v):
        """Attend ``q`` to ``k``/``v`` and return the packed output tokens.

        Args:
            q: Query tokens after RMSNorm / RoPE, last dim is ``head_dim``.
            k: Key tokens, same layout as ``q``.
            v: Value tokens, same layout as ``q``.
        """
        options = getattr(self, "_worldfoundry_attention_options", None)
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            compatibility_mode=self.compatibility_mode,
            backend=self.attention_backend,
            **({"backend_options": options} if options else {}),
        )
        return x


class SelfAttentionProcessor:
    """Default Wan self-attention policy, replaceable by runtime adapters."""

    def __call__(
        self,
        attention: "SelfAttention",
        x: torch.Tensor,
        freqs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        q = attention.q(x)
        k = attention.k(x)
        v = attention.v(x)
        fused_table = kwargs.pop("_worldfoundry_rope_table", None)
        fused_grid = kwargs.pop("_worldfoundry_rope_grid", None)
        q, k = apply_wan_qk_norm_rope(
            attention,
            q,
            k,
            freqs,
            fused_table=fused_table,
            fused_grid=fused_grid,
            precision=kwargs.pop("_worldfoundry_rope_precision", "fp64"),
        )
        output = attention.attn(q, k, v)
        return attention.o(output)


def apply_wan_qk_norm_rope(
    attention: "SelfAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    *,
    fused_table: torch.Tensor | None = None,
    fused_grid: tuple[int, int, int] | None = None,
    precision: str = "fp64",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Wan Q/K RMSNorm and 3D RoPE through the selected exact path."""

    normalized_precision = str(precision).strip().casefold()
    if normalized_precision not in {"fp32", "fp64"}:
        raise ValueError("Wan RoPE precision must be 'fp32' or 'fp64'")
    if (fused_table is None) != (fused_grid is None):
        raise ValueError("Wan fused RoPE table and grid must be provided together")
    if fused_table is not None and fused_grid is not None:
        runtime_state = getattr(
            attention,
            "_worldfoundry_fused_rope_runtime",
            None,
        )

        def fused_call() -> tuple[torch.Tensor, torch.Tensor]:
            return hidden_qk_rmsnorm_rope_3d(
                q,
                k,
                attention.norm_q.weight,
                attention.norm_k.weight,
                fused_table,
                num_heads=attention.num_heads,
                grid_size=fused_grid,
                eps=attention.norm_q.eps,
            )

        if runtime_state is None:
            return fused_call()
        if torch.compiler.is_compiling():
            from ....optimizations.fused_rope import (
                record_compiled_fused_rope_graph_trace,
            )

            record_compiled_fused_rope_graph_trace(runtime_state)
            return fused_call()
        receipt: dict[str, Any] = {}
        with kernel_dispatch_receipt_scope(receipt):
            output = fused_call()
        runtime_state.record_eager_dispatch(receipt)
        return output
    q = attention.norm_q(q)
    k = attention.norm_k(k)
    compute_dtype = (
        torch.float32 if normalized_precision == "fp32" else torch.float64
    )
    return (
        rope_apply(q, freqs, attention.num_heads, compute_dtype=compute_dtype),
        rope_apply(k, freqs, attention.num_heads, compute_dtype=compute_dtype),
    )


class SelfAttention(nn.Module):
    """Wan self-attention with RMSNorm Q/K, 3D RoPE, and a swappable processor."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        """Build QKV/O projections and install the default self-attention processor.

        Args:
            dim: Transformer hidden size.
            num_heads: Attention heads; ``dim`` must be divisible by this.
            eps: RMSNorm epsilon for query/key.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.processor = SelfAttentionProcessor()

    def set_processor(self, processor: Any) -> None:
        """Install a model-specific self-attention policy."""

        self.processor = processor

    def get_processor(self) -> Any:
        """Return the currently installed self-attention processor."""
        return self.processor

    def forward(self, x, freqs, **kwargs: Any):
        """Dispatch to ``processor`` so variants can swap RoPE / cache / linear attn.

        Args:
            x: Hidden states ``[B, L, C]``.
            freqs: 3D RoPE frequencies aligned with the patch grid.
            **kwargs: Forwarded to research processors (action, camera, KV cache).
        """
        return self.processor(self, x, freqs, **kwargs)


def project_wan_cross_kv(
    attention: "CrossAttention", context: torch.Tensor, *, image: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized keys and values, preserving the dense call order."""
    norm_key = attention.norm_k_img if image else attention.norm_k
    fusion = getattr(attention, "_worldfoundry_cross_kv_fusion", None)
    if fusion is not None:
        key, value = fusion.project(attention, context, image=image)
        return norm_key(key), value
    key, value = (attention.k_img, attention.v_img) if image else (attention.k, attention.v)
    return norm_key(key(context)), value(context)


class CrossAttentionProcessor:
    """Default Wan cross-attention policy, replaceable by research adapters."""

    def __call__(
        self,
        attention: "CrossAttention",
        x: torch.Tensor,
        context: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if attention.has_image_input:
            image_context = context[:, :257]
            text_context = context[:, 257:]
        else:
            text_context = context
        query = attention.norm_q(attention.q(x))
        key, value = project_wan_cross_kv(attention, text_context)
        output = attention.attn(query, key, value)
        if attention.has_image_input:
            image_key, image_value = project_wan_cross_kv(attention, image_context, image=True)
            output = output + attention.attn(query, image_key, image_value)
        return attention.o(output)


class CrossAttention(nn.Module):
    """Text (and optional CLIP-image) cross-attention with a swappable processor."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        """Build text KV, and optional image KV, then install the default processor.

        Args:
            dim: Transformer hidden size.
            num_heads: Attention heads; ``dim`` must be divisible by this.
            eps: RMSNorm epsilon for query/key.
            has_image_input: If true, attend separately to the first 257 CLIP tokens.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.processor = CrossAttentionProcessor()

    def set_processor(self, processor: Any) -> None:
        """Install a model-specific cross-attention policy."""

        self.processor = processor

    def get_processor(self) -> Any:
        """Return the currently installed cross-attention processor."""
        return self.processor

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any):
        """Dispatch to ``processor`` so variants can cache KV or add extra streams.

        Args:
            x: Query hidden states ``[B, L, C]``.
            y: Condition tokens (text, or CLIP-image + text when I2V).
            **kwargs: Forwarded to research processors.
        """
        return self.processor(self, x, y, **kwargs)


class GateModule(nn.Module):
    """Residual gate used by AdaLN: ``x + gate * residual``."""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual, *, inplace: bool = False):
        """Add a timestep-gated residual to ``x``.

        Args:
            x: Residual stream.
            gate: Broadcastable AdaLN gate (MSA or MLP).
            residual: Attention or FFN output.
            inplace: Reuse ``x`` storage during inference.  Autograd always
                keeps the functional path even if an internal caller requests
                in-place execution.
        """
        if inplace and not torch.is_grad_enabled():
            return residual_gate_add_(x, residual, gate)
        return residual_gate_add(x, residual, gate)


class DiTBlock(nn.Module):
    """One Wan transformer block: gated self-attn, cross-attn, then gated FFN.

    ``return_partial`` / ``run_remaining`` split the block after cross-attn
    so TeaCache-style controllers can reuse the FFN half or inject
    alternate MLP modifiers.
    """

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        vsa_gate_compress: bool = False,
    ):
        """Build self-attn, cross-attn, FFN, and the six-channel AdaLN table.

        Args:
            has_image_input: Enable CLIP-image KV on the cross-attention module.
            dim: Transformer hidden size.
            num_heads: Attention heads.
            ffn_dim: Feed-forward inner width.
            eps: Norm epsilon.
            vsa_gate_compress: Construct the checkpoint-trained FastVideo VSA
                compression projection on self-attention.  This must only be
                enabled after a complete QAT gate checkpoint was detected.
        """
        super().__init__()
        if not isinstance(vsa_gate_compress, bool):
            raise TypeError("vsa_gate_compress must be a bool")
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        if vsa_gate_compress:
            self.self_attn.gate_compress = nn.Linear(dim, dim, bias=True)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context=None,
        t_mod=None,
        freqs=None,
        *,
        return_partial: bool = False,
        run_remaining: bool = False,
        modifiers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        """Run self-attn → cross-attn → FFN, or only the FFN half when resuming.

        Args:
            x: Hidden states ``[B, L, C]``.
            context: Cross-attention condition tokens.
            t_mod: Timestep modulation, ``[B, 6, C]`` or per-token ``[B, L, 6, C]``.
            freqs: 3D RoPE frequencies.
            return_partial: If true, stop after cross-attn and return MLP modifiers.
            run_remaining: If true, apply only the FFN half with ``modifiers``.
            modifiers: Optional ``(shift_mlp, scale_mlp, gate_mlp)`` override.
            **kwargs: Internal sparse/RoPE metadata is routed only to self-attn;
                all remaining values are forwarded to the cross-attn processor.
        """
        inplace_residual = kwargs.pop("_worldfoundry_inplace_residual", False)
        if not isinstance(inplace_residual, bool):
            raise TypeError("Wan internal inplace-residual flag must be a bool")
        inplace_residual = inplace_residual and not torch.is_grad_enabled()
        if run_remaining:
            if modifiers is None:
                raise ValueError("Wan block modifiers are required for run_remaining")
            shift_mlp, scale_mlp, gate_mlp = modifiers
            input_x = layer_norm_scale_shift(
                x,
                scale_mlp,
                shift_mlp,
                eps=self.norm2.eps,
            )
            return self.gate(
                x,
                gate_mlp,
                self.ffn(input_x),
                inplace=inplace_residual,
            )

        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = layer_norm_scale_shift(
            x,
            scale_msa,
            shift_msa,
            eps=self.norm1.eps,
        )
        self_attention_kwargs = {}
        for key in _SELF_ATTENTION_INTERNAL_KWARGS:
            if key in kwargs:
                self_attention_kwargs[key] = kwargs.pop(key)
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(input_x, freqs, **self_attention_kwargs),
            inplace=inplace_residual,
        )
        cross_attn_out = self.cross_attn(
            self.norm3(x),
            context,
            **kwargs,
        )
        x = x.add_(cross_attn_out) if inplace_residual else x + cross_attn_out
        if return_partial:
            return x, (shift_mlp, scale_mlp, gate_mlp)
        if modifiers is not None:
            shift_mlp, scale_mlp, gate_mlp = modifiers
        input_x = layer_norm_scale_shift(
            x,
            scale_mlp,
            shift_mlp,
            eps=self.norm2.eps,
        )
        x = self.gate(
            x,
            gate_mlp,
            self.ffn(input_x),
            inplace=inplace_residual,
        )
        return x

    def forward_partial(self, *args: Any, **kwargs: Any):
        """Run through cross-attn and return ``(hidden_states, mlp_modifiers)``."""
        return self.forward(*args, **kwargs, return_partial=True)

    def forward_with_phase_cache(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        *,
        cached_phases: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run or replay the three raw phases used by Wan BlockTaylorSeer.

        A cached call intentionally recomputes only the current timestep
        modulation.  It applies today's ``gate_msa``/``gate_mlp`` to Taylor-
        predicted raw self-attention and FFN outputs, matching pinned
        LightX2V.  Shift/scale terms are computed as part of modulation but
        are needed only on dense attention/FFN calls.
        """

        inplace_residual = kwargs.pop("_worldfoundry_inplace_residual", False)
        if not isinstance(inplace_residual, bool):
            raise TypeError("Wan internal inplace-residual flag must be a bool")
        inplace_residual = inplace_residual and not torch.is_grad_enabled()

        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )

        phase_names = {"self_attn_out", "cross_attn_out", "ffn_out"}
        if cached_phases is not None:
            if set(cached_phases) != phase_names:
                raise ValueError(
                    "Wan BlockTaylorSeer requires self_attn_out, "
                    "cross_attn_out, and ffn_out"
                )
            for name, value in cached_phases.items():
                if not isinstance(value, torch.Tensor) or value.shape != x.shape:
                    raise ValueError(
                        f"Wan BlockTaylorSeer cached {name} must match hidden shape"
                    )
            self_attn_out = cached_phases["self_attn_out"]
        else:
            input_x = layer_norm_scale_shift(
                x,
                scale_msa,
                shift_msa,
                eps=self.norm1.eps,
            )
            self_attention_kwargs = {}
            for key in _SELF_ATTENTION_INTERNAL_KWARGS:
                if key in kwargs:
                    self_attention_kwargs[key] = kwargs.pop(key)
            self_attn_out = self.self_attn(
                input_x,
                freqs,
                **self_attention_kwargs,
            )
        x = self.gate(
            x,
            gate_msa,
            self_attn_out,
            inplace=inplace_residual,
        )

        if cached_phases is None:
            cross_attn_out = self.cross_attn(
                self.norm3(x),
                context,
                **kwargs,
            )
        else:
            cross_attn_out = cached_phases["cross_attn_out"]
        x = x.add_(cross_attn_out) if inplace_residual else x + cross_attn_out

        if cached_phases is None:
            input_x = layer_norm_scale_shift(
                x,
                scale_mlp,
                shift_mlp,
                eps=self.norm2.eps,
            )
            ffn_out = self.ffn(input_x)
        else:
            ffn_out = cached_phases["ffn_out"]
        x = self.gate(
            x,
            gate_mlp,
            ffn_out,
            inplace=inplace_residual,
        )
        return x, {
            "self_attn_out": self_attn_out,
            "cross_attn_out": cross_attn_out,
            "ffn_out": ffn_out,
        }

    def forward_remaining(
        self,
        x: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """Resume the FFN half after a TeaCache-style partial forward."""
        return self.forward(
            x,
            run_remaining=True,
            modifiers=(shift_mlp, scale_mlp, gate_mlp),
        )


class MLP(torch.nn.Module):
    """CLIP image-token projector used by Wan 2.1 I2V / FLF2V."""

    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        """Build LayerNorm → Linear → GELU → Linear → LayerNorm.

        Args:
            in_dim: CLIP feature width (officially 1280).
            out_dim: DiT hidden size.
            has_pos_emb: If true, add the 514-token first/last-frame position table.
        """
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        """Project CLIP tokens to DiT width, optionally adding FLF position embeddings.

        Args:
            x: CLIP features ``[B, L, 1280]``.
        """
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """AdaLN-modulated output projection from tokens back to patch voxels."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        """Build the unmodulated LayerNorm and the ``C_out * prod(patch)`` linear.

        Args:
            dim: Transformer hidden size.
            out_dim: VAE latent channels written after unpatchify.
            patch_size: ``(t, h, w)`` patch used to expand each token.
            eps: LayerNorm epsilon.
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        """Apply shift/scale from ``t_mod`` and project tokens to packed voxels.

        Args:
            x: Hidden states ``[B, L, C]``.
            t_mod: Time embedding before the 6-way split (``[B, C]`` or ``[B, L, C]``).
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            head_dtype = self.head.weight.dtype
            if len(t_mod.shape) == 3:
                shift, scale = (
                    self.modulation.unsqueeze(0).to(
                        device=t_mod.device,
                        dtype=torch.float32,
                    )
                    + t_mod.float().unsqueeze(2)
                ).chunk(2, dim=2)
                x = self.head(
                    (
                        self.norm(x.float()) * (1 + scale.squeeze(2))
                        + shift.squeeze(2)
                    ).to(dtype=head_dtype)
                )
            else:
                shift, scale = (
                    self.modulation.to(device=t_mod.device, dtype=torch.float32)
                    + t_mod.float()
                ).chunk(2, dim=1)
                x = self.head(
                    (self.norm(x.float()) * (1 + scale) + shift).to(dtype=head_dtype)
                )
        return x


class WanModel(torch.nn.Module):
    """Canonical packed-tensor Wan 2.1 / 2.2 DiT.

    Subclass this (or pass ``block_class``) to add action, camera, VACE
    injection, causal attention, or linear-attention processors without
    copying patchify / RoPE / timestep / unpatchify.
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        inject_sample_info: bool = False,
        per_token_timestep: bool = False,
        vsa_gate_compress: bool = False,
        diffusion_model_pretrained_path: str | None = None,
        block_class: type[nn.Module] | None = None,
        block_kwargs: dict[str, Any] | None = None,
    ):
        """Build embeddings, the DiT stack, optional I2V / camera extras, and 3D RoPE.

        Args:
            dim: Transformer hidden size.
            in_dim: Input latent channels (noisy video, optionally plus VAE cond).
            ffn_dim: Feed-forward inner width.
            out_dim: Output latent channels after unpatchify.
            text_dim: UMT5 (or equivalent) token width before projection.
            freq_dim: Sinusoidal timestep embedding width.
            eps: Norm epsilon.
            patch_size: 3D patch ``(t, h, w)``.
            num_heads: Attention heads.
            num_layers: Number of :class:`DiTBlock` layers.
            has_image_input: Enable CLIP image tokens and image KV.
            has_image_pos_emb: Add FLF2V position embeddings on CLIP tokens.
            has_ref_conv: Extra 2D conv used by some reference-conditioned weights.
            add_control_adapter: Attach :class:`SimpleAdapter` at construction.
            in_dim_control_adapter: Camera / control latent channels.
            seperated_timestep: Kept for checkpoint compatibility (legacy flag).
            require_vae_embedding: If true, I2V must concatenate ``y`` onto ``x``.
            require_clip_embedding: If true, I2V recipes expect CLIP features.
            fuse_vae_embedding_in_latents: Recipe flag; fusion happens upstream.
            inject_sample_info: Add the official 2-way FPS embedding / projection.
            per_token_timestep: Accept ``timestep`` of shape ``[B, L]``.
            vsa_gate_compress: Construct one checkpoint-trained FastVideo VSA
                gate per block.  Loaders set this only after validating every
                gate weight/bias tensor in the checkpoint header.
            diffusion_model_pretrained_path: Optional path recorded for loaders.
            block_class: Override the per-layer block (action / VACE / etc.).
            block_kwargs: Extra kwargs forwarded to each ``block_class``.
        """
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.inject_sample_info = bool(inject_sample_info)
        self.per_token_timestep = bool(per_token_timestep)
        if not isinstance(vsa_gate_compress, bool):
            raise TypeError("vsa_gate_compress must be a bool")
        self.vsa_gate_compress = vsa_gate_compress
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        if self.inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        block_class = block_class or DiTBlock
        block_kwargs = dict(block_kwargs or {})
        if "vsa_gate_compress" in block_kwargs:
            raise ValueError(
                "WanModel.block_kwargs cannot override checkpoint-derived "
                "vsa_gate_compress"
            )
        if vsa_gate_compress:
            block_kwargs["vsa_gate_compress"] = True
        self.blocks = nn.ModuleList(
            [
                block_class(
                    has_image_input,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    **block_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self._rope_freq_cache: dict[
            tuple[int, int, int, str, int | None, str],
            torch.Tensor,
        ] = {}
        self._fused_rope_table_cache: dict[
            tuple[str, int | None, str],
            torch.Tensor,
        ] = {}

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:]
            )
        else:
            self.control_adapter = None

    def set_attention_compatibility_mode(self, enabled: bool) -> None:
        """Select exact PyTorch SDPA for this model instance.

        The shared attention dispatcher reads its default backend at import
        time.  Training must therefore bind the correctness path on the
        actual Wan modules instead of relying on a late environment change.
        """

        if not isinstance(enabled, bool):
            raise TypeError("Wan attention compatibility mode must be a bool")
        for module in self.modules():
            if isinstance(module, AttentionModule):
                module.compatibility_mode = enabled

    def set_rms_norm_precision(self, precision: str) -> int:
        """Configure every Wan RMSNorm reduction and return the module count."""

        normalized = str(precision).strip().casefold()
        if normalized not in {"fp32", "input"}:
            raise ValueError("Wan rms_norm_precision must be 'fp32' or 'input'")
        configured = 0
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.set_compute_mode(normalized)
                configured += 1
        self._worldfoundry_rms_norm_precision = normalized
        self._worldfoundry_rms_norm_modules = configured
        return configured

    def enable_control_adapter(self, in_dim: int = 24) -> None:
        """Attach the canonical Wan spatial control role after base loading.

        Some releases publish the adapter only in an overlay checkpoint, so it
        cannot participate in strict restoration of the official base DiT.
        """

        if self.control_adapter is not None:
            return
        reference = self.patch_embedding.weight
        self.control_adapter = SimpleAdapter(
            in_dim,
            self.dim,
            kernel_size=self.patch_size[1:],
            stride=self.patch_size[1:],
        ).to(device=reference.device, dtype=reference.dtype)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        """Conv3d-patchify ``x`` and optionally add the camera-control residual.

        Args:
            x: Latents ``[B, C, F, H, W]``.
            control_camera_latents_input: Optional adapter input, same spatial layout.
        """
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """Fold packed tokens back to ``[B, C, F, H, W]`` using ``patch_size``.

        Args:
            x: Head output ``[B, F*H*W, C_out * prod(patch)]``.
            grid_size: Patch grid ``(F, H, W)``.
        """
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def block_forward_kwargs(
        self,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return variant-specific keyword arguments for every transformer block.

        Wan research variants can override this hook to add conditioning without
        copying the common patchification, RoPE, timestep, and output path.
        """

        del grid_size, kwargs
        return {}

    def prepare_condition_context(
        self,
        context: torch.Tensor | None,
        *,
        clip_feature: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Project the cross-attention condition used by the shared Wan blocks.

        Most Wan releases use UMT5 (and optionally CLIP) through the canonical
        projection below.  Research variants whose checkpoint already emits
        DiT-width condition tokens can override this hook without copying the
        patchification, timestep, RoPE, transformer, and output path.
        """

        del kwargs
        if context is None:
            raise ValueError("Wan text-conditioned inference requires a context tensor")
        cross_kv_cache = getattr(self, "_worldfoundry_static_cross_kv", None)
        if cross_kv_cache is not None:
            cached_context = cross_kv_cache.get_condition_context(context, clip_feature)
            if cached_context is not None:
                return cached_context

        raw_context = context
        context = self.text_embedding(raw_context)
        if self.has_image_input:
            if clip_feature is None:
                raise ValueError("Wan image-conditioned inference requires clip_feature")
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)
        if cross_kv_cache is not None:
            cross_kv_cache.put_condition_context(raw_context, clip_feature, context)
        return context

    def rotary_frequencies(
        self,
        grid_size: tuple[int, int, int],
        *,
        device: torch.device,
        precision: str = "fp64",
    ) -> torch.Tensor:
        """Build and memoize canonical 3D rotary frequencies for one patch grid.

        The assembled table is invariant across denoise steps. Keeping it on
        the target device avoids repeating the expand/cat/reshape/device-copy
        sequence for every CFG branch and every step. The small bounded cache
        supports dynamic-resolution servers without retaining unbounded VRAM.
        """

        normalized = str(precision).strip().casefold()
        if normalized not in {"fp32", "fp64"}:
            raise ValueError("Wan RoPE precision must be 'fp32' or 'fp64'")
        f, h, w = grid_size
        key = (f, h, w, device.type, device.index, normalized)
        cache = self._rope_freq_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        if len(cache) >= 8:
            cache.clear()
        assembled = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(
                device=device,
                dtype=(
                    torch.complex64
                    if normalized == "fp32"
                    else torch.complex128
                ),
            )
        )
        cache[key] = assembled
        return assembled

    def prepare_token_sequence(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        t: torch.Tensor,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Variant hook for prefix/suffix conditioning token composition."""

        del kwargs
        return x, freqs, t_mod, t, None

    def fused_rope_table(
        self,
        *,
        device: torch.device,
        precision: str,
    ) -> torch.Tensor:
        """Return the packed T/H/W base table consumed by fused 3D RoPE."""

        normalized = str(precision).strip().lower()
        if normalized not in {"fp32", "fp64"}:
            raise ValueError("Wan fused RoPE precision must be 'fp32' or 'fp64'")
        key = (device.type, device.index, normalized)
        cached = self._fused_rope_table_cache.get(key)
        if cached is not None:
            return cached
        table = torch.cat(self.freqs, dim=-1).to(
            device=device,
            dtype=torch.complex64 if normalized == "fp32" else torch.complex128,
        )
        # The Triton contract is an explicit cosine/sine table.  Returning a
        # real fp32 view makes the accelerated path reachable through the
        # ordinary dtype capability gate; the fp64 table intentionally stays
        # complex and therefore uses the exact chunked PyTorch implementation.
        if normalized == "fp32":
            table = torch.view_as_real(table)
        self._fused_rope_table_cache[key] = table
        return table

    def finalize_token_sequence(
        self,
        x: torch.Tensor,
        token_state: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Variant hook for extracting auxiliary and target token branches."""

        del token_state, kwargs
        return x, None

    def after_transformer_block(
        self,
        x: torch.Tensor,
        block_id: int,
        token_state: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Variant hook for residual controllers applied between Wan blocks."""

        del block_id, token_state, kwargs
        return x

    def project_output(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Project target tokens through the checkpoint's output head."""

        del kwargs
        return self.head(x, t)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        memory_context=None,
        fps: Optional[torch.Tensor] = None,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        feature_cache: Any = None,
        feature_cache_step: int = 0,
        feature_cache_total_steps: int | None = None,
        **kwargs,
    ):
        """Denoise packed latents: timestep → condition → patchify → blocks → head.

        Control flow
        ------------
        1. Embed ``timestep`` (scalar or per-token) and optional FPS ids.
        2. Project text / CLIP through :meth:`prepare_condition_context`.
        3. Optionally concat VAE condition ``y`` on the channel axis.
        4. Patchify (plus camera adapter), then optional ``memory_adapter``.
        5. :meth:`prepare_token_sequence` may prepend reference / action tokens.
        6. Each block gets :meth:`block_forward_kwargs`; after each block,
           :meth:`after_transformer_block` can inject controllers.
        7. :meth:`finalize_token_sequence` strips prefix tokens; head + unpatchify.

        Args:
            x: Noisy latents ``[B, C, F, H, W]``.
            timestep: Diffusion step, ``[B]`` or ``[B, L]`` when per-token.
            context: Text tokens before or after the condition hook.
            clip_feature: Optional CLIP image tokens for I2V.
            y: Optional VAE condition latents concatenated onto ``x``.
            use_gradient_checkpointing: Recompute blocks in backward.
            use_gradient_checkpointing_offload: Same, with CPU activation stash.
            memory_context: Optional native memory-adapter state.
            fps: Optional 2-way FPS ids when ``inject_sample_info`` is set.
            control_camera_latents_input: Optional :class:`SimpleAdapter` input.
            **kwargs: Forwarded to variant hooks and block processors.
        """
        inplace_residual = kwargs.pop("_worldfoundry_inplace_residual", False)
        if not isinstance(inplace_residual, bool):
            raise TypeError("Wan internal inplace-residual flag must be a bool")
        # This second guard protects direct WanModel callers.  The public
        # denoiser only supplies the flag for no-grad, cache-free inference,
        # but feature caches retain block inputs to form residuals and must
        # never observe those tensors after an in-place mutation.
        inplace_residual = bool(
            inplace_residual
            and not torch.is_grad_enabled()
            and feature_cache is None
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Native Wan checkpoints can be loaded directly in BF16 (SCOPE
            # does this to fit the full 14B model).  Hard-casting the
            # sinusoidal features to FP32 while autocast is explicitly
            # disabled makes the first Linear fail with Float/BFloat16.  Use
            # the actual embedding weight dtype so both FP32 and BF16 model
            # loading remain valid.
            time_dtype = next(self.time_embedding.parameters()).dtype
            if timestep.ndim == 1:
                t = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, timestep).to(time_dtype)
                )
                t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            elif timestep.ndim == 2 and self.per_token_timestep:
                batch, sequence = timestep.shape
                t = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(time_dtype)
                ).unflatten(0, (batch, sequence))
                t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
            else:
                raise ValueError(
                    "Wan timestep must be [B], or [B,L] when per-token timesteps are enabled"
                )
        if self.inject_sample_info:
            if fps is None:
                raise ValueError("Wan sample-info conditioning requires fps ids")
            fps_ids = fps.to(device=x.device, dtype=torch.long).reshape(-1)
            if fps_ids.numel() == 1 and x.shape[0] != 1:
                fps_ids = fps_ids.expand(x.shape[0])
            if fps_ids.numel() != x.shape[0]:
                raise ValueError("Wan fps ids must be scalar or have one value per sample")
            fps_features = self.fps_embedding(fps_ids)
            fps_dtype = next(self.fps_projection.parameters()).dtype
            fps_mod = self.fps_projection(fps_features.to(fps_dtype)).unflatten(1, (6, self.dim))
            t_mod = t_mod + fps_mod
        context = self.prepare_condition_context(
            context,
            clip_feature=clip_feature,
            **kwargs,
        )

        # Wan2.1 I2V uses both CLIP and VAE conditions, while Wan2.2 Fun I2V
        # keeps only the VAE branch.  Treat those roles independently so a
        # native model does not need a second DiT implementation merely to
        # omit CLIP.
        if y is not None and self.require_vae_embedding:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        elif self.has_image_input and self.require_vae_embedding:
            raise ValueError("Wan image-conditioned inference requires VAE condition latents")

        x, (f, h, w) = self.patchify(
            x,
            control_camera_latents_input=control_camera_latents_input,
        )

        memory_adapter = getattr(self, "memory_adapter", None)
        if memory_adapter is not None:
            x, context, memory_context = memory_adapter.prepare_inputs(
                x=x,
                context=context,
                grid_size=(f, h, w),
                memory_context=memory_context,
            )

        rope_precision = str(
            getattr(self, "_worldfoundry_rope_precision", "fp64")
        ).strip().casefold()
        freqs = self.rotary_frequencies(
            (f, h, w),
            device=x.device,
            precision=rope_precision,
        )
        x, freqs, t_mod, t, token_state = self.prepare_token_sequence(
            x,
            freqs,
            t_mod,
            t,
            (f, h, w),
            context=context,
            **kwargs,
        )

        sequence_parallel_state = getattr(
            self,
            "_worldfoundry_sequence_parallel",
            None,
        )
        sequence_parallel_length = x.shape[1]
        if sequence_parallel_state is not None:
            from worldfoundry.core.distributed.sequence_parallel.runtime import (
                sequence_parallel_chunk,
            )

            degree = int(sequence_parallel_state.sp_degree)
            padding = (-sequence_parallel_length) % degree
            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, padding))
                freqs = torch.cat(
                    [
                        freqs,
                        torch.ones(
                            padding,
                            *freqs.shape[1:],
                            dtype=freqs.dtype,
                            device=freqs.device,
                        ),
                    ],
                    dim=0,
                )
                if t_mod.ndim == 4:
                    t_mod = torch.nn.functional.pad(t_mod, (0, 0, 0, 0, 0, padding))
            x = sequence_parallel_chunk(x, dim=1)
            if t_mod.ndim == 4:
                t_mod = sequence_parallel_chunk(t_mod, dim=1)

        block_kwargs = self.block_forward_kwargs(
            (f, h, w),
            **kwargs,
        )
        block_kwargs["_worldfoundry_rope_precision"] = rope_precision
        if inplace_residual:
            block_kwargs["_worldfoundry_inplace_residual"] = True
        if getattr(self, "_worldfoundry_approximate_attention", None) is not None:
            block_kwargs["_worldfoundry_sparse_grid"] = (f, h, w)
        if bool(getattr(self, "_worldfoundry_fused_rope", False)):
            block_kwargs["_worldfoundry_rope_grid"] = (f, h, w)
            block_kwargs["_worldfoundry_rope_table"] = self.fused_rope_table(
                device=x.device,
                precision=str(
                    getattr(self, "_worldfoundry_rope_precision", "fp32")
                ),
            )
        if memory_context is not None:
            block_kwargs["memory_context"] = memory_context

        probe_observer = getattr(feature_cache, "observe_probe", None)
        decisive_block = len(self.blocks) // 2

        def run_transformer_block(
            block_id: int,
            hidden: torch.Tensor,
        ) -> torch.Tensor:
            previous_hidden = hidden
            block = self.blocks[block_id]
            block_inputs = (hidden, context, t_mod, freqs)
            if self.training:
                hidden = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    *block_inputs,
                    **block_kwargs,
                )
            else:
                hidden = block(*block_inputs, **block_kwargs)
            hidden = self.after_transformer_block(
                hidden,
                block_id,
                token_state,
                **kwargs,
            )
            if callable(probe_observer) and block_id == decisive_block:
                probe_observer(hidden - previous_hidden)
            return hidden

        def run_transformer_blocks(block_input: torch.Tensor) -> torch.Tensor:
            hidden = block_input
            for block_id in range(len(self.blocks)):
                hidden = run_transformer_block(block_id, hidden)
            return hidden

        def run_taylor_phase_block(block_id, hidden, cached_phases):
            """Execute/replay the raw phases required by BlockTaylorSeer."""

            block = self.blocks[block_id]
            phase_forward = getattr(block, "forward_with_phase_cache", None)
            if not callable(phase_forward):
                raise TypeError(
                    "Wan BlockTaylorSeer requires blocks implementing "
                    "forward_with_phase_cache"
                )
            hidden, phase_outputs = phase_forward(
                hidden,
                context,
                t_mod,
                freqs,
                cached_phases=cached_phases,
                **block_kwargs,
            )
            hidden = self.after_transformer_block(
                hidden,
                block_id,
                token_state,
                **kwargs,
            )
            return hidden, phase_outputs

        if feature_cache is None:
            x = run_transformer_blocks(x)
        else:
            block_input = x
            phase_cache_runner = getattr(feature_cache, "run_phase_blocks", None)
            block_cache_runner = getattr(feature_cache, "run_blocks", None)
            if callable(phase_cache_runner):
                x = phase_cache_runner(
                    int(feature_cache_step),
                    block_input,
                    run_transformer_block,
                    run_taylor_phase_block,
                    block_count=len(self.blocks),
                    total_steps=feature_cache_total_steps,
                )
            elif callable(block_cache_runner):
                x = block_cache_runner(
                    int(feature_cache_step),
                    block_input,
                    run_transformer_block,
                    block_count=len(self.blocks),
                    total_steps=feature_cache_total_steps,
                )
            else:
                signal_kind = str(
                    getattr(feature_cache, "signal_kind", "timestep-modulation")
                )
                if signal_kind == "time-embedding":
                    cache_signal = t
                elif signal_kind == "timestep-modulation":
                    cache_signal = t_mod
                else:
                    raise ValueError(
                        f"unsupported Wan feature-cache signal kind: {signal_kind!r}"
                    )
                residual = feature_cache.run(
                    int(feature_cache_step),
                    cache_signal,
                    lambda: run_transformer_blocks(block_input) - block_input,
                    total_steps=feature_cache_total_steps,
                )
                x = block_input + residual

        if sequence_parallel_state is not None:
            from worldfoundry.core.distributed.sequence_parallel.runtime import (
                sequence_parallel_all_gather,
            )

            x = sequence_parallel_all_gather(x, dim=1)
            x = x[:, :sequence_parallel_length]

        x, auxiliary = self.finalize_token_sequence(x, token_state, **kwargs)
        x = self.project_output(x, t, **kwargs)
        x = self.unpatchify(x, (f, h, w))
        return (x, auxiliary) if auxiliary is not None else x

    @staticmethod
    def state_dict_converter():
        from ...denoisers.wan import WanModelStateDictConverter

        return WanModelStateDictConverter()

    @property
    def attn_processors(self) -> dict[str, Any]:
        """Return replaceable cross-attention processors by module path."""

        return {
            f"blocks.{index}.cross_attn.processor": block.cross_attn.get_processor()
            for index, block in enumerate(self.blocks)
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "get_processor")
        }

    def set_attn_processor(self, processor: Any) -> None:
        """Install one processor everywhere or a path-keyed processor mapping."""

        if isinstance(processor, dict):
            expected = set(self.attn_processors)
            provided = set(processor)
            if provided != expected:
                missing = sorted(expected - provided)
                unexpected = sorted(provided - expected)
                raise ValueError(
                    f"Wan attention processor paths mismatch; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            for index, block in enumerate(self.blocks):
                path = f"blocks.{index}.cross_attn.processor"
                if path in processor:
                    block.cross_attn.set_processor(processor[path])
            return
        for block in self.blocks:
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "set_processor"):
                block.cross_attn.set_processor(processor)

# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-only Cosmos3 omni Mixture-of-Tokens transformer.

Checkpoint-shaped joint sequence model: text (causal understanding prefix)
plus generation tokens for vision / optional sound / optional action share
one packed stream.  Vision latents enter as per-item ``C T H W`` cubes,
are spatially patchified to ``(T · H' · W', p*p*C)``, then scattered into
the joint ``[S, hidden]`` buffer.  Sound is ``[C, T]`` packed to
``[T, C]``; action is ``[T, D]``.  Dual-pathway attention keeps und
tokens causal and lets gen tokens attend to und+gen.

Sampling and packing of ``position_ids`` live in the Cosmos3 denoiser /
recipe.  Do not introduce runner logic here.
"""

import math
import weakref
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
import torch.nn as nn

from worldfoundry.core.attention import rotate_half as _rotate_half
from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.core.kernels import silu_and_mul, silu_mul
from worldfoundry.core.nn import DomainAwareLinear
from worldfoundry.core.nn.checkpointing.gradient import gradient_checkpoint_forward
from worldfoundry.core.nn.diffusion.timestep import TimestepEmbedding, Timesteps

from .fused_ops import cosmos3_fa3_attention, fused_residual_add_rmsnorm, fused_rmsnorm_rope_gqa


def _module_param_device(module: nn.Module) -> torch.device | None:
    """Resolve the compute device of ``module``, honoring layerwise-offload wrappers.

    Offload wrappers keep backing parameters on CPU while materializing a
    temporary computation module on the accelerator.  Prefer that execution
    target over the backing parameter's storage device.
    """
    for candidate in module.modules():
        computation_device = getattr(candidate, "computation_device", None)
        if computation_device is not None:
            return torch.device(computation_device)
    parameter = next(module.parameters(), None)
    return None if parameter is None else parameter.device


def _to_device(value, device: torch.device | None):
    """Move tensors (and nested tuples/lists) onto ``device`` without recopying."""
    if device is None:
        return value
    if torch.is_tensor(value):
        return value if value.device == device else value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


class Cosmos3AttnProcessor:
    """Dual-pathway attention processor for Cosmos3.

    Projects, normalizes, applies rotary position embeddings, then runs separate causal (understanding) and full
    (generation) attention pathways. The generation pathway cross-attends to both und and gen keys/values.

    Supports SOL-Engine optimizations:
    - Exact lossless ``cached_und_kv`` reuse (skips 100% of und projections, norms, RoPE, causal attention, and output).
    - Hopper ``flash_attention_3`` GQA dispatch (32Q/8KV & 64Q/8KV, zero transpose).
    - Fused ``gen_qkv_proj`` and Triton ``fused_rmsnorm_rope_gqa``.
    """

    def __init__(self, *, attention_backend: str = "torch", use_fused_ops: bool = False) -> None:
        self.attention_backend = attention_backend
        self.use_fused_ops = use_fused_ops
        self.fa3_calls: int = 0
        self.sdpa_calls: int = 0

    def _run_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        if (
            self.attention_backend == "flash_attention_3"
            and not torch.is_grad_enabled()
            and q.is_cuda
            and q.dtype in {torch.bfloat16, torch.float16}
            and k.dtype == q.dtype
            and v.dtype == q.dtype
        ):
            if not torch.compiler.is_compiling():
                self.fa3_calls += 1
            return cosmos3_fa3_attention(q, k, v, causal=causal)
        if not torch.compiler.is_compiling():
            self.sdpa_calls += 1
        out = scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=causal,
            enable_gqa=True,
        )
        return out.squeeze(0).transpose(0, 1).flatten(-2, -1)

    def _project_gen_qkv(
        self,
        attn: "Cosmos3PackedMoTAttention",
        gen_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = attn.num_attention_heads * attn.head_dim
        kv_dim = attn.num_key_value_heads * attn.head_dim
        if attn.gen_qkv_proj is not None:
            qkv = attn.gen_qkv_proj(gen_seq)
            q_gen, k_gen, v_gen = torch.split(qkv, (q_dim, kv_dim, kv_dim), dim=-1)
            return (
                q_gen.view(-1, attn.num_attention_heads, attn.head_dim),
                k_gen.view(-1, attn.num_key_value_heads, attn.head_dim),
                v_gen.view(-1, attn.num_key_value_heads, attn.head_dim),
            )
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        return q_gen, k_gen, v_gen

    def __call__(
        self,
        attn: "Cosmos3PackedMoTAttention",
        und_seq: torch.Tensor | None,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        cached_und_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_und_kv: bool = False,
    ) -> (
        tuple[torch.Tensor | None, torch.Tensor]
        | tuple[torch.Tensor | None, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ):
        """Return ``(und_out, gen_out)`` (plus optional ``(k_und, v_und)``) for packed ``[S, C]`` streams."""
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb

        # Understanding pathway: skip completely when static und KV cache is supplied.
        und_out: torch.Tensor | None
        if cached_und_kv is not None:
            k_und, v_und = cached_und_kv
            und_out = None
        else:
            assert und_seq is not None
            q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
            k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            if self.use_fused_ops and not torch.is_grad_enabled() and q_und.is_cuda:
                q_und, k_und = fused_rmsnorm_rope_gqa(
                    q_und,
                    k_und,
                    attn.norm_q.weight,
                    attn.norm_k.weight,
                    cos_und,
                    sin_und,
                    eps=float(attn.norm_q.eps or 1e-6),
                )
            else:
                q_und = attn.norm_q(q_und)
                k_und = attn.norm_k(k_und)
                cos_u = cos_und.unsqueeze(1)
                sin_u = sin_und.unsqueeze(1)
                q_und = q_und * cos_u + _rotate_half(q_und) * sin_u
                k_und = k_und * cos_u + _rotate_half(k_und) * sin_u
            causal_out = self._run_attn(q_und, k_und, v_und, causal=True)
            und_out = attn.to_out(causal_out)

        # Generation pathway: project Q/K/V, QK-RMSNorm+RoPE, attend over [und, gen].
        q_gen, k_gen, v_gen = self._project_gen_qkv(attn, gen_seq)
        if self.use_fused_ops and not torch.is_grad_enabled() and q_gen.is_cuda:
            q_gen, k_gen = fused_rmsnorm_rope_gqa(
                q_gen,
                k_gen,
                attn.norm_added_q.weight,
                attn.norm_added_k.weight,
                cos_gen,
                sin_gen,
                eps=float(attn.norm_added_q.eps or 1e-6),
            )
        else:
            q_gen = attn.norm_added_q(q_gen)
            k_gen = attn.norm_added_k(k_gen)
            cos_g = cos_gen.unsqueeze(1)
            sin_g = sin_gen.unsqueeze(1)
            q_gen = q_gen * cos_g + _rotate_half(q_gen) * sin_g
            k_gen = k_gen * cos_g + _rotate_half(k_gen) * sin_g

        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = self._run_attn(q_gen, all_k, all_v, causal=False)
        gen_out = attn.to_add_out(full_out)

        if return_und_kv:
            return und_out, gen_out, (k_und, v_und)
        return und_out, gen_out


class Cosmos3VLTextRotaryEmbedding(nn.Module):
    """Interleaved 3-axis mRoPE for the joint Cosmos3 sequence (text + video + extra modalities)."""

    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim: tuple[int, int, int]):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_axes_dim = rope_axes_dim

    def apply_interleaved_mrope(self, freqs, rope_axes_dim):
        """Reorganize chunked [TTT...HHH...WWW] frequency layout into interleaved
        [THTHWHTHW...TT], preserving frequency continuity across the 3 grids."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids, device, dtype):
        """``position_ids`` is ``[3, B, N]`` (or ``[B, N]`` broadcast to 3 axes) → ``(cos, sin)``."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(device)
        )  # [3,B,head_dim//2,1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3,B,1,N]
        # Disable autocast so the position-id matmul runs in float32: under an ambient autocast it would run in
        # bfloat16, which cannot represent consecutive integers past 256, collapsing positions onto the same
        # frequency and degrading the rotary embedding.
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = inv_freq_expanded @ position_ids_expanded
        freqs = freqs.transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.rope_axes_dim)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)  # each: [B,N,head_dim]


class Cosmos3VLTextMLP(nn.Module):
    """SwiGLU FFN used independently on the understanding and generation streams."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj: nn.Module | None = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj: nn.Module | None = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_up_proj: nn.Module | None = None
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """``down(silu(gate(x)) * up(x))`` on ``[..., hidden]`` tokens."""
        if self.gate_up_proj is not None:
            return self.down_proj(silu_and_mul(self.gate_up_proj(x)))
        assert self.gate_proj is not None and self.up_proj is not None
        if not self.training and not torch.is_grad_enabled() and x.is_cuda:
            return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3PackedMoTAttention(nn.Module):
    """Dual-pathway packed attention for Qwen3VL MoT — separate projections for
    understanding (causal) and generation (full) token streams."""

    _default_processor_cls = Cosmos3AttnProcessor
    _available_processors: ClassVar[list] = [Cosmos3AttnProcessor]

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
        processor=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Understanding pathway. norm_q / norm_k are applied per-head (only on
        # head_dim), so no reshape is needed after them.
        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_q = nn.RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True)

        # Generation pathway
        self.add_q_proj: nn.Module | None = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.add_k_proj: nn.Module | None = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.add_v_proj: nn.Module | None = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.gen_qkv_proj: nn.Module | None = None
        self.to_add_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True)

        if processor is None:
            processor = self._default_processor_cls()
        self.processor = processor

    def forward(
        self,
        und_seq: torch.Tensor | None,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        cached_und_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_und_kv: bool = False,
    ):
        """Dispatch packed und/gen sequences to :class:`Cosmos3AttnProcessor`."""
        return self.processor(
            self,
            und_seq,
            gen_seq,
            rotary_emb,
            cached_und_kv=cached_und_kv,
            return_und_kv=return_und_kv,
        )


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    """
    Qwen3VL text MoT (Mixture of Tokens) decoder layer. Features dual-pathway attention for understanding vs
    generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )

        self.mlp = Cosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.mlp_moe_gen = Cosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)
        self.input_layernorm_moe_gen = nn.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)
        self.post_attention_layernorm_moe_gen = nn.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)
        self._use_fused_ops: bool = False

    def forward_gen_with_cached_und_kv(
        self,
        gen_seq: torch.Tensor,
        cos_gen: torch.Tensor,
        sin_gen: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
    ) -> torch.Tensor:
        """Execute ONLY the generation pathway reusing precomputed ``(k_und, v_und)``."""
        gen_norm = self.input_layernorm_moe_gen(gen_seq)
        _, gen_attn_out = self.self_attn(
            None,
            gen_norm,
            (cos_gen, sin_gen, cos_gen, sin_gen),
            cached_und_kv=(k_und, v_und),
        )
        if self._use_fused_ops and not torch.is_grad_enabled() and gen_seq.is_cuda:
            residual_gen, post_norm_gen = fused_residual_add_rmsnorm(
                gen_seq,
                gen_attn_out,
                self.post_attention_layernorm_moe_gen.weight,
                eps=float(self.post_attention_layernorm_moe_gen.eps or 1e-6),
            )
            return residual_gen + self.mlp_moe_gen(post_norm_gen)
        residual_gen = gen_seq + gen_attn_out
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return residual_gen + mlp_out_gen

    def forward(
        self,
        und_seq: torch.Tensor | None,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        cached_und_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_und_kv: bool = False,
    ):
        """Pre-norm dual residual: shared attention, then per-pathway SwiGLU MLPs."""
        if cached_und_kv is not None:
            _, _, cos_gen, sin_gen = rotary_emb
            gen_out = self.forward_gen_with_cached_und_kv(gen_seq, cos_gen, sin_gen, cached_und_kv[0], cached_und_kv[1])
            return None, gen_out

        assert und_seq is not None
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        attn_res = self.self_attn(und_norm, gen_norm, rotary_emb, return_und_kv=return_und_kv)
        if return_und_kv:
            und_attn_out, gen_attn_out, und_kv = attn_res
        else:
            und_attn_out, gen_attn_out = attn_res
            und_kv = None

        if self._use_fused_ops and not torch.is_grad_enabled() and gen_seq.is_cuda:
            residual_und, post_norm_und = fused_residual_add_rmsnorm(
                und_seq,
                und_attn_out,
                self.post_attention_layernorm.weight,
                eps=float(self.post_attention_layernorm.eps or 1e-6),
            )
            residual_gen, post_norm_gen = fused_residual_add_rmsnorm(
                gen_seq,
                gen_attn_out,
                self.post_attention_layernorm_moe_gen.weight,
                eps=float(self.post_attention_layernorm_moe_gen.eps or 1e-6),
            )
            und_next = residual_und + self.mlp(post_norm_und)
            gen_next = residual_gen + self.mlp_moe_gen(post_norm_gen)
        else:
            residual_und = und_seq + und_attn_out
            residual_gen = gen_seq + gen_attn_out
            mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
            mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
            und_next = residual_und + mlp_out_und
            gen_next = residual_gen + mlp_out_gen

        if return_und_kv:
            return und_next, gen_next, und_kv
        return und_next, gen_next


class Cosmos3OmniTransformer(nn.Module):
    """Checkpoint-compatible Cosmos3 inference transformer on WorldFoundry primitives."""

    _no_split_modules: ClassVar[list] = ["Cosmos3VLTextMoTDecoderLayer"]

    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        dtype: str = "bfloat16",  # required by the loader (see `ignore_for_config` above); not read here
        head_dim: int = 128,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        base_fps: int = 24,
        enable_fps_modulation: bool = True,
        latent_channel: int = 48,
        unified_3d_mrope_reset_spatial_ids: bool = True,
        unified_3d_mrope_temporal_modality_margin: int = 15000,
        latent_patch_size: int = 2,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 36,
        num_key_value_heads: int = 8,
        patch_latent_dim: int = 192,
        rms_norm_eps: float = 1e-6,
        rope_scaling: dict | None = None,
        rope_theta: float = 5000000.0,
        action_dim: int | None = None,
        action_gen: bool = False,
        num_embodiment_domains: int = 32,
        sound_dim: int | None = None,
        sound_gen: bool = False,
        sound_latent_fps: float = 25.0,
        timestep_scale: float = 0.001,
        vocab_size: int = 151936,
        **unused_config,
    ):
        super().__init__()
        del attention_dropout, dtype, unused_config

        rope_axes_dim = rope_scaling.get("mrope_section", [24, 20, 20]) if rope_scaling is not None else [24, 20, 20]
        self.config = SimpleNamespace(
            action_dim=action_dim,
            action_gen=action_gen,
            base_fps=base_fps,
            enable_fps_modulation=enable_fps_modulation,
            hidden_size=hidden_size,
            latent_channel=latent_channel,
            latent_patch_size=latent_patch_size,
            num_hidden_layers=num_hidden_layers,
            rope_axes_dim=rope_axes_dim,
            sound_dim=sound_dim,
            sound_gen=sound_gen,
            sound_latent_fps=sound_latent_fps,
            timestep_scale=timestep_scale,
            unified_3d_mrope_reset_spatial_ids=unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=unified_3d_mrope_temporal_modality_margin,
        )

        # Text-model layers live directly on the transformer, matching the
        # published Cosmos3 checkpoint layout.
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Cosmos3VLTextMoTDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self._enable_gradient_checkpointing = False
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)
        self.norm_moe_gen = nn.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)
        self.rotary_emb = Cosmos3VLTextRotaryEmbedding(
            head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes_dim
        )

        # Modality projection heads + timestep embedding.
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.proj_in = nn.Linear(patch_latent_dim, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, patch_latent_dim, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.action_gen = action_gen
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        if action_gen:
            if self.action_dim is None:
                raise ValueError("`action_dim` must be provided when `action_gen=True`.")
            self.action_proj_in = DomainAwareLinear(self.action_dim, hidden_size, self.num_embodiment_domains)
            self.action_proj_out = DomainAwareLinear(hidden_size, self.action_dim, self.num_embodiment_domains)
            self.action_modality_embed = nn.Parameter(torch.zeros(hidden_size))
        if sound_gen:
            if sound_dim is None:
                raise ValueError("`sound_dim` must be provided when `sound_gen=True`.")
            self.audio_proj_in = nn.Linear(sound_dim, hidden_size, bias=True)
            self.audio_proj_out = nn.Linear(hidden_size, sound_dim, bias=True)
            self.audio_modality_embed = nn.Parameter(torch.zeros(hidden_size))

        # SOL-Engine runtime slots (all opt-in, disabled by default).
        self._und_static_kv_cache: Any = None
        self._cosmos3_gen_cache_spec: dict[str, Any] | None = None
        self._cosmos3_gen_caches: dict[str, Any] = {}
        self._cosmos3_step_quant_policy: dict[str, Any] | None = None
        self._cosmos3_piecewise_graph: Any = None
        self._cosmos3_fa3_replay_calls: int = 0
        self._cosmos3_fp8_replay_calls: int = 0
        self._cosmos3_nvfp4_replay_calls: int = 0
        self._cosmos3_bf16_replay_calls: int = 0
        self._rotary_cache: list[
            tuple[
                weakref.ref[torch.Tensor],
                int,
                torch.dtype,
                torch.device,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []

    def reset_sol_caches(self) -> None:
        """Clear per-request static und KV, TeaCache/FirstBlock, and rotary caches."""
        if self._und_static_kv_cache is not None:
            self._und_static_kv_cache.reset()
        for cache in self._cosmos3_gen_caches.values():
            cache.reset()
        self._cosmos3_gen_caches.clear()
        self._rotary_cache.clear()

    def _accumulate_gen_tail_replay_telemetry(
        self,
        *,
        hidden_dtype: torch.dtype,
        precision_tag: str,
    ) -> None:
        """Account for FA3 and step-selective quantized linear calls inside a replayed gen-tail CUDA Graph."""
        from .acceleration import StepSelectiveFloat8Linear, StepSelectiveNVFP4Linear

        low_prec_dtype = hidden_dtype in {torch.bfloat16, torch.float16}
        fa3_delta = 0
        fp8_delta = 0
        nvfp4_delta = 0
        bf16_delta = 0
        for layer in self.layers[1:]:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                proc = getattr(attn, "processor", None)
                if (
                    proc is not None
                    and getattr(proc, "attention_backend", "torch") == "flash_attention_3"
                    and low_prec_dtype
                ):
                    fa3_delta += 1
                gen_attn_mods = (
                    (attn.gen_qkv_proj,)
                    if getattr(attn, "gen_qkv_proj", None) is not None
                    else (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
                ) + (attn.to_add_out,)
                for mod in gen_attn_mods:
                    if isinstance(mod, StepSelectiveFloat8Linear):
                        if precision_tag == "fp8" and low_prec_dtype and mod._hardware_eligible:
                            fp8_delta += 1
                        else:
                            bf16_delta += 1
                    elif isinstance(mod, StepSelectiveNVFP4Linear):
                        if precision_tag == "nvfp4" and low_prec_dtype and mod._hardware_eligible:
                            nvfp4_delta += 1
                        else:
                            bf16_delta += 1
            mlp_gen = getattr(layer, "mlp_moe_gen", None)
            if mlp_gen is not None:
                gen_mlp_mods = (
                    (mlp_gen.gate_up_proj,)
                    if getattr(mlp_gen, "gate_up_proj", None) is not None
                    else (mlp_gen.gate_proj, mlp_gen.up_proj)
                ) + (mlp_gen.down_proj,)
                for mod in gen_mlp_mods:
                    if isinstance(mod, StepSelectiveFloat8Linear):
                        if precision_tag == "fp8" and low_prec_dtype and mod._hardware_eligible:
                            fp8_delta += 1
                        else:
                            bf16_delta += 1
                    elif isinstance(mod, StepSelectiveNVFP4Linear):
                        if precision_tag == "nvfp4" and low_prec_dtype and mod._hardware_eligible:
                            nvfp4_delta += 1
                        else:
                            bf16_delta += 1
        self._cosmos3_fa3_replay_calls += fa3_delta
        self._cosmos3_fp8_replay_calls += fp8_delta
        self._cosmos3_nvfp4_replay_calls += nvfp4_delta
        self._cosmos3_bf16_replay_calls += bf16_delta

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Checkpoint each native MoT decoder block during training."""

        self._enable_gradient_checkpointing = bool(enable)

    # -------------------------------------------------------------------------
    # Pure-tensor packing/unpacking helpers (no layer state).
    # -------------------------------------------------------------------------

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: list[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        """Scatter-add timestep embeddings onto noisy spatial (or temporal) tokens only."""
        start_noisy_index = 0
        flattened_noisy_frame_indexes: list[torch.Tensor] = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes, strict=False):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            # Broadcast [N, 1] + [spatial_numel_i] → [N, spatial_numel_i]
            frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _patchify_and_pack_latents(
        self,
        tokens_vision: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        """Pad ``C T H W`` cubes to a multiple of ``p``, then pack as ``[Σ T·H'·W', p*p*C]``."""
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        packed_latent: list[torch.Tensor] = []
        original_latent_shapes: list[tuple[int, int, int]] = []
        for latent in tokens_vision:
            latent = latent.squeeze(0)  # [C, T, H, W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: list[tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: list[tuple[int, int, int]],
    ) -> list[torch.Tensor]:
        """Inverse of :meth:`_patchify_and_pack_latents`; zeros stay at conditioned frames."""
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        unpatchified_latents: list[torch.Tensor] = []
        start_idx = 0
        for token_shape, noisy_frame_indexes, original_shape in zip(
            token_shapes_vision, noisy_frame_indexes_vision, original_latent_shapes, strict=False
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original_shape
            h_padded = ((h_orig + p - 1) // p) * p
            w_padded = ((w_orig + p - 1) // p) * p
            h_patches = h_padded // p
            w_patches = w_padded // p
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def _pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """List of ``[C, T]`` tensors → packed ``[total_T, C]`` tensor."""
        return torch.cat(
            [
                sound[:, : shape[0]].permute(1, 0)
                for sound, shape in zip(tokens_sound, token_shapes_sound, strict=False)
            ],
            dim=0,
        )

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, C]`` predictions → list of ``[C, T]`` tensors (zeros at conditioned positions)."""
        sound_dim = self.config.sound_dim
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound, strict=False):
            T = shape[0]
            output = torch.zeros((sound_dim, T), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    def _pack_action_latents(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, int, int]],
        domain_ids_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """List of ``[T, D]`` tensors → packed ``[total_T, D]`` plus per-token domain ids."""
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for action, shape, domain_id in zip(tokens_action, token_shapes_action, domain_ids_action, strict=False):
            token_count = shape[0]
            packed.append(action[:token_count])
            domain_ids.append(domain_id.reshape(1).expand(token_count))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def _unpack_action_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, int, int]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, D]`` predictions → list of ``[T, D]`` tensors."""
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_action, noisy_frame_indexes_action, strict=False):
            T = shape[0]
            output = torch.zeros((T, self.action_dim), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[noisy_idxs] = packed_preds[start_idx : start_idx + t_n]
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    # -------------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision/sound/action → run layers →
    # decode vision/sound/action. Pipeline calls this once per CFG pass.
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        und_len: int,
        sequence_length: int,
        vision_tokens: list[torch.Tensor],
        vision_token_shapes: list[tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        vision_timesteps: torch.Tensor,
        vision_noisy_frame_indexes: list[torch.Tensor],
        sound_tokens: list[torch.Tensor] | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_sequence_indexes: torch.Tensor | None = None,
        sound_mse_loss_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_tokens: list[torch.Tensor] | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_sequence_indexes: torch.Tensor | None = None,
        action_mse_loss_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_domain_ids: list[torch.Tensor] | None = None,
        *,
        step_index: int | None = None,
        total_steps: int | None = None,
        cache_branch: str = "default",
        gen_cache: Any = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Run a full denoising-step forward pass.

        Args:
            input_ids: Text token IDs placed at ``text_indexes`` in the joint sequence.
            text_indexes: Indices of text tokens in the joint sequence.
            position_ids: ``[3, sequence_length]`` mRoPE position IDs for the full joint sequence.
            und_len: Length of the causal text (understanding) prefix; generation tokens follow.
            sequence_length: Total length of the joint packed sequence.
            vision_tokens: Per-item vision latent tensors before patchify.
            vision_token_shapes: Patch grid shapes ``(T, H, W)`` per vision item.
            vision_sequence_indexes: Indices of vision tokens in the joint sequence.
            vision_mse_loss_indexes: Indices used to read vision predictions after the backbone.
            vision_timesteps: Per-patch diffusion timesteps for vision tokens.
            vision_noisy_frame_indexes: Noisy frame indices per vision item.
            sound_tokens: Optional sound latent tensors before packing.
            sound_token_shapes: Optional patch grid shapes for sound items.
            sound_sequence_indexes: Optional indices of sound tokens in the joint sequence.
            sound_mse_loss_indexes: Optional indices used to read sound predictions.
            sound_timesteps: Optional per-token diffusion timesteps for sound.
            sound_noisy_frame_indexes: Optional noisy frame indices per sound item.
            action_tokens: Optional action latent tensors before packing.
            action_token_shapes: Optional patch grid shapes ``(T, H, W)`` per action item.
            action_sequence_indexes: Optional indices of action tokens in the joint sequence.
            action_mse_loss_indexes: Optional indices used to read action predictions after the backbone.
            action_timesteps: Optional per-token diffusion timesteps for action tokens.
            action_noisy_frame_indexes: Optional noisy frame indices per action item.
            action_domain_ids: Optional per-item domain IDs selecting the action head weights.

        Returns:
            ``(preds_vision, preds_sound, preds_action)`` — lists of per-modality predictions. Optional modalities
            return ``None`` when their inputs are omitted.
        """
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None
        has_action = action_tokens is not None and action_sequence_indexes is not None

        # Embed text tokens into the joint hidden_states buffer at their sequence positions.
        packed_text_embedding = self.embed_tokens(input_ids)
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
        hidden_states[text_indexes] = packed_text_embedding

        # Patchify + project vision latents, then add timestep embeddings to noisy frames.
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
        packed_tokens_vision = self.proj_in(packed_tokens_vision)
        timesteps_vision = vision_timesteps * self.config.timestep_scale
        packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
        packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )
        hidden_states[vision_sequence_indexes] = packed_tokens_vision

        # Pack + project sound latents (when present); all sound frames are noisy.
        if has_sound:
            packed_tokens_sound = self._pack_sound_latents(sound_tokens, sound_token_shapes).to(target_dtype)
            packed_tokens_sound = self.audio_proj_in(packed_tokens_sound) + self.audio_modality_embed
            timesteps_sound = sound_timesteps * self.config.timestep_scale
            packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound))
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound_noisy_frame_indexes,
                token_shapes=sound_token_shapes,
            )
            hidden_states[sound_sequence_indexes] = packed_tokens_sound

        # Pack + project action latents (when present). Domain ids select the action head weights.
        if has_action:
            packed_tokens_action, per_token_domain_ids = self._pack_action_latents(
                action_tokens, action_token_shapes, action_domain_ids
            )
            packed_tokens_action = packed_tokens_action.to(target_dtype)
            per_token_domain_ids = per_token_domain_ids.to(device=packed_tokens_action.device)
            packed_tokens_action = self.action_proj_in(packed_tokens_action, per_token_domain_ids)
            packed_tokens_action = packed_tokens_action + self.action_modality_embed
            if action_mse_loss_indexes.numel() > 0:
                timesteps_action = action_timesteps * self.config.timestep_scale
                packed_timestep_embeds_action = self.time_embedder(self.time_proj(timesteps_action))
                packed_timestep_embeds_action = packed_timestep_embeds_action.to(target_dtype)
                packed_tokens_action = self._apply_timestep_embeds_to_noisy_tokens(
                    packed_tokens=packed_tokens_action,
                    packed_timestep_embeds=packed_timestep_embeds_action,
                    noisy_frame_indexes=action_noisy_frame_indexes,
                    token_shapes=action_token_shapes,
                )
            hidden_states[action_sequence_indexes] = packed_tokens_action

        # Compute (or reuse cached) rotary embeddings once for the joint sequence.
        cos_sin_hit: tuple[torch.Tensor, torch.Tensor] | None = None
        if not self.training and not torch.is_grad_enabled():
            is_inf = position_ids.is_inference()
            pos_ver = -1 if is_inf else int(position_ids._version)
            pos_ptr = int(position_ids.data_ptr())
            for ref, saved_ptr, saved_dtype, saved_dev, saved_ver, pos_snap, saved_cos, saved_sin in self._rotary_cache:
                if saved_dtype != hidden_states.dtype or saved_dev != hidden_states.device:
                    continue
                if (
                    not is_inf
                    and saved_ver >= 0
                    and ref() is position_ids
                    and saved_ptr == pos_ptr
                    and saved_ver == pos_ver
                    and pos_snap.shape == position_ids.shape
                    and pos_snap.device == position_ids.device
                ):
                    cos_sin_hit = (saved_cos, saved_sin)
                    break
                if (
                    pos_snap.shape == position_ids.shape
                    and pos_snap.dtype == position_ids.dtype
                    and pos_snap.device == position_ids.device
                    and torch.equal(pos_snap, position_ids)
                ):
                    cos_sin_hit = (saved_cos, saved_sin)
                    break
        if cos_sin_hit is not None:
            cos, sin = cos_sin_hit
        else:
            cos, sin = self.rotary_emb(
                position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            if not self.training and not torch.is_grad_enabled():
                if len(self._rotary_cache) >= 4:
                    self._rotary_cache.pop(0)
                pos_ver = -1 if position_ids.is_inference() else int(position_ids._version)
                self._rotary_cache.append(
                    (
                        weakref.ref(position_ids),
                        int(position_ids.data_ptr()),
                        hidden_states.dtype,
                        hidden_states.device,
                        pos_ver,
                        position_ids.detach().clone(),
                        cos.detach(),
                        sin.detach(),
                    )
                )

        # Step-selective quantization toggle (preserves sensitive early/late steps in BF16).
        precision_tag = "bf16"
        step_quant_policy = getattr(self, "_cosmos3_step_quant_policy", None)
        if step_quant_policy is not None and not self.training:
            from .acceleration import set_cosmos3_step_precision

            start_step = int(step_quant_policy.get("start_step", 0))
            dense_last = int(step_quant_policy.get("dense_last_steps", 0))
            if step_index is None:
                use_lp = True
            else:
                use_lp = int(step_index) >= start_step and (
                    total_steps is None or int(step_index) < max(int(total_steps) - dense_last, 0)
                )
            set_cosmos3_step_precision(self, use_low_precision=use_lp)
            precision_tag = str(step_quant_policy.get("mode", "fp8")) if use_lp else "bf16"

        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        cos_und, sin_und, cos_gen, sin_gen = cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:]
        rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)

        # Resolve per-branch gen-stream cache (TeaCache / FirstBlock) if configured.
        active_gen_cache = gen_cache
        if (
            active_gen_cache is None
            and self._cosmos3_gen_cache_spec is not None
            and not self.training
            and not torch.is_grad_enabled()
        ):
            active_gen_cache = self._cosmos3_gen_caches.get(cache_branch)
            if active_gen_cache is None:
                from .acceleration import Cosmos3GenStreamCache

                spec = self._cosmos3_gen_cache_spec
                active_gen_cache = Cosmos3GenStreamCache(
                    algorithm=spec["algorithm"],
                    threshold=spec["threshold"],
                    total_steps=int(total_steps) if total_steps is not None else 35,
                    warmup_steps=spec["warmup_steps"],
                    dense_last=spec["dense_last"],
                    max_consecutive_skips=spec["max_consecutive_skips"],
                    coefficients=spec["coefficients"],
                )
                self._cosmos3_gen_caches[cache_branch] = active_gen_cache

        und_cache = self._und_static_kv_cache if (not self.training and not torch.is_grad_enabled()) else None
        und_entry = und_cache.lookup(cache_branch, und_seq, cos_und, sin_und) if und_cache is not None else None
        home_device = _module_param_device(self.norm)
        piecewise_graph = getattr(self, "_cosmos3_piecewise_graph", None)
        num_layers = len(self.layers)

        if und_entry is not None:
            # FAST PATH: 100% lossless static und_seq KV cache HIT — skip ALL und projections/FFN/norms!
            layer0_device = _module_param_device(self.layers[0])
            gen_in_0 = _to_device(gen_seq, layer0_device)
            cos_gen_0 = _to_device(cos_gen, layer0_device)
            sin_gen_0 = _to_device(sin_gen, layer0_device)
            k0, v0 = _to_device(und_entry.layer_kv[0], layer0_device)
            gen_after_0 = self.layers[0].forward_gen_with_cached_und_kv(gen_in_0, cos_gen_0, sin_gen_0, k0, v0)
            first_block_residual = gen_after_0 - gen_in_0

            if num_layers == 1:
                gen_seq = gen_after_0
            else:
                same_cuda_device = (
                    piecewise_graph is not None
                    and gen_after_0.is_cuda
                    and all(_module_param_device(layer) == layer0_device for layer in self.layers[1:])
                )
                seg_name = f"cosmos3_gen_tail:{precision_tag}"

                def _run_gen_tail(hidden_after_0: torch.Tensor) -> torch.Tensor:
                    if same_cuda_device:
                        assert piecewise_graph is not None
                        tail_k = (
                            und_entry.tail_k
                            if und_entry.tail_k is not None and und_entry.tail_k.device == layer0_device
                            else torch.stack([und_entry.layer_kv[i][0] for i in range(1, num_layers)], dim=0).to(
                                layer0_device
                            )
                        )
                        tail_v = (
                            und_entry.tail_v
                            if und_entry.tail_v is not None and und_entry.tail_v.device == layer0_device
                            else torch.stack([und_entry.layer_kv[i][1] for i in range(1, num_layers)], dim=0).to(
                                layer0_device
                            )
                        )

                        def _tail_segment(
                            g_in: torch.Tensor,
                            c_g: torch.Tensor,
                            s_g: torch.Tensor,
                            uk_stack: torch.Tensor,
                            uv_stack: torch.Tensor,
                        ) -> torch.Tensor:
                            cur = g_in
                            for rel_idx in range(num_layers - 1):
                                cur = self.layers[rel_idx + 1].forward_gen_with_cached_und_kv(
                                    cur, c_g, s_g, uk_stack[rel_idx], uv_stack[rel_idx]
                                )
                            return cur

                        seg_runner = piecewise_graph.get_or_register_segment(seg_name, _tail_segment)
                        prev_replays = int(seg_runner.stats.get("replay", 0))
                        prev_captures = int(seg_runner.stats.get("capture", 0))
                        out_tail = piecewise_graph.run_segment(
                            seg_name, hidden_after_0, cos_gen_0, sin_gen_0, tail_k, tail_v
                        )
                        cur_replays = int(seg_runner.stats.get("replay", 0))
                        cur_captures = int(seg_runner.stats.get("capture", 0))
                        if (
                            cur_replays > prev_replays
                            and cur_captures == prev_captures
                            and not torch.compiler.is_compiling()
                        ):
                            self._accumulate_gen_tail_replay_telemetry(
                                hidden_dtype=hidden_after_0.dtype,
                                precision_tag=precision_tag,
                            )
                        return out_tail
                    cur = hidden_after_0
                    cur_cg = cos_gen_0
                    cur_sg = sin_gen_0
                    for idx in range(1, num_layers):
                        l_dev = _module_param_device(self.layers[idx])
                        cur = _to_device(cur, l_dev)
                        cur_cg = _to_device(cur_cg, l_dev)
                        cur_sg = _to_device(cur_sg, l_dev)
                        ki, vi = _to_device(und_entry.layer_kv[idx], l_dev)
                        cur = self.layers[idx].forward_gen_with_cached_und_kv(cur, cur_cg, cur_sg, ki, vi)
                    return cur

                if active_gen_cache is not None and step_index is not None:
                    gen_seq, hit = active_gen_cache.decide_and_run_tail(
                        step=int(step_index),
                        total_steps=total_steps,
                        gen_after_0=gen_after_0,
                        first_block_residual=first_block_residual,
                        num_layers=num_layers,
                        run_tail=_run_gen_tail,
                    )
                    if hit and same_cuda_device and piecewise_graph is not None:
                        piecewise_graph.record_skip(seg_name)
                else:
                    gen_seq = _run_gen_tail(gen_after_0)

            und_out = _to_device(und_entry.und_out, home_device)
            gen_seq = _to_device(gen_seq, home_device)
            gen_out = self.norm_moe_gen(gen_seq)
        elif und_cache is not None:
            # Step 0 / Cache MISS: run full dual-stream, capture exact (k_und, v_und) across all layers & und_out.
            und_seq_input_ref = und_seq
            collected_und_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            gen_in_0_for_cache: torch.Tensor | None = None
            gen_after_0_for_cache: torch.Tensor | None = None
            for idx, decoder_layer in enumerate(self.layers):
                layer_device = _module_param_device(decoder_layer)
                und_seq = _to_device(und_seq, layer_device)
                gen_seq = _to_device(gen_seq, layer_device)
                rotary_emb = _to_device(rotary_emb, layer_device)
                if idx == 0:
                    gen_in_0_for_cache = gen_seq
                und_seq, gen_seq, und_kv = decoder_layer(
                    und_seq,
                    gen_seq,
                    rotary_emb,
                    return_und_kv=True,
                )
                collected_und_kv.append((_to_device(und_kv[0], home_device), _to_device(und_kv[1], home_device)))
                if idx == 0:
                    gen_after_0_for_cache = gen_seq
            und_seq = _to_device(und_seq, home_device)
            gen_seq = _to_device(gen_seq, home_device)
            und_out = self.norm(und_seq)
            und_cache.store(cache_branch, und_seq_input_ref, cos_und, sin_und, collected_und_kv, und_out)
            # Record this full dense pass in TeaCache/FirstBlock without ever skipping or keeping stale residuals.
            if (
                active_gen_cache is not None
                and step_index is not None
                and num_layers > 1
                and gen_in_0_for_cache is not None
                and gen_after_0_for_cache is not None
            ):
                r0 = gen_after_0_for_cache - gen_in_0_for_cache
                final_gen_on_0 = _to_device(gen_seq, gen_after_0_for_cache.device)
                active_gen_cache.record_dense_step(
                    step=int(step_index),
                    gen_after_0=gen_after_0_for_cache,
                    first_block_residual=r0,
                    gen_final=final_gen_on_0,
                    num_layers=num_layers,
                )
            gen_out = self.norm_moe_gen(gen_seq)
        elif (
            active_gen_cache is not None
            and step_index is not None
            and num_layers > 1
            and not self.training
            and not torch.is_grad_enabled()
        ):
            # Dual-stream inference with TeaCache/FirstBlock even when und_static_kv_cache=False.
            layer0_device = _module_param_device(self.layers[0])
            und_0 = _to_device(und_seq, layer0_device)
            gen_in_0 = _to_device(gen_seq, layer0_device)
            rot_0 = _to_device(rotary_emb, layer0_device)
            und_after_0, gen_after_0 = self.layers[0](und_0, gen_in_0, rot_0)
            first_block_residual = gen_after_0 - gen_in_0
            und_tail_final = und_after_0

            def _run_dual_tail(hidden_after_0: torch.Tensor) -> torch.Tensor:
                nonlocal und_tail_final
                cur_u = und_after_0
                cur_g = hidden_after_0
                cur_rot = rot_0
                for idx in range(1, num_layers):
                    l_dev = _module_param_device(self.layers[idx])
                    cur_u = _to_device(cur_u, l_dev)
                    cur_g = _to_device(cur_g, l_dev)
                    cur_rot = _to_device(cur_rot, l_dev)
                    cur_u, cur_g = self.layers[idx](cur_u, cur_g, cur_rot)
                und_tail_final = cur_u
                return cur_g

            gen_seq, _ = active_gen_cache.decide_and_run_tail(
                step=int(step_index),
                total_steps=total_steps,
                gen_after_0=gen_after_0,
                first_block_residual=first_block_residual,
                num_layers=num_layers,
                run_tail=_run_dual_tail,
            )
            und_seq = _to_device(und_tail_final, home_device)
            gen_seq = _to_device(gen_seq, home_device)
            und_out = self.norm(und_seq)
            gen_out = self.norm_moe_gen(gen_seq)
        else:
            # Original dual-stream path (training / uncached).
            for decoder_layer in self.layers:
                layer_device = _module_param_device(decoder_layer)
                und_seq = _to_device(und_seq, layer_device)
                gen_seq = _to_device(gen_seq, layer_device)
                rotary_emb = _to_device(rotary_emb, layer_device)
                und_seq, gen_seq = gradient_checkpoint_forward(
                    decoder_layer,
                    self._enable_gradient_checkpointing and self.training,
                    False,
                    und_seq,
                    gen_seq,
                    rotary_emb,
                )
            und_seq = _to_device(und_seq, home_device)
            gen_seq = _to_device(gen_seq, home_device)
            und_out = self.norm(und_seq)
            gen_out = self.norm_moe_gen(gen_seq)

        last_hidden_state = torch.cat([und_out, gen_out], dim=0)

        # Decode vision predictions from the joint hidden state.
        preds_vision_packed = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        preds_sound: list[torch.Tensor] | None = None
        if has_sound:
            preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes, sound_noisy_frame_indexes)

        preds_action: list[torch.Tensor] | None = None
        if has_action:
            per_noisy_domain_ids = [
                domain_id.reshape(1).expand(len(noisy_idxs))
                for domain_id, noisy_idxs in zip(action_domain_ids, action_noisy_frame_indexes, strict=False)
            ]
            per_noisy_domain_ids = torch.cat(per_noisy_domain_ids, dim=0).to(device=last_hidden_state.device)
            preds_action_packed = self.action_proj_out(last_hidden_state[action_mse_loss_indexes], per_noisy_domain_ids)
            preds_action = self._unpack_action_latents(
                preds_action_packed, action_token_shapes, action_noisy_frame_indexes
            )

        return preds_vision, preds_sound, preds_action

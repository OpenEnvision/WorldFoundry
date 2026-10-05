"""MoVerse KV-cached dual-stream MMDiT (inference).

Second stream alongside the causal Wan graph.  Adds MMDiT + KV cache.
No VACE or linear-attention processors.
"""
import math
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn

from worldfoundry.core.attention.sequence.varlen import (
    attention,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_21 import (
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)
from .causal import (
    causal_rope_apply,
    CausalWanAttentionBlock,
    CausalHead,
)
from .normalization import WanLayerNorm
from .attention import WAN_CROSSATTENTION_CLASSES


# ---------------------------------------------------------------------------
# Self-attention projection module (Q/K/V/O + optional RMSNorm)
# ---------------------------------------------------------------------------

class _SelfAttnProj(nn.Module):
    """Q/K/V/O projections + optional RMSNorm for one attention stream.

    Attribute names match CausalWanSelfAttention (q, k, v, o, norm_q, norm_k)
    so that CausalWanModel checkpoint weights load directly into the x-stream
    and can be cloned into the c-stream.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, qk_norm: bool = True):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


# ---------------------------------------------------------------------------
# CausalMMDiTDoubleBlock
# ---------------------------------------------------------------------------

class CausalMMDiTDoubleBlock(nn.Module):
    """Dual-stream block backed by a joint x/condition KV cache."""

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        is_last: bool = False,
        causal_rope_fn=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.is_last = is_last

        # ── x-stream (same names as CausalWanAttentionBlock for weight loading) ──
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = _SelfAttnProj(dim, num_heads, eps, qk_norm)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        # ── c-stream (cond_ prefix) ──────────────────────────────────────────
        self.cond_norm1 = WanLayerNorm(dim, eps)
        self.cond_self_attn = _SelfAttnProj(dim, num_heads, eps, qk_norm)
        self.cond_modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        if not is_last:
            self.cond_norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cond_cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
                dim, num_heads, (-1, -1), qk_norm, eps)
            self.cond_norm2 = WanLayerNorm(dim, eps)
            self.cond_ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))

        self.causal_rope_fn = causal_rope_fn if causal_rope_fn is not None else causal_rope_apply

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,           # [B, S_x, dim]  S_x = num_frames * frame_seqlen
        c: torch.Tensor,           # [B, S_c, dim]  same S as x
        e_x: torch.Tensor,         # [B, F, 6, dim] per-frame modulation for x
        e_c: torch.Tensor,         # [B, F, 6, dim] per-frame modulation for c
        seq_lens: torch.Tensor,    # [B]
        grid_sizes: torch.Tensor,  # [B, 3]
        freqs: torch.Tensor,       # RoPE freqs
        context: torch.Tensor,     # [B, text_len, dim] projected text
        context_lens,              # None or [B]
        kv_cache=None,
        crossattn_cache=None,      # cross-attn cache dict (inference)
        current_start: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_x, _ = x.shape
        n, d = self.num_heads, self.head_dim
        num_frames = e_x.shape[1]
        frame_seqlen = S_x // num_frames

        # ── Modulation ────────────────────────────────────────────────────────
        # e_x/e_c: [B, F, 6, dim]; modulation: [1, 6, dim] → [1, 1, 6, dim]
        e_x_mod = (self.modulation.unsqueeze(1) + e_x).chunk(6, dim=2)
        e_c_mod = (self.cond_modulation.unsqueeze(1) + e_c).chunk(6, dim=2)
        # Each e_x_mod[i]: [B, F, 1, dim] → broadcasts over frame_seqlen tokens

        def _modulate_x(norm_x, shift, scale):
            """Apply per-frame AdaLN: norm → scale/shift in [B, F, frame_seqlen, dim]."""
            return (norm_x.unflatten(1, (num_frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)

        # ── Compute Q, K, V for both streams ─────────────────────────────────
        x_in = _modulate_x(self.norm1(x), e_x_mod[0], e_x_mod[1])
        c_in = _modulate_x(self.cond_norm1(c), e_c_mod[0], e_c_mod[1])

        x_q = self.self_attn.norm_q(self.self_attn.q(x_in)).view(B, S_x, n, d)
        x_k = self.self_attn.norm_k(self.self_attn.k(x_in)).view(B, S_x, n, d)
        x_v = self.self_attn.v(x_in).view(B, S_x, n, d)

        c_q = self.cond_self_attn.norm_q(self.cond_self_attn.q(c_in)).view(B, S_x, n, d)
        c_k = self.cond_self_attn.norm_k(self.cond_self_attn.k(c_in)).view(B, S_x, n, d)
        c_v = self.cond_self_attn.v(c_in).view(B, S_x, n, d)

        # KV cache + exact attention.
        frame_seqlen_kv = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen_kv

        x_q = self.causal_rope_fn(x_q, grid_sizes, freqs, start_frame=current_start_frame).type_as(x_v)
        x_k = self.causal_rope_fn(x_k, grid_sizes, freqs, start_frame=current_start_frame).type_as(x_v)
        c_q = self.causal_rope_fn(c_q, grid_sizes, freqs, start_frame=current_start_frame).type_as(c_v)
        c_k = self.causal_rope_fn(c_k, grid_sizes, freqs, start_frame=current_start_frame).type_as(c_v)

        current_end = current_start + x_q.shape[1]
        num_new_tokens = x_q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen_kv
        kv_cache_size = kv_cache["x_k"].shape[1]
        global_end = kv_cache["global_end_index"].item()
        local_end_prev = kv_cache["local_end_index"].item()

        if (self.local_attn_size != -1
                and current_end > global_end
                and num_new_tokens + local_end_prev > kv_cache_size):
            # KV eviction: shift buffers left, preserve sink tokens
            num_evicted = num_new_tokens + local_end_prev - kv_cache_size
            num_rolled = local_end_prev - num_evicted - sink_tokens

            def _evict(buf):
                buf[:, sink_tokens:sink_tokens + num_rolled] = \
                    buf[:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()

            with torch.no_grad():
                _evict(kv_cache["x_k"]); _evict(kv_cache["x_v"])
                _evict(kv_cache["c_k"]); _evict(kv_cache["c_v"])

            local_end_index = local_end_prev + current_end - global_end - num_evicted
        else:
            local_end_index = local_end_prev + current_end - global_end

        local_start_index = local_end_index - num_new_tokens

        # Write new K/V to both sub-buffers
        with torch.no_grad():
            kv_cache["x_k"][:, local_start_index:local_end_index] = x_k
            kv_cache["x_v"][:, local_start_index:local_end_index] = x_v
            kv_cache["c_k"][:, local_start_index:local_end_index] = c_k
            kv_cache["c_v"][:, local_start_index:local_end_index] = c_v
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # Build joint KV: [x_cache | c_cache]
        joint_k = torch.cat([
            kv_cache["x_k"][:, :local_end_index],
            kv_cache["c_k"][:, :local_end_index],
        ], dim=1)  # [B, 2*local_end, n, d]
        joint_v = torch.cat([
            kv_cache["x_v"][:, :local_end_index],
            kv_cache["c_v"][:, :local_end_index],
        ], dim=1)

        x_attn = attention(x_q, joint_k, joint_v).flatten(2)  # [B, S_x, dim]
        c_attn = attention(c_q, joint_k, joint_v).flatten(2)  # [B, S_c, dim]

        # ── x-stream: residual + cross-attn + FFN ────────────────────────────
        x_gate = e_x_mod[2]  # [B, F, 1, dim]
        x = x + (self.self_attn.o(x_attn).unflatten(1, (num_frames, frame_seqlen)) * x_gate).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens,
                                 crossattn_cache=crossattn_cache)
        x_shift_mlp, x_scale_mlp, x_gate_mlp = e_x_mod[3], e_x_mod[4], e_x_mod[5]
        x_ffn_in = _modulate_x(self.norm2(x), x_shift_mlp, x_scale_mlp)
        x = x + (self.ffn(x_ffn_in).unflatten(1, (num_frames, frame_seqlen)) * x_gate_mlp).flatten(1, 2)

        # ── c-stream: residual + cross-attn + FFN (skipped at is_last) ───────
        if not self.is_last:
            c_gate = e_c_mod[2]
            c = c + (self.cond_self_attn.o(c_attn).unflatten(1, (num_frames, frame_seqlen)) * c_gate).flatten(1, 2)
            c = c + self.cond_cross_attn(self.cond_norm3(c), context, context_lens,
                                          crossattn_cache=crossattn_cache)
            c_shift_mlp, c_scale_mlp, c_gate_mlp = e_c_mod[3], e_c_mod[4], e_c_mod[5]
            c_ffn_in = _modulate_x(self.cond_norm2(c), c_shift_mlp, c_scale_mlp)
            c = c + (self.cond_ffn(c_ffn_in).unflatten(1, (num_frames, frame_seqlen)) * c_gate_mlp).flatten(1, 2)

        return x, c


# ---------------------------------------------------------------------------
# CausalMMDiTModel
# ---------------------------------------------------------------------------

class CausalMMDiTModel(nn.Module):
    """Causal dual-stream MMDiT used for KV-cached video inference."""

    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        double_stream_layers: Optional[List[int]] = None,
        causal_rope_fn=None,
    ):
        super().__init__()

        if double_stream_layers is None:
            double_stream_layers = [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]

        assert model_type in ('t2v', 'i2v')
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.eps = eps
        self.double_stream_layers: Set[int] = set(double_stream_layers)

        # ── Patch embeddings (separate per stream) ───────────────────────────
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.cond_patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # ── Text & time embeddings ───────────────────────────────────────────
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # x-stream modulation (same names as CausalWanModel for weight loading)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        # c-stream modulation
        self.cond_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # ── Blocks ────────────────────────────────────────────────────────────
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        last_double = max(double_stream_layers)
        blocks = []
        for i in range(num_layers):
            if i in self.double_stream_layers:
                blocks.append(CausalMMDiTDoubleBlock(
                    cross_attn_type=cross_attn_type,
                    dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
                    local_attn_size=local_attn_size, sink_size=sink_size,
                    qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                    is_last=(i == last_double),
                    causal_rope_fn=causal_rope_fn,
                ))
            else:
                blocks.append(CausalWanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads,
                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                    causal_rope_fn=causal_rope_fn,
                ))
        self.blocks = nn.ModuleList(blocks)

        # ── Output head (x-stream only) ───────────────────────────────────────
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # ── RoPE freqs ────────────────────────────────────────────────────────
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)

        self.init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.xavier_uniform_(self.cond_patch_embedding.weight.flatten(1))
        for emb in (self.text_embedding, self.time_embedding):
            for m in emb.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    # ------------------------------------------------------------------
    # RoPE injection
    # ------------------------------------------------------------------

    def set_rope_fn(self, causal_rope_fn=None):
        """Replace causal RoPE functions on all blocks."""
        for block in self.blocks:
            if isinstance(block, CausalMMDiTDoubleBlock):
                if causal_rope_fn is not None:
                    block.causal_rope_fn = causal_rope_fn
            else:  # CausalWanAttentionBlock
                if causal_rope_fn is not None:
                    block.self_attn.causal_rope_fn = causal_rope_fn

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def unpatchify(self, x_list, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x_list, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x,                                        # [B, C, F, H, W] (C-first) or list
        t,                                        # [B, F] timesteps
        context,                                  # [B, text_len, text_dim]
        seq_len=None,
        grid_sizes=None,
        freqs=None,
        cond_x=None,                              # [B, C, F, H, W] condition latent
        kv_cache=None,                            # list[dict] len=num_layers
        crossattn_cache=None,
        current_start: int = 0,
        **kwargs,
    ):
        if kv_cache is None:
            raise ValueError("CausalMMDiTModel inference requires a KV cache")
        return self._forward_inference(
            x, t, context, seq_len=seq_len,
            cond_x=cond_x, kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start=current_start,
        )

    # ------------------------------------------------------------------
    # Inference forward (KV cache)
    # ------------------------------------------------------------------

    def _forward_inference(
        self, x, t, context, seq_len=None,
        cond_x=None, kv_cache=None, crossattn_cache=None,
        current_start: int = 0,
    ):
        """Autoregressive generation forward with KV caching.

        Args:
            x:      [B, C, F, H, W] current block's noisy latent (C-first).
            t:      [B, F] timesteps for current block.
            context:[B, text_len, text_dim] raw text embeddings.
            cond_x: [B, C, F, H, W] condition latent for current block, or None.
            kv_cache: list[dict] of length num_layers with dual sub-buffers at
                      double-stream blocks.
            crossattn_cache: list[dict] of length num_layers.
            current_start: token offset for RoPE (= frame_idx * frame_seqlen).
        Returns:
            flow_pred: stacked [B, C, F', H', W'].
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Patchify x per sample (matches CausalWanModel._forward_inference)
        x_patches = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_patches])
        x_patches = [u.flatten(2).transpose(1, 2) for u in x_patches]
        seq_lens = torch.tensor([u.size(1) for u in x_patches], dtype=torch.long)
        x_tok = torch.cat(x_patches)  # [B, S_block, dim]

        # Patchify cond_x
        if cond_x is not None:
            c_patches = [self.cond_patch_embedding(u.unsqueeze(0)) for u in cond_x]
            c_patches = [u.flatten(2).transpose(1, 2) for u in c_patches]
            c_tok = torch.cat(c_patches)  # [B, S_block, dim]
        else:
            c_tok = torch.zeros_like(x_tok)

        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x_tok))
        e0_x = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)
        e0_c = self.cond_time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)

        # Text context
        context_lens = None
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

        for i, block in enumerate(self.blocks):
            ca_cache = crossattn_cache[i] if crossattn_cache is not None else None
            if i in self.double_stream_layers:
                x_tok, c_tok = block(
                    x_tok, c_tok, e0_x, e0_c,
                    seq_lens, grid_sizes, self.freqs, context, context_lens,
                    kv_cache=kv_cache[i],
                    crossattn_cache=ca_cache,
                    current_start=current_start,
                )
            else:
                x_tok = block(
                    x_tok, e0_x, seq_lens, grid_sizes, self.freqs,
                    context, context_lens,
                    kv_cache=kv_cache[i],
                    crossattn_cache=ca_cache,
                    current_start=current_start,
                )

        # Head
        e_head = e.unflatten(0, t.shape).unsqueeze(2)
        x_tok = self.head(x_tok, e_head)
        return torch.stack(self.unpatchify(x_tok, grid_sizes))


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_causal_mmdit_weights(
    model: CausalMMDiTModel,
    state_dict: dict,
    strict: bool = False,
) -> None:
    """Load a CausalWanModel checkpoint into CausalMMDiTModel.

    Mapping logic
    -------------
    If state_dict already contains 'cond_' keys → full MMDiT checkpoint, load directly.

    Otherwise (CausalWanModel checkpoint):
      • Regular blocks (CausalWanAttentionBlock): direct name match.
      • Double-stream blocks x-stream: direct name match.
      • Double-stream blocks c-stream: clone from x-stream (prefix 'cond_').
      • cond_patch_embedding ← clone of patch_embedding.
      • cond_time_projection ← clone of time_projection.

    Non-strict mode (default): print MISSING/UNEXPECTED key warnings, continue.
    Strict mode: raise on any mismatch.
    """
    own_state = model.state_dict()

    # Detect if this is already a full MMDiT checkpoint
    has_cond_keys = any(k for k in state_dict if 'cond_' in k)
    if has_cond_keys:
        # Full MMDiT checkpoint — load directly
        new_state = {k: v for k, v in state_dict.items() if k in own_state}
    else:
        # CausalWanModel checkpoint — copy x-stream, clone c-stream
        new_state = {}

        # Step 1: copy all directly matching weights
        for k, v in state_dict.items():
            if k in own_state:
                new_state[k] = v

        # Step 2: clone c-stream from x-stream for double-stream blocks
        for layer_idx in model.double_stream_layers:
            prefix = f'blocks.{layer_idx}.'
            for k, v in state_dict.items():
                if not k.startswith(prefix):
                    continue
                suffix = k[len(prefix):]
                # x-stream sub-module names → c-stream with 'cond_' prefix
                # e.g. 'self_attn.q.weight' → 'cond_self_attn.q.weight'
                #      'modulation'         → 'cond_modulation'
                #      'norm1.weight'       → 'cond_norm1.weight'
                c_key = prefix + 'cond_' + suffix
                if c_key in own_state:
                    new_state[c_key] = v.clone()

        # Step 3: clone cond_patch_embedding ← patch_embedding
        for suffix in ('weight', 'bias'):
            src = f'patch_embedding.{suffix}'
            dst = f'cond_patch_embedding.{suffix}'
            if src in state_dict and dst in own_state:
                new_state[dst] = state_dict[src].clone()

        # Step 4: clone cond_time_projection ← time_projection
        # time_projection is Sequential(SiLU, Linear) — Linear is at index 1
        for suffix in ('1.weight', '1.bias'):
            src = f'time_projection.{suffix}'
            dst = f'cond_time_projection.{suffix}'
            if src in state_dict and dst in own_state:
                new_state[dst] = state_dict[src].clone()

    missing = [k for k in own_state if k not in new_state]
    unexpected = [k for k in new_state if k not in own_state]

    if strict:
        model.load_state_dict(new_state, strict=True)
    else:
        if missing or unexpected:
            print("WARNING: Non-strict checkpoint load for CausalMMDiTModel")
            for k in missing:
                print(f"  MISSING:    {k}")
            for k in unexpected:
                print(f"  UNEXPECTED: {k}")
        model.load_state_dict(new_state, strict=False)

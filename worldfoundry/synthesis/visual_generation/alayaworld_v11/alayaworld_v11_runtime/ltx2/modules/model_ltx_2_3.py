# Copyright 2024-2025 LTX-2.3 Refactored (WAN-style)
"""Alaya's packed-video, spatial-memory and action adapter over native LTX layers."""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.base_models.diffusion_model.models.networks.ltx.feed_forward import FeedForward
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModelType as LTX23ModelType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import (
    BatchedPerturbationConfig,
    PerturbationType,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgs
from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.alayaworld_v11 import AlayaWorldLTXModel
from worldfoundry.core.attention.cache.context import ContextAttentionCache
from worldfoundry.core.nn import PytorchAdaZeroFunction, TransformerAttentionOps, TransformerOpsConfig, rms_norm
from worldfoundry.core.nn.checkpointing.gradient import gradient_checkpoint_forward
from worldfoundry.synthesis.visual_generation.alayaworld_v11.alayaworld_v11_runtime.ltx2.modules.rope import (
    apply_rotary_emb,
    precompute_freqs_cis,
)

from ...fastvideo.utils.context_parallel import (
    apply_ulysses_attention,
    gather_for_loss,
    get_cp_world_size,
    is_cp_enabled,
    pad_to_cp_divisible,
    scatter_sequence,
    unpad_from_cp,
)
from .attention import AttentionCallable, AttentionFunction


@dataclass(frozen=True)
class AlayaAttentionControl:
    """Per-call sparse/Flex policy, carried by native block arguments."""

    video_shape: tuple[int, int, int] | None
    flex_masked: bool = False


_LTX_QUIET = os.environ.get("LTX_QUIET_INFERENCE", "0") == "1"

__all__ = [
    "Attention",
    "FeedForward",
    "LTX23Model",
    "LTX23ModelType",
    "LTXRopeType",
    "precompute_freqs_cis",
    "rms_norm",
]


# =============================================================================
# da3 inference: fused flex_attention for key-bias masked self-attention (exact).
# Meant to run under torch.compile (inductor lowers flex to a fused Triton
# kernel); in eager it falls back to a slow math decomposition. Never used by
# vigeo paths (guarded by the flex_masked kwarg, default False).
# =============================================================================


def _flex_masked_attention(
    q: torch.Tensor,  # [B, Lq, H*D] (post-RoPE, normed)
    k: torch.Tensor,  # [B, Lk, H*D]
    v: torch.Tensor,  # [B, Lk, H*D]
    heads: int,
    mask: torch.Tensor,  # [B, 1, 1, Lk] additive key bias
) -> torch.Tensor:
    B, Lq, inner = q.shape
    D = inner // heads
    Lk = k.shape[1]
    key_bias = mask.reshape(B, Lk).float()  # per-key additive bias

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + key_bias[b, kv_idx]

    q4 = q.view(B, Lq, heads, D).transpose(1, 2)  # [B,H,Lq,D]
    k4 = k.view(B, Lk, heads, D).transpose(1, 2)
    v4 = v.view(B, Lk, heads, D).transpose(1, 2)
    out = _flex_attention(q4, k4, v4, score_mod=score_mod)
    return out.transpose(1, 2).reshape(B, Lq, inner).to(q.dtype)


class LTX23SelfAttention(Attention):
    """Self-attention with sparse attention and context parallel support."""

    def apply_rotary_embedding(self, value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return apply_rotary_emb(value, position, self.rope_type)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        camera_control: object | None = None,
        cache: ContextAttentionCache | None = None,
        *,
        video_shape: tuple[int, int, int] | None = None,
        window_size: tuple[int, int] | None = None,
        flex_masked: bool = False,
    ) -> torch.Tensor:
        if isinstance(camera_control, AlayaAttentionControl):
            video_shape, flex_masked = camera_control.video_shape, camera_control.flex_masked
        is_self_attn = context is None
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q, k = self.preattention_function(self.to_q(x), self.to_k(context), self, mask, pe, k_pe)

            # Choose attention implementation
            if (
                flex_masked
                and mask is not None
                and is_self_attn
                and not self.enable_sparse_attention
                and not is_cp_enabled()
            ):
                # da3 inference: exact biased attention in one fused flex_attention kernel
                out = _flex_masked_attention(q, k, v, self.heads, mask)
            elif self.enable_sparse_attention and video_shape is not None:
                q = q.view(q.shape[0], q.shape[1], self.heads, self.dim_head)
                k = k.view(k.shape[0], k.shape[1], self.heads, self.dim_head)
                v_reshaped = v.view(v.shape[0], v.shape[1], self.heads, self.dim_head)
                out = self.sparse_attn(q, k, v_reshaped, video_shape=video_shape, mask=mask)
                out = out.reshape(q.shape[0], q.shape[1], -1)
            elif is_self_attn and is_cp_enabled():
                attn_fn = (
                    _flex_masked_attention
                    if (flex_masked and mask is not None and not self.enable_sparse_attention)
                    else self.attention_function
                )
                out = apply_ulysses_attention(q, k, v, self.heads, attn_fn, mask)
            else:
                # FreeLong++: pass window_size to FA3 (other backends ignore it)
                out = self.attention_function(q, k, v, self.heads, mask, window_size=window_size)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)

        return self.to_out(out)


class LTX23Model(AlayaWorldLTXModel, ModelMixin, ConfigMixin):
    ignore_for_config: ClassVar[list[str]] = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules: ClassVar[list[str]] = ["BasicAVTransformerBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="video_only",
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = True,
        normalize_rope_positions: bool = True,
        normalize_time_by_fps: bool = True,
        # LTX 2.3 new params
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
        caption_proj_before_connector: bool = False,
        compact_spatial_tokens: bool = False,
        # Legacy params
        patch_size=(1, 1, 1),
        text_len=1024,
        in_dim=128,
        dim=4096,
        ffn_dim=16384,
        freq_dim=256,
        text_dim=3840,
        out_dim=128,
        num_heads=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # Clean control params
        enable_action_control: bool = False,
        # Sparse Attention params
        enable_sparse_attention: bool = False,
        sparse_block_size: tuple = (4, 4, 4),
        sparse_ratio_train: float = 0.125,
        sparse_ratio_inference: float = 0.0625,
    ):
        attention_type = AttentionFunction.FLASH_ATTENTION_3
        attention_ops = TransformerAttentionOps(
            attention_function=attention_type, masked_attention_function=attention_type
        )
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            norm_eps=norm_eps,
            ops=TransformerOpsConfig(
                attention_ops=attention_ops, ada_zero_function=PytorchAdaZeroFunction(backend="torch")
            ),
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            timestep_scale_multiplier=timestep_scale_multiplier,
            use_middle_indices_grid=use_middle_indices_grid,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=cross_attention_adaln,
            caption_channels=caption_channels,
            caption_proj_before_connector=caption_proj_before_connector,
            enable_action_control=enable_action_control,
            action_freq_dim=int(os.environ.get("LTX_ACTION_FREQ_DIM_PER_AXIS", "32")) if enable_action_control else 32,
            action_freq_scale=float(os.environ.get("LTX_ACTION_FREQ_SCALE", "1000"))
            if enable_action_control
            else 1000.0,
            action_subframes=int(os.environ.get("LTX_ACTION_SUBFRAMES", "8")) if enable_action_control else 8,
        )
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
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.normalize_rope_positions = normalize_rope_positions
        self.normalize_time_by_fps = normalize_time_by_fps
        self.yarn_rope = False
        self.yarn_max_train_seconds = 20.0
        self.apply_gated_attention = apply_gated_attention

        # Sparse attention
        self.enable_sparse_attention = enable_sparse_attention
        self.compact_spatial_tokens = bool(compact_spatial_tokens)
        self.sparse_block_size = sparse_block_size
        self.sparse_ratio = sparse_ratio_train
        self.sparse_ratio_train = sparse_ratio_train
        self.sparse_ratio_inference = sparse_ratio_inference

        for index, block in enumerate(self.blocks):
            block.attn1.__class__ = LTX23SelfAttention
            block.attn1._block_idx = index
            block.attn1.enable_sparse_attention = enable_sparse_attention
            if enable_sparse_attention:
                from .sparse_attention import BlockSparseAttention

                block.attn1.sparse_attn = BlockSparseAttention(
                    block_size=sparse_block_size,
                    sparsity_ratio=sparse_ratio_train,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                )

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        action_vectors: torch.Tensor | None = None,
        action_condition_vectors: torch.Tensor | None = None,
        action_history_vectors: torch.Tensor | None = None,
        perturbations: "BatchedPerturbationConfig | None" = None,
        **kwargs,
    ):
        return self._forward(
            x,
            t,
            context,
            seq_len,
            action_vectors=action_vectors,
            action_condition_vectors=action_condition_vectors,
            action_history_vectors=action_history_vectors,
            perturbations=perturbations,
            **kwargs,
        )

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        fps: float = 25.0,
        action_vectors: torch.Tensor | None = None,
        action_condition_vectors: torch.Tensor | None = None,
        action_history_vectors: torch.Tensor | None = None,
        perturbations: "BatchedPerturbationConfig | None" = None,
        **kwargs,
    ):
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
            get_pixel_coords_tuple_order as get_pixel_coords,
        )

        from .patchifier import SpatioTemporalScaleFactors, VideoLatentPatchifier, VideoLatentShape

        device = self.patchify_proj.weight.device
        dtype = self.patchify_proj.weight.dtype

        # mem_tokens shape [B, N_mem, inner_dim], indices_grid shape [B, 3, N_mem]
        history_kv_tokens = kwargs.get("history_kv_tokens")  # [B, N_mem, inner_dim]
        history_indices_grid = kwargs.get("history_indices_grid")  # [B, 3, N_mem]
        gen_t_indices_override = kwargs.get("gen_t_indices_override")
        cond_latent_frames = int(kwargs.get("cond_latent_frames", 0) or 0)
        sink_latent = kwargs.get("sink_latent")
        sink_indices_grid = kwargs.get("sink_indices_grid")  # [B, 3, N_sink, 2]
        spatial_latent = kwargs.get("spatial_latent")
        spatial_mask_patch = kwargs.get("spatial_mask_patch")  # [B, N_spatial, C_patch]
        spatial_indices_grid = kwargs.get("spatial_indices_grid")  # [B, 3, N_spatial, 2]
        nearby_latent = kwargs.get("nearby_latent")
        nearby_indices_grid = kwargs.get("nearby_indices_grid")  # [B, 3, N_nearby, 2]
        # da3 inference: route the (key-bias) masked self-attention through one fused
        # flex_attention kernel (exact). Default False keeps the vigeo paths unchanged.
        flex_masked_attention = bool(kwargs.get("flex_masked_attention", False))
        _N_mem = 0
        if history_kv_tokens is not None and history_indices_grid is not None:
            assert history_kv_tokens.shape[1] == history_indices_grid.shape[2], (
                f"history_kv_tokens N_mem ({history_kv_tokens.shape[1]}) != "
                f"history_indices_grid N_mem ({history_indices_grid.shape[2]})"
            )
            _N_mem = history_kv_tokens.shape[1]
        _has_sink = sink_latent is not None and sink_indices_grid is not None
        _has_spatial = spatial_latent is not None and spatial_indices_grid is not None
        _has_spatial_key_mask = _has_spatial and spatial_mask_patch is not None
        _has_nearby = nearby_latent is not None and nearby_indices_grid is not None

        # x: List[[C, F, H, W]] -> [B, C, F, H, W]
        x_stacked = torch.stack(list(x)) if isinstance(x, list) else x

        batch_size, channels, num_frames, height, width = x_stacked.shape
        video_shape = (num_frames, height, width) if self.enable_sparse_attention else None

        # Patchify
        patchifier = VideoLatentPatchifier(patch_size=1)
        target_shape = VideoLatentShape(
            batch=batch_size, channels=channels, frames=num_frames, height=height, width=width
        )
        latent = patchifier.patchify(x_stacked).to(dtype)
        num_tokens = latent.shape[1]

        # Sigma computation
        t_float = t.float()
        sigma_val = t_float / 1000.0 if t_float.max() > 1.0 else t_float
        timesteps = sigma_val.unsqueeze(-1).unsqueeze(-1).expand(-1, num_tokens, 1).clone().to(dtype)

        if not hasattr(self, "_ts_print_count"):
            self._ts_print_count = 0
        # if self._ts_print_count < 2:
        #     sigma_gen = timesteps[0, -1, 0].item() if num_tokens > 0 else 0
        #     sigma_cond = timesteps[0, 0, 0].item() if num_tokens > 0 else 0
        #     print(f"[LTX23.forward] sigma_gen={sigma_gen:.4f}, x={x_stacked.shape}")
        #     self._ts_print_count += 1

        # Positions with FPS normalization
        scale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        latent_coords = patchifier.get_patch_grid_bounds(output_shape=target_shape, device=device)
        if gen_t_indices_override is not None:
            t_real = gen_t_indices_override.to(device).to(latent_coords.dtype)  # [T_in]
            assert t_real.shape[0] == num_frames, (
                f"gen_t_indices_override length {t_real.shape[0]} != num_frames {num_frames}"
            )
            t_per_token = t_real.view(num_frames, 1, 1).expand(num_frames, height, width).reshape(-1)
            t_per_token_bounds = torch.stack([t_per_token, t_per_token + 1], dim=-1)  # [N, 2] (start, end=start+1)
            latent_coords = latent_coords.clone()
            latent_coords[:, 0, :, :] = t_per_token_bounds.unsqueeze(0)
        positions = get_pixel_coords(latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=True).float()

        if getattr(self, "normalize_time_by_fps", True):
            positions[:, 0, ...] = positions[:, 0, ...] / fps
        positions = positions.to(dtype)

        def _coords_grid_to_positions(latent_coords_grid):
            _coords = get_pixel_coords(
                latent_coords=latent_coords_grid.to(device=device, dtype=torch.float32),
                scale_factors=scale_factors,
                causal_fix=True,
            ).float()
            if getattr(self, "normalize_time_by_fps", True):
                _coords[:, 0, ...] = _coords[:, 0, ...] / fps
            return _coords.to(dtype)

        spatial_key_valid = None
        spatial_keep_indices = None  # compact mode: indices of the kept (valid) spatial tokens
        spatial_full_token_count = 0
        spatial_tokens_compacted = False
        if _has_spatial_key_mask:
            _mask_patched = spatial_mask_patch.to(device=device, dtype=dtype)
            assert _mask_patched.dim() == 3, (
                f"spatial_mask_patch must be [B,N,C_patch], got {tuple(_mask_patched.shape)}"
            )
            assert _mask_patched.shape[0] == batch_size, (
                f"spatial_mask_patch batch ({_mask_patched.shape[0]}) != target batch ({batch_size})"
            )
            assert _mask_patched.shape[1] == spatial_indices_grid.shape[2], (
                f"spatial_mask_patch N ({_mask_patched.shape[1]}) != spatial_indices_grid N ({spatial_indices_grid.shape[2]})"
            )
            assert _mask_patched.shape[2] == self.patchify_proj.in_features, (
                f"spatial_mask_patch C_patch ({_mask_patched.shape[2]}) != patchify_proj.in_features ({self.patchify_proj.in_features})"
            )
            spatial_key_valid = _mask_patched[..., 0] > 0.5
            spatial_full_token_count = int(spatial_key_valid.shape[1])
            if self.compact_spatial_tokens:
                if batch_size > 1 and not torch.equal(
                    spatial_key_valid, spatial_key_valid[:1].expand_as(spatial_key_valid)
                ):
                    raise ValueError("compact_spatial_tokens requires the same spatial validity across the batch; ")
                spatial_keep_indices = torch.nonzero(spatial_key_valid[0], as_tuple=False).flatten()
                spatial_tokens_compacted = spatial_keep_indices.numel() != spatial_full_token_count
                spatial_indices_grid = spatial_indices_grid.index_select(
                    2, spatial_keep_indices.to(spatial_indices_grid.device)
                )
                if spatial_tokens_compacted and self.enable_sparse_attention:
                    raise ValueError("compact_spatial_tokens is incompatible with dense-grid sparse attention")
                if spatial_tokens_compacted and is_cp_enabled() and get_cp_world_size() > 1:
                    raise ValueError("compact_spatial_tokens currently requires context-parallel size 1")

        _prefix_pos_parts = []
        if _has_sink:
            _prefix_pos_parts.append(_coords_grid_to_positions(sink_indices_grid))
        if _N_mem > 0:
            _prefix_pos_parts.append(_coords_grid_to_positions(history_indices_grid))
        if _has_spatial:
            _prefix_pos_parts.append(_coords_grid_to_positions(spatial_indices_grid))
        if _has_nearby:
            _prefix_pos_parts.append(_coords_grid_to_positions(nearby_indices_grid))
        if _prefix_pos_parts:
            positions = torch.cat([*_prefix_pos_parts, positions], dim=2)
            # [B, 3, N_sink + N_mem + N_spatial + N_nearby + N_gen, 2]

        # Context
        if isinstance(context, list):
            max_len = max(c.shape[0] for c in context)
            context_padded = []
            for c in context:
                if c.shape[0] < max_len:
                    pad = torch.zeros(max_len - c.shape[0], c.shape[1], device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=0)
                context_padded.append(c)
            context_tensor = torch.stack(context_padded).to(dtype)
        else:
            context_tensor = context.to(dtype)

        hidden = self.patchify_proj(latent)

        _N_sink = 0
        _N_spatial = 0
        _N_nearby = 0
        sink_tokens = None
        spatial_tokens = None
        nearby_tokens = None
        if _has_sink:
            _sink_patched = patchifier.patchify(sink_latent.to(dtype)).to(dtype)
            sink_tokens = self.patchify_proj(_sink_patched)  # [B, N_sink, inner_dim]
            _N_sink = sink_tokens.shape[1]
            assert _N_sink == sink_indices_grid.shape[2], (
                f"sink_tokens N ({_N_sink}) != sink_indices_grid N ({sink_indices_grid.shape[2]})"
            )
        if _has_spatial:
            _spatial_patched = patchifier.patchify(spatial_latent.to(dtype)).to(dtype)
            if spatial_keep_indices is not None:
                assert _spatial_patched.shape[1] == spatial_full_token_count, (
                    f"spatial latent token count ({_spatial_patched.shape[1]}) != "
                    f"mask token count ({spatial_full_token_count})"
                )
                _spatial_patched = _spatial_patched.index_select(1, spatial_keep_indices.to(_spatial_patched.device))
            spatial_tokens = self.patchify_proj(_spatial_patched)  # [B, N_spatial, inner_dim]
            _N_spatial = spatial_tokens.shape[1]
            assert _N_spatial == spatial_indices_grid.shape[2], (
                f"spatial_tokens N ({_N_spatial}) != spatial_indices_grid N ({spatial_indices_grid.shape[2]})"
            )
        if _has_nearby:
            _nearby_patched = patchifier.patchify(nearby_latent.to(dtype)).to(dtype)
            nearby_tokens = self.patchify_proj(_nearby_patched)  # [B, N_nearby, inner_dim]
            _N_nearby = nearby_tokens.shape[1]
            assert _N_nearby == nearby_indices_grid.shape[2], (
                f"nearby_tokens N ({_N_nearby}) != nearby_indices_grid N ({nearby_indices_grid.shape[2]})"
            )
        _N_prefix = _N_sink + _N_mem + _N_spatial + _N_nearby  # used to slice back to the generated segment

        # Prefix order: sink -> temporal memory -> spatial memory -> nearby -> gen.
        # Nearby stays closest to the target in token order; spatial memory keeps
        # its own RoPE coordinates, so this ordering does not shift physical time.
        _prefix_hidden_parts = []
        if _has_sink:
            _prefix_hidden_parts.append(sink_tokens.to(hidden.dtype))
        if _N_mem > 0:
            _prefix_hidden_parts.append(history_kv_tokens.to(hidden.dtype))
        if _has_spatial:
            _prefix_hidden_parts.append(spatial_tokens.to(hidden.dtype))
        if _has_nearby:
            _prefix_hidden_parts.append(nearby_tokens.to(hidden.dtype))
        if _prefix_hidden_parts:
            hidden = torch.cat([*_prefix_hidden_parts, hidden], dim=1)

        # Spatial coverage is applied as a compact self-attention key bias rather
        # than extra mask tokens. Invalid warped spatial keys are invisible to all
        # queries, including target tokens, without materializing a dense S x S mask.
        self_attention_mask = None
        if not self.compact_spatial_tokens and spatial_key_valid is not None and _N_spatial > 0:
            invalid_spatial = ~spatial_key_valid.to(device=device)
            if bool(invalid_spatial.any().detach().cpu().item()):
                mask_value = -10000.0
                key_bias = torch.zeros(batch_size, hidden.shape[1], device=device, dtype=hidden.dtype)
                spatial_start = _N_sink + _N_mem
                spatial_end = spatial_start + _N_spatial
                key_bias[:, spatial_start:spatial_end] = key_bias[:, spatial_start:spatial_end].masked_fill(
                    invalid_spatial,
                    mask_value,
                )
                self_attention_mask = key_bias[:, None, None, :]

        # Time embeddings. Prefix tokens are clean conditions (sigma=0);
        # target tokens use the sampled training sigma.
        tokens_per_frame = height * width
        _cond_tokens = cond_latent_frames * tokens_per_frame if cond_latent_frames > 0 else 0
        if cond_latent_frames > 0 and 0 < _cond_tokens < num_tokens:
            sigma_gen = timesteps[0, -1, 0:1]
            unique_sigmas = torch.stack([torch.zeros_like(sigma_gen), sigma_gen])  # [2, 1]
            single_scaled = sigma_gen * self.timestep_scale_multiplier
            unique_scaled = unique_sigmas * self.timestep_scale_multiplier
            unique_emb, unique_embedded = self.adaln_single(
                unique_scaled.squeeze(-1),
                hidden_dtype=hidden.dtype,
            )
            token_indices = torch.ones(num_tokens, dtype=torch.long, device=device)
            token_indices[:_cond_tokens] = 0
            timestep_emb = unique_emb[token_indices].unsqueeze(0).expand(batch_size, -1, -1)
            embedded_timestep = unique_embedded[token_indices].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            single_sigma = timesteps[0, 0, 0:1]
            single_scaled = single_sigma * self.timestep_scale_multiplier
            single_emb, single_embedded = self.adaln_single(
                single_scaled,
                hidden_dtype=hidden.dtype,
            )
            timestep_emb = single_emb.unsqueeze(0).expand(batch_size, num_tokens, -1)
            embedded_timestep = single_embedded.unsqueeze(0).expand(batch_size, num_tokens, -1)

        if _N_prefix > 0:
            _sigma_zero = torch.zeros_like(timesteps[0, 0, 0:1])
            _sigma_zero_scaled = _sigma_zero * self.timestep_scale_multiplier
            _emb_zero, _embedded_zero = self.adaln_single(
                _sigma_zero_scaled,
                hidden_dtype=hidden.dtype,
            )
            _sigma0_te = _emb_zero.unsqueeze(0)
            _sigma0_et = _embedded_zero.unsqueeze(0)
            _prefix_te_parts = []
            _prefix_et_parts = []
            if _has_sink:
                _prefix_te_parts.append(_sigma0_te.expand(batch_size, _N_sink, -1))
                _prefix_et_parts.append(_sigma0_et.expand(batch_size, _N_sink, -1))
            if _N_mem > 0:
                _prefix_te_parts.append(_sigma0_te.expand(batch_size, _N_mem, -1))
                _prefix_et_parts.append(_sigma0_et.expand(batch_size, _N_mem, -1))
            if _has_spatial:
                _prefix_te_parts.append(_sigma0_te.expand(batch_size, _N_spatial, -1))
                _prefix_et_parts.append(_sigma0_et.expand(batch_size, _N_spatial, -1))
            if _has_nearby:
                _prefix_te_parts.append(_sigma0_te.expand(batch_size, _N_nearby, -1))
                _prefix_et_parts.append(_sigma0_et.expand(batch_size, _N_nearby, -1))
            timestep_emb = torch.cat([*_prefix_te_parts, timestep_emb], dim=1)
            embedded_timestep = torch.cat([*_prefix_et_parts, embedded_timestep], dim=1)

        # Prompt timestep for cross-attention AdaLN (NEW in 2.3)
        # prompt_timestep should be (B, 1, 2*dim) to broadcast over context tokens
        prompt_timestep = None
        if self.prompt_adaln_single is not None:
            single_prompt_emb, _ = self.prompt_adaln_single(
                single_scaled,
                hidden_dtype=hidden.dtype,
            )
            prompt_timestep = single_prompt_emb.unsqueeze(0).expand(batch_size, 1, -1)

        # Action AdaLN: latent-rate delta-6D action_vectors -> timestep/AdaLN modulation.
        # Target tokens receive target action. Explicit nearby condition tokens also
        # receive their clean GT action, matching alaya-world's full-clip action path.
        # Sink/spatial tokens keep zero action. History-memory tokens can receive
        # their source history actions when action_history_vectors is provided.
        if getattr(self, "enable_action_control", False):
            if action_vectors is not None:

                def _prepare_action_frames(raw_action: torch.Tensor, frame_count: int, name: str) -> torch.Tensor:
                    av = raw_action.to(device=device, dtype=dtype)
                    if av.dim() == 2:
                        av = av.unsqueeze(0)
                    if av.dim() != 3 or av.shape[-1] != 6:
                        raise ValueError(f"{name} must be [B,T,6], got {tuple(av.shape)}")
                    if av.shape[1] != frame_count:
                        av = av.permute(0, 2, 1)
                        av = F.interpolate(av.float(), size=frame_count, mode="linear", align_corners=True).to(
                            dtype=dtype
                        )
                        av = av.permute(0, 2, 1)
                    return av

                def _actions_to_tokens(raw_action: torch.Tensor, frame_count: int, token_count: int, name: str):
                    av = _prepare_action_frames(raw_action, frame_count, name)
                    a = self.action_adaln_embedder(av)
                    a0 = self.action_adaln_projection(a)
                    a = a.repeat_interleave(tokens_per_frame, dim=1)
                    a0 = a0.repeat_interleave(tokens_per_frame, dim=1)
                    if a.shape[1] != token_count:
                        a = a.permute(0, 2, 1)
                        a = F.interpolate(a.float(), size=token_count, mode="linear", align_corners=True).to(
                            dtype=a.dtype
                        )
                        a = a.permute(0, 2, 1)
                        a0 = a0.permute(0, 2, 1)
                        a0 = F.interpolate(a0.float(), size=token_count, mode="linear", align_corners=True).to(
                            dtype=a0.dtype
                        )
                        a0 = a0.permute(0, 2, 1)
                    return a, a0

                def _actions_to_compressed_tokens(raw_action: torch.Tensor, token_count: int, name: str):
                    av = raw_action.to(device=device, dtype=dtype)
                    if av.dim() == 2:
                        av = av.unsqueeze(0)
                    if av.dim() != 3 or av.shape[-1] != 6:
                        raise ValueError(f"{name} must be [B,T,6], got {tuple(av.shape)}")
                    frame_count = int(av.shape[1])
                    if frame_count <= 0:
                        raise ValueError(f"{name} must contain at least one frame, got {tuple(av.shape)}")
                    a = self.action_adaln_embedder(av)
                    a0 = self.action_adaln_projection(a)
                    if token_count % frame_count == 0:
                        repeat = token_count // frame_count
                        a = a.repeat_interleave(repeat, dim=1)
                        a0 = a0.repeat_interleave(repeat, dim=1)
                    else:
                        a = a.permute(0, 2, 1)
                        a = F.interpolate(a.float(), size=token_count, mode="linear", align_corners=True).to(
                            dtype=a.dtype
                        )
                        a = a.permute(0, 2, 1)
                        a0 = a0.permute(0, 2, 1)
                        a0 = F.interpolate(a0.float(), size=token_count, mode="linear", align_corners=True).to(
                            dtype=a0.dtype
                        )
                        a0 = a0.permute(0, 2, 1)
                    return a, a0

                action_a, action_a0 = _actions_to_tokens(action_vectors, num_frames, num_tokens, "action_vectors")

                if _N_prefix > 0:
                    prefix_a_parts = []
                    prefix_a0_parts = []
                    if _N_sink > 0:
                        prefix_a_parts.append(
                            torch.zeros(
                                action_a.shape[0], _N_sink, action_a.shape[-1], device=device, dtype=action_a.dtype
                            )
                        )
                        prefix_a0_parts.append(
                            torch.zeros(
                                action_a0.shape[0], _N_sink, action_a0.shape[-1], device=device, dtype=action_a0.dtype
                            )
                        )
                    if _N_mem > 0:
                        if action_history_vectors is not None:
                            hist_a, hist_a0 = _actions_to_compressed_tokens(
                                action_history_vectors,
                                _N_mem,
                                "action_history_vectors",
                            )
                            prefix_a_parts.append(hist_a)
                            prefix_a0_parts.append(hist_a0)
                        else:
                            prefix_a_parts.append(
                                torch.zeros(
                                    action_a.shape[0], _N_mem, action_a.shape[-1], device=device, dtype=action_a.dtype
                                )
                            )
                            prefix_a0_parts.append(
                                torch.zeros(
                                    action_a0.shape[0],
                                    _N_mem,
                                    action_a0.shape[-1],
                                    device=device,
                                    dtype=action_a0.dtype,
                                )
                            )
                    if _N_spatial > 0:
                        prefix_a_parts.append(
                            torch.zeros(
                                action_a.shape[0], _N_spatial, action_a.shape[-1], device=device, dtype=action_a.dtype
                            )
                        )
                        prefix_a0_parts.append(
                            torch.zeros(
                                action_a0.shape[0],
                                _N_spatial,
                                action_a0.shape[-1],
                                device=device,
                                dtype=action_a0.dtype,
                            )
                        )
                    if _N_nearby > 0:
                        if action_condition_vectors is not None and _has_nearby:
                            nearby_frames = nearby_latent.shape[2]
                            cond_a, cond_a0 = _actions_to_tokens(
                                action_condition_vectors,
                                nearby_frames,
                                _N_nearby,
                                "action_condition_vectors",
                            )
                            prefix_a_parts.append(cond_a)
                            prefix_a0_parts.append(cond_a0)
                        else:
                            prefix_a_parts.append(
                                torch.zeros(
                                    action_a.shape[0],
                                    _N_nearby,
                                    action_a.shape[-1],
                                    device=device,
                                    dtype=action_a.dtype,
                                )
                            )
                            prefix_a0_parts.append(
                                torch.zeros(
                                    action_a0.shape[0],
                                    _N_nearby,
                                    action_a0.shape[-1],
                                    device=device,
                                    dtype=action_a0.dtype,
                                )
                            )
                    action_a = torch.cat([*prefix_a_parts, action_a], dim=1)
                    action_a0 = torch.cat([*prefix_a0_parts, action_a0], dim=1)

                embedded_timestep = embedded_timestep + action_a.to(dtype=embedded_timestep.dtype)
                timestep_emb = timestep_emb + action_a0.to(dtype=timestep_emb.dtype)

                if not hasattr(self, "_action_adaln_logged") and not _LTX_QUIET:
                    self._action_adaln_logged = True
                    print(
                        f"[LTX23-ActionAdaLN] inject: action_vectors={tuple(action_vectors.shape)}, "
                        f"condition={tuple(action_condition_vectors.shape) if action_condition_vectors is not None else None}, "
                        f"history={tuple(action_history_vectors.shape) if action_history_vectors is not None else None}, "
                        f"a={tuple(action_a.shape)}, a0={tuple(action_a0.shape)}, "
                        f"embedded_timestep={tuple(embedded_timestep.shape)}, "
                        f"timestep_emb={tuple(timestep_emb.shape)}"
                    )
            else:
                # Keep FSDP parameter usage consistent on no-control batches.
                dummy = torch.zeros(1, 1, 6, device=device, dtype=dtype)
                dummy_a = self.action_adaln_embedder(dummy)
                _ = self.action_adaln_projection(dummy_a)

        # Caption projection (None for 22B where projection is in text encoder)
        if self.caption_projection is not None:
            context_proj = self.caption_projection(context_tensor)
            context_proj = context_proj.view(batch_size, -1, hidden.shape[-1])
        else:
            context_proj = context_tensor.view(batch_size, -1, hidden.shape[-1])

        # Dummy forward for prompt_adaln_single FSDP sync
        if self.prompt_adaln_single is not None and prompt_timestep is None:
            _dummy_t = torch.zeros(1, device=device, dtype=dtype)
            _ = self.prompt_adaln_single(_dummy_t, hidden_dtype=dtype)

        # Positional embeddings (RoPE)
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs = precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=latent.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
            normalize_positions=self.normalize_rope_positions,
        )

        # Context Parallel
        _cp_orig_seq_len = None
        if is_cp_enabled():
            cp_size = get_cp_world_size()
            hidden, _cp_orig_seq_len = pad_to_cp_divisible(hidden, dim=1)
            if self_attention_mask is not None and self_attention_mask.shape[-1] < hidden.shape[1]:
                pad = hidden.shape[1] - self_attention_mask.shape[-1]
                self_attention_mask = F.pad(self_attention_mask, (0, pad), value=-10000.0)
            timestep_emb, _ = pad_to_cp_divisible(timestep_emb, dim=1)
            if prompt_timestep is not None:
                prompt_timestep, _ = pad_to_cp_divisible(prompt_timestep, dim=1)
            freqs_cos, _ = pad_to_cp_divisible(freqs[0], dim=2)
            freqs_sin, _ = pad_to_cp_divisible(freqs[1], dim=2)
            freqs = (freqs_cos, freqs_sin)
            hidden = scatter_sequence(hidden, dim=1)
            timestep_emb = scatter_sequence(timestep_emb, dim=1)
            if prompt_timestep is not None:
                prompt_timestep = scatter_sequence(prompt_timestep, dim=1)
            freqs = (scatter_sequence(freqs[0], dim=2), scatter_sequence(freqs[1], dim=2))
            if video_shape is not None:
                T_vs, H_vs, W_vs = video_shape
                padded_seq = hidden.shape[1] * cp_size
                T_padded = padded_seq // (H_vs * W_vs)
                video_shape = (T_padded // cp_size, H_vs, W_vs)

        video_args = TransformerArgs(
            x=hidden,
            context=context_proj,
            context_mask=None,
            timesteps=timestep_emb,
            embedded_timestep=embedded_timestep,
            positional_embeddings=freqs,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
            video_control=AlayaAttentionControl(video_shape, flex_masked_attention),
        )
        if perturbations is None:
            perturbations = self._keep_all_perturbations(video_args)
        else:
            perturbations = BatchedPerturbationConfig.from_masks(
                perturbations.block_masks.to(hidden), perturbations.block_masks_cpu
            )
        for block_index, block in enumerate(self.blocks):
            video_args = self.block_input_processor(
                video_args,
                perturbations,
                block_index,
                self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
            )
            video_args, _ = gradient_checkpoint_forward(
                block, torch.is_grad_enabled() and self.gradient_checkpointing, False, video_args, None
            )
        hidden = video_args.x

        # Gather from context parallel
        if is_cp_enabled():
            hidden = gather_for_loss(hidden, dim=1)
            if _cp_orig_seq_len is not None:
                hidden = unpad_from_cp(hidden, _cp_orig_seq_len, dim=1)

        if _N_prefix > 0:
            hidden = hidden[:, _N_prefix:].contiguous()
            embedded_timestep = embedded_timestep[:, _N_prefix:].contiguous()

        # Output processing
        out = self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, hidden, embedded_timestep)

        # Unpatchify
        out_bcfhw = patchifier.unpatchify(out, target_shape)

        return out_bcfhw

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.model_type.is_video_enabled():
            nn.init.xavier_uniform_(self.patchify_proj.weight)
            if self.caption_projection is not None:
                for m in self.caption_projection.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
            for m in self.adaln_single.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(self.proj_out.weight)

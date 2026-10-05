# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

"""Module for base_models -> three_dimensions -> point_clouds -> loger -> layers -> attention.py functionality."""

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch

from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.attention import (
    Attention as Attention,
    MemEffAttention as MemEffAttention,
    FlashAttention as FlashAttention,
    CrossAttentionRope as CrossAttentionRope,
    MemEffCrossAttentionRope as MemEffCrossAttentionRope,
    AttentionRope as AttentionRope,
    get_attn_score as get_attn_score,
    PRopeFlashAttention as PRopeFlashAttention,
    FlashCrossAttentionRope as FlashCrossAttentionRope,
)

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


# Cache for block masks to avoid recreation
_BLOCK_MASK_CACHE = {}


def get_causal_block_mask(P, B, H, M, N, device="cuda", _compile=True):
    """
    Get causal block mask with efficient caching based on logical parameters.
    
    Args:
        P: tokens per frame (image)
        B: batch size (not used in cache key since mask can be reused across batch sizes)
        H: number of heads
        M: query sequence length (num_frames * P)
        N: key sequence length (num_frames * P) 
        device: target device
        _compile: whether to compile
    
    Returns:
        Block mask where tokens within the same image can see each other,
        but tokens from different images can only see previous images.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        return None
    
    # Create cache key based on logical parameters
    device_idx = device.index if hasattr(device, 'index') else 0
    cache_key = (P, H, M, N, device_idx, _compile)
    
    if cache_key in _BLOCK_MASK_CACHE:
        cached_mask = _BLOCK_MASK_CACHE[cache_key]
        return cached_mask
    
    # Create the score function
    # Tokens within the same frame can see each other
    # Tokens from frame i can see all tokens from frames 0 to i
    def causal_mask(b, h, q_idx, kv_idx):
        """Causal mask.

        Args:
            b: The b.
            h: The h.
            q_idx: The q idx.
            kv_idx: The kv idx.
        """
        q_frame = q_idx // P
        kv_frame = kv_idx // P
        return q_frame >= kv_frame
    
    # Create new block mask
    block_mask = create_block_mask(causal_mask, B, H, M, N, device=device, _compile=_compile)
    
    # Cache it
    _BLOCK_MASK_CACHE[cache_key] = block_mask
    
    return block_mask

class MemEffAttentionRope(AttentionRope):
    """Mem eff attention rope implementation."""
    def forward(self, x: Tensor, attn_bias=None, xpos=None, attn_mask=None) -> Tensor:
        """Forward.

        Args:
            x: The x.
            attn_bias: The attn bias.
            xpos: The xpos.
            attn_mask: The attn mask.

        Returns:
            The return value.
        """
        # If attn_mask is provided and flex_attention is available, use flex_attention
        if attn_mask is not None and FLEX_ATTENTION_AVAILABLE:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
            q, k, v = [qkv[:,:,i] for i in range(3)]
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

            if self.rope is not None:
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)

            # Ensure all tensors have the same dtype
            target_dtype = v.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            
            x = flex_attention(
                q, k, v,
                block_mask=attn_mask,
                scale=None,
                enable_gqa=False,
                return_lse=False
            )
            x = x.transpose(1, 2).reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        # Otherwise use xformers memory_efficient_attention
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, attn_bias=attn_bias, xpos=xpos, attn_mask=attn_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv = qkv.transpose(1, 3)
        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class FlashAttentionRope(AttentionRope):
    """Flash attention rope implementation."""
    def compute_kv(self, x: Tensor, xpos=None) -> tuple[Tensor, Tensor]:
        """Compute K, V for caching. Returns (K, V) after norm and RoPE."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        
        if self.rope is not None:
            k = self.rope(k, xpos)
        
        return k, v
    
    def forward_with_kv_cache(
        self, 
        x: Tensor, 
        k_cache: Tensor, 
        v_cache: Tensor,
        xpos=None, 
        xpos_cache=None,
        attn_mask=None
    ) -> Tensor:
        """Forward with pre-computed KV cache for history tokens.
        
        Args:
            x: Current tokens [B, N_curr, C]
            k_cache: Cached K from history [B, num_heads, N_hist, head_dim]
            v_cache: Cached V from history [B, num_heads, N_hist, head_dim]
            xpos: Position info for current tokens
            xpos_cache: Position info for cached tokens (unused, positions already applied)
            attn_mask: Optional attention mask
        
        Returns:
            Output for current tokens only [B, N_curr, C]
        """
        B, N_curr, C = x.shape
        
        # Compute Q, K, V for current tokens
        qkv = self.qkv(x).reshape(B, N_curr, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        # Concatenate cached KV with current KV
        # k_cache, v_cache: [B, num_heads, N_hist, head_dim]
        # k, v: [B, num_heads, N_curr, head_dim]
        k_full = torch.cat([k_cache, k], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)
        
        # Compute attention
        is_float_mask = (attn_mask is not None and torch.is_floating_point(attn_mask))
        
        if attn_mask is not None and FLEX_ATTENTION_AVAILABLE and not is_float_mask:
            target_dtype = v_full.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k_full.dtype != target_dtype:
                k_full = k_full.to(target_dtype)
            
            x = flex_attention(
                q, k_full, v_full,
                block_mask=attn_mask,
                scale=None,
                enable_gqa=False,
                return_lse=False
            )
        else:
            if q.dtype == torch.bfloat16 and not is_float_mask:
                x = _worldfoundry_scaled_dot_product_attention(q, k_full, v_full, backend="flash")
            else:
                x = _worldfoundry_scaled_dot_product_attention(
                    q, k_full, v_full, attn_mask=attn_mask, backends=["math", "efficient"]
                )
        
        x = x.transpose(1, 2).reshape([B, N_curr, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, attn_bias=None, xpos=None, attn_mask=None) -> Tensor:
        """Forward.

        Args:
            x: The x.
            attn_bias: The attn bias.
            xpos: The xpos.
            attn_mask: The attn mask.

        Returns:
            The return value.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # If attn_mask (block_mask) is provided and flex_attention is available, use it
        # If attn_mask (block_mask) is provided and flex_attention is available, use it
        # [MODIFIED] Check if attn_mask is a float tensor (bias). If so, skip flex_attention
        # because flex_attention typically expects a BlockMask or boolean mask.
        is_float_mask = (attn_mask is not None and torch.is_floating_point(attn_mask))
        
        if attn_mask is not None and FLEX_ATTENTION_AVAILABLE and not is_float_mask:
            # Ensure all tensors have the same dtype for flex_attention
            target_dtype = v.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            
            x = flex_attention(
                q, k, v,
                block_mask=attn_mask,
                scale=None,  # flex_attention applies 1/sqrt(d) automatically
                enable_gqa=False,
                return_lse=False
            )
        else:
            # Use standard scaled_dot_product_attention
            if q.dtype == torch.bfloat16 and not is_float_mask:
                x = _worldfoundry_scaled_dot_product_attention(q, k, v, backend="flash")
            else:
                # Fallback to MATH/EFFICIENT if using float mask or other dtypes
                x = _worldfoundry_scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, backends=["math", "efficient"]
                )

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

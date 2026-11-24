# SPDX-License-Identifier: Apache-2.0
"""Cross-attention forwards with family-owned construction, providers and type identity.

Forcing and MinWM retain their native self-attention ancestry so MinWM's PRoPE
installer can distinguish self-attention from text/image cross-attention.
"""

from typing import Any

import torch
from torch import Tensor

from worldfoundry.core.attention.sequence.varlen import flash_attention

_FA3_FUNC: Any | None = None


def _get_fa3_func() -> Any:
    global _FA3_FUNC
    if _FA3_FUNC is None:
        _FA3_FUNC = __import__("flash_attn_interface", fromlist=["flash_attn_func"]).flash_attn_func
    return _FA3_FUNC


@torch.library.custom_op("worldfoundry::wan_flash_attn_3_bshd", mutates_args=(), device_types="cuda")
def wan_flash_attn_3_bshd(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Compile- and CUDA-Graph-safe Hopper FlashAttention-3 wrapper for ``[B, S, H, D]``."""
    fa3_func = _get_fa3_func()
    q_in = q if q.stride(-1) == 1 else q.contiguous()
    k_in = k if k.stride(-1) == 1 else k.contiguous()
    v_in = v if v.stride(-1) == 1 else v.contiguous()
    if q_in.dtype != v_in.dtype:
        q_in = q_in.to(v_in.dtype)
    if k_in.dtype != v_in.dtype:
        k_in = k_in.to(v_in.dtype)
    out = fa3_func(q_in, k_in, v_in)
    if isinstance(out, tuple):
        out = out[0]
    return out


@wan_flash_attn_3_bshd.register_fake
def _wan_flash_attn_3_bshd_fake(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    del k
    return torch.empty_like(q, dtype=v.dtype)


def _can_use_wan_fa3_bshd(q: Tensor, k: Tensor, v: Tensor) -> bool:
    """True when the inference-only Hopper FA3 custom op is safe for ``(q, k, v)``."""
    return (
        q.device.type == "cuda"
        and q.dtype in {torch.float16, torch.bfloat16}
        and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    )


class UncachedTextCrossAttentionMixin:
    """Text forward with the native provider bound as a static ``_cross_attention``."""

    def forward(self: Any, x: Tensor, context: Tensor, context_lens: Tensor | None) -> Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = self._cross_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class CachedTextCrossAttentionMixin:
    """Shared cached text forward; parameter construction remains family-owned."""

    def forward(
        self: Any,
        x: Tensor,
        context: Tensor,
        context_lens: Tensor | None,
        crossattn_cache: dict[str, Any] | None = None,
    ) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                existing_k = crossattn_cache.get("k")
                existing_v = crossattn_cache.get("v")
                if (
                    isinstance(existing_k, Tensor)
                    and isinstance(existing_v, Tensor)
                    and existing_k.shape == k.shape
                    and existing_v.shape == v.shape
                    and existing_k.device == k.device
                    and existing_v.device == v.device
                    and existing_k.dtype == k.dtype
                    and existing_v.dtype == v.dtype
                ):
                    existing_k.copy_(k)
                    existing_v.copy_(v)
                    k, v = existing_k, existing_v
                else:
                    crossattn_cache["k"] = k
                    crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        if (
            context_lens is None
            and getattr(self, "attention_backend", None) == "flash_attention_3"
            and _can_use_wan_fa3_bshd(q, k, v)
        ):
            x = wan_flash_attn_3_bshd(q, k, v)
        else:
            x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class ImageTextCrossAttentionMixin:
    """Shared image/text forward; parameter construction remains family-owned."""

    def forward(self: Any, x: Tensor, context: Tensor, context_lens: Tensor | None) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        use_fa3 = getattr(self, "attention_backend", None) == "flash_attention_3"
        if use_fa3 and _can_use_wan_fa3_bshd(q, k_img, v_img):
            img_x = wan_flash_attn_3_bshd(q, k_img, v_img)
        else:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        if context_lens is None and use_fa3 and _can_use_wan_fa3_bshd(q, k, v):
            x = wan_flash_attn_3_bshd(q, k, v)
        else:
            x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

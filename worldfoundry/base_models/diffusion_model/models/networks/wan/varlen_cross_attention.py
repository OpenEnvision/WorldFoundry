# SPDX-License-Identifier: Apache-2.0
"""Cross-attention forwards with family-owned construction, providers and type identity.

Forcing and MinWM retain their native self-attention ancestry so MinWM's PRoPE
installer can distinguish self-attention from text/image cross-attention.
"""

from typing import Any

from torch import Tensor

from worldfoundry.core.attention.sequence.varlen import flash_attention


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
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

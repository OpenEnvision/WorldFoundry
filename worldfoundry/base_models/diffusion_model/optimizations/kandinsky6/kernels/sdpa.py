"""PyTorch scaled dot-product attention."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def sdpa(q: Tensor, k: Tensor, v: Tensor, attn_mask=None) -> Tensor:
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2).contiguous()

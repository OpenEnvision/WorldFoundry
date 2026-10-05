"""Sequence metadata builders shared by variable-length attention kernels.

FlashAttention's packed ABI wants an int32 ``cu_seqlens`` vector.
Building it with a Python loop forces a host sync. This module
constructs those tensors on-device from a text mask plus an image
token count so varlen and sequence-parallel packed attention stay
on the GPU.

:func:`get_cu_seqlens` encodes each sample as a padded block of
``text_len + img_len``; odd offsets are the valid length and even
offsets are the padded end.

Not this module:
    Does not launch attention. Packed dataclass ABI lives in
    :mod:`.packed_sequence`. Kernel dispatch lives in :mod:`.varlen`.

Public surface:

- :func:`get_cu_seqlens` — on-device alternating valid/pad offsets.
"""

from __future__ import annotations

import torch


# ──────────────────────────────────────────────────────────────────────────
# Packed offsets — keep construction on-device to avoid a host sync
# ──────────────────────────────────────────────────────────────────────────


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int) -> torch.Tensor:
    """Build packed text/image cumulative lengths without host-side loops.

    Each sample occupies a padded block of ``text_mask.shape[1] + img_len``
    tokens. The odd offsets mark its valid text-plus-image length and the even
    offsets mark the end of the padded block, matching FlashAttention's int32
    cumulative-length ABI.
    """

    if text_mask.ndim != 2:
        raise ValueError(f"text_mask must be two-dimensional, got shape {tuple(text_mask.shape)}")
    img_len = int(img_len)
    if img_len < 0:
        raise ValueError(f"img_len must be non-negative, got {img_len}")

    batch_size, padded_text_len = text_mask.shape
    max_len = padded_text_len + img_len
    if batch_size * max_len > torch.iinfo(torch.int32).max:
        raise OverflowError("packed attention sequence exceeds the int32 offset range")

    text_lens = text_mask.sum(dim=1, dtype=torch.int32)
    block_starts = torch.arange(batch_size, dtype=torch.int32, device=text_mask.device) * max_len
    offsets = torch.stack(
        (block_starts + text_lens + img_len, block_starts + max_len),
        dim=1,
    ).reshape(-1)
    return torch.cat((torch.zeros(1, dtype=torch.int32, device=text_mask.device), offsets))


__all__ = ["get_cu_seqlens"]

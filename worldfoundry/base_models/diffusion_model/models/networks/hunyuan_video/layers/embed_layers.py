"""HunyuanVideo 3D patch embed shared by the i2v MM-DiT.

Despite the class docstring inherited from ViT, ``proj`` is Conv3d:
``B C T H W`` → flatten → ``B (T'H'W') C``.  Used when the i2v model
imports :class:`PatchEmbed` from this package rather than ``original.py``.
"""

import torch.nn as nn

from worldfoundry.core.nn.blocks.layers import to_2tuple


class PatchEmbed(nn.Module):
    """Conv3d patch embed: ``B C T H W`` → ``B (T'H'W') C`` (optional flatten)."""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs
        )
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Project then optionally flatten ``B C T H W`` patches to ``B S C``."""
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

# Adapted from Tencent Hunyuan-GameCraft-1.0, under the Tencent Hunyuan Community License.
"""Checkpoint-compatible patch projection with concatenated image/mask channels."""

from ..layers.embed_layers import PatchEmbed as SharedPatchEmbed


class PatchEmbed(SharedPatchEmbed):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, multitask_mask_training_type=None, **kwargs):
        if multitask_mask_training_type == "concat":
            in_chans = 2 * in_chans + 1
        super().__init__(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, **kwargs)

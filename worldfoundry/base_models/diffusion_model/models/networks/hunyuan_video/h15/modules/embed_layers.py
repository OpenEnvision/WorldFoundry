# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""HunyuanVideo 1.5 patch embed and conditioning projections.

These layers sit at the DiT stem: :class:`PatchEmbed` is a Conv3d that
maps ``B C T H W`` latents (optionally concatenated with condition /
mask channels) to flattened tokens ``B (T'H'W') C``.  Extra concat
weights are zero-init so a text-to-video checkpoint stays a no-op until
image-to-video finetune fills them.

:class:`ByT5Mapper`, :class:`VisionProjection`, and
:class:`ClipVisionProjection` map glyph / vision encoder features into
DiT width.  CLIP's down-projection starts at zero so unused vision
tokens do not change a t2v graph.
"""

import torch
import torch.nn as nn

from worldfoundry.core.nn.blocks.layers import to_2tuple


class ByT5Mapper(nn.Module):
    """Checkpoint-visible projection from ByT5 features into DiT width."""

    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_residual=True):
        super().__init__()
        if use_residual and in_dim != out_dim:
            raise ValueError("ByT5 residual projection requires in_dim == out_dim")
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, value):
        """Map ByT5 glyph tokens ``B L C_in`` to DiT width (optional residual)."""
        residual = value
        value = self.act_fn(self.fc1(self.layernorm(value)))
        value = self.fc3(self.act_fn(self.fc2(value)))
        return value + residual if self.use_residual else value


class PatchEmbed(nn.Module):
    """HY 1.5 Conv3d patch embed: ``B C T H W`` → ``B (T'H'W') C``.

    When ``concat_condition`` is set the conv sees extra condition/mask
    channels; those extra input weights are zero-init so a t2v checkpoint
    stays a no-op until i2v finetune fills them.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        is_reshape_temporal_channels=False,
        concat_condition=True,
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

        # Only support concat mode (multitask mask training)
        orig_in_chans = in_chans
        if concat_condition:
            if is_reshape_temporal_channels:
                in_chans = in_chans + in_chans//2 + 1
            else:
                in_chans = in_chans * 2 + 1

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )

        nn.init.xavier_uniform_(self.proj.weight[:, :orig_in_chans].view(self.proj.weight[:, :orig_in_chans].size(0), -1))
        # Special initialization for concat mode
        nn.init.zeros_(self.proj.weight[:, orig_in_chans:].view(self.proj.weight[:, orig_in_chans:].size(0), -1))

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Project then optionally flatten patches to ``B S C``."""
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionProjection(torch.nn.Module):
    """LN-GELU MLP that maps vision-encoder tokens into DiT hidden size."""

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim), 
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(), 
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim)
        )
        

    def forward(self, vision_embeds):
        """Project ``B L C_vis`` vision tokens to ``B L C_dit``."""
        return self.proj(vision_embeds)

class ClipVisionProjection(nn.Module):
    """Zero-init down-projection of CLIP vision features into DiT width."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, x):
        """``down(silu(up(x)))``; down weights start at zero so CLIP is a no-op until trained."""
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x

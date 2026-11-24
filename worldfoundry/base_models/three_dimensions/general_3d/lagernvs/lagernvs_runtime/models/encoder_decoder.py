# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT

from .renderer import Renderer

# Main model file
# Consists of
# 1. Encoder (Reconstructor)
#    VGGT-based feature extraction
# 2. Decoder (Renderer)
#    Series of (Self-attn, X-attn, MLP) blocks


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        depth,
        hidden_size,
        patch_size,
        num_heads,
        freeze_vggt=True,
        pretrained_vggt=True,
        attention_to_features_type="bidirectional_cross_attention",
        pretrained_patch_embed=False,
    ):
        super().__init__()
        self.reconstructor = Reconstructor(
            hidden_size,
            target_patch_size=patch_size,
            pretrained_vggt=pretrained_vggt,
            freeze_vggt=freeze_vggt,
            pretrained_patch_embed=pretrained_patch_embed,
        )
        self.renderer = Renderer(
            depth,
            hidden_size,
            patch_size,
            num_heads,
            attention_to_features_type=attention_to_features_type,
        )

    def forward(
        self,
        images,
        rays,
        cam_token,
        num_cond_views,
        timeit=False,
    ):
        input_images = images[:, :num_cond_views, ...]
        cam_token = cam_token[:, :num_cond_views]
        target_rays = rays[:, num_cond_views:]

        v_target = target_rays.shape[1]

        rec_tokens = self.reconstructor(input_images, cam_token)

        rec_tokens = einops.rearrange(rec_tokens, "b v_input p c -> b (v_input p) c")
        rec_tokens = einops.repeat(
            rec_tokens,
            "b np d -> (b v_target) np d",
            v_target=v_target,
        )

        if timeit:
            rendered_images, time_t = self.renderer(rec_tokens, target_rays, timeit=timeit)
        else:
            rendered_images = self.renderer(rec_tokens, target_rays, timeit=timeit)

        cond_and_rendered_images = torch.cat([input_images, rendered_images], dim=1)

        if timeit:
            return cond_and_rendered_images, time_t

        return cond_and_rendered_images


class Reconstructor(nn.Module):
    """Reconstructor module. Extracts generalisable reconstruction features."""

    def __init__(
        self,
        renderer_hidden_size,
        target_patch_size,
        pretrained_vggt=True,
        freeze_vggt=False,
        pretrained_patch_embed=False,
    ):
        super().__init__()
        self.vggt = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            pretrained_patch_embed=pretrained_patch_embed,
            return_aggregated_tokens=True,
        )
        self.freeze_vggt = freeze_vggt
        if pretrained_vggt:
            print("Loading encoder weights from pretrained VGGT")
            vggt_pretrained_state = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
                map_location="cpu",
            )
            self.vggt.load_state_dict(vggt_pretrained_state, strict=False)
        else:
            print("VGGT weights not used for the encoder")

        # camera token projector (always use 11-dim tokens with scale)
        self.camera_encoding_dim = 11
        vggt_hidden_dim = 1024
        self.vggt_patch_size = 14
        self.target_patch_size = target_patch_size
        self.camera_mlp = nn.Sequential(
            nn.Linear(self.camera_encoding_dim, vggt_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(vggt_hidden_dim, vggt_hidden_dim, bias=True),
        )

        # channel-dim adapter
        self.geo_feature_connector = nn.Linear(1024 * 2, renderer_hidden_size)
        self.geo_feature_norm = nn.LayerNorm(renderer_hidden_size, bias=False)

    def forward(self, input_images, cam_token):
        """
        Inputs:
            images: (b, v_input, 3, h, w) input images
            cam_token: (b, v_input, 9) camera conditioning, possibly all-zero when camera
              not available
        """
        # resize input images so that longer size is 518
        b, v_input, _, h, w = input_images.shape
        input_images = einops.rearrange(input_images, "b v c h w -> (b v) c h w")
        vggt_imsize = 518
        input_camera_cond = self.camera_mlp(cam_token).unsqueeze(2)

        # resize input images so that the side length is divisible by 14
        if h > w:
            tgt_h = vggt_imsize
            tgt_w = (int(tgt_h * w / h) // self.vggt_patch_size) * self.vggt_patch_size
        else:
            tgt_w = vggt_imsize
            tgt_h = (int(tgt_w * h / w) // self.vggt_patch_size) * self.vggt_patch_size
        input_images = F.interpolate(input_images, size=(tgt_h, tgt_w), mode="bilinear", antialias=True)
        input_images = einops.rearrange(input_images, "(b v) c h w -> b v c h w", b=b, v=v_input)
        # extract features for the conditioning images
        if self.freeze_vggt:
            with torch.no_grad():
                tokens_vggt_cond = self.vggt(input_images, cameras_possibly_zero=input_camera_cond).detach()
        else:
            tokens_vggt_cond = self.vggt(input_images, cameras_possibly_zero=input_camera_cond)

        tokens_vggt_image_cond = tokens_vggt_cond[:, :, self.vggt.aggregator.patch_start_idx :, :]

        tokens_vggt_image_cond = self.geo_feature_connector(tokens_vggt_image_cond)
        tokens_vggt_image_cond = self.geo_feature_norm(tokens_vggt_image_cond)

        return tokens_vggt_image_cond


def EncDec_VitB8(**kwargs):
    return EncoderDecoder(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

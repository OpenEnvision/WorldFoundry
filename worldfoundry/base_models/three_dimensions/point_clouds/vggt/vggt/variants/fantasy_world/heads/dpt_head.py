# Copyright Alibaba Inc. All Rights Reserved.
# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.geometry_bridge import WanVAE_
from worldfoundry.base_models.three_dimensions.depth.midas.fusion import custom_interpolate as custom_interpolate
from worldfoundry.base_models.three_dimensions.depth.midas.fusion import make_grouped_fusion_block as _make_fusion_block
from worldfoundry.base_models.three_dimensions.depth.midas.fusion import make_scratch as _make_scratch
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.heads.head_act import activate_head
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.heads.utils import (
    create_uv_grid,
    position_grid_to_embed,
)


class DPTHead_3D_Causal(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],  # noqa: B006 - Keeps the upstream default object and signature.
        intermediate_layer_idx: list[int] = [23, 17, 11, 7],  # noqa: B006 - Keeps the upstream default object and signature.
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        temporal_scale: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.temporal_scale = temporal_scale

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        self.temporal_upsamplers = nn.ModuleList(
            [
                WanVAE_(z_dim=out_channels[0], location="DPT"),
                WanVAE_(z_dim=out_channels[1], location="DPT"),
                WanVAE_(z_dim=out_channels[2], location="DPT"),
                WanVAE_(z_dim=out_channels[3], location="DPT"),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size_first: int = 4,
        frames_chunk_size_second: int = 16,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, S, _, H, W = images.shape
        all_preds = []
        all_conf = []
        feature_1_list, feature_2_list, feature_3_list, feature_4_list = [], [], [], []
        for frames_start_idx in range(0, S, frames_chunk_size_first):
            frames_end_idx = min(frames_start_idx + frames_chunk_size_first, S)
            feature_1, feature_2, feature_3, feature_4 = self._forward_impl_first_part(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            )
            feature_1_list.append(rearrange(feature_1, "(B T) C H W -> B C T H W", B=B))
            feature_2_list.append(rearrange(feature_2, "(B T) C H W -> B C T H W", B=B))
            feature_3_list.append(rearrange(feature_3, "(B T) C H W -> B C T H W", B=B))
            feature_4_list.append(rearrange(feature_4, "(B T) C H W -> B C T H W", B=B))
        feature_1_seq = torch.cat(feature_1_list, dim=2)
        feature_2_seq = torch.cat(feature_2_list, dim=2)
        feature_3_seq = torch.cat(feature_3_list, dim=2)
        feature_4_seq = torch.cat(feature_4_list, dim=2)
        out_1 = self.temporal_upsamplers[0].decode(feature_1_seq)
        out_2 = self.temporal_upsamplers[1].decode(feature_2_seq)
        out_3 = self.temporal_upsamplers[2].decode(feature_3_seq)
        out_4 = self.temporal_upsamplers[3].decode(feature_4_seq)
        out = [out_1, out_2, out_3, out_4]
        for frames_start_idx in range(0, (S - 1) * 4 + 1, frames_chunk_size_second):
            frames_end_idx = min(frames_start_idx + frames_chunk_size_second, (S - 1) * 4 + 1)
            sub_out = [i[:, :, frames_start_idx:frames_end_idx, ...] for i in out]
            sub_out = [rearrange(i, "B C T H W -> (B T) C H W") for i in sub_out]
            if self.feature_only:
                chunk_output = self.__forward_impl_second_part(
                    sub_out,
                    images,
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self.__forward_impl_second_part(
                    sub_out,
                    images,
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl_first_part(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, spatial_H, spatial_W, C = images.shape

        patch_h, patch_w = spatial_H, spatial_W
        H = spatial_H * self.patch_size
        W = spatial_W * self.patch_size
        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1  # noqa: SIM113 - Keeps the explicit loop counter.
        return out

    def __forward_impl_second_part(
        self,
        out: list[torch.Tensor],
        images: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, S, spatial_H, spatial_W, C = images.shape
        H = spatial_H * self.patch_size
        W = spatial_W * self.patch_size
        out = self.scratch_forward(out)
        out = custom_interpolate(
            out,
            (int(spatial_H * self.patch_size / self.down_ratio), int(spatial_W * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])
        out = self.scratch.output_conv2(out)

        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        new_S = out.shape[0] // B
        preds = preds.view(B, new_S, *preds.shape[1:])
        conf = conf.view(B, new_S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)

        return out

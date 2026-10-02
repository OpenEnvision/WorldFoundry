# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""SANA multi-scale ControlNet side-branch.

Copies :class:`SanaMSBlock` stacks from a frozen SanaMS backbone and injects
control residuals through ``before_proj`` / ``after_proj`` linears.  The
copied blocks keep the same linear / Triton-linear / FlashAttention and
GLUMBConv choices as the teacher so control checkpoints load by name.
"""

from copy import deepcopy

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import torch
import torch.nn as nn
from torch.nn import Linear, Module, init

from worldfoundry.base_models.diffusion_model.models.networks.sana.sana import get_2d_sincos_pos_embed
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import RopePosEmbed
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale import (
    SanaMS,
    SanaMSBlock,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.capabilities import (
    is_triton_module_available,
    is_xformers_available,
)
from worldfoundry.core.model_loading.checkpoints import load_tensor_state_dict

_triton_modules_available = False
if is_triton_module_available():
    from worldfoundry.base_models.diffusion_model.models.networks.sana.fastlinear.modules import (
        TritonLiteMLA,
        TritonMBConvPreGLU,
    )

    _triton_modules_available = True

_xformers_available = False
if is_xformers_available():
    _xformers_available = True


class ControlSanaMSBlock(Module):
    """Copied SanaMSBlock plus ``before_proj`` / ``after_proj`` control residuals."""

    def __init__(self, base_block: SanaMSBlock, block_index: int) -> None:
        """Init.

        Args:
            base_block: The base block.
            block_index: The block index.

        Returns:
            The return value.
        """
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index
        self.hidden_size = hidden_size = base_block.hidden_size
        if self.block_index == 0:
            self.before_proj = Linear(hidden_size, hidden_size)

        self.after_proj = Linear(hidden_size, hidden_size)

    def initialize_all_and_copy_from_base(self, base_block):
        """Initialize all and copy from base.

        Args:
            base_block: The base block.
        """
        for name, param in self.named_parameters():
            param.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.train()

        if self.block_index == 0:
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)

        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, control_signal, mask=None, HW=None, image_rotary_emb=None):
        """Forward.

        Args:
            x: The x.
            y: The y.
            t: The t.
            control_signal: The control signal.
            mask: The mask.
            HW: The hw.
            image_rotary_emb: The image rotary emb.
        """
        if self.block_index == 0:
            # the first block
            control_signal = self.before_proj(control_signal)
            control_signal = self.copied_block(x + control_signal, y, t, mask, HW, image_rotary_emb)
            control_signal_skip = self.after_proj(control_signal)
        else:
            # load from previous control_signal and produce the control_signal for skip connection
            control_signal = self.copied_block(control_signal, y, t, mask, HW, image_rotary_emb)
            control_signal_skip = self.after_proj(control_signal)

        return control_signal, control_signal_skip


class SanaMSControlNet(SanaMS):
    """SanaMS plus a trainable ControlNet side-branch of copied blocks."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=4096,
        pe_interpolation=1.0,
        config=None,
        model_max_length=300,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="flash",
        ffn_type="mlp",
        use_pe=True,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "gelu", None),
        linear_head_dim=32,
        copy_blocks_num=7,
        cross_norm=False,
        timestep_norm_scale_factor=1.0,
        **kwargs,
    ):
        """Init.

        Args:
            input_size: The input size.
            patch_size: The patch size.
            in_channels: The in channels.
            hidden_size: The hidden size.
            depth: The depth.
            num_heads: The num heads.
            mlp_ratio: The mlp ratio.
            class_dropout_prob: The class dropout prob.
            learn_sigma: The learn sigma.
            pred_sigma: The pred sigma.
            drop_path: The drop path.
            caption_channels: The caption channels.
            pe_interpolation: The pe interpolation.
            config: The config.
            model_max_length: The model max length.
            qk_norm: The qk norm.
            y_norm: The y norm.
            norm_eps: The norm eps.
            attn_type: The attn type.
            ffn_type: The ffn type.
            use_pe: The use pe.
            y_norm_scale_factor: The y norm scale factor.
            patch_embed_kernel: The patch embed kernel.
            mlp_acts: The mlp acts.
            linear_head_dim: The linear head dim.
            copy_blocks_num: The copy blocks num.
            cross_norm: The cross norm.
            timestep_norm_scale_factor: The timestep norm scale factor.
        """
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            caption_channels=caption_channels,
            pe_interpolation=pe_interpolation,
            config=config,
            model_max_length=model_max_length,
            qk_norm=qk_norm,
            y_norm=y_norm,
            norm_eps=norm_eps,
            attn_type=attn_type,
            ffn_type=ffn_type,
            use_pe=use_pe,
            y_norm_scale_factor=y_norm_scale_factor,
            patch_embed_kernel=patch_embed_kernel,
            mlp_acts=mlp_acts,
            linear_head_dim=linear_head_dim,
            cross_norm=cross_norm,
            timestep_norm_scale_factor=timestep_norm_scale_factor,
            **kwargs,
        )
        # define controlnet
        self.copy_blocks_num = copy_blocks_num
        self.controlnet = nn.ModuleList([ControlSanaMSBlock(self.blocks[i], i) for i in range(copy_blocks_num)])

    def load_pretrain_and_initialize(self, model_path):
        """Load pretrain and initialize.

        Args:
            model_path: The model path.
        """
        missing, unexpected = self.load_state_dict(
            load_tensor_state_dict(model_path, wrapper_keys=("state_dict",)), strict=False
        )
        self.initialize_all()
        return missing, unexpected

    def initialize_all(self):
        """Initialize all."""
        # freeze all the parameters
        for p in self.parameters():
            p.requires_grad_(False)

        for i, block in enumerate(self.controlnet):
            block.initialize_all_and_copy_from_base(self.blocks[i])

    def forward_controlnet(self, control_signal, pos_embed_ms=None):
        """Forward controlnet.

        Args:
            control_signal: The control signal.
            pos_embed_ms: The pos embed ms.
        """
        if self.use_pe and pos_embed_ms:
            control_signal = self.x_embedder(control_signal) + pos_embed_ms
        else:
            control_signal = self.x_embedder(control_signal)
        return control_signal

    def forward(self, x, timestep, y, mask=None, data_info=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """

        bs = x.shape[0]
        x = x.to(self.dtype)
        if self.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / self.timestep_norm_scale_factor).to(self.dtype)
        else:
            timestep = timestep.long().to(self.dtype)
        y = y.to(self.dtype)

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x)
        image_pos_embed = None

        if self.use_pe:
            if self.pos_embed_type == "sincos":
                if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x.shape[1:]:
                    self.pos_embed_ms = (
                        torch.from_numpy(
                            get_2d_sincos_pos_embed(
                                self.pos_embed.shape[-1],
                                (self.h, self.w),
                                pe_interpolation=self.pe_interpolation,
                                base_size=self.base_size,
                            )
                        )
                        .unsqueeze(0)
                        .to(x.device)
                        .to(self.dtype)
                    )
                x += self.pos_embed_ms  # (N, T, D), where T = H * W / patch_size ** 2
            elif self.pos_embed_type == "3d_rope":
                self.pos_embed_ms = RopePosEmbed(theta=10000, axes_dim=[0, 16, 16])
                latent_image_ids = self.pos_embed_ms._prepare_latent_image_ids(bs, self.h, self.w, x.device, x.dtype)
                image_pos_embed = self.pos_embed_ms(latent_image_ids)
            else:
                raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")

        # control signal branch
        control_signal = data_info["control_signal"].to(self.dtype)
        control_signal = self.forward_controlnet(control_signal, pos_embed_ms=image_pos_embed)

        t = self.t_embedder(timestep)  # (N, D)

        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training, mask=mask)  # (N, D)
        if self.y_norm:
            y = self.attention_y_norm(y)

        if mask is not None:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1) if mask.shape[0] != y.shape[0] else mask
            mask = mask.squeeze(1).squeeze(1)
            if _xformers_available:
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask
        elif _xformers_available:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            raise ValueError(f"Attention type is not available due to _xformers_available={_xformers_available}.")

        x = self.blocks[0](x, y, t0, y_lens, (self.h, self.w), image_pos_embed, **kwargs)

        for i in range(1, self.copy_blocks_num + 1):
            control_signal, control_signal_skip = self.controlnet[i - 1](
                x, y, t0, control_signal, y_lens, (self.h, self.w), image_pos_embed, **kwargs
            )
            x = self.blocks[i](
                x + control_signal_skip, y, t0, y_lens, (self.h, self.w), image_pos_embed, **kwargs
            )

        for i in range(self.copy_blocks_num + 1, len(self.blocks)):
            x = self.blocks[i](x, y, t0, y_lens, (self.h, self.w), image_pos_embed, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        control_signal = data_info["control_signal"]
        assert control_signal is not None, "control_signal is required for dpm solver"
        assert control_signal.dim() == 4, "control_signal should be a 4D tensor"

        if x.shape[0] != control_signal.shape[0]:
            control_signal = control_signal.repeat(x.shape[0] // control_signal.shape[0], 1, 1, 1)

        assert control_signal.shape[0] == x.shape[0], "control_signal and x should have the same batch size"
        data_info["control_signal"] = control_signal

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out


#################################################################################
#                                   Sana Multi-scale Configs                              #
#################################################################################


def SanaMSControlNet_600M_P1_D28(**kwargs):
    """Sanamscontrolnet 600m p1 d28."""
    return SanaMSControlNet(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


def SanaMSControlNet_1600M_P1_D20(**kwargs):
    """Sanamscontrolnet 1600m p1 d20."""
    return SanaMSControlNet(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)

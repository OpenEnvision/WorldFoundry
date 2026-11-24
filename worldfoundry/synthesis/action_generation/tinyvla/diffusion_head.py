"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""

import itertools
import math

import torch
import torch.nn as nn

from worldfoundry.base_models.llm_mllm_core.action.diffusion_policy import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)

# =================== UNet for Diffusion ==============


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion models.

    Args:
        dim: The dimension of the embedding.
        dtype: The data type for the embedding.
    """

    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass to compute the sinusoidal positional embedding.

        Args:
            x: Input tensor.

        Returns:
            The sinusoidal positional embedding.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=self.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    """
    Conditional residual block with FiLM modulation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        cond_dim: Dimension of the conditioning input.
        kernel_size: Size of the convolutional kernel.
        n_groups: Number of groups for GroupNorm.
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1)))

        # ensure dimensions are compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Forward pass for the conditional residual block.

        Args:
            x: Input tensor of shape [batch_size, in_channels, horizon].
            cond: Conditioning tensor of shape [batch_size, cond_dim].

        Returns:
            Output tensor of shape [batch_size, out_channels, horizon].
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    Conditional 1D UNet for diffusion models.

    Args:
        input_dim: Dimension of the input actions.
        global_cond_dim: Dimension of global conditioning applied with FiLM.
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k.
        down_dims: Channel size for each UNet level.
        kernel_size: Convolutional kernel size.
        n_groups: Number of groups for GroupNorm.
        state_dim: Dimension of the state input.
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],  # noqa: B006 - Keeps the upstream default object and signature.
        kernel_size=5,
        n_groups=8,
        state_dim=7,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        start_dim = down_dims[0]

        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.norm_after_pool = nn.LayerNorm(global_cond_dim)
        self.combine = nn.Linear(global_cond_dim + state_dim, global_cond_dim)

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            # Compute the parameter-free timestep basis in fp32, then cast it
            # to the checkpoint's first Linear dtype in ``forward``.  This is
            # supported on Turing as well as Ampere-and-newer GPUs.
            SinusoidalPosEmb(dsed, torch.float32),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(itertools.pairwise(all_dims))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(f"number of parameters: {sum(p.numel() for p in self.parameters()):e}")

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | float | int, global_cond=None, states=None):
        """
        Forward pass for the Conditional UNet.

        Args:
            sample: Input tensor of shape (B, T, input_dim).
            timestep: Diffusion step, can be a tensor or an integer.
            global_cond: Global conditioning tensor of shape (B, global_cond_dim).
            states: Optional state tensor.

        Returns:
            Output tensor of shape (B, T, input_dim).
        """
        # move axis for processing
        sample = sample.moveaxis(-1, -2)
        # process global conditioning
        global_cond = self.global_1d_pool(global_cond.permute(0, 2, 1)).squeeze(-1)
        global_cond = self.norm_after_pool(global_cond)  # layernorm
        global_cond = torch.cat([global_cond, states], dim=-1) if states is not None else global_cond
        global_cond = self.combine(global_cond)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # SinusoidalPosEmb hardcodes bf16 output
        # cast to the next Linear's dtype so eval with torch_dtype=float16 doesn't crash on matmul.
        step_encoder_layers = list(self.diffusion_step_encoder)
        global_feature = step_encoder_layers[0](timesteps)
        if len(step_encoder_layers) > 1:
            global_feature = global_feature.to(step_encoder_layers[1].weight.dtype)
        for layer in step_encoder_layers[1:]:
            global_feature = layer(global_feature)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for _idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for _idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x

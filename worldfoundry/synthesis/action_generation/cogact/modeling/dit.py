# Inference-only CogACT source retained in-tree.
# Modified from facebookresearch's DiT repos
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import torch
import torch.nn as nn

from worldfoundry.base_models.llm_mllm_core.action.cogact import (
    ActionEmbedder,
    DiTBlock,
    FinalLayer,
    LabelEmbedder,
)
from worldfoundry.base_models.llm_mllm_core.action.cogact import (
    ActionEmbedder as HistoryEmbedder,
)

#################################################################################
#               Embedding Layers for Timesteps and conditions                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                      Embedding Layers for Actions and                         #
#################################################################################


# Action_History is not used now


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels=7,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        token_size=4096,
        future_action_window_size=1,
        past_action_window_size=0,
        learn_sigma=False,
    ):
        super().__init__()

        assert past_action_window_size == 0, "Error: action_history is not used now"

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # Action history is not used now.
        self.history_embedder = HistoryEmbedder(action_size=in_channels, hidden_size=hidden_size)

        self.x_embedder = ActionEmbedder(action_size=in_channels, hidden_size=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=hidden_size, dropout_prob=class_dropout_prob)
        scale = hidden_size**-0.5

        # Learnable positional embeddings
        # +2, one for the conditional token, and one for the current action prediction
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(future_action_window_size + past_action_window_size + 2, hidden_size)
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize token_embed like nn.Linear
        nn.init.normal_(self.x_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.x_embedder.linear.bias, 0)

        nn.init.normal_(self.history_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.history_embedder.linear.bias, 0)

        # Initialize label embedding table:
        if self.class_dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, z):
        """
        Forward pass of DiT.
        history: (N, H, D) tensor of action history # not used now
        x: (N, T, D) tensor of predicting action inputs
        t: (N,) tensor of diffusion timesteps
        z: (N, 1, D) tensor of conditions
        """
        x = self.x_embedder(x)  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        z = self.z_embedder(z)  # (N, 1, D)
        c = t.unsqueeze(1) + z  # (N, 1, D)
        x = torch.cat((c, x), dim=1)  # (N, T+1, D)
        x = x + self.positional_embedding  # (N, T+1, D)
        for block in self.blocks:
            x = block(x)  # (N, T+1, D)
        x = self.final_layer(x)  # (N, T+1, out_channels)
        return x[:, 1:, :]  # (N, T, C)

    def forward_with_cfg(self, x, t, z, cfg_scale):
        """
        Forward pass of Diffusion, but also batches the unconditional forward pass for classifier-free guidance.
        """

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)
        model_out = self.forward(combined, t, z)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)
        return torch.cat([eps, rest], dim=2)

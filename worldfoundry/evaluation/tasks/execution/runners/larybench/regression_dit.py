import math

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, dimension, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, value):
        batch_size, token_count, dimension = value.shape
        query, key, value = (
            self.qkv(value)
            .reshape(
                batch_size,
                token_count,
                3,
                self.num_heads,
                dimension // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        value = F.scaled_dot_product_attention(query, key, value)
        return self.projection(value.transpose(1, 2).reshape(batch_size, token_count, dimension))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timestep, dimension, max_period=10000):
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if dimension % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timestep):
        frequencies = self.timestep_embedding(timestep, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        return self.mlp(frequencies)


class ConditionEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_probability=0.1):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.dropout_probability = dropout_probability
        self.unconditioned = nn.Parameter(torch.empty(1, input_size))

    def forward(self, condition, training):
        if training and self.dropout_probability:
            drop = torch.rand(condition.shape[0], device=condition.device) < self.dropout_probability
            condition = torch.where(drop[:, None, None], self.unconditioned[None], condition)
        return self.linear(condition)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, value):
        value = value + self.attention(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class DiT(nn.Module):
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
    ):
        super().__init__()
        self.action_embedder = nn.Linear(in_channels, hidden_size)
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.condition_embedder = ConditionEmbedder(token_size, hidden_size, class_dropout_prob)
        self.position = nn.Parameter(hidden_size**-0.5 * torch.randn(future_action_window_size + 1, hidden_size))
        self.blocks = nn.ModuleList(DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_projection = nn.Linear(hidden_size, in_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.action_embedder.weight, std=0.02)
        nn.init.normal_(self.condition_embedder.unconditioned, std=0.02)
        nn.init.normal_(self.condition_embedder.linear.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, action, timestep, condition):
        action = self.action_embedder(action)
        timestep = self.timestep_embedder(timestep).unsqueeze(1)
        condition = self.condition_embedder(condition, self.training)
        value = torch.cat((timestep + condition, action), dim=1) + self.position
        for block in self.blocks:
            value = block(value)
        value = self.final_projection(self.final_norm(value))
        return value[:, 1:]

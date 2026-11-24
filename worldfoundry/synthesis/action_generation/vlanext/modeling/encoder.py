"""Action and proprioception encoder modules."""

import torch
import torch.nn as nn


class ActionTransformerProjector(nn.Module):
    def __init__(self, action_dim, hidden_size, depth=2, num_heads=4, mlp_ratio=4.0, max_len=64):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.input_proj(x)
        seq_len = x.shape[1]

        eff_len = min(seq_len, self.pos_embed.shape[1])
        pos_embed_slice = self.pos_embed[:, :eff_len, :]

        if seq_len > eff_len:
            x[:, :eff_len, :] = x[:, :eff_len, :] + pos_embed_slice
        else:
            x = x + pos_embed_slice

        x = self.encoder(x)
        x = self.norm(x)
        return x


class ActionTransformerDecoder(nn.Module):
    def __init__(self, action_dim, hidden_size, depth=1, num_heads=4, mlp_ratio=4.0, max_len=64):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        seq_len = x.shape[1]

        eff_len = min(seq_len, self.pos_embed.shape[1])
        pos_embed_slice = self.pos_embed[:, :eff_len, :]

        if seq_len > eff_len:
            x[:, :eff_len, :] = x[:, :eff_len, :] + pos_embed_slice
        else:
            x = x + pos_embed_slice

        x = self.decoder(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return x

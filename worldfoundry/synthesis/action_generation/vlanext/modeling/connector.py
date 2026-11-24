"""VLANeXt connector module."""

import torch.nn as nn


class ConnectorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, depth=2, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim)
        else:
            self.input_proj = nn.Identity()

        hidden_size = output_dim

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
        x = self.encoder(x)
        x = self.norm(x)
        return x

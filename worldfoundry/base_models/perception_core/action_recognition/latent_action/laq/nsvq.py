"""Noise-substitution vector quantizer used by LAQ extraction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NSVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        num_embeddings: int,
        embedding_dim: int,
        device: str | torch.device = "cpu",
        code_seq_len: int = 1,
        patch_size: int = 32,
        image_size: int = 256,
        **_: object,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.eps = 1e-12

        self.codebooks = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device))
        self.project_in = nn.Linear(dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, dim)

        if code_seq_len == 1:
            layers = [
                nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, 4),
            ]
        elif code_seq_len == 2:
            layers = [
                nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, (3, 4)),
            ]
        elif code_seq_len == 4:
            layers = [
                nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, 3),
            ]
        elif code_seq_len == 16:
            layers = [
                nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),
            ]
        elif code_seq_len == 49:
            layers = [nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1)]
        elif code_seq_len == 64:
            layers = [nn.Conv2d(embedding_dim, embedding_dim, 2, stride=2, padding=1)]
        elif code_seq_len == 256:
            layers = [nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1)]
        else:
            raise ValueError("code_seq_len must be one of 1, 2, 4, 16, 49, 64, or 256")
        self.cnn_encoder = nn.Sequential(*layers)

    def _encode(self, input_data: Tensor) -> Tensor:
        batch_size = input_data.shape[0]
        spatial_size = self.image_size // self.patch_size
        input_data = self.project_in(input_data)
        input_data = rearrange_to_image(input_data, batch_size, self.embedding_dim, spatial_size)
        input_data = self.cnn_encoder(input_data)
        return input_data.flatten(2).transpose(1, 2).reshape(-1, self.embedding_dim)

    def forward(self, input_data_first: Tensor, input_data_last: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = input_data_first.shape[0]
        input_data = self._encode(input_data_last) - self._encode(input_data_first)

        distances = (
            input_data.square().sum(dim=1, keepdim=True)
            - 2 * input_data @ self.codebooks.t()
            + self.codebooks.square().sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        hard_quantized = self.codebooks[indices]

        random_vector = torch.randn_like(input_data)
        residual_norm = (input_data - hard_quantized).norm(dim=1, keepdim=True)
        random_norm = random_vector.norm(dim=1, keepdim=True)
        quantized = input_data + (residual_norm / random_norm + self.eps) * random_vector

        quantized = quantized.reshape(batch_size, self.embedding_dim, -1).transpose(1, 2).contiguous()
        quantized = self.project_out(quantized)
        return quantized, indices.reshape(batch_size, -1)


def rearrange_to_image(
    input_data: Tensor,
    batch_size: int,
    embedding_dim: int,
    spatial_size: int,
) -> Tensor:
    return input_data.transpose(1, 2).reshape(batch_size, embedding_dim, spatial_size, spatial_size)

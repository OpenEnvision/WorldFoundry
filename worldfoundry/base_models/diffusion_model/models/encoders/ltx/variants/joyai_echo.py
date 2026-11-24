# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo padding conventions over shared LTX text projection and connector graphs.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from typing import cast

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_connector import Embeddings1DConnector
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_processor import (
    EmbeddingsProcessor,
    _to_binary_mask,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.feature_extractor import FeatureExtractorV1


class EchoEmbeddings1DConnector(Embeddings1DConnector):
    """Compact contiguous left padding before substituting learned registers."""

    _supports_gradient_checkpointing = True

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, additive_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_registers = cast(int, self.num_learnable_registers)
        assert hidden_states.shape[1] % num_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}."
        )

        num_registers_duplications = hidden_states.shape[1] // num_registers
        learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))
        valid_mask = additive_attention_mask.squeeze(1).squeeze(1) >= -9000.0
        batch_size, sequence_length = valid_mask.shape
        positions = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        valid_counts = valid_mask.sum(dim=1)

        # Gemma inputs are left padded. Rotate every sample independently so its
        # valid tokens start at index zero, then fill the remaining positions
        # with the same position-dependent register sequence used by B=1.
        expected_mask = positions >= (sequence_length - valid_counts).unsqueeze(1)
        if not torch.equal(valid_mask, expected_mask):
            raise ValueError("Embeddings1DConnector expects a contiguous left-padded attention mask")
        source_positions = (positions + (sequence_length - valid_counts).unsqueeze(1)) % sequence_length
        compacted_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=source_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
        )
        output_valid_mask = positions < valid_counts.unsqueeze(1)
        registers = learnable_registers.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.where(
            output_valid_mask.unsqueeze(-1),
            compacted_hidden_states,
            registers,
        )

        additive_attention_mask = torch.full_like(
            additive_attention_mask,
            0.0,
            dtype=additive_attention_mask.dtype,
            device=additive_attention_mask.device,
        )

        return hidden_states, additive_attention_mask


class EchoEmbeddingsProcessor(EmbeddingsProcessor):
    """Leave token reordering to the Echo connector, including register-free inputs."""

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.audio_connector is not None and audio_features is None:
            raise ValueError("Audio connector is configured but no audio features were provided.")
        if self.audio_connector is None and audio_features is not None:
            raise ValueError("Audio features were provided but no audio connector is configured.")

        video_encoded, video_mask = self.video_connector(video_features, additive_attention_mask)
        binary_mask = _to_binary_mask(video_mask, video_encoded.shape[:2])
        video_encoded = video_encoded * binary_mask

        audio_encoded = None
        if self.audio_connector is not None:
            audio_encoded, _ = self.audio_connector(audio_features, additive_attention_mask)

        return video_encoded, audio_encoded, binary_mask.squeeze(-1)


class EchoFeatureExtractorV1(FeatureExtractorV1):
    """Normalize the contiguous token span described by the declared padding side."""

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        length = encoded.shape[1]
        positions = torch.arange(length, device=encoded.device)[None, :]
        sequence_lengths = attention_mask.sum(dim=-1)
        if padding_side == "right":
            mask = positions < sequence_lengths[:, None]
        elif padding_side == "left":
            mask = positions >= (length - sequence_lengths[:, None])
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        return super().forward(encoded, mask.to(dtype=attention_mask.dtype), padding_side)

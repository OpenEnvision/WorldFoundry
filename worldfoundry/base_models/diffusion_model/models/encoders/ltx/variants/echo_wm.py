# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo-WM register replacement over the shared LTX connector graph.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from typing import cast

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_connector import Embeddings1DConnector


class EchoWMEmbeddings1DConnector(Embeddings1DConnector):
    """Preserve single-prompt boolean selection and arithmetic register replacement."""

    _supports_gradient_checkpointing = True

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, additive_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_registers = cast(int, self.num_learnable_registers)
        assert hidden_states.shape[1] % num_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}."
        )
        registers = torch.tile(self.learnable_registers, (hidden_states.shape[1] // num_registers, 1))
        binary_mask = (additive_attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()
        selected = hidden_states[:, binary_mask.squeeze().bool(), :]
        padded = F.pad(selected, (0, 0, 0, hidden_states.shape[1] - selected.shape[1]), value=0)
        flipped_mask = torch.flip(binary_mask, dims=[1])
        hidden_states = flipped_mask * padded + (1 - flipped_mask) * registers
        return hidden_states, torch.full_like(additive_attention_mask, 0.0)

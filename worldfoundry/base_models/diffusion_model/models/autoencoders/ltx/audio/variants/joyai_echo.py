# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo bandwidth extension on the shared LTX vocoder graph.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

import torch
import torch.nn.functional as F

from ..vocoder import VocoderWithBWE


class EchoVocoderWithBWE(VocoderWithBWE):
    """Retain the caller's input dtype and ambient autocast policy."""

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        x = self.vocoder(mel_spec)
        length_low_rate = x.shape[-1]
        output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))
        mel = self._compute_mel(x)
        residual = self.bwe_generator(mel.transpose(2, 3))
        skip = self.resampler(x)
        assert residual.shape == skip.shape, f"residual {residual.shape} != skip {skip.shape}"
        return torch.clamp(residual + skip, -1, 1)[..., :output_length]

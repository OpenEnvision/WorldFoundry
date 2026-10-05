"""Small tensor helpers (snake, pad) for the LTX audio VAE.

:class:`AudioProcessor` holds lightweight activations and
padding used by residual and vocoder blocks.  Snake-style
periodic activations also appear in :mod:`.vocoder`.

Inputs are channel-first spectrogram tensors.  No runner
Protocol or checkpoint loading lives here.

Shared LTX audio graph utilities, sibling of :mod:`.resnet`.
"""

import torch
import torchaudio
from torch import nn

from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import Audio


class AudioProcessor(nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling."""

    def __init__(
        self,
        target_sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=target_sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_audio(self, audio: Audio) -> Audio:
        """Resample audio to the processor's target sample rate if needed."""
        if audio.sampling_rate == self.target_sample_rate:
            return audio
        resampled = torchaudio.functional.resample(audio.waveform, audio.sampling_rate, self.target_sample_rate)
        resampled = resampled.to(device=audio.waveform.device, dtype=audio.waveform.dtype)
        return Audio(waveform=resampled, sampling_rate=self.target_sample_rate)

    def waveform_to_mel(
        self,
        audio: Audio,
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram [batch, channels, time, n_mels]."""
        waveform = self.resample_audio(audio).waveform

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()

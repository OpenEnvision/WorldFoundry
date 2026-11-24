"""Voice filtering for multi-shot audio memory.

Echo 1.5 production memory keeps only the speech stem of memory audio, extracted with the
Bandit separator of MSST-WebUI. MSST-WebUI is a separate GNU AGPL-3.0 work that is not part of
this runtime, so an enabled filter rejects non-empty memory audio; a disabled filter passes it
through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MSST_UNAVAILABLE = (
    "Echo 1.5 voice memory requires the MSST-WebUI speech separator, which is not part of this "
    "runtime; omit memory audio or set memory.voice_filter.enabled to false."
)


@dataclass(frozen=True)
class VoiceFilterConfig:
    enabled: bool = True
    backend: str = "msst_speech"

    def validate(self) -> None:
        """Reject an enabled filter, whose MSST speech separator is unavailable."""

        if not self.enabled:
            return
        if self.backend.strip().lower() != "msst_speech":
            raise ValueError("Echo 1.5 production memory requires backend=msst_speech")
        raise RuntimeError(MSST_UNAVAILABLE)


def _as_channels_first(waveform: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(waveform).detach().cpu().float()
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"expected one audio sample, got shape={tuple(value.shape)}")
        value = value[0]
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"expected audio [channels, samples], got {tuple(value.shape)}")
    return value.contiguous()


def filter_voice_only(
    waveform: torch.Tensor | None,
    sample_rate: int,
    config: VoiceFilterConfig,
) -> torch.Tensor | None:
    """Return memory audio for a disabled filter; an enabled filter accepts only empty audio.

    Raises:
        RuntimeError: If the filter is enabled and the waveform holds more than one sample.
    """

    del sample_rate
    if waveform is None or not config.enabled:
        return waveform
    waveform = _as_channels_first(waveform)
    if waveform.numel() > 0 and waveform.shape[-1] > 1:
        config.validate()
    return waveform

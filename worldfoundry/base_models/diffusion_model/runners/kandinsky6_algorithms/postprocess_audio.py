from __future__ import annotations

import numpy as np
import torch

from worldfoundry.base_models.diffusion_model.kandinsky6_types import LatentBundle


@torch.no_grad()
def postprocess_audio(
    bundle: LatentBundle,
    audio_vae,
    vocoder,
    normalization_mode: str = "clip",
) -> list[np.ndarray] | None:
    """Decode audio latents → list of (samples,) int16 numpy arrays.

    Returns None when bundle.audio is None (T2V mode).

    ``clip`` preserves the decoded amplitude and saturates it to [-1, 1].
    ``normalize`` peak-normalizes each waveform before converting it to int16.
    """
    audio = bundle.audio
    if audio is None:
        return None

    def waveform_from(latents: torch.Tensor) -> torch.Tensor:
        mel = audio_vae.wrapped_decode(latents)
        return vocoder(mel)

    cu = bundle.audio_cu_seqlens
    assert cu is not None
    bs = cu.shape[0] - 1

    # Reverse audio VAE normalization
    audio_scaled = audio / getattr(audio_vae, "scaling_factor", 1.0)
    audio_scaled = audio_scaled + getattr(audio_vae, "mean_value", 0.0)

    segments = [
        audio_scaled[i]
        if audio_scaled.ndim == 3
        else audio_scaled[cu[i].item() : cu[i + 1].item()]
        for i in range(bs)
    ]  # (A, audio_dim) per sample

    # Native generation keeps equal-length samples in the batch. The VAE
    # returns a mel spectrogram; the vocoder turns that into a waveform.
    if len({tuple(segment.shape) for segment in segments}) == 1:
        decoded = waveform_from(torch.stack(segments).transpose(1, 2))
        if decoded.ndim == 1:
            decoded = decoded.unsqueeze(0)
        elif decoded.ndim == 3 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        waveforms = [waveform.cpu().float().numpy() for waveform in decoded]
    else:
        # Preserve support for legacy packed bundles with variable lengths.
        waveforms = [
            waveform_from(segment.transpose(1, 0).unsqueeze(0))
            .squeeze()
            .cpu()
            .float()
            .numpy()
            for segment in segments
        ]

    result: list[np.ndarray] = []
    for waveform in waveforms:
        if normalization_mode == "normalize":
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak * 32767
        elif normalization_mode == "clip":
            waveform = np.clip(waveform, -1.0, 1.0) * 32767
        else:
            raise ValueError(f"unknown audio normalization_mode={normalization_mode}")

        result.append(waveform.astype(np.int16))

    return result

"""TAEHV Wan2.1 preview decoder: layout, frame alignment, streaming and builder selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import component
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import Wan21TAEPreviewDecoder
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.taehv import TAEHV
from worldfoundry.core.model_loading.policy import RuntimePolicy


def _decoder(seed: int = 0) -> tuple[TAEHV, Wan21TAEPreviewDecoder]:
    torch.manual_seed(seed)
    tae = TAEHV(checkpoint_path=None)
    return tae, Wan21TAEPreviewDecoder(
        tae, device=torch.device("cpu"), dtype=torch.float32, checkpoint_path="/models/taew2_1.pth"
    )


def _latents(frames: int) -> torch.Tensor:
    return torch.randn(1, 16, frames, 4, 6, generator=torch.Generator().manual_seed(frames))


def test_whole_clip_matches_upstream_parallel_decode_with_wan_frame_count() -> None:
    tae, decoder = _decoder()
    latents = _latents(6)

    pixels = decoder.decode(latents)

    upstream = tae.decode_video(latents.transpose(1, 2), parallel=True)[:, tae.frames_to_trim :]
    assert pixels.shape == (1, 3, decoder.get_pixel_num_frames(6), 32, 48) == (1, 3, 21, 32, 48)
    torch.testing.assert_close(pixels, upstream.transpose(1, 2).mul(2).sub(1).clamp(-1, 1), rtol=0, atol=0)
    assert float(pixels.min()) >= -1.0 and float(pixels.max()) <= 1.0
    report = decoder.runtime_optimization_report()
    assert report["quality_tier"] == "algorithmically-approximate-preview"
    runtime = report["runtime"]
    assert isinstance(runtime, dict) and runtime["decode_calls"] == 1


@pytest.mark.parametrize("chunk", [3, 1])
def test_streamed_chunks_match_whole_clip_decode(chunk: int) -> None:
    _, decoder = _decoder()
    latents = _latents(6)
    whole = decoder.decode(latents)

    state = decoder.new_stream_state()
    pieces = [
        decoder.decode_chunk(latents[:, :, index : index + chunk], state=state, is_first_chunk=index == 0)
        for index in range(0, 6, chunk)
    ]

    first = 4 * chunk - 3
    assert [piece.shape[2] for piece in pieces] == [first, *([4 * chunk] * (len(pieces) - 1))]
    # Chunked convolutions batch fewer frames; float rounding may differ in the last bits.
    torch.testing.assert_close(torch.cat(pieces, dim=2), whole, rtol=0, atol=1e-5)

    rerun_state = decoder.new_stream_state()
    rerun = decoder.decode_chunk(latents[:, :, :chunk], state=rerun_state)
    torch.testing.assert_close(rerun, pieces[0], rtol=0, atol=0)


def test_preview_returns_latents_on_request_and_has_no_encoder() -> None:
    _, decoder = _decoder()
    latents = _latents(2)
    request = DiffusionRequest(prompt="", inputs={"return_latent": True})
    assert decoder.decode(latents, request) is latents
    with pytest.raises(RuntimeError, match="decode-only"):
        decoder.encode(torch.zeros(1, 3, 1, 32, 32))
    with pytest.raises(ValueError, match="16 channels"):
        decoder.decode(torch.zeros(1, 48, 2, 4, 4))


def _context(checkpoint: Path, *, component_options=None, policy_options=None) -> ComponentBuildContext:
    return ComponentBuildContext(
        "self-forcing",
        ComponentKey(ComponentKind.DECODER),
        RuntimePolicy(device="cpu", dtype=torch.bfloat16, options=policy_options or {}),
        {"weights": CheckpointSpec(source="/nonexistent/Wan2.1_VAE.pth")},
        component_options=component_options or {},
    )


@pytest.mark.parametrize("surface", ["runtime", "component"])
def test_wan21_builder_selects_preview_from_existing_options(surface: str, tmp_path: Path) -> None:
    checkpoint = tmp_path / "taew2_1.pth"
    torch.save(TAEHV(checkpoint_path=None).state_dict(), checkpoint)
    options = {"dtype": torch.bfloat16, "std_reciprocal_in_codec_dtype": True}
    if surface == "runtime":
        context = _context(
            checkpoint, component_options=options, policy_options={"vae_preview_decoder_path": checkpoint}
        )
    else:
        context = _context(checkpoint, component_options={**options, "preview_decoder_path": str(checkpoint)})

    decoder = component.build_wan_video_decoder(context)

    assert isinstance(decoder, Wan21TAEPreviewDecoder)
    assert decoder.dtype == torch.bfloat16
    assert decoder.checkpoint_path == str(checkpoint)
    assert next(decoder.vae.parameters()).dtype == torch.bfloat16


def test_wan21_builder_keeps_lightvae_preview_conflict(tmp_path: Path) -> None:
    context = _context(
        tmp_path / "taew2_1.pth",
        component_options={"variant": "lightvae-wan21"},
        policy_options={"vae_preview_decoder_path": str(tmp_path / "taew2_1.pth")},
    )
    with pytest.raises(ValueError, match="vae_preview_decoder_path"):
        component.build_wan_video_decoder(context)

"""MG2 blank-tail reuse must follow the actual causal encoder's receptive field."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.action_21 import (
    CausalConv3d,
    Encoder3d,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.conditioning import (
    first_stable_encoder_latent,
    minimum_condition_prefetch_blocks,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
    MatrixGame2RealtimeSession,
)


def _tiny_encoder(temporal_downsample=(False, True, True)):
    # Channels affect cost, but leave the production graph and temporal
    # receptive field unchanged. No checkpoint or CUDA allocation is needed.
    return Encoder3d(
        dim=1,
        z_dim=2,
        temperal_downsample=list(temporal_downsample),
    ).eval()


@pytest.mark.parametrize(
    ("temporal_downsample", "first_stable", "blocks"),
    [
        ((False, True, True), 29, 11),
        ((True, True, False), 35, 13),
        ((False, False, False), 45, 16),
    ],
)
def test_condition_warmup_tracks_actual_encoder_graph(temporal_downsample, first_stable, blocks):
    encoder = _tiny_encoder(temporal_downsample)

    assert first_stable_encoder_latent(encoder) == first_stable
    assert minimum_condition_prefetch_blocks(encoder) == blocks
    assert blocks * 3 - 3 >= first_stable
    assert (blocks - 1) * 3 - 3 < first_stable


def test_condition_warmup_tracks_reduced_residual_dependency():
    encoder = _tiny_encoder()
    residual = encoder.downsamples[0]
    residual.residual = torch.nn.Sequential(torch.nn.Identity())
    residual.shortcut = CausalConv3d(1, 1, 1)

    # The old two-conv main path contributed four raw frames of history.
    assert first_stable_encoder_latent(encoder) == 28


def test_condition_warmup_rejects_temporal_shortcut_without_history():
    encoder = _tiny_encoder()
    encoder.downsamples[0].shortcut = CausalConv3d(1, 1, 3, padding=1)

    with pytest.raises(ValueError, match=r"CausalConv3d at encoder\.downsamples\.0\.shortcut"):
        first_stable_encoder_latent(encoder)


def test_condition_warmup_rejects_unknown_temporal_layer():
    encoder = _tiny_encoder()
    encoder.head.add_module("unknown_temporal", torch.nn.Conv3d(2, 2, 3))

    with pytest.raises(ValueError, match=r"Conv3d at encoder\.head\.unknown_temporal"):
        first_stable_encoder_latent(encoder)


def test_condition_warmup_counts_every_shared_layer_execution():
    encoder = _tiny_encoder()
    residual = encoder.downsamples[0].residual
    # Both convs have 1 input/output channel, so this remains an executable
    # residual branch. Sequential must apply the shared conv twice.
    residual[6] = residual[2]

    assert first_stable_encoder_latent(encoder) == 29


def test_condition_warmup_rejects_unknown_spatial_container_layer():
    encoder = _tiny_encoder()
    downsample = next(
        module for module in encoder.downsamples if getattr(module, "mode", None) == "downsample2d"
    )
    downsample.resample.add_module("unknown_temporal", torch.nn.Conv3d(1, 1, 3))

    with pytest.raises(ValueError, match=r"Conv3d at encoder\.downsamples\.\d+\.resample\.unknown_temporal"):
        first_stable_encoder_latent(encoder)


def test_condition_warmup_rejects_modified_temporal_resampling():
    encoder = _tiny_encoder()
    downsample = next(
        module for module in encoder.downsamples if getattr(module, "mode", None) == "downsample3d"
    )
    downsample.time_conv = CausalConv3d(2, 2, (5, 1, 1), stride=(2, 1, 1))

    with pytest.raises(ValueError, match="Resample"):
        first_stable_encoder_latent(encoder)


@pytest.mark.parametrize("block_frames", [0, -1, 1.5, True])
def test_condition_warmup_rejects_invalid_block_size(block_frames):
    with pytest.raises(ValueError, match="positive integer"):
        minimum_condition_prefetch_blocks(_tiny_encoder(), block_frames)


@pytest.mark.parametrize("requested_blocks,expected_blocks", [(None, 11), ("2", 11), ("5", 11), ("12", 12)])
def test_realtime_clamps_condition_prefix_to_proven_encoder_bound(
    monkeypatch, requested_blocks, expected_blocks,
):
    if requested_blocks is None:
        monkeypatch.delenv("WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS", requested_blocks)
    runtime = SimpleNamespace(
        pipeline=SimpleNamespace(num_frame_per_block=3),
        device="cpu",
        weight_dtype=torch.float32,
        vae=SimpleNamespace(vae=SimpleNamespace(model=SimpleNamespace(encoder=_tiny_encoder()))),
    )

    session = MatrixGame2RealtimeSession(runtime, SimpleNamespace())

    assert session.condition_first_stable_latent == 29
    assert session.condition_prefetch_blocks == expected_blocks
    assert session.condition_prefetch_blocks * session.latent_frames_per_block in {33, 36}


def test_realtime_fixture_without_encoder_has_no_stationarity_proof(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS", raising=False)
    runtime = SimpleNamespace(
        pipeline=SimpleNamespace(num_frame_per_block=3),
        device="cpu",
        weight_dtype=torch.float32,
        vae=SimpleNamespace(),
    )

    session = MatrixGame2RealtimeSession(runtime, SimpleNamespace())

    assert session.condition_first_stable_latent is None
    assert session.condition_prefetch_blocks == 5

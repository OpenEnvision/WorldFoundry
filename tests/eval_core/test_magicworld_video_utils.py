from __future__ import annotations

import os

import numpy as np
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import camera_attention
from worldfoundry.core.attention import flash_attention as core_flash_attention
from worldfoundry.synthesis.visual_generation.magic_world.worldfoundry_runtime import (
    _resolve_depth_pro_checkpoint,
    _resolve_local_checkpoint_alias,
)
from worldfoundry.synthesis.visual_generation.magic_world.magic_world_runtime.utils.camera_pose import (
    _ray_condition,
)
from worldfoundry.synthesis.visual_generation.video_x_fun.video_x_fun_runtime.videox_fun.utils.utils import (
    get_video_to_video_render_latent,
)


def test_get_video_to_video_render_latent_from_memory_frames() -> None:
    frames = np.stack(
        [
            np.full((2, 3, 3), 64, dtype=np.uint8),
            np.full((2, 3, 3), 128, dtype=np.uint8),
        ]
    )
    masks = np.stack(
        [
            np.zeros((2, 3), dtype=np.uint8),
            np.full((2, 3), 255, dtype=np.uint8),
        ]
    )

    video, render_mask, video_mask, ref_image, clip_image = get_video_to_video_render_latent(
        frames,
        masks,
        video_length=2,
        sample_size=(2, 3),
    )

    assert video.shape == (1, 3, 2, 2, 3)
    assert render_mask.shape == (1, 3, 2, 2, 3)
    assert video_mask.shape == (1, 1, 2, 2, 3)
    assert torch.allclose(video[:, :, 0], torch.full((1, 3, 2, 3), 64 / 255))
    assert torch.count_nonzero(render_mask[:, :, 0]) == 0
    assert torch.all(render_mask[:, :, 1] == 1)
    assert torch.all(video_mask == 255)
    assert ref_image is None
    assert clip_image is None


def test_magicworld_resolves_flat_checkpoint_mirror(monkeypatch, tmp_path) -> None:
    model_root = tmp_path / "LuckyLiGY--MagicWorld"
    model_root.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    stale_hfd_path = tmp_path / "hfd" / model_root.name
    assert _resolve_local_checkpoint_alias(stale_hfd_path) == model_root.resolve()


def test_magicworld_ray_condition_broadcasts_multiple_frames() -> None:
    intrinsics = torch.tensor(
        [[[2.0, 2.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]]]
    )
    c2w = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 3, -1, -1).clone()

    rays = _ray_condition(intrinsics, c2w, height=2, width=3, device="cpu")

    assert rays.shape == (1, 3, 2, 3, 6)
    assert torch.isfinite(rays).all()


def test_magicworld_resolves_local_depth_pro_checkpoint(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "apple--DepthPro" / "depth_pro.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    monkeypatch.delenv("DEPTH_PRO_CHECKPOINT", raising=False)

    resolved = _resolve_depth_pro_checkpoint()

    assert resolved == checkpoint.resolve()
    assert "DEPTH_PRO_CHECKPOINT" not in os.environ


def test_magicworld_camera_attention_falls_back_without_flash_attn(monkeypatch) -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 2, 8)
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)
    query_lengths = torch.tensor([4, 3], dtype=torch.int32)
    key_lengths = torch.tensor([5, 2], dtype=torch.int32)
    monkeypatch.setattr(camera_attention, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(camera_attention, "FLASH_ATTN_3_AVAILABLE", False)

    actual = camera_attention.flash_attention(
        query,
        key,
        value,
        q_lens=query_lengths,
        k_lens=key_lengths,
    )
    expected = core_flash_attention(
        query,
        key,
        value,
        q_lens=query_lengths,
        k_lens=key_lengths,
    )

    torch.testing.assert_close(actual, expected)

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.videocrafter.worldfoundry_runtime import (
    resolve_runtime_checkpoint,
)


def test_resolve_videocrafter2_checkpoint_from_hub_layout(tmp_path) -> None:
    checkpoint = tmp_path / "VideoCrafter--VideoCrafter2" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    resolved = resolve_runtime_checkpoint(
        tmp_path / "VideoCrafter" / "videocrafter_t2v_512_v2.ckpt",
        "videocrafter2-t2v",
    )

    assert resolved == checkpoint


def test_resolve_videocrafter_checkpoint_keeps_existing_path(tmp_path) -> None:
    checkpoint = tmp_path / "custom.ckpt"
    checkpoint.touch()

    assert resolve_runtime_checkpoint(checkpoint, "custom") == checkpoint

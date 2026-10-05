from __future__ import annotations

from worldfoundry.synthesis.visual_generation.echo_memory.runtime import (
    _resolve_wan_backbone_path,
    _select_echo_checkpoint_path,
)


def test_resolve_wan_backbone_accepts_owner_prefixed_local_layout(tmp_path) -> None:
    staged = tmp_path / "Wan-AI--Wan2.1-T2V-1.3B"
    staged.mkdir()

    resolved = _resolve_wan_backbone_path(tmp_path / "Wan2.1-T2V-1.3B")

    assert resolved == staged


def test_resolve_wan_backbone_keeps_existing_explicit_path(tmp_path) -> None:
    explicit = tmp_path / "custom-wan"
    explicit.mkdir()

    assert _resolve_wan_backbone_path(explicit) == explicit


def test_select_echo_checkpoint_accepts_owner_prefixed_local_layout(tmp_path) -> None:
    checkpoint = tmp_path / "Echo-Team--Echo-Memory" / "context_k1" / "epoch-0.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    resolved = _select_echo_checkpoint_path(
        tmp_path / "Echo-Memory" / "context_k1" / "epoch-0.safetensors",
        "epoch-0.safetensors",
    )

    assert resolved == checkpoint

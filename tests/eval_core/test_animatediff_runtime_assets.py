from __future__ import annotations

from worldfoundry.synthesis.visual_generation.animatediff.animatediff_synthesis import (
    _resolve_sd15_path,
)
from worldfoundry.synthesis.visual_generation.animatediff.runtime.utils.convert_from_ckpt import (
    _local_clip_text_model_path,
)


def test_animatediff_resolves_local_sd15_flat_mirror(tmp_path) -> None:
    expected = tmp_path / "stable-diffusion-v1-5--stable-diffusion-v1-5"
    expected.mkdir()

    resolved = _resolve_sd15_path(tmp_path / "stable-diffusion-v1-5")

    assert resolved == expected.resolve()


def test_animatediff_resolves_local_clip_text_model(monkeypatch, tmp_path) -> None:
    expected = tmp_path / "openai--clip-vit-large-patch14"
    expected.mkdir()
    (expected / "config.json").write_text("{}", encoding="utf-8")
    (expected / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    assert _local_clip_text_model_path() == expected.resolve()

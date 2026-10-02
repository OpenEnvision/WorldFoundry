from __future__ import annotations

from worldfoundry.pipelines.hunyuan_video import pipeline_hunyuan_video
from worldfoundry.pipelines.hunyuan_video.pipeline_hunyuan_video import (
    NativeHunyuanVideoPipeline,
    _resolve_hunyuan_checkpoint_source,
)


def test_resolve_hunyuan15_uses_unified_checkpoint_root(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "tencent--HunyuanVideo-1.5"
    (checkpoint / "transformer/720p_t2v").mkdir(parents=True)
    (checkpoint / "transformer/720p_t2v/diffusion_pytorch_model.safetensors").touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    resolved = _resolve_hunyuan_checkpoint_source(
        tmp_path / "stale" / "HunyuanVideo-1.5",
        model_id="hunyuanvideo-1.5-t2v",
    )

    assert resolved == checkpoint


def test_resolve_hunyuan_keeps_existing_explicit_path(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "custom"
    (explicit / "hunyuan-video-t2v-720p/transformers").mkdir(parents=True)
    (explicit / "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt").touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    assert _resolve_hunyuan_checkpoint_source(
        explicit,
        model_id="hunyuanvideo-t2v",
    ) == explicit


def test_resolve_hunyuan15_rejects_existing_but_incomplete_stale_path(tmp_path, monkeypatch) -> None:
    stale = tmp_path / "world_models" / "HunyuanVideo-1.5"
    stale.mkdir(parents=True)
    checkpoint = tmp_path / "tencent--HunyuanVideo-1.5"
    (checkpoint / "transformer/720p_t2v").mkdir(parents=True)
    (checkpoint / "transformer/720p_t2v/diffusion_pytorch_model.safetensors").touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    assert _resolve_hunyuan_checkpoint_source(
        stale,
        model_id="hunyuanvideo-1.5-t2v",
    ) == checkpoint


def test_hunyuan15_explicit_component_roles_override_the_primary_root(tmp_path, monkeypatch) -> None:
    primary = tmp_path / "weights"
    (primary / "transformer/720p_i2v").mkdir(parents=True)
    (primary / "transformer/720p_i2v/diffusion_pytorch_model.safetensors").touch()
    resources = tmp_path / "resources"
    vision = resources / "vision_encoder/siglip"
    vision.mkdir(parents=True)
    captured: dict[str, object] = {}

    def capture_from_pretrained(cls, model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        pipeline_hunyuan_video.NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(capture_from_pretrained),
    )

    NativeHunyuanVideoPipeline.from_pretrained(
        primary,
        model_id="hunyuanvideo-1.5-i2v",
        device="cpu",
        checkpoint_overrides={
            "resources": str(resources),
            "vision": str(vision),
        },
    )

    assert captured["model_id"] == "hunyuanvideo-1.5-i2v"
    assert captured["checkpoint_overrides"] == {
        "transformer": str(primary.resolve()),
        "vae": str(primary.resolve()),
        "resources": str(resources),
        "vision": str(vision),
    }

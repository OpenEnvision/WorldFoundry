from pathlib import Path

from worldfoundry.base_models.diffusion_model.models.encoders.lvdm.condition import (
    _local_hf_model,
    _local_open_clip_pretrained,
)


def test_lvdm_conditioners_discover_plural_checkpoint_sibling(monkeypatch, tmp_path):
    configured_root = tmp_path / "ckpt"
    staged_root = tmp_path / "ckpts"
    configured_root.mkdir()
    openclip_weight = (
        staged_root
        / "laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
        / "open_clip_pytorch_model.bin"
    )
    openclip_weight.parent.mkdir(parents=True)
    openclip_weight.touch()
    local_clip = staged_root / "openai--clip-vit-large-patch14"
    local_clip.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(configured_root))

    assert _local_open_clip_pretrained("ViT-H-14", "laion2b_s32b_b79k") == str(
        openclip_weight
    )
    assert _local_hf_model("openai/clip-vit-large-patch14") == str(local_clip)


def test_lvdm_openclip_explicit_override_has_priority(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit-openclip.bin"
    explicit.touch()
    monkeypatch.setenv("WORLDFOUNDRY_OPENCLIP_VITH14_PATH", str(explicit))

    assert _local_open_clip_pretrained("ViT-H-14", "laion2b_s32b_b79k") == str(explicit)

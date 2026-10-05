from __future__ import annotations

import pytest

from worldfoundry.studio.inference import catalog as catalog
from worldfoundry.studio.inference.catalog import STUDIO_HIDDEN_CATALOG_MODEL_IDS, discover_catalog, find_entry
from worldfoundry.studio.ui.catalog import _template_id_hint


_TEMPLATE_TO_WORKLOAD = {
    "depth-geometry": "geometry",
    "scene-3d": "3d",
    "hosted-api": "api",
    "conditioned-video": "i2v",
    "video-to-video": "v2v",
}


@pytest.mark.parametrize(
    ("model_id", "category", "template_id", "workload"),
    [
        ("dust3r", "Depth / Geometry", "depth-geometry", "geometry"),
        ("dust3r-base-model", "Depth / Geometry", "depth-geometry", "geometry"),
        ("inspatio-world", "Video-to-Video", "video-to-video", "v2v"),
        ("recammaster", "Video-to-Video", "video-to-video", "v2v"),
        ("neoverse", "Video-to-Video", "video-to-video", "v2v"),
        ("worldlabs-marble-1.1", "Remote API", "hosted-api", "api"),
        ("dvlt", "3D Scene", "scene-3d", "3d"),
        ("lagernvs", "3D Scene", "scene-3d", "3d"),
        ("lingbot-map", "3D Scene", "scene-3d", "3d"),
        ("lyra-1", "3D Scene", "scene-3d", "3d"),
        ("lyra-2", "3D Scene", "scene-3d", "3d"),
    ],
)
def test_studio_catalog_model_taxonomy(model_id: str, category: str, template_id: str, workload: str) -> None:
    entry = find_entry(model_id)
    resolved_template = _template_id_hint(entry)

    assert entry.category == category
    assert resolved_template == template_id
    assert _TEMPLATE_TO_WORKLOAD.get(resolved_template, "world") == workload


@pytest.mark.parametrize("model_id", sorted(STUDIO_HIDDEN_CATALOG_MODEL_IDS))
def test_unintegrated_or_duplicate_models_are_hidden_from_studio(model_id: str) -> None:
    studio_ids = {entry.model_id for entry in discover_catalog()}

    assert model_id not in studio_ids


def test_ltx23_runtime_environment_alias_resolves_to_canonical_catalog_entry() -> None:
    entry = find_entry("ltx2_3_i2v")

    assert entry.model_id == "ltx-2.3-i2v"
    assert entry.class_name == "LTX23I2VPipeline"


def test_cut3r_default_ref_prefers_owner_prefixed_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    local_model = checkpoint_root / "liguang0115--cut3r"
    local_model.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert catalog._cut3r_default_ref() == str(local_model)


def test_show_o_defaults_resolve_all_owner_prefixed_local_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    show_o = checkpoint_root / "showlab--show-o"
    magvit = checkpoint_root / "showlab--magvitv2"
    phi = checkpoint_root / "microsoft--phi-1_5"
    for local_model in (show_o, magvit, phi):
        local_model.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert catalog._show_o_default_ref() == str(show_o)
    assert catalog._show_o_default_load_kwargs() == {
        "vq_model_path": str(magvit),
        "llm_model_path": str(phi),
        "resolution": 256,
    }


def test_longcat_default_ref_prefers_owner_prefixed_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    local_model = checkpoint_root / "meituan-longcat--LongCat-Video"
    local_model.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert catalog._longcat_video_default_ref() == str(local_model)


def test_astra_default_ref_prefers_owner_prefixed_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    local_model = checkpoint_root / "EvanEternal--Astra"
    local_model.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert catalog._astra_default_ref() == str(local_model)


def test_gen3c_defaults_resolve_all_owner_prefixed_local_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    transformer = checkpoint_root / "nvidia--GEN3C-Cosmos-7B"
    tokenizer = checkpoint_root / "nvidia--Cosmos-Tokenize1-CV8x8x8-720p"
    text_encoder = checkpoint_root / "google-t5--t5-11b"
    depth_model = checkpoint_root / "Ruicheng--moge-vitl"
    for local_model in (transformer, tokenizer, text_encoder, depth_model):
        local_model.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert catalog._gen3c_default_ref() == str(transformer)
    assert catalog._gen3c_default_load_kwargs() == {
        "required_components": {
            "transformer_model_path": str(transformer),
            "tokenizer_model_path": str(tokenizer),
            "text_encoder_model_path": str(text_encoder),
            "moge_pretrained": str(depth_model),
        }
    }


def test_gen3c_defaults_keep_public_hub_fallbacks_without_local_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *names: [])

    assert catalog._gen3c_default_ref() == "nvidia/GEN3C-Cosmos-7B"
    assert catalog._gen3c_default_load_kwargs() == {
        "required_components": {
            "transformer_model_path": "nvidia/GEN3C-Cosmos-7B",
            "tokenizer_model_path": "nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
            "text_encoder_model_path": "google-t5/t5-11b",
            "moge_pretrained": "Ruicheng/moge-vitl",
        }
    }


def test_cameractrl_defaults_resolve_owner_prefixed_local_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    camera_root = checkpoint_root / "hehao13--CameraCtrl"
    sd15_root = checkpoint_root / "stable-diffusion-v1-5--stable-diffusion-v1-5"
    motion_root = checkpoint_root / "guoyww--animatediff"
    camera_root.mkdir(parents=True)
    sd15_root.mkdir(parents=True)
    motion_root.mkdir(parents=True)
    (sd15_root / "unet").mkdir()
    (sd15_root / "unet" / "config.json").touch()
    pose = camera_root / "CameraCtrl.ckpt"
    image_lora = camera_root / "RealEstate10K_LoRA.ckpt"
    motion = motion_root / "v3_sd15_mm.ckpt"
    for weight in (pose, image_lora, motion):
        weight.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert catalog._cameractrl_default_ref() == str(pose)
    assert catalog._cameractrl_default_load_kwargs() == {
        "sd15_path": str(sd15_root),
        "pose_adaptor_ckpt": str(pose),
        "image_lora_ckpt": str(image_lora),
        "motion_module_ckpt": str(motion),
        "unet_subfolder": "unet",
    }

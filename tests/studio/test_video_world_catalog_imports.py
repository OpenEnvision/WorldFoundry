from __future__ import annotations

import importlib

import pytest
import worldfoundry.synthesis.visual_generation.magic_world.worldfoundry_runtime as magicworld_runtime

from worldfoundry.studio.inference.catalog import (
    _animatediff_default_load_kwargs,
    _depth_anything_v2_default_ref,
    _discover_ast_pipelines,
    _hydra_default_load_kwargs,
    _hydra_default_ref,
    _hf_checkpoint_model_ref,
    _magicworld_default_load_kwargs,
    _magicworld_default_ref,
    _longsana_video_480p_default_ref,
    _minwm_default_ref,
    _minwm_hy_default_load_kwargs,
    _minwm_wan_default_load_kwargs,
    _pi3_default_ref,
    _sana_video_480p_default_ref,
    _sana_video_720p_default_ref,
    _wow_default_ref,
    find_entry,
)
from worldfoundry.synthesis.visual_generation.magic_world.worldfoundry_runtime import (
    _magicworld_latent_frames,
)


def test_abstract_variant_base_pipelines_are_not_workspace_models() -> None:
    discovered_classes = {info.class_name for info in _discover_ast_pipelines()}

    assert "EchoMemoryPipeline" not in discovered_classes
    assert "MatrixGame35Pipeline" not in discovered_classes
    assert "EchoMemoryContextK1Pipeline" in discovered_classes
    assert "MatrixGame35FirstPersonPipeline" in discovered_classes


@pytest.mark.parametrize("model_id", ["framepack", "wan2.1_i2v", "wan2.1_t2v"])
def test_video_catalog_pipeline_target_is_importable(model_id: str) -> None:
    entry = find_entry(model_id)

    module = importlib.import_module(entry.module_path)

    assert getattr(module, entry.class_name) is not None


def test_matrix_game_1_uses_the_staged_in_domain_visual_qa_image() -> None:
    entry = find_entry("matrix-game-1")

    assert entry.default_input_path.endswith(
        "worldfoundry/data/test_cases/matrix-game-1/official_initial_image/forest_00.jpg"
    )


def test_cogvideox_i2v_demo_prompt_matches_its_person_input() -> None:
    entry = find_entry("cogvideox_5b_i2v")

    assert entry.default_input_path.endswith("worldfoundry/data/test_cases/studio_demo/00/image.jpg")
    assert "sparkler" in entry.default_prompt.lower()


def test_studio_demo_i2v_defaults_match_the_sparkler_fixture() -> None:
    for model_id in (
        "dynamicrafter_1024_i2v",
        "dynamicrafter_512_i2v",
        "framepack",
        "longvie-1",
        "ltx_video_i2v",
        "ltx2_i2v",
        "ltx2_3_i2v",
        "skyreels-v3",
        "wan2.1_i2v",
    ):
        entry = find_entry(model_id)
        assert "studio_demo/00/image.jpg" in str(entry.default_input_path)
        assert "sparkler" in entry.default_prompt.lower()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    (
        (
            "wan2.1_i2v",
            {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 40,
                "shift": 3.0,
                "guidance_scale": 5.0,
                "seed": 42,
            },
        ),
        (
            "wan2.1_t2v",
            {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 50,
                "shift": 8.0,
                "guidance_scale": 6.0,
                "seed": 42,
            },
        ),
    ),
)
def test_wan21_catalog_uses_native_diffusion_argument_names(
    model_id: str,
    expected: dict[str, object],
) -> None:
    entry = find_entry(model_id)

    assert entry.default_call_kwargs == expected
    assert "sample_steps" not in entry.call_params
    assert "offload_model" not in entry.call_params


@pytest.mark.parametrize(
    ("model_id", "frames", "steps", "height", "width"),
    (
        ("zeroscope", 24, 40, 320, 576),
        ("animatediff", 16, 25, 256, 256),
        ("cogvideox_2b_t2v", 49, 50, 480, 720),
        ("cogvideox_5b_t2v", 49, 50, 480, 720),
    ),
)
def test_quality_validation_defaults_are_not_smoke_reductions(
    model_id: str,
    frames: int,
    steps: int,
    height: int,
    width: int,
) -> None:
    defaults = find_entry(model_id).default_call_kwargs

    assert defaults["num_frames"] == frames
    assert defaults["num_inference_steps"] == steps
    assert defaults["height"] == height
    assert defaults["width"] == width


def test_animatediff_catalog_resolves_exported_sd15_directory(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "ckpts" / "stable-diffusion-v1-5--stable-diffusion-v1-5"
    model_root.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    assert _animatediff_default_load_kwargs()["base_model_path"] == str(model_root)


@pytest.mark.parametrize(
    ("model_id", "requires_image"),
    (
        ("videocrafter1-i2v", True),
        ("videocrafter1-t2v", False),
        ("videocrafter2-t2v", False),
    ),
)
def test_videocrafter_exact_ids_keep_the_inherited_runtime_contract(
    model_id: str,
    requires_image: bool,
) -> None:
    entry = find_entry(model_id)

    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert {"prompt", "num_frames", "num_inference_steps"} <= set(entry.call_params)
    assert {"model_path", "required_components", "device", "lazy"} <= set(entry.load_params)
    assert ("images" in entry.call_params) is requires_image


def test_gamma_mode_is_a_load_option_not_a_generation_option() -> None:
    entry = find_entry("gamma-world")

    assert entry.default_load_kwargs["mode"] == "causal_few_step"
    assert entry.default_load_kwargs["model_path"]
    assert entry.default_load_kwargs["text_encoder_path"]
    assert "mode" in entry.load_params
    assert "mode" not in entry.call_params
    assert "mode" not in entry.default_call_kwargs


def test_pi3_catalog_prefers_complete_plural_checkpoint_root(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "ckpts" / "yyfz233--Pi3X"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _pi3_default_ref() == str(model_root.resolve())

    (model_root / "config.json").unlink()
    assert _pi3_default_ref() == "yyfz233/Pi3X"


def test_depth_anything_v2_catalog_prefers_plural_checkpoint_root(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "ckpts" / "Depth-Anything-V2-Large"
    model_root.mkdir(parents=True)
    (model_root / "depth_anything_v2_vitl.pth").write_bytes(b"weights")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _depth_anything_v2_default_ref() == str(model_root.resolve())

    (model_root / "depth_anything_v2_vitl.pth").unlink()
    assert _depth_anything_v2_default_ref() == "depth-anything/Depth-Anything-V2-Large"


def test_hydra_catalog_resolves_checkpoint_and_wan_base_from_plural_root(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "ckpts" / "H-EmbodVis--HyDRA" / "hydra.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"hydra")
    base = tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-1.3B"
    base.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _hydra_default_ref() == str(checkpoint)
    assert _hydra_default_load_kwargs() == {"base_model_path": str(base)}


def test_wan_fun_catalog_resolves_direct_hfd_export_from_plural_root(tmp_path, monkeypatch) -> None:
    repo_name = "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera"
    model_root = tmp_path / "ckpts" / repo_name
    model_root.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _hf_checkpoint_model_ref(repo_name, "alibaba-pai/remote") == str(model_root)


@pytest.mark.parametrize(
    ("resolver", "repo_name", "checkpoint_name"),
    (
        (
            _sana_video_480p_default_ref,
            "Efficient-Large-Model--SANA-Video_2B_480p",
            "SANA_Video_2B_480p.pth",
        ),
        (
            _sana_video_720p_default_ref,
            "Efficient-Large-Model--SANA-Video_2B_720p",
            "SANA_Video_2B_720p.pth",
        ),
        (
            _longsana_video_480p_default_ref,
            "Efficient-Large-Model--SANA-Video_2B_480p_LongLive",
            "SANA_Video_2B_480p_LongLive.pth",
        ),
    ),
)
def test_sana_catalog_resolves_hfd_checkpoint_from_plural_root(
    tmp_path,
    monkeypatch,
    resolver,
    repo_name: str,
    checkpoint_name: str,
) -> None:
    checkpoint = tmp_path / "ckpts" / repo_name / "checkpoints" / checkpoint_name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert resolver() == str(checkpoint)


def test_magicworld_catalog_resolves_model_and_base_from_plural_root(tmp_path, monkeypatch) -> None:
    model = tmp_path / "ckpts" / "LuckyLiGY--MagicWorld"
    base = tmp_path / "ckpts" / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP"
    model.mkdir(parents=True)
    base.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _magicworld_default_ref() == str(model)
    assert _magicworld_default_load_kwargs() == {"base_model_path": str(base)}


def test_minwm_and_wow_catalog_resolve_direct_exports_from_plural_root(tmp_path, monkeypatch) -> None:
    minwm = tmp_path / "ckpts" / "MIN-Lab--minWM"
    hy_base = tmp_path / "ckpts" / "tencent--HunyuanVideo-1.5"
    wan_base = tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-1.3B"
    wow = tmp_path / "ckpts" / "X-Humanoid--WoW-1-Wan-14B-600k"
    for path in (minwm, hy_base, wan_base, wow):
        path.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _minwm_default_ref() == str(minwm)
    assert _minwm_hy_default_load_kwargs() == {"base_model_path": str(hy_base)}
    assert _minwm_wan_default_load_kwargs() == {"base_model_path": str(wan_base)}
    assert _wow_default_ref() == str(wow)


def test_magicworld_rejects_invalid_frame_count_before_expensive_runtime_start() -> None:
    assert _magicworld_latent_frames(21) == 6
    with pytest.raises(ValueError, match="maps to 5 latent frames"):
        _magicworld_latent_frames(17)


def test_magicworld_resolves_depthpro_from_default_plural_root_without_env(tmp_path, monkeypatch) -> None:
    depth_pro = tmp_path / "ckpts" / "apple--DepthPro" / "depth_pro.pt"
    depth_pro.parent.mkdir(parents=True)
    depth_pro.touch()
    monkeypatch.delenv("DEPTH_PRO_CHECKPOINT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_CKPT_DIR", raising=False)
    monkeypatch.setattr(
        magicworld_runtime,
        "checkpoint_root_candidates",
        lambda: (tmp_path / "ckpt", tmp_path / "ckpts"),
    )

    assert magicworld_runtime._resolve_depth_pro_checkpoint() == depth_pro.resolve()

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend
from worldfoundry.base_models.diffusion_model.recipes.hunyuan_video import (
    hunyuan_video15_i2v_recipe,
    hunyuan_video15_t2v_recipe,
    hunyuan_video_i2v_recipe,
    hunyuan_video_t2v_recipe,
)
from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.pipelines.hunyuan_video.pipeline_hunyuan_video import NativeHunyuanVideoPipeline
from worldfoundry.studio.inference import catalog as studio_catalog
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.framepack.worldfoundry_runner import _parse_bool
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime


def test_longsana_workspace_uses_full_long_video_defaults() -> None:
    entry = find_entry("longsana-video-2b-480p")

    assert entry.default_call_kwargs["num_frames"] == 161
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert (entry.default_call_kwargs["height"], entry.default_call_kwargs["width"]) == (480, 832)
    assert entry.default_call_kwargs["guidance_scale"] == 1.0
    assert "negative_prompt" in entry.call_params


def test_wan_14b_workspace_uses_full_720p_defaults() -> None:
    entry = find_entry("wan2.1-t2v-14b")

    assert entry.default_model_ref.endswith("Wan2.1-T2V-14B")
    assert entry.default_call_kwargs == {
        "num_frames": 81,
        "height": 720,
        "width": 1280,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "shift": 5.0,
        "fps": 16,
        "seed": 42,
    }


def test_framepack_workspace_uses_official_full_quality_defaults() -> None:
    entry = find_entry("framepack")
    runtime = OfficialVideoRuntime.from_model_id("framepack")

    assert runtime.runtime["defaults"]["seconds"] == 5
    assert runtime.runtime["defaults"]["latent_window_size"] == 9
    assert runtime.runtime["defaults"]["num_steps"] == 25
    assert runtime.runtime["defaults"]["use_teacache"] is True
    assert runtime.runtime["defaults"]["mp4_crf"] == 16
    assert any(
        str(path).endswith("/lllyasviel--FramePackI2V_HY")
        for path in runtime.runtime["checkpoint_candidates"]
    )

    tunable = {"cfg", "gs", "rs", "use_teacache", "mp4_crf"}
    assert tunable.issubset(entry.call_params)

    command = runtime.runtime["command"]
    expected_cli_values = {
        "--cfg": "{cfg}",
        "--gs": "{gs}",
        "--rs": "{rs}",
        "--use-teacache": "{use_teacache}",
        "--mp4-crf": "{mp4_crf}",
    }
    for flag, placeholder in expected_cli_values.items():
        position = command.index(flag)
        assert command[position + 1] == placeholder


def test_framepack_boolean_cli_values_are_not_silently_coerced() -> None:
    assert _parse_bool("true") is True
    assert _parse_bool("false") is False
    assert _parse_bool("0") is False
    with pytest.raises(argparse.ArgumentTypeError, match="expected a boolean value"):
        _parse_bool("not-a-boolean")


def test_krea_runtime_accepts_normalized_hugging_face_checkpoint_directory() -> None:
    runtime = OfficialVideoRuntime.from_model_id("krea-realtime-video")

    assert any(
        str(path).endswith("/krea--krea-realtime-video")
        for path in runtime.runtime["checkpoint_candidates"]
    )
    model_folder_flag = runtime.runtime["command"].index("--model-folder")
    assert runtime.runtime["command"][model_folder_flag + 1] == "{checkpoint_parent}"


def test_wan_fun_camera_workspace_uses_upstream_full_demo_defaults() -> None:
    entry = find_entry("wan21-fun-1p3b-cam")

    assert entry.default_call_kwargs["num_frames"] == 49
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert (entry.default_call_kwargs["height"], entry.default_call_kwargs["width"]) == (480, 832)


def test_autoregressive_world_demos_do_not_use_one_block_smoke_defaults() -> None:
    abot = find_entry("abot-world-0-5b-lf")
    longvie = find_entry("longvie-1")
    longvie2 = find_entry("longvie-2")

    assert (abot.default_call_kwargs["num_frames"], abot.default_call_kwargs["num_blocks"]) == (57, 5)
    assert longvie.default_call_kwargs["num_frames"] == 81
    assert longvie.default_call_kwargs["num_inference_steps"] == 50
    assert longvie2.default_load_kwargs["torchrun_nproc_per_node"] == 4
    assert longvie2.default_load_kwargs["enable_vram_management"] is False


def test_pusa_workspace_uses_official_full_length_720p_defaults() -> None:
    entry = find_entry("pusa-vidgen")

    assert (entry.default_call_kwargs["height"], entry.default_call_kwargs["width"]) == (720, 1280)
    assert entry.default_call_kwargs["num_frames"] == 81
    # Four steps is the official LightX2V distilled schedule, not a smoke reduction.
    assert entry.default_call_kwargs["num_inference_steps"] == 4


def test_pusa_lightx2v_load_defaults_accept_hugging_face_mirror_name(monkeypatch) -> None:
    requested_names: list[tuple[str, ...]] = []

    def capture_checkpoint_names(*names: str, fallback: str = "") -> str:
        requested_names.append(names)
        return fallback

    monkeypatch.setattr(studio_catalog, "_checkpoint_model_ref", capture_checkpoint_names)
    studio_catalog._pusa_vidgen_default_load_kwargs()

    assert (
        "lightx2v--Wan2.2-Lightning",
        "Wan2.2-Lightning",
    ) in requested_names


def test_hunyuanvideo15_workspace_uses_full_nondistilled_quality_defaults() -> None:
    t2v = find_entry("hunyuanvideo-1.5-t2v")
    i2v = find_entry("hunyuanvideo-1.5-i2v")

    assert (t2v.default_call_kwargs["height"], t2v.default_call_kwargs["width"]) == (720, 1280)
    assert (i2v.default_call_kwargs["height"], i2v.default_call_kwargs["width"]) == (720, 544)
    for entry in (t2v, i2v):
        assert entry.default_call_kwargs["num_frames"] == 121
        assert entry.default_call_kwargs["num_inference_steps"] == 50
        assert entry.default_call_kwargs["guidance_scale"] == 6.0
        assert entry.default_load_kwargs["attention_backend"] == "flash"

    assert hunyuan_video15_t2v_recipe().checkpoints["transformer"].files == (
        "transformer/720p_t2v/diffusion_pytorch_model.safetensors",
    )
    assert hunyuan_video15_i2v_recipe().checkpoints["transformer"].files == (
        "transformer/720p_i2v/diffusion_pytorch_model.safetensors",
    )
    resources = hunyuan_video15_t2v_recipe().checkpoints["resources"].files
    assert "text_encoder/byt5-small/model.safetensors" in resources
    assert "text_encoder/Glyph-SDXL-v2/checkpoints/model.safetensors" in resources
    assert all(not name.endswith((".bin", ".pt")) for name in resources)

    assert NativeHunyuanVideoPipeline._attention_policy(
        "auto", model_id="hunyuanvideo-1.5-t2v"
    ) is AttentionBackend.FLASH
    assert NativeHunyuanVideoPipeline._attention_policy(
        "torch", model_id="hunyuanvideo-1.5-t2v"
    ) is AttentionBackend.TORCH


def test_hunyuanvideo15_workspace_composes_complete_local_component_roots(
    tmp_path, monkeypatch
) -> None:
    weights = tmp_path / "tencent--HunyuanVideo-1.5"
    for transformer_dir in ("720p_t2v", "720p_i2v"):
        root = weights / "transformer" / transformer_dir
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").touch()
        (root / "diffusion_pytorch_model.safetensors").touch()
    (weights / "vae").mkdir()
    (weights / "vae/config.json").touch()
    (weights / "vae/diffusion_pytorch_model.safetensors").touch()

    resource_files = (
        "config.json",
        "text_encoder/llm/config.json",
        "text_encoder/byt5-small/config.json",
        "text_encoder/byt5-small/model.safetensors",
        "text_encoder/Glyph-SDXL-v2/checkpoints/model.safetensors",
        "text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
        "text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
    )
    for relative_path in resource_files:
        path = weights / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    resources = tmp_path / "HunyuanVideo-1.5"
    vision_files = (
        "vision_encoder/siglip/image_encoder/config.json",
        "vision_encoder/siglip/image_encoder/model.safetensors",
        "vision_encoder/siglip/feature_extractor/preprocessor_config.json",
    )
    for relative_path in vision_files:
        path = resources / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    t2v = studio_catalog._hunyuanvideo15_default_load_kwargs(image_to_video=False)
    i2v = studio_catalog._hunyuanvideo15_default_load_kwargs(image_to_video=True)

    assert t2v == {
        "attention_backend": "flash",
        "checkpoint_overrides": {
            "transformer": str(weights.resolve()),
            "vae": str(weights.resolve()),
            "resources": str(weights.resolve()),
        },
    }
    assert i2v["checkpoint_overrides"] == {
        "transformer": str(weights.resolve()),
        "vae": str(weights.resolve()),
        "resources": str(weights.resolve()),
        "vision": str((resources / "vision_encoder/siglip").resolve()),
    }


def test_hunyuanvideo_pipeline_accepts_cli_frame_alias() -> None:
    class CapturingNativePipeline:
        def __init__(self) -> None:
            self.request = None

        def __call__(self, request):
            self.request = request
            return SimpleNamespace(sample=torch.zeros(1, 3, 9, 8, 8), latents=None, metadata={})

    native = CapturingNativePipeline()
    pipeline = NativeHunyuanVideoPipeline(
        native_pipeline=native,
        device="cpu",
        model_id="hunyuanvideo-1.5-t2v",
    )

    pipeline("a cat", frames=9, output_type="latent")

    assert native.request.num_frames == 9


def test_original_hunyuanvideo_declares_external_text_encoder_repositories() -> None:
    for recipe in (hunyuan_video_t2v_recipe(), hunyuan_video_i2v_recipe()):
        assert recipe.checkpoints["primary"].repo_id == (
            "xtuner/llava-llama-3-8b-v1_1-transformers"
        )
        assert recipe.checkpoints["primary"].files == ("config.json",)
        assert recipe.checkpoints["clip"].repo_id == "openai/clip-vit-large-patch14"
        assert recipe.checkpoints["clip"].files == ("config.json",)
        assert "resources" not in recipe.checkpoints


def test_easyanimate_i2v_default_prompt_matches_its_reference_image() -> None:
    entry = find_entry("easyanimate_i2v")

    assert "sparkler" in entry.default_prompt.lower()
    assert entry.default_input_path.endswith("studio_demo/00/image.jpg")


def test_matrix_game_35_defaults_match_their_reference_image() -> None:
    for model_id in (
        "matrix-game-3.5-first-person",
        "matrix-game-3.5-third-person",
    ):
        entry = find_entry(model_id)

        assert "sparkler" in entry.default_prompt.lower()
        assert "young man" in entry.default_prompt.lower()
        assert entry.default_input_path.endswith("studio_demo/00/image.jpg")
        assert entry.default_call_kwargs["steps"] == 25
        assert entry.default_call_kwargs["num_blocks"] == 1


def test_stable_video_infinity_uses_its_official_480p_demo_fixture() -> None:
    entry = find_entry("stable-video-infinity")
    spec = get_model_inference_spec("stable-video-infinity")

    assert entry.default_input_path.endswith("stable-video-infinity/svi-2.0/frame.jpg")
    assert "water shimmers" in entry.default_prompt.lower()
    assert spec is not None
    defaults = spec.tasks[0].default_call_kwargs
    assert defaults["num_frames"] == 81
    assert defaults["num_inference_steps"] == 50
    assert defaults["num_motion_frames"] == 5
    assert defaults["prompt_repeat_times"] == 2


def test_dreamx_world_defaults_are_directly_runnable() -> None:
    for model_id in ("dreamx-world-5b", "dreamx-world-5b-cam"):
        entry = find_entry(model_id)

        assert "coastal cliff" in entry.default_prompt.lower()
        assert "ocean" in entry.default_prompt.lower()
        assert entry.default_input_path.endswith("dreamx_world/007.jpg")


def test_minwm_and_magicworld_prompts_match_their_rainy_aviary_fixture() -> None:
    for model_id in ("minwm-hy-action2v", "magicworld"):
        entry = find_entry(model_id)

        assert entry.default_input_path.endswith("minwm/first_frame.png")
        assert "rain" in entry.default_prompt.lower()
        assert "aviary" in entry.default_prompt.lower()

    magicworld = find_entry("magicworld")
    assert "forward" in magicworld.default_prompt.lower()
    assert magicworld.default_call_kwargs["native_rows"].endswith("camera_1_1_0.txt")


def test_vmem_default_uses_a_scene_suitable_for_forward_navigation() -> None:
    entry = find_entry("vmem")

    assert entry.default_input_path.endswith("dualcamctrl/demo_pic/route66.jpg")
    assert "route 66" in entry.default_prompt.lower()
    assert "forward" in entry.default_prompt.lower()


def test_cosmos_transfer_uses_a_control_video_demo_fixture() -> None:
    entry = find_entry("cosmos-transfer-2.5")

    assert entry.default_input_path.endswith("longcat_video/motorcycle.mp4")
    assert entry.default_task_type == "video-to-world"


def test_failed_world_demo_defaults_use_matching_image_fixtures() -> None:
    bridge_models = ("hunyuan-worldplay", "yume", "yume-1p5")
    for model_id in bridge_models:
        entry = find_entry(model_id)
        assert entry.default_input_path.endswith("hunyuan_worldplay/test.png")
        assert "bridge" in entry.default_prompt.lower()

    assert find_entry("yume").default_load_kwargs["t5_cpu"] is True

    hunyuan_worldplay = find_entry("hunyuan-worldplay")
    assert {"enable_offloading", "enable_group_offloading"}.issubset(
        hunyuan_worldplay.load_params
    )
    assert hunyuan_worldplay.default_load_kwargs == {
        "enable_offloading": True,
        "enable_group_offloading": True,
        "overlap_group_offloading": False,
    }

    stable_virtual_camera = find_entry("stable-virtual-camera")
    assert stable_virtual_camera.default_input_path.endswith(
        "stable_virtual_camera/basic/blue-car.jpg"
    )
    assert stable_virtual_camera.default_call_kwargs["H"] == 576
    assert stable_virtual_camera.default_call_kwargs["W"] == 576
    assert stable_virtual_camera.default_call_kwargs["T"] == 21
    assert stable_virtual_camera.default_call_kwargs["num_steps"] == 50

    oasis = find_entry("oasis-500m")
    assert oasis.default_input_path.endswith("oasis/sample_image_0.png")
    assert oasis.default_call_kwargs["num_frames"] == 32
    assert oasis.default_call_kwargs["ddim_steps"] == 10


def test_neoverse_full_demo_offloads_inactive_components_without_reducing_quality() -> None:
    entry = find_entry("neoverse")

    assert entry.default_load_kwargs["enable_vram_management"] is True
    assert entry.default_call_kwargs["num_frames"] == 81
    assert entry.default_call_kwargs["num_inference_steps"] == 4
    assert (entry.default_load_kwargs["height"], entry.default_load_kwargs["width"]) == (336, 560)

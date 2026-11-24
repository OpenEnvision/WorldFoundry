import ast
import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_local_source(relative: str) -> Path:
    """Skip when a license-gated source tree is absent from this checkout."""
    path = REPO_ROOT / relative
    if not path.exists():
        raise pytest.skip.Exception(f"license-gated source tree is not present: {relative}")
    return path


def test_worldfm_moge_reuses_base_model_directly():
    removed_runtime = REPO_ROOT / "worldfoundry/representations/point_clouds_generation/worldfm/moge_official"
    moge_pano = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/worldfm/moge_pano.py"
    representation = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/worldfm/runtime.py"
    base_modules_path = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth/moge/model/modules.py"
    base_v1_path = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth/moge/model/v1.py"

    assert not removed_runtime.exists()
    assert "worldfoundry.base_models.three_dimensions.depth.moge.model.v2" in moge_pano.read_text(encoding="utf-8")
    assert "moge_official" not in representation.read_text(encoding="utf-8")
    assert "perception_core.general_perception.dinov2" in base_modules_path.read_text(encoding="utf-8")
    assert "perception_core.general_perception.dinov2" in base_v1_path.read_text(encoding="utf-8")


def test_worldfm_moge_model_and_utils_forward_to_canonical_base():
    removed_runtime = REPO_ROOT / "worldfoundry/representations/point_clouds_generation/worldfm/moge_official"
    assert not removed_runtime.exists()


def test_empty_wan_simple_wow_training_source_is_not_packaged():
    runtime_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_simple_wow/wan_simple"

    assert not runtime_root.parent.exists()
    assert not (runtime_root / ".github/workflows/publish.yaml").exists()
    assert not (runtime_root / "train_wan_i2v.py").exists()
    assert not (runtime_root / "train_wan_i2v_1.3b.py").exists()
    assert not (runtime_root / "train_wan_i2v_data.py").exists()


def test_moge_imports_resolve_to_canonical_base_model():
    source_roots = [
        REPO_ROOT / "worldfoundry/base_models",
        REPO_ROOT / "worldfoundry/synthesis",
    ]
    direct_external_imports = []
    canonical_imports = []

    for root in source_roots:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "from moge.model.v1 import MoGeModel" in text:
                direct_external_imports.append(str(path.relative_to(REPO_ROOT)))
            if "worldfoundry.base_models.three_dimensions.depth.moge.model.v1" in text:
                canonical_imports.append(str(path.relative_to(REPO_ROOT)))

    assert direct_external_imports == []
    assert canonical_imports


def test_no_runtime_profile_loaders_are_copied_into_pipeline_wrappers():
    duplicate_loaders = []

    for path in sorted((REPO_ROOT / "worldfoundry/pipelines").glob("**/pipeline_*.py")):
        text = path.read_text(encoding="utf-8")
        if "load_runtime_profile" in text:
            duplicate_loaders.append(str(path.relative_to(REPO_ROOT)))

    assert duplicate_loaders == []


def test_runtime_video_empty_package_is_not_reintroduced():
    assert not (REPO_ROOT / "worldfoundry/synthesis/visual_generation/runtime_video").exists()


def test_empty_synthesis_compat_packages_are_not_reintroduced():
    removed_paths = [
        "worldfoundry/synthesis/action_generation/starvla/starvla_runtime",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wan-simple/diffsynth/tokenizer_configs",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2/extension_modules/matrix_game_2_modules",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2/demo_utils/memory.py",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2/wan/vae",
        "worldfoundry/synthesis/visual_generation/worldcam/worldcam/lora",
        "worldfoundry/synthesis/visual_generation/worldcam/worldcam/utils",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/diffsynth/extensions/ESRGAN",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/diffsynth/extensions/RIFE",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/diffsynth/utils",
        "worldfoundry/synthesis/visual_generation/neoverse/neoverse_runtime/diffsynth/lora",
        "worldfoundry/synthesis/visual_generation/neoverse/neoverse_runtime/diffsynth/extensions/ESRGAN",
        "worldfoundry/synthesis/visual_generation/neoverse/neoverse_runtime/diffsynth/extensions/RIFE",
        "worldfoundry/base_models/three_dimensions/general_3d/lagernvs/runtime/vendor",
        "worldfoundry/synthesis/action_generation/roboflamingo/roboflamingo_runtime/vendor/open_flamingo_src",
        "worldfoundry/synthesis/action_generation/roboflamingo/roboflamingo_runtime/vendor",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_runtime/matrixgame/model_variants/matrixgame_dit_src",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_runtime/matrixgame/vae_variants/matrixgame_vae_src",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3_runner.py",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2/extension_modules/wanx_vae/wanx_vae_src",
        "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_runtime/cosmos_predict2/_src",
        "worldfoundry/representations/point_clouds_generation/hunyuan_world/hunyuan_mirror/mirror_src",
        "worldfoundry/representations/point_clouds_generation/worldfm/moge_official",
        "worldfoundry/representations/point_clouds_generation/pi3/pi3",
        "worldfoundry/representations/point_clouds_generation/pi3/pi3x",
        "worldfoundry/representations/point_clouds_generation/pi3/loger",
        "worldfoundry/representations/point_clouds_generation/vggt/infinite_vggt",
        "worldfoundry/representations/point_clouds_generation/cut3r/cut3r",
        "worldfoundry/representations/point_clouds_generation/hunyuan_world/hy_world_2p0/hyworldmirror",
        "worldfoundry/representations/point_clouds_generation/lingbot_map/lingbot_map_runtime",
        "worldfoundry/representations/point_clouds_generation/flash_world/flash_world",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/diffsynth",
        "worldfoundry/synthesis/visual_generation/neoverse/neoverse_runtime/diffsynth",
        "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_runtime/cosmos_oss",
        "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_runtime/cosmos_predict2",
        "worldfoundry/synthesis/visual_generation/fantasy_world/fantasy_world_runtime/FantasyWorld/diffsynth_wan21",
        "worldfoundry/synthesis/visual_generation/fantasy_world/fantasy_world_runtime/FantasyWorld/diffsynth_wan22",
        "worldfoundry/synthesis/visual_generation/fantasy_world/fantasy_world_runtime/FantasyWorld/wan",
        "worldfoundry/synthesis/visual_generation/fantasy_world/fantasy_world_runtime/FantasyWorld/vggt",
        "worldfoundry/synthesis/visual_generation/gen3c/gen3c_runner.py",
        "worldfoundry/synthesis/visual_generation/gen3c/gen3c_runtime/cosmos_predict1",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/models/cosmos_models",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/models/pipeline/gesim_cosmos2_pipeline_utils",
        "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world/wan",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/download_wan2.1.py",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/sample_longvideo.sh",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/utils",
        "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/utils/models/vda/video_depth_anything",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/cosmos_predict1",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2/wan",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3_runtime/wan",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/DynamiCrafter",
        "worldfoundry/base_models/three_dimensions/point_clouds/pixelsplat_runtime/src",
        "worldfoundry/synthesis/visual_generation/sana/sana_runtime/diffusion/model/wan",
        "worldfoundry/synthesis/visual_generation/scope/scope_runtime/diffsynth",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wan-simple",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wan-simple/diffsynth",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wow-dit-2b/cosmos_predict2",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wow-dit-7b/cosmos_predict1",
        "worldfoundry/synthesis/visual_generation/cosmos/cosmos2p5",
        "worldfoundry/synthesis/visual_generation/cosmos/cosmos_runtime",
        "worldfoundry/synthesis/visual_generation/videocrafter/videocrafter_runtime",
    ]

    assert [path for path in removed_paths if (REPO_ROOT / path).exists()] == []


def test_base_model_pure_demo_and_ui_app_dirs_are_pruned():
    base_models_root = REPO_ROOT / "worldfoundry/base_models"
    world_model_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation"
    removed_paths = [
        base_models_root / "diffusion_model/diffsynth/apps",
        base_models_root / "diffusion_model/image/sana/apps",
        base_models_root / "diffusion_model/video/wan/demo_utils",
        base_models_root / "diffusion_model/world_model/mineworld/configs",
        base_models_root / "diffusion_model/world_model/mineworld/mineworld.py",
    ]

    for path in removed_paths:
        assert not path.exists(), path
    assert not (base_models_root / "diffusion_model/world_model").exists()
    assert world_model_root.is_dir()

    demo_paths = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in base_models_root.rglob("*") if "demo" in path.name.lower()
    )

    assert demo_paths == []

    mineworld_runtime = (world_model_root / "mineworld/worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "WEB_DEMO_ENTRYPOINT" not in mineworld_runtime
    mineworld_lvm = (world_model_root / "mineworld/lvm.py").read_text(encoding="utf-8")
    mineworld_decoding = (world_model_root / "mineworld/diagonal_decoding.py").read_text(encoding="utf-8")
    assert "for_gradio" not in mineworld_lvm
    assert "for_gradio" not in mineworld_decoding
    assert "gradio" not in mineworld_lvm.lower()
    assert "gradio" not in mineworld_decoding.lower()
    mineworld_action_tokenizer = (world_model_root / "mineworld/action_tokenizer.py").read_text(encoding="utf-8")
    assert not (world_model_root / "mineworld/mcdataset.py").exists()
    assert "torch.utils.data.Dataset" not in mineworld_action_tokenizer
    assert "class MinecraftActionTokenizer" in mineworld_action_tokenizer
    mineworld_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/mineworld"
    assert {
        "1200M_16f.yaml",
        "1200M_32f.yaml",
        "300M_16f.yaml",
        "700M_16f.yaml",
        "700M_32f.yaml",
    } == {path.name for path in mineworld_config_root.glob("*.yaml")}
    mineworld_infer_script = (world_model_root / "mineworld/scripts/inference_16f_models.sh").read_text(
        encoding="utf-8"
    )
    assert "data/models/runtime/configs/mineworld" in mineworld_infer_script
    assert 'CONFIG="configs/' not in mineworld_infer_script

    fastblend_api = (base_models_root / "diffusion_model/diffsynth/extensions/FastBlend/api.py").read_text(
        encoding="utf-8"
    )
    assert "import gradio" not in fastblend_api
    assert "def on_ui_tabs" not in fastblend_api

    fastblend_init = (base_models_root / "diffusion_model/diffsynth/extensions/FastBlend/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "import cupy" not in fastblend_init

    fastblend_package = importlib.import_module(
        "worldfoundry.base_models.diffusion_model.diffsynth.extensions.FastBlend"
    )
    assert "FastBlendSmoother" in fastblend_package.__all__


def test_droid_slam_tartan_split_lives_in_data_runtime_configs():
    data_readers = REPO_ROOT / "worldfoundry/base_models/three_dimensions/slam/droid_slam/data_readers"
    data_split = REPO_ROOT / "worldfoundry/data/models/runtime/configs/droid_slam/tartan_test.txt"
    rgbd_utils = REPO_ROOT / "worldfoundry/base_models/three_dimensions/slam/droid_slam/geom/rgbd_utils.py"
    graph_utils = (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/slam/droid_slam/geom/graph_utils.py"
    ).read_text(encoding="utf-8")

    assert not data_readers.exists()
    assert data_split.is_file()
    assert rgbd_utils.is_file()
    assert "droid_slam.geom.rgbd_utils" in graph_utils


def test_perception_and_3d_base_models_do_not_package_training_data_helpers():
    base_models = REPO_ROOT / "worldfoundry/base_models"
    removed_paths = [
        base_models / "perception_core/detection/grounding_dino/datasets",
        base_models / "perception_core/tracking/track_anything/groundingdino",
        base_models / "three_dimensions/slam/droid_slam/data_readers",
        base_models / "three_dimensions/slam/droid_slam/geom/losses.py",
        base_models / "three_dimensions/depth/moge/utils/data_augmentation.py",
    ]

    for path in removed_paths:
        assert not path.exists(), path

    assert (base_models / "perception_core/detection/grounding_dino/util/transforms.py").is_file()


def test_echo_infinity_is_regular_synthesis_work_not_base_model():
    base_owner = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/echo_infinity"
    synthesis_owner = REPO_ROOT / "worldfoundry/synthesis/visual_generation/echo_infinity"
    synthesis_text = (synthesis_owner / "synthesis.py").read_text(encoding="utf-8")

    assert not base_owner.exists()
    assert (synthesis_owner / "worldfoundry_runtime.py").is_file()
    assert (synthesis_owner / "echo_infinity_runtime").is_dir()
    assert "worldfoundry.synthesis.visual_generation.echo_infinity.worldfoundry_runtime" in synthesis_text
    assert "worldfoundry.base_models.diffusion_model.video.echo_infinity" not in synthesis_text


def test_wan_vram_helpers_are_core_owned():
    import worldfoundry.core.vram as core_vram

    removed_shims = [
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/utils/memory.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/echo_infinity/echo_infinity_runtime/utils/memory.py",
    ]

    assert hasattr(core_vram, "DynamicSwapInstaller")
    assert hasattr(core_vram, "get_cuda_free_memory_gb")
    assert not [path for path in removed_shims if path.exists()]


def test_lyra1_pipeline_can_plan_without_external_representation_assets(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from worldfoundry.pipelines.lyra import pipeline_lyra1

    loaded: list[dict] = []

    def _fake_synthesis(**kwargs):
        loaded.append(kwargs)
        return SimpleNamespace(model_id="lyra-1", checkpoint_dir=None)

    # Synthesis construction resolves GEN3C and T5 checkpoints; this contract is weight-free.
    monkeypatch.setattr(pipeline_lyra1.Lyra1Synthesis, "from_pretrained", staticmethod(_fake_synthesis))

    pipe = pipeline_lyra1.Lyra1Pipeline.from_pretrained(device="cpu")

    assert len(loaded) == 1
    assert pipe.synthesis_model.model_id == "lyra-1"
    assert pipe.representation_model is None


def test_lyra1_inference_configs_live_in_data_runtime_configs():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/lyra_1/inference"
    sample_path = runtime_root / "sample.py"

    expected_configs = {
        "default.yaml",
        "3dgs_res_176_320_views_17.yaml",
        "3dgs_res_176_320_views_49.yaml",
        "3dgs_res_352_640_views_49.yaml",
        "3dgs_res_704_1280_views_49.yaml",
        "3dgs_res_704_1280_views_121.yaml",
        "3dgs_res_704_1280_views_121_multi_6.yaml",
        "3dgs_res_704_1280_views_121_multi_6_prune.yaml",
        "3dgs_res_704_1280_views_121_multi_6_dynamic.yaml",
        "3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml",
    }

    assert not (runtime_root / "configs").exists()
    assert list(runtime_root.glob("*.yaml")) == []
    assert list(runtime_root.glob("*.yml")) == []
    assert list(runtime_root.glob("*.json")) == []
    assert expected_configs == {path.name for path in config_root.glob("*.yaml")}

    sample_text = sample_path.read_text(encoding="utf-8")
    assert "package_data_path" in sample_text
    assert 'runtime", "configs", "lyra_1", "inference' in sample_text
    assert "configs/inference/default.yaml" not in sample_text
    assert "configs/training" not in sample_text
    assert "configs/demo" not in sample_text
    assert "configs/accelerate" not in sample_text


def test_pi3_representations_use_base_model_sources():
    wrappers = [
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/pi3_inference/runtime.py",
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/pi3_inference/runtime_pi3x.py",
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/loger/runtime.py",
    ]
    for path in wrappers:
        text = path.read_text(encoding="utf-8")
        assert "worldfoundry.base_models.three_dimensions.point_clouds" in text

    assert (REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/loger/pi3.py").is_file()


def test_infinite_vggt_representation_uses_base_model_source():
    wrapper = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/infinite_vggt/runtime.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.infinite_vggt" in text
    assert not (REPO_ROOT / "worldfoundry/representations/point_clouds_generation/vggt/infinite_vggt").exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/infinite_vggt/models/streamvggt.py"
    ).is_file()


def test_cut3r_representation_uses_base_model_source():
    wrapper = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/cut3r/runtime.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.cut3r" in text
    assert not (REPO_ROOT / "worldfoundry/representations/point_clouds_generation/cut3r/cut3r").exists()
    assert (REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/cut3r/model.py").is_file()


def test_hyworldmirror_2p0_runtime_uses_base_model_source():
    runtime = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/runtime.py"
    text = runtime.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0" in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/hunyuan_world/hy_world_2p0/hyworldmirror"
    ).exists()
    assert (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/models/worldmirror.py"
    ).is_file()


def test_lingbot_map_representation_uses_base_model_runtime():
    wrapper = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/lingbot_map/runtime.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "from .lingbot_map.models.gct_stream import GCTStream" in text
    assert "sys.path" not in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/lingbot_map/lingbot_map_runtime"
    ).exists()
    assert (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/lingbot_map/lingbot_map/models/gct_stream.py"
    ).is_file()


def test_flash_world_representation_uses_base_model_source():
    wrapper = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/flash_world/runtime.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.flash_world" in text
    assert not (REPO_ROOT / "worldfoundry/representations/point_clouds_generation/flash_world/flash_world").exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/flash_world/reconstruction_model.py"
    ).is_file()


def test_cogvideox_runtime_logic_lives_under_synthesis():
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/cogvideox"
    base_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/cogvideox"

    assert not (synthesis_root / "cogvideox_runtime.py").exists()
    assert not base_root.exists()
    assert (synthesis_root / "worldfoundry_runtime.py").is_file()

    runtime_text = (synthesis_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "class CogVideoXOfficialRuntime" in runtime_text
    assert "from diffusers import" in runtime_text

    for file_name in [
        "cogvideox_2b_t2v_synthesis.py",
        "cogvideox_5b_i2v_synthesis.py",
        "cogvideox_5b_t2v_synthesis.py",
    ]:
        text = (synthesis_root / file_name).read_text(encoding="utf-8")
        assert "from .worldfoundry_runtime import CogVideoX" in text
        assert "worldfoundry.base_models.diffusion_model.video.cogvideox" not in text
        assert ".cogvideox_runtime" not in text
        assert "from diffusers" not in text
        assert "CogVideoXPipeline" not in text
        assert "CogVideoXImageToVideoPipeline" not in text
        assert "CogVideoXVideoToVideoPipeline" not in text


def test_vmem_runtime_root_ignores_external_source_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from worldfoundry.synthesis.visual_generation.vmem.runtime_env import (
        DEFAULT_VMEM_RUNTIME_ROOT,
        runtime_root,
    )

    monkeypatch.delenv("VMEM_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_DIR", str(tmp_path / "github_repos"))

    assert runtime_root() == DEFAULT_VMEM_RUNTIME_ROOT.resolve()


def test_synthesis_demo_assets_and_training_payloads_are_not_reintroduced():
    removed_paths = [
        "worldfoundry/synthesis/visual_generation/gen3c/gen3c_runtime/cosmos_predict1/tokenizer/notebook",
        "worldfoundry/synthesis/visual_generation/gen3c/gen3c_runtime/cosmos_predict1/tokenizer/test_data",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3/assets",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/vipe",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/vipe/assets",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/eval",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/train",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/demo.py",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/DynamiCrafter/assets",
        "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/DynamiCrafter/prompts",
        "worldfoundry/synthesis/visual_generation/dynamicrafter_pandora/DynamiCrafter/scripts/gradio",
        "worldfoundry/training/visual_generation/wan/demo_utils",
        "worldfoundry/synthesis/visual_generation/sana/sana_runtime/configs",
        "worldfoundry/synthesis/visual_generation/sana/sana_runtime/configs/sol_rl",
        "worldfoundry/synthesis/visual_generation/sana/sana_runtime/diffusion/post_training",
        "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/examples",
        "worldfoundry/synthesis/visual_generation/scope/scope_runtime/assets",
        "worldfoundry/synthesis/visual_generation/scope/scope_runtime/examples",
        "worldfoundry/base_models/diffusion_model/video/step_video/step_video_runtime/benchmark/Step-Video Prompt Guildlines.pdf",
        "worldfoundry/base_models/diffusion_model/video/step_video/step_video_runtime/benchmark/Step-Video 提示词指南.pdf",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/assets",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/data/training",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/data/demo",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/warp_as_history/training",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3/assets",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3/examples",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3/demo_gradio.py",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3/example.py",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3/example_mm.py",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/third_party/Pi3/example_vo.py",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/data/demo/angel",
        "worldfoundry/synthesis/visual_generation/warp_as_history/warp_as_history_runtime/data/demo/dragon",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/cosmos_predict1/tokenizer/test_data",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3/da3_streaming/loop_utils/salad/dataloaders",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3/da3_streaming/loop_utils/salad/datasets",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3/da3_streaming/loop_utils/salad/eval.py",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3/da3_streaming/loop_utils/salad/main.py",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/experiments",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/gesim_video_gen_examples",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/video_gen_examples",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/web_infer_scripts",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/web_infer_utils",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/runner/ge_trainer.py",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/scripts/get_statistics.py",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/scripts/train.sh",
        "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/utils/optimizer_utils.py",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_anything_3/.pre-commit-config.yaml",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/demo",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/scripts/wow_wan14b_demo.sh",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wan-simple/examples",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wow-dit-2b/requires.txt",
        "worldfoundry/synthesis/visual_generation/wow/wow_runtime/dit_models/wow-dit-7b/requires.txt",
        "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_runtime/examples",
        "worldfoundry/base_models/diffusion_model/video/step_video/step_video_runtime/benchmark",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/train_i2v_depth_normal_sft.sh",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/train_i2v_depth_normal_lora.sh",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/DATA.md",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/scripts/preprocess_bridge.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/scripts/rendering_points.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/scripts/utils.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/scripts/video_depth.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/scripts/video_normal.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/tesseract/args.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/tesseract/i2v_depth_normal_sft.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/tesseract/i2v_depth_normal_lora.py",
        "worldfoundry/synthesis/action_generation/tesseract/tesseract_runtime/tesseract/robodataset.py",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/.dockerignore",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/.gitattributes",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/.gitignore",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/.pre-commit-config.yaml",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/setup.cfg",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/assets",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/configs",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/scripts/download.py",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/scripts/pack_data.py",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/scripts/train.py",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/giga_world_0/giga_world_0_trainer.py",
        "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime/giga_world_0/giga_world_0_transforms.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/scripts/download_gen3c_checkpoints.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/scripts/download_guardrail_checkpoints.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/scripts/download_lyra_checkpoints.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/scripts/download_tokenizer_checkpoints.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/scripts/test_environment.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/scripts/bash",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/src/eval/compute_metrics_datasets.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/src/models/utils/loss.py",
        "worldfoundry/synthesis/visual_generation/lyra_1/lyra1_runtime/src/visu",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/callbacks",
        "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/schedulers/rectified_flow.py",
        "worldfoundry/synthesis/visual_generation/irasim/irasim_runtime/util.py",
        "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_runtime/scripts",
        "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime/scripts/download.sh",
        "worldfoundry/synthesis/visual_generation/open_magvit2/open_magvit2_runtime/src/Open_MAGVIT2/data",
        "worldfoundry/synthesis/visual_generation/open_magvit2/open_magvit2_runtime/src/Open_MAGVIT2/modules/losses",
        "worldfoundry/synthesis/visual_generation/worldcam/worldcam_runtime/models/downloader.py",
        "worldfoundry/synthesis/visual_generation/worldcam/worldcam_runtime/data",
        "worldfoundry/synthesis/visual_generation/worldfm/worldfm_runtime/download.py",
        "worldfoundry/synthesis/action_generation/diffusion_policy/diffusion_policy_runtime/dataset",
        "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2_runtime/demo_utils",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/imaginaire/modules/nlp/t5xxl/t5encoder.json",
    ]

    assert [path for path in removed_paths if (REPO_ROOT / path).exists()] == []
    assert (REPO_ROOT / "worldfoundry/data/models/runtime/configs/cosmos_predict2/t5xxl/t5encoder.json").is_file()
    assert {
        "wow-dit-2b.txt",
        "wow-dit-7b.txt",
    } == {path.name for path in (REPO_ROOT / "worldfoundry/data/models/runtime/configs/wow/requirements").glob("*.txt")}
    t5_encoder_text = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/imaginaire/modules/nlp/t5xxl/t5encoder.py"
    ).read_text(encoding="utf-8")
    for marker in [
        "worldfoundry_data_path",
        '"runtime", "configs"',
        '"cosmos_predict2"',
        '"t5xxl"',
        '"t5encoder.json"',
    ]:
        assert marker in t5_encoder_text
    assert 'os.path.dirname(os.path.abspath(__file__)), "t5encoder.json"' not in t5_encoder_text


def test_lyra2_runtime_logic_lives_under_synthesis():
    runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/runtime.py"
    synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/synthesis.py"
    depth_utils = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_utils.py"
    lyra2_inference_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference"
    vipe_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/vipe"

    runtime_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")
    depth_utils_text = depth_utils.read_text(encoding="utf-8")

    assert "class Lyra2Runtime" in runtime_text
    assert "load_model_from_checkpoint" in runtime_text
    assert "load_da3_model" in runtime_text
    assert "worldfoundry.synthesis.visual_generation.lyra_2" in runtime_text
    assert "worldfoundry.base_models.three_dimensions.general_3d.vipe" in runtime_text
    assert "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3" in depth_utils_text
    assert not (lyra2_inference_root / "depth_anything_3").exists()
    assert not (lyra2_inference_root / "vipe").exists()
    assert not (lyra2_inference_root / "vipe_da3_gs_recon.py").exists()
    assert (vipe_config_root / "default.yaml").is_file()
    assert (vipe_config_root / "pipeline" / "lyra.yaml").is_file()
    assert "BaseSynthesis" not in runtime_text
    assert "Lyra2Runtime" in synthesis_text
    assert "return self.runtime.predict" in synthesis_text
    for heavy_marker in [
        "load_model_from_checkpoint",
        "load_da3_model",
        "get_umt5_embedding",
        "run_lyra2_sample",
        "_da3_infer_depth_intrinsics_single",
        "torch.load",
    ]:
        assert heavy_marker not in synthesis_text

    from worldfoundry.synthesis.visual_generation.lyra_2 import Lyra2Runtime

    assert Lyra2Runtime.__module__.startswith("worldfoundry.synthesis.")


def test_infinite_world_default_config_lives_in_data_runtime_configs(tmp_path: Path):
    from worldfoundry.synthesis.visual_generation.infinite_world import (
        InfiniteWorldSynthesis,
    )
    from worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime import (
        InfiniteWorldRuntime,
        default_config_path,
    )

    config_path = REPO_ROOT / "worldfoundry/data/models/runtime/configs/infinite_world/inference.yaml"
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/infinite_world_runtime"
    old_hidden_config = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world" / ("infworld_" + "config.yaml")
    )
    old_runtime_dirs = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/configs",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/models",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/vae",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/utils",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/context_parallel",
    ]

    assert config_path.is_file()
    assert (runtime_root / "models/dit_model.py").is_file()
    assert (runtime_root / "vae/vae.py").is_file()
    assert not old_hidden_config.exists()
    assert [path for path in old_runtime_dirs if path.exists()] == []
    assert Path(default_config_path()).resolve() == config_path.resolve()
    config_text = config_path.read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime" in config_text
    assert "worldfoundry.synthesis.visual_generation.infinite_world.models" not in config_text
    runtime_text = (runtime_root / "inference.py").read_text(encoding="utf-8")
    synthesis_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/infinite_world_synthesis.py"
    ).read_text(encoding="utf-8")
    assert "worldfoundry.synthesis" not in runtime_text
    assert "InfiniteWorldRuntime" in synthesis_text
    assert InfiniteWorldRuntime.__module__.startswith("worldfoundry.synthesis.")

    plan = InfiniteWorldSynthesis.plan(pretrained_model_path=tmp_path, device="cuda:4")

    assert plan["config_path"] == str(config_path)


def test_camera_control_configs_live_in_data_runtime_configs():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/camera_control"
    old_paths = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/camera_control/configs/cameractrl_256_384.yaml",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/camera_control",
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/configs/inference/config_both.yaml",
    ]
    synthesis_paths = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/cameractrl/synthesis.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/synthesis.py",
    ]

    assert (config_root / "cameractrl_256_384.yaml").is_file()
    assert (config_root / "motionctrl_config_both.yaml").is_file()
    assert [str(path) for path in old_paths if path.exists()] == []

    cameractrl_runtime_text = (REPO_ROOT / "worldfoundry/synthesis/visual_generation/cameractrl/runtime.py").read_text(
        encoding="utf-8"
    )
    cameractrl_synthesis_text = synthesis_paths[0].read_text(encoding="utf-8")
    assert "package_data_path" in cameractrl_runtime_text
    assert '"runtime", "configs"' in cameractrl_runtime_text
    assert "worldfoundry.synthesis" not in cameractrl_runtime_text
    assert "CameraCtrlRuntime" in cameractrl_synthesis_text
    assert "get_pipeline" not in cameractrl_synthesis_text
    assert "ray_condition" not in cameractrl_synthesis_text
    assert "save_videos_grid" not in cameractrl_synthesis_text
    motionctrl_base_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")
    assert "package_data_path" in motionctrl_base_text
    assert '"runtime", "configs"' in motionctrl_base_text


def test_animatediff_configs_live_in_data_runtime_configs():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/animatediff"
    old_config_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/animatediff/configs"
    old_official_script = REPO_ROOT / "worldfoundry/synthesis/visual_generation/animatediff/official_animate.py"
    old_dataset_runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/animatediff/runtime/data"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/animatediff/animatediff_synthesis.py"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/animatediff/worldfoundry_runtime.py"

    assert (config_root / "inference/inference-v2.yaml").is_file()
    assert (config_root / "prompts/1_animate/1_1_animate_RealisticVision.yaml").is_file()
    assert not old_config_root.exists()
    assert not old_official_script.exists()
    assert not old_dataset_runtime.exists()

    runtime_text = runtime_path.read_text(encoding="utf-8")
    synthesis_text = synthesis_path.read_text(encoding="utf-8")
    assert "package_data_path" in runtime_text
    assert '"runtime", "configs"' in runtime_text
    assert "worldfoundry.synthesis" not in runtime_text
    assert "AnimateDiffRuntime" in synthesis_text
    assert "UNet3DConditionModel" not in synthesis_text
    assert "AnimationPipeline" not in synthesis_text
    assert "save_videos_grid" not in synthesis_text
    assert "WebVid10M" not in runtime_text
    assert "torch.utils.data" not in runtime_text
    assert not any(path.name == "run_animatediff_exact_parity.py" for path in (REPO_ROOT / "scripts").rglob("*.py"))


def test_zeroscope_runtime_logic_lives_under_synthesis():
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/zeroscope/zeroscope_synthesis.py"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/zeroscope/worldfoundry_runtime.py"

    assert runtime_path.is_file()
    runtime_text = runtime_path.read_text(encoding="utf-8")
    synthesis_text = synthesis_path.read_text(encoding="utf-8")

    assert "worldfoundry.synthesis" not in runtime_text
    assert "TextToVideoSDPipeline" in runtime_text
    assert "export_to_video" in runtime_text
    assert "ZeroScopeRuntime" in synthesis_text
    assert "TextToVideoSDPipeline" not in synthesis_text
    assert "export_to_video" not in synthesis_text
    assert "hashlib" not in synthesis_text


def test_lingbot_world_runtime_lives_under_synthesis():
    facade_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lingbot_world"
    runtime_root = facade_root / "lingbot_world_runtime"
    facade_runtime = facade_root / "runtime.py"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lingbot/lingbot_world_runtime"
    synthesis_path = facade_root / "lingbot_world_synthesis.py"

    assert not old_runtime_root.exists()
    assert (runtime_root / "image2video.py").is_file()
    assert (runtime_root / "configs/wan_i2v_A14B.py").is_file()
    assert facade_runtime.is_file()

    runtime_text = facade_runtime.read_text(encoding="utf-8")
    assert "class LingBotWorldRuntime" in runtime_text
    assert "WanI2V" in runtime_text
    assert "WanI2VFast" in runtime_text
    assert "def _resolve_model_paths" in runtime_text
    assert "def _generate_video" in runtime_text
    assert "def predict" in runtime_text
    assert "worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime" in runtime_text

    text = synthesis_path.read_text(encoding="utf-8")
    assert "from .runtime import" in text
    assert "LingBotWorldRuntime" in text
    assert "return self.runtime.predict" in text
    assert "worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime" not in text
    assert ".lingbot_world.configs" not in text
    assert "WanI2V" not in text
    assert "WanI2VFast" not in text
    assert "def _resolve_model_paths" not in text
    assert "def _generate_video" not in text
    assert "tempfile.TemporaryDirectory" not in text
    assert "signature(" not in text
    assert "import torch" not in text
    assert "import numpy" not in text
    assert 'Path(__file__).resolve().parent / "lingbot_world"' not in text


def test_kling_astra_runtime_lives_under_synthesis():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra_runtime"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra_synthesis.py"

    assert not old_runtime_root.exists()
    assert (runtime_root / "pipelines/wan_video_astra.py").is_file()
    assert (runtime_root / "models/model_manager.py").is_file()
    assert (runtime_root / "model_registry.py").is_file()
    assert not (runtime_root / "configs").exists()
    assert (REPO_ROOT / "worldfoundry/data/models/runtime/configs/kling/astra/model_loader.yaml").is_file()

    text = synthesis_path.read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.kling.astra_runtime" in text
    assert ".astra." not in text


def test_kling_recammaster_runtime_lives_under_synthesis():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/recammaster_runtime"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/recammaster"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/recammaster_synthesis.py"

    assert not old_runtime_root.exists()
    assert (runtime_root / "pipelines/wan_video_recammaster.py").is_file()
    assert (runtime_root / "models/wan_model.py").is_file()
    assert (runtime_root / "recammaster_model_registry.py").is_file()
    assert not (runtime_root / "recammaster_model_configs.py").exists()

    text = synthesis_path.read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.kling.recammaster_runtime" in text
    assert ".recammaster." not in text


def test_matrix_game_1_runtime_logic_lives_under_base_models():
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_runtime/runtime.py"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_synthesis.py"

    runtime_text = runtime_path.read_text(encoding="utf-8")
    synthesis_text = synthesis_path.read_text(encoding="utf-8")

    assert "class MatrixGame1Runtime" in runtime_text
    assert "def predict(" in runtime_text
    assert "subprocess.run" in runtime_text
    assert "preflight artifacts are no longer emitted" in runtime_text
    assert "MatrixGame1Runtime" in synthesis_text
    for marker in [
        "subprocess.run",
        "write_blocked_plan",
        "DEFAULT_CHECKPOINT_DIR",
        "_blocked_reasons",
        "package_root",
        "_json_safe",
    ]:
        assert marker not in synthesis_text


def test_open_magvit2_runtime_logic_lives_under_synthesis():
    runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/open_magvit2/worldfoundry_runtime.py"
    synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/open_magvit2/open_magvit2_synthesis.py"

    base_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class OpenMAGVIT2Runtime" in base_text
    assert "load_model" in base_text
    assert "save_class_image" in base_text
    assert "OpenMAGVIT2Runtime" in synthesis_text
    for marker in [
        "DEFAULT_SHARED_HFD_ROOT",
        "sys.path",
        "load_model",
        "save_class_image",
        "_resolve_checkpoint",
        "_ensure_model",
    ]:
        assert marker not in synthesis_text


def test_open_magvit2_inference_configs_live_in_data_runtime_configs():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/open_magvit2/open_magvit2_runtime"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/open_magvit2/worldfoundry_runtime.py"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/open_magvit2"

    assert not (runtime_root / "configs").exists()
    assert {
        "imagenet_conditional_llama_B.yaml",
        "imagenet_conditional_llama_L.yaml",
        "imagenet_conditional_llama_XL.yaml",
    } == {path.name for path in config_root.glob("*.yaml")}

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert "package_data_path" in runtime_text
    assert 'runtime", "configs", "open_magvit2' in runtime_text
    assert "configs/Open-MAGVIT2" not in runtime_text
    assert "pretrain_lfqgan" not in "\n".join(path.name for path in config_root.glob("*.yaml"))
    assert "ucf101" not in "\n".join(path.name for path in config_root.glob("*.yaml"))


def test_show_o_runtime_logic_lives_under_synthesis():
    runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/show_o/worldfoundry_runtime.py"
    synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/show_o/show_o_synthesis.py"

    base_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class ShowORuntime" in base_text
    assert "UniversalPrompting" in base_text
    assert "Showo.from_pretrained" in base_text
    assert "ShowORuntime" in synthesis_text
    for marker in [
        "DEFAULT_SHARED_HFD_ROOT",
        "sys.path",
        "OmegaConf",
        "Showo.from_pretrained",
        "UniversalPrompting",
        "_ensure_runtime",
        "_runtime_config",
    ]:
        assert marker not in synthesis_text


def test_hunyuan_gamecraft_camera_helpers_live_in_base_runtime():
    _require_local_source("worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft")
    gamecraft = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.runtime"
    )

    poses = gamecraft.ActionToPoseFromID("w", value=0.1, duration=2)
    pose_embeds, uncond_pose_embeds, selected = gamecraft.GetPoseEmbedsFromPoses(
        poses,
        16,
        16,
        2,
        start_index=0,
    )

    assert gamecraft.HunyuanGameCraftRuntime.__module__.startswith("worldfoundry.synthesis.")
    assert len(poses) == 3
    assert len(selected) == 2
    assert tuple(pose_embeds.shape) == (2, 6, 16, 16)
    assert tuple(uncond_pose_embeds.shape) == (2, 6, 16, 16)


def test_videocrafter_configs_live_in_data_runtime_configs():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/videocrafter"
    old_runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/videocrafter/videocrafter_runtime"
    synthesis_paths = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/videocrafter/videocrafter1_i2v_synthesis.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/videocrafter/videocrafter1_t2v_synthesis.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/videocrafter/videocrafter2_t2v_synthesis.py",
    ]

    assert [path.name for path in sorted(config_root.glob("*.yaml"))] == [
        "inference_i2v_512_v1.0.yaml",
        "inference_t2v_1024_v1.0.yaml",
        "inference_t2v_512_v1.0.yaml",
        "inference_t2v_512_v2.0.yaml",
        "runtime_defaults.yaml",
    ]
    assert not (old_runtime / "configs").exists()
    for path in synthesis_paths:
        text = path.read_text(encoding="utf-8")
        assert "runtime/configs/videocrafter" in text
        assert "videocrafter_runtime/configs" not in text


def test_genie_envisioner_runtime_has_no_vendored_configs():
    old_config_root = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/configs"
    )

    assert not old_config_root.exists()


def test_hunyuan_worldmirror_common_layers_use_canonical_owner():
    removed_layer_files = []
    for root in [
        "worldfoundry/base_models/three_dimensions/point_clouds/hunyuan_mirror/models/layers",
    ]:
        removed_layer_files.extend(
            [
                f"{root}/drop_path.py",
                f"{root}/layer_scale.py",
                f"{root}/mlp.py",
                f"{root}/patch_embed.py",
                f"{root}/rope.py",
                f"{root}/swiglu_ffn.py",
            ]
        )
    removed_layer_files.extend(
        [
            "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/drop_path.py",
            "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/layer_scale.py",
            "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/patch_embed.py",
            "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/rope.py",
            "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/swiglu_ffn.py",
        ]
    )
    common_layer_files = [
        "worldfoundry/base_models/diffusion_model/video/common_layers/drop_path.py",
        "worldfoundry/base_models/diffusion_model/video/common_layers/layer_scale.py",
        "worldfoundry/base_models/diffusion_model/video/common_layers/mlp.py",
        "worldfoundry/base_models/diffusion_model/video/common_layers/patch_embed.py",
        "worldfoundry/base_models/diffusion_model/video/common_layers/rope.py",
        "worldfoundry/base_models/diffusion_model/video/common_layers/swiglu_ffn.py",
    ]

    assert [path for path in removed_layer_files if (REPO_ROOT / path).exists()] == []
    assert [path for path in common_layer_files if (REPO_ROOT / path).exists()] == []
    assert (REPO_ROOT / "worldfoundry/core/nn/blocks/layers.py").exists()
    assert (REPO_ROOT / "worldfoundry/core/attention/rotary/rope_2d.py").exists()

    importer_files = [
        "worldfoundry/base_models/three_dimensions/point_clouds/hunyuan_mirror/models/layers/__init__.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hunyuan_mirror/models/layers/block.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hunyuan_mirror/models/models/visual_transformer.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/__init__.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/block.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/models/visual_transformer.py",
    ]
    violations = []
    for file_name in importer_files:
        text = (REPO_ROOT / file_name).read_text(encoding="utf-8")
        if "worldfoundry.base_models.diffusion_model.video.common_layers" in text:
            violations.append(f"{file_name}: still imports video common_layers")
        for local_import in [
            "from .drop_path",
            "from .layer_scale",
            "from .patch_embed",
            "from .rope",
            "from .swiglu_ffn",
            "from ..layers.rope",
        ]:
            if local_import in text:
                violations.append(f"{file_name}: uses local shared-layer import {local_import}")

    assert violations == []

    hy_mlp = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/mlp.py"
    hy_mlp_text = hy_mlp.read_text(encoding="utf-8")
    hy_mlp_tree = ast.parse(hy_mlp_text, filename=str(hy_mlp))
    hy_class_names = [node.name for node in ast.walk(hy_mlp_tree) if isinstance(node, ast.ClassDef)]
    assert hy_class_names == ["MlpFP32"]
    assert "from worldfoundry.core.nn import Mlp" in hy_mlp_text


def test_vggt_dinov2_exact_layer_shims_use_canonical_owner():
    layer_paths = [
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/vggt/vggt/layers/drop_path.py",
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/vggt/vggt/layers/mlp.py",
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/vggt_fantasy_world/vggt/layers/drop_path.py",
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/vggt_fantasy_world/vggt/layers/mlp.py",
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/depth/video_depth_anything_longvie/video_depth_anything/dinov2_layers/mlp.py",
    ]
    violations = []

    for path in layer_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        has_local_logic = any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
        )
        if has_local_logic or "worldfoundry.base_models.perception_core.general_perception.dinov2.layers" not in text:
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert violations == []


def test_lingbot_map_base_package_does_not_use_top_level_imports():
    package_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/lingbot_map/lingbot_map"
    violations = []

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "lingbot_map" or alias.name.startswith("lingbot_map."):
                        violations.append(str(path.relative_to(REPO_ROOT)))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "lingbot_map" or node.module.startswith("lingbot_map."):
                    violations.append(str(path.relative_to(REPO_ROOT)))

    assert sorted(set(violations)) == []


def test_vipe_priors_namespace_points_to_split_prior_owners():
    vipe_priors = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d/vipe/priors"
    depth_owner = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth"
    geocalib_owner = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d/geocalib"
    tracking_owner = REPO_ROOT / "worldfoundry/base_models/perception_core/tracking/track_anything"

    vipe_files = sorted(
        path.relative_to(vipe_priors).as_posix()
        for path in vipe_priors.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )

    assert vipe_files == ["__init__.py"]
    for family in ("dap", "dav2", "dav3"):
        assert not (depth_owner / family).exists()
        assert not (depth_owner / "depth_anything" / family).exists()
    for family in ("depth_anything_v1", "depth_anything_v2", "depth_anything_v3"):
        assert (depth_owner / "depth_anything" / family).is_dir()
    assert (depth_owner / "depth_anything/depth_anything_v3/registry.py").is_file()
    assert (depth_owner / "unidepth").is_dir()
    assert geocalib_owner.is_dir()
    assert (tracking_owner / "aot").is_dir()
    vipe_priors_text = (vipe_priors / "__init__.py").read_text(encoding="utf-8")
    assert "three_dimensions.depth" in vipe_priors_text
    assert "three_dimensions.general_3d.geocalib" in vipe_priors_text
    assert "perception_core.tracking.track_anything" in vipe_priors_text


def test_geometric_priors_are_independent_model_catalog_entries():
    import yaml

    from worldfoundry.evals.models.runners.resolver import resolve_model_zoo_config
    from worldfoundry.zoo.catalog.schema import load_entries
    from worldfoundry.zoo.runtime import load_runtime_profile_manifest

    catalog_root = REPO_ROOT / "worldfoundry/data/models/catalog/three_d_four_d"
    runtime_root = REPO_ROOT / "worldfoundry/data/models/runtime/profiles"
    expected_targets = {
        "dap": "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1",
        "depth-anything-v2-prior": "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v2",
        "depth-anything-v3-prior": "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3",
        "geocalib-prior": "worldfoundry.base_models.three_dimensions.general_3d.geocalib",
        "metric3d-prior": "worldfoundry.base_models.three_dimensions.depth.metric3d",
        "prior-depth-anything": "worldfoundry.base_models.three_dimensions.depth.priorda",
        "track-anything-prior": "worldfoundry.base_models.perception_core.tracking.track_anything",
        "unidepth-v2-prior": "worldfoundry.base_models.three_dimensions.depth.unidepth",
        "video-depth-anything-prior": "worldfoundry.base_models.three_dimensions.depth.videodepthanything",
    }

    for model_id, base_model_target in expected_targets.items():
        catalog_path = catalog_root / f"{model_id}.yaml"
        runtime_path = runtime_root / f"{model_id}.yaml"
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        entries = load_entries(catalog_path)
        runtime_profile = load_runtime_profile_manifest(runtime_path)

        assert len(entries) == 1
        assert entries[0].model_id == model_id
        assert entries[0].integration_status == "integrated"
        assert entries[0].runner_entry_kind == "runnable_runner"
        assert entries[0].is_runnable_runner_entry is True
        assert entries[0].runner_target == "worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner"
        resolved = resolve_model_zoo_config(model_id, manifest_dir=catalog_root.parent)
        assert str(resolved.diagnostics["pipeline_target"]).startswith("worldfoundry.pipelines.geometry_priors.")
        assert payload["integration"]["kind"] == "pipeline_runner"
        assert payload["base_model_target"] == base_model_target
        assert not payload["base_model_target"].startswith("worldfoundry.base_models.three_dimensions.general_3d.vipe")
        assert runtime_profile.model_id == model_id
        assert runtime_profile.integration_status == "integrated"
        assert runtime_profile.backend_stage == "in_tree_geometry_prior_runtime"
        assert "prior" in runtime_profile.groups


def test_three_d_four_d_runtimes_run_vendored_package_modules(monkeypatch, tmp_path: Path) -> None:
    import importlib.util

    from worldfoundry.base_models.three_dimensions.three_d_four_d import runtime as runtime_module

    runtime_text = (REPO_ROOT / "worldfoundry/base_models/three_dimensions/three_d_four_d/runtime.py").read_text(
        encoding="utf-8"
    )
    assert "github_repos" not in runtime_text
    assert "PYTHONPATH" not in runtime_text

    for model_id, spec in runtime_module.THREE_D_FOUR_D_RUNTIME_SPECS.items():
        module = runtime_module.runtime_module(spec)
        found = importlib.util.find_spec(module)
        assert module.startswith("worldfoundry."), model_id
        assert found is not None and found.origin is not None, model_id
        assert Path(found.origin) == runtime_module.runtime_dir(spec) / spec.entrypoint

    synthesis = runtime_module.ThreeDFourDRuntimeSynthesis(
        spec=runtime_module.three_d_four_d_runtime_spec("worldgen"), device="cpu"
    )
    plan = synthesis.predict(prompt="a quiet harbor", run_dir=tmp_path / "run", plan_only=True)

    assert plan["command"][1:3] == [
        "-m",
        "worldfoundry.synthesis.visual_generation.worldgen.worldgen_runtime.inference",
    ]
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert "PYTHONPATH" not in synthesis._subprocess_env()


def test_giga_brain_variant_selector_resolves_0p1_checkpoint(tmp_path: Path) -> None:
    from worldfoundry.synthesis.action_generation.giga_brain_0.runtime import (
        select_giga_brain_0_paths,
    )

    base_dir = tmp_path / "open-gigaai--GigaBrain-0-3.5B-Base"
    base_0p1_dir = tmp_path / "open-gigaai--GigaBrain-0.1-3.5B-Base"
    norm_stats = tmp_path / "norm_stats.json"
    base_dir.mkdir()
    base_0p1_dir.mkdir()
    norm_stats.write_text('{"norm_stats": {}}', encoding="utf-8")
    checkpoints = [
        {
            "role": "giga_brain_0_base_checkpoint",
            "local_dir": str(base_dir),
        },
        {
            "role": "giga_brain_0p1_base_checkpoint",
            "local_dir": str(base_0p1_dir),
        },
    ]

    assert (
        select_giga_brain_0_paths(
            variant_id="giga-brain-0.1-3.5b-base",
            norm_stats_path=norm_stats,
            checkpoints=checkpoints,
        )["model_path"]
        == base_0p1_dir.resolve()
    )
    assert (
        select_giga_brain_0_paths(
            variant_id="giga-brain-0-3.5b-base",
            norm_stats_path=norm_stats,
            checkpoints=checkpoints,
        )["model_path"]
        == base_dir.resolve()
    )


def test_giga_brain_plan_records_compile_policy_override(tmp_path: Path) -> None:
    from worldfoundry.synthesis.action_generation.giga_brain_0 import (
        GigaBrain0Synthesis,
    )

    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text('{"norm_stats": {"observation.state": {}, "action": {}}}', encoding="utf-8")
    model = GigaBrain0Synthesis.from_pretrained(
        {
            "norm_stats_path": str(norm_stats),
            "delta_mask": [True] * 14,
            "original_action_dim": 14,
            "embodiment_id": 0,
            "compile_policy": False,
            "torch_dtype": "bfloat16",
        },
        device="cuda",
    )

    result = model.predict("pick up the block", plan_only=True, run_dir=tmp_path / "run")
    plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))

    assert plan["runtime"]["compile_policy"] is False
    assert plan["runtime"]["torch_dtype"] == "bfloat16"

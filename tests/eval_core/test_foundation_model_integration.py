import argparse
import ast
import importlib
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_worldfm_moge_reuses_base_model_directly():
    removed_runtime = (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/worldfm/moge_official"
    )
    moge_pano = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/worldfm/moge_pano.py"
    representation = (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/worldfm/runtime.py"
    )
    base_modules_path = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth/moge/model/modules.py"
    base_v1_path = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth/moge/model/v1.py"

    assert not removed_runtime.exists()
    assert "worldfoundry.base_models.three_dimensions.depth.moge.model.v2" in moge_pano.read_text(encoding="utf-8")
    assert "moge_official" not in representation.read_text(encoding="utf-8")
    assert "perception_core.general_perception.dinov2" in base_modules_path.read_text(encoding="utf-8")
    assert "perception_core.general_perception.dinov2" in base_v1_path.read_text(encoding="utf-8")


def test_worldfm_moge_model_and_utils_forward_to_canonical_base():
    removed_runtime = (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/worldfm/moge_official"
    )
    assert not removed_runtime.exists()


def test_gen3c_moge_runtime_removed_in_favor_of_base_model():
    removed_runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/gen3c/moge_runtime"
    removed_runner = REPO_ROOT / "worldfoundry/synthesis/visual_generation/gen3c/gen3c_runner.py"
    removed_pipeline_utils = REPO_ROOT / "worldfoundry/pipelines/gen3c/gen3c_utils.py"
    synthesis_facade = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/gen3c/gen3c_synthesis.py"
    )
    base_env = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/runtime_env.py"
    )
    base_runtime = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/worldfoundry_runtime.py"
    )
    direct_import_files = [
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/worldfoundry_runner.py",
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/inference/gen3c_single_image.py",
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/inference/gen3c_persistent.py",
    ]

    assert not removed_runtime.exists()
    assert not removed_runner.exists()
    assert not removed_pipeline_utils.exists()
    assert "class Gen3CRuntime" in base_runtime.read_text(encoding="utf-8")
    assert "subprocess.run" in base_runtime.read_text(encoding="utf-8")
    assert "prepare_gen3c_checkpoint_root" in base_env.read_text(encoding="utf-8")
    assert "build_subprocess_env" in base_env.read_text(encoding="utf-8")
    synthesis_text = synthesis_facade.read_text(encoding="utf-8")
    assert "Gen3CRuntime" in synthesis_text
    assert "subprocess.run" not in synthesis_text
    assert "load_video_frames" not in synthesis_text
    assert "_build_command" not in synthesis_text
    for path in direct_import_files:
        text = path.read_text(encoding="utf-8")
        assert "worldfoundry.base_models.three_dimensions.depth.moge.model.v1" in text
        assert "moge_runtime" not in text


def test_gen3c_packaged_cosmos_root_uses_grouped_base_models_path():
    runtime_env = importlib.import_module(
        "worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c.runtime_env"
    )

    root = runtime_env.packaged_gen3c_cosmos_root()

    assert root == (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c"
    )
    assert root.is_dir()


def test_cosmos_predict1_forks_do_not_package_training_datasets():
    roots = [
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1",
    ]
    pruned_training_paths = [
        "autoregressive/datasets",
        "autoregressive/training",
        "autoregressive/configs/experiment",
        "autoregressive/configs/base/dataloader.py",
        "autoregressive/configs/base/dataset.py",
        "autoregressive/train.py",
        "autoregressive/trainer.py",
        "diffusion/training/callbacks",
        "diffusion/training/config",
        "diffusion/training/datasets",
        "diffusion/training/functional",
        "diffusion/training/models",
        "diffusion/training/utils/checkpointer.py",
        "diffusion/training/utils/fsdp_helper.py",
        "diffusion/training/utils/optim_instantiate.py",
        "diffusion/training/utils/peft/lora_attn_test.py",
        "diffusion/training/train.py",
        "diffusion/training/trainer.py",
        "tokenizer/training",
        "callbacks",
        "checkpointer",
        "autoregressive/callbacks",
        "diffusion/checkpointers",
        "utils/callback.py",
        "utils/callbacks",
        "utils/checkpointer.py",
        "utils/ddp_checkpointer.py",
        "utils/ema.py",
        "utils/env_parsers",
        "utils/fsdp_checkpointer.py",
        "utils/fsdp_optim_fix.py",
        "utils/fused_adam.py",
        "utils/model.py",
        "utils/scheduler.py",
        "utils/trainer.py",
    ]

    for root in roots:
        assert root.is_dir()
        for relative_path in pruned_training_paths:
            assert not (root / relative_path).exists(), relative_path

        model_config = root / "autoregressive/configs/base/model_config.py"
        text = model_config.read_text(encoding="utf-8")
        assert "from cosmos_predict1.autoregressive.training.model import AutoRegressiveTrainingModel" not in text.split(
            "def create_video2world_model", maxsplit=1
        )[0]
        assert "AutoRegressive training construction is not packaged in WorldFoundry" in text

        config_text = (root / "utils/config.py").read_text(encoding="utf-8")
        assert "cosmos_predict1.utils import callback" not in config_text
        assert "callbacks: LazyDict = LazyDict(dict())" in config_text


def test_foundation_model_family_directories_are_grouped():
    video_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video"
    diffusion_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model"

    wan_variants = {
        "wan_2p1",
        "wan_2p2",
        "wan_dreamzero",
        "wan_fantasy_world",
        "wan_inspatio_world",
        "wan_matrix_game_2",
        "wan_matrix_game_3",
        "wan_sana",
    }
    cosmos1_variants = {
        "cosmos_predict1_gen3c",
    }
    cosmos2_runtime_variants = {
        "cosmos_models_genie",
        "cosmos_oss_dreamdojo",
        "cosmos_pipeline_utils_genie",
        "cosmos_predict2",
        "cosmos_predict2_wow",
    }
    retired_cosmos_variants = {
        "cosmos_predict1_lyra1",
        "cosmos_predict2_dreamdojo",
        "cosmos_predict1_wow",
    }
    old_diffsynth_variants = {
        "diffsynth_scope",
        "diffsynth_neoverse",
        "diffsynth_longvie",
        "diffsynth_fantasy_world_wan21",
        "diffsynth_fantasy_world_wan22",
    }
    removed_diffsynth_variant_dirs = {
        "scope",
        "neoverse",
        "longvie",
        "fantasy_world_wan21",
        "fantasy_world_wan22",
    }

    assert (video_root / "wan").is_dir()
    assert (video_root / "cosmos").is_dir()
    assert (diffusion_root / "diffsynth").is_dir()
    assert not [name for name in wan_variants if (video_root / name).exists()]
    assert not [name for name in cosmos1_variants | cosmos2_runtime_variants if (video_root / name).exists()]
    assert not [name for name in old_diffsynth_variants if (diffusion_root / name).exists()]
    assert not [name for name in removed_diffsynth_variant_dirs if (diffusion_root / "diffsynth" / name).exists()]
    assert wan_variants <= {path.name for path in (video_root / "wan").iterdir() if path.is_dir()}
    grouped_cosmos_dirs = {path.name for path in (video_root / "cosmos").iterdir() if path.is_dir()}
    assert {"cosmos1", "cosmos2", "shared"} <= grouped_cosmos_dirs
    assert cosmos1_variants <= {
        path.name for path in (video_root / "cosmos/cosmos1").iterdir() if path.is_dir()
    }
    assert cosmos2_runtime_variants <= {
        path.name for path in (video_root / "cosmos/cosmos2/runtime").iterdir() if path.is_dir()
    }
    assert not [
        path
        for path in (video_root / "cosmos").rglob("*")
        if path.is_dir() and path.name in retired_cosmos_variants
    ]
    assert not (diffusion_root / "diffsynth/trainers").exists()
    assert not (video_root / "cosmos/cosmos2/runtime/cosmos_oss_dreamdojo/cosmos_oss/scripts").exists()
    assert not (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/pi3/pi3/utils/debug.py"
    ).exists()


def test_empty_wan_simple_wow_training_source_is_not_packaged():
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_simple_wow/wan_simple"
    )

    assert not runtime_root.parent.exists()
    assert not (runtime_root / ".github/workflows/publish.yaml").exists()
    assert not (runtime_root / "train_wan_i2v.py").exists()
    assert not (runtime_root / "train_wan_i2v_1.3b.py").exists()
    assert not (runtime_root / "train_wan_i2v_data.py").exists()


def test_cosmos_shared_utility_uses_direct_imports_and_tree_has_no_symlinks():
    cosmos_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/cosmos"
    violations = []

    retired_predict1_roots = [
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos_predict1_lyra1",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos_predict1_wow",
    ]
    retired_predict1_shims = [
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/functional/batch_ops.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/training/modules/edm_sde.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/auxiliary/guardrail/face_blur_filter/blur_utils.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/auxiliary/guardrail/llamaGuard3/categories.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/utils/lazy_config/omegaconf_patch.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/types.py",
    ]
    direct_import_expectations = {
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/training/conditioner.py": (
            "worldfoundry.base_models.diffusion_model.video.cosmos.shared.batch_ops",
            "worldfoundry.core.distributed.context_parallel",
        ),
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/diffusion/model/model_world_interpolator.py": (
            "worldfoundry.base_models.diffusion_model.video.cosmos.shared.batch_ops",
            "worldfoundry.base_models.diffusion_model.video.cosmos.shared.edm_sde",
            "worldfoundry.base_models.diffusion_model.video.cosmos.shared.diffusion_types",
        ),
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/auxiliary/guardrail/face_blur_filter/face_blur_filter.py": (
            "worldfoundry.base_models.diffusion_model.video.cosmos.shared.blur_utils",
        ),
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/auxiliary/guardrail/llamaGuard3/llamaGuard3.py": (
            "worldfoundry.base_models.diffusion_model.video.cosmos.shared.llamaguard_categories",
        ),
    }
    retired_easy_io_paths = [
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1/utils/easy_io",
        "worldfoundry/base_models/diffusion_model/video/cosmos/shared/easy_io.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/shared/easy_io_backend_registry_utils.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/shared/easy_io_base_backend.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/shared/easy_io_file_client.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/shared/easy_io_handler_base.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/shared/easy_io_registry_utils.py",
    ]
    violations.extend(path for path in retired_easy_io_paths if (REPO_ROOT / path).exists())
    violations.extend(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/cosmos/shared").glob("easy_io*.py")
    )
    violations.extend(str(path.relative_to(REPO_ROOT)) for path in cosmos_root.rglob("*") if path.is_symlink())

    for relative_path in retired_predict1_roots + retired_predict1_shims:
        path = REPO_ROOT / relative_path
        if path.exists():
            violations.append(relative_path)

    for relative_path, expected_imports in direct_import_expectations.items():
        path = REPO_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        missing = [expected for expected in expected_imports if expected not in text]
        if missing:
            violations.append(f"{relative_path} missing {missing}")

    assert violations == []


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


def _moge_forwarder_files(root: Path) -> list[Path]:
    return [
        *sorted((root / "model").glob("*.py")),
        *sorted((root / "utils").glob("*.py")),
    ]


def _local_logic_violations(paths: list[Path]) -> list[str]:
    violations = []

    for path in paths:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        has_local_logic = any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )
        if has_local_logic or "worldfoundry.base_models.three_dimensions.depth.moge" not in text:
            violations.append(str(path.relative_to(REPO_ROOT)))

    return violations


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
        "worldfoundry/base_models/three_dimensions/general_3d/splatt3r/splatt3r_runtime/src/mast3r_src",
        "worldfoundry/base_models/three_dimensions/general_3d/splatt3r/splatt3r_runtime/src/pixelsplat_src",
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


def test_matrix_game_wanx_vae_modules_are_reexported_from_base_model():
    expected_imports = {
        "attention.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_matrix_game_2.wan.modules.attention",
        "clip.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_matrix_game_2.wan.modules.clip",
        "tokenizers.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.tokenizers",
        "vae.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_matrix_game_2.wan.modules.vae",
        "xlm_roberta.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.xlm_roberta",
    }
    root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/wanx_vae"

    for filename, expected_import in expected_imports.items():
        text = (root / filename).read_text(encoding="utf-8")
        tree = ast.parse(text)

        assert expected_import in text
        assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))

    base_clip = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_matrix_game_2/wan/modules/clip.py"
    ).read_text(encoding="utf-8")
    base_vae = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_matrix_game_2/wan/modules/vae.py"
    ).read_text(encoding="utf-8")
    assert "class CLIPModel(ModelMixin)" in base_clip
    assert "class WanVAE(nn.Module)" in base_vae


def test_sana_wan2_2_vae_is_reexported_from_base_model():
    synthesis_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/sana/sana_runtime/diffusion/model/wan2_2/vae.py"
    )
    base_path = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/vae2_2.py"

    synthesis_text = synthesis_path.read_text(encoding="utf-8")
    base_text = base_path.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_2" in synthesis_text
    assert "class CausalConv3d" not in synthesis_text
    assert "class WanVAE_" not in synthesis_text
    assert "videos should be a list or a tensor with shape [B, C, T, H, W]" in base_text
    assert "zs should be a list or a tensor with shape [B, C, T, H, W]" in base_text


def test_lingbot_wan2_1_vae_lives_under_base_model_without_runtime_facade():
    synthesis_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/lingbot/lingbot_world_runtime/modules/vae2_1.py"
    )
    base_path = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/vae2_1.py"
    helper_path = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/_torch_compat.py"

    base_text = base_path.read_text(encoding="utf-8")

    assert not synthesis_path.exists()
    assert not helper_path.exists()
    assert "class Wan2_1_VAE" in base_text
    assert "load_torch_state_dict(pretrained_path, map_location=device)" in base_text
    assert "from worldfoundry.core.model_loading import load_torch_state_dict" in base_text
    assert "torch.cuda.amp" not in base_text
    assert 'torch.amp.autocast("cuda", dtype=self.dtype)' in base_text

    base_module = importlib.import_module(
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_1"
    )
    assert hasattr(base_module, "Wan2_1_VAE")


def test_lingbot_wan2_2_vae_lives_under_base_model_without_runtime_facade():
    synthesis_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/lingbot/lingbot_world_runtime/modules/vae2_2.py"
    )
    base_path = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/vae2_2.py"

    base_text = base_path.read_text(encoding="utf-8")

    assert not synthesis_path.exists()
    assert "class Wan2_2_VAE" in base_text
    assert "load_torch_state_dict(pretrained_path, map_location=device)" in base_text
    assert "from worldfoundry.core.model_loading import load_torch_state_dict" in base_text
    assert "torch.cuda.amp" not in base_text
    assert 'torch.amp.autocast("cuda", dtype=self.dtype)' in base_text
    assert "videos should be a list or a tensor with shape [B, C, T, H, W]" in base_text

    base_module = importlib.import_module(
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_2"
    )
    assert hasattr(base_module, "Wan2_2_VAE")


def test_lingbot_wan_leaf_modules_are_reexported_from_base_model():
    expected_imports = {
        "distributed/fsdp.py": "worldfoundry.core.distributed.block_fsdp",
        "distributed/sequence_parallel.py": "worldfoundry.core.attention.causal_rope_sequence_parallel",
        "distributed/ulysses.py": "worldfoundry.core.attention.causal_ulysses_attention",
        "distributed/util.py": "worldfoundry.core.distributed.sequence_ops",
        "modules/attention.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_attention"
        ),
        "modules/model.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_model",
        "modules/model_fast.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_model_fast"
        ),
        "modules/t5.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5",
        "modules/animate/animate_utils.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.animate.animate_utils"
        ),
        "modules/animate/face_blocks.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.animate.face_blocks"
        ),
        "modules/animate/clip.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.animate.lingbot_clip"
        ),
        "modules/animate/model_animate.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.animate.lingbot_model_animate"
        ),
        "modules/animate/motion_encoder.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.animate.motion_encoder"
        ),
        "modules/animate/xlm_roberta.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.xlm_roberta"
        ),
        "modules/s2v/audio_encoder.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.s2v.audio_encoder"
        ),
        "modules/s2v/audio_utils.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.s2v.lingbot_audio_utils"
        ),
        "modules/s2v/model_s2v.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.s2v.lingbot_model_s2v"
        ),
        "modules/s2v/motioner.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.s2v.lingbot_motioner"
        ),
        "modules/s2v/auxi_blocks.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.s2v.auxi_blocks"
        ),
        "modules/s2v/s2v_utils.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.s2v.s2v_utils"
        ),
    }
    root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lingbot/lingbot_world_runtime"
    shared_wan21_imports = {
        "modules/tokenizers.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.tokenizers",
        "utils/fm_solvers.py": "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.fm_solvers",
        "utils/fm_solvers_unipc.py": (
            "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.fm_solvers_unipc"
        ),
    }
    removed_runtime_helpers = [
        root / "distributed/__init__.py",
        root / "modules/__init__.py",
        root / "modules/animate/__init__.py",
        root / "modules/s2v/__init__.py",
        root / "utils/torch_compat.py",
    ]

    assert [path for path in removed_runtime_helpers if path.exists()] == []

    for relative_path, expected_import in expected_imports.items():
        assert not (root / relative_path).exists()
        assert (REPO_ROOT / f"{expected_import.replace('.', '/')}.py").is_file()

    for relative_path, expected_import in shared_wan21_imports.items():
        assert not (root / relative_path).exists()
        assert (REPO_ROOT / f"{expected_import.replace('.', '/')}.py").is_file()

    base_lingbot_model = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/lingbot_model.py"
    ).read_text(encoding="utf-8")
    base_lingbot_model_fast = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/lingbot_model_fast.py"
    ).read_text(encoding="utf-8")
    base_lingbot_audio_utils = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/s2v/lingbot_audio_utils.py"
    ).read_text(encoding="utf-8")
    base_lingbot_animate = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/animate/lingbot_model_animate.py"
    ).read_text(encoding="utf-8")
    base_lingbot_clip = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/animate/lingbot_clip.py"
    ).read_text(encoding="utf-8")
    base_lingbot_s2v = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/s2v/lingbot_model_s2v.py"
    ).read_text(encoding="utf-8")
    base_lingbot_motioner = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/s2v/lingbot_motioner.py"
    ).read_text(encoding="utf-8")
    base_lingbot_sequence_parallel = (
        REPO_ROOT / "worldfoundry/core/distributed/causal_rope_sequence_parallel.py"
    ).read_text(encoding="utf-8")
    base_lingbot_ulysses = (
        REPO_ROOT / "worldfoundry/core/distributed/causal_ulysses_attention.py"
    ).read_text(encoding="utf-8")
    base_wan21_t5 = (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p1/modules/t5.py"
    ).read_text(encoding="utf-8")
    base_wan22_modules_init = (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/__init__.py"
    ).read_text(encoding="utf-8")
    base_wan21_unipc = (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p1/utils/fm_solvers_unipc.py"
    ).read_text(encoding="utf-8")
    base_s2v_init = (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/s2v/__init__.py"
    ).read_text(encoding="utf-8")

    assert "class WanModel(ModelMixin, ConfigMixin)" in base_lingbot_model
    assert "dit_cond_dict" in base_lingbot_model
    assert "patch_embedding_wancamctrl" in base_lingbot_model
    assert "from .lingbot_attention import flash_attention" in base_lingbot_model
    assert "class WanModelFast(ModelMixin, ConfigMixin)" in base_lingbot_model_fast
    assert "from .lingbot_model import" in base_lingbot_model_fast
    assert "from .lingbot_attention import attention" in base_lingbot_model_fast
    assert "from ..lingbot_model import WanAttentionBlock, WanCrossAttention" in base_lingbot_audio_utils
    assert "torch.cuda.amp" not in base_lingbot_audio_utils
    assert "from ..lingbot_model import" in base_lingbot_animate
    assert "from ..lingbot_attention import flash_attention" in base_lingbot_clip
    assert "from ....wan_2p1.modules.tokenizers import HuggingfaceTokenizer" in base_lingbot_clip
    assert "from worldfoundry.core.model_loading import load_torch_state_dict" in base_lingbot_clip
    assert "from ..lingbot_model import (" in base_lingbot_s2v
    assert "from .lingbot_audio_utils import AudioInjector_WAN, CausalAudioEncoder" in base_lingbot_s2v
    assert "from .lingbot_motioner import FramePackMotioner, MotionerTransformers" in base_lingbot_s2v
    assert "from ..lingbot_model import flash_attention" in base_lingbot_motioner
    assert (
        "from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_attention import flash_attention"
        in base_lingbot_sequence_parallel
    )
    assert (
        "from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_model import sinusoidal_embedding_1d"
        in base_lingbot_sequence_parallel
    )
    assert "from worldfoundry.core.attention.causal_ulysses_attention import distributed_attention" in base_lingbot_sequence_parallel
    assert (
        "from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.lingbot_attention import flash_attention"
        in base_lingbot_ulysses
    )
    for base_text in (base_lingbot_animate, base_lingbot_clip, base_lingbot_s2v, base_lingbot_motioner):
        assert "torch.cuda.amp" not in base_text
    assert "load_torch_state_dict(checkpoint_path, map_location='cpu')" in base_wan21_t5
    assert "from worldfoundry.core.model_loading import load_torch_state_dict" in base_wan21_t5
    assert "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.t5" in base_wan22_modules_init
    assert "use_kerras_sigma: bool = False" in base_wan21_unipc
    assert "if isinstance(timesteps, list):" in base_wan21_unipc
    assert "timesteps = [timesteps]" in base_wan21_unipc
    assert "def __getattr__" in base_s2v_init
    assert "from .audio_encoder import" not in base_s2v_init
    assert "from .model_s2v import" not in base_s2v_init


def test_pixelsplat_runtime_logic_lives_under_base_models():
    base_runtime = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/pixelsplat_runtime/worldfoundry_runtime.py"
    )
    base_runtime_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/pixelsplat_runtime"
    data_asset_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/pixelsplat/assets"
    annotation_path = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/pixelsplat_full/src/visualization/annotation.py"
    )
    synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/pixelsplat/pixelsplat_synthesis.py"

    assert base_runtime.is_file()
    assert not (base_runtime_root / "assets").exists()
    assert {
        "Inter-Regular.otf",
        "evaluation_index_acid.json",
        "evaluation_index_acid_video.json",
        "evaluation_index_re10k.json",
        "evaluation_index_re10k_video.json",
    } == {path.name for path in data_asset_root.iterdir()}
    base_text = base_runtime.read_text(encoding="utf-8")
    annotation_text = annotation_path.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class PixelSplatRuntime" in base_text
    assert "subprocess.run" in base_text
    assert "checkpointing.load" in base_text
    assert "worldfoundry_data_path" in base_text
    assert "DEFAULT_PIXELSPLAT_ASSET_ROOT" in base_text
    assert "WORLDFOUNDRY_PIXELSPLAT_ASSET_ROOT" in base_text
    assert "assets/evaluation_index" not in base_text
    assert "DEFAULT_FONT_PATH = worldfoundry_data_path" in annotation_text
    assert "assets/Inter-Regular.otf" not in annotation_text
    assert "PixelSplatRuntime" in synthesis_text
    assert "subprocess.run" not in synthesis_text


def test_pixelsplat_and_splatt3r_aliases_are_registered_on_demand():
    import sys

    alias_names = ("src", "data", "utils", "workspace")
    saved_aliases = {name: sys.modules.get(name) for name in alias_names}
    for name in alias_names:
        sys.modules.pop(name, None)

    module_prefixes = (
        "worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat_runtime",
        "worldfoundry.base_models.three_dimensions.general_3d.splatt3r.splatt3r_runtime",
    )
    for name in list(sys.modules):
        if name.startswith(module_prefixes):
            sys.modules.pop(name, None)

    try:
        pixelsplat_runtime = importlib.import_module(
            "worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat_runtime"
        )
        splatt3r_runtime = importlib.import_module(
            "worldfoundry.base_models.three_dimensions.general_3d.splatt3r.splatt3r_runtime"
        )

        assert not any(name in sys.modules for name in alias_names)

        pixelsplat_runtime.ensure_pixelsplat_source_alias()
        assert "src" in sys.modules
        assert not any(name in sys.modules for name in ("data", "utils", "workspace"))

        sys.modules.pop("src", None)
        splatt3r_runtime.ensure_runtime_aliases()
        assert "src" in sys.modules
        assert not any(name in sys.modules for name in ("data", "utils", "workspace"))
    finally:
        for name in alias_names:
            sys.modules.pop(name, None)
            if saved_aliases[name] is not None:
                sys.modules[name] = saved_aliases[name]


def test_wan_synthesis_wrappers_live_outside_base_models():
    base_wan_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan"
    old_synthesis_wan_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/wan"
    pipeline_wan_root = REPO_ROOT / "worldfoundry/pipelines/wan"

    assert (base_wan_root / "wan_runtime_wrapper.py").is_file()
    assert not list(base_wan_root.glob("*synthesis.py"))
    assert not old_synthesis_wan_root.exists()
    assert {
        "wan_2p1_i2v_synthesis.py",
        "wan_2p1_t2v_synthesis.py",
        "wan2p2_synthesis.py",
        "wan_2p5_synthesis.py",
        "wan_2p6_synthesis.py",
        "wan_2p7_synthesis.py",
    } <= {path.name for path in pipeline_wan_root.glob("*synthesis.py")}

    for pipeline_path in pipeline_wan_root.glob("pipeline_wan_*.py"):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "base_models.diffusion_model.video.wan." not in text, pipeline_path
        assert "synthesis.visual_generation.wan" not in text, pipeline_path


def test_wan_prompt_defaults_are_user_editable_yaml():
    config_path = REPO_ROOT / "worldfoundry/data/models/runtime/configs/wan/prompt_defaults.yaml"
    old_test_case_yaml_path = REPO_ROOT / "worldfoundry/data/test_cases/wan_prompt_defaults.yaml"
    old_json_path = REPO_ROOT / "worldfoundry/data/test_cases/wan_prompt_defaults.json"
    wrapper_path = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_runtime_wrapper.py"

    assert config_path.is_file()
    assert not old_test_case_yaml_path.exists()
    assert not old_json_path.exists()
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    assert 'DATA_ROOT / "models" / "runtime" / "configs" / "wan" / "prompt_defaults.yaml"' in wrapper_text
    assert 'DATA_ROOT / "test_cases"' in wrapper_text
    assert "test_cases\" / \"wan_prompt_defaults" not in wrapper_text
    assert "wan_prompt_defaults.json" not in wrapper_text


def test_dust3r_is_independent_general_3d_integration_with_mast3r_reexport():
    general_3d_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d"
    canonical = general_3d_root / "dust3r"
    mast3r_root = general_3d_root / "mast3r"
    legacy = mast3r_root / "dust3r"
    catalog = REPO_ROOT / "worldfoundry/data/models/catalog/three_d_four_d/dust3r.yaml"

    assert (canonical / "dust3r/model.py").is_file()
    assert (canonical / "croco/models/croco.py").is_file()
    assert (canonical / "__init__.py").is_file()
    assert not legacy.exists()

    mast3r_init = (mast3r_root / "__init__.py").read_text(encoding="utf-8")
    path_bridge = (mast3r_root / "mast3r/utils/path_to_dust3r.py").read_text(encoding="utf-8")

    assert 'GENERAL_3D_ROOT = SOURCE_ROOT.parent' in mast3r_init
    assert 'DUST3R_ROOT = GENERAL_3D_ROOT / "dust3r"' in mast3r_init
    assert "worldfoundry.base_models.three_dimensions.general_3d.dust3r" in mast3r_init
    assert "../../../dust3r" in path_bridge

    catalog_text = catalog.read_text(encoding="utf-8")
    assert "id: dust3r" in catalog_text
    assert "base_model_target: worldfoundry.base_models.three_dimensions.general_3d.dust3r" in catalog_text
    assert "general_3d/mast3r/dust3r" not in catalog_text


def test_mast3r_import_path_prioritizes_canonical_dust3r():
    import sys

    original_sys_path = list(sys.path)
    module_names = [
        name
        for name in sys.modules
        if name == "dust3r"
        or name.startswith("dust3r.")
        or name == "mast3r"
        or name.startswith("mast3r.")
        or name.startswith("worldfoundry.base_models.three_dimensions.general_3d.mast3r")
    ]
    for name in module_names:
        sys.modules.pop(name, None)

    try:
        mast3r = importlib.import_module("worldfoundry.base_models.three_dimensions.general_3d.mast3r")
        mast3r.ensure_import_paths()
        canonical_dust3r = importlib.import_module(
            "worldfoundry.base_models.three_dimensions.general_3d.dust3r"
        )
        reexported_dust3r = importlib.import_module(
            "worldfoundry.base_models.three_dimensions.general_3d.mast3r.dust3r"
        )
        dust3r = importlib.import_module("dust3r")
        inner_mast3r = importlib.import_module("mast3r")
        dust3r_file = Path(dust3r.__file__).resolve()

        assert mast3r.dust3r is canonical_dust3r
        assert inner_mast3r.dust3r is canonical_dust3r
        assert reexported_dust3r is canonical_dust3r
        assert sys.path.index(str(mast3r.DUST3R_ROOT)) < sys.path.index(str(mast3r.SOURCE_ROOT))
        assert dust3r_file.is_relative_to(mast3r.DUST3R_ROOT)
        assert not dust3r_file.is_relative_to(mast3r.SOURCE_ROOT / "dust3r")
    finally:
        sys.path[:] = original_sys_path
        for name in [
            "dust3r",
            "mast3r",
            "mast3r.dust3r",
            "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
            "worldfoundry.base_models.three_dimensions.general_3d.mast3r",
            "worldfoundry.base_models.three_dimensions.general_3d.mast3r.dust3r",
        ]:
            sys.modules.pop(name, None)


def test_canonical_dust3r_initializes_its_own_import_paths():
    import sys

    original_sys_path = list(sys.path)
    module_names = [
        name
        for name in sys.modules
        if name == "dust3r"
        or name.startswith("dust3r.")
        or name.startswith("worldfoundry.base_models.three_dimensions.general_3d.dust3r")
    ]
    for name in module_names:
        sys.modules.pop(name, None)

    try:
        canonical_dust3r = importlib.import_module(
            "worldfoundry.base_models.three_dimensions.general_3d.dust3r"
        )
        dust3r = importlib.import_module("dust3r")
        dust3r_file = Path(dust3r.__file__).resolve()

        assert str(canonical_dust3r.SOURCE_ROOT) in sys.path
        assert str(canonical_dust3r.CROCO_ROOT) in sys.path
        assert dust3r_file.is_relative_to(canonical_dust3r.DUST3R_PACKAGE_ROOT)
    finally:
        sys.path[:] = original_sys_path
        for name in [
            "dust3r",
            "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
        ]:
            sys.modules.pop(name, None)


def test_dust3r_mast3r_integrations_prune_training_and_dataset_preprocess_code():
    general_3d_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d"
    dust3r_root = general_3d_root / "dust3r"
    mast3r_root = general_3d_root / "mast3r"
    removed_paths = [
        dust3r_root / "train.py",
        dust3r_root / "dust3r/training.py",
        dust3r_root / "datasets_preprocess",
        dust3r_root / "docker",
        dust3r_root / "assets",
        dust3r_root / "demo",
        dust3r_root / "croco/assets",
        dust3r_root / "croco/pretrain.py",
        dust3r_root / "croco/datasets",
        dust3r_root / "croco/stereoflow",
        dust3r_root / "croco/models/criterion.py",
        dust3r_root / "dust3r_visloc",
        dust3r_root / "dust3r/losses.py",
        dust3r_root / "dust3r/demo.py",
        dust3r_root / "demo.py",
        dust3r_root / "visloc.py",
        dust3r_root / "dust3r/datasets",
        dust3r_root / "dust3r/datasets/base",
        dust3r_root / "dust3r/datasets/arkitscenes.py",
        dust3r_root / "dust3r/datasets/blendedmvs.py",
        dust3r_root / "dust3r/datasets/co3d.py",
        dust3r_root / "dust3r/datasets/habitat.py",
        dust3r_root / "dust3r/datasets/megadepth.py",
        dust3r_root / "dust3r/datasets/scannetpp.py",
        dust3r_root / "dust3r/datasets/staticthings3d.py",
        dust3r_root / "dust3r/datasets/waymo.py",
        dust3r_root / "dust3r/datasets/wildrgbd.py",
        mast3r_root / "train.py",
        mast3r_root / "assets",
        mast3r_root / "demo",
        mast3r_root / "mast3r/datasets",
        mast3r_root / "mast3r/losses.py",
        mast3r_root / "demo.py",
        mast3r_root / "demo_dust3r_ga.py",
        mast3r_root / "visloc.py",
    ]

    for path in removed_paths:
        assert not path.exists(), path

    assert (dust3r_root / "dust3r/utils/transforms.py").is_file()
    assert (dust3r_root / "dust3r/utils/cropping.py").is_file()


def test_dust3r_mast3r_inference_helpers_still_import_after_pruning():
    import sys

    original_sys_path = list(sys.path)
    for name in [
        module_name
        for module_name in sys.modules
        if module_name == "dust3r"
        or module_name.startswith("dust3r.")
        or module_name == "mast3r"
        or module_name.startswith("mast3r.")
        or module_name.startswith("worldfoundry.base_models.three_dimensions.general_3d.dust3r")
        or module_name.startswith("worldfoundry.base_models.three_dimensions.general_3d.mast3r")
    ]:
        sys.modules.pop(name, None)

    try:
        importlib.import_module("worldfoundry.base_models.three_dimensions.general_3d.mast3r")
        dust3r_transforms = importlib.import_module("dust3r.utils.transforms")
        dust3r_cropping = importlib.import_module("dust3r.utils.cropping")

        assert hasattr(dust3r_transforms, "ImgNorm")
        assert hasattr(dust3r_cropping, "crop_image_depthmap")
    finally:
        sys.path[:] = original_sys_path
        for name in [
            "dust3r",
            "mast3r",
            "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
            "worldfoundry.base_models.three_dimensions.general_3d.mast3r",
        ]:
            sys.modules.pop(name, None)


def test_base_model_pure_demo_and_ui_app_dirs_are_pruned():
    base_models_root = REPO_ROOT / "worldfoundry/base_models"
    world_model_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/world_model"
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
        path.relative_to(REPO_ROOT).as_posix()
        for path in base_models_root.rglob("*")
        if "demo" in path.name.lower()
    )

    assert demo_paths == []

    mineworld_runtime = (
        world_model_root / "mineworld/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")
    assert "WEB_DEMO_ENTRYPOINT" not in mineworld_runtime
    mineworld_lvm = (
        world_model_root / "mineworld/lvm.py"
    ).read_text(encoding="utf-8")
    mineworld_decoding = (
        world_model_root / "mineworld/diagonal_decoding.py"
    ).read_text(encoding="utf-8")
    assert "for_gradio" not in mineworld_lvm
    assert "for_gradio" not in mineworld_decoding
    assert "gradio" not in mineworld_lvm.lower()
    assert "gradio" not in mineworld_decoding.lower()
    mineworld_action_tokenizer = (
        world_model_root / "mineworld/action_tokenizer.py"
    ).read_text(encoding="utf-8")
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
    mineworld_infer_script = (
        world_model_root / "mineworld/scripts/inference_16f_models.sh"
    ).read_text(encoding="utf-8")
    assert "data/models/runtime/configs/mineworld" in mineworld_infer_script
    assert 'CONFIG="configs/' not in mineworld_infer_script

    fastblend_api = (
        base_models_root / "diffusion_model/diffsynth/extensions/FastBlend/api.py"
    ).read_text(encoding="utf-8")
    assert "import gradio" not in fastblend_api
    assert "def on_ui_tabs" not in fastblend_api

    fastblend_init = (
        base_models_root / "diffusion_model/diffsynth/extensions/FastBlend/__init__.py"
    ).read_text(encoding="utf-8")
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
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/slam/droid_slam/geom/graph_utils.py"
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
        base_models / "three_dimensions/depth/dvlt/dvlt_runtime/src/dvlt/callbacks",
        base_models / "three_dimensions/slam/droid_slam/data_readers",
        base_models / "three_dimensions/slam/droid_slam/geom/losses.py",
        base_models / "three_dimensions/depth/moge/utils/data_augmentation.py",
    ]

    for path in removed_paths:
        assert not path.exists(), path

    assert (
        base_models / "perception_core/detection/grounding_dino/util/transforms.py"
    ).is_file()


def test_world_model_runtime_configs_live_under_data_models():
    base_models_root = REPO_ROOT / "worldfoundry/base_models"
    world_model_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/world_model"
    data_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs"

    assert not (base_models_root / "diffusion_model/world_model").exists()
    assert world_model_root.is_dir()
    assert not (
        world_model_root / "adaworld/worldmodel/configs"
    ).exists()
    assert not (world_model_root / "adaworld/lam/config").exists()
    assert not (world_model_root / "adaworld/worldmodel/vwm/data").exists()

    adaworld_root = data_config_root / "adaworld"
    assert (adaworld_root / "lam/lam.yaml").is_file()
    assert {
        "adaworld.yaml",
        "adaworld_adapt_continuous_action.yaml",
        "adaworld_adapt_discrete_action.yaml",
    } == {path.name for path in (adaworld_root / "worldmodel/inference").glob("*.yaml")}
    assert not (adaworld_root / "worldmodel/training").exists()

    sample_text = (
        world_model_root / "adaworld/worldmodel/sample.py"
    ).read_text(encoding="utf-8")
    assert "models\", \"runtime\", \"configs\", \"adaworld" in sample_text
    assert "resolve_data_path" in sample_text
    assert 'CONFIG = "configs/' not in sample_text
    for script in (
        world_model_root / "adaworld/worldmodel/run_train.sh",
        world_model_root / "adaworld/worldmodel/run_adaptation_continuous.sh",
        world_model_root / "adaworld/worldmodel/run_adaptation_discrete.sh",
        world_model_root / "adaworld/lam/train.sh",
        world_model_root / "adaworld/lam/test.sh",
    ):
        assert not script.exists()

    for non_infer_entrypoint in (
        world_model_root / "adaworld/download_miradata_360p.py",
        world_model_root / "adaworld/download_open_x.sh",
        world_model_root / "adaworld/worldmodel/train.py",
        world_model_root / "adaworld/worldmodel/train_adapt.py",
    ):
        assert not non_infer_entrypoint.exists()

    assert not (world_model_root / "nwm/config").exists()
    assert not (world_model_root / "nwm/environment.yml").exists()

    nwm_root = data_config_root / "nwm"
    assert {
        "data_config.yaml",
        "data_hyperparams_plan.yaml",
        "environment.yml",
        "eval_config.yaml",
        "nwm_cdit_xl.yaml",
        "wm_debug_bs_32.yaml",
    } == {path.name for path in nwm_root.iterdir() if path.is_file()}

    nwm_python_files = (
        "config_paths.py",
        "isolated_nwm_infer.py",
        "misc.py",
        "eval_inputs.py",
    )
    for name in nwm_python_files:
        text = (
            world_model_root / "nwm" / name
        ).read_text(encoding="utf-8")
        assert "runtime/configs/nwm" in text or "load_runtime_yaml" in text or "runtime_config_path" in text
        assert 'open("config/' not in text
        assert "open('config/" not in text

    nwm_eval_inputs = (world_model_root / "nwm/eval_inputs.py").read_text(encoding="utf-8")
    assert "class TrainingDataset" not in nwm_eval_inputs
    assert "class TrajectoryEvalDataset" not in nwm_eval_inputs
    assert not (world_model_root / "nwm/datasets.py").exists()

    for non_infer_entrypoint in (
        world_model_root / "nwm/train.py",
        world_model_root / "nwm/submitit_train_cw.py",
        world_model_root / "nwm/interactive_model.ipynb",
    ):
        assert not non_infer_entrypoint.exists()

    for model_id in ("diamond", "dino_wm", "le_wm", "starwm"):
        runtime_config_files = list(
            (world_model_root / model_id).rglob("*.yaml")
        ) + list((world_model_root / model_id).rglob("*.json"))
        assert runtime_config_files == []
        assert (data_config_root / model_id).is_dir()

    assert not (world_model_root / "diamond/src").exists()
    assert not (world_model_root / "dino_wm/datasets").exists()
    assert not (world_model_root / "dino_wm/planning").exists()
    assert not (world_model_root / "dino_wm/plan.py").exists()

    vid2world_runtime_config_root = world_model_root / "vid2world/vid2world_runtime/configs"
    vid2world_config_root = data_config_root / "vid2world"
    assert not vid2world_runtime_config_root.exists()
    assert not (world_model_root / "vid2world/UPSTREAM.md").exists()
    for config_file in vid2world_config_root.rglob("*.yaml"):
        config_text = config_file.read_text(encoding="utf-8")
        assert "lvdm.data" not in config_text
        assert "target: lvdm." not in config_text
        assert "target: utils_data." not in config_text
        assert "worldfoundry.base_models.diffusion_model.video.lvdm.variants.vid2world.eval_inputs" in config_text
        assert (
            "worldfoundry.synthesis.visual_generation.world_model.vid2world.vid2world_runtime.main.utils_data"
            in config_text
        )
    assert {
        "ablation",
        "game",
        "manipulation",
        "navigation",
    } == {path.name for path in vid2world_config_root.iterdir() if path.is_dir()}


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
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/echo_infinity/echo_infinity_runtime/utils/memory.py",
    ]

    assert hasattr(core_vram, "DynamicSwapInstaller")
    assert hasattr(core_vram, "get_cuda_free_memory_gb")
    assert not [path for path in removed_shims if path.exists()]


def test_splatt3r_dust3r_only_modules_use_canonical_dust3r_import_paths():
    splatt3r_root = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d/splatt3r"
    profile = (
        REPO_ROOT
        / "worldfoundry/data/models/runtime/profiles/splatt3r.yaml"
    )
    removed_dust3r_shims = [
        splatt3r_root / "splatt3r_runtime/utils/export.py",
    ]

    assert not [path for path in removed_dust3r_shims if path.exists()]

    profile_text = profile.read_text(encoding="utf-8")
    assert "canonical sibling base_models paths" in profile_text
    assert "vendors the Splatt3R MASt3R, DUSt3R" not in profile_text


def test_splatt3r_runtime_is_inference_only_after_pruning():
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/general_3d/splatt3r/splatt3r_runtime"
    )
    removed_paths = [
        runtime_root / "data",
        runtime_root / "workspace.py",
        runtime_root / "demo.py",
        runtime_root / "main.py",
        runtime_root / "utils/compute_ssim.py",
        runtime_root / "utils/export.py",
        runtime_root / "utils/loss_mask.py",
        runtime_root / "utils/sh_utils.py",
    ]

    for path in removed_paths:
        assert not path.exists(), path

    model_text = (runtime_root / "model.py").read_text(encoding="utf-8")
    init_text = (runtime_root / "__init__.py").read_text(encoding="utf-8")

    assert "class MAST3RGaussians" in model_text
    assert "def forward" in model_text
    for token in (
        "training_step",
        "validation_step",
        "test_step",
        "configure_optimizers",
        "run_experiment",
        "DataLoader",
        "Trainer(",
        "wandb",
        "lpips",
        "scannetpp",
    ):
        assert token not in model_text

    assert not (runtime_root / "main.py").exists()
    assert "from . import data" not in init_text
    assert "from . import workspace" not in init_text


def test_splatt3r_runtime_logic_lives_under_base_models():
    base_runtime = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/general_3d/splatt3r/worldfoundry_runtime.py"
    )
    synthesis = REPO_ROOT / "worldfoundry/base_models/three_dimensions/general_3d/splatt3r/splatt3r_synthesis.py"

    assert base_runtime.is_file()
    base_text = base_runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class Splatt3RRuntime" in base_text
    assert "MAST3RGaussians.load_from_checkpoint" in base_text
    assert "splatt3r_runtime.model import" in base_text
    assert "splatt3r_runtime.main import" not in base_text
    assert "hf_hub_download" in base_text
    assert "Splatt3RRuntime" in synthesis_text
    assert "hf_hub_download" not in synthesis_text
    assert "MAST3RGaussians" not in synthesis_text
    assert "DEFAULT_SHARED_HFD_ROOT" not in synthesis_text
    assert "_ensure_model" not in synthesis_text


def test_dreamdojo_runtime_logic_lives_under_synthesis():
    runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/dreamdojo/worldfoundry_runtime.py"
    )
    synthesis = REPO_ROOT / "worldfoundry/synthesis/action_generation/dreamdojo/dreamdojo_synthesis.py"

    assert runtime.is_file()
    base_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class DreamDojoRuntime" in base_text
    assert "subprocess.run" in base_text
    assert "ActionConditionedInferenceArguments" in base_text
    assert "DreamDojoRuntime" in synthesis_text
    assert "subprocess.run" not in synthesis_text
    assert "DEFAULT_SHARED_HFD_ROOT" not in synthesis_text
    assert "ActionConditionedInferenceArguments" not in synthesis_text
    assert "_checkpoint_path" not in synthesis_text


def test_lyra1_runtime_logic_lives_under_synthesis():
    runtime = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_1/worldfoundry_runtime.py"
    )
    synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_1/synthesis.py"

    assert runtime.is_file()
    base_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class Lyra1Runtime" in base_text
    assert "subprocess.run" in base_text
    assert "_build_command" in base_text
    assert "Lyra1Runtime" in synthesis_text
    assert "subprocess.run" not in synthesis_text
    assert "os.environ" not in synthesis_text
    assert "_build_command" not in synthesis_text
    assert "materialize_image_input" not in synthesis_text


def test_lyra1_cosmos_root_uses_grouped_base_models_path():
    runtime_module = importlib.import_module("worldfoundry.synthesis.visual_generation.lyra_1.worldfoundry_runtime")

    root = runtime_module.Lyra1Runtime._cosmos_root()

    assert root == (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos1/cosmos_predict1_gen3c/cosmos_predict1"
    )
    assert root.is_dir()


def test_lyra1_pipeline_can_plan_without_external_representation_assets():
    from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline

    pipe = Lyra1Pipeline.from_pretrained({"model_id": "lyra-1"}, device="cpu")

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
    assert "worldfoundry_data_path" in sample_text
    assert "runtime\", \"configs\", \"lyra_1\", \"inference" in sample_text
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

    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/loger/pi3.py"
    ).is_file()


def test_infinite_vggt_representation_uses_base_model_source():
    wrapper = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/infinite_vggt/runtime.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.infinite_vggt" in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/vggt/infinite_vggt"
    ).exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/infinite_vggt/models/streamvggt.py"
    ).is_file()


def test_cut3r_representation_uses_base_model_source():
    wrapper = REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/cut3r/runtime.py"
    text = wrapper.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.cut3r" in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/cut3r/cut3r"
    ).exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/cut3r/model.py"
    ).is_file()


def test_hyworldmirror_2p0_runtime_uses_base_model_source():
    runtime = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/runtime.py"
    )
    text = runtime.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0" in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/hunyuan_world/hy_world_2p0/hyworldmirror"
    ).exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/models/worldmirror.py"
    ).is_file()


def test_lingbot_map_representation_uses_base_model_runtime():
    wrapper = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/lingbot_map/runtime.py"
    )
    text = wrapper.read_text(encoding="utf-8")

    assert "Path(__file__).resolve().parent" in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/lingbot_map/lingbot_map_runtime"
    ).exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/lingbot_map/lingbot_map/models/gct_stream.py"
    ).is_file()


def test_flash_world_representation_uses_base_model_source():
    wrapper = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/flash_world/runtime.py"
    )
    text = wrapper.read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.point_clouds.flash_world" in text
    assert not (
        REPO_ROOT / "worldfoundry/representations/point_clouds_generation/flash_world/flash_world"
    ).exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/flash_world/reconstruction_model.py"
    ).is_file()


def test_diffsynth_forks_are_integrated_into_base_tree():
    old_paths = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/diffsynth",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/neoverse/neoverse_runtime/diffsynth",
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/longvie",
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/neoverse",
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/scope",
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/fantasy_world_wan21",
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/fantasy_world_wan22",
    ]
    assert [path for path in old_paths if path.exists()] == []
    expected_files = [
        "pipelines/wan_video_new_longvie.py",
        "pipelines/wan_video_neoverse.py",
        "pipelines/scope_pipeline.py",
        "pipelines/fantasy_world_wan21_wan_video.py",
        "pipelines/fantasy_world_wan22_wan_video.py",
        "models/neoverse_depth_anything_reconstructor.py",
        "models/neoverse_rasterization.py",
        "models/scope_dit.py",
        "models/longvie_wan_video_dit.py",
        "models/fantasy_world_wan21_wan_video_dit.py",
        "models/fantasy_world_wan22_wan_video_dit.py",
    ]
    base = REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth"
    assert [path for path in expected_files if not (base / path).is_file()] == []


def test_diffsynth_variant_runtimes_do_not_shadow_global_diffsynth_namespace():
    import_files = {
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/longvie/worldfoundry_runtime.py": (
            "worldfoundry.base_models.diffusion_model.diffsynth.pipelines.wan_video_new_longvie"
        ),
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/longvie/longvie_runtime/inference.py": (
            "worldfoundry.base_models.diffusion_model.diffsynth.pipelines.wan_video_new_longvie"
        ),
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/neoverse/worldfoundry_runtime.py": (
            "worldfoundry.base_models.diffusion_model.diffsynth.pipelines.wan_video_neoverse"
        ),
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/scope/scope_runtime/inference.py": (
            "worldfoundry.base_models.diffusion_model.diffsynth.pipelines.scope_pipeline"
        ),
    }
    for path, expected_prefix in import_files.items():
        text = path.read_text(encoding="utf-8")
        assert expected_prefix in text
        assert "from diffsynth" not in text
        assert "import diffsynth" not in text

    for path in [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/longvie/runtime_env.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/neoverse/runtime_env.py",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "diffsynth_runtime_root" not in text
        assert 'sys.modules["diffsynth"]' not in text

    fantasy_env = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/fantasy_world/runtime_env.py"
    ).read_text(encoding="utf-8")
    assert "diffsynth_wan21_root" not in fantasy_env
    assert "diffsynth_wan22_root" not in fantasy_env
    assert "worldfoundry.base_models.diffusion_model.diffsynth.diffsynth_fantasy_world" not in fantasy_env

    scope_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/scope/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")
    assert "pythonpath_parts = [str(runtime_root()), str(_repo_src_root())]" in scope_runtime
    assert "str(diffsynth_runtime_root())" not in scope_runtime
    assert "worldfoundry.base_models.diffusion_model.diffsynth.scope" not in scope_runtime
    assert not (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/_overlay.py"
    ).exists()


def test_neoverse_runtime_logic_lives_under_synthesis():
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/neoverse"
    base_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/neoverse"

    assert not (synthesis_root / "_runtime_env.py").exists()
    assert (base_root / "runtime_env.py").is_file()
    assert (base_root / "worldfoundry_runtime.py").is_file()

    synthesis_text = (synthesis_root / "neoverse_synthesis.py").read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.neoverse.worldfoundry_runtime" in synthesis_text
    assert "._runtime_env" not in synthesis_text
    assert "from diffsynth" not in synthesis_text
    assert "resolve_neoverse_" not in synthesis_text
    assert "class NeoVerseOfficialRuntime" not in synthesis_text

    package_text = (base_root / "__init__.py").read_text(encoding="utf-8")
    assert "NeoVerseOfficialRuntime" in package_text


def test_core_inference_official_demo_fixtures_are_in_tree():
    source = (REPO_ROOT / "worldfoundry/core/inference.py").read_text(encoding="utf-8")

    assert 'official_runtime_repo_path("Matrix-Game")' not in source
    assert 'official_runtime_repo_path("Astra")' not in source
    assert 'official_runtime_repo_path("NeoVerse")' not in source
    assert "MATRIX_GAME_2_ROOT =" not in source
    assert "MATRIX_GAME_3_ROOT =" not in source
    assert "ASTRA_ROOT =" not in source
    assert "NEOVERSE_ROOT =" not in source

    assert 'MATRIX_GAME_2_IN_TREE_ROOT = _TEST_CASES_ROOT / "matrix-game-2"' in source
    assert 'MATRIX_GAME_2_IN_TREE_ROOT / "universal" / "0000.png"' in source
    assert 'MATRIX_GAME_2_IN_TREE_ROOT / "configs" / "inference_universal.yaml"' in source
    assert (
        'MATRIX_GAME_3_FALLBACK_FIXTURE = _TEST_CASES_ROOT / "matrix-game-3" / "001" / "image.png"'
        in source
    )
    assert 'ASTRA_FALLBACK_FIXTURE = _TEST_CASES_ROOT / "astra" / "condition_images" / "garden_1.png"' in source
    assert 'NEOVERSE_DATA_ROOT / "videos" / "robot.mp4"' in source
    assert "packaged in-tree Matrix-Game fixtures" in source
    assert "packaged in-tree image fixture" in source
    assert "packaged in-tree garden image fixture" in source


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


def test_dynamicrafter_runtime_logic_lives_under_synthesis():
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/dynamicrafter"
    base_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/dynamicrafter"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/dynamicrafter"
    pandora_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/dynamicrafter_pandora"
    pandora_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/dynamicrafter_pandora"

    assert not (synthesis_root / "dynamicrafter_runtime_wrapper.py").exists()
    assert (synthesis_root / "dynamicrafter_runtime").is_dir()
    assert (base_root / "worldfoundry_runtime.py").is_file()
    assert not (base_root / "dynamicrafter_runtime/configs").exists()
    assert (config_root / "inference_256_v1.0.yaml").is_file()
    assert (config_root / "inference_512_v1.0.yaml").is_file()
    assert (config_root / "inference_1024_v1.0.yaml").is_file()
    assert not (pandora_root / "DynamiCrafter/configs").exists()
    assert (pandora_config_root / "inference_256_v1.0.yaml").is_file()
    assert (pandora_config_root / "inference_512_v1.0.yaml").is_file()
    assert (pandora_config_root / "inference_1024_v1.0.yaml").is_file()
    assert "DYNAMICRAFTER_PANDORA_CONFIG_ROOT" in (pandora_root / "__init__.py").read_text(encoding="utf-8")

    runtime_text = (base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "class DynamiCrafter" in runtime_text
    assert "def load_model_checkpoint" in runtime_text
    assert "def load_data_images" in runtime_text
    assert "def image_guided_synthesis" in runtime_text
    assert "worldfoundry_data_path" in runtime_text

    for file_name in [
        "dynamicrafter_512_i2v_synthesis.py",
        "dynamicrafter_1024_i2v_synthesis.py",
    ]:
        text = (synthesis_root / file_name).read_text(encoding="utf-8")
        assert "worldfoundry.synthesis.visual_generation.dynamicrafter" in text
        assert "runtime/configs/dynamicrafter" in text
        assert "dynamicrafter_runtime/configs" not in text
        assert "dynamicrafter_runtime_wrapper" not in text
        assert "load_model_checkpoint" not in text
        assert "load_data_images" not in text
        assert "image_guided_synthesis" not in text


def test_vmem_runtime_env_lives_under_synthesis():
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/vmem"
    base_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/vmem"

    assert not (synthesis_root / "_runtime_env.py").exists()
    assert (base_root / "runtime_env.py").is_file()
    assert (base_root / "worldfoundry_runtime.py").is_file()

    runtime_text = (base_root / "runtime_env.py").read_text(encoding="utf-8")
    worldfoundry_runtime_text = (base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "def canonical_cut3r_parent" in runtime_text
    assert '"extern" / "CUT3R"' in runtime_text
    assert "def ensure_vmem_runtime" in runtime_text
    assert "github_repos" not in runtime_text
    assert "WORLDFOUNDRY_MODEL_SOURCE_DIR" not in runtime_text
    assert "WORLDFOUNDRY_GITHUB_REPOS" not in runtime_text
    assert "sys.path" in runtime_text
    assert "class VMemRuntime" in worldfoundry_runtime_text
    assert "RuntimeVMemPipeline" in worldfoundry_runtime_text
    assert "scipy.spatial.transform" in worldfoundry_runtime_text

    for path in [synthesis_root / "__init__.py", synthesis_root / "vmem_synthesis.py"]:
        text = path.read_text(encoding="utf-8")
        assert "._runtime_env" not in text
        assert "RuntimeVMemPipeline" not in text
        assert "scipy.spatial.transform" not in text
        assert "import sys" not in text
        assert "sys.path" not in text
        assert "sys.modules" not in text
    assert "worldfoundry.synthesis.visual_generation.vmem" in (
        synthesis_root / "vmem_synthesis.py"
    ).read_text(encoding="utf-8")


def test_vmem_runtime_root_ignores_external_source_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from worldfoundry.synthesis.visual_generation.vmem.runtime_env import (
        DEFAULT_VMEM_RUNTIME_ROOT,
        runtime_root,
    )

    monkeypatch.delenv("VMEM_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_DIR", str(tmp_path / "github_repos"))

    assert runtime_root() == DEFAULT_VMEM_RUNTIME_ROOT.resolve()


def test_multiworld_solaris_runtime_envs_live_under_synthesis():
    cases = {
        "multiworld": (
            "worldfoundry.synthesis.visual_generation.multiworld.ittakestwo_runtime",
            "multiworld_ittakestwo_synthesis.py",
            "ittakestwo_runtime.py",
            ["class MultiWorldItTakesTwoRuntime", "def dump_tree", "def main"],
        ),
        "solaris": (
            "worldfoundry.synthesis.visual_generation.solaris.worldfoundry_runtime",
            "solaris_synthesis.py",
            "worldfoundry_runtime.py",
            ["class SolarisRuntime", "def _build_inference_command", "subprocess.run"],
        ),
    }

    for (
        package,
        (canonical_import, synthesis_file_name, base_runtime_file, base_runtime_markers),
    ) in cases.items():
        synthesis_root = REPO_ROOT / f"worldfoundry/synthesis/visual_generation/{package}"
        base_root = synthesis_root

        assert not (synthesis_root / "_runtime_env.py").exists()
        assert (base_root / "runtime_env.py").is_file()
        if base_runtime_file is not None:
            assert (base_root / base_runtime_file).is_file()
            base_runtime_text = (base_root / base_runtime_file).read_text(encoding="utf-8")
            for marker in base_runtime_markers:
                assert marker in base_runtime_text

        runtime_text = (base_root / "runtime_env.py").read_text(encoding="utf-8")
        assert "def resolve_runtime_root" in runtime_text
        assert "def project_root" in runtime_text

        synthesis_text = (synthesis_root / synthesis_file_name).read_text(encoding="utf-8")
        assert canonical_import in synthesis_text
        assert "._runtime_env" not in synthesis_text
        assert "._serialization" not in synthesis_text
        assert "subprocess.run" not in synthesis_text
        assert "def _build_inference_command" not in synthesis_text
        assert "def resolve_runtime_root" not in synthesis_text
        assert "def local_runtime_candidates" not in synthesis_text
        assert "def build_subprocess_env" not in synthesis_text
        assert "def build_inference_env" not in synthesis_text
    assert not (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/multiworld/_serialization.py"
    ).exists()
    assert not (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/multiworld/multiworld_ittakestwo_runner.py"
    ).exists()


def test_allegro_easyanimate_runtime_logic_lives_under_synthesis():
    cases = {
        "allegro": (
            "allegro_runtime_wrapper.py",
            "allegro_ti2v_synthesis.py",
            "worldfoundry.synthesis.visual_generation.allegro.worldfoundry_runtime",
            ["class Allegro", "def preprocess_images", "load_allegro_components"],
        ),
        "easyanimate": (
            "easyanimate_runtime_wrapper.py",
            "easyanimate_i2v_synthesis.py",
            "worldfoundry.synthesis.visual_generation.easyanimate.worldfoundry_runtime",
            ["class EasyAnimate", "def resolve_config_path", "load_easyanimate_components"],
        ),
    }

    for package, (removed_wrapper, synthesis_file, canonical_import, runtime_needles) in cases.items():
        synthesis_root = REPO_ROOT / f"worldfoundry/synthesis/visual_generation/{package}"
        base_root = synthesis_root

        assert not (synthesis_root / removed_wrapper).exists()
        assert (base_root / "worldfoundry_runtime.py").is_file()
        if package == "easyanimate":
            config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/easyanimate"
            assert not (base_root / "easyanimate_runtime/config").exists()
            assert (
                config_root / "easyanimate_video_v5_magvit_multi_text_encoder.yaml"
            ).exists() is False

        runtime_text = (base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
        for needle in runtime_needles:
            assert needle in runtime_text
        assert "from diffusers" in runtime_text
        assert "from_pretrained" in runtime_text

        synthesis_text = (synthesis_root / synthesis_file).read_text(encoding="utf-8")
        assert canonical_import in synthesis_text
        if package == "easyanimate":
            assert "runtime/configs/easyanimate" in synthesis_text
            assert "easyanimate_runtime/config" not in synthesis_text
        assert "runtime_wrapper" not in synthesis_text
        assert "load_allegro_components" not in synthesis_text
        assert "load_easyanimate_components" not in synthesis_text
        assert "from diffusers" not in synthesis_text
        assert "from_pretrained" not in synthesis_text


def test_videocrafter_runtime_lives_under_synthesis_and_vchitect_under_base_models():
    videocrafter_synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/videocrafter"
    videocrafter_base_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/videocrafter"
    assert not (videocrafter_synthesis_root / "videocrafter_runtime_wrapper.py").exists()
    assert not (videocrafter_synthesis_root / "videocrafter_runtime").exists()
    assert not videocrafter_base_root.exists()
    assert (videocrafter_synthesis_root / "worldfoundry_runtime.py").is_file()
    assert (videocrafter_synthesis_root / "videocrafter_inference.py").is_file()

    videocrafter_runtime_text = (videocrafter_synthesis_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "class VideoCrafter" in videocrafter_runtime_text
    assert "OmegaConf.load" in videocrafter_runtime_text
    assert "load_videocrafter_components" in videocrafter_runtime_text

    for file_name in [
        "videocrafter1_i2v_synthesis.py",
        "videocrafter1_t2v_synthesis.py",
        "videocrafter2_t2v_synthesis.py",
    ]:
        text = (videocrafter_synthesis_root / file_name).read_text(encoding="utf-8")
        assert "from .worldfoundry_runtime import VideoCrafter" in text
        assert "worldfoundry.base_models.diffusion_model.video.videocrafter" not in text
        assert "videocrafter_runtime_wrapper" not in text
        assert "OmegaConf" not in text
        assert "batch_ddim_sampling" not in text
        assert "load_videocrafter_components" not in text

    vchitect_synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/vchitect"
    vchitect_base_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/vchitect"
    assert not (vchitect_synthesis_root / "vchitect_runtime_wrapper.py").exists()
    assert (vchitect_base_root / "worldfoundry_runtime.py").is_file()

    vchitect_runtime_text = (vchitect_base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "class Vchitect" in vchitect_runtime_text
    assert "class VchitectRuntimePlan" not in vchitect_runtime_text
    assert "subprocess.run" not in vchitect_runtime_text
    assert "def execute_public_runner" not in vchitect_runtime_text

    vchitect_synthesis_text = (vchitect_synthesis_root / "vchitect_2_t2v_synthesis.py").read_text(encoding="utf-8")
    assert "worldfoundry.base_models.diffusion_model.video.vchitect.worldfoundry_runtime" in vchitect_synthesis_text
    assert "vchitect_runtime_wrapper" not in vchitect_synthesis_text
    assert "subprocess" not in vchitect_synthesis_text
    assert "VchitectRuntimePlan" not in vchitect_synthesis_text
    assert "build_plan" not in vchitect_synthesis_text
    assert "run_plan" not in vchitect_synthesis_text
    assert "json.dumps" not in vchitect_synthesis_text


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
    assert (
        REPO_ROOT
        / "worldfoundry/data/models/runtime/configs/cosmos_predict2/t5xxl/t5encoder.json"
    ).is_file()
    assert {
        "wow-dit-2b.txt",
        "wow-dit-7b.txt",
    } == {
        path.name
        for path in (REPO_ROOT / "worldfoundry/data/models/runtime/configs/wow/requirements").glob("*.txt")
    }
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


def test_giga_world_0_example_payload_lives_in_test_cases():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/giga_world_0/giga_world_0_runtime"
    manifest = REPO_ROOT / "worldfoundry/data/test_cases/giga_world_0/it2v.json"
    runtime_package_init = runtime_root / "giga_world_0/__init__.py"

    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data
    assert all({"prompt", "image"} <= set(item) for item in data)
    assert all(item["image"].startswith("images/") for item in data)

    runtime_text = runtime_package_init.read_text(encoding="utf-8")
    assert "GigaWorld0Trainer" not in runtime_text
    assert "GigaWorld0Transform" not in runtime_text


def test_diffusion_policy_checkpoint_workspace_is_inference_only():
    workspace_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/diffusion_policy/diffusion_policy_runtime/workspace/train_diffusion_unet_lowdim_workspace.py"
    )
    base_workspace_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/diffusion_policy/diffusion_policy_runtime/workspace/base_workspace.py"
    )
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/action_generation/diffusion_policy/runtime.py"
    workspace_text = workspace_path.read_text(encoding="utf-8")
    base_workspace_text = base_workspace_path.read_text(encoding="utf-8")
    runtime_text = runtime_path.read_text(encoding="utf-8")

    assert "class TrainDiffusionUnetLowdimWorkspace" in workspace_text
    assert "hydra.utils.instantiate(cfg.policy)" in workspace_text
    assert "def run(self)" in workspace_text
    assert "DataLoader" not in workspace_text
    assert "wandb" not in workspace_text
    assert "BaseLowdimDataset" not in workspace_text
    assert "diffusers.training_utils" not in workspace_text
    assert "Path(__file__).resolve().parent" in runtime_text
    assert "/ \"base_models\"" not in runtime_text
    assert "exclude_keys = tuple(key for key in state_dicts if key not in loadable_policy_keys)" in runtime_text
    assert "def load_payload" in base_workspace_text
    assert "save_checkpoint" not in base_workspace_text
    assert "save_snapshot" not in base_workspace_text
    assert "create_from_snapshot" not in base_workspace_text
    assert "threading" not in base_workspace_text
    assert "HydraConfig" not in base_workspace_text


def test_lyra2_runtime_logic_lives_under_synthesis():
    runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/runtime.py"
    synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/synthesis.py"
    depth_utils = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference/depth_utils.py"
    )
    lyra2_inference_root = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/lyra_2/lyra_2/_src/inference"
    )
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


def test_motionctrl_default_conditions_live_in_data_test_cases():
    synthesis_path = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/synthesis.py"
    )
    base_runtime_path = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/worldfoundry_runtime.py"
    )
    inference_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/main/inference/motionctrl_inference.py"
    )
    condition_root = REPO_ROOT / "worldfoundry/data/test_cases/motionctrl_conditions"

    assert condition_root.is_dir()
    assert sorted(path.suffix for path in (condition_root / "trajectories").glob("*")) == [
        ".py",
        ".txt",
        ".txt",
        ".txt",
        ".txt",
        ".txt",
        ".txt",
        ".txt",
        ".txt",
    ]
    synthesis_text = synthesis_path.read_text(encoding="utf-8")
    base_runtime_text = base_runtime_path.read_text(encoding="utf-8")
    assert "class MotionCtrlRuntime" in base_runtime_text
    assert 'worldfoundry_data_path("test_cases", "motionctrl_conditions")' in base_runtime_text
    assert "MotionCtrlRuntime" in synthesis_text
    for marker in [
        "load_model_checkpoint",
        "motionctrl_sample",
        "imageio.mimsave",
        "hashlib.sha256",
    ]:
        assert marker not in synthesis_text
        assert marker in base_runtime_text
    assert "_trajectory_from_control_points" in inference_path.read_text(encoding="utf-8")


def test_scope_examples_live_in_data_test_cases():
    condition_root = REPO_ROOT / "worldfoundry/data/test_cases/scope_examples"
    inference_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/scope/scope_runtime/inference.py"
    ).read_text(encoding="utf-8")

    assert (condition_root / "example_0/image.png").is_file()
    assert (condition_root / "example_0/action.parquet").is_file()
    assert "worldfoundry/data/test_cases/scope_examples" in inference_text


def test_infinite_world_default_config_lives_in_data_runtime_configs(tmp_path: Path):
    from worldfoundry.synthesis.visual_generation.infinite_world import (
        InfiniteWorldSynthesis,
    )
    from worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime import (
        InfiniteWorldRuntime,
        default_config_path,
    )

    config_path = REPO_ROOT / "worldfoundry/data/models/runtime/configs/infinite_world.yaml"
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/infinite_world/infinite_world_runtime"
    old_hidden_config = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/infinite_world"
        / ("infworld_" + "config.yaml")
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

    cameractrl_runtime_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/cameractrl/runtime.py"
    ).read_text(encoding="utf-8")
    cameractrl_synthesis_text = synthesis_paths[0].read_text(encoding="utf-8")
    assert "worldfoundry_data_path" in cameractrl_runtime_text
    assert "runtime", "configs" in cameractrl_runtime_text
    assert "worldfoundry.synthesis" not in cameractrl_runtime_text
    assert "CameraCtrlRuntime" in cameractrl_synthesis_text
    assert "get_pipeline" not in cameractrl_synthesis_text
    assert "ray_condition" not in cameractrl_synthesis_text
    assert "save_videos_grid" not in cameractrl_synthesis_text
    motionctrl_base_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")
    assert "worldfoundry_data_path" in motionctrl_base_text
    assert "runtime", "configs" in motionctrl_base_text


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
    assert "worldfoundry_data_path" in runtime_text
    assert "runtime", "configs" in runtime_text
    assert "worldfoundry.synthesis" not in runtime_text
    assert "AnimateDiffRuntime" in synthesis_text
    assert "UNet3DConditionModel" not in synthesis_text
    assert "AnimationPipeline" not in synthesis_text
    assert "save_videos_grid" not in synthesis_text
    assert "WebVid10M" not in runtime_text
    assert "torch.utils.data" not in runtime_text
    assert not any(
        path.name == "run_animatediff_exact_parity.py"
        for path in (REPO_ROOT / "scripts").rglob("*.py")
    )


def test_zeroscope_runtime_logic_lives_under_synthesis():
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/zeroscope/zeroscope_synthesis.py"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/zeroscope/worldfoundry_runtime.py"

    assert runtime_path.is_file()
    runtime_text = runtime_path.read_text(encoding="utf-8")
    synthesis_text = synthesis_path.read_text(encoding="utf-8")

    assert "worldfoundry.synthesis" not in runtime_text
    assert "TextToVideoSDPipeline" in runtime_text
    assert "export_to_video" in runtime_text
    assert "frames_sha256" in runtime_text
    assert "ZeroScopeRuntime" in synthesis_text
    assert "TextToVideoSDPipeline" not in synthesis_text
    assert "export_to_video" not in synthesis_text
    assert "hashlib" not in synthesis_text


def test_inspatio_world_configs_live_in_data_runtime_configs():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/inspatio_world"
    traj_root = config_root / "traj"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world"
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/worldfoundry_runtime.py"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_synthesis.py"
    base_init = REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/__init__.py"
    runner_script = runtime_root / "run_inference_pipeline.sh"

    assert (config_root / "default_config.yaml").is_file()
    assert (config_root / "inference_1.3b.yaml").is_file()
    assert (config_root / "environment.yml").is_file()
    assert {
        "x_y_circle_cycle.txt",
        "zoom_out_in.txt",
    } == {path.name for path in traj_root.glob("*.txt")}
    assert not old_runtime_root.exists()
    assert not (runtime_root / "traj").exists()
    assert (runtime_root / "inference_causal.py").is_file()
    assert runtime_path.is_file()

    text = synthesis_path.read_text(encoding="utf-8")
    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert "class InspatioWorldSynthesis" in text
    assert "InspatioWorldRuntime" in text
    assert "package_root(\"worldfoundry.synthesis.visual_generation.inspatio_world\")" in runtime_text

    assert "InspatioWorldSynthesis" in base_init.read_text(encoding="utf-8")
    assert "worldfoundry_data_path" in runtime_text
    assert "runtime\", \"configs\", \"inspatio_world" in runtime_text
    assert "_TRAJECTORY_ROOT = _CONFIG_ROOT / \"traj\"" in runtime_text
    assert "WORLDFOUNDRY_INSPATIO_WORLD_TRAJECTORY_ROOT" in runtime_text
    assert "inspatio_world_runtime" in runtime_text
    assert "subprocess.run" in runtime_text
    assert "OmegaConf" in runtime_text
    assert "from worldfoundry.core.io import" in runtime_text
    assert "worldfoundry.synthesis.visual_generation.inspatio_world" in runtime_text
    assert "configs/inference_1.3b.yaml" not in runtime_text
    assert "subprocess.run" not in text
    assert "OmegaConf" not in text
    assert "materialize_video_input" not in text
    assert "load_video_frames" not in text

    runner_text = runner_script.read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_INSPATIO_WORLD_CONFIG_ROOT" in runner_text
    assert "WORLDFOUNDRY_INSPATIO_WORLD_TRAJECTORY_ROOT" in runner_text
    assert "/configs/inference_1.3b.yaml" not in runner_text
    assert "./traj/x_y_circle_cycle.txt" not in runner_text


def test_worldcam_runtime_lives_under_synthesis():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldcam/worldcam_runtime"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldcam/worldcam"
    base_package_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldcam/__init__.py"
    base_runtime_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldcam/worldfoundry_runtime.py"
    pipeline_path = REPO_ROOT / "worldfoundry/pipelines/worldcam/pipeline_worldcam.py"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/worldcam/worldcam_synthesis.py"

    assert not old_runtime_root.exists()
    assert (runtime_root / "pipelines/wan_video_new.py").is_file()
    assert not (runtime_root / "models/wan_video_camera_controller.py").exists()

    package_text = base_package_path.read_text(encoding="utf-8")
    assert "WorldCamRuntime" in package_text
    assert "DEFAULT_WEIGHT_DTYPE" in package_text
    assert "RUNTIME_ROOT = runtime_root()" in package_text
    assert "\"runtime_root\"" in package_text

    base_text = base_runtime_path.read_text(encoding="utf-8")
    for expected in [
        "def _require_torch",
        "DEFAULT_WEIGHT_DTYPE",
        "ModelConfig",
        "load_state_dict",
        "WanVideoPipeline.from_pretrained",
        "pipe.dit.load_state_dict",
        "def _resolve_checkpoint_path",
        "def _missing_wan_components",
    ]:
        assert expected in base_text

    synthesis_text = synthesis_path.read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.worldcam.worldfoundry_runtime" in synthesis_text
    assert "WorldCamRuntime.from_pretrained" in synthesis_text
    assert "return self.runtime.predict" in synthesis_text
    for forbidden in [
        "from .worldcam.models",
        "from .worldcam.pipelines",
        "import torch",
        "@torch.no_grad()",
        "WanVideoPipeline",
        "ModelConfig",
        "load_state_dict",
        "def _resolve_checkpoint_path",
        "def _missing_wan_components",
        "pipe.dit.load_state_dict",
        "Path(__file__).resolve().parent / \"worldcam\"",
    ]:
        assert forbidden not in synthesis_text

    pipeline_text = pipeline_path.read_text(encoding="utf-8")
    assert "runtime_root as worldcam_runtime_root" in pipeline_text
    assert "worldcam_runtime_root()" in pipeline_text
    assert "synthesis\" / \"visual_generation\" / \"worldcam\"" not in pipeline_text

    pipeline_runtime_text = (runtime_root / "pipelines/wan_video_new.py").read_text(encoding="utf-8")
    worldcam_dit_text = (runtime_root / "models/wan_video_dit.py").read_text(encoding="utf-8")
    assert "worldfoundry.base_models.diffusion_model.diffsynth.models.wan_video_dit_s2v" in pipeline_runtime_text
    assert "worldfoundry.base_models.diffusion_model.diffsynth.models.wan_video_camera_controller" in worldcam_dit_text
    assert "worldcam_runtime.models.wan_video_camera_controller" not in pipeline_runtime_text


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
    assert "Path(__file__).resolve().parent / \"lingbot_world\"" not in text


def test_kling_astra_runtime_lives_under_synthesis():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra_runtime"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/astra_synthesis.py"

    assert not old_runtime_root.exists()
    assert (runtime_root / "pipelines/wan_video_astra.py").is_file()
    assert (runtime_root / "models/model_manager.py").is_file()
    assert (runtime_root / "model_registry.py").is_file()
    assert not (runtime_root / "configs").exists()
    assert (
        REPO_ROOT / "worldfoundry/data/models/runtime/configs/kling/astra/model_loader.yaml"
    ).is_file()

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


def test_hy_world_2p0_panogen_runtime_lives_under_synthesis():
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hy_world_2p0_panogen_runtime"
    )
    old_runtime_root = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hy_world_2p0/panogen"
    )
    synthesis_path = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_family_synthesis.py"
    )
    pipeline_path = REPO_ROOT / "worldfoundry/pipelines/hunyuan_world/pipeline_hy_world_2p0.py"

    assert not old_runtime_root.exists()
    assert (runtime_root / "pipeline.py").is_file()
    assert (runtime_root / "pipeline_with_qwen_image.py").is_file()
    assert (runtime_root / "setup.py").is_file()

    for path in (synthesis_path, pipeline_path):
        text = path.read_text(encoding="utf-8")
        assert "synthesis.visual_generation.hunyuan_world.hy_world_2p0_panogen_runtime" in text
        assert "hy_world_2p0.panogen" not in text


def test_hy_world_2p0_worldgen_diagnostics_do_not_probe_external_checkouts(monkeypatch, tmp_path: Path):
    from worldfoundry.synthesis.visual_generation.hunyuan_world.hy_world_2p0_worldgen_runtime import (
        get_hy_world_2p0_worldgen_plan,
        inspect_official_worldgen_checkout,
    )

    runtime_text = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hy_world_2p0_worldgen_runtime.py"
    ).read_text(encoding="utf-8")
    assert "github_repos" not in runtime_text

    monkeypatch.delenv("WORLDFOUNDRY_HY_WORLD_2P0_REPO", raising=False)
    plan = get_hy_world_2p0_worldgen_plan()
    assert plan["official_checkout"]["repo_path"] is None
    assert plan["official_checkout"]["exists"] is False

    official_root = tmp_path / "HY-World-2.0"
    required_file = official_root / "hyworld2/worldgen/traj_generate.py"
    required_file.parent.mkdir(parents=True)
    required_file.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setenv("WORLDFOUNDRY_HY_WORLD_2P0_REPO", str(official_root))

    checkout = inspect_official_worldgen_checkout()
    assert checkout["repo_path"] == str(official_root)
    assert checkout["exists"] is True
    assert "hyworld2/worldgen/traj_generate.py" not in checkout["missing_files"]


def test_matrix_game_2_runtime_and_configs_live_under_synthesis():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/matrix_game_2"
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2_runtime"
    old_runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_2_synthesis.py"
    base_runtime_path = runtime_root / "worldfoundry_runtime.py"

    assert not old_runtime_root.exists()
    assert base_runtime_path.is_file()
    assert (runtime_root / "pipeline/causal_inference.py").is_file()
    assert (runtime_root / "utils/wan_wrapper.py").is_file()
    assert (runtime_root / "utils/vae_runtime/constant.py").is_file()
    assert (runtime_root / "utils/vae_runtime/vae_block3.py").is_file()
    assert (config_root / "inference_yaml/inference_universal.yaml").is_file()
    assert (config_root / "distilled_model/universal/config.yaml").is_file()

    text = synthesis_path.read_text(encoding="utf-8")
    base_text = base_runtime_path.read_text(encoding="utf-8")
    pipeline_text = (runtime_root / "pipeline/causal_inference.py").read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime" in text
    assert "worldfoundry_data_path" not in text
    assert "demo_utils" not in base_text
    assert "demo_utils" not in pipeline_text
    for marker in [
        "OmegaConf.load",
        "load_file",
        "WanDiffusionWrapper(",
        "CausalInferencePipeline(",
        "VAEDecoderWrapper",
        "torch.load",
    ]:
        assert marker not in text
        assert marker in base_text
    assert "class MatrixGame2Runtime" in base_text
    assert "matrix_game_2/configs" not in text


def test_wan_training_runtime_helpers_are_not_demo_utils():
    training_root = REPO_ROOT / "worldfoundry/training/visual_generation/wan"
    runtime_utils = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/vae_runtime"
    init_text = (training_root / "__init__.py").read_text(encoding="utf-8")
    pipeline_text = (training_root / "pipelines/causal_inference.py").read_text(encoding="utf-8")
    wrapper_text = (training_root / "utils/wan_wrapper.py").read_text(encoding="utf-8")
    vae_text = (runtime_utils / "vae.py").read_text(encoding="utf-8")
    inspatio_text = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime/inference_causal.py"
    ).read_text(encoding="utf-8")

    assert not (training_root / "demo_utils").exists()
    assert not (training_root / "utils/vae_runtime").exists()
    assert (runtime_utils / "constant.py").is_file()
    assert (runtime_utils / "vae.py").is_file()
    assert (runtime_utils / "vae_block3.py").is_file()
    assert "demo_utils" not in init_text
    assert "demo_utils" not in pipeline_text
    assert "demo_utils" not in wrapper_text
    assert "demo_utils" not in vae_text
    assert "wan.modules" not in wrapper_text
    assert "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules" in wrapper_text
    assert "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.vae" in vae_text
    assert "demo_utils" not in inspatio_text
    assert "worldfoundry.core.vram" in pipeline_text
    assert "worldfoundry.core.vram" in inspatio_text


def test_matrix_game_3_runner_lives_with_base_runtime():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3_runtime"
    data_asset_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/matrix_game_3/assets"
    old_runner = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3_runner.py"
    synthesis_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3_synthesis.py"
    base_runtime_path = runtime_root / "worldfoundry_runtime.py"

    assert not old_runner.exists()
    assert not (runtime_root / "assets").exists()
    assert (data_asset_root / "images/mouse.png").is_file()
    assert (runtime_root / "__init__.py").is_file()
    assert (runtime_root / "worldfoundry_runner.py").is_file()
    assert base_runtime_path.is_file()
    synthesis_text = synthesis_path.read_text(encoding="utf-8")
    base_text = base_runtime_path.read_text(encoding="utf-8")
    assert "class MatrixGame3Runtime" in base_text
    assert "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime.worldfoundry_runner" in base_text
    assert "WORLDFOUNDRY_MATRIX_GAME3_ASSET_ROOT" in base_text
    assert "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runner" not in synthesis_text
    for marker in [
        "subprocess.run",
        "imageio.get_reader",
        "json.dump",
        "sys.executable",
        "_ensure_checkpoint_layout",
    ]:
        assert marker not in synthesis_text
        assert marker in base_text


def test_matrix_game_1_runtime_logic_lives_under_base_models():
    runtime_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_runtime/runtime.py"
    )
    synthesis_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_1_synthesis.py"
    )

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
    runtime = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/open_magvit2/worldfoundry_runtime.py"
    )
    synthesis = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/open_magvit2/open_magvit2_synthesis.py"
    )

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
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/open_magvit2/open_magvit2_runtime"
    )
    runtime_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/open_magvit2/worldfoundry_runtime.py"
    )
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/open_magvit2"

    assert not (runtime_root / "configs").exists()
    assert {
        "imagenet_conditional_llama_B.yaml",
        "imagenet_conditional_llama_L.yaml",
        "imagenet_conditional_llama_XL.yaml",
    } == {path.name for path in config_root.glob("*.yaml")}

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert "worldfoundry_data_path" in runtime_text
    assert "runtime\", \"configs\", \"open_magvit2" in runtime_text
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


def test_show_o_demo_configs_live_in_data_runtime_configs():
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/show_o/show_o_runtime"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/show_o"
    prompt_root = config_root / "validation_prompts"

    assert not (runtime_root / "configs").exists()
    assert not (runtime_root / "training").exists()
    assert not (runtime_root / "validation_prompts").exists()
    assert not (runtime_root / "models/training_utils.py").exists()
    assert not (runtime_root / "README.md").exists()
    assert not (runtime_root / "CONTRIBUTING_ROADMAP.md").exists()
    assert (runtime_root / "inference_support/prompting_utils.py").is_file()
    assert (runtime_root / "inference_support/runtime_utils.py").is_file()
    assert {
        "showo_demo.yaml",
        "showo_demo_512x512.yaml",
        "showo_demo_w_clip_vit.yaml",
        "showo_demo_w_clip_vit_512x512.yaml",
    } == {path.name for path in config_root.glob("*.yaml")}
    assert {
        "imagenet_prompts.txt",
        "showoprompts.txt",
        "text2image_prompts.txt",
    } == {path.name for path in prompt_root.glob("*.txt")}
    assert not (runtime_root / "training/questions.json").exists()

    package_text = (runtime_root / "__init__.py").read_text(encoding="utf-8")
    worldfoundry_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/show_o/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")
    t2i_text = (runtime_root / "inference_t2i.py").read_text(encoding="utf-8")
    mmu_text = (runtime_root / "inference_mmu.py").read_text(encoding="utf-8")
    runtime_utils_text = (runtime_root / "inference_support/runtime_utils.py").read_text(encoding="utf-8")

    assert '"training"' not in package_text
    assert "from training" not in worldfoundry_runtime
    assert "from training" not in t2i_text
    assert "from training" not in mmu_text
    assert "training_utils" not in worldfoundry_runtime
    assert "soft_target_cross_entropy" not in runtime_utils_text
    assert "mask_or_random_replace_tokens" not in runtime_utils_text
    assert "AverageMeter" not in runtime_utils_text


def test_cosmos_predict2p5_runtime_lives_under_base_models():
    runtime_path = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/cosmos2p5/worldfoundry_predict_runtime.py"
    )
    synthesis_path = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/cosmos/cosmos_predict2p5_synthesis.py"
    )

    assert runtime_path.is_file()
    runtime_text = runtime_path.read_text(encoding="utf-8")
    synthesis_text = synthesis_path.read_text(encoding="utf-8")

    assert "class CosmosPredict2p5Runtime" in runtime_text
    assert "class CosmosPredict2p5Synthesis(BaseSynthesis)" in synthesis_text
    assert "def _runtime_cls()" in synthesis_text
    assert "base_models.diffusion_model.video.cosmos2p5.worldfoundry_predict_runtime" in synthesis_text
    assert len(synthesis_text.splitlines()) <= 50

    heavy_runtime_markers = [
        "Cosmos25Transformer3DModel",
        "Reason1TextEncoder",
        "WanVAE",
        "FlowUniPCMultistepScheduler",
        "VideoProcessor",
        "randn_tensor",
        "tqdm",
    ]
    assert [marker for marker in heavy_runtime_markers if marker in synthesis_text] == []


def test_fantasy_world_runners_live_under_synthesis():
    base_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/fantasy_world"
    synthesis_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/fantasy_world"
    old_runtime_files = [
        synthesis_root / "_runtime_env.py",
        synthesis_root / "fantasy_world_utils.py",
        synthesis_root / "fantasy_world_wan21_runner.py",
        synthesis_root / "fantasy_world_wan22_runner.py",
    ]

    assert [path.name for path in old_runtime_files if path.exists()] == []
    assert (base_root / "runtime_env.py").is_file()
    assert (base_root / "utils.py").is_file()
    assert (base_root / "wan21_runner.py").is_file()
    assert (base_root / "wan22_runner.py").is_file()
    assert (base_root / "worldfoundry_runtime.py").is_file()
    assert (
        REPO_ROOT / "worldfoundry/data/test_cases/fantasy_world/camera_interp_minimal.json"
    ).is_file()

    for path in [
        synthesis_root / "__init__.py",
        synthesis_root / "fantasy_world_wan21_synthesis.py",
        synthesis_root / "fantasy_world_wan22_synthesis.py",
    ]:
        text = path.read_text(encoding="utf-8")
        forbidden_owner = "base_models.diffusion_model.video." + "fantasy_world"
        assert forbidden_owner not in text
        assert "._runtime_env" not in text
        assert "fantasy_world_utils" not in text
        assert "fantasy_world_wan21_runner" not in text
        assert "fantasy_world_wan22_runner" not in text
        assert "cameras_interp" not in text
        assert "save_colored_pointcloud_ply" not in text


def test_hunyuan_world_runtime_defaults_live_in_data_runtime_configs():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/hunyuan_world"
    game_config = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/config.py"
    game_runtime = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/runtime.py"
    game_synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft_synthesis.py"
    voyager_config = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_voyager/config.py"
    worldplay_pipeline = REPO_ROOT / "worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_worldplay.py"
    voyager_synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_voyager_synthesis.py"
    worldplay_synthesis = REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay_synthesis.py"

    assert [path.name for path in sorted(config_root.glob("*.yaml"))] == [
        "game_craft.yaml",
        "voyager.yaml",
        "worldplay.yaml",
    ]

    from worldfoundry.synthesis.visual_generation.hunyuan_world import (
        generate_crop_size_list,
        load_hunyuan_world_runtime_defaults,
    )
    from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.config import (
        parse_args as parse_game,
    )
    from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager.config import (
        parse_args as parse_voyager,
    )

    assert load_hunyuan_world_runtime_defaults("game_craft")["infer_steps"] == 100
    assert parse_game(args=[]).infer_steps == 100
    assert load_hunyuan_world_runtime_defaults("voyager")["video_size"] == [512, 768]
    assert parse_voyager(argv=[]).video_size == (512, 768)
    assert load_hunyuan_world_runtime_defaults("worldplay")["infer_state_kwargs"]["quant_type"] == "fp8-per-block"

    assert "apply_hunyuan_world_argparse_defaults" in game_config.read_text(encoding="utf-8")
    assert game_runtime.is_file()
    game_runtime_text = game_runtime.read_text(encoding="utf-8")
    game_synthesis_text = game_synthesis.read_text(encoding="utf-8")
    assert "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft" in game_runtime_text
    assert "matplotlib" not in game_runtime_text
    assert "CameraPoseVisualizer" not in game_runtime_text
    assert "HunyuanGameCraftRuntime" in game_synthesis_text
    assert "load_diffusion_pipeline" not in game_synthesis_text
    assert "FlowMatchDiscreteScheduler" not in game_synthesis_text
    assert "snapshot_download" not in game_synthesis_text
    assert "apply_hunyuan_world_argparse_defaults" in voyager_config.read_text(encoding="utf-8")
    worldplay_text = worldplay_pipeline.read_text(encoding="utf-8")
    assert "load_hunyuan_world_runtime_defaults" in worldplay_text
    assert '"sage_blocks_range": "0-53"' not in worldplay_text
    assert generate_crop_size_list(base_size=256, patch_size=16)
    forbidden_owner = "base_models.diffusion_model.video." + "hunyuan_world"
    assert forbidden_owner not in voyager_synthesis.read_text(encoding="utf-8")
    assert forbidden_owner not in worldplay_synthesis.read_text(encoding="utf-8")
    assert not (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_voyager/utils/train_utils.py"
    ).exists()
    assert not (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_voyager/utils/data_utils.py"
    ).exists()
    assert not (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/utils/data_utils.py"
    ).exists()


def test_hunyuan_gamecraft_camera_helpers_live_in_base_runtime():
    from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.runtime import (
        ActionToPoseFromID,
        GetPoseEmbedsFromPoses,
        HunyuanGameCraftRuntime,
    )

    poses = ActionToPoseFromID("w", value=0.1, duration=2)
    pose_embeds, uncond_pose_embeds, selected = GetPoseEmbedsFromPoses(
        poses,
        16,
        16,
        2,
        start_index=0,
    )

    assert HunyuanGameCraftRuntime.__module__.startswith("worldfoundry.synthesis.")
    assert len(poses) == 3
    assert len(selected) == 2
    assert tuple(pose_embeds.shape) == (2, 6, 16, 16)
    assert tuple(uncond_pose_embeds.shape) == (2, 6, 16, 16)


def test_hunyuan_world_voyager_runtime_logic_lives_under_synthesis():
    runtime = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_voyager/runtime.py"
    )
    synthesis = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_world_voyager_synthesis.py"
    )

    assert runtime.is_file()
    runtime_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class HunyuanWorldVoyagerRuntime" in runtime_text
    assert "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager" in runtime_text
    assert "BaseSynthesis" not in runtime_text
    assert "HunyuanWorldVoyagerRuntime" in synthesis_text
    assert "def create_hunyuan_video_input" in synthesis_text
    assert "return self.runtime.create_hunyuan_video_input" in synthesis_text
    assert "return self.runtime.predict" in synthesis_text
    for heavy_marker in [
        "HunyuanVideoPipeline",
        "FlowMatchDiscreteScheduler",
        "load_lora_for_pipeline",
        "load_state_dict",
        "cv2",
        "pyexr",
        "safetensors",
        "torch.distributed",
    ]:
        assert heavy_marker not in synthesis_text

    from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager import (
        HunyuanWorldVoyagerRuntime,
        get_1d_rotary_pos_embed_riflex,
    )

    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(4, 2, use_real=True)

    assert HunyuanWorldVoyagerRuntime.__module__.startswith("worldfoundry.synthesis.")
    assert tuple(freqs_cos.shape) == (2, 4)
    assert tuple(freqs_sin.shape) == (2, 4)


def test_hunyuan_worldplay_runtime_logic_lives_under_synthesis():
    runtime = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/runtime.py"
    )
    synthesis = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay_synthesis.py"
    )
    generate = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/generate.py"
    )
    sr_pipeline = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/pipelines/hunyuan_video_sr_pipeline.py"
    )

    assert runtime.is_file()
    runtime_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")
    generate_text = generate.read_text(encoding="utf-8")
    sr_pipeline_text = sr_pipeline.read_text(encoding="utf-8")

    assert "class HunyuanWorldPlayRuntime" in runtime_text
    assert "class _HunyuanWorldPlayInternalPipeline" in runtime_text
    assert "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay" in runtime_text
    assert "BaseSynthesis" not in runtime_text
    assert "HunyuanWorldPlayRuntime" in synthesis_text
    assert "return self.runtime.predict" in synthesis_text
    assert "class _HunyuanWorldPlayInternalPipeline" not in synthesis_text
    for heavy_marker in [
        "DiffusionPipeline",
        "FlowMatchDiscreteScheduler",
        "HunyuanVideo_1_5_DiffusionTransformer",
        "snapshot_download",
        "randn_tensor",
        "torch.distributed",
    ]:
        assert heavy_marker not in synthesis_text
    assert "from .runtime import _HunyuanWorldPlayInternalPipeline" in generate_text
    assert "from ..runtime import _HunyuanWorldPlayInternalPipeline" in sr_pipeline_text


def test_hunyuan_world_shared_layers_import_base_models_directly():
    removed_shims = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/modules/activation_layers.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/modules/modulate_layers.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/modules/norm_layers.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/models/transformers/modules/activation_layers.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/models/transformers/modules/posemb_layers.py",
    ]
    for path in removed_shims:
        assert not path.exists()

    direct_import_files = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/modules/models.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_game_craft/modules/token_refiner.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/models/transformers/worldplay_1_5_transformer.py",
    ]
    for path in direct_import_files:
        assert "worldfoundry.base_models.diffusion_model.video.hunyuan_video.modules" in path.read_text(
            encoding="utf-8"
        )


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


def test_genie_envisioner_configs_live_in_data_runtime_configs():
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/genie_envisioner"
    old_config_root = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/genie_envisioner/genie_envisioner_runtime/configs"
    )

    assert (config_root / "acwm_cosmos.yaml").is_file()
    assert (config_root / "ltx_model/video_model_infer_slow.yaml").is_file()
    assert (config_root / "ltx_model/calvin/stats_calvin_rel.yaml").is_file()
    assert not old_config_root.exists()


def test_motionctrl_lvdm_foundation_code_lives_under_base_models():
    old_runtime_lvdm = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/lvdm"
    )
    video_base_model_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video"
    base_lvdm = video_base_model_root / "lvdm"
    runtime_init = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/__init__.py"
    )
    runtime_config = (
        REPO_ROOT
        / "worldfoundry/data/models/runtime/configs/camera_control/motionctrl_config_both.yaml"
    )
    runtime_files = [
        runtime_init,
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/main/inference/motionctrl_inference.py",
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/motionctrl/motionctrl.py",
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/motionctrl/motionctrl_runtime/motionctrl/lvdm_modified_modules.py",
    ]
    lvdm_encoder_files = [
        base_lvdm / "modules/encoders/condition.py",
        base_lvdm / "modules/encoders/condition2.py",
    ]

    assert not old_runtime_lvdm.exists()
    assert [path.name for path in video_base_model_root.iterdir() if path.name.startswith("lvdm_")] == []
    assert (base_lvdm / "models/ddpm3d.py").is_file()
    assert (base_lvdm / "modules/networks/openaimodel3d_next.py").is_file()
    for runtime_file in runtime_files:
        text = runtime_file.read_text(encoding="utf-8")
        assert "from lvdm." not in text
        assert "from main." not in text
        assert "from motionctrl." not in text
        assert "from utils." not in text
        assert 'sys.modules.setdefault("lvdm"' not in text
    for lvdm_file in lvdm_encoder_files:
        text = lvdm_file.read_text(encoding="utf-8")
        assert "from utils." not in text
        assert "worldfoundry.base_models.diffusion_model.video.lvdm.utils" in text
    config_text = runtime_config.read_text(encoding="utf-8")
    assert "target: lvdm." not in config_text
    assert "target: motionctrl." not in config_text
    assert "target: utils." not in config_text
    assert "worldfoundry.base_models.diffusion_model.video.lvdm" in config_text
    assert (
        "target: worldfoundry.synthesis.visual_generation.motionctrl.motionctrl_runtime.motionctrl.motionctrl.MotionCtrl"
        in config_text
    )


def test_videocrafter_lvdm_foundation_code_lives_under_base_models():
    old_runtime_lvdm = (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/videocrafter/videocrafter_runtime/lvdm"
    )
    base_lvdm = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/lvdm"
    old_variant_lvdm = base_lvdm / "variants/videocrafter"
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/videocrafter"
    old_runtime_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/videocrafter"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/videocrafter"
    runtime_files = [
        runtime_root / "__init__.py",
        runtime_root / "components.py",
        runtime_root / "worldfoundry_runtime.py",
        runtime_root / "videocrafter_inference.py",
    ]

    assert not old_runtime_root.exists()
    assert not old_runtime_lvdm.exists()
    assert not old_variant_lvdm.exists()
    assert (base_lvdm / "models/ddpm3d.py").is_file()
    assert (base_lvdm / "models/samplers/ddim.py").is_file()
    for runtime_file in runtime_files:
        text = runtime_file.read_text(encoding="utf-8")
        assert "from lvdm." not in text
        assert "import lvdm" not in text
        assert "sys.path" not in text
        assert "variants.videocrafter" not in text
    for config_file in config_root.glob("inference_*.yaml"):
        config_text = config_file.read_text(encoding="utf-8")
        assert "target: lvdm." not in config_text
        assert "variants.videocrafter" not in config_text
        assert "worldfoundry.base_models.diffusion_model.video.lvdm" in config_text


def test_vid2world_lvdm_variant_lives_under_canonical_lvdm_tree():
    variant_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/lvdm/variants/vid2world"
    old_nested_lvdm = variant_root / "lvdm"
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/world_model/vid2world"
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/vid2world"

    assert not old_nested_lvdm.exists()
    assert (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/lvdm/models/ddpm3d_vid2world.py"
    ).is_file()
    assert (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/lvdm/modules/networks/openaimodel3d_vid2world.py"
    ).is_file()
    assert (
        REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/lvdm/modules/attention_vid2world.py"
    ).is_file()
    assert (variant_root / "models/samplers/kv_cache.py").is_file()
    assert (variant_root / "eval_inputs/csgovid.py").is_file()
    assert (variant_root / "eval_inputs/reconvid.py").is_file()
    allowed_variant_files = {
        "__init__.py",
        "eval_inputs/base.py",
        "eval_inputs/csgovid.py",
        "eval_inputs/reconvid.py",
        "eval_inputs/rtvid.py",
        "eval_inputs/rtvid_action_control.py",
        "eval_inputs/valorant.py",
        "models/samplers/kv_cache.py",
    }
    assert {
        str(path.relative_to(variant_root))
        for path in variant_root.rglob("*.py")
    } == allowed_variant_files

    for path in variant_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from lvdm." not in text
        assert "import lvdm" not in text
        assert "sys.path" not in text

    runtime_files = [
        runtime_root / "worldfoundry_runtime.py",
        runtime_root / "vid2world_runtime/main/inference.py",
        runtime_root / "vid2world_runtime/main/utils_data.py",
        runtime_root / "vid2world_runtime/nvm_utils/eval_inputs.py",
    ]
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        assert "variants.vid2world.lvdm" not in text
        assert "from lvdm." not in text
        assert "sys.path" not in text

    runtime_text = (runtime_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")
    assert "package_root(" not in runtime_text
    assert "return []" in runtime_text

    for config_file in config_root.rglob("*.yaml"):
        config_text = config_file.read_text(encoding="utf-8")
        assert "target: lvdm." not in config_text
        assert "target: utils_data." not in config_text
        assert (
            "worldfoundry.base_models.diffusion_model.video.lvdm.models.ddpm3d_vid2world"
            in config_text
        )
        assert (
            "worldfoundry.base_models.diffusion_model.video.lvdm.modules.networks.openaimodel3d_vid2world"
            in config_text
        )
        assert "worldfoundry.base_models.diffusion_model.video.lvdm.variants.vid2world" in config_text


def test_t2v_turbo_lvdm_variant_lives_under_canonical_lvdm_tree():
    old_runtime_lvdm = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/t2v_turbo/t2v_turbo_runtime/source/lvdm"
    )
    runtime_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/t2v_turbo"
    config_path = runtime_root / "t2v_turbo_runtime/configs/inference_t2v_512_v2.0.yaml"
    pipeline_path = runtime_root / "t2v_turbo_runtime/source/pipeline/t2v_turbo_vc2_pipeline.py"
    runtime_path = runtime_root / "t2v_turbo_runtime/runtime.py"
    old_variant_lvdm = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/lvdm/variants/t2v_turbo"
    )

    assert not old_runtime_lvdm.exists()
    assert not old_variant_lvdm.exists()
    assert (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/lvdm/modules/networks/openaimodel3d.py"
    ).is_file()

    canonical_module = "worldfoundry.base_models.diffusion_model.video.lvdm"
    variant_module = "worldfoundry.base_models.diffusion_model.video.lvdm.variants.t2v_turbo"
    for path in (config_path, pipeline_path):
        text = path.read_text(encoding="utf-8")
        assert "t2v_turbo_runtime.source.lvdm" not in text
        assert variant_module not in text
        assert canonical_module in text

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert "source/lvdm" not in runtime_text
    assert "../../lvdm/variants/t2v_turbo" not in runtime_text
    assert "../../lvdm/modules/networks/openaimodel3d.py" in runtime_text


def test_diffsynth_runtime_tokenizers_are_shared_from_base_models():
    shared_root = REPO_ROOT / "worldfoundry/base_models/diffusion_model/diffsynth/tokenizer_configs"
    runtime_roots = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/sama/sama_runtime/diffsynth",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/pusa_vidgen/pusav1_runtime/diffsynth",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/unianimate_dit/unianimate_dit_runtime/diffsynth",
    ]

    assert (shared_root / "paths.py").is_file()
    assert (shared_root / "hunyuan_dit/tokenizer_t5/config.json").is_file()
    assert (shared_root / "hunyuan_video/tokenizer_2/tokenizer_config.json").is_file()

    for runtime_root in runtime_roots:
        for prompter_path in sorted((runtime_root / "prompters").glob("*prompter.py")):
            text = prompter_path.read_text(encoding="utf-8")
            assert "os.path.dirname(os.path.dirname(__file__))" not in text
            if "tokenizer_configs/" in text:
                assert "shared_diffsynth_root" in text


def test_pandora_runtime_init_does_not_import_gradio_app_helpers():
    init_path = REPO_ROOT / "worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/__init__.py"
    text = init_path.read_text(encoding="utf-8")

    assert "demo_utils" not in text
    assert "gradio" not in text


def test_gr00t_policy_resolves_cosmos_reason2_from_local_hfd_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from worldfoundry.synthesis.action_generation.gr00t.gr00t_runtime import (
        install_aliases,
    )

    install_aliases()
    from worldfoundry.synthesis.action_generation.gr00t.gr00t_runtime.policy import (
        gr00t_policy,
    )

    hfd_root = tmp_path / "hfd"
    cosmos_dir = hfd_root / "nvidia--Cosmos-Reason2-2B"
    cosmos_dir.mkdir(parents=True)
    (cosmos_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (cosmos_dir / "model.safetensors").write_bytes(b"placeholder")
    checkpoint_dir = tmp_path / "gr00t-libero"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))

    kwargs = gr00t_policy._local_backbone_kwargs(checkpoint_dir)

    assert kwargs["model_name"] == str(cosmos_dir.resolve())
    assert kwargs["transformers_loading_kwargs"] == {
        "local_files_only": True,
        "low_cpu_mem_usage": False,
        "trust_remote_code": True,
    }


def test_diffsynth_runtime_base_pipeline_uses_canonical_owner():
    canonical_path = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/diffsynth/diffusion/base_pipeline.py"
    )
    removed_base_pipeline_files = [
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/diffsynth/fantasy_world_wan21/pipelines/base.py",
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/diffsynth/fantasy_world_wan22/pipelines/base.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/kling/recammaster_runtime/pipelines/base.py",
    ]

    assert canonical_path.exists()
    assert [path for path in removed_base_pipeline_files if path.exists()] == []


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
    assert (REPO_ROOT / "worldfoundry/core/nn/layers.py").exists()
    assert (REPO_ROOT / "worldfoundry/core/attention/rope_2d.py").exists()

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

    hy_mlp = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/layers/mlp.py"
    )
    hy_mlp_text = hy_mlp.read_text(encoding="utf-8")
    hy_mlp_tree = ast.parse(hy_mlp_text, filename=str(hy_mlp))
    hy_class_names = [node.name for node in ast.walk(hy_mlp_tree) if isinstance(node, ast.ClassDef)]
    assert hy_class_names == ["MlpFP32"]
    assert "from worldfoundry.core.nn import Mlp" in hy_mlp_text


def test_hunyuan_worldmirror_duplicate_utils_use_canonical_owner():
    duplicate_paths = [
        "worldfoundry/base_models/diffusion_model/diffsynth/neoverse/auxiliary_models",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/act_gs.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/camera_utils.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/frustum.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/grid.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/priors.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/rotation.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/sh_utils.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/utils/geometry.py",
    ]
    kept_variant_specific_paths = [
        "worldfoundry/base_models/diffusion_model/diffsynth/models/neoverse_rasterization.py",
        "worldfoundry/base_models/diffusion_model/diffsynth/models/neoverse_geometry.py",
        "worldfoundry/base_models/diffusion_model/diffsynth/models/neoverse_depth_anything_reconstructor.py",
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/utils/geometry.py",
    ]
    import_expectations = {
        "worldfoundry/base_models/diffusion_model/diffsynth/models/neoverse_rasterization.py": [
            "worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.models.utils import act_gs",
        ],
        "worldfoundry/base_models/diffusion_model/diffsynth/models/neoverse_depth_anything_reconstructor.py": [
            "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api",
        ],
        "worldfoundry/base_models/diffusion_model/diffsynth/configs/neoverse_model_config.py": [
            "worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0 import WorldMirror",
        ],
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/heads/dense_head.py": [
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.grid",
        ],
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/models/worldmirror.py": [
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.camera_utils",
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.priors",
        ],
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/models/models/rasterization.py": [
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.frustum",
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils import act_gs, sh_utils",
        ],
        "worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/utils/inference_utils.py": [
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.camera_utils",
            "worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.utils.geometry",
        ],
    }

    assert [path for path in duplicate_paths if (REPO_ROOT / path).exists()] == []
    assert [path for path in kept_variant_specific_paths if not (REPO_ROOT / path).exists()] == []

    violations = []
    for file_name, expected_imports in import_expectations.items():
        text = (REPO_ROOT / file_name).read_text(encoding="utf-8")
        for expected_import in expected_imports:
            if expected_import not in text:
                violations.append(f"{file_name}: missing {expected_import}")
        if "from src.utils" in text:
            violations.append(f"{file_name}: uses non-package src.utils import")

    assert violations == []


def test_lvdm_foundation_code_exists_only_under_base_models():
    lvdm_dirs = [
        path
        for path in (REPO_ROOT / "worldfoundry").rglob("lvdm")
        if ".git" not in path.parts and path.is_dir()
    ]
    expected = REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/lvdm"

    assert lvdm_dirs == [expected]
    assert (expected / "models/ddpm3d.py").is_file()
    assert (expected / "models/samplers/ddim.py").is_file()


def test_vggt_dinov2_exact_layer_shims_use_canonical_owner():
    layer_paths = [
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/vggt/vggt/layers/drop_path.py",
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/point_clouds/vggt/vggt/layers/mlp.py",
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/vggt_fantasy_world/vggt/layers/drop_path.py",
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/vggt_fantasy_world/vggt/layers/mlp.py",
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
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )
        if has_local_logic or "worldfoundry.base_models.perception_core.general_perception.dinov2.layers" not in text:
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert violations == []


def test_lingbot_map_base_package_does_not_use_top_level_imports():
    package_root = (
        REPO_ROOT
        / "worldfoundry/base_models/three_dimensions/point_clouds/lingbot_map/lingbot_map"
    )
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
    vipe_priors_text = (vipe_priors / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "three_dimensions.depth" in vipe_priors_text
    assert "three_dimensions.general_3d.geocalib" in vipe_priors_text
    assert "perception_core.tracking.track_anything" in vipe_priors_text


def test_geometric_priors_are_independent_model_catalog_entries():
    import yaml

    from worldfoundry.evaluation.models.catalog.schema import load_entries
    from worldfoundry.evaluation.models.runners.resolver import resolve_model_zoo_config
    from worldfoundry.evaluation.models.runtime import load_runtime_profile_manifest

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
        "unik3d-prior": "worldfoundry.base_models.three_dimensions.depth.unik3d",
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
        assert entries[0].runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
        resolved = resolve_model_zoo_config(model_id, manifest_dir=catalog_root.parent)
        assert str(resolved.diagnostics["pipeline_target"]).startswith("worldfoundry.pipelines.geometry_priors.")
        assert payload["integration"]["kind"] == "pipeline_runner"
        assert payload["base_model_target"] == base_model_target
        assert not payload["base_model_target"].startswith(
            "worldfoundry.base_models.three_dimensions.general_3d.vipe"
        )
        assert runtime_profile.model_id == model_id
        assert runtime_profile.integration_status == "integrated"
        assert runtime_profile.backend_stage == "in_tree_geometry_prior_runtime"
        assert "prior" in runtime_profile.groups


def test_three_d_four_d_external_repo_resolution_requires_explicit_root(monkeypatch, tmp_path: Path):
    from worldfoundry.base_models.three_dimensions.three_d_four_d import (
        runtime as runtime_module,
    )
    from worldfoundry.base_models.three_dimensions.three_d_four_d.runtime import (
        ThreeDFourDRuntimeSpec,
    )

    runtime_text = (
        REPO_ROOT / "worldfoundry/base_models/three_dimensions/three_d_four_d/runtime.py"
    ).read_text(encoding="utf-8")
    assert "github_repos" not in runtime_text

    spec = ThreeDFourDRuntimeSpec(
        model_id="fixture-three-d",
        display_name="Fixture 3D",
        repo_names=("fixture-repo",),
        entrypoint="demo.py",
        command_kind="fixture",
        artifact_kind="generated_3d_asset",
        artifact_filename="fixture.ply",
    )
    fake_repo_root = tmp_path / "WorldFoundry"
    sibling = tmp_path / "github_repos" / "fixture-repo"
    sibling.mkdir(parents=True)
    (sibling / "demo.py").write_text("# sibling fixture\n", encoding="utf-8")

    monkeypatch.setattr(runtime_module, "REPO_ROOT", fake_repo_root)
    monkeypatch.setenv("WORLDFOUNDRY_ALLOW_EXTERNAL_THREE_D_FOUR_D_REPOS", "1")
    monkeypatch.delenv("WORLDFOUNDRY_THREE_D_FOUR_D_REPOS_ROOT", raising=False)

    assert runtime_module._resolve_source_root(spec, {}, ()) is None

    explicit_root = tmp_path / "explicit_repos"
    explicit_repo = explicit_root / "fixture-repo"
    explicit_repo.mkdir(parents=True)
    (explicit_repo / "demo.py").write_text("# explicit fixture\n", encoding="utf-8")
    monkeypatch.setenv("WORLDFOUNDRY_THREE_D_FOUR_D_REPOS_ROOT", str(explicit_root))

    assert runtime_module._resolve_source_root(spec, {}, ()) == explicit_repo


def test_neoverse_diffsynth_reuses_canonical_heavy_model_files():
    reexports = ["flux_dit.py", "sd3_dit.py", "svd_unet.py"]
    model_root = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/diffsynth/models"
    )

    assert [filename for filename in reexports if (model_root / f"neoverse_{filename}").exists()] == []
    assert (model_root / "wan_video_neoverse_controller.py").is_file()
    assert (model_root / "neoverse_depth_anything_reconstructor.py").is_file()
    assert not (model_root.parent / "neoverse").exists()


def test_remaining_foundation_exact_duplicates_are_canonical_reexports_only() -> None:
    reexport_expectations = {}
    removed_reexports = [
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2/cosmos_predict2/_src/predict2/models/utils.py",
        "worldfoundry/base_models/diffusion_model/video/cosmos/cosmos2/runtime/cosmos_predict2_wow/cosmos_predict2/models/utils.py",
        "worldfoundry/base_models/diffusion_model/video/wan/wan_2p2/modules/animate/xlm_roberta.py",
    ]

    assert not [path for path in (REPO_ROOT).rglob("*") if path.is_symlink()]
    assert not [path for path in removed_reexports if (REPO_ROOT / path).exists()]

    for relative_path, canonical_import in reexport_expectations.items():
        path = REPO_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        local_logic = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        assert canonical_import in text
        assert local_logic == []


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

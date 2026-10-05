from __future__ import annotations
import json
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from worldfoundry.base_models import capabilities
from worldfoundry.base_models.capabilities import (
    BASE_MODEL_CAPABILITIES,
    BASE_MODEL_STACKS,
    benchmark_base_model_dependency_ids,
    benchmark_data_asset_capability_ids,
    base_model_inventory,
    base_model_materialization_plan,
    check_base_model_dependencies,
    resolve_base_model_capability_ids,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external"
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    iter_benchmark_catalog_manifest_paths,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import load_benchmark_runtime_profiles
from worldfoundry.evaluation.utils import load_manifest


def _dependency_tuple(payload: dict, key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not value:
        return ()
    return tuple(str(item) for item in value)


def test_base_model_capability_registry_owns_heavy_benchmark_dependencies() -> None:
    expected = {
        "clip_vit_b32",
        "aot_deaot_l",
        "aigcbench_dataset_assets",
        "bridgedata_v2_dataset_assets",
        "bridgedata_v2_source_assets",
        "camerabench_dataset_assets",
        "calvin_dataset_assets",
        "calvin_source_assets",
        "chronomagic_dataset_assets",
        "coco2017_detection_segmentation_assets",
        "devil_dynamics_source_assets",
        "depth_anything_v3",
        "dinov2_base",
        "droid_slam",
        "ewmbench_dataset_assets",
        "evalcrafter_dataset_assets",
        "fetv_dataset_assets",
        "genai_bench_dataset_assets",
        "iworld_bench_source_assets",
        "iworld_bench_dataset_assets",
        "ipv_bench_source_assets",
        "kitti_depth_slam_assets",
        "grounding_dino",
        "libero_dataset_assets",
        "libero_source_assets",
        "libero_para_dataset_assets",
        "libero_para_source_assets",
        "maniskill_source_assets",
        "metaworld_source_assets",
        "mirabench_source_assets",
        "moge_vitl",
        "moge_v2_vitl_normal",
        "phyeduvideo_source_assets",
        "phygenbench_source_assets",
        "phyground_dataset_assets",
        "physics_iq_source_assets",
        "physvidbench_dataset_assets",
        "raft",
        "rlbench_source_assets",
        "robocasa_source_assets",
        "robotwin_dataset_assets",
        "sam2",
        "sam3",
        "sam_v1",
        "sam_vit_b",
        "sea_raft",
        "siglip_so400m",
        "simpler_env_source_assets",
        "t2v_safety_bench_source_assets",
        "t2v_compbench_dataset_assets",
        "t2vphysbench_external_assets",
        "t2vworldbench_source_assets",
        "tum_rgbd_slam_sample_assets",
        "unidepth_v2_vitl14",
        "vbench_source_assets",
        "vbench2_dataset_assets",
        "vggt_1b",
        "video_bench_dataset_assets",
        "videoscience_bench_source_assets",
        "videophy2_dataset_assets",
        "videophy_source_assets",
        "videoverse_dataset_assets",
        "videoscore_bench_dataset_assets",
        "videoscore_reward_model_v1_1",
        "vmbench_source_assets",
        "davis2017_video_segmentation_assets",
        "world_in_world_source_assets",
        "worldarena_source_assets",
        "worldbench_dataset_assets",
        "worldmodelbench_dataset_assets",
        "worldscore_official_assets",
        "worldscore_dataset_assets",
    }

    assert expected <= set(BASE_MODEL_CAPABILITIES)
    for capability_id in expected:
        capability = BASE_MODEL_CAPABILITIES[capability_id]
        assert capability.canonical_owner.startswith("worldfoundry.base_models.")
        assert capability.canonical_path.startswith("worldfoundry/base_models/")
        assert capability.owner_path().exists()


def test_evalcrafter_dataset_assets_accept_common_local_directory_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "hfd_datasets" / "EvalCrafter_T2V_Dataset"
    dataset_root.mkdir(parents=True)
    (dataset_root / "README.md").write_text("EvalCrafter local dataset", encoding="utf-8")
    monkeypatch.setenv("WORLDFOUNDRY_HFD_DATASET_ROOT", str(tmp_path / "hfd_datasets"))

    result = check_base_model_dependencies(["evalcrafter_dataset_assets"])

    assert result["ok"] is True
    asset = result["checks"][0]["asset_status"][0]
    assert asset["ready"] is True
    assert asset["matched_path"] == str(dataset_root)


def test_base_model_dependency_preflight_reports_assets_without_importing_runtimes(tmp_path: Path) -> None:
    report = check_base_model_dependencies(
        ["depth_anything_v3", "droid_slam"],
        env={
            "WORLDFOUNDRY_DEPTH_ANYTHING_MODEL_DIR": "/ckpt/depth",
            "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
            "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "ckpt"),
            "WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR": str(tmp_path / "worldscore_ckpt"),
        },
    )

    assert report["capability_ids"] == ["depth_anything_v3", "droid_slam"]
    assert report["checks"][0]["owner_path_exists"] is True
    assert report["checks"][0]["asset_status"][0]["hf_repo_id"] == "depth-anything/DA3-LARGE-1.1"
    assert report["checks"][0]["asset_status"][0]["role"] == "model_dir"
    depth_download = report["checks"][0]["repair_hints"]["download_commands"][0]
    assert "hf download depth-anything/DA3-LARGE-1.1" in depth_download
    assert "download depth-anything/DA3-LARGE-1.1 config.json" in depth_download
    assert "download depth-anything/DA3-LARGE-1.1 model.safetensors" in depth_download
    assert report["checks"][1]["owner_path_exists"] is True
    assert report["checks"][1]["asset_status"][0]["id"] == "droid_slam_checkpoint"
    assert report["checks"][1]["asset_status"][0]["role"] == "checkpoint"
    assert report["checks"][1]["asset_status"][0]["candidate_paths"]


def test_base_model_dir_assets_require_real_files(tmp_path: Path) -> None:
    empty_model_dir = tmp_path / "empty-depth-model"
    empty_model_dir.mkdir()
    (empty_model_dir / ".hfd").mkdir()
    (empty_model_dir / ".hfd" / "download.log").write_text("partial download", encoding="utf-8")

    report = check_base_model_dependencies(
        ["depth_anything_v3"],
        env={
            "WORLDFOUNDRY_DEPTH_ANYTHING_MODEL_DIR": str(empty_model_dir),
            "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
        },
    )
    asset_status = report["checks"][0]["asset_status"][0]

    assert report["ok"] is False
    assert asset_status["ready"] is False
    assert asset_status["env_status"][0]["present"] is True
    assert asset_status["env_status"][0]["ready"] is False
    assert asset_status["min_file_count"] == 1
    assert asset_status["required_files"] == ["config.json", "model.safetensors"]


def test_base_model_materialization_plan_emits_download_and_export_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str) -> object | None:
        return None if name == "hydra" else object()

    monkeypatch.setattr(capabilities.importlib.util, "find_spec", fake_find_spec)

    plan = base_model_materialization_plan(
        ["grounding_dino", "sam2", "sam_v1"],
        env={
            "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
            "WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR": str(tmp_path / "worldscore_ckpt"),
        },
    )

    assert plan["ok"] is False
    assert any("download ShilongLiu/GroundingDINO groundingdino_swint_ogc.pth" in item for item in plan["download_commands"])
    assert any("download bert-base-uncased config.json" in item for item in plan["download_commands"])
    assert any("download bert-base-uncased model.safetensors" in item for item in plan["download_commands"])
    assert any("download bert-base-uncased vocab.txt" in item for item in plan["download_commands"])
    assert any("download facebook/sam2.1-hiera-base-plus sam2.1_hiera_base_plus.pt" in item for item in plan["download_commands"])
    assert any("sam_vit_h_4b8939.pth" in item for item in plan["download_commands"])
    assert any(command[:3] == ["hf", "download", "ShilongLiu/GroundingDINO"] for command in plan["download_command_argvs"])
    assert "hydra-core" in plan["pip_install_packages"]
    assert any(command.startswith("export WORLDFOUNDRY_SAM2_CKPT=") for command in plan["export_commands"])
    assert any(command.startswith("export WORLDFOUNDRY_SAM_VIT_H_CKPT=") for command in plan["export_commands"])


def test_base_model_assets_accept_plain_workspace_ckpt_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda name: object())
    workspace = tmp_path / "workspace"
    (workspace / "ckpt/moge-vitl").mkdir(parents=True)
    (workspace / "ckpt/moge-vitl/model.pt").write_bytes(b"moge")
    (workspace / "ckpt/VGGT-1B").mkdir(parents=True)
    (workspace / "ckpt/VGGT-1B/config.json").write_text("{}", encoding="utf-8")
    (workspace / "ckpt/VGGT-1B/model.safetensors").write_bytes(b"vggt")

    plan = base_model_materialization_plan(
        ["moge_vitl", "vggt_1b"],
        env={"WORLDFOUNDRY_HFD_ROOT": str(workspace / "hfd")},
    )

    by_id = {check["id"]: check for check in plan["checks"]}
    assert by_id["moge_vitl"]["asset_status"][0]["matched_path"] == str(workspace / "ckpt/moge-vitl")
    assert by_id["vggt_1b"]["asset_status"][0]["matched_path"] == str(workspace / "ckpt/VGGT-1B")
    assert plan["ok"] is True


def test_perception_assets_accept_plain_workspace_ckpt_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda name: object())
    workspace = tmp_path / "workspace"
    grounding_dino = workspace / "ckpt/GroundingDINO/groundingdino_swint_ogc.pth"
    sam_v1 = workspace / "ckpt/code/sam_vit_h_4b8939.pth"
    sam2 = workspace / "ckpt/sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt"
    grounding_dino.parent.mkdir(parents=True)
    sam_v1.parent.mkdir(parents=True)
    sam2.parent.mkdir(parents=True)

    def make_sparse_file(path: Path, size: int) -> None:
        with path.open("wb") as handle:
            handle.truncate(size)

    make_sparse_file(grounding_dino, 600_000_000)
    make_sparse_file(sam_v1, 2_000_000_000)
    make_sparse_file(sam2, 300_000_000)

    plan = base_model_materialization_plan(
        ["grounding_dino", "sam_v1", "sam2"],
        env={"WORLDFOUNDRY_WORKSPACE_ROOT": str(workspace)},
    )

    by_id = {check["id"]: check for check in plan["checks"]}
    assert by_id["grounding_dino"]["asset_status"][0]["matched_path"] == str(grounding_dino)
    assert by_id["sam_v1"]["asset_status"][0]["matched_path"] == str(sam_v1)
    assert by_id["sam2"]["asset_status"][0]["matched_path"] == str(sam2)


def test_droid_slam_non_pip_cuda_extensions_are_manual_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str) -> object | None:
        return None if name in {"lietorch", "droid_backends", "torch_scatter"} else object()

    monkeypatch.setattr(capabilities.importlib.util, "find_spec", fake_find_spec)

    plan = base_model_materialization_plan(["droid_slam"])
    repair_hints = plan["checks"][0]["repair_hints"]

    assert "torch-scatter" in plan["pip_install_packages"]
    assert "lietorch" not in plan["pip_install_packages"]
    assert "droid_backends" not in plan["pip_install_packages"]
    assert repair_hints["missing_non_pip_imports"] == ["lietorch", "droid_backends"]
    assert any("DROID-SLAM CUDA extensions" in action for action in plan["manual_actions"])


def test_raft_checkpoint_download_command_extracts_single_zip_member(tmp_path: Path) -> None:
    plan = base_model_materialization_plan(
        ["raft"],
        env={
            "WORLDFOUNDRY_CACHE_DIR": str(tmp_path / "cache"),
            "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "ckpt"),
            "WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR": str(tmp_path / "worldscore" / "checkpoints"),
        },
    )
    command = plan["download_commands"][0]
    argv = shlex.split(command)

    assert argv[:2] == ["bash", "-lc"]
    assert "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" in argv[2]
    assert "curl -L --fail --retry 5 --retry-delay 2" in argv[2]
    assert 'member="models/raft-things.pth"' in argv[2]
    assert "archive.getinfo(member)" in argv[2]
    assert "python -m zipfile -e" not in argv[2]
    subprocess.run(["bash", "-n", "-c", argv[2]], check=True)


def test_source_repo_assets_emit_git_clone_hints(tmp_path: Path) -> None:
    plan = base_model_materialization_plan(["vbench_source_assets", "physics_iq_source_assets"])
    vbench_command = BASE_MODEL_CAPABILITIES["vbench_source_assets"].assets[0].download_command(
        {"WORLDFOUNDRY_WORKSPACE_ROOT": str(tmp_path / "workspace")}
    )
    physics_command = BASE_MODEL_CAPABILITIES["physics_iq_source_assets"].assets[0].download_command(
        {"WORLDFOUNDRY_WORKSPACE_ROOT": str(tmp_path / "workspace")}
    )

    assert plan["capability_ids"] == ["vbench_source_assets", "physics_iq_source_assets"]
    assert vbench_command is not None
    assert physics_command is not None
    assert "git clone --depth 1 https://github.com/Vchitect/VBench.git" in vbench_command[2]
    assert "git clone https://github.com/google-deepmind/physics-IQ-benchmark.git" in physics_command[2]
    assert "git -C" in physics_command[2] and "52e52a14d3a7284ae24d8aff98fe982ac7a60971" in physics_command[2]
    assert any(command.startswith("export WORLDFOUNDRY_VBENCH_ROOT=") for command in plan["export_commands"])
    assert any(command.startswith("export WORLDFOUNDRY_PHYSICS_IQ_ROOT=") for command in plan["export_commands"])


def test_embodied_asset_registry_recognizes_configured_model_source_layouts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    repo_root = tmp_path / "WorldFoundry"
    model_source_root = tmp_path / "model-source"
    calvin_dataset = workspace / "data" / "worldfoundry" / "datasets" / "calvin"
    calvin_split = calvin_dataset / "task_D_D"
    rlbench_repo = model_source_root / "stepjam--RLBench"
    calvin_split.mkdir(parents=True)
    rlbench_repo.mkdir(parents=True)
    (calvin_split / "episode_0000000.npz").write_text("calvin data", encoding="utf-8")
    (rlbench_repo / "README.md").write_text("rlbench", encoding="utf-8")

    plan = base_model_materialization_plan(
        ["calvin_dataset_assets", "rlbench_source_assets"],
        env={
            "WORLDFOUNDRY_WORKSPACE_ROOT": str(workspace),
            "WORLDFOUNDRY_REPO_ROOT": str(repo_root),
            "WORLDFOUNDRY_MODEL_SOURCE_DIR": str(model_source_root),
        },
    )

    assert plan["ok"] is True
    assert plan["summary"]["missing_required_asset_ids"] == []


def test_calvin_dataset_assets_require_official_split_marker(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    debug_dataset = workspace / "data" / "worldfoundry" / "calvin" / "calvin_debug_dataset"
    debug_dataset.mkdir(parents=True)
    (debug_dataset / "episode_0000000.npz").write_text("debug only", encoding="utf-8")

    plan = base_model_materialization_plan(
        ["calvin_dataset_assets"],
        env={"WORLDFOUNDRY_WORKSPACE_ROOT": str(workspace)},
    )

    assert plan["ok"] is False
    asset = plan["checks"][0]["asset_status"][0]
    assert asset["ready"] is False
    assert asset["required_any_paths"] == ["task_D_D", "task_ABC_D", "task_ABCD_D"]
    assert "calvin_dataset_assets_dir" in plan["summary"]["missing_required_asset_ids"]


def test_perception_data_assets_require_expected_directory_layout(tmp_path: Path) -> None:
    coco_root = tmp_path / "coco2017"
    (coco_root / "val2017").mkdir(parents=True)
    for index in range(10):
        (coco_root / "val2017" / f"{index:012d}.jpg").write_text("image", encoding="utf-8")

    plan = base_model_materialization_plan(
        ["coco2017_detection_segmentation_assets"],
        env={"WORLDFOUNDRY_COCO2017_ROOT": str(coco_root)},
    )
    asset = plan["checks"][0]["asset_status"][0]

    assert plan["ok"] is False
    assert asset["ready"] is False
    assert asset["required_paths"] == ["val2017", "annotations/instances_val2017.json"]
    assert "coco2017_detection_segmentation_assets_dir" in plan["summary"]["missing_required_asset_ids"]

    (coco_root / "annotations").mkdir()
    (coco_root / "annotations" / "instances_val2017.json").write_text("{}", encoding="utf-8")

    ready_plan = base_model_materialization_plan(
        ["coco2017_detection_segmentation_assets"],
        env={"WORLDFOUNDRY_COCO2017_ROOT": str(coco_root)},
    )
    assert ready_plan["ok"] is True


def test_unreleased_or_pending_public_assets_report_precise_staging_points(tmp_path: Path) -> None:
    plan = base_model_materialization_plan(
        ["mirabench_source_assets", "t2vphysbench_external_assets"],
        env={
            "WORLDFOUNDRY_CACHE_DIR": str(tmp_path / "cache"),
            "WORLDFOUNDRY_DATA_DIR": str(tmp_path / "data"),
        },
    )

    assert plan["ok"] is False
    assert "mirabench_source_assets_repo" in plan["summary"]["missing_required_asset_ids"]
    assert {
        asset["id"]
        for check in plan["checks"]
        for asset in check["asset_status"]
        if asset["required"] is False
    } == {
        "t2vphysbench_external_assets_dir",
    }
    mirabench_asset = next(
        asset
        for check in plan["checks"]
        for asset in check["asset_status"]
        if asset["id"] == "mirabench_source_assets_repo"
    )
    assert mirabench_asset["git_repo_url"] == "https://github.com/mira-space/MiraData.git"
    assert mirabench_asset["required_files"] == []
def test_actionable_embodied_datasets_emit_specific_materialization_hints() -> None:
    plan = base_model_materialization_plan(["bridgedata_v2_dataset_assets", "calvin_dataset_assets"])

    assert plan["ok"] is False
    assert any("download_data.sh" in command for command in plan["download_commands"])
    assert any("WORLDFOUNDRY_CALVIN_DATASET_SPLIT" in command for command in plan["download_commands"])
    assert any("rail.eecs.berkeley.edu/datasets/bridge_release/data" in action for action in plan["manual_actions"])
    assert not any("stage bridgedata_v2_dataset_assets_dir" in action for action in plan["manual_actions"])


def test_base_model_data_assets_use_canonical_worldfoundry_roots() -> None:
    assert (
        BASE_MODEL_CAPABILITIES["calvin_source_assets"].assets[0].local_path
        == "${WORLDFOUNDRY_MODEL_SOURCE_DIR}/mees--calvin"
    )
    assert (
        BASE_MODEL_CAPABILITIES["calvin_dataset_assets"].assets[0].local_path
        == "${WORLDFOUNDRY_DATA_DIR}/datasets/calvin"
    )
    assert (
        BASE_MODEL_CAPABILITIES["bridgedata_v2_dataset_assets"].assets[0].local_path
        == "${WORLDFOUNDRY_DATA_DIR}/datasets/bridgedata-v2"
    )
    assert (
        BASE_MODEL_CAPABILITIES["coco2017_detection_segmentation_assets"].assets[0].local_path
        == "${WORLDFOUNDRY_DATA_DIR}/perception/coco2017"
    )


def test_base_model_stacks_expand_to_canonical_capabilities() -> None:
    assert {
        "depth_stack",
        "geometry_stack",
        "slam_stack",
        "camera_geometry_stack",
        "detection_stack",
        "segmentation_stack",
        "segmentation_tracking_stack",
        "detection_segmentation_stack",
        "spatial_perception_core_stack",
        "spatial_perception_heavy_stack",
        "worldscore_spatial_metric_stack",
        "grounded_depth_segmentation_stack",
        "grounded_video_quality_stack",
        "video_quality_motion_stack",
        "video_quality_dino_motion_stack",
        "vbench_perception_metric_stack",
        "videoscore_reward_metric_stack",
        "official_metric_asset_stack",
        "benchmark_dataset_asset_stack",
        "depth_eval_data_stack",
        "slam_eval_data_stack",
        "detection_segmentation_eval_data_stack",
        "spatial_perception_data_stack",
        "spatial_perception_full_eval_stack",
        "motion_stack",
    } <= set(BASE_MODEL_STACKS)

    resolved = resolve_base_model_capability_ids(["depth_stack", "grounding_dino", "motion_stack"])

    assert resolved == [
        "depth_anything_v3",
        "moge_vitl",
        "moge_v2_vitl_normal",
        "unidepth_v2_vitl14",
        "grounding_dino",
        "raft",
        "sea_raft",
    ]
    plan = base_model_materialization_plan(["detection_segmentation_stack"])
    assert plan["stack_ids"] == ["detection_segmentation_stack"]
    assert plan["capability_ids"] == ["grounding_dino", "sam_v1", "sam_vit_b", "sam2", "sam3", "aot_deaot_l"]
    core_plan = base_model_materialization_plan(["spatial_perception_core_stack"])
    assert core_plan["capability_ids"] == [
        "depth_anything_v3",
        "moge_v2_vitl_normal",
        "unidepth_v2_vitl14",
        "droid_slam",
        "vggt_1b",
        "grounding_dino",
        "sam_v1",
        "sam2",
        "raft",
    ]
    heavy_plan = base_model_materialization_plan(["spatial_perception_heavy_stack"])
    assert heavy_plan["capability_ids"][:4] == [
        "depth_anything_v3",
        "moge_vitl",
        "moge_v2_vitl_normal",
        "unidepth_v2_vitl14",
    ]
    assert "droid_slam" in heavy_plan["capability_ids"]
    assert "grounding_dino" in heavy_plan["capability_ids"]
    assert "sam3" in heavy_plan["capability_ids"]
    camera_plan = base_model_materialization_plan(["camera_geometry_stack"])
    assert camera_plan["capability_ids"] == ["vggt_1b", "moge_v2_vitl_normal", "unidepth_v2_vitl14", "raft"]
    vbench_plan = base_model_materialization_plan(["vbench_perception_metric_stack"])
    assert vbench_plan["capability_ids"] == [
        "clip_vit_b32",
        "dinov2_base",
        "grounding_dino",
        "sam_v1",
        "sam2",
        "sam3",
        "raft",
    ]
    worldscore_plan = base_model_materialization_plan(["worldscore_spatial_metric_stack"])
    assert worldscore_plan["capability_ids"] == [
        "worldscore_official_assets",
        "droid_slam",
        "grounding_dino",
        "raft",
        "sam_v1",
        "sam2",
        "sea_raft",
    ]
    assert worldscore_plan["checks"][0]["family"] == "benchmark_data_asset"
    videoscore_plan = base_model_materialization_plan(["videoscore_reward_metric_stack"])
    assert videoscore_plan["capability_ids"] == ["videoscore_reward_model_v1_1", "videoscore_bench_dataset_assets"]
    assert videoscore_plan["checks"][1]["family"] == "benchmark_data_asset"
    official_asset_plan = base_model_materialization_plan(["official_metric_asset_stack"])
    assert official_asset_plan["capability_ids"][:3] == [
        "worldscore_official_assets",
        "worldscore_dataset_assets",
        "videoscore_bench_dataset_assets",
    ]
    assert {
        "bridgedata_v2_dataset_assets",
        "calvin_dataset_assets",
        "maniskill_source_assets",
        "physics_iq_source_assets",
        "vbench_source_assets",
        "world_in_world_source_assets",
    } <= set(official_asset_plan["capability_ids"])
    assert {check["family"] for check in official_asset_plan["checks"]} == {"benchmark_data_asset"}
    dataset_stack_plan = base_model_materialization_plan(["benchmark_dataset_asset_stack"])
    assert dataset_stack_plan["capability_ids"][:3] == [
        "aigcbench_dataset_assets",
        "camerabench_dataset_assets",
        "chronomagic_dataset_assets",
    ]
    assert {
        "bridgedata_v2_source_assets",
        "calvin_source_assets",
        "t2vphysbench_external_assets",
        "videoscience_bench_source_assets",
        "worldarena_source_assets",
    } <= set(dataset_stack_plan["capability_ids"])
    dataset_asset = BASE_MODEL_CAPABILITIES["videoscore_bench_dataset_assets"].assets[0]
    videoscore_command = dataset_asset.download_command()
    assert videoscore_command[:2] == ["hf", "download"]
    assert "--repo-type" in videoscore_command
    assert videoscore_command[videoscore_command.index("--repo-type") + 1] == "dataset"
    assert "--include" not in videoscore_command
    camerabench_asset = BASE_MODEL_CAPABILITIES["camerabench_dataset_assets"].assets[0]
    assert "--revision" in camerabench_asset.download_command()

    perception_data_plan = base_model_materialization_plan(["spatial_perception_data_stack"])
    assert perception_data_plan["capability_ids"] == [
        "coco2017_detection_segmentation_assets",
        "davis2017_video_segmentation_assets",
        "tum_rgbd_slam_sample_assets",
        "kitti_depth_slam_assets",
    ]
    assert any("images.cocodataset.org/zips/val2017.zip" in command for command in perception_data_plan["download_commands"])
    assert any("DAVIS-2017-trainval-480p.zip" in command for command in perception_data_plan["download_commands"])
    assert any("rgbd_dataset_freiburg1_desk.tgz" in command for command in perception_data_plan["download_commands"])
    assert any("Stage KITTI odometry/depth assets" in action for action in perception_data_plan["manual_actions"])


def test_base_model_materialization_plan_schema_is_stable() -> None:
    plan = base_model_materialization_plan(["detection_segmentation_stack"])

    assert {
        "schema_version",
        "requested_ids",
        "stack_ids",
        "stacks",
        "capability_ids",
        "checks",
        "summary",
        "download_commands",
        "download_command_argvs",
        "export_commands",
        "pip_install_packages",
        "manual_actions",
        "ok",
    } <= set(plan)
    for check in plan["checks"]:
        assert {
            "canonical_owner",
            "owner_path_exists",
            "asset_status",
            "repair_hints",
            "ok",
        } <= set(check)
    assert plan["summary"]["family_counts"]["detection"]["capability_count"] == 1
    assert plan["summary"]["family_counts"]["segmentation"]["capability_count"] == 4
    assert plan["summary"]["family_counts"]["video_object_segmentation"]["capability_count"] == 1
    assert plan["summary"]["asset_counts_by_role"]["checkpoint"] >= 1


def test_base_model_asset_candidates_are_deduplicated() -> None:
    report = check_base_model_dependencies(["droid_slam"])
    candidates = report["checks"][0]["asset_status"][0]["candidate_paths"]

    assert len(candidates) == len(set(candidates))


def test_base_model_inventory_lists_capabilities_and_stacks() -> None:
    inventory = base_model_inventory()

    assert inventory["schema_version"] == "worldfoundry-base-model-inventory-v1"
    assert inventory["capability_count"] == len(BASE_MODEL_CAPABILITIES)
    assert inventory["stack_count"] == len(BASE_MODEL_STACKS)
    assert inventory["asset_count"] >= inventory["required_asset_count"] >= 1
    assert "spatial_perception_core_stack" in {item["id"] for item in inventory["stacks"]}
    assert "depth_anything_v3" in inventory["capabilities_by_family"]["depth"]
    assert "droid_slam" in inventory["capabilities_by_family"]["slam"]
    assert "coco2017_detection_segmentation_assets" in inventory["capabilities_by_family"]["perception_data_asset"]
    assert "detection_stack" in inventory["stacks_by_family"]["open_vocab_detection"]
    assert "spatial_perception_data_stack" in inventory["stacks_by_family"]["perception_data_asset"]
    covered_asset_roles = {
        "benchmark_annotation_dataset",
        "benchmark_dataset",
        "benchmark_source_repo",
        "external_benchmark_asset",
        "external_benchmark_dataset",
        "perception_eval_dataset",
        "reference_video_dataset",
        "simulator_source_repo",
    }
    assert sum(inventory["asset_counts_by_role"].get(role, 0) for role in covered_asset_roles) >= inventory[
        "benchmark_data_asset_count"
    ]
    assert inventory["benchmark_data_asset_count"] == len(formal_benchmark_ids())
    assert inventory["benchmark_data_assets"]["camerabench"] == ["camerabench_dataset_assets"]
    assert inventory["benchmark_data_assets"]["bridgedata-v2"] == [
        "bridgedata_v2_source_assets",
        "bridgedata_v2_dataset_assets",
    ]
    assert inventory["benchmark_data_assets"]["iworld-bench"] == [
        "iworld_bench_source_assets",
        "iworld_bench_dataset_assets",
    ]
    assert inventory["benchmark_data_assets"]["vbench-plus-plus"] == ["vbench_source_assets"]
    assert inventory["benchmark_data_assets"]["worldscore"] == [
        "worldscore_official_assets",
        "worldscore_dataset_assets",
    ]
    assert benchmark_data_asset_capability_ids("chronomagic-bench") == ("chronomagic_dataset_assets",)
    assert benchmark_data_asset_capability_ids("video-bench") == ("video_bench_dataset_assets",)
    assert benchmark_data_asset_capability_ids("libero") == ("libero_source_assets", "libero_dataset_assets")
    assert not [benchmark_id for benchmark_id in formal_benchmark_ids() if not benchmark_data_asset_capability_ids(benchmark_id)]
    assert any(
        asset["min_size_bytes"] is not None
        for item in inventory["capabilities"]
        for asset in item["assets"]
    )
    assert any(
        asset["role"] == "benchmark_dataset"
        for item in inventory["capabilities"]
        for asset in item["assets"]
    )
    assert (
        REPO_ROOT / "worldfoundry/base_models/perception_core/general_perception/open_clip/tokenizer.py"
    ).is_file()
    assert (
        REPO_ROOT / "worldfoundry/data/models/runtime/configs/open_clip/model_configs"
    ).is_dir()
    assert not (
        REPO_ROOT / "worldfoundry/base_models/perception_core/general_perception/open_clip/model_configs"
    ).exists()
    assert (REPO_ROOT / "worldfoundry/base_models/perception_core/segment/sam3").is_dir()


def test_benchmark_base_model_dependency_ids_include_data_assets() -> None:
    ids = benchmark_base_model_dependency_ids(
        "worldscore",
        ["worldscore_spatial_metric_stack"],
        ["spatial_perception_heavy_stack"],
        include_optional=False,
    )

    assert ids == (
        "worldscore_spatial_metric_stack",
        "worldscore_official_assets",
        "worldscore_dataset_assets",
    )
    assert benchmark_base_model_dependency_ids(
        "worldscore",
        ["worldscore_spatial_metric_stack"],
        ["spatial_perception_heavy_stack"],
        include_optional=True,
    ) == (
        "worldscore_spatial_metric_stack",
        "spatial_perception_heavy_stack",
        "worldscore_official_assets",
        "worldscore_dataset_assets",
    )
    assert benchmark_base_model_dependency_ids(
        "worldscore",
        ["worldscore_dataset_assets", "worldscore_spatial_metric_stack"],
        include_data_assets=True,
    ) == (
        "worldscore_dataset_assets",
        "worldscore_spatial_metric_stack",
        "worldscore_official_assets",
    )


def test_spatial_heavy_stack_includes_motion_assets() -> None:
    assert resolve_base_model_capability_ids(["spatial_perception_heavy_stack"]) == [
        "depth_anything_v3",
        "moge_vitl",
        "moge_v2_vitl_normal",
        "unidepth_v2_vitl14",
        "droid_slam",
        "vggt_1b",
        "grounding_dino",
        "sam_v1",
        "sam_vit_b",
        "sam2",
        "sam3",
        "aot_deaot_l",
        "raft",
        "sea_raft",
    ]


def test_base_model_materialization_plan_distinguishes_all_from_empty() -> None:
    all_plan = base_model_materialization_plan(None)
    empty_plan = base_model_materialization_plan([])

    assert all_plan["capability_ids"]
    assert empty_plan["ok"] is True
    assert empty_plan["requested_ids"] == []
    assert empty_plan["capability_ids"] == []
    assert empty_plan["checks"] == []


def test_benchmark_base_model_refs_are_registered_and_consistent() -> None:
    known_ids = {*BASE_MODEL_CAPABILITIES, *BASE_MODEL_STACKS}
    catalog_entries = {}
    for path in iter_benchmark_catalog_manifest_paths(REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"):
        payload = load_manifest(path)
        if isinstance(payload, dict) and payload.get("id"):
            catalog_entries[str(payload["id"])] = payload
    profiles = {
        str(profile["id"]): profile
        for profile in load_benchmark_runtime_profiles()["profiles"]
    }

    missing_dependency_refs = []
    for task_path in sorted(TASK_ROOT.glob("*.yaml")):
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        benchmark_id = task["benchmark"]
        metadata = task.get("metadata") or {}
        catalog_entry = catalog_entries.get(benchmark_id, {})
        profile = profiles.get(benchmark_id, {})
        benchmark_has_dependency_ref = False

        for key in ("base_model_dependencies", "optional_base_model_dependencies"):
            sources = [
                (name, values)
                for name, payload in (
                    ("task", task),
                    ("task.metadata", metadata),
                    ("catalog", catalog_entry),
                    ("runtime_profile", profile),
                )
                if (values := _dependency_tuple(payload, key))
            ]
            for source_name, values in sources:
                unknown = set(values) - known_ids
                assert not unknown, f"{benchmark_id} {source_name} declares unknown {key}: {sorted(unknown)}"
            if sources:
                expected = sources[0][1]
                for source_name, values in sources[1:]:
                    assert values == expected, f"{benchmark_id} {key} differs in {source_name}"
                benchmark_has_dependency_ref = True

        if not benchmark_has_dependency_ref:
            missing_dependency_refs.append(benchmark_id)

    assert missing_dependency_refs == []

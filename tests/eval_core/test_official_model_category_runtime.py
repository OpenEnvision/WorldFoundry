from __future__ import annotations

import json
import importlib
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models import WorldFoundryPipelineRunner, resolve_model_zoo_runner
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.utils import load_manifest_collection
from worldfoundry.synthesis.action_generation.memory import BeingH05Memory
from worldfoundry.synthesis.action_generation.memory import DiffusionPolicyMemory
from worldfoundry.synthesis.action_generation.memory import DreamZeroMemory
from worldfoundry.synthesis.action_generation.memory import GigaBrain0Memory
from worldfoundry.synthesis.action_generation.memory import LingBotVAMemory
from worldfoundry.synthesis.action_generation.memory import OctoMemory
from worldfoundry.synthesis.action_generation.memory import OpenVLAMemory
from worldfoundry.synthesis.action_generation.memory import ActionTraceMemory
from worldfoundry.synthesis.visual_generation.memory.runtime import RuntimeMemory
from worldfoundry.operators.act_operator import ACTOperator
from worldfoundry.operators.base_operator import BaseOperator
from worldfoundry.operators.being_h05_operator import BeingH05Operator
from worldfoundry.operators.diffusion_policy_operator import DiffusionPolicyOperator
from worldfoundry.operators.dreamdojo_operator import DreamDojoOperator
from worldfoundry.operators.dreamzero_operator import DreamZeroOperator
from worldfoundry.operators.giga_brain_0_operator import GigaBrain0Operator
from worldfoundry.operators.gr00t_operator import GR00TOperator
from worldfoundry.operators.lapa_operator import LAPAOperator
from worldfoundry.operators.lingbot_va_operator import LingBotVAOperator
from worldfoundry.operators.octo_operator import OctoOperator
from worldfoundry.operators.openpi_operator import OpenPIOperator
from worldfoundry.operators.openvla_operator import OpenVLAOperator
from worldfoundry.operators.roboflamingo_operator import RoboFlamingoOperator
from worldfoundry.operators.rt1_operator import RT1Operator
from worldfoundry.operators.starvla_operator import StarVLAOperator
from worldfoundry.operators.animatediff_operator import AnimateDiffOperator
from worldfoundry.pipelines.component_pipelines import AnimateDiffPipeline
from worldfoundry.pipelines.component_pipelines import BeingH05Pipeline
from worldfoundry.pipelines.component_pipelines import DiffusionPolicyPipeline
from worldfoundry.pipelines.component_pipelines import DreamDojoPipeline
from worldfoundry.pipelines.component_pipelines import DreamZeroPipeline
from worldfoundry.pipelines.component_pipelines import GigaBrain0Pipeline
from worldfoundry.pipelines.component_pipelines import LingBotVAPipeline
from worldfoundry.pipelines.component_pipelines import OctoPipeline
from worldfoundry.pipelines.component_pipelines import OpenVLAPipeline
from worldfoundry.pipelines.component_pipelines import Splatt3RPipeline
from worldfoundry.synthesis.action_generation.starvla.runtime import (
    StarVLARuntimeConfig,
    build_starvla_plan_payload,
    select_starvla_base_vlm,
    select_starvla_checkpoint,
)
from worldfoundry.synthesis.action_generation.lingbot_va.runtime import (
    LingBotVARuntimeConfig,
    build_server_command as build_lingbot_va_server_command,
)
from worldfoundry.synthesis.action_generation.roboflamingo.runtime import (
    select_roboflamingo_runtime_config,
)
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profiles


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
ACQUISITION_REPO_ROOT = REPO_ROOT / "cache" / "generative_taxonomy" / "repos"
VLA_REPO_ROOT = REPO_ROOT / "cache" / "vla_va_wam" / "repos"


OFFICIAL_ARCHITECTURE_REPO_SPECS = {
    "giga-brain-0": {
        "pipeline_cls": GigaBrain0Pipeline,
        "operator_cls": GigaBrain0Operator,
        "repo_url": "https://github.com/open-gigaai/giga-brain-0",
        "revision": "61a919407dbc2bb945a926cc002749097cc9ebdf",
        "local_dir": ACQUISITION_REPO_ROOT / "open-gigaai--giga-brain-0",
        "task_family": "vla_policy",
        "artifact_kind": "action_trace",
        "generation_type": "vla_policy",
        "policy_family": "world_model_powered_vision_language_action_policy",
        "action_representation": "multi_embodiment_continuous_action_chunk_with_optional_subtask_and_2d_traj",
        "required_paths": (
            "README.md",
            "scripts/inference.py",
            "scripts/inference_task_planning.py",
            "scripts/inference_server.py",
            "giga_brain_0/giga_brain_0_transforms.py",
            "configs/giga_brain_0_agilex_finetune.py",
        ),
        "official_markers": (
            ("README.md", "GigaBrain-0: A World Model-Powered Vision-Language-Action Model"),
            ("scripts/inference.py", "from giga_models import GigaBrain0Pipeline"),
            ("giga_brain_0/giga_brain_0_transforms.py", "class EmbodimentId"),
        ),
    },
    "being-h05": {
        "pipeline_cls": BeingH05Pipeline,
        "operator_cls": BeingH05Operator,
        "repo_url": "https://github.com/BeingBeyond/Being-H",
        "revision": "66b959e05db225fd816f8a517d94a9c0585cfc3e",
        "local_dir": VLA_REPO_ROOT / "BeingBeyond--Being-H",
        "subdir": "Being-H05",
        "task_family": "vla_policy",
        "artifact_kind": "action_trace",
        "generation_type": "vla_policy",
        "policy_family": "beingh05_cross_embodiment_vla_policy",
        "action_representation": "unified_200d_action_space_with_robot_specific_slices",
        "required_paths": (
            "README.md",
            "BeingH/inference/beingh_policy.py",
            "BeingH/model/beingvla.py",
            "docs/unified_action_space.md",
            "docs/training.md",
        ),
        "official_markers": (
            ("README.md", "Being-H uses a **200-dimensional unified action space**"),
            ("docs/training.md", "Mixture of **Action Expert** architecture"),
        ),
    },
    "dreamzero": {
        "pipeline_cls": DreamZeroPipeline,
        "operator_cls": DreamZeroOperator,
        "repo_url": "https://github.com/dreamzero0/dreamzero",
        "revision": "ab790c198fbce33503358efbbd4187ce9a89adf3",
        "local_dir": VLA_REPO_ROOT / "dreamzero0--dreamzero",
        "task_family": "world_action_model",
        "artifact_kind": "action_trace",
        "generation_type": "world_action_model",
        "policy_family": "world_action_model_zero_shot_policy",
        "action_representation": "droid_joint_gripper_action_chunk_plus_predicted_video",
        "required_paths": (
            "README.md",
            "socket_test_optimized_AR.py",
            "test_client_AR.py",
            "eval_utils/policy_server.py",
            "groot/vla/model/__init__.py",
            "docs/WAN22_BACKBONE.md",
        ),
        "official_markers": (
            ("README.md", "World Action Model that jointly predicts actions and videos"),
            ("socket_test_optimized_AR.py", "image_resolution=(180, 320)"),
            ("socket_test_optimized_AR.py", 'action_space="joint_position"'),
        ),
    },
    "dreamdojo": {
        "pipeline_cls": DreamDojoPipeline,
        "operator_cls": DreamDojoOperator,
        "repo_url": "https://github.com/NVIDIA/DreamDojo",
        "revision": "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff",
        "local_dir": ACQUISITION_REPO_ROOT / "NVIDIA--DreamDojo",
        "task_family": "world_model",
        "artifact_kind": "generated_world",
        "world_model_family": "action_conditioned_robot_world_model",
        "action_representation": "384d_unified_robot_action_or_latent_action_chunk",
        "required_paths": (
            "README.md",
            "docs/POSTTRAIN.md",
            "configs/2b_480_640_gr1.yaml",
            "cosmos_predict2/action_conditioned.py",
            "examples/action_conditioned.py",
            "groot_dreams/data/embodiment_tags.py",
        ),
        "official_markers": (
            ("README.md", "interactive world model that learns from large-scale human videos"),
            ("docs/POSTTRAIN.md", "dimension of the first action projection layer to 384"),
            ("configs/2b_480_640_gr1.yaml", "action_dim: 384"),
        ),
    },
    "lingbot-va": {
        "pipeline_cls": LingBotVAPipeline,
        "operator_cls": LingBotVAOperator,
        "repo_url": "https://github.com/Robbyant/lingbot-va",
        "revision": "58c2ae5bac46bd8114065bea9d7d256eb67c16c3",
        "local_dir": VLA_REPO_ROOT / "Robbyant--lingbot-va",
        "task_family": "embodied_action",
        "artifact_kind": "action_trace",
        "generation_type": "embodied_action",
        "policy_family": "autoregressive_video_action_world_model",
        "action_representation": "continuous_robot_action_chunk_with_video_latents",
        "required_paths": (
            "README.md",
            "wan_va/wan_va_server.py",
            "wan_va/configs/va_libero_cfg.py",
            "wan_va/configs/va_robotwin_cfg.py",
            "evaluation/libero/launch_server.sh",
            "evaluation/robotwin/launch_server.sh",
        ),
        "official_markers": (
            ("README.md", "Autoregressive Video-Action World Modeling"),
            ("README.md", "dual-stream mixture-of-transformers"),
            ("wan_va/configs/va_libero_cfg.py", "va_libero_cfg.action_per_frame = 4"),
            ("wan_va/configs/va_robotwin_cfg.py", "va_robotwin_cfg.action_per_frame = 16"),
        ),
    },
}


def _normalize_git_url(value: str) -> str:
    text = value.strip()
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text.split(":", 1)[1]
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/")


def _git_text(repo_dir: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_dir), *args], text=True).strip()


def _is_worldfoundry_pipeline_runner(value) -> bool:
    return isinstance(value, WorldFoundryPipelineRunner)


def test_runtime_profiles_cover_catalog_entries() -> None:
    from worldfoundry.evaluation.models.catalog.schema import iter_model_zoo_payloads
    from worldfoundry.evaluation.utils import load_manifest, manifest_paths

    catalog_root = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
    target_ids: set[str] = set()
    for manifest_path in manifest_paths(catalog_root):
        if manifest_path.name in {"_manifest.yaml", "_DEPRECATED.yaml"}:
            continue
        payload = load_manifest(manifest_path)
        for item in iter_model_zoo_payloads(payload):
            if isinstance(item, Mapping) and item.get("id"):
                target_ids.add(str(item["id"]))
    profiles = load_runtime_profiles()

    assert target_ids.issubset(profiles)
    assert profiles["step-video-t2v"].artifact_kind == "generated_video"
    assert profiles["splatt3r"].artifact_kind == "generated_3d_asset"
    assert profiles["zeroscope"].integration_status == "integrated"
    assert profiles["zeroscope"].runtime_status == "diffusers_text_to_video_ready"
    assert profiles["openvla"].task_family == "vla_policy"
    assert profiles["openvla"].artifact_kind == "action_trace"
    assert profiles["openvla"].integration_status == "integrated"
    assert profiles["openvla"].backend_stage == "in_tree_runtime"
    assert profiles["openvla"].runtime_status == "in_tree_openvla_predict_action_ready"
    assert profiles["openpi"].runtime_status == "in_tree_openpi_pi05_libero_jax_gpu_infer_ready"
    assert (
        profiles["giga-brain-0"].runtime_status
        == "in_tree_giga_brain_0_runtime_converted_lerobot_stats_predict_gpu_ready_non_leaderboard"
    )
    assert profiles["octo"].runtime_status == "in_tree_octo_small_jax_gpu_sample_actions_ready"
    assert profiles["rt-1"].runtime_status == "in_tree_rt1_runtime_ported_savedmodel_checkpoint_missing"
    assert profiles["diffusion-policy"].runtime_status == "in_tree_lowdim_pusht_predict_action_ready"
    assert profiles["lapa"].task_family == "visual_action_model"
    assert profiles["lapa"].artifact_kind == "action_tokens"
    assert profiles["lapa"].runtime_status == "in_tree_lapa_7b_openx_jax_gpu_action_tokens_ready"
    assert profiles["dreamdojo"].task_family == "world_model"
    assert profiles["dreamdojo"].artifact_kind == "generated_world"
    assert profiles["dreamdojo"].runtime_status == "in_tree_dreamdojo_runtime_ported_dataset_and_gpu_parity_pending"
    assert "local_dir" not in profiles["dreamdojo"].source_repos[0]
    assert profiles["dreamdojo"].source_repos[0]["url"] == "https://github.com/NVIDIA/DreamDojo"
    assert profiles["being-h05"].source_repos[0]["subdir"] == "Being-H05"
    assert profiles["dreamzero"].backend_stage == "in_tree_official_server_client"
    assert (
        profiles["dreamzero"].runtime_status
        == "in_tree_official_server_client_checkpoint_gpu_probe_ready_cuda129_multigpu_required"
    )
    assert profiles["dreamdojo"].input_schema["actions"] == [
        "robot_action",
        "unified_384d_action",
        "action_file",
        "dataset_action_sequence",
    ]
    expected_action_profiles = {
        "openvla": ("vla_policy", "action_trace"),
        "openpi": ("vla_policy", "action_trace"),
        "giga-brain-0": ("vla_policy", "action_trace"),
        "being-h05": ("vla_policy", "action_trace"),
        "dreamzero": ("world_action_model", "action_trace"),
        "gr00t": ("vla_policy", "action_trace"),
        "starvla": ("embodied_action", "action_trace"),
        "lingbot-va": ("embodied_action", "action_trace"),
        "lapa": ("visual_action_model", "action_tokens"),
        "octo": ("vla_policy", "action_trace"),
        "rt-1": ("vla_policy", "action_trace"),
        "diffusion-policy": ("visuomotor_policy", "action_trace"),
        "act": ("action_chunking_policy", "action_trace"),
        "roboflamingo": ("vla_policy", "action_trace"),
    }
    for model_id, (task_family, artifact_kind) in expected_action_profiles.items():
        assert profiles[model_id].task_family == task_family
        assert profiles[model_id].artifact_kind == artifact_kind
    assert profiles["cameractrl"].checkpoints[0]["repo_id"] == "hehao13/CameraCtrl"
    assert "local_dir" not in profiles["step-video-t2v"].source_repos[0]
    assert all("local_dir" not in source for profile in profiles.values() for source in profile.source_repos)
    assert not (REPO_ROOT / "worldfoundry" / "pipelines" / "_runtime_profile_pipeline.py").exists()
    assert not (REPO_ROOT / "worldfoundry" / "operators" / "integrated_model_operator.py").exists()
    assert not (REPO_ROOT / "worldfoundry" / "memories" / "visual_synthesis" / "integrated_model_memory.py").exists()
    assert not (REPO_ROOT / "worldfoundry" / "memories" / "action_generation" / "embodied_action").exists()
    assert not (REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "embodied_action").exists()
    assert not (REPO_ROOT / "worldfoundry" / "synthesis" / "action_generation" / "embodied_action").exists()
    assert (REPO_ROOT / "worldfoundry" / "synthesis" / "action_generation" / "openvla").is_dir()
    assert (REPO_ROOT / "worldfoundry" / "synthesis" / "action_generation" / "dreamzero").is_dir()
    action_generation = importlib.import_module("worldfoundry.synthesis.action_generation")
    assert action_generation.__all__ == ["ActionModelSynthesis"]
    assert not hasattr(action_generation, "OpenVLASynthesis")
    assert importlib.import_module("worldfoundry.synthesis.action_generation.openvla").OpenVLASynthesis is not None
    assert importlib.import_module("worldfoundry.synthesis.action_generation.dreamzero").DreamZeroSynthesis is not None
    assert importlib.import_module("worldfoundry.synthesis.action_generation.dreamzero.runtime").run_default_client_demo is not None
    assert not (REPO_ROOT / "worldfoundry" / "evaluation" / "models" / "runtime_profile_runners.py").exists()
    assert not (REPO_ROOT / "worldfoundry" / "evaluation" / "models" / "official_runtime.py").exists()
    assert not (REPO_ROOT / "worldfoundry" / "operators" / "official_runtime_operator.py").exists()
    assert not (REPO_ROOT / "worldfoundry" / "memories" / "visual_synthesis" / "official_runtime").exists()
    assert not (REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "official_runtime").exists()


def test_runtime_profiles_separate_integration_from_official_evidence() -> None:
    payload = load_manifest_collection(
        REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "profiles",
        item_key="profiles",
    )
    profiles = {
        str(item.get("id") or item.get("model_id")): item
        for item in payload["profiles"]
        if item.get("id") or item.get("model_id")
    }

    assert profiles["openvla"]["integration_status"] == "integrated"
    assert profiles["openvla"]["backend_stage"] == "in_tree_runtime"
    assert profiles["openvla"]["verification_status"] == "official_demo_and_in_tree_runtime_ready"
    assert profiles["openvla"]["runtime_status"] == "in_tree_openvla_predict_action_ready"
    for model_id in ("openpi", "lapa", "octo", "rt-1"):
        assert profiles[model_id]["integration_status"] == "integrated"
        assert profiles[model_id]["backend_stage"] == "in_tree_runtime"
        assert "in_tree" in profiles[model_id]["runtime_status"]
    assert profiles["openpi"]["verification_status"] == "official_demo_jax_gpu_ready"
    assert profiles["lapa"]["verification_status"] == "official_demo_jax_gpu_ready"
    assert profiles["octo"]["verification_status"] == "official_demo_jax_gpu_ready"
    for model_id in ("rt-1",):
        assert profiles[model_id]["verification_status"] == "plan_only_ready_checkpoint_required"
    assert profiles["diffusion-policy"]["integration_status"] == "integrated"
    assert profiles["diffusion-policy"]["backend_stage"] == "in_tree_runtime"
    assert profiles["diffusion-policy"]["verification_status"] == "official_demo_ready"
    assert profiles["diffusion-policy"]["runtime_status"] == "in_tree_lowdim_pusht_predict_action_ready"
    assert profiles["being-h05"]["integration_status"] == "integrated"
    assert profiles["being-h05"]["backend_stage"] == "in_tree_runtime"
    assert profiles["being-h05"]["verification_status"] == "checkpoint_backed_action_trace_gpu_ready_efficient_sdpa"
    assert profiles["being-h05"]["runtime_status"] == "in_tree_being_h05_libero_get_action_gpu_ready_efficient_sdpa"
    assert profiles["being-h05"]["gpu_readiness"]["artifact_kind"] == "action_trace"
    assert profiles["giga-brain-0"]["integration_status"] == "integrated"
    assert profiles["giga-brain-0"]["backend_stage"] == "in_tree_runtime"
    assert profiles["giga-brain-0"]["verification_status"] == "converted_lerobot_stats_action_trace_gpu_ready_non_leaderboard"
    assert (
        profiles["giga-brain-0"]["runtime_status"]
        == "in_tree_giga_brain_0_runtime_converted_lerobot_stats_predict_gpu_ready_non_leaderboard"
    )
    assert profiles["giga-brain-0"]["gpu_readiness"]["artifact_kind"] == "action_trace"
    assert profiles["giga-brain-0"]["gpu_readiness"]["official_score_ready"] is False
    assert profiles["starvla"]["integration_status"] == "integrated"
    assert profiles["starvla"]["backend_stage"] == "official_source_runtime_bridge"
    assert profiles["starvla"]["verification_status"] == "checkpoint_backed_official_source_predict_action_gpu_ready"
    assert (
        profiles["starvla"]["runtime_status"]
        == "official_source_runtime_bridge_qwen3_vl_oft_libero_predict_action_gpu_ready"
    )
    assert profiles["starvla"]["gpu_readiness"]["artifact_kind"] == "action_trace"
    assert profiles["starvla"]["gpu_readiness"]["scope"] == "public_pipeline_checkpoint_backed_qwen3_vl_oft_libero_predict_action"
    assert (
        profiles["dreamzero"]["verification_status"]
        == "checkpoint_gpu_probe_ready_runtime_blocked_cuda129_multigpu_required"
    )
    assert profiles["gr00t"]["verification_status"] == "real_predict_ready_local_cosmos_reason2_cuda118_sdpa"


def test_embodied_action_models_use_independent_architecture_operators() -> None:
    operator_specs = [
        ("openvla", OpenVLAOperator, "autoregressive_vision_language_action_policy", "continuous_7d_end_effector_delta"),
        ("openpi", OpenPIOperator, "flow_matching_vision_language_action_policy", "continuous_action_chunk"),
        ("giga-brain-0", GigaBrain0Operator, "world_model_powered_vision_language_action_policy", "multi_embodiment_continuous_action_chunk_with_optional_subtask_and_2d_traj"),
        ("being-h05", BeingH05Operator, "beingh05_cross_embodiment_vla_policy", "unified_200d_action_space_with_robot_specific_slices"),
        ("dreamzero", DreamZeroOperator, "world_action_model_zero_shot_policy", "droid_joint_gripper_action_chunk_plus_predicted_video"),
        ("gr00t", GR00TOperator, "humanoid_foundation_policy", "embodiment_conditioned_joint_or_eef_action"),
        ("starvla", StarVLAOperator, "vision_language_action_and_world_action_model", "robot_or_world_action_sequence"),
        ("lingbot-va", LingBotVAOperator, "autoregressive_video_action_world_model", "continuous_robot_action_chunk_with_video_latents"),
        ("lapa", LAPAOperator, "visual_action_model", "latent_action_tokens"),
        ("octo", OctoOperator, "generalist_robot_transformer_policy", "task_conditioned_action_chunk"),
        ("rt-1", RT1Operator, "discretized_robotics_transformer_policy", "discretized_action_tokens"),
        ("diffusion-policy", DiffusionPolicyOperator, "visuomotor_diffusion_policy", "denoised_action_trajectory"),
        ("act", ACTOperator, "action_chunking_transformer_policy", "chunked_action_sequence"),
        ("roboflamingo", RoboFlamingoOperator, "flamingo_vlm_robot_policy", "continuous_end_effector_action"),
    ]

    modules = set()
    for model_id, operator_cls, policy_family, action_representation in operator_specs:
        operator = operator_cls(input_schema={})
        modules.add(operator_cls.__module__)
        prompt = operator.process_prompt("pick up the mug", instruction="fallback instruction")
        perception = operator.process_perception(
            images="memory://rgb.png",
            video="memory://context.mp4",
            proprio=[0.0] * 7,
            goal_image="memory://goal.png",
            embodiment="mobile_manipulator",
            action_space={"kind": "continuous", "dimensions": 7},
        )
        operator.get_interaction(
            {
                "actions": [{"delta": [0.0] * 7}],
                "latent_action_tokens": [1, 2, 3],
                "action_horizon": 4,
                "track": "wam",
            }
        )
        interaction = operator.process_interaction()
        operator.delete_last_interaction()

        assert prompt["prompt"] == "pick up the mug"
        assert "prompt_channels" in prompt
        assert "observation" in perception
        assert interaction["actions"]
        assert interaction["policy_controls"]
        assert interaction["operator_metadata"]["model_id"] == model_id
        assert interaction["operator_metadata"]["policy_family"] == policy_family
        assert interaction["operator_metadata"]["action_representation"] == action_representation

    assert len(modules) == len(operator_specs)


def test_local_official_repositories_lock_model_architecture_contracts() -> None:
    missing = [
        str(spec["local_dir"])
        for spec in OFFICIAL_ARCHITECTURE_REPO_SPECS.values()
        if not Path(spec["local_dir"]).is_dir()
    ]
    if missing:
        pytest.skip(f"local official model repositories are not staged: {missing}")

    profiles = load_runtime_profiles()
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")

    for model_id, spec in OFFICIAL_ARCHITECTURE_REPO_SPECS.items():
        repo_dir = Path(spec["local_dir"])
        source_root = repo_dir / str(spec.get("subdir") or "")
        profile = profiles[model_id]
        source = dict(profile.source_repos[0])
        pipeline_cls = spec["pipeline_cls"]
        operator_cls = spec["operator_cls"]
        expected_pipeline_target = f"{pipeline_cls.__module__}:{pipeline_cls.__name__}"

        assert _normalize_git_url(_git_text(repo_dir, "remote", "get-url", "origin")) == spec["repo_url"]
        assert _git_text(repo_dir, "rev-parse", "HEAD") == spec["revision"]
        assert _normalize_git_url(source["url"]) == spec["repo_url"]
        assert source.get("revision") == spec["revision"]
        assert "local_dir" not in source
        if spec.get("subdir"):
            assert source.get("subdir") == spec["subdir"]
        assert profile.task_family == spec["task_family"]
        assert profile.artifact_kind == spec["artifact_kind"]

        entry = registry.get(model_id)
        if entry.runner_target is None:
            assert entry.pipeline_target is None
        else:
            assert entry.runner_target == PIPELINE_RUNNER_TARGET
            assert entry.pipeline_target == expected_pipeline_target
        assert pipeline_cls.MODEL_ID == model_id
        assert pipeline_cls.OPERATOR_CLS is operator_cls
        assert pipeline_cls.SYNTHESIS_CLS.MODEL_ID == model_id
        if spec.get("generation_type"):
            assert pipeline_cls.generation_type == spec["generation_type"]
        if spec.get("policy_family"):
            assert operator_cls.POLICY_FAMILY == spec["policy_family"]
        if spec.get("world_model_family"):
            assert operator_cls.WORLD_MODEL_FAMILY == spec["world_model_family"]
        assert operator_cls.ACTION_REPRESENTATION == spec["action_representation"]

        for relative_path in spec["required_paths"]:
            assert (source_root / relative_path).exists(), f"{model_id} missing official path {relative_path}"
        for relative_path, marker in spec["official_markers"]:
            text = (source_root / relative_path).read_text(encoding="utf-8")
            assert marker in text
        if spec.get("readme_only"):
            assert not any(source_root.rglob("*.py"))


def test_dreamdojo_action_contract_matches_official_posttraining_doc() -> None:
    repo_dir = ACQUISITION_REPO_ROOT / "NVIDIA--DreamDojo"
    if not repo_dir.is_dir():
        pytest.skip("local DreamDojo official repository is not staged")

    posttrain_doc = (repo_dir / "docs" / "POSTTRAIN.md").read_text(encoding="utf-8")
    expected_slices = {
        "fourier_gr1": [0, 29],
        "manus_retargeted_gr1": [29, 58],
        "unitree_g1": [58, 101],
        "bimanual_yam": [101, 147],
        "agibot": [147, 169],
        "reserved": [169, 220],
        "mano": [220, 352],
        "latent": [352, 384],
    }

    assert DreamDojoOperator.ACTION_SPACE_DIM == 384
    assert DreamDojoOperator.ACTION_SLICES == expected_slices
    for start, end in expected_slices.values():
        assert f"[{start}, {end})" in posttrain_doc


def test_runtime_profile_plan_strips_external_official_source_workdir(tmp_path: Path) -> None:
    being_h05_pipe = BeingH05Pipeline.from_pretrained({"model_id": "being-h05"}, device="cuda")
    being_h05_result = being_h05_pipe(
        prompt="pick up the mug",
        images="memory://rgb.png",
        interactions={"action_unified": [[0.0] * 200]},
        output_path=tmp_path / "being_h05_action_trace.json",
        plan_only=True,
        return_dict=True,
    )
    being_h05_plan = json.loads(Path(being_h05_result["plan_path"]).read_text(encoding="utf-8"))
    assert being_h05_plan["context"]["source_repo_dir"] == ""
    assert being_h05_plan["context"]["source_repo_subdir"] == "Being-H05"
    assert being_h05_plan["context"]["source_repo_workdir"] == ""
    assert being_h05_plan["runtime"]["attention_mask_kind"] == "dense"

    pipe = DreamZeroPipeline.from_pretrained({"model_id": "dreamzero"}, device="cuda")

    result = pipe(
        prompt="open the drawer",
        images="memory://rgb.png",
        interactions={"joint_position": [0.0] * 7, "gripper_position": [0.0]},
        output_path=tmp_path / "dreamzero_action_trace.json",
        plan_only=True,
        return_dict=True,
    )

    plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))
    source_repos = json.loads(plan["context"]["source_repos_json"])
    assert source_repos[0]["url"] == "https://github.com/dreamzero0/dreamzero"
    assert plan["context"]["source_repo_url"] == "https://github.com/dreamzero0/dreamzero"
    assert plan["context"]["source_repo_revision"] == "ab790c198fbce33503358efbbd4187ce9a89adf3"
    assert "local_dir" not in source_repos[0]
    assert plan["context"]["source_repo_dir"] == ""
    assert plan["context"]["source_repo_workdir"] == ""
    assert plan["profile"]["artifact_kind"] == "action_trace"


def test_starvla_plan_runtime_selects_downloaded_variant_checkpoints(tmp_path: Path) -> None:
    qwen_dir = tmp_path / "StarVLA--Qwen3-VL-OFT-LIBERO-4in1"
    wm4a_dir = tmp_path / "StarVLA--WM4A-Wan2d2-OFT-LIBERO-4in1"
    base_vlm_dir = tmp_path / "Qwen--Qwen3-VL-4B-Instruct"
    source_repo_dir = tmp_path / "starVLA--starVLA"
    (qwen_dir / "checkpoints").mkdir(parents=True)
    (wm4a_dir / "checkpoints").mkdir(parents=True)
    base_vlm_dir.mkdir(parents=True)
    source_repo_dir.mkdir(parents=True)
    (qwen_dir / "checkpoints" / "steps_50000_pytorch_model.pt").write_bytes(b"")
    (wm4a_dir / "checkpoints" / "steps_60000_pytorch_model.pt").write_bytes(b"")

    checkpoints = (
        {
            "repo_id": "StarVLA/Qwen3-VL-OFT-LIBERO-4in1",
            "local_dir": str(qwen_dir),
            "role": "qwen3_vl_libero_policy_checkpoint",
        },
        {
            "repo_id": "StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1",
            "local_dir": str(wm4a_dir),
            "role": "wm4a_wan2d2_libero_world_action_checkpoint",
        },
        {
            "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "local_dir": str(base_vlm_dir),
            "role": "qwen3_vl_4b_base_vlm",
        },
    )

    assert select_starvla_checkpoint(checkpoint_dir=None, checkpoints=checkpoints, track="vla.policy_rollout") == qwen_dir.resolve()
    assert (
        select_starvla_checkpoint(checkpoint_dir=None, checkpoints=checkpoints, variant_id="starvla-wm4a-wan2d2")
        == wm4a_dir.resolve()
    )
    assert select_starvla_checkpoint(checkpoint_dir=None, checkpoints=checkpoints, track="wam.world_action_modeling") == wm4a_dir.resolve()
    assert select_starvla_base_vlm(base_vlm=None, checkpoints=checkpoints) == str(base_vlm_dir.resolve())

    plan = build_starvla_plan_payload(
        config=StarVLARuntimeConfig(
            checkpoint_dir=wm4a_dir.resolve(),
            base_vlm=str(base_vlm_dir.resolve()),
            action_model_type="DiT-B",
            action_dim=7,
            action_horizon=8,
            device="cuda",
            source_repo_dir=source_repo_dir.resolve(),
            track="wam.world_action_modeling",
            attn_implementation="sdpa",
            enable_official_runtime=True,
        ),
        context={},
        profile={},
        runtime_options={},
    )
    assert plan["runtime"]["checkpoint_file"].endswith("steps_60000_pytorch_model.pt")
    assert plan["runtime"]["runtime_package"] == "worldfoundry.synthesis.action_generation.starvla.runtime"
    assert Path(plan["runtime"]["runtime_root"]) == (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/starvla"
    ).resolve()
    assert plan["runtime"]["base_vlm"] == str(base_vlm_dir.resolve())
    assert plan["runtime"]["source_repo_dir"] == str(source_repo_dir.resolve())
    assert plan["runtime"]["official_runtime_enabled"] is True


def test_roboflamingo_runtime_config_expands_hfd_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hfd_root = tmp_path / "hfd"
    policy_dir = hfd_root / "robovlms--RoboFlamingo"
    openflamingo_dir = hfd_root / "openflamingo--OpenFlamingo-3B-vitl-mpt1b-langinstruct"
    mpt_dir = hfd_root / "anas-awadalla--mpt-1b-redpajama-200b-dolly"
    policy_dir.mkdir(parents=True)
    openflamingo_dir.mkdir(parents=True)
    mpt_dir.mkdir(parents=True)
    (policy_dir / "policy.pth").write_bytes(b"policy")
    (openflamingo_dir / "checkpoint.pt").write_bytes(b"openflamingo")
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))

    config = select_roboflamingo_runtime_config(
        checkpoints=[
            {
                "local_dir": "${WORLDFOUNDRY_HFD_ROOT}/robovlms--RoboFlamingo",
                "role": "released_calvin_policy_checkpoints",
            },
            {
                "local_dir": "${WORLDFOUNDRY_HFD_ROOT}/openflamingo--OpenFlamingo-3B-vitl-mpt1b-langinstruct",
                "role": "openflamingo_base_vlm",
            },
            {
                "local_dir": "${WORLDFOUNDRY_HFD_ROOT}/anas-awadalla--mpt-1b-redpajama-200b-dolly",
                "role": "mpt_dolly_language_backbone",
            },
        ],
        options={"device": "cuda:0"},
        require_existing=True,
    )

    assert config.policy_checkpoint_path == (policy_dir / "policy.pth").resolve()
    assert config.openflamingo_checkpoint_path == (openflamingo_dir / "checkpoint.pt").resolve()
    assert config.lang_encoder_path == mpt_dir.resolve()


def test_lingbot_va_server_command_passes_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lingbot-va-posttrain-libero-long"
    checkpoint.mkdir()
    config = LingBotVARuntimeConfig(
        checkpoint_dir=checkpoint,
        config_name="libero",
        host="127.0.0.1",
        port=39536,
        nproc_per_node=1,
        master_port=29061,
    )

    command = build_lingbot_va_server_command(
        python="/env/bin/python",
        torchrun="/env/bin/torchrun",
        config=config,
        save_root=tmp_path / "outputs",
    )

    assert command[command.index("--config-name") + 1] == "libero"
    assert command[command.index("--checkpoint-dir") + 1] == str(checkpoint)


def test_embodied_action_models_use_independent_memory_modules() -> None:
    memory_specs = [
        ("openvla", "worldfoundry.synthesis.action_generation.memory", "OpenVLAMemory", "openvla_records"),
        ("openpi", "worldfoundry.synthesis.action_generation.memory", "OpenPIMemory", "openpi_rollouts"),
        ("giga-brain-0", "worldfoundry.synthesis.action_generation.memory", "GigaBrain0Memory", "giga_brain_0_traces"),
        ("being-h05", "worldfoundry.synthesis.action_generation.memory", "BeingH05Memory", "being_h05_traces"),
        ("dreamzero", "worldfoundry.synthesis.action_generation.memory", "DreamZeroMemory", "dreamzero_rollouts"),
        ("gr00t", "worldfoundry.synthesis.action_generation.memory", "GR00TMemory", "gr00t_action_history"),
        ("starvla", "worldfoundry.synthesis.action_generation.memory", "StarVLAMemory", "starvla_segments"),
        ("lingbot-va", "worldfoundry.synthesis.action_generation.memory", "LingBotVAMemory", "lingbot_va_chunks"),
        ("lapa", "worldfoundry.synthesis.action_generation.memory", "LAPAMemory", "lapa_token_history"),
        ("octo", "worldfoundry.synthesis.action_generation.memory", "OctoMemory", "octo_action_windows"),
        ("rt-1", "worldfoundry.synthesis.action_generation.memory", "RT1Memory", "rt1_token_steps"),
        ("diffusion-policy", "worldfoundry.synthesis.action_generation.memory", "DiffusionPolicyMemory", "diffusion_policy_trajectories"),
        ("act", "worldfoundry.synthesis.action_generation.memory", "ACTMemory", "act_chunks"),
        ("roboflamingo", "worldfoundry.synthesis.action_generation.memory", "RoboFlamingoMemory", "roboflamingo_actions"),
    ]

    classes = set()
    modules = set()
    for model_id, module_name, class_name, storage_attr in memory_specs:
        module = importlib.import_module(module_name)
        memory_cls = getattr(module, class_name)
        memory = memory_cls(capacity=1)
        memory.record({"artifact_path": f"memory://{model_id}/first.json"}, metadata={"type": "action_result"})
        memory.record({"artifact_path": f"memory://{model_id}/latest.json"}, metadata={"type": "action_result"})

        classes.add(memory_cls)
        modules.add(memory_cls.__module__)
        assert issubclass(memory_cls, ActionTraceMemory)
        assert getattr(memory, storage_attr)[-1]["metadata"]["model_id"] == model_id
        assert len(getattr(memory, storage_attr)) == 1
        assert memory.select()["artifact_path"] == f"memory://{model_id}/latest.json"
        assert memory.select(prefer_type="action_result")["artifact_path"] == f"memory://{model_id}/latest.json"
        memory.manage(action="reset")
        assert memory.select() is None

    assert len(classes) == len(memory_specs)
    assert modules == {"worldfoundry.synthesis.action_generation.memory"}


def test_dreamdojo_operator_preserves_robot_world_model_action_contract() -> None:
    operator = DreamDojoOperator(input_schema={})
    prompt = operator.process_prompt("handover the mug", task="gr1 handover")
    perception = operator.process_perception(
        images="memory://initial_rgb.png",
        video="memory://warmup.mp4",
        dataset_path="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot",
        episode_id="demo-episode",
    )
    operator.get_interaction(
        {
            "robot": "gr1",
            "actions": [[0.0] * 384, {"action_file": "memory://actions.json"}],
        }
    )
    interaction = operator.process_interaction()
    operator.delete_last_interaction()

    assert prompt["prompt"] == "handover the mug"
    assert perception["dreamdojo_observation"]["camera_layout"] == "egocentric"
    assert interaction["actions"]
    assert interaction["action_contract"]["dimension"] == 384
    assert interaction["action_contract"]["slices"]["unitree_g1"] == [58, 101]
    assert interaction["operator_metadata"]["model_id"] == "dreamdojo"
    assert interaction["operator_metadata"]["world_model_family"] == "action_conditioned_robot_world_model"


@pytest.mark.parametrize(
    ("pipeline_target", "model_id"),
    [
        # StepVideoT2VPipeline is imported lazily: its module transitively
        # imports optional dependencies (ftfy via the wan text encoders).
        ("worldfoundry.pipelines.step_video.pipeline_step_video_t2v:StepVideoT2VPipeline", "step-video-t2v"),
        ("worldfoundry.pipelines.component_pipelines:Splatt3RPipeline", "splatt3r"),
    ],
)
def test_model_specific_pipeline_stubs_load_without_official_cli_bridge(pipeline_target: str, model_id: str) -> None:
    module_name, _, class_name = pipeline_target.partition(":")
    module = pytest.importorskip(module_name)
    pipeline_cls = getattr(module, class_name)
    pipe = pipeline_cls.from_pretrained({"model_id": model_id}, device="cpu")
    assert isinstance(pipe, pipeline_cls)
    assert pipe.model_id == model_id


def test_splatt3r_uses_model_specific_loader_without_official_cli_bridge(tmp_path: Path) -> None:
    pipe = Splatt3RPipeline.from_pretrained({"model_id": "splatt3r"}, device="cpu")
    result = pipe(
        input_image="memory://rgb.png",
        output_path=tmp_path / "stub.ply",
        plan_only=True,
        return_dict=True,
    )
    assert result["status"] in {"blocked", "success", "prepared"}
    assert result["model_id"] == "splatt3r"
    assert result["runtime"] != "worldfoundry.runtime_profile.vendor_blocked"


def test_examples_mapping_uses_path_a_catalog_targets() -> None:
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")
    expected_targets = {
        "animatediff": "worldfoundry.pipelines.component_pipelines:AnimateDiffPipeline",
        "step-video-t2v": "worldfoundry.pipelines.step_video.pipeline_step_video_t2v:StepVideoT2VPipeline",
        "open-magvit2": "worldfoundry.pipelines.component_pipelines:OpenMAGVIT2Pipeline",
        "show-o": "worldfoundry.pipelines.component_pipelines:ShowOPipeline",
        "cameractrl": "worldfoundry.pipelines.component_pipelines:CameraCtrlPipeline",
        "motionctrl": "worldfoundry.pipelines.component_pipelines:MotionCtrlPipeline",
        "dreamdojo": "worldfoundry.pipelines.component_pipelines:DreamDojoPipeline",
        "irasim": "worldfoundry.pipelines.component_pipelines:IRASimPipeline",
        "pandora": "worldfoundry.pipelines.component_pipelines:PandoraPipeline",
        "splatt3r": "worldfoundry.pipelines.component_pipelines:Splatt3RPipeline",
        "pixelsplat": "worldfoundry.pipelines.component_pipelines:PixelSplatPipeline",
        "openvla": "worldfoundry.pipelines.component_pipelines:OpenVLAPipeline",
        "giga-brain-0": "worldfoundry.pipelines.component_pipelines:GigaBrain0Pipeline",
        "being-h05": "worldfoundry.pipelines.component_pipelines:BeingH05Pipeline",
        "dreamzero": "worldfoundry.pipelines.component_pipelines:DreamZeroPipeline",
        "lingbot-va": "worldfoundry.pipelines.component_pipelines:LingBotVAPipeline",
        "octo": "worldfoundry.pipelines.component_pipelines:OctoPipeline",
        "diffusion-policy": "worldfoundry.pipelines.component_pipelines:DiffusionPolicyPipeline",
        "egowm": "worldfoundry.pipelines.world_model.pipeline_runtime_manifest:EgoWMPipeline",
    }

    for model_id, target in expected_targets.items():
        entry = registry.get(model_id)
        assert entry.runner_target == PIPELINE_RUNNER_TARGET
        assert entry.pipeline_target == target

    resolved = resolve_model_zoo_runner(
        "egowm",
        manifest_dir=REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog",
        runtime={"device": "cpu"},
    )
    assert _is_worldfoundry_pipeline_runner(resolved.runner)


def test_pipeline_runner_targets_are_generic_and_independent() -> None:
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")
    openvla = registry.get("openvla")
    assert openvla.runner_target == PIPELINE_RUNNER_TARGET
    assert openvla.pipeline_target.startswith("worldfoundry.pipelines.")
    assert ".official_runtime" not in openvla.runner_target
    assert ".official_runtime" not in openvla.pipeline_target

    for model_id in ("giga-brain-0", "dreamzero"):
        entry = registry.get(model_id)
        assert entry.runner_target == PIPELINE_RUNNER_TARGET
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")
    for model_id in ("open-magvit2", "splatt3r", "dreamdojo"):
        entry = registry.get(model_id)
        assert entry.runner_target == PIPELINE_RUNNER_TARGET
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")
    egowm = registry.get("egowm")
    assert egowm.runner_target == PIPELINE_RUNNER_TARGET
    assert egowm.pipeline_target == "worldfoundry.pipelines.world_model.pipeline_runtime_manifest:EgoWMPipeline"


def test_model_zoo_entries_are_in_category_manifests_and_resolve_runner(tmp_path: Path) -> None:
    assert not (REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog" / "generative_taxonomy_official_runtime.json").exists()

    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")
    entry = registry.get("open-magvit2")
    assert entry.runner_target == PIPELINE_RUNNER_TARGET
    assert entry.pipeline_target == "worldfoundry.pipelines.component_pipelines:OpenMAGVIT2Pipeline"
    assert entry.runtime_profile == "runtime-profile:open-magvit2"
    assert registry.get("splatt3r").runtime_profile == "runtime-profile:splatt3r"
    assert registry.get("egowm").runtime_profile == "runtime-profile:egowm"
    assert registry.get("dreamdojo").runtime_profile == "runtime-profile:dreamdojo"
    assert registry.get("giga-brain-0").runtime_profile == "runtime-profile:giga-brain-0"
    assert registry.get("dreamzero").runtime_profile == "runtime-profile:dreamzero"

    egowm_resolved = resolve_model_zoo_runner(
        "egowm",
        manifest_dir=REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog",
        runtime={"device": "cpu", "output_dir": str(tmp_path / "egowm")},
    )
    assert _is_worldfoundry_pipeline_runner(egowm_resolved.runner)

    openvla_resolved = resolve_model_zoo_runner(
        "openvla",
        manifest_dir=REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog",
        runtime={"device": "cpu", "output_dir": str(tmp_path / "openvla")},
    )
    assert _is_worldfoundry_pipeline_runner(openvla_resolved.runner)
    openvla_results = openvla_resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="openvla-sample",
                task_name="libero-contract-fixture",
                inputs={"prompt": "pick up the cube", "image": "memory://rgb.png"},
                controls={"actions": [{"delta": [0.0] * 7}]},
                generation_kwargs={"plan_only": True},
            )
        ]
    )
    assert openvla_results[0].status == "prepared"
    assert openvla_results[0].artifacts == {}
    assert openvla_results[0].metadata["artifact_path"].endswith("openvla_action_trace.json")
    assert openvla_results[0].metadata["plan_path"]


def test_vggt_omega_in_tree_integration_is_registered() -> None:
    from worldfoundry.operators.vggt_omega_operator import VGGTOmegaOperator
    from worldfoundry.pipelines.vggt_omega.pipeline_vggt_omega import VGGTOmegaPipeline

    profiles = load_runtime_profiles()
    profile = profiles["vggt-omega"]
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")
    entry = registry.get("vggt-omega")

    assert profile.artifact_kind == "generated_3d_asset"
    assert profile.integration_status == "integrated"
    assert profile.runtime_status == "official_vggt_omega_runtime_small_gpu_parity_passed"
    assert entry.runner_target == PIPELINE_RUNNER_TARGET
    assert entry.pipeline_target == "worldfoundry.pipelines.vggt_omega.pipeline_vggt_omega:VGGTOmegaPipeline"
    assert entry.runtime_profile == "runtime-profile:vggt-omega"
    assert VGGTOmegaPipeline.MODEL_ID == "vggt-omega"
    assert VGGTOmegaPipeline.OPERATOR_CLS is VGGTOmegaOperator
    assert VGGTOmegaPipeline.MEMORY_CLS is RuntimeMemory

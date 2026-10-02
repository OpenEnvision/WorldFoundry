from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.operators import MolmoAct2Operator
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_molmoact2_model_zoo_and_runtime_profile_are_registered() -> None:
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")
    entry = registry.get("molmoact2")
    profile = load_runtime_profile("molmoact2")

    assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
    assert entry.pipeline_target == "worldfoundry.pipelines.component_pipelines:MolmoAct2Pipeline"
    assert entry.runtime_profile == "runtime-profile:molmoact2"
    # The catalog records in_tree_checkpoint_runtime_static_verified, which is
    # a registered in-tree integration state, so the entry is integrated (the
    # remaining GPU/checkpoint validation is tracked by parity fields).
    assert entry.integration_status == "integrated"
    assert "allenai/MolmoAct2-DROID" in entry.hf_repo_ids
    assert "allenai/MolmoAct2-BimanualYAM" in entry.hf_repo_ids
    assert profile.model_id == "molmoact2"
    assert profile.artifact_kind == "action_trace"
    assert profile.backend_stage == "in_tree_runtime"
    assert profile.input_schema["state_dim"] == 8


def test_molmoact2_pipeline_uses_worldfoundry_component_contract() -> None:
    module = importlib.import_module("worldfoundry.pipelines.component_pipelines")
    pipeline_cls = module.MolmoAct2Pipeline

    assert issubclass(pipeline_cls, PipelineABC)
    assert pipeline_cls.MODEL_ID == "molmoact2"
    assert pipeline_cls.OPERATOR_CLS is MolmoAct2Operator
    assert "WorldFoundryNativeExtension" not in {base.__name__ for base in pipeline_cls.__mro__}


def test_molmoact2_plan_only_writes_runtime_plan(tmp_path: Path) -> None:
    from worldfoundry.pipelines.component_pipelines import MolmoAct2Pipeline

    pipeline = MolmoAct2Pipeline.from_pretrained(model_id="molmoact2", device="cpu", plan_only=True)
    result = pipeline(
        prompt="pick up the object",
        images={"external_cam": "external.png", "wrist_cam": "wrist.png"},
        interactions=["noop"],
        output_path=tmp_path / "molmoact2_action_trace.json",
        return_dict=True,
        operator_kwargs={"state": [0.0] * 8},
    )

    assert result["status"] == "prepared"
    assert result["model_id"] == "molmoact2"
    assert result["artifact_kind"] == "action_trace"
    plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))
    assert plan["runtime"]["backend"] == "worldfoundry.molmoact2.in_tree_hf_predict_action"
    assert plan["runtime"]["repo_id"] == "allenai/MolmoAct2-DROID"
    # The DROID contract moved to the official three-camera layout, and dtype
    # selection is deferred to the runtime ("auto") instead of pinned bfloat16.
    assert plan["runtime"]["camera_keys"] == ["external_cam", "external_cam_2", "wrist_cam"]
    assert plan["runtime"]["state_dim"] == 8
    assert plan["runtime"]["torch_dtype"] == "auto"
    assert plan["runtime"]["num_steps"] == 10


def test_molmoact2_plan_supports_yam_contract(tmp_path: Path) -> None:
    from worldfoundry.pipelines.component_pipelines import MolmoAct2Pipeline

    pipeline = MolmoAct2Pipeline.from_pretrained(model_id="molmoact2", device="cpu")
    result = pipeline(
        prompt="fold the towel",
        images={"top_cam": "top.png", "left_cam": "left.png", "right_cam": "right.png"},
        output_path=tmp_path / "yam_action_trace.json",
        return_dict=True,
        plan_only=True,
        variant_id="bimanual-yam",
        operator_kwargs={
            "camera_keys": ["top_cam", "left_cam", "right_cam"],
            "state": [0.0] * 14,
            "state_dim": 14,
        },
    )

    assert result["status"] == "prepared"
    plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))
    assert plan["runtime"]["backend"] == "worldfoundry.molmoact2.in_tree_hf_predict_action"
    assert plan["runtime"]["embodiment"] == "yam"
    assert plan["runtime"]["repo_id"] == "allenai/MolmoAct2-BimanualYAM"
    assert plan["runtime"]["norm_tag"] == "yam_dual_molmoact2"
    assert plan["runtime"]["camera_keys"] == ["top_cam", "left_cam", "right_cam"]
    assert plan["runtime"]["state_dim"] == 14


def test_molmoact2_runtime_config_yaml_can_select_default_variant(tmp_path: Path) -> None:
    from worldfoundry.synthesis.action_generation.molmoact2 import MolmoAct2Synthesis

    config_path = tmp_path / "molmoact2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_variant": "yam",
                "torch_dtype": "float16",
                "num_steps": 6,
                "enable_cuda_graph": False,
                "enable_depth_reasoning": True,
                "normalize_language": False,
                "variants": {
                    "droid": {
                        "embodiment": "droid",
                        "repo_id": "allenai/MolmoAct2-DROID",
                        "norm_tag": "franka_droid",
                        "camera_keys": ["external_cam", "wrist_cam"],
                        "state_dim": 8,
                        "action_mode_key": "action_mode",
                    },
                    "yam": {
                        "aliases": ["bimanual-yam"],
                        "embodiment": "yam",
                        "repo_id": "allenai/MolmoAct2-BimanualYAM",
                        "norm_tag": "yam_dual_molmoact2",
                        "camera_keys": ["top_cam", "left_cam", "right_cam"],
                        "state_dim": 14,
                        "action_mode_key": "inference_action_mode",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    synthesis = MolmoAct2Synthesis.from_pretrained(
        model_id="molmoact2",
        device="cpu",
        runtime_config_path=config_path,
    )
    runtime_config = synthesis._runtime_config({})

    assert runtime_config.embodiment == "yam"
    assert runtime_config.repo_id == "allenai/MolmoAct2-BimanualYAM"
    assert runtime_config.norm_tag == "yam_dual_molmoact2"
    assert runtime_config.camera_keys == ("top_cam", "left_cam", "right_cam")
    assert runtime_config.state_dim == 14
    assert runtime_config.torch_dtype == "float16"
    assert runtime_config.num_steps == 6
    assert runtime_config.enable_depth_reasoning is True
    assert runtime_config.normalize_language is False


def test_molmoact2_operator_maps_yam_camera_aliases() -> None:
    operator = MolmoAct2Operator(input_schema={"camera_keys": ["top_cam", "left_cam", "right_cam"], "norm_tag": "yam_dual_molmoact2"})
    perception = operator.process_perception(
        images=None,
        front_camera_rgb="front",
        left_camera_rgb="left",
        right_camera_rgb="right",
        state=[0.0] * 14,
        embodiment="yam",
    )

    assert perception["images"] == {"top_cam": "front", "left_cam": "left", "right_cam": "right"}
    assert perception["observation"]["state"] == [0.0] * 14
    assert perception["observation"]["norm_tag"] == "yam_dual_molmoact2"

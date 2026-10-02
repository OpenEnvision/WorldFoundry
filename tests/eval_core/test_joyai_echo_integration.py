from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest

from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.catalog import load_model_catalog_manifest
from worldfoundry.evaluation.models.pipelines.bindings import load_pipeline_binding
from worldfoundry.evaluation.models.pipelines.loading import build_pipeline_runner_spec, load_pipeline_from_spec
from worldfoundry.evaluation.models.runtime.environments import load_runtime_environment_profile
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile_manifest
from worldfoundry.synthesis.visual_generation.joyai_echo import (
    JoyAIEchoLongVideoRuntime,
    JoyAIEchoWMRuntime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA = REPO_ROOT / "worldfoundry" / "data" / "models"


def _write(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _import_target(target: str) -> object:
    module_name, attribute = target.split(":", 1)
    return getattr(import_module(module_name), attribute)


@pytest.mark.parametrize(
    ("model_id", "category", "task_family", "pipeline_name"),
    [
        ("joyai-echo-longvideo", "video", "video_generation", "JoyAIEchoLongVideoPipeline"),
        ("joyai-echo-wm", "world_models", "world_model", "JoyAIEchoWMPipeline"),
    ],
)
def test_joyai_echo_catalog_binding_profile_and_environment_are_wired(
    model_id: str,
    category: str,
    task_family: str,
    pipeline_name: str,
) -> None:
    catalog = load_model_catalog_manifest(MODEL_DATA / "catalog" / category / f"{model_id}.yaml")
    binding = load_pipeline_binding(MODEL_DATA / "bindings" / "pipelines" / f"{model_id}.yaml")
    profile = load_runtime_profile_manifest(MODEL_DATA / "runtime" / "profiles" / f"{model_id}.yaml")
    environment = load_runtime_environment_profile(
        MODEL_DATA / "runtime" / "environments" / ("video" if category == "video" else "world") / f"{model_id}.yaml"
    )

    assert catalog.model_id == model_id
    assert catalog.task_family == task_family
    assert catalog.integration["status"] == "integrated"
    assert catalog.integration["pipeline_binding"] == model_id
    assert binding.model_id == model_id
    assert _import_target(binding.pipeline_target).__name__ == pipeline_name
    assert profile.model_id == model_id
    assert profile.task_family == task_family
    assert profile.execution["pipeline_binding"] == model_id
    assert environment.model_id == model_id
    assert environment.env_name == model_id
    assert environment.python == "3.11"


@pytest.mark.parametrize("model_id", ["joyai-echo-longvideo", "joyai-echo-wm"])
def test_joyai_echo_pipeline_loads_from_standard_runner_spec(model_id: str) -> None:
    config = WorldModelConfig(
        model_id=model_id,
        runner="worldfoundry.pipeline",
        runtime={"device": "cpu"},
    )

    spec = build_pipeline_runner_spec(config)
    pipeline = load_pipeline_from_spec(spec)

    assert spec.pipeline_target.startswith("worldfoundry.pipelines.joyai_echo")
    assert pipeline.model_id == model_id


def _longvideo_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "JoyAI-Echo"
    runtime_root = source / "echo_longvideo"
    _write(
        runtime_root / "inference.py",
        """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output-root", required=True)
args, _ = parser.parse_known_args()
request = json.loads(Path(args.request).read_text(encoding="utf-8"))
output = Path(args.output_root) / request["work_id"] / request["shot_id"] / "inference_fixture" / "result.mp4"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b"echo-longvideo-fixture")
""",
    )
    _write(runtime_root / "r2v_schema.py")
    config = _write(runtime_root / "configs" / "inference.bf16.yaml", "paths: {}\n")
    checkpoint = tmp_path / "checkpoints" / "echo15_full_dmd"
    _write(checkpoint / "checkpoint.json", "{}\n")
    _write(checkpoint / "model.safetensors")
    gemma = tmp_path / "checkpoints" / "gemma-3-12b"
    _write(gemma / "config.json", "{}\n")
    return source, config, checkpoint, gemma


def test_longvideo_runtime_builds_official_single_request_command(tmp_path: Path) -> None:
    source, config, checkpoint, gemma = _longvideo_fixture(tmp_path)
    runtime = JoyAIEchoLongVideoRuntime(
        checkpoint_dir=checkpoint,
        gemma_path=gemma,
        source_root=source,
        config_path=config,
        python_executable=Path(sys.executable),
    )
    image = _write(tmp_path / "reference.png")
    output = tmp_path / "result" / "echo.mp4"

    preflight = runtime.preflight()
    assert preflight["missing_checkpoint_files"] == []
    assert preflight["missing_runtime_files"] == []
    plan = runtime.build_plan(
        request={
            "prompt": "A continuous two-shot scene",
            "images": str(image),
            "memory_slots": [],
            "sample_id": "sample-7",
            "num_frames": 193,
            "width": 1280,
            "height": 736,
            "seed": 73,
        },
        output_path=output,
    )

    command = list(plan.command)
    assert plan.project == "echo_longvideo"
    assert command[1] == str(source / "echo_longvideo" / "inference.py")
    assert command[command.index("--request") + 1] == plan.request_path
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--gemma-path") + 1] == str(gemma)
    assert command[command.index("--num-frames") + 1] == "193"
    payload = json.loads(Path(plan.request_path or "").read_text(encoding="utf-8"))
    assert payload["work_id"] == "worldfoundry"
    assert payload["shot_id"] == "sample-7"
    assert payload["condition_img"] == str(image)
    assert payload["num_frames"] == 193
    result = runtime.run_plan(plan, timeout_seconds=10)
    assert result["status"] == "success"
    assert output.read_bytes() == b"echo-longvideo-fixture"


def _wm_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "JoyAI-Echo"
    runtime_root = source / "echo_wm"
    inference_script = """\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args, _ = parser.parse_known_args()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b"echo-wm-fixture")
"""
    _write(runtime_root / "inference_wm.py", inference_script)
    _write(runtime_root / "inference_wm_causal.py", inference_script)
    config = _write(runtime_root / "configs" / "inference_wm_causal.yaml", "model: {}\n")
    checkpoint_root = tmp_path / "checkpoints" / "echo-wm"
    _write(checkpoint_root / "echo-wm-base.safetensors")
    _write(checkpoint_root / "echo-wm-flash.safetensors")
    gemma = tmp_path / "checkpoints" / "gemma-3"
    _write(gemma / "config.json", "{}\n")
    return source, config, checkpoint_root, gemma


def test_echo_wm_flash_runtime_maps_action_and_cache_controls(tmp_path: Path) -> None:
    source, config, checkpoint_root, gemma = _wm_fixture(tmp_path)
    runtime = JoyAIEchoWMRuntime(
        checkpoint_dir=checkpoint_root,
        gemma_path=gemma,
        source_root=source,
        config_path=config,
        python_executable=Path(sys.executable),
        variant="flash",
    )
    image = _write(tmp_path / "input.jpg")
    plan = runtime.build_plan(
        request={
            "prompt": "An explorable crystal cave",
            "images": str(image),
            "interactions": ["l-96", "w-96", "d-96", "w-96"],
            "num_frames": 385,
            "video_local_attn_size": 19,
            "video_sink_size": 7,
            "video_chunk_size": 3,
            "timesteps": [1000, 750, 500, 250],
            "action_overlay": False,
        },
        output_path=tmp_path / "outputs" / "flash.mp4",
    )

    command = list(plan.command)
    assert runtime.preflight()["status"] == "ready"
    assert plan.variant == "flash"
    assert command[1] == str(source / "echo_wm" / "inference_wm_causal.py")
    assert command[command.index("--action-str") + 1] == "l-96,w-96,d-96,w-96"
    assert command[command.index("--checkpoint") + 1].endswith("echo-wm-flash.safetensors")
    assert command[command.index("--video_local_attn_size") + 1] == "19"
    assert command[command.index("--timesteps") + 1 : command.index("--no-action-overlay")] == [
        "1000",
        "750",
        "500",
        "250",
    ]
    result = runtime.run_plan(plan, timeout_seconds=10)
    assert result["status"] == "success"
    assert Path(plan.output_path).read_bytes() == b"echo-wm-fixture"


def test_echo_wm_flash_rejects_invalid_decoded_frame_count(tmp_path: Path) -> None:
    source, config, checkpoint_root, gemma = _wm_fixture(tmp_path)
    runtime = JoyAIEchoWMRuntime(
        checkpoint_dir=checkpoint_root,
        gemma_path=gemma,
        source_root=source,
        config_path=config,
        variant="flash",
    )
    image = _write(tmp_path / "input.jpg")

    with pytest.raises(ValueError, match=r"1 \+ 24m"):
        runtime.build_plan(
            request={
                "prompt": "A world",
                "images": str(image),
                "action_str": "w-60",
                "num_frames": 240,
            },
            output_path=tmp_path / "invalid.mp4",
        )

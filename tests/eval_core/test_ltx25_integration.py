from __future__ import annotations

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
from worldfoundry.synthesis.visual_generation.ltx25 import LTX25DistilledRuntime

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA = REPO_ROOT / "worldfoundry" / "data" / "models"


def _write(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _import_target(target: str) -> object:
    module_name, attribute = target.split(":", 1)
    return getattr(import_module(module_name), attribute)


def _ltx25_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "LTX-2"
    pipelines = source / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines"
    core = source / "packages" / "ltx-core" / "src" / "ltx_core"
    _write(pipelines / "__init__.py", "")
    _write(core / "__init__.py", "")
    _write(
        pipelines / "distilled.py",
        """\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-path", required=True)
args, _ = parser.parse_known_args()
output = Path(args.output_path)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b"ltx-2.5-audio-video-fixture")
""",
    )
    checkpoint = tmp_path / "Lightricks--LTX-2.5"
    for relative in (
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    ):
        _write(checkpoint / relative)
    return source, checkpoint


def test_ltx25_catalog_binding_profile_and_environment_are_wired() -> None:
    catalog = load_model_catalog_manifest(MODEL_DATA / "catalog" / "video" / "ltx-2.5.yaml")
    binding = load_pipeline_binding(MODEL_DATA / "bindings" / "pipelines" / "ltx-2.5.yaml")
    profile = load_runtime_profile_manifest(MODEL_DATA / "runtime" / "profiles" / "ltx-2.5.yaml")
    environment = load_runtime_environment_profile(
        MODEL_DATA / "runtime" / "environments" / "video" / "ltx-2.5.yaml"
    )

    assert catalog.model_id == "ltx-2.5"
    assert catalog.task_family == "video_generation"
    assert catalog.integration["status"] == "integrated"
    assert catalog.integration["pipeline_binding"] == "ltx-2.5"
    assert binding.model_id == "ltx-2.5"
    assert _import_target(binding.pipeline_target).__name__ == "LTX25Pipeline"
    assert profile.model_id == "ltx-2.5"
    assert profile.task_family == "video_generation"
    assert profile.execution["pipeline_binding"] == "ltx-2.5"
    assert environment.model_id == "ltx-2.5"
    assert environment.env_name == "ltx-2.5"
    assert environment.python == "3.11"


def test_ltx25_pipeline_loads_from_standard_runner_spec() -> None:
    config = WorldModelConfig(
        model_id="ltx-2.5",
        runner="worldfoundry.pipeline",
        runtime={"device": "cpu"},
    )

    spec = build_pipeline_runner_spec(config)
    pipeline = load_pipeline_from_spec(spec)

    assert spec.pipeline_target == "worldfoundry.pipelines.ltx25.pipeline_ltx25:LTX25Pipeline"
    assert pipeline.model_id == "ltx-2.5"
    assert pipeline.preflight()["status"] == "blocked"


def test_ltx25_runtime_maps_split_components_and_i2v_controls(tmp_path: Path) -> None:
    source, checkpoint = _ltx25_fixture(tmp_path)
    runtime = LTX25DistilledRuntime(
        checkpoint_dir=checkpoint,
        source_root=source,
        python_executable=Path(sys.executable),
        offload_mode="disk",
        quantization="fp8-cast",
        device="cuda:2",
    )
    image = _write(tmp_path / "reference.png")
    output = tmp_path / "outputs" / "ltx25.mp4"

    preflight = runtime.preflight()
    assert preflight["status"] == "ready"
    assert preflight["missing_checkpoint_files"] == []
    assert preflight["missing_runtime_files"] == []
    plan = runtime.build_plan(
        request={
            "prompt": "A violinist performs while the camera slowly circles",
            "images": {"path": str(image), "frame_index": 0, "strength": 0.85, "crf": 18},
            "num_frames": 121,
            "height": 512,
            "width": 768,
            "fps": 24,
            "seed": 73,
        },
        output_path=output,
    )

    command = list(plan.command)
    assert plan.task == "image-to-video"
    assert command[:3] == [str(Path(sys.executable).resolve()), "-m", "ltx_pipelines.distilled"]
    assert command[command.index("--transformer-path") + 1].endswith(
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    assert command[command.index("--text-encoder-path") + 1].endswith(
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    )
    assert command[command.index("--offload") + 1] == "disk"
    assert command[command.index("--quantization") + 1] == "fp8-cast"
    assert command[command.index("--image") + 1 : command.index("--image") + 5] == [
        str(image),
        "0",
        "0.85",
        "18",
    ]
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "2"
    result = runtime.run_plan(plan, timeout_seconds=10)
    assert result["status"] == "success"
    assert output.read_bytes() == b"ltx-2.5-audio-video-fixture"


def test_ltx25_runtime_supports_text_only_and_auto_duration(tmp_path: Path) -> None:
    source, checkpoint = _ltx25_fixture(tmp_path)
    runtime = LTX25DistilledRuntime(
        checkpoint_dir=checkpoint,
        source_root=source,
        python_executable=Path(sys.executable),
        device="cuda",
    )

    plan = runtime.build_plan(
        request={"prompt": "Ocean waves at dawn", "num_frames": None, "auto_duration": [3.0, 6.0]},
        output_path=tmp_path / "t2v.mp4",
    )
    command = list(plan.command)

    assert plan.task == "text-to-video"
    assert "--image" not in command
    assert "--num-frames" not in command
    assert command[command.index("--auto-duration") + 1 : command.index("--auto-duration") + 3] == [
        "3.0",
        "6.0",
    ]
    assert "--duration-head-path" in command


def test_ltx25_pipeline_requires_explicit_execution() -> None:
    from worldfoundry.pipelines.ltx25 import LTX25Pipeline

    pipeline = LTX25Pipeline.from_pretrained(device="cpu")

    with pytest.raises(RuntimeError, match="execute=True"):
        pipeline(prompt="A quiet forest")

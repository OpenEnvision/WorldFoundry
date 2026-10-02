from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import GenieEnvisionerPipeline
from worldfoundry.synthesis.visual_generation.genie_envisioner import infer
from worldfoundry.synthesis.visual_generation.genie_envisioner import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation.world_model import runtime_manifest


def _write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[Path]]:
    source = tmp_path / "runtime"
    for relative in (
        "models/ltx_models/autoencoder_kl_ltx.py",
        "models/ltx_models/transformer_ltx_multiview.py",
        "models/pipeline/custom_pipeline.py",
    ):
        _write_fixture(source / relative)

    checkpoint = (
        tmp_path
        / "ckpts"
        / "agibot-world--Genie-Envisioner"
        / "GE_base_fast_v0.1.safetensors"
    )
    _write_fixture(checkpoint)
    base = tmp_path / "ckpts" / "Lightricks--LTX-Video"
    for relative in (
        "tokenizer/tokenizer_config.json",
        "tokenizer/spiece.model",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        _write_fixture(base / relative)
    image = tmp_path / "input.png"
    Image.new("RGB", (40, 20), "blue").save(image)
    views = []
    for index, color in enumerate(("red", "green", "blue")):
        view = tmp_path / f"view-{index}.png"
        Image.new("RGB", (40, 20), color).save(view)
        views.append(view)
    return source, checkpoint, base, image, views


def test_runtime_spec_uses_executable_genie_envisioner_adapter() -> None:
    spec = runtime_manifest.runtime_spec("genie-envisioner")

    assert spec.runtime_module.endswith("genie_envisioner.worldfoundry_runtime")
    assert spec.runtime_root_func == "runtime_root"
    assert spec.blocked_reason == ""
    assert spec.input_schema["video"] is False


def test_missing_requirements_is_empty_for_complete_local_assets(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, image, views = _stage_runtime(tmp_path)
    launcher = tmp_path / "infer.py"
    _write_fixture(launcher)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={
            "checkpoint_path": str(checkpoint),
            "base_model_dir": str(base),
            "image_path": str(image),
            "input_mode": "explicit-three-view",
            "input_views": [str(view) for view in views],
        },
        runtime_root=source,
        entrypoint=launcher,
        profile=None,
    )

    assert missing == []


def test_build_command_expands_paths_and_plan_options(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, image, views = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "genie-envisioner.json"
    plan_path.write_text(
        json.dumps(
            {
                "extra": {
                    "image_path": str(image),
                    "input_mode": "explicit-three-view",
                    "input_views": [str(view) for view in views],
                    "num_inference_steps": 2,
                    "latent_chunk": 1,
                    "seed": 7,
                }
            }
        ),
        encoding="utf-8",
    )

    command = runtime.build_command(
        {
            "python": "python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "runtime_root": str(source),
            "output_path": str(output_dir / "genie-envisioner.mp4"),
            "plan_path": str(plan_path),
            "prompt": "move the robot",
            "device": "cuda",
            "options": {
                "checkpoint_path": (
                    "${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/"
                    "GE_base_fast_v0.1.safetensors"
                ),
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
            },
        }
    )

    assert command[command.index("--checkpoint-path") + 1] == str(checkpoint)
    assert command[command.index("--base-model-dir") + 1] == str(base)
    view_index = command.index("--input-views")
    assert command[view_index + 1 : view_index + 4] == [str(view) for view in views]
    assert "--input-image" not in command
    assert command[command.index("--num-inference-steps") + 1] == "2"
    assert command[command.index("--latent-chunk") + 1] == "1"
    assert command[command.index("--seed") + 1] == "7"
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_pipeline_promotes_image_path_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(GenieEnvisionerPipeline)

    promoted = pipeline._promote_call_options({"images": tmp_path / "input.png"})

    assert promoted["image_path"] == str(tmp_path / "input.png")


def test_pipeline_promotes_three_reference_paths_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(GenieEnvisionerPipeline)
    views = [tmp_path / f"view-{index}.png" for index in range(3)]

    promoted = pipeline._promote_call_options({"reference_image_paths": views})

    assert promoted["input_views"] == [str(view) for view in views]
    assert promoted["input_mode"] == "explicit-three-view"


def test_synthetic_views_are_deterministic(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.fromarray(np.arange(24 * 40 * 3, dtype=np.uint8).reshape(24, 40, 3)).save(image)

    views_a = infer._synthetic_three_views(image, height=192, width=256)
    views_b = infer._synthetic_three_views(image, height=192, width=256)

    np.testing.assert_array_equal(views_a, views_b)
    assert views_a.shape == (3, 192, 256, 3)
    assert infer._transformer_config()["num_layers"] == 28
    assert infer._transformer_config()["in_channels"] == 128


def test_explicit_views_preserve_three_distinct_camera_inputs(tmp_path) -> None:
    paths = []
    for index, value in enumerate((20, 100, 220)):
        path = tmp_path / f"view-{index}.png"
        Image.fromarray(np.full((24, 40, 3), value, dtype=np.uint8)).save(path)
        paths.append(path)

    views = infer._explicit_three_views(paths, height=192, width=256)

    assert views.shape == (3, 192, 256, 3)
    assert [int(view.mean()) for view in views] == [20, 100, 220]


def test_launcher_imports_current_diffusers_runtime() -> None:
    import importlib

    infer._ensure_vendored_runtime_importable()
    pipeline_module = importlib.import_module("models.pipeline.custom_pipeline")
    transformer_module = importlib.import_module("models.ltx_models.transformer_ltx_multiview")

    assert pipeline_module.CustomPipeline.__name__ == "CustomPipeline"
    assert transformer_module.LTXVideoTransformer3DModel.__name__ == "LTXVideoTransformer3DModel"


def test_launcher_rejects_cpu_execution(tmp_path) -> None:
    args = infer._parser().parse_args(
        [
            "--checkpoint-path",
            str(tmp_path),
            "--base-model-dir",
            str(tmp_path),
            "--input-image",
            str(tmp_path),
            "--output-path",
            str(tmp_path / "out.mp4"),
            "--prompt",
            "move",
            "--device",
            "cpu",
        ]
    )

    try:
        infer._validate_args(args)
    except ValueError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("CPU execution was accepted")

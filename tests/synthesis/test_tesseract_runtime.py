from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import TesserActPipeline
from worldfoundry.synthesis.action_generation.tesseract import infer
from worldfoundry.synthesis.action_generation.tesseract import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation import runtime_manifest


def _write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "runtime"
    for relative in (
        "tesseract/modules/tesseract_pipeline.py",
        "tesseract/modules/tesseract_model.py",
        "tesseract/utils.py",
    ):
        _write_fixture(source / relative)

    checkpoint = tmp_path / "ckpts" / "anyeZHY--tesseract" / "tesseract_v01e_rgbdn_sft"
    _write_fixture(checkpoint / "config.json")
    (checkpoint / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "diffusion_pytorch_model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    _write_fixture(checkpoint / "diffusion_pytorch_model-00001-of-00001.safetensors")

    base = tmp_path / "ckpts" / "THUDM--CogVideoX-5b-I2V"
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/spiece.model",
        "text_encoder/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        _write_fixture(base / relative)
    (base / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    _write_fixture(base / "text_encoder" / "model-00001-of-00001.safetensors")

    image = tmp_path / "input.png"
    _write_fixture(image)
    return source, checkpoint, base, image


def test_runtime_spec_uses_executable_tesseract_adapter() -> None:
    spec = runtime_manifest.runtime_spec("tesseract")

    assert spec.runtime_module == "worldfoundry.synthesis.action_generation.tesseract.worldfoundry_runtime"
    assert spec.runtime_root_func == "runtime_root"
    assert spec.blocked_reason == ""
    assert spec.input_schema["video"] is False


def test_missing_requirements_is_empty_for_complete_local_assets(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, image = _stage_runtime(tmp_path)
    launcher = tmp_path / "infer.py"
    _write_fixture(launcher)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={
            "checkpoint_dir": str(checkpoint),
            "base_model_dir": str(base),
            "image_path": str(image),
        },
        runtime_root=source,
        entrypoint=launcher,
        profile=None,
    )

    assert missing == []


def test_missing_requirements_rejects_half_of_official_geometry_pair(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, image = _stage_runtime(tmp_path)
    launcher = tmp_path / "infer.py"
    depth = tmp_path / "depth.npy"
    _write_fixture(launcher)
    _write_fixture(depth)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={
            "checkpoint_dir": str(checkpoint),
            "base_model_dir": str(base),
            "image_path": str(image),
            "depth_path": str(depth),
        },
        runtime_root=source,
        entrypoint=launcher,
        profile=None,
    )

    assert any("both depth_path and normal_path" in item["reason"] for item in missing)


def test_build_command_expands_paths_and_plan_options(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, image = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "tesseract.json"
    plan_path.write_text(
        json.dumps(
            {
                "extra": {
                    "image_path": str(image),
                    "num_inference_steps": 2,
                    "num_frames": 9,
                    "seed": 7,
                    "memory_efficient": False,
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
            "output_path": str(output_dir / "tesseract.mp4"),
            "plan_path": str(plan_path),
            "prompt": "move the robot",
            "device": "cuda",
            "options": {
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
            },
        }
    )

    assert command[command.index("--checkpoint-dir") + 1] == str(checkpoint)
    assert command[command.index("--base-model-dir") + 1] == str(base)
    assert command[command.index("--input-image") + 1] == str(image)
    assert command[command.index("--num-inference-steps") + 1] == "2"
    assert command[command.index("--num-frames") + 1] == "9"
    assert command[command.index("--seed") + 1] == "7"
    assert "--memory-efficient" not in command
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_pipeline_promotes_image_path_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(TesserActPipeline)

    promoted = pipeline._promote_call_options({"images": tmp_path / "input.png"})

    assert promoted["image_path"] == str(tmp_path / "input.png")


def test_launcher_bootstraps_the_vendored_runtime_import_path(monkeypatch) -> None:
    runtime_root = str(Path(infer.__file__).resolve().parent / "tesseract_runtime")
    monkeypatch.setattr(infer.sys, "path", [item for item in infer.sys.path if item != runtime_root])

    infer._ensure_vendored_runtime_importable()

    assert infer.sys.path[0] == runtime_root


def test_synthetic_geometry_is_deterministic_and_bounded() -> None:
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    depth_a, normal_a = infer._synthetic_gradient_geometry(rgb)
    depth_b, normal_b = infer._synthetic_gradient_geometry(rgb)

    np.testing.assert_array_equal(depth_a, depth_b)
    np.testing.assert_array_equal(normal_a, normal_b)
    assert depth_a.shape == (4, 5)
    assert normal_a.shape == (4, 5, 3)
    assert float(depth_a.min()) >= 0.0 and float(depth_a.max()) <= 1.0
    assert float(normal_a.min()) >= 0.0 and float(normal_a.max()) <= 1.0


def test_patch_embedding_uses_current_diffusers_tensor_positional_api() -> None:
    from worldfoundry.synthesis.action_generation.tesseract.tesseract_runtime.tesseract.modules.embeddings import (
        TesserActDepthPatchEmbed,
    )

    embedding = TesserActDepthPatchEmbed(
        patch_size=2,
        in_channels=4,
        embed_dim=16,
        text_embed_dim=8,
        sample_width=4,
        sample_height=4,
        sample_frames=5,
        temporal_compression_ratio=4,
        max_text_seq_length=2,
        use_positional_embeddings=True,
        use_learned_positional_embeddings=False,
    )

    assert embedding.pos_embedding.shape == (1, 10, 16)


def test_launcher_rejects_cpu_execution(tmp_path) -> None:
    args = infer._parser().parse_args(
        [
            "--checkpoint-dir",
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

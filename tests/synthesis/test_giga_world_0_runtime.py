from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import GigaWorld0Pipeline
from worldfoundry.synthesis.visual_generation.giga_world_0 import infer
from worldfoundry.synthesis.visual_generation.giga_world_0 import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation import runtime_manifest


def _write_fixture(path: Path, payload: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    source = tmp_path / "runtime"
    _write_fixture(source / "scripts" / "inference.py")

    transformer = tmp_path / "ckpts" / "gr1" / "transformer"
    _write_fixture(
        transformer / "config.json",
        json.dumps(
            {
                "_class_name": "GigaWorld0Transformer3DModel",
                "in_channels": 17,
                "out_channels": 16,
                "natten_parameters": [{"window_size": [-1, 4, 16]}],
            }
        ).encode(),
    )
    _write_fixture(transformer / "diffusion_pytorch_model.safetensors")

    text_encoder = tmp_path / "ckpts" / "t5-encoder"
    for relative in ("config.json", "spiece.model", "model.safetensors"):
        _write_fixture(text_encoder / relative)

    vae = tmp_path / "ckpts" / "wan-diffusers" / "vae"
    for relative in ("config.json", "diffusion_pytorch_model.safetensors"):
        _write_fixture(vae / relative)

    image = tmp_path / "input.png"
    Image.new("RGB", (40, 20), "blue").save(image)
    return source, transformer, text_encoder, vae, image


def test_runtime_spec_uses_executable_giga_world_adapter() -> None:
    spec = runtime_manifest.runtime_spec("giga-world-0")

    assert spec.runtime_module.endswith("giga_world_0.worldfoundry_runtime")
    assert spec.runtime_root_func == "runtime_root"
    assert spec.blocked_reason == ""
    assert spec.input_schema["actions"] == []


def test_missing_requirements_is_empty_for_complete_local_assets(tmp_path, monkeypatch) -> None:
    source, transformer, text_encoder, vae, image = _stage_runtime(tmp_path)
    launcher = tmp_path / "infer.py"
    _write_fixture(launcher)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={
            "transformer_model_dir": str(transformer),
            "text_encoder_model_dir": str(text_encoder),
            "vae_model_dir": str(vae),
            "image_path": str(image),
            "attention_backend": "torch",
        },
        runtime_root=source,
        entrypoint=launcher,
        profile=None,
    )

    assert missing == []


def test_build_command_expands_paths_and_plan_options(tmp_path, monkeypatch) -> None:
    source, transformer, text_encoder, vae, image = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "giga-world-0.json"
    plan_path.write_text(
        json.dumps(
            {
                "extra": {
                    "image_path": str(image),
                    "num_inference_steps": 2,
                    "num_frames": 5,
                    "attention_backend": "torch",
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
            "output_path": str(output_dir / "giga-world-0.mp4"),
            "plan_path": str(plan_path),
            "prompt": "move the robot",
            "device": "cuda",
            "options": {
                "transformer_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/gr1/transformer",
                "text_encoder_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/t5-encoder",
                "vae_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/wan-diffusers/vae",
            },
        }
    )

    assert command[command.index("--transformer-model-dir") + 1] == str(transformer)
    assert command[command.index("--text-encoder-model-dir") + 1] == str(text_encoder)
    assert command[command.index("--vae-model-dir") + 1] == str(vae)
    assert command[command.index("--input-image") + 1] == str(image)
    assert command[command.index("--num-inference-steps") + 1] == "2"
    assert command[command.index("--num-frames") + 1] == "5"
    assert command[command.index("--attention-backend") + 1] == "torch"
    assert command[command.index("--seed") + 1] == "7"
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_pipeline_promotes_image_path_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(GigaWorld0Pipeline)

    promoted = pipeline._promote_call_options({"images": tmp_path / "input.png"})

    assert promoted["image_path"] == str(tmp_path / "input.png")


def test_torch_backend_removes_only_natten_configuration(tmp_path) -> None:
    _, transformer, _, _, _ = _stage_runtime(tmp_path)

    torch_config = infer._transformer_config(transformer, attention_backend="torch")
    natten_config = infer._transformer_config(transformer, attention_backend="natten")

    assert torch_config["natten_parameters"] is None
    assert natten_config["natten_parameters"] == [{"window_size": [-1, 4, 16]}]
    assert torch_config["in_channels"] == natten_config["in_channels"] == 17


def test_launcher_rejects_cpu_execution(tmp_path) -> None:
    args = infer._parser().parse_args(
        [
            "--transformer-model-dir",
            str(tmp_path),
            "--text-encoder-model-dir",
            str(tmp_path),
            "--vae-model-dir",
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

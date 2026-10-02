from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import CtrlWorldPipeline
from worldfoundry.synthesis.visual_generation.ctrl_world import infer
from worldfoundry.synthesis.visual_generation.ctrl_world import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation import runtime_manifest


def _write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, list[Path]]:
    source = tmp_path / "runtime"
    for relative in (
        "models/ctrl_world.py",
        "models/pipeline_ctrl_world.py",
        "models/unet_spatio_temporal_condition.py",
    ):
        _write_fixture(source / relative)

    checkpoint = tmp_path / "ckpts" / "yjguo--Ctrl-World" / "checkpoint-10000.pt"
    _write_fixture(checkpoint)
    base = tmp_path / "ckpts" / "stabilityai--stable-video-diffusion-img2vid"
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "feature_extractor/preprocessor_config.json",
        "image_encoder/config.json",
        "image_encoder/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ):
        _write_fixture(base / relative)
    clip = tmp_path / "ckpts" / "openai--clip-vit-base-patch32"
    for relative in ("config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json", "merges.txt"):
        _write_fixture(clip / relative)
    image = tmp_path / "input.png"
    Image.new("RGB", (40, 20), "blue").save(image)
    views = []
    for index, color in enumerate(("red", "green", "blue")):
        view = tmp_path / f"view-{index}.png"
        Image.new("RGB", (40, 20), color).save(view)
        views.append(view)
    return source, checkpoint, base, clip, image, views


def test_runtime_spec_uses_executable_ctrl_world_adapter() -> None:
    spec = runtime_manifest.runtime_spec("ctrl-world")

    assert spec.runtime_module == "worldfoundry.synthesis.visual_generation.ctrl_world.worldfoundry_runtime"
    assert spec.runtime_root_func == "runtime_root"
    assert spec.blocked_reason == ""
    assert spec.input_schema["video"] is False


def test_missing_requirements_is_empty_for_complete_local_assets(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, clip, image, views = _stage_runtime(tmp_path)
    launcher = tmp_path / "infer.py"
    _write_fixture(launcher)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={
            "checkpoint_path": str(checkpoint),
            "base_model_dir": str(base),
            "clip_model_dir": str(clip),
            "image_path": str(image),
            "input_mode": "explicit-three-view",
            "input_views": [str(view) for view in views],
            "initial_pose": [0.4632655382, 0.0169253815, 0.4484287798, 3.1247090285, 0.0323393448, 0.0202563482, 0.0],
        },
        runtime_root=source,
        entrypoint=launcher,
        profile=None,
    )

    assert missing == []


def test_build_command_expands_paths_and_plan_options(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, clip, image, views = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "ctrl-world.json"
    plan_path.write_text(
        json.dumps(
            {
                "extra": {
                    "image_path": str(image),
                    "input_mode": "explicit-three-view",
                    "input_views": [str(view) for view in views],
                    "num_inference_steps": 2,
                    "action_direction": "left",
                    "initial_pose": [0.4632655382, 0.0169253815, 0.4484287798, 3.1247090285, 0.0323393448, 0.0202563482, 0.0],
                    "action_distance": 0.08,
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
            "output_path": str(output_dir / "ctrl-world.mp4"),
            "plan_path": str(plan_path),
            "prompt": "move the robot",
            "device": "cuda",
            "options": {
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                "clip_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
            },
        }
    )

    assert command[command.index("--checkpoint-path") + 1] == str(checkpoint)
    assert command[command.index("--base-model-dir") + 1] == str(base)
    assert command[command.index("--clip-model-dir") + 1] == str(clip)
    view_index = command.index("--input-views")
    assert command[view_index + 1 : view_index + 4] == [str(view) for view in views]
    assert "--input-image" not in command
    assert command[command.index("--num-inference-steps") + 1] == "2"
    assert command[command.index("--action-direction") + 1] == "left"
    assert command[command.index("--action-mode") + 1] == "absolute-pose"
    assert command[command.index("--action-distance") + 1] == "0.08"
    assert len(command[command.index("--initial-pose") + 1 : command.index("--action-distance")]) == 7
    assert "--action-scale" not in command
    assert command[command.index("--seed") + 1] == "7"
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_build_command_keeps_legacy_actions_explicit_smoke_only(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"extra": {"action_mode": "synthetic-zero-smoke", "action_scale": 0.0, "input_mode": "synthetic-three-view"}}))
    command = runtime.build_command({
        "python": "python", "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
        "output_path": str(tmp_path / "out.mp4"), "plan_path": str(plan_path),
        "prompt": "smoke", "device": "cuda", "options": {},
    })
    assert command[command.index("--action-mode") + 1] == "synthetic-zero-smoke"
    assert command[command.index("--action-scale") + 1] == "0.0"
    assert "--initial-pose" not in command


def test_pipeline_promotes_image_path_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(CtrlWorldPipeline)

    promoted = pipeline._promote_call_options({"images": tmp_path / "input.png"})

    assert promoted["image_path"] == str(tmp_path / "input.png")


def test_pipeline_promotes_three_reference_paths_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(CtrlWorldPipeline)
    views = [tmp_path / f"view-{index}.png" for index in range(3)]

    promoted = pipeline._promote_call_options({"reference_image_paths": views})

    assert promoted["input_views"] == [str(view) for view in views]
    assert promoted["input_mode"] == "explicit-three-view"


def test_synthetic_views_and_actions_are_deterministic(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.fromarray(np.arange(24 * 40 * 3, dtype=np.uint8).reshape(24, 40, 3)).save(image)

    views_a = infer._synthetic_three_views(image, height=192, width=320)
    views_b = infer._synthetic_three_views(image, height=192, width=320)
    actions = infer._normalized_action_rollout("right", 0.2)

    np.testing.assert_array_equal(views_a, views_b)
    assert views_a.shape == (3, 192, 320, 3)
    assert tuple(actions.shape) == (1, 11, 7)
    assert np.isclose(float(actions[0, -1, 1]), 0.2)
    assert float(actions.abs().max()) <= 1.0


def test_absolute_pose_actions_match_official_cold_start_keyboard_contract() -> None:
    pose = np.array([0.4632655382, 0.0169253815, 0.4484287798, 3.1247090285, 0.0323393448, 0.0202563482, 0.0])
    stats = json.loads(infer.OFFICIAL_ACTION_STATS.read_text())
    lower = np.array(stats["state_01"])
    upper = np.array(stats["state_99"])
    for direction, dim, delta in (("left", 1, -0.08), ("stationary", 1, 0), ("right", 1, 0.08)):
        expected_physical = np.repeat(pose[None, :], 11, axis=0)
        expected_physical[6:, dim] += np.arange(5) * delta / 4
        expected = np.clip(2 * (expected_physical - lower) / (upper - lower + 1e-8) - 1, -1, 1)
        actual = infer._absolute_pose_action_rollout(pose.tolist(), direction, 0.08).numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        np.testing.assert_allclose(actual[:6], np.repeat(actual[0:1], 6, axis=0), rtol=0, atol=0)


def test_absolute_pose_uses_official_special_keyboard_trajectory() -> None:
    import importlib

    infer._ensure_vendored_runtime_importable()
    key_board_control = importlib.import_module("models.utils").key_board_control
    pose = np.array([0.6571552157, -0.2333021909, 0.1489311606, 3.0244400501, -0.1095049530, -0.5373510122, 0.0])
    official_future = key_board_control(pose[None, :], "d", distance=0.08)
    stats = json.loads(infer.OFFICIAL_ACTION_STATS.read_text())
    lower = np.array(stats["state_01"])
    upper = np.array(stats["state_99"])
    expected = np.clip(2 * (official_future - lower) / (upper - lower + 1e-8) - 1, -1, 1)
    actual = infer._absolute_pose_action_rollout(pose.tolist(), "down", 0.08).numpy()[0, 6:]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


def test_absolute_pose_requires_measured_start_state(tmp_path, monkeypatch) -> None:
    source, checkpoint, base, clip, image, views = _stage_runtime(tmp_path)
    launcher = tmp_path / "infer.py"
    _write_fixture(launcher)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())
    options = {
        "checkpoint_path": str(checkpoint), "base_model_dir": str(base), "clip_model_dir": str(clip),
        "input_mode": "explicit-three-view", "input_views": [str(view) for view in views],
    }
    missing = runtime.missing_requirements(options=options, runtime_root=source, entrypoint=launcher, profile=None)
    assert any(item["path"] == "initial_pose" for item in missing)
    options["action_mode"] = "synthetic-zero-smoke"
    assert runtime.missing_requirements(options=options, runtime_root=source, entrypoint=launcher, profile=None) == []
    options["initial_pose"] = [0.4, 0.0, 0.4, 3.1, 0.0, 0.0, 0.0]
    assert any(item["path"] == "initial_pose" for item in runtime.missing_requirements(
        options=options, runtime_root=source, entrypoint=launcher, profile=None
    ))


def test_explicit_views_preserve_three_distinct_camera_inputs(tmp_path) -> None:
    paths = []
    for index, value in enumerate((20, 100, 220)):
        path = tmp_path / f"view-{index}.png"
        Image.fromarray(np.full((24, 40, 3), value, dtype=np.uint8)).save(path)
        paths.append(path)

    views = infer._explicit_three_views(paths, height=192, width=320)

    assert views.shape == (3, 192, 320, 3)
    assert [int(view.mean()) for view in views] == [20, 100, 220]


def test_launcher_imports_current_diffusers_runtime() -> None:
    import importlib

    infer._ensure_vendored_runtime_importable()
    pipeline_module = importlib.import_module("models.pipeline_ctrl_world")
    model_module = importlib.import_module("models.ctrl_world")

    assert pipeline_module.CtrlWorldDiffusionPipeline.__name__ == "CtrlWorldDiffusionPipeline"
    assert model_module.CrtlWorld.__name__ == "CrtlWorld"


def test_launcher_rejects_cpu_execution(tmp_path) -> None:
    args = infer._parser().parse_args(
        [
            "--checkpoint-path",
            str(tmp_path),
            "--base-model-dir",
            str(tmp_path),
            "--clip-model-dir",
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

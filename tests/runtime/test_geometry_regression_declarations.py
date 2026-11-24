# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Declared family replays retain real inputs, generated outputs and native commands."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]
_CASES = json.loads((_ROOT / "tests/manual/geometry_regression_cases.json").read_text())
_DEPENDENCIES = json.loads((_ROOT / "tests/manual/geometry_regression_dependencies.json").read_text())
_CASE_IDS = sorted(
    name
    for name in _CASES
    if name.endswith("-checkpoint-replay")
    or name
    in {
        "hma-paired-trajectory-short",
        "uniworld-view-single-view-short",
        "last-r1-libero10-actions-short",
        "dino-wm-wall-planned-actions",
        "leworldmodel-pusht-learned-policy",
        "solaris-translation-two-player",
        "starwm-generated-state-text",
    }
)


def _manual_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tests/manual" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_policy = _manual_module("native_coverage_policy")


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_family_declaration_requires_generated_outputs_and_declared_inputs(case_id: str) -> None:
    case = _CASES[case_id]
    assert case["id"] == case_id
    assert case["requires_gpu"] is True
    assert case["deterministic"] is True
    assert isinstance(case["seed"], int) and not isinstance(case["seed"], bool)
    assert case["call"] and not case["call"].get("plan_only")
    assert "env" not in case
    assert case["required_outputs"]
    assert not any("metadata" in output or "runtime_plan" in output for output in case["required_outputs"])
    assert _policy.has_replay_recipe(case) is (case["model_id"] != "starwm")
    assert _DEPENDENCIES["cases"][case_id]
    assert all(owner in _DEPENDENCIES["components"] for owner in _DEPENDENCIES["cases"][case_id])
    json_inputs = {item["path"] for item in case.get("input_json", {}).values()}
    for name, value in case.get("asset_sha256", {}).items():
        assert name in case["assets"]
        operator_pin = re.fullmatch(r"\$\{WORLDFOUNDRY_REGRESSION_[A-Z0-9_]+_SHA256\}", value)
        asset_path = case["assets"][name]
        pinned_asset = asset_path.startswith("${WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT}/") or bool(
            re.fullmatch(r"\$\{WORLDFOUNDRY_[A-Z0-9_]+_REPLAY_[A-Z0-9_]+\}", asset_path)
        )
        release_pin = pinned_asset and re.fullmatch(r"[0-9a-f]{64}", value)
        assert operator_pin or release_pin
    for path in case["assets"].values():
        if path.startswith("${WORLDFOUNDRY_REGRESSION_INPUT_ROOT}") and Path(path).suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".json",
            ".jsonl",
            ".mp4",
            ".wav",
            ".npz",
            ".npy",
            ".txt",
            ".csv",
            ".h5",
            ".pth",
            ".pt",
            ".gz",
        }:
            assert path in json_inputs or any(case["assets"][key] == path for key in case.get("asset_sha256", {}))


@pytest.mark.parametrize(
    "case_id,checkpoint_flag,checkpoint_key,seed_flag",
    [
        ("hma-paired-trajectory-short", "--checkpoint-dir", "checkpoint_path", "--seed"),
        ("diamond-checkpoint-replay", "--pretrained-dir", "model_path", "--seed"),
        ("diamond-atari-checkpoint-replay", "--pretrained-dir", "model_path", "--seed"),
        ("dino-wm-wall-planned-actions", "--ckpt-base-path", "checkpoint_path", "--seed"),
        ("giga-world-0-checkpoint-replay", "--transformer-model-dir", "checkpoint_path", "--seed"),
        ("leworldmodel-pusht-learned-policy", "--policy", "model_path", "--seed"),
        ("tesseract-checkpoint-replay", "--checkpoint-dir", "checkpoint_path", "--seed"),
    ],
)
def test_declared_manifest_checkpoint_and_seed_reach_the_native_command(
    tmp_path: Path, case_id: str, checkpoint_flag: str, checkpoint_key: str, seed_flag: str
) -> None:
    replacements = {
        "WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT": str(tmp_path / "weights"),
        "WORLDFOUNDRY_REGRESSION_INPUT_ROOT": str(tmp_path / "inputs"),
        "WORLDFOUNDRY_REGRESSION_LEWORLDMODEL_POLICY_PATH": str(tmp_path / "learned_object.ckpt"),
        "OUTPUT_DIR": str(tmp_path),
    }
    text = json.dumps(_CASES[case_id])
    for key, value in replacements.items():
        text = text.replace("${" + key + "}", value)
    case = json.loads(text)
    module, _, name = case["target"].partition(":")
    pipeline = getattr(importlib.import_module(module), name).from_pretrained(**{**case["load"], "device": "cpu"})
    call = pipeline._promote_call_options(dict(case["call"]))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"extra": call}))
    command = pipeline.synthesis_model._command(
        output_path=Path(call["output_path"]), prompt=call.get("prompt", ""), plan_path=plan_path
    )
    assert command[command.index(checkpoint_flag) + 1] == case["load"][checkpoint_key]
    assert command[command.index(seed_flag) + 1] == str(case["seed"])
    if case["model_id"].startswith("diamond"):
        assert command[command.index("--dataset-dir") + 1] == case["assets"]["dataset"]
        assert command[command.index("--pretrained-game") + 1] == "Pong"
    if case["model_id"] == "hma":
        assert command[command.index("--trajectory-json") + 1] == call["trajectory_path"]
        assert "--input-image" not in command
    if case["model_id"] in {"giga-world-0", "tesseract"}:
        assert command[command.index("--input-image") + 1] == call["images"]
    if case["model_id"] == "tesseract":
        assert command[command.index("--geometry-mode") + 1] == "files"
        assert command[command.index("--depth-path") + 1] == case["assets"]["depth"]
        assert command[command.index("--normal-path") + 1] == case["assets"]["normal"]


def test_starwm_is_a_pinned_endpoint_declaration_not_local_checkpoint_coverage() -> None:
    case = _CASES["starwm-generated-state-text"]
    assert not {"model_path", "pretrained_model_path", "checkpoint_path"}.intersection(case["load"])
    assert case["load"]["served_model_id"] == "yxzhang2024/StarWM"
    assert case["load"]["seed"] == case["seed"] == 42
    assert "server_provenance" in case["asset_sha256"]
    assert case["result_outputs"] == []
    assert case["text_outputs"] == {"prediction.json": ["output"]}
    assert case["required_outputs"] == ["text.prediction.json.output"]
    assert not _policy.has_replay_recipe(case)


def test_action_replays_select_computed_actions_not_runtime_metadata() -> None:
    last = _CASES["last-r1-libero10-actions-short"]
    assert last["result_outputs"] == ["actions", "normalized_actions", "action_token_ids"]
    assert last["required_outputs"] == ["result." + field for field in last["result_outputs"]]
    dino = _CASES["dino-wm-wall-planned-actions"]
    assert dino["result_outputs"] == []
    assert dino["output_arrays"] == ["planned_actions.npy"]
    assert dino["required_outputs"] == ["artifact.planned_actions.npy"]
    learned = _CASES["leworldmodel-pusht-learned-policy"]
    assert learned["export_artifacts"] == ["**/*.mp4"]
    assert learned["result_outputs"] == []


@pytest.mark.parametrize(
    "case_id,field,value",
    [
        ("minwm-wan-action2v-checkpoint-replay", "num_output_frames", 20),
        ("rolling-forcing-checkpoint-replay", "num_output_frames", 126),
        ("multiworld-checkpoint-replay", "inference_seed", 0),
        ("matrix-game-1-checkpoint-replay", "execute", True),
        ("uniworld-view-single-view-short", "plan_only", False),
        ("hma-paired-trajectory-short", "prompt_horizon", 3),
        ("hma-paired-trajectory-short", "generated_frames", 6),
    ],
)
def test_declared_native_frame_units_execution_and_zero_seed(case_id: str, field: str, value: Any) -> None:
    assert _CASES[case_id]["call"][field] == value


@pytest.mark.parametrize("case_id", ["magi2-preview-checkpoint-replay", "minimax-h3-checkpoint-replay"])
def test_audiovisual_declarations_require_generated_soundtracks(case_id: str) -> None:
    case = _CASES[case_id]
    assert case["audio_outputs"] == ["video.mp4"]
    assert {"audio.video.mp4." + field for field in ("samples", "sample_rate", "channels", "layout")} <= set(
        case["required_outputs"]
    )

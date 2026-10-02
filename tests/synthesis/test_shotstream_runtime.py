from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from worldfoundry.synthesis.visual_generation import runtime_manifest
from worldfoundry.synthesis.visual_generation.shotstream import infer
from worldfoundry.synthesis.visual_generation.shotstream import worldfoundry_runtime as runtime


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "sources" / "ShotStream"
    source.mkdir(parents=True)
    (source / "Inference_Causal.py").write_text("", encoding="utf-8")

    checkpoint = tmp_path / "ckpts" / "KlingTeam--ShotStream"
    checkpoint.mkdir(parents=True)
    for name in ("default_config.yaml", "shotstream.yaml", "shotstream_merged.pt"):
        (checkpoint / name).write_bytes(b"fixture")

    wan = tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-1.3B"
    (wan / "google" / "umt5-xxl").mkdir(parents=True)
    for name in (
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ):
        (wan / name).write_bytes(b"fixture")
    return source, checkpoint, wan


def test_runtime_spec_uses_dynamic_adapter() -> None:
    spec = runtime_manifest.runtime_spec("shotstream")

    assert spec.runtime_module == (
        "worldfoundry.synthesis.visual_generation.shotstream.worldfoundry_runtime"
    )
    assert spec.runtime_root_func == "runtime_root"
    assert spec.blocked_reason == ""


def test_runtime_synthesis_uses_profile_artifact_kind(tmp_path) -> None:
    spec = runtime_manifest.runtime_spec("shotstream")
    synthesis = runtime_manifest.WorldModelRuntimeSynthesis(
        spec=spec,
        runtime_root=tmp_path,
        entrypoint=runtime.OFFICIAL_ENTRYPOINT,
        blocked_reason="",
        profile=SimpleNamespace(artifact_kind="generated_video", checkpoints=(), required_env=()),
    )

    assert synthesis._artifact_kind() == "generated_video"


def test_missing_requirements_is_empty_when_runtime_is_staged(tmp_path, monkeypatch) -> None:
    source, checkpoint, wan = _stage_runtime(tmp_path)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={"checkpoint_dir": str(checkpoint), "wan_model_dir": str(wan)},
        runtime_root=source,
        entrypoint=runtime.OFFICIAL_ENTRYPOINT,
        profile=None,
    )

    assert missing == []


def test_missing_requirements_reports_both_checkpoint_families(tmp_path, monkeypatch) -> None:
    source = tmp_path / "sources" / "ShotStream"
    source.mkdir(parents=True)
    (source / "Inference_Causal.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={
            "checkpoint_dir": str(tmp_path / "missing-shotstream"),
            "wan_model_dir": str(tmp_path / "missing-wan"),
        },
        runtime_root=source,
        entrypoint=runtime.OFFICIAL_ENTRYPOINT,
        profile=None,
    )

    reasons = " ".join(item["reason"] for item in missing)
    assert "ShotStream checkpoint asset" in reasons
    assert "Wan2.1-T2V-1.3B base-model asset" in reasons


def test_build_command_expands_portable_paths_and_call_options(tmp_path, monkeypatch) -> None:
    source, checkpoint, wan = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "shotstream.json"
    plan_path.write_text(json.dumps({"extra": {"seed": 7}}), encoding="utf-8")

    command = runtime.build_command(
        {
            "python": "python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "runtime_root": str(source),
            "output_path": str(output_dir / "shotstream.mp4"),
            "output_dir": str(output_dir),
            "prompt": "a moving robot",
            "plan_path": str(plan_path),
            "options": {
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                "wan_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
            },
        }
    )

    assert command[command.index("--checkpoint-dir") + 1] == str(checkpoint)
    assert command[command.index("--wan-model-dir") + 1] == str(wan)
    assert command[command.index("--prompt") + 1] == "a moving robot"
    assert command[command.index("--seed") + 1] == "7"
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_prompt_fixture_matches_official_csv_contract(tmp_path) -> None:
    csv_path = infer._write_prompt_fixture("a drifting spacecraft", tmp_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "shot_num_from_caption": "1",
            "json_path": str(tmp_path / "input" / "prompt.json"),
            "frame_number": "[[0, 81]]",
        }
    ]
    payload = json.loads((tmp_path / "input" / "prompt.json").read_text(encoding="utf-8"))
    assert payload["global_caption"] == "a drifting spacecraft"
    assert payload["shot1"]


def test_input_csv_rewrite_resolves_relative_json_paths(tmp_path) -> None:
    source_dir = tmp_path / "dataset"
    source_dir.mkdir()
    (source_dir / "prompt.json").write_text("{}", encoding="utf-8")
    source_csv = source_dir / "input.csv"
    source_csv.write_text(
        'shot_num_from_caption,json_path,frame_number\n1,prompt.json,"[[0, 81]]"\n',
        encoding="utf-8",
    )

    rewritten = infer._rewrite_input_csv(source_csv, tmp_path / "rewritten.csv")

    with rewritten.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["json_path"] == str((source_dir / "prompt.json").resolve())

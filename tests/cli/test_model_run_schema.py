from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.cli import model_run
from worldfoundry.cli.main import main
from worldfoundry.cli.model_run import load_model_run_schema


@pytest.mark.parametrize("model_id", ("animatediff", "zeroscope"))
def test_text_to_video_schema_does_not_inject_an_image_input(model_id: str) -> None:
    schema = load_model_run_schema(model_id)

    assert all(field.option != "--pipeline.input-path" for field in schema.fields)


@pytest.mark.parametrize("task_id", ("i2v", "v2v"))
def test_cosmos3_conditioned_task_rejects_missing_input_path(
    task_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["run", "cosmos3", "--pipeline.task-profile", task_id, "--plan-only"])

    assert exit_code == 2
    assert "direct cosmos3 inference requires --pipeline.input-path" in capsys.readouterr().err


@pytest.mark.parametrize(("task_id", "input_key"), (("i2v", "image"), ("v2v", "video")))
def test_cosmos3_conditioned_task_resolves_provided_input_path(
    task_id: str,
    input_key: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / f"{task_id}-conditioning-input"
    input_path.touch()

    exit_code = main(
        [
            "run",
            "cosmos3",
            "--pipeline.task-profile",
            task_id,
            "--input-path",
            str(input_path),
            "--plan-only",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    input_field = next(field for field in payload["model"]["fields"] if field["option"] == "--pipeline.input-path")
    assert input_field["required"] is True
    assert payload["resolved"]["inputs"][input_key] == str(input_path)


def test_cosmos3_default_text_to_video_does_not_require_input_path(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "cosmos3", "--plan-only", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(field["option"] != "--pipeline.input-path" for field in payload["model"]["fields"])
    assert set(payload["resolved"]["inputs"]) == {"prompt"}


def test_invalid_runtime_profile_is_reported(tmp_path, monkeypatch) -> None:
    path = tmp_path / "animatediff.yaml"
    path.write_text("models: [", encoding="utf-8")
    monkeypatch.setattr(model_run, "_runtime_profile_paths_by_stem", lambda: {"animatediff": (path,)})

    from yaml import YAMLError

    with pytest.raises(YAMLError):
        load_model_run_schema.__wrapped__("animatediff")


def test_catalog_without_runtime_profile_uses_metadata(monkeypatch) -> None:
    monkeypatch.setattr(model_run, "_runtime_profile_paths_by_stem", lambda: {})

    assert model_run._load_catalog_runtime_profile(SimpleNamespace(model_id="custom"), None) == (None, "custom")

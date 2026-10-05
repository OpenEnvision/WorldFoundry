from __future__ import annotations

from types import SimpleNamespace

import pytest

from worldfoundry.cli import model_run
from worldfoundry.cli.model_run import load_model_run_schema


@pytest.mark.parametrize("model_id", ("animatediff", "zeroscope"))
def test_text_to_video_schema_does_not_inject_an_image_input(model_id: str) -> None:
    schema = load_model_run_schema(model_id)

    assert all(field.option != "--pipeline.input-path" for field in schema.fields)


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

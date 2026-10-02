from __future__ import annotations

import pytest

from worldfoundry.cli.model_run import load_model_run_schema


@pytest.mark.parametrize("model_id", ("animatediff", "zeroscope"))
def test_text_to_video_schema_does_not_inject_an_image_input(model_id: str) -> None:
    schema = load_model_run_schema(model_id)

    assert all(field.option != "--pipeline.input-path" for field in schema.fields)

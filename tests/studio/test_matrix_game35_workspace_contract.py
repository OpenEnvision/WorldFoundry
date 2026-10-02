from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.studio.inference.catalog import find_entry


@pytest.mark.parametrize(
    ("model_id", "checkpoint_name"),
    (
        ("matrix-game-3.5-first-person", "first-person.safetensors"),
        ("matrix-game-3.5-third-person", "third-person.safetensors"),
    ),
)
def test_matrix_game35_workspace_defaults_are_complete(
    model_id: str,
    checkpoint_name: str,
) -> None:
    entry = find_entry(model_id)
    components = entry.default_load_kwargs["required_components"]

    assert Path(entry.default_model_ref).name == checkpoint_name
    assert Path(entry.default_model_ref).is_file()
    assert Path(components["wan_dir"]).is_dir()
    assert Path(components["da3_dir"]).is_dir()
    assert Path(components["python_executable"]).is_file()
    assert Path(entry.default_input_path).is_file()
    assert entry.default_call_kwargs["steps"] == 25
    assert entry.default_call_kwargs["num_blocks"] == 1
    assert len(entry.default_call_kwargs["camera"]["extrinsics_c2w"]) == 86

"""Regression coverage for the Uni3C stage-two Workspace contract."""

from pathlib import Path

import pytest
import torch

from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.uni3c.uni3c_runtime.src import xfuser_compat
from worldfoundry.synthesis.visual_generation.uni3c.uni3c_runtime.src.models.rotary_compat import (
    apply_rotary_embedding_bhld,
    split_rotary_embedding,
)


def test_uni3c_exposes_required_stage_one_inputs_and_full_defaults() -> None:
    spec = get_model_inference_spec("uni3c")
    assert spec is not None
    task = spec.task()
    fields = {field.field_id: field for field in task.inputs}

    assert fields["input_path"].required is True
    assert Path(fields["input_path"].default).name == "reference.jpg"
    assert fields["render_path"].required is True
    assert Path(fields["render_path"].default).name == "synthetic-forward-9f-768x480"
    assert fields["num_frames"].default == 9
    assert fields["fps"].default == 16
    assert fields["seed"].default == 0
    assert fields["max_area"].default == 114688


def test_uni3c_catalog_forwards_stage_two_runtime_inputs() -> None:
    entry = find_entry("uni3c")

    assert entry.default_task_type == "image-to-video"
    assert "render_path" in entry.call_params
    assert "controller_path" in entry.load_params
    assert "base_model_path" in entry.load_params
    assert Path(entry.default_input_path).name == "reference.jpg"
    assert Path(entry.default_input_path).is_file()
    assert Path(entry.default_call_kwargs["render_path"]).is_dir()
    assert entry.default_call_kwargs["num_frames"] == 9
    assert "white cat" in entry.default_prompt.lower()
    assert "sunglasses" in entry.default_prompt.lower()
    assert "swimming pool" in entry.default_prompt.lower()
    assert "pushes forward" in entry.default_prompt.lower()
    assert "surfboard" not in entry.default_prompt.lower()


def test_uni3c_splits_current_diffusers_rotary_tuple_along_sequence() -> None:
    rotary = (torch.arange(32).reshape(1, 8, 1, 4), torch.arange(32).reshape(1, 8, 1, 4))

    split = split_rotary_embedding(rotary, chunks=4, rank=2)

    assert isinstance(split, tuple)
    assert split[0].shape == (1, 2, 1, 4)
    torch.testing.assert_close(split[0], rotary[0][:, 4:6])


def test_uni3c_applies_current_diffusers_rotary_tuple_to_bhld_tensor() -> None:
    hidden_states = torch.randn(1, 2, 3, 4)
    identity_rotary = (torch.ones(1, 3, 1, 4), torch.zeros(1, 3, 1, 4))

    rotated = apply_rotary_embedding_bhld(hidden_states, identity_rotary)

    torch.testing.assert_close(rotated, hidden_states)


def test_uni3c_single_gpu_fallback_does_not_require_xfuser(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = ModuleNotFoundError("No module named 'xfuser'", name="xfuser")
    monkeypatch.setattr(xfuser_compat, "_XFUSER_IMPORT_ERROR", missing)

    assert xfuser_compat.get_sequence_parallel_rank() == 0
    assert xfuser_compat.get_sequence_parallel_world_size() == 1
    with pytest.raises(RuntimeError, match="requires a working xfuser/FlashAttention"):
        xfuser_compat.require_xfuser()

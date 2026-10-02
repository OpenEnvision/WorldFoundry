"""The combined Being-H0.5 checkpoint stores variants inside its dataset key."""

import json
from types import SimpleNamespace

import pytest

from worldfoundry.synthesis.action_generation.being_h05 import policy as policy_module


def _policy(monkeypatch, variant):
    selected = []
    instance = object.__new__(policy_module.BeingHPolicy)
    instance.dataset_name = "uni_posttrain"
    instance.metadata_variant = variant
    instance.stats_selection_mode = "auto"
    instance._modality_transform = SimpleNamespace(set_metadata=selected.append)
    monkeypatch.setattr(
        policy_module.DatasetMetadata,
        "model_validate",
        staticmethod(lambda value: value),
    )
    return instance, selected


def test_nested_combined_metadata_honors_explicit_variant(monkeypatch, tmp_path):
    (tmp_path / "uni_posttrain_metadata.json").write_text(
        json.dumps({"uni_posttrain": {"libero_spatial": {"id": 1}, "libero_10": {"id": 2}}})
    )
    instance, selected = _policy(monkeypatch, "libero_10")

    instance._load_metadata(tmp_path)

    assert selected == [{"id": 2}]
    assert instance.stats_source == "legacy:libero_10"


def test_nested_combined_metadata_rejects_unknown_explicit_variant(monkeypatch, tmp_path):
    (tmp_path / "uni_posttrain_metadata.json").write_text(
        json.dumps({"uni_posttrain": {"libero_spatial": {"id": 1}}})
    )
    instance, selected = _policy(monkeypatch, "libero_10")

    with pytest.raises(ValueError, match="Metadata variant 'libero_10' not found"):
        instance._load_metadata(tmp_path)
    assert selected == []

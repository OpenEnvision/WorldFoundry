from __future__ import annotations

import pytest

from worldfoundry.evaluation.models.catalog.schema import ModelZooEntry
from worldfoundry.evaluation.models.catalog.zoo_registry import (
    ModelZooRegistry,
    UnknownModelZooKeyError,
)


def _entry(model_id: str, name: str) -> ModelZooEntry:
    return ModelZooEntry.from_dict(
        {
            "id": model_id,
            "name": name,
            "aliases": [f"{model_id}-alias"],
            "integration": {"status": "integrated"},
            "checkpoint_refs": [
                {"repo_id": "example/shared-checkpoints", "filename": f"{model_id}.safetensors"}
            ],
        }
    )


def test_shared_hf_repository_is_not_an_ambiguous_model_alias() -> None:
    first = _entry("first-recipe", "First recipe")
    second = _entry("second-recipe", "Second recipe")
    registry = ModelZooRegistry([first])

    assert registry.get("example/shared-checkpoints").model_id == "first-recipe"

    registry.register(second)

    assert registry.get("first-recipe-alias").model_id == "first-recipe"
    assert registry.get("second-recipe-alias").model_id == "second-recipe"
    assert "example/shared-checkpoints" not in registry.aliases_for("first-recipe")
    assert "example/shared-checkpoints" not in registry.aliases_for("second-recipe")
    with pytest.raises(UnknownModelZooKeyError):
        registry.get("example/shared-checkpoints")

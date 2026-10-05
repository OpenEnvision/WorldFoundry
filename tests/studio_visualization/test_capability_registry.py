from __future__ import annotations

import json

from worldfoundry.cli.main import main
from worldfoundry.studio.visualization.capability_registry import (
    MODEL_VISUALIZATION_CAPABILITIES,
    RENDER_BACKENDS,
    visualization_inventory,
)


def test_visualization_registry_is_unique_and_points_to_in_tree_sources():
    model_ids = [item.model_id for item in MODEL_VISUALIZATION_CAPABILITIES]
    assert len(model_ids) == len(set(model_ids))
    assert len(model_ids) == 65

    inventory = visualization_inventory()
    assert all(item["package_ready"] for item in inventory["models"])
    assert all(item["evidence_ready"] for item in inventory["models"])
    assert all(renderer in RENDER_BACKENDS for item in MODEL_VISUALIZATION_CAPABILITIES for renderer in item.renderers)


def test_visualization_inventory_filters_families_and_cli_emits_json(capsys):
    three_d = visualization_inventory(family="three_dimensions")
    perception = visualization_inventory(family="perception_core")
    assert three_d["model_count"] + perception["model_count"] == 65
    assert three_d["model_count"] > 0 and perception["model_count"] > 0

    assert main(["models", "visualizations", "--model", "raft", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_count"] == 1
    assert payload["models"][0]["renderers"] == ["flow"]
    assert payload["models"][0]["backends"] == ["media"]

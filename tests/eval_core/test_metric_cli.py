from __future__ import annotations

import json

import pytest

from worldfoundry import cli


def test_metric_help_exposes_list_show_validate(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["metric", "--help"])
    output = capsys.readouterr().out

    assert exc_info.value.code == 0
    assert "list" in output
    assert "show" in output
    assert "validate" in output


def test_metric_list_and_show_emit_registry_metadata(capsys) -> None:
    assert cli.main(["metric", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_id = {item["id"]: item for item in payload}

    assert "artifact_count" in by_id
    assert by_id["has_artifact"]["parameterized_prefix"] == "has_artifact:"

    assert cli.main(["metric", "show", "has_artifact:video", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == "has_artifact"
    assert shown["resolved_metric_id"] == "has_artifact:video"


def test_metric_validate_returns_nonzero_for_unknown_metrics(capsys) -> None:
    exit_code = cli.main(
        ["metric", "validate", "artifact-count", "numeric:quality", "has_artifact:", "unknown", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert [item["metric_id"] for item in payload["metrics"]] == ["artifact-count", "numeric:quality"]
    assert [item["canonical_metric_id"] for item in payload["metrics"]] == ["artifact_count", "numeric:quality"]
    assert payload["unknown_metrics"] == ["has_artifact:", "unknown"]

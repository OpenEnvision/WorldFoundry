from __future__ import annotations

import json

import pytest

from worldfoundry import cli


def _write_cli_task_catalog(root) -> None:
    root.mkdir()
    (root / "task.yaml").write_text(
        """
name: cli-task
benchmark_name: cli-benchmark
protocol: open_loop
evaluation_protocol: reference_metrics
tags: [cli, smoke]
input_keys: [prompt]
output_keys: [generated_video]
metric_ids: [quality]
description: CLI task fixture.
data:
  metadata_path: manifests/cli.jsonl
""",
        encoding="utf-8",
    )


def test_task_help_exposes_list_show_validate(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["task", "--help"])
    output = capsys.readouterr().out

    assert exc_info.value.code == 0
    assert "list" in output
    assert "show" in output
    assert "validate" in output
    assert "materialize" in output


def test_task_list_reads_filesystem_yaml_registry(tmp_path, capsys) -> None:
    task_root = tmp_path / "tasks"
    _write_cli_task_catalog(task_root)

    exit_code = cli.main(["task", "list", "--task-root", str(task_root), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload) == 1
    assert payload[0]["task_name"] == "cli-task"
    assert payload[0]["benchmark_name"] == "cli-benchmark"
    assert payload[0]["evaluation_protocol"] == ["reference_metrics"]
    assert payload[0]["source_path"].endswith("task.yaml")


def test_task_show_prints_catalog_task_json(tmp_path, capsys) -> None:
    task_root = tmp_path / "tasks"
    _write_cli_task_catalog(task_root)

    exit_code = cli.main(
        [
            "task",
            "show",
            "cli-task",
            "--task-root",
            str(task_root),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["task_name"] == "cli-task"
    assert payload["protocol"] == "open_loop"
    assert payload["metric_ids"] == ["quality"]
    assert payload["data"]["metadata_path"] == "manifests/cli.jsonl"


def test_task_validate_reports_ok_for_valid_yaml(tmp_path, capsys) -> None:
    task_root = tmp_path / "tasks"
    _write_cli_task_catalog(task_root)

    exit_code = cli.main(["task", "validate", str(task_root / "task.yaml"), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["valid_count"] == 1
    assert payload["items"][0]["task_names"] == ["cli-task"]


def test_task_validate_returns_nonzero_for_invalid_yaml(tmp_path, capsys) -> None:
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("schema_version: future-task\nname: bad\n", encoding="utf-8")

    exit_code = cli.main(["task", "validate", str(invalid_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["invalid_count"] == 1
    assert payload["items"][0]["error_type"] == "ValueError"


def test_task_materialize_writes_generation_request_rows(tmp_path, capsys) -> None:
    task_root = tmp_path / "tasks"
    data_root = tmp_path / "data"
    requests_path = tmp_path / "requests.jsonl"
    _write_cli_task_catalog(task_root)
    (data_root / "manifests").mkdir(parents=True)
    (data_root / "manifests" / "cli.jsonl").write_text(
        '{"sample_id": "sample-a", "prompt": "orbit left"}\n'
        '{"sample_id": "sample-b", "prompt": "orbit right"}\n',
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "task",
            "materialize",
            "cli-task",
            "--task-root",
            str(task_root),
            "--dataset-root",
            str(data_root),
            "--num-samples",
            "1",
            "--output-jsonl",
            str(requests_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    rows = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert exit_code == 0
    assert payload["schema_version"] == "worldfoundry-materialized-requests"
    assert payload["task_type"] == "cli-task"
    assert payload["benchmark_name"] == "cli-benchmark"
    assert payload["sample_count"] == 1
    assert rows[0]["sample_id"] == "sample-a"
    assert rows[0]["task_name"] == "cli-task"
    assert rows[0]["inputs"]["prompt"] == "orbit left"
    assert rows[0]["output_schema"]["generated_video"]["kind"] == "generated_video"


def test_task_list_loads_default_benchmark_zoo_manifests(capsys) -> None:
    exit_code = cli.main(["task", "list", "--benchmark", "vbench", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "vbench_t2v_standard" in {item["task_name"] for item in payload}
    item = next(item for item in payload if item["task_name"] == "vbench_t2v_standard")
    assert item["metadata"]["source_kind"] == "benchmark_zoo"
    assert item["evaluation_protocol"] == ["external_official_runner"]


def test_task_include_path_adds_external_catalog(tmp_path, capsys) -> None:
    default_root = tmp_path / "default"
    include_root = tmp_path / "include"
    _write_cli_task_catalog(default_root)
    include_root.mkdir()
    (include_root / "external.yaml").write_text(
        """
name: external-task
benchmark_name: external-benchmark
protocol: external_official_benchmark
evaluation_protocol: external_official_runner
tags: [external]
""",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "task",
            "list",
            "--task-root",
            str(default_root),
            "--include-path",
            str(include_root),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert {item["task_name"] for item in payload} == {"cli-task", "external-task"}

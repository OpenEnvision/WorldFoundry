from __future__ import annotations

import json
from pathlib import Path

from worldfoundry import cli


def _write_samples(path: Path) -> None:
    path.write_text(
        '{"sample_id": "sample-a", "prompt": "move forward", "control_sequence": [{"action": "forward"}]}\n'
        '{"sample_id": "sample-b", "prompt": "turn right"}\n',
        encoding="utf-8",
    )


def test_dataset_create_show_validate_and_materialize_cli(tmp_path: Path, capsys) -> None:
    samples_path = tmp_path / "samples.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    requests_path = tmp_path / "requests.jsonl"
    _write_samples(samples_path)

    exit_code = cli.main(
        [
            "dataset",
            "create",
            "--samples-path",
            str(samples_path),
            "--root",
            str(tmp_path),
            "--dataset-id",
            "cli-dataset",
            "--split",
            "smoke",
            "--license",
            "mit",
            "--access",
            "status=public",
            "--output-json",
            str(manifest_path),
            "--json",
        ]
    )
    created = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert manifest_path.is_file()
    assert created["schema_version"] == "worldfoundry-dataset-manifest"
    assert created["dataset_id"] == "cli-dataset"
    assert created["samples_path"] == "samples.jsonl"
    assert created["sample_count"] == 2
    assert created["access"] == {"status": "public"}

    assert cli.main(["dataset", "show", str(manifest_path), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["sha256"] == created["sha256"]

    assert cli.main(["dataset", "validate", str(manifest_path), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["ok"] is True
    assert validation["sample_count"] == 2

    assert (
        cli.main(
            [
                "dataset",
                "materialize",
                str(manifest_path),
                "--task-name",
                "cli-task",
                "--input-key",
                "prompt",
                "--output-jsonl",
                str(requests_path),
                "--num-samples",
                "1",
                "--json",
            ]
        )
        == 0
    )
    materialized = json.loads(capsys.readouterr().out)
    request_rows = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert materialized["sample_count"] == 1
    assert request_rows[0]["sample_id"] == "sample-a"
    assert request_rows[0]["task_name"] == "cli-task"
    assert request_rows[0]["split"] == "smoke"
    assert request_rows[0]["inputs"]["prompt"] == "move forward"


def test_dataset_validate_cli_reports_manifest_drift(tmp_path: Path, capsys) -> None:
    samples_path = tmp_path / "samples.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    _write_samples(samples_path)
    cli.main(
        [
            "dataset",
            "create",
            "--samples-path",
            str(samples_path),
            "--root",
            str(tmp_path),
            "--dataset-id",
            "cli-dataset",
            "--output-json",
            str(manifest_path),
            "--json",
        ]
    )
    capsys.readouterr()
    samples_path.write_text('{"sample_id": "sample-a", "prompt": "changed"}\n', encoding="utf-8")

    exit_code = cli.main(["dataset", "validate", str(manifest_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert "sample_count mismatch: manifest=2 actual=1" in payload["issues"]
    assert "sha256 mismatch" in payload["issues"]

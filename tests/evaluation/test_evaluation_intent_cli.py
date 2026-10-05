from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.cli.main import main
from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest


def test_generate_score_cli_plans_custom_dataset_without_loading_model(tmp_path: Path, capsys) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"sample_id": "sample-1", "prompt": "A cube rolls down a ramp."}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(
        build_dataset_manifest(samples_path=samples, dataset_id="custom-prompts"),
        manifest_path,
    )

    exit_code = main(
        [
            "generate-score",
            "--model",
            "hailuo-2p3",
            "--dataset-manifest",
            str(manifest_path),
            "--metric",
            "artifact_count",
            "--output-dir",
            str(tmp_path / "output"),
            "--plan-only",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["intent_kind"] == "generate_and_score"
    assert payload["execution"]["kind"] == "evaluate"


def test_reproduce_cli_resolves_recipe(tmp_path: Path, capsys) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """
schema_version: worldfoundry-reproduction-recipe-v1
id: vbench-zeroscope-aesthetic
benchmark:
  id: vbench
  revision: benchmark-commit
model:
  id: zeroscope
  revision: model-commit
generation:
  num_frames: 8
data:
  num_samples: 1
evaluation:
  mode: official-run
reference:
  leaderboard_valid: false
""".lstrip(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "reproduce",
            "--recipe",
            str(recipe),
            "--output-dir",
            str(tmp_path / "output"),
            "--plan-only",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["config_sources"]["recipe_id"] == "vbench-zeroscope-aesthetic"

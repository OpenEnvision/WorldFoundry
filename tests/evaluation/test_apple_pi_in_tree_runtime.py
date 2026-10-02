from __future__ import annotations

import json

from PIL import Image

from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.framework.integration import IntegrationTier, integration_spec
from worldfoundry.evaluation.tasks.execution.runners.apple_pi.run_apple_pi_official_runner import main


def test_apple_pi_is_model_backed_and_does_not_require_upstream_runtime():
    assert integration_spec("apple-pi").tier is IntegrationTier.MODEL_BACKED
    assert get_external_benchmark_contract("apple-pi").requires_upstream_runtime is False


def test_apple_pi_native_mock_protocol(tmp_path):
    gt_root = tmp_path / "gt"
    case_dir = gt_root / "cases" / "case_000"
    case_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(case_dir / "input.png")
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input_image": "input.png",
                "physics_type": "projectile",
                "formula_info": {"choices": ["A", "B", "C", "D"]},
            }
        ),
        encoding="utf-8",
    )
    (gt_root / "dataset.json").write_text(
        json.dumps(
            {
                "name": "Apple-PI",
                "version": "fixture",
                "num_rollouts": 3,
                "cases": [{"case_id": "case_000", "path": "cases/case_000"}],
            }
        ),
        encoding="utf-8",
    )

    predictions = tmp_path / "predictions"
    for rollout in range(3):
        output_dir = predictions / "cases" / "case_000" / "perception_text"
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(output_dir / f"rollout_{rollout:02d}.png")
    (predictions / "submission.json").write_text(
        json.dumps({"model": "fixture", "protocol": "image", "num_rollouts": 3}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    assert main(
        [
            "--run-official",
            "--gt-dir",
            str(gt_root),
            "--pred-dir",
            str(predictions),
            "--subtrack",
            "perception_text",
            "--judge-backend",
            "mock",
            "--no-foundation-models",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["normalization_ok"] is True
    assert scorecard["normalizer_only"] is False
    assert scorecard["evaluation"]["kind"] == "apple_pi_in_tree_model_backed"

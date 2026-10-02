from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.reporting import (
    has_official_runtime_evidence,
    inspect_scorecard_runtime_flags,
)


def test_runtime_evidence_requires_explicit_true_flags(tmp_path: Path) -> None:
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "run": {"status": "succeeded"},
                "evaluation": {"kind": "normalizer"},
            }
        ),
        encoding="utf-8",
    )

    flags = inspect_scorecard_runtime_flags(scorecard_path)

    assert flags["found"] is True
    assert flags["run_status"] == "succeeded"
    assert flags["evaluation_kind"] == "normalizer"
    assert "official_benchmark_verified" not in flags
    assert has_official_runtime_evidence(flags) is False


def test_runtime_evidence_contract_validation_is_not_leaderboard_runtime(tmp_path: Path) -> None:
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "official_benchmark_verified": True,
                "integration_evidence": False,
                "worldfoundry_contract_validation_evidence": True,
                "run": {"status": "official_verified", "command": None},
            }
        ),
        encoding="utf-8",
    )
    flags = inspect_scorecard_runtime_flags(scorecard_path)
    assert flags["worldfoundry_contract_validation_evidence"] is True
    assert has_official_runtime_evidence(flags) is False


def test_runtime_evidence_accepts_explicit_official_runtime_flags(tmp_path: Path) -> None:
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "official_benchmark_verified": True,
                "integration_evidence": True,
                "run": {"status": "succeeded", "command": ["python", "-m", "official_eval"]},
            }
        ),
        encoding="utf-8",
    )

    flags = inspect_scorecard_runtime_flags(scorecard_path)

    assert flags["official_benchmark_verified"] is True
    assert flags["integration_evidence"] is True
    assert flags["run_command_present"] is True
    assert has_official_runtime_evidence(flags) is True

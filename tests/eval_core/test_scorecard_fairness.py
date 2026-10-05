from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.reporting.scorecard import build_scorecard


def _successful_scorecard(**kwargs: Any) -> dict[str, Any]:
    payload = {
        "run": {"run_id": "unit-scorecard", "status": "succeeded"},
        "benchmark": {"benchmark_name": "unit-benchmark"},
        "model": {"model_id": "unit-model"},
        "dataset": {"sample_count": 2},
        "generation": {"num_requests": 2, "successful": 2, "failed": 0},
        "metrics_summary": {
            "sample_count": 2,
            "successful_samples": 2,
            "failed_samples": 0,
            "leaderboard": {"quality": 0.75},
            "per_metric": {
                "quality": {
                    "mean": 0.75,
                    "sample_count": 2,
                    "higher_is_better": True,
                }
            },
            "groups": {},
        },
        "artifacts": {},
    }
    payload.update(kwargs)
    return build_scorecard(**payload)


def test_generic_scorecard_preserves_scores_but_blocks_leaderboard_without_evidence() -> None:
    scorecard = _successful_scorecard()

    assert scorecard["metrics"]["leaderboard"]["quality"] == 0.75
    assert scorecard["evaluation"]["leaderboard_metrics"]["quality"] == 0.75
    assert scorecard["eligibility"]["score_valid"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert scorecard["eligibility"]["leaderboard_eligible"] is False
    assert scorecard["eligibility"]["evidence_gate"]["present"] is False
    assert "missing official/full-suite leaderboard evidence gate" in scorecard["eligibility"]["reasons"]
    assert scorecard["evaluation"]["mode"] == "metric_only"
    assert scorecard["comparison_identity"]["schema_version"] == "worldfoundry-comparison-identity-v1"


def test_scorecard_classifies_official_protocol_new_model_from_provenance() -> None:
    scorecard = _successful_scorecard(
        provenance={
            "producer": "catalog_model",
            "fidelity": {"generation": "custom", "data": "official", "evaluation": "official"},
            "claim": {"leaderboard_candidate": True},
        }
    )

    assert scorecard["evaluation"]["kind"] == "existing_results"
    assert scorecard["evaluation"]["mode"] == "new_model_evaluation"
    assert scorecard["comparison_identity"]["protocol_fidelity"] == "official"
    assert scorecard["comparison_identity"]["data_fidelity"] == "official"


def test_generic_scorecard_allows_leaderboard_with_explicit_official_full_suite_gate() -> None:
    scorecard = _successful_scorecard(
        leaderboard_evidence={"gate": "official_full_suite", "passed": True, "source": "full-suite-report"}
    )

    assert scorecard["metrics"]["leaderboard"]["quality"] == 0.75
    assert scorecard["eligibility"]["leaderboard_valid"] is True
    assert scorecard["eligibility"]["leaderboard_eligible"] is True
    assert scorecard["eligibility"]["reasons"] == []
    assert scorecard["eligibility"]["evidence_gate"]["present"] is True
    assert scorecard["eligibility"]["evidence_gate"]["source_paths"] == ["leaderboard_evidence"]


def test_generic_scorecard_blocks_failed_samples_even_with_evidence_gate() -> None:
    scorecard = _successful_scorecard(
        metrics_summary={
            "sample_count": 2,
            "successful_samples": 1,
            "failed_samples": 1,
            "leaderboard": {"quality": 0.75},
            "per_metric": {"quality": {"mean": 0.75, "sample_count": 1, "higher_is_better": True}},
            "groups": {},
        },
        leaderboard_evidence={"official_full_suite_evidence": True},
    )

    assert scorecard["eligibility"]["score_valid"] is False
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert scorecard["eligibility"]["evidence_gate"]["present"] is True
    assert "1 sample(s) failed" in scorecard["eligibility"]["reasons"]

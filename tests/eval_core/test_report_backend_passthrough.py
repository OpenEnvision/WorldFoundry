"""Scorecard ``backend`` passthrough into every aggregated report surface.

Mock-capable official runners (phygenbench, iworldbench, mirabench, ewmbench,
genai_bench, phyfps_bench_gen, ...) record which backend produced the scores,
but historically the reporting layer dropped that field, so a mock (fixture)
scorecard aggregated identically to an official one.  These tests pin the
contract: the backend is extracted from every known scorecard shape, surfaces
in summary/report/index/comparison/browser artifacts, and mock scores are
never leaderboard-comparable after aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.reporting import (
    MOCK_BACKEND_BLOCKING_REASON,
    build_markdown_comparison,
    build_markdown_report,
    build_markdown_run_index,
    build_run_browser_html,
    build_run_comparison,
    build_run_index,
    build_run_summary,
    build_scorecard,
    evaluation_backend_from_payload,
    is_mock_backend,
)


def _scorecard(
    *,
    run_id: str,
    run_extra: dict[str, Any] | None = None,
    evaluation_extra: dict[str, Any] | None = None,
    leaderboard_valid: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "worldfoundry-scorecard",
        "run": {"run_id": run_id, "status": "succeeded", **(run_extra or {})},
        "benchmark": {"benchmark_id": "backend-benchmark", "benchmark_name": "backend-benchmark"},
        "model": {"model_id": f"{run_id}-model", "model_name": f"{run_id}-model"},
        "dataset": {"dataset_id": "backend-dataset", "sample_count": 2},
        "generation": {"num_requests": 2, "successful": 2, "failed": 0},
        "metrics": {
            "leaderboard": {"quality": 0.9},
            "per_metric": {"quality": {"mean": 0.9, "higher_is_better": True}},
            "summary": {
                "sample_count": 2,
                "successful_samples": 2,
                "failed_samples": 0,
                "failed_sample_ids": [],
            },
        },
        "evaluation": {"kind": "existing_results", **(evaluation_extra or {})},
        "eligibility": {
            "score_valid": True,
            "leaderboard_valid": leaderboard_valid,
            "leaderboard_eligible": leaderboard_valid,
            "reasons": [],
            "blocking_reasons": [],
        },
        "artifacts": {"scorecard": f"/tmp/{run_id}/scorecard.json"},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_backend_extraction_covers_known_scorecard_shapes() -> None:
    # Direct fields.
    assert evaluation_backend_from_payload({"evaluation": {"backend": " Official "}}) == "official"
    assert evaluation_backend_from_payload({"run": {"backend": "mock"}}) == "mock"
    assert evaluation_backend_from_payload({"generation": {"backend": "mock"}}) == "mock"
    # Nested stage summaries used by the mock-capable runners.
    assert (
        evaluation_backend_from_payload({"run": {"judge_summary": {"backend": "mock"}}}) == "mock"
    )
    assert (
        evaluation_backend_from_payload({"run": {"runtime_summary": {"backend": "official"}}})
        == "official"
    )
    assert (
        evaluation_backend_from_payload({"evaluation": {"predict_summary": {"backend": "MOCK"}}})
        == "mock"
    )
    # Direct fields win over nested stage summaries.
    assert (
        evaluation_backend_from_payload(
            {
                "evaluation": {"backend": "official"},
                "run": {"judge_summary": {"backend": "mock"}},
            }
        )
        == "official"
    )
    assert evaluation_backend_from_payload({}) is None
    assert is_mock_backend("mock") is True
    assert is_mock_backend(" MOCK ") is True
    assert is_mock_backend("official") is False
    assert is_mock_backend(None) is False


def test_build_run_summary_propagates_backend_and_blocks_mock_leaderboard() -> None:
    scorecard = _scorecard(
        run_id="mock-run",
        run_extra={"judge_summary": {"backend": "mock"}},
        leaderboard_valid=True,
    )

    summary = build_run_summary(scorecard)

    assert summary["evaluation"]["backend"] == "mock"
    assert summary["eligibility"]["leaderboard_valid"] is False
    assert summary["eligibility"]["leaderboard_eligible"] is False
    assert MOCK_BACKEND_BLOCKING_REASON in summary["eligibility"]["reasons"]
    assert MOCK_BACKEND_BLOCKING_REASON in summary["eligibility"]["blocking_reasons"]


def test_build_run_summary_keeps_official_backend_eligibility() -> None:
    scorecard = _scorecard(
        run_id="official-run",
        run_extra={"runtime_summary": {"backend": "official"}},
        leaderboard_valid=True,
    )

    summary = build_run_summary(scorecard)

    assert summary["evaluation"]["backend"] == "official"
    assert summary["eligibility"]["leaderboard_valid"] is True
    assert MOCK_BACKEND_BLOCKING_REASON not in summary["eligibility"]["reasons"]


def test_markdown_report_shows_backend_and_mock_warning() -> None:
    mock_summary = build_run_summary(
        _scorecard(run_id="mock-run", run_extra={"judge_summary": {"backend": "mock"}})
    )
    official_summary = build_run_summary(
        _scorecard(run_id="official-run", evaluation_extra={"backend": "official"})
    )

    mock_report = build_markdown_report(mock_summary)
    official_report = build_markdown_report(official_summary)

    assert "- Backend: mock" in mock_report
    assert MOCK_BACKEND_BLOCKING_REASON in mock_report
    assert "- Backend: official" in official_report
    assert MOCK_BACKEND_BLOCKING_REASON not in official_report


def test_run_index_propagates_backend_and_flags_mock_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "mock-run" / "scorecard.json",
        _scorecard(run_id="mock-run", run_extra={"judge_summary": {"backend": "mock"}}),
    )
    _write_json(
        tmp_path / "official-run" / "scorecard.json",
        _scorecard(run_id="official-run", evaluation_extra={"backend": "official"}),
    )

    index = build_run_index(tmp_path)

    backends = {row["run_id"]: row["backend"] for row in index["rows"]}
    assert backends == {"mock-run": "mock", "official-run": "official"}
    assert index["backends"] == ["mock", "official"]
    mock_issues = [issue for issue in index["issues"] if issue.startswith("mock backend:")]
    assert len(mock_issues) == 1
    assert "mock-run" in mock_issues[0]
    assert MOCK_BACKEND_BLOCKING_REASON in mock_issues[0]

    markdown = build_markdown_run_index(index)
    assert "| Backend |" in markdown
    assert "| mock |" in markdown
    assert "| official |" in markdown


def test_run_comparison_propagates_backend_and_flags_mock_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "mock-run" / "scorecard.json",
        _scorecard(run_id="mock-run", run_extra={"judge_summary": {"backend": "mock"}}),
    )
    _write_json(
        tmp_path / "official-run" / "scorecard.json",
        _scorecard(run_id="official-run", evaluation_extra={"backend": "official"}),
    )

    comparison = build_run_comparison(
        [tmp_path / "mock-run", tmp_path / "official-run"],
        labels=["mock-run", "official-run"],
    )

    backends = {row["label"]: row["backend"] for row in comparison["rows"]}
    assert backends == {"mock-run": "mock", "official-run": "official"}
    assert comparison["backends"] == ["mock", "official"]
    mock_issues = [issue for issue in comparison["issues"] if issue.startswith("mock backend:")]
    assert len(mock_issues) == 1
    assert "mock-run" in mock_issues[0]

    markdown = build_markdown_comparison(comparison)
    assert "| Backend |" in markdown
    assert "mock backend: mock-run" in markdown


def test_build_scorecard_records_backend_and_blocks_mock() -> None:
    common = {
        "benchmark": {"benchmark_id": "backend-benchmark"},
        "model": {"model_id": "backend-model"},
        "dataset": {"dataset_id": "backend-dataset"},
        "metrics_summary": {
            "sample_count": 1,
            "successful_samples": 1,
            "failed_samples": 0,
            "leaderboard": {"quality": 0.5},
        },
        "artifacts": {},
        "leaderboard_evidence": {"official_full_suite_evidence": True},
    }

    mock_scorecard = build_scorecard(
        run={"run_id": "mock-run", "judge_summary": {"backend": "mock"}},
        generation={"num_requests": 1, "successful": 1, "failed": 0},
        **common,
    )
    official_scorecard = build_scorecard(
        run={"run_id": "official-run"},
        generation={"num_requests": 1, "successful": 1, "failed": 0, "backend": "official"},
        **common,
    )

    assert mock_scorecard["evaluation"]["backend"] == "mock"
    assert mock_scorecard["eligibility"]["leaderboard_valid"] is False
    assert MOCK_BACKEND_BLOCKING_REASON in mock_scorecard["eligibility"]["reasons"]

    assert official_scorecard["evaluation"]["backend"] == "official"
    assert official_scorecard["eligibility"]["leaderboard_valid"] is True
    assert MOCK_BACKEND_BLOCKING_REASON not in official_scorecard["eligibility"]["reasons"]


def test_run_browser_renders_backend_column(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "mock-run" / "scorecard.json",
        _scorecard(run_id="mock-run", run_extra={"judge_summary": {"backend": "mock"}}),
    )

    html = build_run_browser_html(build_run_index(tmp_path))

    assert "<th>Backend</th>" in html
    assert "row.backend" in html

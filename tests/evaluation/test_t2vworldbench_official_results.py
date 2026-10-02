from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.t2vworldbench.run_t2vworldbench_official_runner import (
    CONFIG,
    extract_metrics,
    main,
)


def test_extracts_real_official_csv_fields_with_their_published_scales() -> None:
    payload = [
        {
            "grid_image_name": "sample-1.png",
            "min_quality_score": "4",
            "min_realism_score": "3",
            "min_relevance_score": "5",
            "min_consistency_score": "2",
            "total_min_score": "14",
            "final_min_score": "14",
        },
        {
            "grid_image_name": "sample-2.png",
            "min_quality_score": "2",
            "min_realism_score": "5",
            "min_relevance_score": "3",
            "min_consistency_score": "4",
            "final_min_score": "14",
        },
        # ``model_score`` appends this footer row to the official CSV.  It
        # must not be mistaken for a per-video dimension score.
        {
            "grid_image_name": "Overall Average Model Score:",
            "original_t2v_prompt": "14.0000",
        },
    ]

    metrics = extract_metrics(payload, Path("official_video_assessment_scores.csv"))

    assert tuple(metrics) == ("quality", "realism", "relevance", "consistency", "final")
    assert metrics["quality"]["raw_score"] == 3.0
    assert metrics["quality"]["normalized_score"] == 0.6
    assert metrics["realism"]["normalized_score"] == 0.8
    assert metrics["relevance"]["normalized_score"] == 0.8
    assert metrics["consistency"]["normalized_score"] == 0.6
    assert metrics["final"]["raw_score"] == 14.0
    assert metrics["final"]["normalized_score"] == 0.7
    assert all(row["sample_count"] == 2 for row in metrics.values())
    assert not {"physics_knowledge", "world_knowledge_average"}.intersection(metrics)


def test_runner_schema_exposes_only_fields_emitted_by_official_runtime() -> None:
    assert CONFIG.metric_order == ("quality", "realism", "relevance", "consistency", "final")
    assert "physics_knowledge" not in CONFIG.metric_specs


def test_official_result_import_writes_matching_scorecard_schema(tmp_path) -> None:
    results_path = tmp_path / "model_video_assessment_scores.csv"
    results_path.write_text(
        "grid_image_name,min_quality_score,min_realism_score,min_relevance_score,min_consistency_score,final_min_score\n"
        "sample.png,4,3,5,2,14\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    assert (
        main(
            [
                "--official-results-path",
                str(results_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    per_metric = scorecard["metrics"]["per_metric"]
    assert set(per_metric) == {"quality", "realism", "relevance", "consistency", "final"}
    assert per_metric["quality"]["raw_score"] == 4.0
    assert per_metric["quality"]["normalized_score"] == 0.8
    assert per_metric["final"]["raw_score"] == 14.0
    assert per_metric["final"]["normalized_score"] == 0.7
    assert "physics_knowledge" not in per_metric

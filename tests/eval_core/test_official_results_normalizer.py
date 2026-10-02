from __future__ import annotations

import json
from pathlib import Path

import yaml

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import load_benchmark_catalog_shard_entries
from worldfoundry.evaluation.tasks.catalog.schema import BenchmarkMetricSpec, BenchmarkZooEntry
from worldfoundry.evaluation.tasks.execution.framework.result_normalizer import (
    OfficialMetricMapping,
    OfficialResultsNormalizer,
    build_official_metric_mappings,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_VIDEO_WORLD_CONTRACT_IDS = (
    "aigcbench",
    "mirabench",
    "devil-dynamics",
    "genai-bench",
    "phygenbench",
    "videophy",
    "videophy2",
    "physics-iq",
    "t2v-safety-bench",
    "ipv-bench",
    "videoscience-bench",
    "phyeduvideo",
    # t2vphysbench was removed from the external benchmark catalog.
    "physvidbench",
    "t2vworldbench",
    "worldarena",
    "world-in-world",
    "phyground",
    "ewmbench",
)


def test_official_results_normalizer_imports_toy_json(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"results": [{"sample_id": "a", "quality": 0.8}, {"sample_id": "b", "quality": 0.6}]}),
        encoding="utf-8",
    )
    normalizer = OfficialResultsNormalizer(
        "toy-benchmark",
        (OfficialMetricMapping(metric_id="quality", source_fields=("quality",), required_fields=("quality",)),),
    )

    result = normalizer.normalize_file(path)

    assert [item.sample_id for item in result.per_sample_results] == ["a", "b"]
    assert result.aggregate_results[0].metric_id == "quality"
    assert result.aggregate_results[0].valid is True
    assert result.aggregate_results[0].normalized_value == 0.7
    assert result.raw_metric_rows()[0]["available"] is True
    assert result.scorecard_metrics()["quality"]["normalized_score"] == 0.7


def test_official_results_normalizer_imports_toy_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"sample_id": "a", "accuracy": 80}),
                json.dumps({"sample_id": "b", "accuracy": 100}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    normalizer = OfficialResultsNormalizer(
        "toy-benchmark",
        (
            OfficialMetricMapping(
                metric_id="accuracy",
                source_fields=("accuracy",),
                normalizer="percent_or_fraction_to_unit",
            ),
        ),
    )

    result = normalizer.normalize_file(path)

    assert len(result.per_sample_results) == 2
    assert result.aggregate_results[0].normalized_value == 0.9


def test_official_results_normalizer_imports_toy_csv(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("sample_id,score\nsample-a,1.0\nsample-b,0.5\n", encoding="utf-8")
    normalizer = OfficialResultsNormalizer(
        "toy-benchmark",
        (OfficialMetricMapping(metric_id="score", source_fields=("score",), required_fields=("score",)),),
    )

    result = normalizer.normalize_file(path)

    assert len(result.per_sample_results) == 2
    assert result.aggregate_results[0].normalized_value == 0.75


def test_official_results_normalizer_blocks_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("sample_id,other_score\nsample-a,1.0\n", encoding="utf-8")
    normalizer = OfficialResultsNormalizer(
        "toy-benchmark",
        (OfficialMetricMapping(metric_id="score", source_fields=("score",), required_fields=("score",)),),
    )

    result = normalizer.normalize_file(path)

    assert result.per_sample_results == ()
    assert result.aggregate_results[0].valid is False
    assert result.aggregate_results[0].skip_reason == "missing_required_official_result_field"
    assert result.raw_metric_rows()[0]["available"] is False


def test_official_results_normalizer_blocks_unknown_metric() -> None:
    normalizer = OfficialResultsNormalizer(
        "toy-benchmark",
        (OfficialMetricMapping(metric_id="known_score", source_fields=("known_score",)),),
        requested_metric_ids=("known_score", "missing_score"),
    )

    result = normalizer.normalize_records(({"sample_id": "a", "known_score": 1.0},))

    by_metric = {item.metric_id: item for item in result.aggregate_results}
    assert by_metric["known_score"].valid is True
    assert by_metric["missing_score"].valid is False
    assert by_metric["missing_score"].skip_reason == "unknown_metric"


def test_benchmark_metric_aliases_are_alternatives_not_required_columns() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "alias-benchmark",
            "metrics": [
                {
                    "id": "physics_knowledge",
                    "leaderboard_key": "physics",
                }
            ],
        }
    )
    normalizer = OfficialResultsNormalizer.from_benchmark_entry(entry)

    result = normalizer.normalize_records(({"sample_id": "a", "physics": 0.7},))

    assert result.aggregate_results[0].valid is True
    assert result.aggregate_results[0].normalized_value == 0.7
    assert result.per_sample_results[0].components["source_field"] == "physics"


def test_official_results_normalizer_aggregates_category_rows() -> None:
    normalizer = OfficialResultsNormalizer(
        "toy-benchmark",
        (
            OfficialMetricMapping(
                metric_id="physics",
                source_fields=("score",),
                category_field="category",
                category_values=("physics",),
            ),
        ),
    )

    result = normalizer.normalize_records(
        (
            {"sample_id": "p1", "category": "physics", "score": 1.0},
            {"sample_id": "p2", "category": "physics", "score": 0.0},
            {"sample_id": "c1", "category": "chemistry", "score": 1.0},
        )
    )

    assert [item.sample_id for item in result.per_sample_results] == ["p1", "p2"]
    assert result.aggregate_results[0].normalized_value == 0.5
    assert result.aggregate_results[0].components["category_values"] == ["physics"]


def test_benchmark_metric_schema_preserves_official_results_hook() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "schema-hook-benchmark",
            "metrics": [
                {
                    "id": "score",
                    "leaderboard_key": "Score",
                    "normalizer": "scale_max:5",
                    "official_results": {
                        "source_fields": ["Score"],
                        "required_columns": ["sample_id", "Score"],
                        "aggregation": "mean",
                    },
                }
            ],
        }
    )
    mapping = OfficialMetricMapping.from_metric_spec(entry.metrics[0])

    assert entry.metrics[0].official_results["source_fields"] == ["Score"]
    assert mapping.source_fields == ("Score",)
    assert mapping.required_fields == ("sample_id", "Score")
    assert mapping.normalizer == "scale_max:5"


def test_new_video_world_contracts_have_configurable_official_result_mappings() -> None:
    entries = {entry.benchmark_id: entry for entry in load_benchmark_catalog_shard_entries("video")}

    for benchmark_id in NEW_VIDEO_WORLD_CONTRACT_IDS:
        mappings = build_official_metric_mappings(entries[benchmark_id])

        assert len(mappings) == len(entries[benchmark_id].metrics)
        assert all(mapping.metric_id for mapping in mappings)
        assert all(mapping.source_fields for mapping in mappings)

from __future__ import annotations

import json
from pathlib import Path

import worldfoundry.evaluation.tasks.embodied.normalizer as normalizer


def test_normalize_json_results_writes_scorecard_and_metrics(tmp_path: Path) -> None:
    results = tmp_path / "libero.json"
    results.write_text(
        json.dumps(
            {
                "summary": {"official_success_rate": 0.75},
                "results": [
                    {"task_id": "libero_10", "episode_id": "e1", "success": True, "latency_score": 0.8},
                    {"task_id": "libero_10", "episode_id": "e2", "success": False, "latency_score": 0.6},
                    {"task_id": "libero_spatial", "episode_id": "e3", "success": "100%", "latency_score": 0.4},
                ],
            }
        ),
        encoding="utf-8",
    )

    scorecard = normalizer.normalize_results(
        input_paths=[results],
        output_dir=tmp_path / "out",
        benchmark_id="libero",
        track="vla",
    )

    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["benchmark"]["normalizer_first"] is True
    assert scorecard["evaluation"]["sample_count"] == 3
    assert scorecard["evaluation"]["task_count"] == 2
    assert scorecard["metrics"]["leaderboard"]["success"] == 2 / 3
    assert scorecard["metrics"]["per_task"]["libero_10"]["success"] == 0.5
    assert (tmp_path / "out" / "scorecard.json").is_file()
    assert (tmp_path / "out" / "raw_results.jsonl").read_text(encoding="utf-8").count("\n") == 6


def test_normalize_csv_supports_metric_value_rows_and_normalizer_override(tmp_path: Path) -> None:
    csv_path = tmp_path / "wam.csv"
    csv_path.write_text(
        "sample_id,task_id,metric_id,value\n"
        "s1,branch,world_state_consistency,70\n"
        "s2,branch,world_state_consistency,90\n",
        encoding="utf-8",
    )

    scorecard = normalizer.normalize_results(
        input_paths=[csv_path],
        output_dir=tmp_path / "out",
        benchmark_id="wam-bench",
        track="wam",
        normalizers={"world_state_consistency": "scale_max:100"},
    )

    assert scorecard["metrics"]["leaderboard"]["world_state_consistency"] == 0.8
    assert scorecard["evaluation"]["per_task"]["branch"]["world_state_consistency"] == 0.8


def test_normalize_embodied_benchmark_common_fields(tmp_path: Path) -> None:
    results = tmp_path / "rlbench.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "reach_target",
                        "episode_id": "e1",
                        "sequence_success": "100%",
                        "episode_success": True,
                        "normalized_return": 0.7,
                    }
                ),
                json.dumps(
                    {
                        "task": "reach_target",
                        "episode_id": "e2",
                        "sequence_success": "0%",
                        "episode_success": False,
                        "normalized_return": 0.3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scorecard = normalizer.normalize_results(
        input_paths=[results],
        output_dir=tmp_path / "out",
        benchmark_id="rlbench",
        track="vla",
    )

    assert scorecard["metrics"]["leaderboard"]["sequence_success"] == 0.5
    assert scorecard["metrics"]["leaderboard"]["episode_success"] == 0.5
    assert scorecard["metrics"]["leaderboard"]["normalized_return"] == 0.5


def test_normalize_robotwin_result_txt_tree(tmp_path: Path) -> None:
    result_file = (
        tmp_path
        / "eval_result"
        / "handover_block"
        / "ACT"
        / "demo_clean"
        / "ckpt100"
        / "20260525"
        / "_result.txt"
    )
    result_file.parent.mkdir(parents=True)
    result_file.write_text("Instruction Type: demo\n0.8\n", encoding="utf-8")

    scorecard = normalizer.normalize_results(
        input_paths=[tmp_path / "eval_result"],
        output_dir=tmp_path / "out",
        benchmark_id="robotwin",
        track="vla",
    )

    assert scorecard["metrics"]["leaderboard"]["success_rate"] == 0.8
    assert scorecard["metrics"]["per_task"]["handover_block"]["success_rate"] == 0.8


def test_normalizer_main_rejects_invalid_track(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text('{"sample_id":"s1","success":true}\n', encoding="utf-8")

    rc = normalizer.main(
        [
            "--input",
            str(results),
            "--output-dir",
            str(tmp_path / "out"),
            "--benchmark-id",
            "bad",
            "--track",
            "vla",
        ]
    )

    assert rc == 0
    payload = json.loads((tmp_path / "out" / "scorecard.json").read_text(encoding="utf-8"))
    assert payload["benchmark"]["track"] == "vla"

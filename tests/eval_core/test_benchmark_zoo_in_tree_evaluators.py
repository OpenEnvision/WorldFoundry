from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import (
    BenchmarkZooInTreeEvaluator,
    evaluate_benchmark_metrics,
)
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import run_evaluate


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external"
TARGET_BENCHMARKS = (
    "t2v-safety-bench",
    "videoscience-bench",
    "phyeduvideo",
    "worldarena",
    "world-in-world",
    "ewmbench",
)
NEW_BENCHMARKS_WITH_EVALUATORS = (
    # t2vphysbench was removed from the external benchmark catalog.
    "evalcrafter",
    "fetv",
    "t2vworldbench",
    "videoverse",
    "physvidbench",
    "aigcbench",
    "mirabench",
    "devil-dynamics",
    "genai-bench",
    "phygenbench",
    "videophy2",
    "physics-iq",
    "ipv-bench",
    "phyground",
    "ewmbench",
)


def _video_result(metadata: dict | None = None) -> GenerationResult:
    return GenerationResult(
        sample_id="sample-1",
        artifacts={"generated_video": ArtifactRef(uri="memory://sample.mp4", kind="video")},
        metadata=metadata or {},
    )


def _world_in_world_result(metadata: dict | None = None) -> GenerationResult:
    return GenerationResult(
        sample_id="sample-1",
        artifacts={
            "generated_video": ArtifactRef(uri="memory://sample.mp4", kind="video"),
            "interaction_trace": ArtifactRef(uri="memory://trace.json", kind="structured_trace"),
        },
        metadata=metadata or {},
    )


def test_t2v_safety_evaluator_scores_rule_labels_and_violation_manifest() -> None:
    request = GenerationRequest(sample_id="sample-1", task_name="t2v-safety")
    result = _video_result(
        {
            "nsfw_labels": {
                "pornography": "safe",
                "borderline_pornography": "safe",
                "violence": "safe",
                "gore": "safe",
                "public_figures": "safe",
                "discrimination": "safe",
                "political_sensitivity": "safe",
                "illegal_activities": "safe",
                "disturbing_content": "safe",
                "misinformation_falsehoods": "safe",
                "copyright_trademark": "safe",
            },
            "violation_manifest": {"temporal_risk": ["unsafe escalation"]},
        }
    )

    rows = evaluate_benchmark_metrics("t2v-safety-bench", request, result)
    by_metric = {row["metric_id"]: row for row in rows}

    assert by_metric["pornography_nsfw_rate"]["status"] == "scored"
    assert by_metric["pornography_nsfw_rate"]["score"] == 0.0
    assert by_metric["temporal_risk_nsfw_rate"]["score"] == 1.0
    assert by_metric["nsfw_average"]["score"] == 1 / 12
    assert by_metric["nsfw_average"]["evidence"]["components"]


def test_t2v_safety_evaluator_requires_judge_when_no_labels_exist() -> None:
    request = GenerationRequest(sample_id="sample-1", task_name="t2v-safety")
    rows = evaluate_benchmark_metrics("t2v-safety-bench", request, _video_result())
    statuses = {row["status"] for row in rows}

    assert statuses == {"judge_required"}
    assert all(row["blocked_reason"] for row in rows)


def test_reasoning_evaluator_scores_mcq_and_rubric_manifest() -> None:
    request = GenerationRequest(
        sample_id="sample-1",
        task_name="videoscience",
        inputs={
            "answer_key": {"prompt_consistency": "B"},
            "rubric_manifest": {
                "phenomenon_congruency": {"score": 4, "max_score": 5},
                "correct_dynamism": 0.5,
                "immutability": 100,
            },
            "checklist_scores": {
                "spatio_temporal_coherence": {"satisfied": 3, "total": 4},
            },
        },
    )
    result = _video_result({"predictions": {"prompt_consistency": "option B"}})

    rows = evaluate_benchmark_metrics("videoscience-bench", request, result)
    by_metric = {row["metric_id"]: row for row in rows}

    assert by_metric["prompt_consistency"]["score"] == 1.0
    assert by_metric["phenomenon_congruency"]["score"] == 0.8
    assert by_metric["correct_dynamism"]["score"] == 0.5
    assert by_metric["immutability"]["score"] == 1.0
    assert by_metric["spatio_temporal_coherence"]["score"] == 0.75
    assert by_metric["videoscience_average"]["score"] == pytest.approx(0.81)


def test_reasoning_evaluator_blocks_without_official_judge_or_rubric() -> None:
    request = GenerationRequest(sample_id="sample-1", task_name="phyeduvideo")
    rows = evaluate_benchmark_metrics("phyeduvideo", request, _video_result())

    assert {row["status"] for row in rows} == {"blocked"}
    assert all(row["blocked_reason"] for row in rows)


def test_world_in_world_requires_interaction_trace_artifact() -> None:
    request = GenerationRequest(sample_id="sample-1", task_name="world-in-world")
    rows = evaluate_benchmark_metrics("world-in-world", request, _video_result())

    assert {row["status"] for row in rows} == {"blocked"}
    assert rows[0]["blocked_reason"] == "required artifact presence or format check failed"
    assert any(
        check["artifact"] == "interaction_trace" and check["present"] is False
        for check in rows[0]["evidence"]["artifact_checks"]
    )


def test_world_in_world_scores_task_success_and_trace_consistency() -> None:
    request = GenerationRequest(
        sample_id="sample-1",
        task_name="world-in-world",
        inputs={"action_trace": ["turn_left", "move_forward", "stop"]},
    )
    result = _world_in_world_result(
        {
            "closed_loop_results": {
                "active_recognition_success_rate": {"success_rate": 50},
                "image_goal_navigation_success_rate": 0.25,
                "image_goal_navigation_spl": 0.2,
                "active_embodied_qa_score": 0.75,
                "active_embodied_qa_spl": 0.5,
                "robotic_manipulation_success_rate": True,
            },
            "interaction_trace": {"actions": ["turn_left", "move_forward", "stop"]},
        }
    )

    rows = evaluate_benchmark_metrics("world-in-world", request, result)
    by_metric = {row["metric_id"]: row for row in rows}

    assert by_metric["active_recognition_success_rate"]["score"] == 0.5
    assert by_metric["robotic_manipulation_success_rate"]["score"] == 1.0
    assert by_metric["interaction_trace_consistency"]["score"] == 1.0
    assert by_metric["world_in_world_average"]["status"] == "scored"


def test_in_tree_evaluator_runs_through_existing_results_runner(tmp_path: Path) -> None:
    request = GenerationRequest(
        sample_id="sample-1",
        task_name="ewmbench",
        inputs={"rubric_manifest": {"scene_consistency": 1, "motion_correctness": 1, "semantic_alignment": 0.5, "diversity": 0.5}},
    )
    result = _video_result()

    run_result = run_evaluate(
        output_dir=tmp_path,
        mode="existing-results",
        benchmark_id="ewmbench",
        requests=[request.to_dict()],
        results=[result.to_dict()],
        metrics=[
            "scene_consistency",
            "motion_correctness",
            "semantic_alignment",
            "diversity",
            "ewmbench_average",
        ],
    )
    scorecard = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))

    assert run_result.status == "succeeded"
    assert scorecard["metrics"]["leaderboard"]["ewmbench_average"] == 0.75
    assert scorecard["metrics"]["per_metric"]["ewmbench_average"]["sample_count"] == 1


def test_target_benchmark_task_yaml_runtime_roots_are_env_names() -> None:
    for benchmark_id in TARGET_BENCHMARKS:
        manifest = yaml.safe_load((TASK_ROOT / f"{benchmark_id}.yaml").read_text(encoding="utf-8"))
        runtime = manifest["metadata"]["runtime"]
        if "root_env" not in runtime:
            # world-in-world switched to bundled assets (asset_override_env /
            # prompt_manifest_env) and no longer exposes a runtime repo root.
            assert benchmark_id == "world-in-world"
            continue
        root_env = runtime["root_env"]

        assert isinstance(root_env, str)
        assert root_env.startswith("WORLDFOUNDRY_")
        assert "/" not in root_env
        assert root_env == root_env.upper()


def test_videoscience_bench_runtime_root_env_regression() -> None:
    manifest = yaml.safe_load((TASK_ROOT / "videoscience-bench.yaml").read_text(encoding="utf-8"))

    assert manifest["metadata"]["runtime"]["root_env"] == "WORLDFOUNDRY_VIDEOSCIENCE_BENCH_ROOT"


def test_metric_protocol_returns_uniform_blocked_result() -> None:
    evaluator = BenchmarkZooInTreeEvaluator("videoscience-bench")
    request = GenerationRequest(sample_id="sample-1", task_name="videoscience")
    metric_results = evaluator.compute_sample(request, _video_result({"requires_official_judge": True}))

    assert metric_results[0].metric_id == "prompt_consistency"
    assert metric_results[0].valid is False
    assert metric_results[0].diagnostics["status"] == "blocked"
    assert metric_results[0].diagnostics["blocked_reason"] == "official judge outputs are required"


def test_new_benchmark_metric_evaluator_paths_are_discoverable() -> None:
    from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
    from worldfoundry.evaluation.tasks.metrics import list_external_metric_evaluators
    from worldfoundry.evaluation.tasks.execution.framework.video_quality_contract import (
        supports_video_quality_benchmark,
    )

    in_tree_ids = set(TARGET_BENCHMARKS)
    for benchmark_id in NEW_BENCHMARKS_WITH_EVALUATORS:
        contract = get_external_benchmark_contract(benchmark_id)
        evaluator_ids = {
            entry.metric_id
            for entry in list_external_metric_evaluators(benchmark_id)
            if entry.benchmark_id == benchmark_id
        }

        assert evaluator_ids == set(contract.metric_ids)
        # physics-video benchmark family (is_physics_video_benchmark) was removed
        # from the catalog together with tasks.official.physics_video.
        assert (
            benchmark_id in in_tree_ids
            or supports_video_quality_benchmark(benchmark_id)
            or evaluator_ids
        )

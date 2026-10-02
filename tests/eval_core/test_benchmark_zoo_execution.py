from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from worldfoundry.cli import main as eval_cli_main
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids
from worldfoundry.evaluation.tasks.execution.orchestration import benchmark_runner as benchmark_runner_module
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
    build_benchmark_runner_registry,
    run_benchmark_execution,
)
from worldfoundry.evaluation.tasks.execution.orchestration.interfaces import BenchmarkRunner, OfficialBenchmarkRunner
from worldfoundry.evaluation.tasks.execution.runners import workspace_registry
from worldfoundry.evaluation.utils import benchmark_task_sample_path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ZOO_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"
MANIFEST_PATH = BENCHMARK_ZOO_DIR


def _official_repo_dir(name: str) -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_OFFICIAL_REPOS_DIR", REPO_ROOT.parent / "github_repos")) / name


def test_runner_registry_discovers_contract_runners_without_promoting_official_status() -> None:
    registry = build_benchmark_runner_registry(MANIFEST_PATH)

    integrated = [entry.benchmark_id for entry in registry.integrated()]
    planned = [entry.benchmark_id for entry in registry.planned()]
    blocked = [entry.benchmark_id for entry in registry.blocked()]

    assert "video-bench" in integrated
    assert "video-bench" not in blocked
    assert "vbench" in {*planned, *integrated}
    assert "worldscore" in {*planned, *integrated}
    assert "iworld-bench" in {*planned, *integrated}
    assert registry.has_runner("vbench") is True
    assert registry.has_runner("worldscore") is True
    assert registry.has_runner("video-bench") is True
    assert registry.has_runner("iworld-bench") is True
    assert registry.has_official_runner("vbench") is False
    assert registry.has_official_runner("worldscore") is False
    assert registry.has_official_runner("video-bench") is False
    assert registry.has_official_runner("iworld-bench") is False

    videobench_entry = registry.zoo.get("video-bench")
    assert registry.get_runner("video-bench").benchmark_id == "video-bench"
    assert videobench_entry.official_benchmark_verified is False
    assert videobench_entry.integration_evidence is False
    assert videobench_entry.leaderboard_valid is False
    assert videobench_entry.runner_availability["surface"] == "videobench_official_in_tree"
    assert videobench_entry.runner_availability["normalizer_available"] is True
    assert registry.get_runner("worldscore").benchmark_id == "worldscore"


def test_default_runner_registry_covers_formal_benchmark_inventory() -> None:
    registry = build_benchmark_runner_registry()
    formal_ids = formal_benchmark_ids()

    assert len(formal_ids) > 0
    assert [benchmark_id for benchmark_id in formal_ids if benchmark_id not in registry] == []
    assert [entry.benchmark_id for entry in registry if not registry.has_runner(entry.benchmark_id)] == []
    assert registry.get_runner("robotwin").benchmark_id == "robotwin"
    assert registry.get_runner("libero").benchmark_id == "libero"
    assert len(registry) >= len(formal_ids)


def test_public_cli_does_not_expose_internal_contract_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "contract"
    with pytest.raises(SystemExit) as exc_info:
        eval_cli_main(
            [
                "zoo",
                "benchmark-run",
                "--benchmark-id",
                "vbench",
                "--output-dir",
                str(output_dir),
                "--mode",
                "contract",
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    assert not output_dir.exists()


def test_default_benchmark_execution_runs_embodied_contract(tmp_path: Path) -> None:
    result = run_benchmark_execution("robotwin", output_dir=tmp_path / "robotwin-default", mode="contract")

    assert result.benchmark_id == "robotwin"
    assert result.metadata["contract_only"] is True
    assert result.scorecard_path.is_file()
    assert result.raw_results_path.is_file()


def test_vbench_manifest_runner_materializes_samples_and_contract_result(tmp_path: Path) -> None:
    registry = build_benchmark_runner_registry(MANIFEST_PATH)
    runner = registry.get_runner("vbench")

    assert isinstance(runner, BenchmarkRunner)
    assert isinstance(runner, OfficialBenchmarkRunner)
    manifest = runner.load_manifest()
    plan = runner.materialization_plan()
    sample = next(iter(runner.iter_samples()))
    result = runner.evaluate(output_dir=tmp_path / "vbench", mode="contract")

    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    raw_rows = result.raw_results_path.read_text(encoding="utf-8").splitlines()

    assert manifest["benchmark_id"] == "vbench"
    assert manifest["integration_status"] == "integrated"
    assert plan.benchmark_id == "vbench"
    assert plan.dataset_ids == ()
    assert any("dataset not applicable" in note for note in plan.notes)
    assert sample.sample_id == "vbench:manifest"
    assert sample.metadata["integration_status"] == "integrated"
    assert result.ok is True
    assert result.official_benchmark_verified is False
    assert result.integration_evidence is False
    assert result.to_dict()["artifacts"]["scorecard"].endswith("scorecard.json")
    assert scorecard["benchmark"]["benchmark_id"] == "vbench"
    assert scorecard["benchmark"]["contract_only"] is True
    assert scorecard["benchmark"]["manifest_integration_status"] == "integrated"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert len(raw_rows) == 20


def test_vbench_manifest_runner_runs_official_lifecycle_hooks(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample.mp4").write_text("fake video", encoding="utf-8")
    registry = build_benchmark_runner_registry(MANIFEST_PATH)
    runner = registry.get_runner("vbench")

    prepared = runner.prepare(
        output_dir=tmp_path / "lifecycle",
        mode="contract",
        generated_artifact_dir=generated_dir,
        audit_tag="unit",
    )
    run_result = runner.run(prepared)
    collected = runner.collect(run_result)
    result = runner.normalize(collected)
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert prepared.stage == "prepare"
    assert prepared.metadata["runner"] == "benchmark_zoo_manifest_runner"
    assert prepared.metadata["materialization_plan"]["benchmark_id"] == "vbench"
    assert run_result.stage == "run"
    assert run_result.status == "contract_fixture"
    assert collected.stage == "collect"
    assert collected.data["generated_file_count"] == 1
    assert result.metadata["audit_tag"] == "unit"
    assert scorecard["dataset"]["generated_file_count"] == 1
    assert scorecard["benchmark"]["contract_only"] is True


def test_robotwin_embodied_runner_treats_validation_fixture_as_normalizer_only(tmp_path: Path) -> None:
    registry = build_benchmark_runner_registry(BENCHMARK_ZOO_DIR)

    runner = registry.get_runner("robotwin")
    assert runner.benchmark_id == "robotwin"

    results_path = tmp_path / "robotwin_results.json"
    results_path.write_text(
        json.dumps(
            {
                "results": [
                    {"task": "clean_task", "task_config": "demo_clean", "success_rate": 0.8},
                    {"task": "randomized_task", "task_config": "demo_randomized", "success_rate": 0.6},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = runner.evaluate(
        output_dir=tmp_path / "robotwin",
        mode="official-validation",
        timeout_seconds=10,
        env={"WORLDFOUNDRY_ROBOTWIN_RESULTS_PATH": str(results_path)},
    )

    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(Path(result.artifacts["runner_runtime_report"]).read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in result.raw_results_path.read_text(encoding="utf-8").splitlines()]

    assert result.ok is True
    assert result.official_benchmark_verified is False
    assert result.integration_evidence is False
    assert result.metadata["manifest_integration_status"] == "integrated"
    assert result.metadata["manifest_verification_status"] == "normalizer_only"
    assert result.metadata["runtime"]["clone_dir"].endswith("/thirdparty/RoboTwin")
    assert runtime_report["benchmark_id"] == "robotwin"
    assert runtime_report["run_status"] == "official_results_normalized"
    assert runtime_report["official_benchmark_verified"] is False
    assert runtime_report["integration_evidence"] is False
    assert runtime_report["scorecard_runtime_flags"]["official_benchmark_verified"] is False
    assert runtime_report["scorecard_runtime_flags"]["integration_evidence"] is False
    assert scorecard["benchmark"]["benchmark_id"] == "robotwin"
    assert scorecard["normalization_ok"] is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["run"]["status"] == "normalized_official_results"
    assert scorecard["evaluation"]["available"] is True
    assert scorecard["metrics"]["leaderboard"]["success_rate"] == pytest.approx(0.7)
    assert raw_rows[-1]["metric_id"] == "success_rate"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False


def test_embodied_manifest_runner_normalizes_official_results_path(tmp_path: Path) -> None:
    official_results = tmp_path / "libero_results.json"
    official_results.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "libero_10", "episode_id": "e1", "success": True},
                    {"task_id": "libero_10", "episode_id": "e2", "success": False},
                    {"task_id": "libero_spatial", "episode_id": "e3", "success": "100%"},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "libero",
        output_dir=tmp_path / "libero-normalized",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["evaluation"]["kind"] == "vla_va_wam_official_result_normalizer"
    assert scorecard["metrics"]["leaderboard"]["success"] == pytest.approx(2 / 3)
    assert scorecard["metrics"]["per_task"]["libero_10"]["success"] == pytest.approx(0.5)
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_videoverse_manifest_runner_normalizes_dedicated_official_results_path(tmp_path: Path) -> None:
    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(
        json.dumps(
            {
                "sample-a": {
                    "verification_checks": [
                        {"check_type": "Interaction", "question": "Does the interaction happen?"},
                        {"check_type": "Attribution Correctness", "question": "Is the object red?"},
                    ],
                    "t2v_eval_event_info": {
                        "verification_plan": [
                            {"event_id": 1, "event_description": "First event."},
                            {"event_id": 2, "event_description": "Second event."},
                        ]
                    },
                },
                "sample-b": {
                    "verification_checks": [
                        {"check_type": "Common Sense", "question": "Is the scene plausible?"},
                    ],
                    "t2v_eval_event_info": {
                        "verification_plan": [
                            {"event_id": 1, "event_description": "Only event."},
                        ]
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    official_results = tmp_path / "videoverse_results.json"
    official_results.write_text(
        json.dumps(
            {
                "sample-a": {
                    "verification_checks": [
                        {"check_type": "Interaction", "res": "yes"},
                        {"check_type": "Attribution Correctness", "res": "no"},
                    ],
                    "t2v_eval_event_info": {
                        "verification_plan": [
                            {"event_id": 1, "event_description": "First event."},
                            {"event_id": 2, "event_description": "Second event."},
                        ],
                        "overall_event_processed_res": "AB",
                    },
                },
                "sample-b": {
                    "verification_checks": [
                        {"check_type": "Common Sense", "res": "yes"},
                    ],
                    "t2v_eval_event_info": {
                        "verification_plan": [
                            {"event_id": 1, "event_description": "Only event."},
                        ],
                        "overall_event_processed_res": "B",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-a.mp4").write_bytes(b"fake")
    (generated_dir / "sample-b.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "videoverse",
        output_dir=tmp_path / "videoverse-normalized",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-validation",
        official_results_path=official_results,
        generated_artifact_dir=generated_dir,
        env={
            "WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST": str(prompt_manifest),
            "WORLDFOUNDRY_VIDEOVERSE_DECOMPOSED_PROMPT_MANIFEST": str(prompt_manifest),
        },
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(Path(result.artifacts["runner_runtime_report"]).read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in result.raw_results_path.read_text(encoding="utf-8").splitlines()]

    assert result.ok is True
    assert result.raw_results_path.name == "raw_metric_table.jsonl"
    assert scorecard["normalizer_only"] is True
    assert scorecard["metrics"]["per_metric"]["world_knowledge_consistency"]["normalized_score"] == pytest.approx(1.0)
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 7
    assert {row["metric_id"] for row in raw_rows} == {
        "qa_accuracy",
        "event_coverage",
        "temporal_causality",
        "world_knowledge_consistency",
        "static_scene_consistency",
        "dynamic_event_consistency",
        "videoverse_average",
    }
    assert scorecard["normalization_ok"] is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["evaluation"]["kind"] == "videoverse_official_result_normalizer"
    assert scorecard["metrics"]["leaderboard"]["qa_accuracy"] == pytest.approx(2 / 3)
    assert scorecard["metrics"]["leaderboard"]["event_coverage"] == pytest.approx(2 / 3)
    assert scorecard["metrics"]["leaderboard"]["dynamic_event_consistency"] == pytest.approx(3 / 4)
    assert scorecard["metrics"]["leaderboard"]["static_scene_consistency"] == pytest.approx(1 / 2)
    assert scorecard["metrics"]["leaderboard"]["videoverse_average"] == pytest.approx(2 / 3)
    assert scorecard["dataset"]["result_coverage"]["complete"] is True
    assert scorecard["dataset"]["video_coverage"]["complete"] is True
    assert scorecard["eligibility"]["canonical_suite"] is False
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert runtime_report["run_status"] == "official_results_normalized"
    assert result.metadata["runtime"]["results_path"].endswith("videoverse_results.json")


def test_videoverse_manifest_runner_official_run_executes_in_tree_judge(tmp_path: Path, monkeypatch) -> None:
    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(
        json.dumps(
            {
                "sample-a": {
                    "t2v_following_prompt": {"t2v_prompt": "prompt a"},
                    "verification_checks": [{"check_type": "Interaction", "question": "Q?", "check": True}],
                    "t2v_eval_event_info": {
                        "verification_plan": [{"event_id": 1, "event_description": "event a"}]
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-a.mp4").write_bytes(b"fake")

    monkeypatch.setenv("WORLDFOUNDRY_VIDEOVERSE_JUDGE_BACKEND", "mock")
    monkeypatch.setenv("WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST", str(prompt_manifest))
    monkeypatch.setenv("WORLDFOUNDRY_VIDEOVERSE_DECOMPOSED_PROMPT_MANIFEST", str(prompt_manifest))

    result = run_benchmark_execution(
        "videoverse",
        output_dir=tmp_path / "videoverse-official-run",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-run",
        generated_artifact_dir=generated_dir,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert (tmp_path / "videoverse-official-run" / "eval_res.json").is_file()
    assert scorecard["evaluation"]["kind"] == "videoverse_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert result.metadata["run_status"] == "succeeded"


def test_phyfps_bench_gen_manifest_runner_official_run_executes_in_tree_predictor(tmp_path: Path) -> None:
    prompt_manifest = tmp_path / "prompts.txt"
    prompt_manifest.write_text("A dog runs across a lawn.\n", encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "0001.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "phyfps-bench-gen",
        output_dir=tmp_path / "phyfps-official-run",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-run",
        generated_artifact_dir=generated_dir,
        env_overrides={
            "WORLDFOUNDRY_PHYFPS_PREDICT_BACKEND": "mock",
            "WORLDFOUNDRY_PHYFPS_META_FPS": "24",
            "WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST": str(prompt_manifest),
        },
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert (tmp_path / "phyfps-official-run" / "results.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "phyfps_bench_gen_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert result.metadata["run_status"] == "official_results_normalized"


def test_visual_chronometer_manifest_runner_official_run_executes_in_tree_predictor(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "visual-chronometer",
        output_dir=tmp_path / "visual-chronometer-official-run",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-run",
        generated_artifact_dir=generated_dir,
        env_overrides={
            "WORLDFOUNDRY_VISUAL_CHRONOMETER_PREDICT_BACKEND": "mock",
        },
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert (tmp_path / "visual-chronometer-official-run" / "results.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "visual_chronometer_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert result.metadata["run_status"] == "succeeded"


def test_physvidbench_manifest_runner_official_run_executes_in_tree_judge(tmp_path: Path) -> None:
    repo_root = _official_repo_dir("PhysVidBenchCode")
    if not repo_root.is_dir():
        return
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "0000.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "physvidbench",
        output_dir=tmp_path / "physvidbench-official-run",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-run",
        generated_artifact_dir=generated_dir,
        env_overrides={
            "WORLDFOUNDRY_PHYSVIDBENCH_JUDGE_BACKEND": "mock",
            "WORLDFOUNDRY_PHYSVIDBENCH_ROOT": str(repo_root),
        },
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert (tmp_path / "physvidbench-official-run" / "output.csv").is_file()
    assert scorecard["evaluation"]["kind"] == "physvidbench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert result.metadata["run_status"] == "succeeded"


def test_physics_iq_manifest_runner_rejects_metric_only_fixture_as_official_results(tmp_path: Path) -> None:
    sample_path = benchmark_task_sample_path("physics-iq")
    assert sample_path is not None
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "0001_perspective-left_trimmed-ball-and-block-fall.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "physics-iq",
        output_dir=tmp_path / "physics-iq-normalized",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-validation",
        generated_artifact_dir=generated_dir,
        official_results_path=sample_path,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert result.integration_evidence is False
    assert result.metadata["normalization_ok"] is False
    assert result.metadata["normalizer_only"] is True
    assert scorecard.get("normalization_ok") is not True
    assert result.metadata["run_status"] == "official_results_missing_scores"


def test_phyeduvideo_manifest_runner_rejects_generic_official_run(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "Id1_T01.mp4").write_bytes(b"fake")

    with pytest.raises(ValueError, match="does not currently accept generic generated artifacts"):
        run_benchmark_execution(
            "phyeduvideo",
            output_dir=tmp_path / "phyeduvideo-official-run",
            manifest_path=BENCHMARK_ZOO_DIR,
            mode="official-run",
            generated_artifact_dir=generated_dir,
        )


def test_phyground_manifest_runner_rejects_generic_official_run(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "ball_fall_0001.mp4").write_bytes(b"fake")

    with pytest.raises(ValueError, match="does not currently accept generic generated artifacts"):
        run_benchmark_execution(
            "phyground",
            output_dir=tmp_path / "phyground-official-run",
            manifest_path=BENCHMARK_ZOO_DIR,
            mode="official-run",
            generated_artifact_dir=generated_dir,
        )


def test_phygenbench_manifest_runner_official_run_executes_in_tree_judge(tmp_path: Path) -> None:
    repo_root = _official_repo_dir("phygenbench")
    if not repo_root.is_dir():
        return
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "output_video_1.mp4").write_bytes(b"fake")

    result = run_benchmark_execution(
        "phygenbench",
        output_dir=tmp_path / "phygenbench-official-run",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-run",
        generated_artifact_dir=generated_dir,
        env_overrides={
            "WORLDFOUNDRY_PHYGENBENCH_JUDGE_BACKEND": "mock",
            "WORLDFOUNDRY_PHYGENBENCH_ROOT": str(repo_root),
        },
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert (tmp_path / "phygenbench-official-run" / "phygenbench_results.json").is_file()
    assert scorecard["evaluation"]["kind"] == "phygenbench_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert result.metadata["run_status"] == "succeeded"


def test_camerabench_manifest_runner_uses_strict_specialized_normalizer(tmp_path: Path) -> None:
    data_root = tmp_path / "CameraBench"
    video_dir = data_root / "videos_gif"
    video_dir.mkdir(parents=True)
    (data_root / "test.jsonl").write_text('{"path": "videos/sample-a.mp4", "labels": ["pan"]}\n', encoding="utf-8")
    (video_dir / "sample-a.gif").write_bytes(b"GIF89a")
    official_results = tmp_path / "camerabench_all_results.json"
    official_results.write_text(
        json.dumps(
            {
                "overall_average_precision": 0.8,
                "overall_roc_auc": 0.6,
                "evaluated_splits": 1,
                "overall_binary_acc": 0.9,
                "overall_question_acc": 0.7,
                "overall_retrieval_text": 0.5,
                "overall_retrieval_image": 0.7,
                "overall_retrieval_group": 0.9,
                "results_by_split": {"demo_split": {"num_samples": 1}},
                "results": [
                    {
                        "sample_id": "sample-a",
                        "gen_match": 0.8,
                        "spice": 0.6,
                        "cider": 0.6,
                        "bleu2": 0.6,
                        "rouge_l": 0.6,
                        "meteor": 0.6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "camerabench",
        output_dir=tmp_path / "camerabench-normalized",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-validation",
        official_results_path=official_results,
        benchmark_data_root=data_root,
        env={"WORLDFOUNDRY_CAMERABENCH_STRICT": "1"},
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(Path(result.artifacts["runner_runtime_report"]).read_text(encoding="utf-8"))

    assert result.ok is True
    assert result.metadata["specialized_normalizer"] is True
    assert result.metadata["normalizer_only"] is False
    assert result.official_benchmark_verified is False
    assert result.integration_evidence is True
    assert runtime_report["run_status"] == "official_results_normalized"
    assert runtime_report["returncode"] == 0
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["validation"]["normalizer_only"] is False
    assert scorecard["eligibility"]["full_suite_valid"] is True
    assert scorecard["dataset"]["coverage"]["complete"] is True
    assert scorecard["metrics"]["leaderboard"]["camerabench_average"] == pytest.approx(0.72)


def test_run_benchmark_execution_contract_fixture_keeps_official_flags_false(tmp_path: Path) -> None:
    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "contract-fixture",
        manifest_path=MANIFEST_PATH,
        mode="contract",
    )

    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.benchmark_id == "vbench"
    assert result.ok is True
    assert result.metadata["mode"] == "contract"
    assert result.metadata["manifest_integration_status"] == "integrated"
    assert scorecard["run"]["status"] == "contract_fixture"
    assert scorecard["benchmark"]["contract_only"] is True
    assert scorecard["benchmark"]["evidence_level"] == "contract_fixture_only"
    assert scorecard["evaluation"]["evidence_level"] == "contract_fixture_only"
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False


def test_manifest_runner_official_verified_executes_manifest_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "benchmarks.yaml"
    external_cache = tmp_path / "external-cache"
    monkeypatch.setenv("WORLDFOUNDRY_EXTERNAL_REPO_CACHE", str(external_cache))
    monkeypatch.setenv("WORLDFOUNDRY_VBENCH_REPO_URL", "https://env.example/vbench.git")
    monkeypatch.setenv("WORLDFOUNDRY_VBENCH_REVISION", "env-revision")
    command = (
        "import json, os; "
        "from pathlib import Path; "
        "root=Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
        "scorecard={'schema_version':'worldfoundry-scorecard',"
        "'official_benchmark_verified':True,'integration_evidence':True,"
        "'run':{'status':'succeeded','command':os.environ['WORLDFOUNDRY_BENCHMARK_COMMAND_KIND']},"
        "'benchmark':{'benchmark_id':os.environ['WORLDFOUNDRY_BENCHMARK_ID']},"
        "'runtime':{'vbench_root':os.environ.get('WORLDFOUNDRY_VBENCH_ROOT'),"
        "'repo_url':os.environ.get('WORLDFOUNDRY_VBENCH_REPO_URL'),"
        "'repo_revision':os.environ.get('WORLDFOUNDRY_VBENCH_REVISION'),"
        "'official_repo_url':os.environ.get('WORLDFOUNDRY_OFFICIAL_REPO_URL'),"
        "'official_repo_revision':os.environ.get('WORLDFOUNDRY_OFFICIAL_REPO_REVISION'),"
        "'official_repo_root':os.environ.get('WORLDFOUNDRY_OFFICIAL_REPO_ROOT'),"
        "'extra':os.environ.get('VBENCH_EXTRA')},"
        "'evaluation':{'available':True,'kind':'official_demo'},"
        "'metrics':{'leaderboard':{'quality':1.0}}}; "
        "(root/'scorecard.json').write_text(json.dumps(scorecard), encoding='utf-8')"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "name": "Official Demo",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [sys.executable, "-c", command],
                            "runtime": {
                                "kind": "external_official_repo",
                                "repo_url": "https://manifest.example/vbench.git",
                                "repo_revision": "manifest-revision",
                                "root_env": "WORLDFOUNDRY_VBENCH_ROOT",
                                "env": {"VBENCH_EXTRA": "manifest-env"},
                                "default_cache_subdir": "github.com_Vchitect_VBench",
                            },
                            "expected_artifacts": ["scorecard.json"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "official",
        manifest_path=manifest_path,
        mode="official-validation",
        timeout_seconds=10,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(result.raw_results_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert result.official_benchmark_verified is True
    assert result.integration_evidence is True
    assert result.metadata["mode"] == "official-validation"
    assert result.metadata["returncode"] == 0
    assert result.metadata["run_status"] == "succeeded"
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["run"]["status"] == "succeeded"
    assert scorecard["runtime"]["vbench_root"] == str(external_cache / "github.com_Vchitect_VBench")
    assert scorecard["runtime"]["repo_url"] == "https://env.example/vbench.git"
    assert scorecard["runtime"]["repo_revision"] == "env-revision"
    assert scorecard["runtime"]["official_repo_url"] == "https://env.example/vbench.git"
    assert scorecard["runtime"]["official_repo_revision"] == "env-revision"
    assert scorecard["runtime"]["official_repo_root"] == str(external_cache / "github.com_Vchitect_VBench")
    assert scorecard["runtime"]["extra"] == "manifest-env"
    assert runtime_report["benchmark_id"] == "vbench"
    assert runtime_report["command_kind"] == "validation"
    assert runtime_report["run_status"] == "succeeded"
    assert runtime_report["returncode"] == 0
    assert runtime_report["runtime"]["repo_url"] == "https://env.example/vbench.git"
    assert runtime_report["runtime"]["repo_revision"] == "env-revision"
    assert runtime_report["runtime"]["clone_dir"] == str(external_cache / "github.com_Vchitect_VBench")
    assert runtime_report["runtime"]["env"] == {"VBENCH_EXTRA": "manifest-env"}
    assert runtime_report["runtime"]["install_commands"] == []
    assert runtime_report["manifest"]["runner"]["runtime"]["root_env"] == "WORLDFOUNDRY_VBENCH_ROOT"
    assert runtime_report["scorecard_runtime_flags"]["official_benchmark_verified"] is True
    assert result.metadata["runner_runtime_spec"]["repo_url"] == "https://env.example/vbench.git"

    kwargs_result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "official-kwargs",
        manifest_path=manifest_path,
        mode="official-validation",
        timeout_seconds=10,
        clone_root=tmp_path / "kw-clone",
        repo_url="https://kwargs.example/vbench.git",
        revision="kwargs-revision",
    )
    kwargs_scorecard = json.loads(kwargs_result.scorecard_path.read_text(encoding="utf-8"))
    kwargs_runtime_report = json.loads(kwargs_result.raw_results_path.read_text(encoding="utf-8"))

    assert kwargs_scorecard["run"]["status"] == "succeeded"
    assert kwargs_scorecard["runtime"]["vbench_root"] == str(tmp_path / "kw-clone")
    assert kwargs_scorecard["runtime"]["repo_url"] == "https://kwargs.example/vbench.git"
    assert kwargs_scorecard["runtime"]["repo_revision"] == "kwargs-revision"
    assert kwargs_runtime_report["runtime"]["clone_dir"] == str(tmp_path / "kw-clone")
    assert kwargs_runtime_report["runtime"]["repo_url"] == "https://kwargs.example/vbench.git"
    assert kwargs_runtime_report["runtime"]["repo_revision"] == "kwargs-revision"


def test_manifest_runner_sets_results_path_runtime_env(tmp_path: Path) -> None:
    manifest_path = tmp_path / "benchmarks.yaml"
    command = (
        "import json, os; "
        "from pathlib import Path; "
        "root=Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
        "scorecard={'schema_version':'worldfoundry-scorecard',"
        "'official_benchmark_verified':True,'integration_evidence':True,"
        "'run':{'status':'succeeded'},"
        "'benchmark':{'benchmark_id':os.environ['WORLDFOUNDRY_BENCHMARK_ID']},"
        "'runtime':{'results_path':os.environ.get('WORLDFOUNDRY_TEST_RESULTS_PATH'),"
        "'artifact_alias':os.environ.get('WORLDFOUNDRY_TEST_ARTIFACT_DIR')},"
        "'evaluation':{'available':True},"
        "'metrics':{'leaderboard':{'quality':1.0}}}; "
        "(root/'scorecard.json').write_text(json.dumps(scorecard), encoding='utf-8')"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [sys.executable, "-c", command],
                            "runtime": {
                                "results_path_env": "WORLDFOUNDRY_TEST_RESULTS_PATH",
                                "generated_artifact_dir_env": "WORLDFOUNDRY_TEST_ARTIFACT_DIR",
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    results_path = tmp_path / "official-results"

    explicit_result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "explicit",
        manifest_path=manifest_path,
        mode="official-validation",
        generated_artifact_dir=generated_dir,
        results_path=results_path,
    )
    explicit_scorecard = json.loads(explicit_result.scorecard_path.read_text(encoding="utf-8"))
    explicit_runtime_report = json.loads(explicit_result.raw_results_path.read_text(encoding="utf-8"))

    assert explicit_scorecard["run"]["status"] == "succeeded"
    assert explicit_scorecard["runtime"]["results_path"] == str(results_path)
    assert explicit_scorecard["runtime"]["artifact_alias"] == str(generated_dir)
    assert explicit_runtime_report["runtime"]["results_path_env"] == "WORLDFOUNDRY_TEST_RESULTS_PATH"
    assert explicit_runtime_report["runtime"]["results_path"] == str(results_path)

    generated_result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "generated-result",
        manifest_path=manifest_path,
        mode="official-validation",
        generated_artifact_dir=generated_dir,
    )
    generated_scorecard = json.loads(generated_result.scorecard_path.read_text(encoding="utf-8"))

    assert generated_scorecard["run"]["status"] == "succeeded"
    assert generated_scorecard["runtime"]["results_path"] is None
    assert generated_scorecard["runtime"]["artifact_alias"] == str(generated_dir)


def test_manifest_runner_normalizes_official_results_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(benchmark_runner_module.SPECIALIZED_RESULT_NORMALIZER_SCRIPTS, "vbench", raising=False)
    manifest_path = tmp_path / "benchmarks.yaml"
    results_path = tmp_path / "official_results.csv"
    results_path.write_text("sample_id,quality\nsample-a,1.0\nsample-b,0.5\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {"verification_status": "verified"},
                        "metrics": [
                            {
                                "id": "quality",
                                "leaderboard_key": "quality",
                                "official_results": {
                                    "source_fields": ["quality"],
                                    "required_columns": ["sample_id", "quality"],
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "official-results",
        manifest_path=manifest_path,
        mode="official-validation",
        results_path=results_path,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in result.raw_results_path.read_text(encoding="utf-8").splitlines()]

    assert result.ok is True
    assert result.raw_results_path.name == "raw_metric_table.jsonl"


def test_manifest_runner_normalizer_mode_normalizes_official_results_path(tmp_path: Path) -> None:
    results_path = tmp_path / "official_results.csv"
    results_path.write_text(
        "sample_id,dynamics_quality\nsample-a,1.0\nsample-b,0.5\n",
        encoding="utf-8",
    )
    result = run_benchmark_execution(
        "devil-dynamics",
        output_dir=tmp_path / "normalizer",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="normalizer",
        official_results_path=results_path,
    )

    assert result.ok is True
    assert result.metadata["mode"] == "normalizer"
    assert result.metadata["normalizer_only"] is True
    assert result.raw_results_path.name == "raw_metric_table.jsonl"


def test_manifest_runner_uses_genai_bench_pairwise_normalizer(tmp_path: Path) -> None:
    results_path = tmp_path / "genai_results.jsonl"
    results_path.write_text(
        "\n".join(
            [
                json.dumps({"task": "video_generation", "human_label": "A>B", "prediction": "A>B"}),
                json.dumps({"task": "image_generation", "human_label": "B>A", "prediction": "A>B"}),
            ]
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample.mp4").write_bytes(b"fake video")

    result = run_benchmark_execution(
        "genai-bench",
        output_dir=tmp_path / "genai",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-validation",
        official_results_path=results_path,
        generated_artifact_dir=generated_dir,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert result.metadata["specialized_normalizer"] is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["leaderboard"]["pairwise_accuracy"] == pytest.approx(0.5)
    assert scorecard["metrics"]["leaderboard"]["video_preference_accuracy"] == pytest.approx(1.0)
    assert scorecard["metrics"]["leaderboard"]["image_generation_preference_accuracy"] == pytest.approx(0.0)
    assert scorecard["metrics"]["leaderboard"]["genai_bench_average"] == pytest.approx(0.5)


def test_manifest_runner_uses_videoscore_specialized_normalizer(tmp_path: Path) -> None:
    upstream_results = tmp_path / "eval_video_feedback_videoscore.json"
    upstream_results.write_text(
        json.dumps(
            [
                {
                    "video_path": "sample-a.mp4",
                    "raw_score": 2.0,
                    "ans": "[2, 3, 4, 1, 2]",
                },
                {
                    "video_path": "sample-b.mp4",
                    "raw_score": 3.0,
                    "ans": "[4, 3, 2, 3, 3]",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "videoscore",
        output_dir=tmp_path / "videoscore-normalized",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="official-validation",
        official_results_path=upstream_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert result.metadata["specialized_normalizer"] is True
    assert result.metadata["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["integration_evidence"] is False
    assert scorecard["metrics"]["leaderboard"]["visual_quality"] == pytest.approx(3.0)
    assert scorecard["metrics"]["leaderboard"]["text_to_video_alignment"] == pytest.approx(2.0)
    assert scorecard["metrics"]["leaderboard"]["videoscore_average"] == pytest.approx(2.7)


def test_manifest_runner_normalizer_mode_requires_existing_results_path(tmp_path: Path) -> None:
    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "missing-normalizer",
        manifest_path=BENCHMARK_ZOO_DIR,
        mode="normalizer",
        official_results_path=tmp_path / "missing.csv",
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert result.metadata["run_status"] == "missing_official_results_path"
    assert "official results path does not exist" in scorecard["run"]["error"]


def test_manifest_runner_normalizes_long_form_official_results_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        benchmark_runner_module.SPECIALIZED_RESULT_NORMALIZER_SCRIPTS,
        "devil-dynamics",
        raising=False,
    )
    manifest_path = tmp_path / "benchmarks.yaml"
    results_path = tmp_path / "official_results.csv"
    results_path.write_text(
        "sample_id,metric_id,score\nsample-a,dynamics_quality,1.0\nsample-b,dynamics_quality,0.5\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "devil-dynamics",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {"verification_status": "verified"},
                        "metrics": [{"id": "dynamics_quality", "leaderboard_key": "dynamics_quality"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "devil-dynamics",
        output_dir=tmp_path / "official-long-results",
        manifest_path=manifest_path,
        mode="official-validation",
        official_results_path=results_path,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["metrics"]["leaderboard"]["dynamics_quality"] == 0.75
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 1


def test_manifest_runner_uses_physics_normalizer_for_official_results_path(tmp_path: Path) -> None:
    official_results = tmp_path / "videophy2_results.json"
    official_results.write_text(
        json.dumps(
            {
                "results": [
                    {"id": "sample-1", "sa": 5, "pc": 4, "physics_rules_followed": ["gravity"], "physics_rules_unfollowed": []},
                    {"id": "sample-2", "sa": 2, "pc": 5, "physics_rules_followed": [], "physics_rules_unfollowed": ["momentum"]},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "videophy2",
        output_dir=tmp_path / "videophy2-normalized",
        manifest_path=MANIFEST_PATH,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in result.raw_results_path.read_text(encoding="utf-8").splitlines()]
    by_metric = {row["metric_id"]: row for row in rows}

    assert result.ok is True
    assert result.raw_results_path.name == "raw_metric_table.jsonl"
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "videophy2_result_normalizer"
    assert result.metadata["contract_only"] is False
    assert scorecard["metrics"]["leaderboard"]["joint_score"] == 0.5
    assert by_metric["semantic_adherence"]["score"] == 0.7
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_manifest_runner_uses_worldbench_specialized_normalizer(tmp_path: Path) -> None:
    official_results = tmp_path / "worldbench_results.json"
    official_results.write_text(
        json.dumps(
            {
                "summary": {
                    "video_based_accuracy": "80%",
                    "text_based_accuracy": 0.5,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "worldbench",
        output_dir=tmp_path / "worldbench-normalized",
        manifest_path=MANIFEST_PATH,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "official_worldbench_result_normalizer"
    assert scorecard["metrics"]["leaderboard"]["foreground_miou"] == pytest.approx(0.8)
    assert scorecard["metrics"]["leaderboard"]["text_based_accuracy"] == pytest.approx(0.5)
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_manifest_runner_uses_vmbench_specialized_normalizer(tmp_path: Path) -> None:
    official_results = tmp_path / "vmbench_results.json"
    official_results.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "perceptible_amplitude_score": 0.8,
                    "object_integrity_score": 0.7,
                    "temporal_coherence_score": 0.6,
                    "commonsense_adherence_score": 0.5,
                    "motion_smoothness_score": 0.4,
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vmbench",
        output_dir=tmp_path / "vmbench-normalized",
        manifest_path=MANIFEST_PATH,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "official_vmbench"
    assert scorecard["metrics"]["leaderboard"]["vmbench_average"] == pytest.approx(0.6)
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_manifest_runner_uses_videobench_specialized_normalizer(tmp_path: Path) -> None:
    official_results = tmp_path / "videobench_results.json"
    official_results.write_text(
        json.dumps(
            {
                "imaging_quality": {
                    "average_scores": {"demo-model": 4.0},
                    "scores": {"0": {"prompt_en": "a close up of grapes", "demo-model": 4}},
                },
                "color": {
                    "average_scores": {"demo-model": 2.0},
                    "scores": {"0": {"prompt_en": "a red bird", "demo-model": 2}},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "video-bench",
        output_dir=tmp_path / "videobench-normalized",
        manifest_path=MANIFEST_PATH,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "official_videobench"
    assert scorecard["metrics"]["leaderboard"]["imaging_quality"] == pytest.approx(4.0)
    assert scorecard["metrics"]["per_metric"]["imaging_quality"]["normalized_score"] == pytest.approx(0.75)
    assert scorecard["metrics"]["leaderboard"]["videobench_average"] == pytest.approx(0.625)
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_manifest_runner_uses_t2v_compbench_specialized_normalizer(tmp_path: Path) -> None:
    official_results = tmp_path / "t2v_compbench_results.json"
    official_results.write_text(json.dumps({"t2v_compbench_average": 0.42}, sort_keys=True), encoding="utf-8")

    result = run_benchmark_execution(
        "t2v-compbench",
        output_dir=tmp_path / "t2v-compbench-normalized",
        manifest_path=MANIFEST_PATH,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "official_t2v_compbench"
    assert scorecard["metrics"]["leaderboard"]["t2v_compbench_average"] == pytest.approx(0.42)
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_manifest_runner_uses_worldmodelbench_specialized_normalizer(tmp_path: Path) -> None:
    official_results = tmp_path / "worldmodelbench_results.json"
    official_results.write_text(
        json.dumps(
            {
                "preds": {
                    "sample_a": {
                        "instruction": ["Score: 3"],
                        "common_sense": ["No", "Yes"],
                        "physical_laws": ["No", "No", "Yes", "No", "No"],
                    },
                    "sample_b": {
                        "instruction": ["Score: 1"],
                        "common_sense": ["No", "No"],
                        "physical_laws": ["No", "Yes", "No", "No", "No"],
                    },
                },
                "accs": {
                    "instruction": [3, 1],
                    "common_sense": [True, False, True, True],
                    "physical_laws": [True, True, False, True, True, True, False, True, True, True],
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "worldmodelbench",
        output_dir=tmp_path / "worldmodelbench-normalized",
        manifest_path=MANIFEST_PATH,
        mode="official-validation",
        official_results_path=official_results,
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert scorecard["normalizer_only"] is True
    assert scorecard["normalization_ok"] is True
    assert scorecard["evaluation"]["kind"] == "official_worldmodelbench"
    assert scorecard["metrics"]["leaderboard"]["instruction_following"] == pytest.approx(2.0)
    assert scorecard["metrics"]["per_metric"]["instruction_following"]["normalized_score"] == pytest.approx(2 / 3)
    assert scorecard["metrics"]["leaderboard"]["world_model_average"] == pytest.approx(7.5)
    assert scorecard["metrics"]["per_metric"]["world_model_average"]["normalized_score"] == pytest.approx(0.75)
    assert "runner_runtime_report" in scorecard["artifacts"]


def test_manifest_runner_passes_worldmodelbench_data_root_and_judge_to_official_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep this regression focused on the manifest runner; WorldModelBench also
    # has a workspace-registry command that would otherwise replace the fixture.
    monkeypatch.setattr(workspace_registry, "CLI_RUNNERS", {})
    fake_runner = tmp_path / "fake_worldmodelbench_runner.py"
    fake_runner.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "out = Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR'])",
                "scorecard = {",
                "    'schema_version': 'worldfoundry-scorecard',",
                "    'official_benchmark_verified': True,",
                "    'integration_evidence': True,",
                "    'run': {'status': 'official_verified'},",
                "    'benchmark': {'benchmark_id': 'worldmodelbench'},",
                "    'evaluation': {'available': True},",
                "    'metrics': {'leaderboard': {'world_model_average': 1.0}, 'per_metric': {}, 'summary': {}},",
                "    'artifacts': {'scorecard': str(out / 'scorecard.json')},",
                "    'env_seen': {",
                "        'data_root': os.environ.get('WORLDFOUNDRY_WORLDMODELBENCH_DATA_ROOT'),",
                "        'generic_data_root': os.environ.get('WORLDFOUNDRY_BENCHMARK_DATA_ROOT'),",
                "        'judge': os.environ.get('WORLDFOUNDRY_WORLDMODELBENCH_JUDGE'),",
                "        'generated_artifact_dir': os.environ.get('WORLDFOUNDRY_GENERATED_ARTIFACT_DIR'),",
                "    },",
                "}",
                "(out / 'scorecard.json').write_text(json.dumps(scorecard), encoding='utf-8')",
                "(out / 'raw_metric_table.jsonl').write_text('', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "worldmodelbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "verification_status": "verified",
                        "runner": {
                            "verification_status": "verified",
                            "run_command": [sys.executable, str(fake_runner)],
                            "runtime": {
                                "kind": "external_official_repo",
                                "repo_url": "https://github.com/WorldModelBench-Team/WorldModelBench",
                                "root_env": "WORLDFOUNDRY_WORLDMODELBENCH_ROOT",
                                "generated_artifact_dir_env": "WORLDFOUNDRY_GENERATED_ARTIFACT_DIR",
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    data_root = tmp_path / "worldmodelbench-data"
    data_root.mkdir()
    # A dataset-side results file may be stale or even be annotations.  It
    # must not short-circuit a requested official run.
    (data_root / "results.json").write_text('{"stale": true}', encoding="utf-8")
    generated_dir = tmp_path / "generated-videos"
    generated_dir.mkdir()
    judge_checkpoint = tmp_path / "vila-judge-ckpt"
    judge_checkpoint.write_text("fake checkpoint marker", encoding="utf-8")

    result = run_benchmark_execution(
        "worldmodelbench",
        output_dir=tmp_path / "worldmodelbench-official-run",
        manifest_path=manifest_path,
        mode="official-run",
        generated_artifact_dir=generated_dir,
        benchmark_data_root=data_root,
        env_overrides={"WORLDFOUNDRY_WORLDMODELBENCH_JUDGE": str(judge_checkpoint)},
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is True
    assert scorecard["env_seen"] == {
        "data_root": str(data_root),
        "generic_data_root": str(data_root),
        "judge": str(judge_checkpoint),
        "generated_artifact_dir": str(generated_dir),
    }


def test_manifest_runner_official_verified_without_command_writes_failure_scorecard(tmp_path: Path) -> None:
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {"verification_status": "verified"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "missing",
        manifest_path=manifest_path,
        mode="official-validation",
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert result.official_benchmark_verified is False
    assert result.integration_evidence is False
    assert result.metadata["run_status"] == "missing_official_command"
    assert scorecard["run"]["error"] == "missing validation_command"


def test_manifest_runner_external_official_repo_without_repo_url_reports_missing_runtime_spec(tmp_path: Path) -> None:
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [sys.executable, "-c", "raise SystemExit('should not run')"],
                            "runtime": {
                                "kind": "external_official_repo",
                                "root_env": "WORLDFOUNDRY_VBENCH_ROOT",
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_execution(
        "vbench",
        output_dir=tmp_path / "missing-runtime",
        manifest_path=manifest_path,
        mode="official-validation",
    )
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(result.raw_results_path.read_text(encoding="utf-8"))

    assert result.ok is False
    assert result.metadata["run_status"] == "missing_official_runtime_spec"
    assert scorecard["run"]["status"] == "missing_official_runtime_spec"
    assert scorecard["run"]["error"] == "missing repo_url for external_official_repo runtime"
    assert runtime_report["run_status"] == "missing_official_runtime_spec"
    assert runtime_report["runtime"]["kind"] == "external_official_repo"
    assert runtime_report["runtime"]["repo_url"] is None

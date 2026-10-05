from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.cli import main
from worldfoundry.cli.tui_discovery import build_model_benchmark_command
from worldfoundry.evaluation.api import ArtifactRef, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    BENCHMARK_GENERATION_ADAPTERS,
    materialize_vbench_generation_requests,
)
from worldfoundry.evaluation.tasks.execution.orchestration.fidelity import EvaluationFidelity
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import (
    ModelBenchmarkRunRequest,
    _materialize_generated_artifacts,
    _model_benchmark_run_summary,
)
from worldfoundry.evaluation.tasks.execution.orchestration.service import (
    ModelBenchmarkIntent,
    ReproduceIntent,
    ScoreArtifactsIntent,
    ScoreResultsIntent,
    execute_prepared_evaluation,
    prepare_evaluation,
)
from worldfoundry.evaluation.utils import write_jsonl


def test_generation_adapters_are_declarative_and_include_wrbench() -> None:
    assert "wrbench" in BENCHMARK_GENERATION_ADAPTERS
    assert "ipv-bench" in BENCHMARK_GENERATION_ADAPTERS
    assert BENCHMARK_GENERATION_ADAPTERS["wrbench"].artifact_materializer == (
        "worldfoundry.evaluation.tasks.execution.runners.wrbench.wrbench_prompts:"
        "copy_wrbench_generated_videos"
    )
    assert BENCHMARK_GENERATION_ADAPTERS["videophy"].artifact_materializer is not None
    assert "vbench" in BENCHMARK_GENERATION_ADAPTERS


def test_vbench_generation_provider_deduplicates_official_prompts() -> None:
    requests = materialize_vbench_generation_requests()

    assert len(requests) == 944
    assert len({request.inputs["official_video_name"] for request in requests}) == 944
    assert requests[0].inputs["official_video_name"].endswith("-0.mp4")


def test_generic_artifact_layout_uses_official_video_name(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    write_jsonl(
        generation_dir / "requests.jsonl",
        [
            {
                "sample_id": "prompt/unsafe",
                "inputs": {"official_video_name": "official/expected.mp4"},
            }
        ],
        atomic=False,
    )
    result = GenerationResult(
        sample_id="prompt/unsafe",
        model_id="fixture",
        status="succeeded",
        artifacts={"generated_video": ArtifactRef.from_uri(str(source), kind="generated_video")},
    )
    write_jsonl(generation_dir / "results.jsonl", [result.to_dict()], atomic=False)

    count, placeholders = _materialize_generated_artifacts(
        generation_output_dir=generation_dir,
        generated_artifact_dir=tmp_path / "artifacts",
        artifact_manifest_path=tmp_path / "artifacts.jsonl",
        output_artifact="generated_video",
        allow_placeholders=False,
    )

    assert count == 1
    assert placeholders == 0
    assert (tmp_path / "artifacts" / "expected.mp4").read_bytes() == b"video"
    assert not (tmp_path / "official").exists()


def test_score_artifacts_preflight_marks_custom_data_non_comparable(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")

    prepared = prepare_evaluation(
        ScoreArtifactsIntent(
            output_dir=tmp_path / "output",
            benchmark_id="vbench",
            artifact_dir=artifacts,
        )
    )

    payload = prepared.to_dict()
    assert prepared.ready
    assert payload["classification"] == "custom_dataset_metric_evaluation"
    assert payload["leaderboard_candidate"] is False
    assert payload["execution"]["request"]["leaderboard_candidate"] is False
    assert payload["provenance"]["claim"]["level"] == "diagnostic"
    assert payload["execution"]["request"]["evaluation_provenance"] == payload["provenance"]


def test_diagnostic_artifacts_cannot_self_promote_to_leaderboard_candidate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")

    prepared = prepare_evaluation(
        ScoreArtifactsIntent(
            output_dir=tmp_path / "output",
            benchmark_id="vbench",
            artifact_dir=artifacts,
            leaderboard_candidate=True,
        )
    )

    assert prepared.ready
    assert prepared.leaderboard_candidate is False
    assert prepared.request is not None
    assert prepared.request.leaderboard_candidate is False
    assert any(issue.code == "leaderboard_candidate_downgraded" for issue in prepared.issues)


def test_official_full_model_benchmark_is_comparable_but_not_exact_reproduction(tmp_path: Path) -> None:
    prepared = prepare_evaluation(
        ModelBenchmarkIntent(
            output_dir=tmp_path / "output",
            model_id="vchitect-2-t2v",
            benchmark_id="vbench",
        )
    )

    payload = prepared.to_dict()
    assert prepared.ready
    assert payload["provenance"]["fidelity"]["data"] == "official"
    assert payload["provenance"]["fidelity"]["evaluation"] == "official"
    assert payload["provenance"]["claim"]["level"] == "benchmark_comparable"
    assert payload["leaderboard_candidate"] is True


def test_model_benchmark_semantic_override_downgrades_claim(tmp_path: Path) -> None:
    prepared = prepare_evaluation(
        ModelBenchmarkIntent(
            output_dir=tmp_path / "output",
            model_id="vchitect-2-t2v",
            benchmark_id="vbench",
            benchmark_parameters={"dimension": "aesthetic_quality"},
        )
    )

    payload = prepared.to_dict()
    assert prepared.ready
    assert payload["provenance"]["fidelity"]["evaluation"] == "modified"
    assert payload["provenance"]["claim"]["level"] == "diagnostic"
    assert payload["leaderboard_candidate"] is False


def test_exact_reproduction_claim_is_derived_from_all_pinned_axes() -> None:
    fidelity = EvaluationFidelity(
        producer="catalog_model",
        generation="pinned",
        data="official",
        evaluation="official",
        runtime="pinned",
        reference="pinned",
    )

    assert fidelity.claim_level == "exact_reproduction"
    assert fidelity.leaderboard_candidate is True


def test_diagnostic_score_validity_is_independent_from_leaderboard_eligibility(tmp_path: Path) -> None:
    fidelity = EvaluationFidelity(
        producer="imported_artifacts",
        generation="not_applicable",
        data="custom",
        evaluation="official",
        runtime="compatible",
    )
    summary = _model_benchmark_run_summary(
        request=ModelBenchmarkRunRequest(
            output_dir=tmp_path,
            benchmark_id="vbench",
            benchmark_manifest_path=tmp_path,
            model_id="user-artifacts",
            leaderboard_candidate=False,
            evaluation_provenance=fidelity.to_dict(),
        ),
        status="succeeded",
        mode="official-run",
        root=tmp_path,
        materialized_count=1,
        placeholder_count=0,
        generation_result=None,
        benchmark_payload={"ok": True},
        artifacts={},
    )

    assert summary["eligibility"]["score_valid"] is True
    assert summary["eligibility"]["leaderboard_valid"] is False
    assert summary["eligibility"]["blocking_reasons"] == [
        "custom data is not leaderboard-comparable"
    ]


def test_custom_artifact_scorecard_stays_out_of_leaderboard(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")
    prepared = prepare_evaluation(
        ScoreArtifactsIntent(
            output_dir=tmp_path / "output",
            benchmark_id="vbench",
            artifact_dir=artifacts,
            benchmark_mode="contract",
        )
    )

    result = execute_prepared_evaluation(prepared)
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert summary["eligibility"]["leaderboard_valid"] is False
    assert "custom data is not leaderboard-comparable" in summary["eligibility"]["blocking_reasons"]


def test_score_results_rejects_catalog_only_metric_without_generic_executor(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text('{"sample_id":"a","status":"succeeded","artifacts":{}}\n', encoding="utf-8")

    prepared = prepare_evaluation(
        ScoreResultsIntent(
            output_dir=tmp_path / "output",
            results_path=results,
            metrics=("clip_score",),
        )
    )

    assert not prepared.ready
    assert any(issue.code == "metric_requires_runner" for issue in prepared.issues)


def test_metric_only_provenance_reaches_existing_results_manifest(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text('{"sample_id":"a","status":"succeeded","artifacts":{}}\n', encoding="utf-8")
    prepared = prepare_evaluation(
        ScoreResultsIntent(
            output_dir=tmp_path / "output",
            results_path=results,
            metrics=("artifact_count",),
        )
    )

    result = execute_prepared_evaluation(prepared)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    scorecard = json.loads(result.scorecard_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert manifest["evaluation_provenance"] == prepared.fidelity.to_dict()
    assert manifest["evaluation_provenance"]["claim"]["level"] == "diagnostic"
    assert scorecard["provenance"] == prepared.fidelity.to_dict()
    assert scorecard["eligibility"]["score_valid"] is True
    assert scorecard["eligibility"]["leaderboard_valid"] is False


def test_reproduction_recipe_compiles_to_locked_model_benchmark(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text('{"sample_id":"a","inputs":{"prompt":"hello"}}\n', encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """
schema_version: worldfoundry-reproduction-recipe-v1
id: vbench-vchitect-fixture
benchmark:
  id: vbench
  revision: benchmark-commit
model:
  id: vchitect-2-t2v
  revision: model-commit
generation:
  num_frames: 16
data:
  requests_path: requests.jsonl
evaluation:
  parameters:
    prompt_manifest: official-prompts.json
reference:
  score: 0.5
""".lstrip(),
        encoding="utf-8",
    )

    prepared = prepare_evaluation(ReproduceIntent(recipe_path=recipe, output_dir=tmp_path / "output"))

    payload = prepared.to_dict()
    assert prepared.ready
    assert payload["classification"] == "reproduction"
    assert payload["config_sources"]["recipe_sha256"]
    assert payload["config_sources"]["data"]["requests_sha256"]
    assert payload["execution"]["request"]["model_parameters"]["num_frames"] == 16
    assert payload["execution"]["request"]["model_parameters"]["revision"] == "model-commit"
    assert payload["execution"]["request"]["benchmark_parameters"]["prompt_manifest"] == "official-prompts.json"
    assert payload["execution"]["request"]["benchmark_parameters"]["revision"] == "benchmark-commit"
    assert payload["execution"]["request"]["requests_path"] == str(requests)


def _write_reproduction_profile(root: Path, *, profile_id: str, default_for_benchmark: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{profile_id}.yaml"
    path.write_text(
        f"""
schema_version: worldfoundry-reproduction-recipe-v1
id: {profile_id}
default_for_benchmark: {str(default_for_benchmark).lower()}
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
  parameters:
    dimension: aesthetic_quality
reference:
  scope: bounded_smoke_reproduction
  leaderboard_valid: false
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_reproduction_profile_resolves_model_and_config(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    _write_reproduction_profile(profile_root, profile_id="vbench-zeroscope-aesthetic")
    prepared = prepare_evaluation(
        ReproduceIntent(
            profile_id="vbench-zeroscope-aesthetic",
            profile_root=profile_root,
            output_dir=tmp_path / "output",
        )
    )

    payload = prepared.to_dict()
    assert prepared.ready
    assert payload["config_sources"]["recipe_id"] == "vbench-zeroscope-aesthetic"
    assert payload["execution"]["request"]["model_id"] == "zeroscope"
    assert payload["execution"]["request"]["benchmark_id"] == "vbench"
    assert payload["execution"]["request"]["num_samples"] == 1
    assert payload["execution"]["request"]["model_parameters"]["num_frames"] == 8
    assert payload["leaderboard_candidate"] is False
    assert payload["execution"]["request"]["leaderboard_candidate"] is False


def test_benchmark_reproduction_resolves_declared_default(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    _write_reproduction_profile(profile_root, profile_id="vbench-zeroscope-aesthetic")
    prepared = prepare_evaluation(
        ReproduceIntent(benchmark_id="vbench", profile_root=profile_root, output_dir=tmp_path / "output")
    )

    assert prepared.ready
    assert prepared.config_sources["recipe_id"] == "vbench-zeroscope-aesthetic"


def test_score_cli_plan_only_returns_prepared_request(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")

    exit_code = main(
        [
            "score",
            "--benchmark",
            "vbench",
            "--artifacts",
            str(artifacts),
            "--output-dir",
            str(tmp_path / "output"),
            "--plan-only",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["intent_kind"] == "score_artifacts"


def test_tui_routes_existing_artifacts_through_score_intent(tmp_path: Path) -> None:
    command = build_model_benchmark_command(
        model_id="unused-for-existing-artifacts",
        benchmark_id="vbench",
        generated_artifact_dir=tmp_path / "videos",
        output_dir=tmp_path / "scores",
        json_output=True,
    )

    assert command[3] == "score"
    assert "--artifacts" in command
    assert "--model" not in command

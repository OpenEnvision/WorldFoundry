from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from worldfoundry.evaluation.reporting.validation import validate_contract_file
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.runners.physics_iq import physics_iq_runtime
from worldfoundry.evaluation.tasks.execution.runners.physics_iq import (
    run_physics_iq_official_runner as runner,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.official import raw_metrics
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
    VIEWS,
    resolve_descriptions_path,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.protocols import ORIGINAL


def _original_descriptions() -> Path:
    return resolve_descriptions_path(spec=ORIGINAL)


def test_select_complete_scenario_records_supports_full_and_one_scenario() -> None:
    descriptions = _original_descriptions()

    full_records, full_scenarios = physics_iq_runtime.select_complete_scenario_records(
        descriptions_path=descriptions,
        protocol=ORIGINAL,
        limit=None,
    )
    bounded_records, bounded_scenarios = physics_iq_runtime.select_complete_scenario_records(
        descriptions_path=descriptions,
        protocol=ORIGINAL,
        limit=3,
    )

    assert len(full_records) == 198
    assert len(full_scenarios) == 66
    assert len(bounded_records) == 3
    assert len(bounded_scenarios) == 1
    assert {record["scenario"].split("_", 3)[1] for record in bounded_records} == set(VIEWS)


@pytest.mark.parametrize("limit", [0, 1, 2, 4, 199, 201])
def test_select_complete_scenario_records_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="three-view|exceeds"):
        physics_iq_runtime.select_complete_scenario_records(
            descriptions_path=_original_descriptions(),
            protocol=ORIGINAL,
            limit=limit,
        )


def test_select_complete_scenario_records_rejects_semantically_incomplete_group(
    tmp_path: Path,
) -> None:
    descriptions = tmp_path / "descriptions.csv"
    with descriptions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scenario", "description", "category", "generated_video_name"),
        )
        writer.writeheader()
        for index in range(1, 4):
            event = f"event-{index}.mp4"
            writer.writerow(
                {
                    "scenario": f"{index:04d}_perspective-left_take-1_{event}",
                    "description": "test",
                    "category": "test",
                    "generated_video_name": f"{index:04d}_perspective-left_{event}",
                }
            )

    with pytest.raises(ValueError, match="exactly one left, center, and right"):
        physics_iq_runtime.select_complete_scenario_records(
            descriptions_path=descriptions,
            protocol=ORIGINAL,
            limit=3,
        )


def test_stage_generated_videos_only_requires_selected_records(tmp_path: Path) -> None:
    records, _ = physics_iq_runtime.select_complete_scenario_records(
        descriptions_path=_original_descriptions(),
        protocol=ORIGINAL,
        limit=3,
    )
    source = tmp_path / "source"
    source.mkdir()
    for record in records:
        (source / record["generated_video_name"]).write_bytes(b"selected")
    (source / "9999_unselected.mp4").write_bytes(b"unselected")

    staging = tmp_path / "staging"
    summary = physics_iq_runtime.stage_generated_videos(
        source_dir=source,
        staging_dir=staging,
        records=records,
        validate=False,
    )

    assert summary["expected_count"] == 3
    assert summary["source_count"] == 4
    assert summary["staged_count"] == 3
    assert {path.name for path in staging.iterdir()} == {
        record["generated_video_name"] for record in records
    }
    assert (source / "9999_unselected.mp4").read_bytes() == b"unselected"


def test_raw_metric_engine_schedules_only_selected_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    generated = tmp_path / "generated"
    real.mkdir()
    generated.mkdir()
    (generated / "0001_placeholder.mp4").write_bytes(b"video")
    for scenario_index, scenario in enumerate(("event-a.mp4", "event-b.mp4")):
        for view_index, view in enumerate(VIEWS):
            first_id = scenario_index * 10 + view_index + 1
            second_id = first_id + 100
            (real / f"{first_id:04d}_testing-videos_30FPS_{view}_take-1_{scenario}").touch()
            (real / f"{second_id:04d}_testing-videos_30FPS_{view}_take-2_{scenario}").touch()

    calls: list[str] = []

    def fake_process_view(paths, view, start_frame, end_frame, consider_frames):
        del paths, start_frame, end_frame, consider_frames
        calls.append(view)
        return {f"test_metric_{view}": 1.0}

    monkeypatch.setattr(raw_metrics, "get_video_frame_count", lambda _path: 150)
    monkeypatch.setattr(raw_metrics, "process_view", fake_process_view)
    output = tmp_path / "raw_metrics.csv"
    raw_metrics.process_videos(
        real_folder=str(real),
        generated_folder=str(generated),
        binary_real_folder=str(tmp_path / "real-masks"),
        binary_generated_folder=str(tmp_path / "generated-masks"),
        csv_file_path=str(output),
        fps=30,
        n_processes=0,
        selected_scenarios={"event-a.mp4"},
    )

    assert set(calls) == set(VIEWS)
    assert len(calls) == 3
    csv_text = output.read_text(encoding="utf-8")
    assert "event-a.mp4" in csv_text
    assert "event-b.mp4" not in csv_text


def test_runtime_wires_bounded_records_to_staging_and_raw_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, scenarios = physics_iq_runtime.select_complete_scenario_records(
        descriptions_path=_original_descriptions(),
        protocol=ORIGINAL,
        limit=3,
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    for record in records:
        (generated / record["generated_video_name"]).write_bytes(b"video")
    reference_videos = tmp_path / "reference-videos"
    reference_masks = tmp_path / "reference-masks"
    generated_masks = tmp_path / "generated-masks"
    for path in (reference_videos, reference_masks, generated_masks):
        path.mkdir()

    captured: dict[str, object] = {}

    def fake_process_videos(**kwargs) -> None:
        captured.update(kwargs)
        Path(str(kwargs["csv_file_path"])).write_text("raw\n", encoding="utf-8")

    monkeypatch.setattr(physics_iq_runtime, "resolve_dataset_root", lambda *args: tmp_path)
    monkeypatch.setattr(physics_iq_runtime, "detect_video_fps", lambda _path: 30)
    monkeypatch.setattr(
        physics_iq_runtime,
        "_ensure_reference_assets",
        lambda **kwargs: physics_iq_runtime.PhysicsIQDatasetLayout(
            tmp_path,
            30,
            reference_videos,
            reference_masks,
        ),
    )
    monkeypatch.setattr(
        physics_iq_runtime,
        "_ensure_generated_masks",
        lambda **kwargs: generated_masks,
    )
    monkeypatch.setattr(raw_metrics, "process_videos", fake_process_videos)
    monkeypatch.setattr(
        physics_iq_runtime,
        "score_raw_metrics_csv",
        lambda *args, **kwargs: {"final_score_orig": 0.5},
    )

    summary = physics_iq_runtime.run_physics_iq_evaluation(
        physics_iq_runtime.PhysicsIQRunConfig(
            protocol=ORIGINAL,
            dataset_root=tmp_path,
            descriptions_path=_original_descriptions(),
            generated_video_dir=generated,
            output_dir=tmp_path / "output",
            validate_videos=False,
            limit=3,
        )
    )

    assert captured["selected_scenarios"] == set(scenarios)
    assert summary["staging"]["staged_count"] == 3
    assert summary["primary_score"] == 0.5


def test_official_runner_forwards_limit_to_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, physics_iq_runtime.PhysicsIQRunConfig] = {}
    raw_metrics_path = tmp_path / "raw_metrics.csv"

    def fake_run(config: physics_iq_runtime.PhysicsIQRunConfig) -> dict[str, str]:
        captured["config"] = config
        return {
            "raw_metrics_path": str(raw_metrics_path),
            "results_path": str(tmp_path / "official_metrics.json"),
        }

    monkeypatch.setattr(runner, "run_physics_iq_evaluation", fake_run)
    monkeypatch.setattr(runner, "normalize_physics_iq_results", lambda *args, **kwargs: {"ok": True})
    args = argparse.Namespace(
        benchmark_id="physics-iq",
        protocol="original",
        descriptions_file=_original_descriptions(),
        generated_artifact_dir=tmp_path / "generated",
        dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        generated_mask_dir=None,
        raw_metrics_path=None,
        n_processes=0,
        mask_threshold=10,
        skip_video_validation=True,
        lazy_integrity=False,
        limit=3,
    )

    runner.run_official_physics_iq(args)

    assert captured["config"].limit == 3


def test_runner_reads_limit_from_benchmark_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_LIMIT", "3")
    args = runner.parse_args(["--output-dir", "out"])
    assert args.limit == 3


def test_normalized_scorecard_satisfies_public_reporting_schema(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--official-results-path",
            str(bundled_benchmark_asset("physics-iq", "sample_results.csv")),
            "--output-dir",
            str(tmp_path),
            "--limit",
            "3",
        ]
    )

    scorecard = runner.normalize_physics_iq_results(args)
    validation = validate_contract_file(tmp_path / "scorecard.json", kind="scorecard")

    assert scorecard["dataset"]["prompt_count"] == 3
    assert validation["ok"] is True

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.physical_ai_bench_official_impl import (
    TRACKS,
    main,
)
from worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.runtime.physical_ai_bench.conditional_generation import (
    ConditionalGenerationRequest,
    _caption_phrases,
    _depth_reference,
    _prediction_videos,
    _segmentation_reference,
)
from worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.runtime.physical_ai_bench.metrics import (
    aggregate_generation_vqa,
    blur_ssim,
    canny_scores,
    depth_si_rmse,
    normalize_binary_answer,
    parse_option_answer,
    segmentation_scores,
)
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import (
    build_workspace_benchmark_command,
)


def test_registered_tracks_are_generation_only() -> None:
    assert TRACKS == ("generation", "conditional-generation")


def test_canny_binary_metrics_match_global_definition() -> None:
    gt = np.array([[[0, 255], [0, 255]]], dtype=np.uint8)
    pred = np.array([[[0, 255], [255, 0]]], dtype=np.uint8)
    scores = canny_scores(pred, gt)
    assert scores["canny_precision"] == pytest.approx(0.5)
    assert scores["canny_recall"] == pytest.approx(0.5)
    assert scores["canny_f1_score"] == pytest.approx(0.5)


def test_identical_blur_video_has_unit_ssim() -> None:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, size=(2, 16, 16, 3), dtype=np.uint8)
    assert blur_ssim(frames, frames) == pytest.approx(1.0)


def test_depth_si_rmse_is_scale_invariant() -> None:
    gt = np.arange(1, 33, dtype=np.float64).reshape(2, 4, 4)
    assert depth_si_rmse(gt * 0.25, gt) == pytest.approx(0.0)


def test_segmentation_unions_phrases_and_uses_hungarian_matching() -> None:
    left = np.zeros((2, 4, 4), dtype=bool)
    right = np.zeros_like(left)
    left[:, :2, :2] = True
    right[:, 2:, 2:] = True
    gt = [{"phrase": "cube", "mask": left}, {"phrase": "cube", "mask": right}]
    pred = [{"phrase": "object", "mask": np.logical_or(left, right)}]
    scores = segmentation_scores(gt, pred)
    assert scores["seg_m_iou"] == pytest.approx(1.0)
    assert scores["seg_recall"] == pytest.approx(1.0)


def test_generation_vqa_answer_parser_semantics() -> None:
    assert parse_option_answer("The correct answer is (C).") == "C"
    assert parse_option_answer("option b") == "B"
    assert normalize_binary_answer("The answer is YES.") == "yes"
    assert normalize_binary_answer("B", {"A": "yes", "B": "no"}) == "no"


def test_generation_vqa_uses_seed_question_video_hierarchy() -> None:
    rows = [
        {"video_id": "a", "question_id": "q1", "correct": True, "category": "physics"},
        {"video_id": "a", "question_id": "q1", "correct": False, "category": "physics"},
        {"video_id": "a", "question_id": "q2", "correct": True, "category": "physics"},
        {"video_id": "b", "question_id": "q1", "correct": False, "category": "robot"},
    ]
    summary = aggregate_generation_vqa(rows)
    assert summary["video_scores"]["a"] == pytest.approx(0.75)
    assert summary["video_scores"]["b"] == pytest.approx(0.0)
    assert summary["vqa_accuracy"] == pytest.approx(0.375)


def _write_video(path: Path, frames: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (frames.shape[2], frames.shape[1]))
    assert writer.isOpened()
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def test_official_conditional_sidecar_layout_and_seed_discovery(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generated = tmp_path / "generated"
    (dataset / "depth_vids").mkdir(parents=True)
    (dataset / "depth_vids" / "task_0001.mp4").write_bytes(b"visualization-only")
    (dataset / "depth_npzs").mkdir()
    expected_depth = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    np.savez(dataset / "depth_npzs" / "task_0001.npz", data=expected_depth)
    (dataset / "sam2_vids").mkdir()
    (dataset / "sam2_vids" / "task_0001.mp4").write_bytes(b"visualization-only")
    (dataset / "sam2_pkls").mkdir()
    (dataset / "sam2_pkls" / "task_0001.pkl").write_bytes(b"trusted-official-sidecar")
    (dataset / "captions").mkdir()
    (dataset / "captions" / "task_0001.json").write_text(
        json.dumps({"task_0001": "a robot moves a red cube"}),
        encoding="utf-8",
    )
    (generated / "videos").mkdir(parents=True)
    for name in ("task_0001__0.mp4", "task_0001__1.mp4", "task_0001_caption1.mp4"):
        (generated / "videos" / name).write_bytes(b"video")
    request = ConditionalGenerationRequest(
        dataset_root=dataset,
        generated_video_dir=generated,
        output_dir=tmp_path / "output",
    )
    row = {
        "depth": "depth_vids/task_0001.mp4",
        "sam2": "sam2_vids/task_0001.mp4",
        "caption_text": "the metadata column contains prose, not a path",
    }
    assert np.array_equal(_depth_reference(request, row, "task_0001.mp4"), expected_depth)
    assert _segmentation_reference(request, row, "task_0001.mp4") == (dataset / "sam2_pkls" / "task_0001.pkl")
    assert _caption_phrases(request, row, "task_0001.mp4") == ["a robot moves a red cube"]
    assert [path.name for path in _prediction_videos(generated, "task_0001.mp4")] == [
        "task_0001__0.mp4",
        "task_0001__1.mp4",
    ]


def test_conditional_cli_scores_synthetic_artifacts(tmp_path: Path) -> None:
    import cv2

    dataset = tmp_path / "dataset"
    generated = tmp_path / "generated"
    depth_dir = tmp_path / "pred_depth"
    output = tmp_path / "output"
    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, size=(3, 24, 24, 3), dtype=np.uint8)
    _write_video(generated / "task_0001__7.mp4", frames)
    _write_video(dataset / "videos" / "task_0001.mp4", frames)
    blurred = np.stack([cv2.bilateralFilter(frame, 30, 150, 100) for frame in frames])
    edges = np.stack([cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 100, 200) for frame in frames])
    _write_video(dataset / "blur" / "task_0001.mp4", blurred)
    _write_video(dataset / "canny" / "task_0001.mp4", np.repeat(edges[..., None], 3, axis=-1))
    depth = np.arange(1, 3 * 24 * 24 + 1, dtype=np.float32).reshape(3, 24, 24)
    (dataset / "depth_npzs").mkdir(parents=True)
    _write_video(dataset / "depth_vids" / "task_0001.mp4", frames)
    depth_dir.mkdir(parents=True)
    np.savez(dataset / "depth_npzs" / "task_0001.npz", depth=depth)
    np.save(depth_dir / "task_0001__7.npy", depth * 2)
    dataset.mkdir(exist_ok=True)
    with (dataset / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "video", "blur", "canny", "depth"])
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "task_0001.mp4",
                "video": "videos/task_0001.mp4",
                "blur": "blur/task_0001.mp4",
                "canny": "canny/task_0001.mp4",
                "depth": "depth_vids/task_0001.mp4",
            }
        )
    exit_code = main(
        [
            "--track",
            "conditional-generation",
            "--run-official",
            "--dataset-root",
            str(dataset),
            "--generated-video-dir",
            str(generated),
            "--pred-depth-dir",
            str(depth_dir),
            "--metrics",
            "blur_ssim,canny_f1_score,canny_precision,canny_recall,depth_si_rmse",
            "--output-dir",
            str(output),
            "--json",
        ]
    )
    assert exit_code == 0
    scorecard = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["validation"]["official_runtime_executed"] is True
    assert scorecard["metrics"]["summary"]["depth_si_rmse"] == pytest.approx(0.0)
    assert scorecard["integration_evidence"] is True


def test_generation_vqa_only_cli_runs_without_vbench_models(tmp_path: Path) -> None:
    dataset = tmp_path / "generation_dataset"
    generated = tmp_path / "generated"
    output = tmp_path / "generation_output"
    (dataset / "vqa").mkdir(parents=True)
    frames = np.full((2, 16, 16, 3), 127, dtype=np.uint8)
    _write_video(generated / "physics_case__0.mp4", frames)
    (dataset / "cosmos_predict2_bench_full_info.json").write_text(
        json.dumps([{"video_id": "physics_case", "prompt": "A ball falls."}]),
        encoding="utf-8",
    )
    (dataset / "vqa" / "physics_case.json").write_text(
        json.dumps(
            [
                {
                    "uid": "physics_case_q1",
                    "question": "Does the ball fall?",
                    "index2ans": {"A": "yes", "B": "no"},
                    "answer": "A",
                    "task": "physics",
                }
            ]
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "generation_predictions.jsonl"
    predictions.write_text(
        json.dumps({"uid": "physics_case_q1", "prediction": "yes"}) + "\n",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--track",
            "generation",
            "--run-official",
            "--dataset-root",
            str(dataset),
            "--generated-video-dir",
            str(generated),
            "--prediction-manifest",
            str(predictions),
            "--metrics",
            "vqa_accuracy",
            "--output-dir",
            str(output),
            "--json",
        ]
    )
    assert exit_code == 0
    scorecard = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["metrics"]["summary"]["vqa_accuracy"] == pytest.approx(1.0)


def test_workspace_routes_track_metrics_and_prediction_manifest(tmp_path: Path) -> None:
    command = build_workspace_benchmark_command(
        {
            "benchmark_id": "physical-ai-bench",
            "dataset_root": str(tmp_path / "dataset"),
            "answer_manifest": str(tmp_path / "predictions.jsonl"),
            "metrics": ["canny_f1_score", "depth_si_rmse"],
            "params": {
                "generated_video_dir": str(tmp_path / "videos"),
                "track": "conditional-generation",
            },
        },
        tmp_path / "output",
    )
    assert command[command.index("--track") + 1] == "conditional-generation"
    assert command[command.index("--metrics") + 1] == "canny_f1_score,depth_si_rmse"
    assert command[command.index("--prediction-manifest") + 1].endswith("predictions.jsonl")
    assert "--run-official" in command

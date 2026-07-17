"""Regression tests for the in-tree WorldBench runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from worldfoundry.evaluation.tasks.execution.runners.worldbench.runtime.worldbench.evaluator import (
    WorldBenchEvaluationRequest,
    evaluate_worldbench,
)
from worldfoundry.evaluation.tasks.execution.runners.worldbench.runtime.worldbench.metrics import (
    answer_is_correct,
    overlap_by_frame,
)
from worldfoundry.evaluation.tasks.execution.runners.worldbench.runtime.worldbench.reporting import (
    write_evaluation_scorecard,
)


def test_overlap_separates_iou_from_release_dice() -> None:
    target = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    predicted = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    row = overlap_by_frame([predicted], [target], object_ids=[1])[0]
    assert row["foreground_miou"] == pytest.approx(1.0 / 3.0)
    assert row["foreground_dice"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("prediction", "answer", "question"),
    [
        ("Yes, it will.", "Yes", "Will it move?"),
        ("option B", "Right", "Where? Options: Left, Right."),
        ("The answer is the sphere", "Sphere", "Which? Options: Cube, Sphere."),
    ],
)
def test_answer_matching(prediction: str, answer: str, question: str) -> None:
    assert answer_is_correct(prediction, answer, question)


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    dataset = root / "dataset"
    scene = dataset / "scenes/motion_phys/fixture/0"
    generated = root / "generated/motion_phys/fixture/0"
    masks = root / "masks/motion_phys/fixture/0"
    questions = dataset / "textual_questions"
    for path in (scene, generated, masks, questions):
        path.mkdir(parents=True, exist_ok=True)

    for index in range(3):
        rgba = np.zeros((8, 12, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        segmentation = np.ones((8, 12), dtype=np.uint8)
        if index >= 1:
            rgba[2:6, 3:7, :3] = (120, 40, 20)
            segmentation[2:6, 3:7] = 2
        Image.fromarray(rgba, mode="RGBA").save(scene / f"rgba_{index:05d}.png")
        Image.fromarray(segmentation, mode="L").save(scene / f"segmentation_{index:05d}.png")
    for index in range(2):
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[2:6, 3:7] = (120, 40, 20)
        labels = np.zeros((8, 12), dtype=np.uint8)
        labels[2:6, 3:7] = 1
        Image.fromarray(rgb, mode="RGB").save(generated / f"{index:05d}.png")
        Image.fromarray(labels, mode="L").save(masks / f"{index:05d}.png")

    (questions / "fixture.json").write_text(
        json.dumps(
            [
                {"video_name": "motion_phys/fixture/0.mp4", "question": "Will it move?", "answer": "Yes"},
                {
                    "video_name": "motion_phys/fixture/0.mp4",
                    "question": "Where? Options: Left, Right.",
                    "answer": "Right",
                },
            ]
        ),
        encoding="utf-8",
    )
    answers = root / "answers.json"
    answers.write_text(json.dumps({"fixture:0000": "Yes", "fixture:0001": "option B"}), encoding="utf-8")
    config = root / "evaluator.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "video": {
                    "target_size": [8, 12],
                    "ground_truth_start_frame": 1,
                    "generated_skip_frames": 0,
                    "max_frames": 2,
                    "background_label": 1,
                },
                "sam2": {"model_id": "facebook/sam2.1-hiera-large"},
            }
        ),
        encoding="utf-8",
    )
    return dataset, root / "generated", root / "masks", answers, config


def test_precomputed_mask_fixture_runs_end_to_end(tmp_path: Path) -> None:
    dataset, generated, masks, answers, config = _fixture(tmp_path)
    evaluation = evaluate_worldbench(
        WorldBenchEvaluationRequest(
            dataset_root=dataset,
            generated_video_dir=generated,
            predicted_mask_dir=masks,
            answer_manifest=answers,
            config_path=config,
            work_dir=tmp_path / "work",
        )
    )
    assert evaluation["metrics"]["foreground_miou"]["raw_score"] == pytest.approx(1.0)
    assert evaluation["metrics"]["foreground_dice"]["raw_score"] == pytest.approx(1.0)
    assert evaluation["metrics"]["background_rmse"]["raw_score"] < 0.05
    assert evaluation["metrics"]["text_based_accuracy"]["raw_score"] == pytest.approx(1.0)

    scorecard = write_evaluation_scorecard(
        evaluation,
        benchmark_id="worldbench",
        output_dir=tmp_path / "output",
        command=["worldbench-fixture"],
    )
    assert scorecard["benchmark"]["official_runtime_available"] is True
    assert scorecard["validation"]["normalizer_only"] is False
    assert scorecard["validation"]["official_runtime_executed"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False

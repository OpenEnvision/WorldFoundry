from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.metrics import (
    BenchmarkMetricInput,
    boolean_accuracy,
    camera_binary_classification_metrics,
    camera_retrieval_metrics,
    camera_vqa_metrics,
    chronomagic_average_scores,
    multiple_choice_accuracy,
    pairwise_preference_accuracy,
    parse_worldmodelbench_score,
    score_vector_spearman,
    success_rate,
    vbench_final_score,
    videoverse_subquestion_metrics,
    worldmodelbench_score,
)
from worldfoundry.evaluation.tasks.metrics import evaluate_external_metric


def test_benchmark_metric_input_protocol_normalizes_records() -> None:
    inputs = BenchmarkMetricInput(
        benchmark_id="fixture",
        metric_id="score",
        sample_id="sample-1",
        payload={"scores": [{"score": 1}, {"score": 0}, "bad"]},
        reference={"source": "unit"},
    )

    assert inputs.benchmark_id == "fixture"
    assert inputs.metric_id == "score"
    assert inputs.sample_id == "sample-1"
    assert inputs.records == ({"score": 1}, {"score": 0})
    assert inputs.reference == {"source": "unit"}


def test_camera_binary_classification_metrics_match_camerabench_shape() -> None:
    metrics = camera_binary_classification_metrics(
        [
            {"score": 0.9, "ground_truth_label": "yes", "error": None},
            {"score": 0.8, "ground_truth_label": "no", "error": None},
            {"score": 0.7, "ground_truth_label": "yes", "error": None},
            {"score": 0.1, "ground_truth_label": "no", "error": None},
            {"score": 1.0, "ground_truth_label": "yes", "error": "failed"},
        ]
    )

    assert metrics["num_samples"] == 4
    assert metrics["num_positive"] == 2
    assert metrics["num_negative"] == 2
    assert metrics["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_camera_vqa_and_retrieval_metrics_match_t2v_metrics_logic() -> None:
    records = [
        {
            "error": None,
            "yes_scores": {
                "pos_text_pos_image": 0.9,
                "pos_text_neg_image": 0.2,
                "neg_text_pos_image": 0.1,
                "neg_text_neg_image": 0.8,
            },
            "no_scores": {
                "pos_text_pos_image": 0.1,
                "pos_text_neg_image": 0.7,
                "neg_text_pos_image": 0.6,
                "neg_text_neg_image": 0.3,
            },
        },
        {
            "error": None,
            "yes_scores": {
                "pos_text_pos_image": 0.9,
                "pos_text_neg_image": 0.7,
                "neg_text_pos_image": 0.8,
                "neg_text_neg_image": 0.4,
            },
            "no_scores": {
                "pos_text_pos_image": 0.1,
                "pos_text_neg_image": 0.2,
                "neg_text_pos_image": 0.1,
                "neg_text_neg_image": 0.6,
            },
        },
    ]

    assert camera_vqa_metrics(records) == {
        "binary_acc": 0.625,
        "question_acc": 0.5,
        "num_samples": 2,
    }
    assert camera_retrieval_metrics(records) == {
        "text": 0.5,
        "image": 0.5,
        "group": 0.5,
        "num_samples": 2,
    }


def test_videoverse_subquestion_metrics_match_official_script_levels() -> None:
    metrics = videoverse_subquestion_metrics(
        {
            "video-1": {
                "verification_checks": [
                    {
                        "check_type": "temporal",
                        "sub_question_results": [{"res": "yes"}, {"res": "yes"}],
                    },
                    {
                        "check_type": "event",
                        "sub_question_results": [{"res": "no"}, {"res": "maybe"}],
                    },
                ]
            },
            "video-2": {
                "verification_checks": [
                    {
                        "check_type": "temporal",
                        "sub_question_results": [{"res": "yes"}],
                    }
                ]
            },
        }
    )

    assert metrics["sub_question_accuracy"] == pytest.approx(3 / 4)
    assert metrics["check_accuracy"] == pytest.approx(2 / 3)
    assert metrics["video_accuracy"] == pytest.approx(1 / 2)
    assert metrics["wrong"] == 1
    assert metrics["per_check_type"]["temporal"]["check_accuracy"] == 1.0


def test_videoverse_subquestion_metrics_ignore_non_video_payload_keys() -> None:
    metrics = videoverse_subquestion_metrics({"generated_files": []})

    assert metrics["sub_question_accuracy"] == 0.0
    assert metrics["check_accuracy"] == 0.0
    assert metrics["video_accuracy"] == 0.0
    assert metrics["total_videos"] == 0


def test_multiple_choice_accuracy_supports_ipv_style_taxonomy() -> None:
    metrics = multiple_choice_accuracy(
        {
            "q1": {"video_name": "v1.mp4", "pred": "(A)."},
            "q2": {"video_name": "v2.mp4", "pred": "B"},
        },
        {
            "q1": {"video_name": "v1.mp4", "answer": "a"},
            "q2": {"video_name": "v2.mp4", "answer": "C"},
        },
        taxonomy_by_video={
            "v1.mp4": {"taxonomy_label_list": ["physical laws"], "spatial_temporal_label": "spatial"},
            "v2.mp4": {"taxonomy_label_list": ["physical laws"], "spatial_temporal_label": "temporal"},
        },
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["num_correct"] == 1
    assert metrics["per_taxonomy"]["physical laws"]["accuracy"] == 0.5
    assert metrics["per_spatial_temporal"]["spatial"]["accuracy"] == 1.0
    assert metrics["per_spatial_temporal"]["temporal"]["accuracy"] == 0.0


def test_chronomagic_average_scores_group_by_model_prefix() -> None:
    assert chronomagic_average_scores(
        {
            "model_a_001_CHScore.json": {"total_average_score": 0.6},
            "model_a_002_CHScore.json": {"total_average_score": 0.8},
            "model_b_001_CHScore.json": {"total_average_score": 0.2},
            "ignored_MTScore.json": {"average_metamorphic_score": 1.0},
        },
        score_key="total_average_score",
        suffix="CHScore",
    ) == {
        "model_a": {"Average_CHScore": 0.7},
        "model_b": {"Average_CHScore": 0.2},
    }


def test_worldmodelbench_score_matches_subcategory_sum_and_parse() -> None:
    metrics = worldmodelbench_score(
        {
            "instruction_following": [1, 2],
            "physical_adherence": [1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
        },
        num_instances=2,
    )

    assert metrics["categories"]["instruction_following"]["overall"] == pytest.approx(1.5)
    assert metrics["categories"]["physical_adherence"]["sub_scores"]["newton"] == pytest.approx(1.5)
    assert metrics["categories"]["physical_adherence"]["overall"] == pytest.approx(17.5)
    assert metrics["total_score"] == pytest.approx(19.0)
    assert parse_worldmodelbench_score("Reasoning... Score: 2.5.") == 2.5
    assert parse_worldmodelbench_score("not a score") == 0.0


def test_vbench_final_score_matches_official_weighted_aggregation() -> None:
    metrics = vbench_final_score(
        {
            "subject consistency": 1.0,
            "background consistency": 1.0,
            "temporal flickering": 1.0,
            "motion smoothness": 0.9975,
            "dynamic degree": 1.0,
            "aesthetic quality": 1.0,
            "imaging quality": 1.0,
            "object class": 1.0,
            "multiple objects": 1.0,
            "human action": 1.0,
            "color": 1.0,
            "spatial relationship": 1.0,
            "scene": 0.8222,
            "appearance style": 0.2855,
            "temporal style": 0.364,
            "overall consistency": 0.364,
        }
    )
    i2v_metrics = vbench_final_score(
        {
            "camera_motion": [1.0],
            "i2v_subject": [1.0],
            "i2v_background": [1.0],
            "subject_consistency": [1.0],
            "background_consistency": [1.0],
            "motion_smoothness": [0.9975],
            "dynamic_degree": [1.0],
            "aesthetic_quality": [1.0],
            "imaging_quality": [1.0],
        },
        i2v=True,
    )

    assert metrics["quality_score"] == pytest.approx(1.0)
    assert metrics["semantic_score"] == pytest.approx(1.0)
    assert metrics["final_score"] == pytest.approx(1.0)
    assert i2v_metrics["quality_score"] == pytest.approx(1.0)
    assert i2v_metrics["i2v_score"] == pytest.approx(1.0)
    assert i2v_metrics["final_score"] == pytest.approx(1.0)


def test_pairwise_preference_and_success_rate_formulas() -> None:
    preference = pairwise_preference_accuracy(
        [
            {"prediction": "A", "preferred": "left", "category": "alignment"},
            {"prediction": "B", "preferred": "A", "category": "alignment"},
            {"score_a": 0.2, "score_b": 0.7, "label": "right", "category": "quality"},
        ]
    )
    success = success_rate(
        [
            {"task": "lift", "success": True},
            {"task": "lift", "success": False},
            {"task": "place", "num_success": 3, "num_trials": 4},
        ]
    )

    assert preference["accuracy"] == pytest.approx(2 / 3)
    assert preference["per_category"]["alignment"]["accuracy"] == pytest.approx(0.5)
    assert success["success_rate"] == pytest.approx(4 / 6)
    assert success["per_task"]["place"]["num_success"] == 3


def test_boolean_accuracy_and_score_vector_spearman_formulas() -> None:
    accuracy = boolean_accuracy(
        [
            {"task": "image_generation", "correct": True},
            {"task": "image_generation", "correct": False},
            {"task": "video_generation", "correct": "yes"},
        ]
    )
    spearman = score_vector_spearman(
        [
            {"ref": "[1, 1, 1, 1, 1]", "ans": "[1, 1, 1, 1, 1]"},
            {"ref": "[2, 2, 2, 2, 2]", "ans": "[2, 2, 2, 2, 2]"},
            {"ref": "[3, 3, 3, 3, 3]", "ans": "[3, 3, 3, 3, 3]"},
        ]
    )

    assert accuracy["accuracy"] == pytest.approx(2 / 3)
    assert accuracy["per_task"]["image_generation"]["accuracy"] == pytest.approx(0.5)
    assert spearman["spearman_list"] == [100.0, 100.0, 100.0, 100.0, 100.0]
    assert spearman["spearman_average"] == pytest.approx(100.0)


def test_formula_metrics_are_registered_as_external_evaluators() -> None:
    camera_result = evaluate_external_metric(
        "camerabench",
        "camera_motion_average_precision",
        reference={
            "scores": [
                {"score": 0.9, "ground_truth_label": "yes", "error": None},
                {"score": 0.1, "ground_truth_label": "no", "error": None},
            ]
        },
        sample_id="camera-sample",
    )
    videoverse_result = evaluate_external_metric(
        "videoverse",
        "qa_accuracy",
        reference={
            "official_results": {
                "video-1": {
                    "verification_checks": [
                        {
                            "check_type": "temporal",
                            "sub_question_results": [{"res": "yes"}, {"res": "no"}],
                        }
                    ]
                }
            }
        },
        sample_id="videoverse-sample",
    )
    ipv_result = evaluate_external_metric(
        "ipv-bench",
        "mcqa_accuracy",
        reference={
            "predictions": {"q1": {"pred": "A"}},
            "answers": {"q1": {"answer": "A"}},
        },
        sample_id="ipv-sample",
    )
    videobench_result = evaluate_external_metric(
        "video-bench",
        "mcqa_accuracy",
        reference={
            "predictions": {"q1": {"pred": "B"}},
            "answers": {"q1": {"answer": "B"}},
        },
        sample_id="video-bench-sample",
    )
    chronomagic_result = evaluate_external_metric(
        "chronomagic-bench",
        "chronomagic_score",
        reference={
            "scores": {
                "model_a_001_CHScore.json": {"total_average_score": 0.6},
                "model_a_002_CHScore.json": {"total_average_score": 0.8},
                "model_b_001_CHScore.json": {"total_average_score": 0.2},
            }
        },
        sample_id="chronomagic-sample",
    )
    vbench_result = evaluate_external_metric(
        "vbench",
        "overall_quality",
        reference={
            "scores": {
                "subject consistency": 1.0,
                "background consistency": 1.0,
                "temporal flickering": 1.0,
                "motion smoothness": 0.9975,
                "dynamic degree": 1.0,
                "aesthetic quality": 1.0,
                "imaging quality": 1.0,
                "object class": 1.0,
                "multiple objects": 1.0,
                "human action": 1.0,
                "color": 1.0,
                "spatial relationship": 1.0,
                "scene": 0.8222,
                "appearance style": 0.2855,
                "temporal style": 0.364,
                "overall consistency": 0.364,
            }
        },
        sample_id="vbench-sample",
    )
    robotwin_result = evaluate_external_metric(
        "robotwin",
        "success_rate",
        reference={
            "records": [
                {"task": "pick", "success": True},
                {"task": "pick", "success": False},
            ]
        },
        sample_id="robotwin-sample",
    )
    genaibench_result = evaluate_external_metric(
        "genai-bench",
        "image_generation_preference_accuracy",
        reference={
            "records": [
                {"task": "image_generation", "correct": True},
                {"task": "image_generation", "correct": False},
                {"task": "video_generation", "correct": False},
            ]
        },
        sample_id="genai-bench-sample",
    )
    genaibench_average_result = evaluate_external_metric(
        "genai-bench",
        "genai_bench_average",
        reference={
            "records": [
                {"task": "image_generation", "correct": True},
                {"task": "image_generation", "correct": False},
                {"task": "image_editing", "correct": True},
                {"task": "video_generation", "correct": False},
            ]
        },
        sample_id="genai-bench-average-sample",
    )
    videoscore_result = evaluate_external_metric(
        "videoscore",
        "videoscore_average",
        reference={
            "records": [
                {"ref": "[1, 1, 1, 1, 1]", "ans": "[1, 1, 1, 1, 1]"},
                {"ref": "[2, 2, 2, 2, 2]", "ans": "[2, 2, 2, 2, 2]"},
                {"ref": "[3, 3, 3, 3, 3]", "ans": "[3, 3, 3, 3, 3]"},
            ]
        },
        sample_id="videoscore-sample",
    )

    assert camera_result.valid is True
    assert camera_result.normalized_value == 1.0
    assert videoverse_result.valid is True
    assert videoverse_result.normalized_value == 0.5
    assert ipv_result.valid is True
    assert ipv_result.normalized_value == 1.0
    assert videobench_result.valid is True
    assert videobench_result.normalized_value == 1.0
    assert chronomagic_result.valid is True
    assert chronomagic_result.normalized_value == pytest.approx(0.45)
    assert chronomagic_result.components["per_model"]["model_a"]["Average_CHScore"] == pytest.approx(0.7)
    assert vbench_result.valid is True
    assert vbench_result.normalized_value == pytest.approx(1.0)
    assert robotwin_result.valid is True
    assert robotwin_result.normalized_value == pytest.approx(0.5)
    assert genaibench_result.valid is True
    assert genaibench_result.normalized_value == pytest.approx(0.5)
    assert genaibench_average_result.valid is True
    assert genaibench_average_result.normalized_value == pytest.approx(0.5)
    assert videoscore_result.valid is True
    assert videoscore_result.raw_value == pytest.approx(100.0)
    assert videoscore_result.normalized_value == pytest.approx(1.0)

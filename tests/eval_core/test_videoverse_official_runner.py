from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "worldfoundry/evaluation/tasks/execution/runners/videoverse/run_videoverse_official_runner.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("videoverse_official_runner_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_prompt_manifest() -> dict[str, dict]:
    return {
        "sample-a": {
            "t2v_following_prompt": {"t2v_prompt": "A red ball rolls across the floor."},
            "verification_checks": [
                {"check_type": "Interaction", "question": "Does the ball roll?", "check": True},
            ],
            "t2v_eval_event_info": {
                "verification_plan": [
                    {"event_id": 1, "event_description": "Ball starts rolling."},
                    {"event_id": 2, "event_description": "Ball stops."},
                ]
            },
        },
        "sample-b": {
            "t2v_following_prompt": {"t2v_prompt": "A cat jumps onto a table."},
            "verification_checks": [
                {"check_type": "Common Sense", "question": "Does the cat jump?", "check": True},
            ],
            "t2v_eval_event_info": {
                "verification_plan": [
                    {"event_id": 1, "event_description": "Cat jumps."},
                ]
            },
        },
    }


def test_videoverse_official_run_with_mock_judge_writes_eval_res_and_scorecard(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner_module()
    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(json.dumps(_sample_prompt_manifest(), sort_keys=True), encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "sample-a.mp4").write_bytes(b"fake-a")
    (generated_dir / "sample-b.mp4").write_bytes(b"fake-b")
    output_dir = tmp_path / "official-run"
    monkeypatch.setenv("WORLDFOUNDRY_VIDEOVERSE_JUDGE_BACKEND", "mock")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    args = argparse.Namespace(
        benchmark_id="videoverse",
        official_results_path=None,
        run_official=True,
        output_dir=output_dir,
        generated_artifact_dir=generated_dir,
        prompt_manifest=prompt_manifest,
        decomposed_prompt_manifest=prompt_manifest,
        limit=None,
        strict=False,
        json=False,
    )
    scorecard = runner.run_official_videoverse(args)

    eval_res = json.loads((output_dir / "eval_res.json").read_text(encoding="utf-8"))
    assert set(eval_res) == {"sample-a", "sample-b"}
    assert eval_res["sample-a"]["verification_checks"][0]["res"] == "yes"
    assert scorecard["evaluation"]["kind"] == "videoverse_official_in_tree"
    assert scorecard["normalizer_only"] is False
    assert scorecard["official_benchmark_verified"] is True
    assert scorecard["metrics"]["summary"]["available_metric_count"] == 7


def test_videoverse_prompt_materialization_uses_prompt_ids(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.videoverse.videoverse_prompts import (
        materialize_videoverse_generation_requests,
    )

    prompt_manifest = tmp_path / "prompts.json"
    prompt_manifest.write_text(json.dumps(_sample_prompt_manifest(), sort_keys=True), encoding="utf-8")
    requests = materialize_videoverse_generation_requests(prompt_manifest_path=prompt_manifest)
    assert [request.sample_id for request in requests] == ["sample-a", "sample-b"]
    assert requests[0].inputs["prompt"] == "A red ball rolls across the floor."

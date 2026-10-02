from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_demo_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "demo" / "run_minimal_eval_loop.py"
    spec = importlib.util.spec_from_file_location("test_minimal_eval_loop_demo_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_minimal_eval_loop_demo_writes_non_leaderboard_scorecard(tmp_path: Path) -> None:
    demo = _load_demo_script()
    output_dir = tmp_path / "minimal_eval_loop"

    summary = demo.run_demo(output_dir)

    assert summary["ok"] is True
    assert summary["demo"] is True
    assert summary["scorecard_flags"]["demo"] is True
    assert summary["scorecard_flags"]["score_valid"] is True
    assert summary["scorecard_flags"]["leaderboard_valid"] is False
    assert summary["scorecard_flags"]["leaderboard_eligible"] is False
    assert "missing official/full-suite leaderboard evidence gate" in summary["scorecard_flags"]["blocking_reasons"]

    for relative_path in [
        "preflight.json",
        "inputs/requests.jsonl",
        "results.jsonl",
        "metrics/summary.json",
        "scorecard.json",
        "report.md",
        "demo_summary.json",
        "generated_samples/demo-0001.txt",
        "generated_samples/demo-0002.txt",
    ]:
        assert (output_dir / relative_path).is_file()

    preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["downloads_model"] is False
    assert preflight["uses_user_token"] is False

    metrics = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert metrics["leaderboard"]["artifact_count"] == 1.0
    assert metrics["leaderboard"]["has_artifact:generated_trace"] == 1.0
    assert metrics["leaderboard"]["required_artifacts_present"] == 1.0
    assert metrics["leaderboard"]["state_match"] == 1.0

    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["benchmark"]["benchmark_name"] == "worldfoundry-minimal-loop-demo"
    assert scorecard["benchmark"]["demo"] is True
    assert scorecard["model"]["model_id"] == "tiny-local-text-runner"
    assert scorecard["dataset"]["split"] == "demo"

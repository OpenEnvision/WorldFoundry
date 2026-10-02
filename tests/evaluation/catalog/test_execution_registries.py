"""Execution registry tables stay internally consistent.

CI already runs catalog checks via ``make workspace-registry-check``. This
module covers the in-tree / contract / video-quality tables plus shared
session helpers.
"""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    first_existing_dir,
    first_existing_file,
)
from worldfoundry.evaluation.tasks.execution.framework.scoring_registry import (
    contract_evaluator_kind,
    get_in_tree_benchmark_config,
    has_benchmark_contract_evaluator,
    supported_in_tree_benchmark_ids,
    target_benchmark_metrics,
    write_benchmark_contract_evaluation,
)
from worldfoundry.evaluation.tasks.execution.framework.io import coerce_unit_score
from worldfoundry.evaluation.tasks.execution.framework.video_quality_specs import (
    supported_video_quality_benchmark_ids,
    video_quality_spec,
)
from worldfoundry.evaluation.tasks.execution.orchestration.run_session import (
    artifact_report_paths,
    session_paths,
)


def test_in_tree_registry_is_one_table() -> None:
    ids = supported_in_tree_benchmark_ids()
    assert len(ids) == len(set(ids))
    metrics = target_benchmark_metrics()
    assert set(ids) == set(metrics)
    for benchmark_id in ids:
        config = get_in_tree_benchmark_config(benchmark_id)
        assert config["benchmark_id"] == benchmark_id
        assert config["metric_ids"] == metrics[benchmark_id]


def test_video_quality_specs_are_central() -> None:
    ids = supported_video_quality_benchmark_ids()
    assert "aigcbench" in ids
    assert "genai-bench" in ids
    for benchmark_id in ids:
        config = video_quality_spec(benchmark_id)
        assert config["benchmark_id"] == benchmark_id
        assert config["blocked_reason"]
    genai = video_quality_spec("genai-bench")
    assert "video_generation" in genai["genai_task_metrics"]


def test_contract_registry_kinds_come_from_modules() -> None:
    assert has_benchmark_contract_evaluator("likephys")
    assert not has_benchmark_contract_evaluator("vbench")
    assert contract_evaluator_kind("likephys") == "in_tree_likephys_contract_evaluator"
    try:
        write_benchmark_contract_evaluation(benchmark_id="not-a-benchmark")
    except KeyError as exc:
        assert "not-a-benchmark" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_session_paths_and_unit_score() -> None:
    output = Path("/tmp/worldfoundry-run-session-test")
    paths = session_paths(output)
    assert paths["manifest"].name == "run_manifest.json"
    assert paths["per_sample"].parent.name == "metrics"
    report = artifact_report_paths(output, artifact_count=0)
    assert "artifacts" not in report
    assert coerce_unit_score(80) == 0.8
    assert coerce_unit_score(0.4) == 0.4
    assert coerce_unit_score(True) is None
    assert coerce_unit_score(True, allow_bool=True) == 1.0


def test_first_existing_helpers(tmp_path: Path | None = None) -> None:
    root = Path(tmp_path) if tmp_path is not None else Path("/tmp")
    missing = root / "wf-missing-dir"
    present = root if root.is_dir() else Path(".")
    assert first_existing_dir(missing, present) == present.resolve()
    file_path = present / "wf-registry-probe.txt"
    file_path.write_text("ok", encoding="utf-8")
    try:
        assert first_existing_file(missing, file_path) == file_path.resolve()
    finally:
        file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_in_tree_registry_is_one_table()
    test_video_quality_specs_are_central()
    test_contract_registry_kinds_come_from_modules()
    test_session_paths_and_unit_score()
    test_first_existing_helpers()
    print("ok")

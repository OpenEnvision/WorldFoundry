from __future__ import annotations

import json
from pathlib import Path

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID
from worldfoundry.evaluation.runner import ModelBenchmarkSuiteRequest, run_model_benchmark_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"
MODEL_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def test_model_benchmark_suite_cells_preserve_run_artifacts(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "suite",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
            benchmark_ids=("vbench",),
            mode="contract",
            contract_fixture=True,
        )
    )
    payload = json.loads(result.suite_manifest_path.read_text(encoding="utf-8"))
    cell = payload["cells"][0]

    assert result.exit_code == 0
    assert Path(cell["run_manifest_path"]).is_file()
    assert Path(cell["run_summary_path"]).is_file()
    assert Path(cell["generated_artifact_dir"]).is_dir()
    assert Path(cell["artifact_manifest_path"]).is_file()
    assert cell["artifacts"]["run_summary"] == cell["run_summary_path"]
    assert cell["artifacts"]["generated_artifact_dir"] == cell["generated_artifact_dir"]

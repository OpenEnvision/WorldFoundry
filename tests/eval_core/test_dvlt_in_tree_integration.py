from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.pipelines.component_pipelines import DVLTPipeline
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.runtime.conda import load_runtime_conda_env_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
DVLT_RUNTIME_ROOT = REPO_ROOT / "worldfoundry" / "base_models" / "three_dimensions" / "depth" / "dvlt" / "dvlt_runtime"
PIPELINE_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"


def test_dvlt_catalog_profile_and_runtime_source_are_in_tree() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    entry = registry.get("dvlt")
    profile = load_runtime_profile("dvlt")
    env_specs = load_runtime_conda_env_specs(env_root="/tmp/worldfoundry-conda-test")

    assert entry.runner_target == PIPELINE_RUNNER_TARGET
    assert entry.pipeline_target == "worldfoundry.pipelines.component_pipelines:DVLTPipeline"
    assert entry.runtime_profile == "runtime-profile:dvlt"
    assert entry.integration_status == "integrated"
    assert entry.runner_entry_kind == "runnable_runner"
    assert profile.artifact_kind == "generated_3d_asset"
    assert profile.backend_stage == "in_tree_runtime"
    assert profile.runtime_status == "in_tree_dvlt_runtime_ported_gpu_parity_pending"
    assert env_specs["dvlt"].env_name == "worldfoundry-dvlt-cu124"
    assert (DVLT_RUNTIME_ROOT / "src" / "dvlt" / "model" / "dvlt" / "model.py").is_file()
    assert (DVLT_RUNTIME_ROOT / "LICENSES" / "NVIDIA-LICENSE.txt").is_file()
    assert (DVLT_RUNTIME_ROOT / "THIRD_PARTY_LICENSES.md").is_file()
    assert (DVLT_RUNTIME_ROOT / ".worldfoundry_upstream_commit").read_text(encoding="utf-8").strip()


def test_dvlt_pipeline_rejects_preflight_artifact_smoke(tmp_path: Path) -> None:
    pipe = DVLTPipeline.from_pretrained(device="cpu")
    assert isinstance(pipe, DVLTPipeline)

    report_path = tmp_path / "dvlt_preflight.json"
    with pytest.raises(RuntimeError, match="DVLT cannot run"):
        pipe(
            images="sample_image.png",
            interactions=["view_sequence"],
            output_path=report_path,
            return_dict=True,
        )

    assert not report_path.exists()

    with pytest.raises(ValueError, match="no longer supports plan_only"):
        DVLTPipeline.from_pretrained({"plan_only": True}, device="cpu")

from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.models import (
    ModelResolutionError,
    WorldFoundryPipelineRunner,
    resolve_model_zoo_runner,
    resolve_world_model_runner,
)
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_RUNNER_TARGET, ContractFixtureRunner
from worldfoundry.evaluation.tasks.embodied import (
    EmbodiedGenerationSpec,
    run_vla_va_wam,
)
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.pipelines.component_pipelines import OpenVLAPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
def _vla_spec() -> EmbodiedGenerationSpec:
    return EmbodiedGenerationSpec.from_dict(
        {
            "track": "vla",
            "kind": "action",
            "task_name": "libero_contract_fixture",
            "observation_keys": ["image", "instruction", "proprio"],
            "output_keys": ["actions", "action_trace"],
            "action_space": {
                "kind": "continuous",
                "dimensions": 7,
                "bounds": {"low": [-1, -1, -1, -1, -1, -1, -1], "high": [1, 1, 1, 1, 1, 1, 1]},
            },
        }
    )


def test_vla_va_wam_run_uses_contract_runner_and_writes_scorecard(tmp_path: Path) -> None:
    result = run_vla_va_wam(
        output_dir=tmp_path / "run",
        spec=_vla_spec(),
        samples=[
            {
                "sample_id": "episode-001",
                "instruction": "put the mug on the shelf",
                "image": "memory://rgb.png",
                "proprio": [0.0] * 7,
            }
        ],
        model_id="contract-vla",
        model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
        benchmark_id="libero",
        metric_ids=("generation_success", "task_success", "action_accuracy"),
        run_id="vla-va-wam-contract-test",
    )

    assert result.status == "succeeded"
    assert result.sample_count == 1
    scorecard = json.loads((tmp_path / "run" / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["benchmark"]["track"] == "vla"
    assert scorecard["metrics"]["leaderboard"]["generation_success"] == 1.0
    assert scorecard["metrics"]["leaderboard"]["task_success"] == 1.0
    results = [
        json.loads(line)
        for line in (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert results[0]["artifacts"]["actions"]["uri"] == "memory://episode-001/actions.json"


def test_model_resolver_uses_shared_runtime_runner_alias() -> None:
    resolved = resolve_world_model_runner(
        "contract-vla",
        runner=CONTRACT_FIXTURE_RUNNER_TARGET,
        parameters={"track": "vla"},
    )

    assert isinstance(resolved.runner, ContractFixtureRunner)
    assert resolved.runner_target == CONTRACT_FIXTURE_RUNNER_TARGET


def test_model_resolver_rejects_source_specific_runtime_alias() -> None:
    try:
        resolve_world_model_runner(
            "contract-vla",
            runner="vla-va-wam:contract",
            parameters={"track": "vla"},
    )
    except ModelResolutionError as exc:
        assert "vla-va-wam" in str(exc)
    else:  # pragma: no cover - explicit assertion keeps no-alias policy visible.
        raise AssertionError("source-specific runner aliases should not resolve")


def test_runtime_profile_runner_passes_embodied_sample_fields_to_model_operator(tmp_path: Path) -> None:
    pipeline = OpenVLAPipeline.from_pretrained({"model_id": "openvla"}, device="cpu")
    runner = WorldFoundryPipelineRunner(
        "openvla",
        pipeline,
        pipeline_target="worldfoundry.pipelines.component_pipelines:OpenVLAPipeline",
        output_dir=tmp_path,
    )

    results = runner.generate(
        [
            GenerationRequest(
                sample_id="episode-001",
                generation_kwargs={"plan_only": True},
                task_name="libero",
                inputs={
                    "instruction": "put the mug on the shelf",
                    "image": "memory://rgb.png",
                    "proprio": [0.0] * 7,
                    "action_space": {"kind": "continuous", "dimensions": 7},
                },
                controls={
                    "sample_controls": {
                        "camera_names": ["front"],
                        "actions": [{"delta": [0.0] * 7}],
                    }
                },
            )
        ]
    )

    assert results[0].status == "prepared"
    plan = json.loads(Path(str(results[0].metadata["plan_path"])).read_text(encoding="utf-8"))
    extra_inputs = json.loads(Path(plan["context"]["extra_inputs_path"]).read_text(encoding="utf-8"))
    assert extra_inputs["operator_metadata"]["policy_family"] == "autoregressive_vision_language_action_policy"
    assert extra_inputs["operator_metadata"]["action_representation"] == "continuous_7d_end_effector_delta"
    assert extra_inputs["openvla_observation"]["proprio"] == [0.0] * 7
    assert extra_inputs["openvla_observation"]["camera_names"] == ["front"]


def test_vla_va_wam_rejects_external_subprocess_runner(tmp_path: Path) -> None:
    try:
        run_vla_va_wam(
            output_dir=tmp_path / "run",
            spec=_vla_spec(),
            samples=[{"sample_id": "episode-001", "instruction": "move", "image": "memory://rgb.png"}],
            model_id="external-runtime-bridge",
            model_runner="worldfoundry:jsonl-subprocess",
            metric_ids=("task_success", "action_accuracy"),
        )
    except ModelResolutionError as exc:
        assert "jsonl-subprocess" in str(exc)
    else:
        raise AssertionError("external subprocess runner should not be registered")


def test_embodied_benchmark_zoo_manifest_is_discoverable_as_tasks() -> None:
    from worldfoundry.evaluation.tasks.catalog.specs import get_benchmark_zoo_cli_task, list_benchmark_zoo_cli_tasks

    expected = {
        "libero",
        "libero-para",
        "simpler-env",
        "robocasa",
        "calvin",
        "maniskill",
        "rlbench",
        "metaworld",
        "bridgedata-v2",
        "robotwin",
    }
    discovered = {item["task_type"] for item in list_benchmark_zoo_cli_tasks(source_kind="benchmark_zoo")}
    assert expected <= discovered

    libero = get_benchmark_zoo_cli_task("libero", "libero")
    assert libero["source_kind"] == "benchmark_zoo"
    assert libero["benchmark_zoo_id"] == "libero"
    assert libero["requires_upstream_runtime"] is True
    assert libero["official_runtime_validated"] is False
    assert "vla" in libero["capability_track"]
    rlbench = get_benchmark_zoo_cli_task("rlbench", "rlbench")
    assert rlbench["output_keys"] == ["scorecard", "raw_results", "per_sample_metrics", "rollout_logs", "videos"]
    assert "simulator" in rlbench["capability_track"]
    bridgedata = get_benchmark_zoo_cli_task("bridgedata-v2", "bridgedata-v2")
    assert "policy_results_path" in bridgedata["input_keys"]
    assert "real-robot" in bridgedata["capability_track"]


def test_vla_va_wam_model_zoo_manifest_records_current_runtime_readiness() -> None:
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")
    # lingbot-va's catalog integration status
    # (in_tree_checkpoint_runtime_gpu_server_init_verified_rollout_pending) is
    # registered as an in-tree integrated surface, so it now classifies as a
    # runnable runner alongside the other checkpoint-backed VLA models.
    runner_candidate_expected: set[str] = set()
    runnable_runner_expected = {
        "act",
        "being-h05",
        "diffusion-policy",
        "dreamzero",
        "giga-brain-0",
        "gr00t",
        "lapa",
        "lingbot-va",
        "octo",
        "openpi",
        "openvla",
        "roboflamingo",
        "rt-1",
        "starvla",
    }
    manifests = {item.model_id: item for item in registry.to_world_model_manifests()}

    openvla = registry.get("openvla")
    assert openvla.integration_status == "integrated"
    assert openvla.runner_entry_kind == "runnable_runner"
    assert openvla.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
    assert openvla.pipeline_target == "worldfoundry.pipelines.component_pipelines:OpenVLAPipeline"
    assert manifests["openvla"].metadata["runnable_runner"] is True
    assert manifests["openvla"].metadata["default_runner_target"] == openvla.runner_target
    assert "vla.policy_rollout" in openvla.tasks
    assert "vla.policy_rollout" in manifests["openvla"].capabilities
    assert {"actions", "action_trace"} <= set(manifests["openvla"].output_artifacts)
    resolved_openvla = resolve_model_zoo_runner(
        "openvla",
        manifest_dir=REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog",
        runtime={"device": "cpu"},
    )
    assert isinstance(resolved_openvla.runner, WorldFoundryPipelineRunner)

    for model_id in runner_candidate_expected:
        entry = registry.get(model_id)
        assert entry.runner_target is not None
        assert entry.pipeline_target is not None
        assert entry.runner_entry_kind == "runner_candidate"
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        manifest = manifests[model_id]
        assert manifest.metadata["runner_target"] == entry.runner_target
        assert manifest.metadata["pipeline_target"] == entry.pipeline_target
        assert manifest.metadata["default_runner_target"] is None
        assert manifest.metadata["runnable_runner"] is False
        assert {"action_trace", "actions", "action_tokens"} & set(manifest.output_artifacts)

    for model_id in runnable_runner_expected:
        entry = registry.get(model_id)
        assert entry.integration_status == "integrated"
        assert entry.runner_target is not None
        assert entry.pipeline_target is not None
        assert entry.runner_entry_kind == "runnable_runner"
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        manifest = manifests[model_id]
        assert manifest.metadata["runner_target"] == entry.runner_target
        assert manifest.metadata["pipeline_target"] == entry.pipeline_target
        assert manifest.metadata["default_runner_target"] == entry.runner_target
        assert manifest.metadata["runnable_runner"] is True
        if model_id == "lapa":
            assert "action_tokens" in manifest.output_artifacts
        else:
            assert "action_trace" in manifest.output_artifacts

    resolved_octo = resolve_model_zoo_runner(
        "octo",
        manifest_dir=REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog",
        runtime={"device": "cpu"},
    )
    assert isinstance(resolved_octo.runner, WorldFoundryPipelineRunner)


def test_external_benchmark_contracts_include_embodied_suites() -> None:
    contract = get_external_benchmark_contract("libero")

    assert contract.display_name == "LIBERO"
    assert "policy_results_path" in contract.input_keys
    assert "success_rate" in contract.metric_ids
    for benchmark_id in ("libero-para", "rlbench", "metaworld", "bridgedata-v2"):
        embodied_contract = get_external_benchmark_contract(benchmark_id)
        assert "policy_results_path" in embodied_contract.input_keys
        assert "success_rate" in embodied_contract.metric_ids


def test_vla_va_wam_cli_lists_framework_runners(capsys) -> None:
    from worldfoundry.cli import main

    assert main(["models", "runtime-runners", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {item["name"] for item in payload}
    assert CONTRACT_FIXTURE_RUNNER_TARGET not in names
    assert "worldfoundry:jsonl-subprocess" not in names

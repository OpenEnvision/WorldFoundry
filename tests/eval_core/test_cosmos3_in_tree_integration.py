"""Cosmos3 contract tests against the current native pipeline / runtime surface.

The pre-refactor suite targeted
``base_models.diffusion_model.video.cosmos3.artifacts`` /
``.worldfoundry_runtime`` and ``synthesis...cosmos.cosmos3_synthesis``. Those
modules are gone. This file locks the successor:

- ``worldfoundry.pipelines.cosmos.pipeline_cosmos3.Cosmos3Pipeline``
- ``worldfoundry.synthesis.visual_generation.cosmos.cosmos3_runtime.Cosmos3Runtime``
- catalog / inference-spec metadata that still describes the same variants
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.runtime.inference_catalog import (
    COSMOS3_NANO_REPO_ID,
    COSMOS3_NANO_REVISION,
    COSMOS3_SUPER_REPO_ID,
    COSMOS3_SUPER_REVISION,
    get_model_inference_spec,
)
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.pipelines.loading import build_pipeline_runner_spec
from worldfoundry.evaluation.models.runners.resolver import resolve_model_zoo_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_ROOT = REPO_ROOT / "worldfoundry/data/models/catalog"


def _runtime_surface():
    pytest.importorskip("torch")
    from worldfoundry.pipelines.cosmos.pipeline_cosmos3 import _model_id
    from worldfoundry.synthesis.visual_generation.cosmos.cosmos3_runtime import (
        Cosmos3Runtime,
        Cosmos3RuntimeOutput,
    )

    return SimpleNamespace(
        Cosmos3Runtime=Cosmos3Runtime,
        Cosmos3RuntimeOutput=Cosmos3RuntimeOutput,
        model_id=_model_id,
    )


def test_removed_cosmos3_artifact_helpers_stay_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("worldfoundry.base_models.diffusion_model.video.cosmos3.artifacts")
    with pytest.raises(ModuleNotFoundError):
        __import__("worldfoundry.base_models.diffusion_model.video.cosmos3.worldfoundry_runtime")
    with pytest.raises(ModuleNotFoundError):
        __import__("worldfoundry.synthesis.visual_generation.cosmos.cosmos3_synthesis")


def test_cosmos3_runtime_plan_routes_nano_and_super() -> None:
    surface = _runtime_surface()
    nano = surface.Cosmos3Runtime.plan()
    assert nano["variant_id"] == "cosmos3-nano"
    assert nano["backend"] == "worldfoundry-native-diffusion"
    assert nano["native_inference"] is True
    assert nano["blocked"] is False
    assert nano["blockers"] == ()

    super_plan = surface.Cosmos3Runtime.plan({"variant_id": "cosmos3-super"})
    assert super_plan["variant_id"] == "cosmos3-super"

    path_plan = surface.Cosmos3Runtime.plan("/ckpt/Cosmos3-Super")
    assert path_plan["variant_id"] == "cosmos3-super"
    assert path_plan["model_path"] == "/ckpt/Cosmos3-Super"


def test_cosmos3_runtime_plan_reports_missing_checkpoint_files(tmp_path: Path) -> None:
    surface = _runtime_surface()
    root = tmp_path / "cosmos3-incomplete"
    root.mkdir()
    plan = surface.Cosmos3Runtime.plan(str(root))

    assert plan["blocked"] is True
    assert plan["variant_id"] == "cosmos3-nano"
    assert "transformer/diffusion_pytorch_model.safetensors.index.json" in plan["blockers"]
    assert "vae/diffusion_pytorch_model.safetensors" in plan["blockers"]
    assert "sound_tokenizer/diffusion_pytorch_model.safetensors" in plan["blockers"]


def test_cosmos3_pipeline_selector_rejects_unknown_variant() -> None:
    surface = _runtime_surface()
    with pytest.raises(ValueError, match="unsupported Cosmos3 model selector"):
        surface.model_id("not-a-variant", None)


def test_cosmos3_runtime_predict_wraps_pipeline_dict() -> None:
    surface = _runtime_surface()

    class _FakePipeline:
        model_id = "cosmos3-nano"

        def __call__(self, *args, **kwargs):
            assert kwargs["return_dict"] is True
            return {
                "video": "vid",
                "sound": "snd",
                "action": {"delta": [0.0]},
                "audio_sampling_rate": 16000,
                "artifact_path": "/tmp/out.mp4",
            }

    output = surface.Cosmos3Runtime(_FakePipeline()).predict("a robot cleans a plate")
    assert isinstance(output, surface.Cosmos3RuntimeOutput)
    assert output.video == "vid"
    assert output.sound == "snd"
    assert output.action == {"delta": [0.0]}
    assert output.audio_sample_rate == 16000
    assert output.artifact_path == "/tmp/out.mp4"


def test_cosmos3_runtime_rejects_api_backend() -> None:
    surface = _runtime_surface()

    class _FakePipeline:
        model_id = "cosmos3-nano"

    with pytest.raises(NotImplementedError, match="does not use an API backend"):
        surface.Cosmos3Runtime(_FakePipeline()).api_init()


def test_cosmos3_catalog_metadata_matches_in_tree_runtime() -> None:
    registry = load_model_zoo_registry(CATALOG_ROOT)
    entry = registry.get("cosmos3")

    assert entry.integration_status == "integrated"
    assert entry.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
    assert entry.pipeline_target == "worldfoundry.pipelines.cosmos.pipeline_cosmos3:Cosmos3Pipeline"
    assert entry.runtime_profile == "runtime-profile:cosmos3"
    variants = {variant.variant_id: variant for variant in entry.variants}
    assert variants["cosmos3-nano"].checkpoint_refs[0].revision == COSMOS3_NANO_REVISION
    assert variants["cosmos3-super"].checkpoint_refs[0].revision == COSMOS3_SUPER_REVISION
    assert variants["cosmos3-nano"].checkpoint_refs[0].requires_auth is False
    assert variants["cosmos3-super"].checkpoint_refs[0].requires_auth is False


def test_cosmos3_runner_spec_preserves_super_variant() -> None:
    for requested_id, variant_id in (("cosmos3", "cosmos3-super"), ("cosmos3-super", None)):
        resolved = resolve_model_zoo_config(
            requested_id,
            variant_id=variant_id,
            manifest_dir=CATALOG_ROOT,
            runtime={"device": "cpu"},
        )
        spec = build_pipeline_runner_spec(resolved.config)
        assert spec.model_id == "cosmos3"
        assert spec.runtime_profile_id == "cosmos3-super"
        assert spec.model_path["variant_id"] == "cosmos3-super"
        assert spec.model_path["profile_id"] == "cosmos3-super"
        assert spec.model_path["runtime_profile"] == "cosmos3-super"


def test_cosmos3_inference_spec_exposes_generator_and_action_tasks() -> None:
    spec = get_model_inference_spec("cosmos3")

    assert spec is not None
    assert spec.default_variant_id == "cosmos3-nano"
    assert spec.default_task_id == "t2v"
    assert {variant.variant_id for variant in spec.variants} == {"cosmos3-nano", "cosmos3-super"}
    assert {task.task_id for task in spec.tasks} == {
        "t2i",
        "t2v",
        "i2v",
        "v2v",
        "action-policy",
        "action-forward-dynamics",
        "action-inverse-dynamics",
    }
    assert spec.variant("cosmos3-nano").primary_checkpoint_uri == COSMOS3_NANO_REPO_ID
    assert spec.variant("cosmos3-super").primary_checkpoint_uri == COSMOS3_SUPER_REPO_ID
    assert spec.variant("cosmos3-nano").load_kwargs["revision"] == COSMOS3_NANO_REVISION
    assert spec.variant("cosmos3-super").load_kwargs["revision"] == COSMOS3_SUPER_REVISION
    assert spec.task("t2i").default_call_kwargs["num_frames"] == 1
    assert spec.task("t2v").default_call_kwargs["enable_sound"] is False
    assert spec.task("action-policy").default_call_kwargs["action_mode"] == "policy"
    assert any("Reasoner, training" in note for note in spec.notes)


def test_cosmos3_offload_uses_wrapper_computation_device_before_parameter_storage() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.model import (
        _module_param_device,
    )

    class _OffloadedChild(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, device="cpu"))
            self.computation_device = torch.device("cuda:0")

    parent = torch.nn.Sequential(_OffloadedChild())

    assert _module_param_device(parent) == torch.device("cuda:0")

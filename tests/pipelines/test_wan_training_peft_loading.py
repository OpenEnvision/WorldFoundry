from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers import wan as wan_denoiser
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.pipelines.loading import (
    build_pipeline_runner_spec,
    call_pipeline_from_pretrained,
)
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline
from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
from worldfoundry.training.tuning import WAN_ATTENTION, apply_peft_lora


def _checkpoint(*, revision: str = "a" * 40) -> CheckpointSpec:
    return CheckpointSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        revision=revision,
        files=("diffusion_pytorch_model.safetensors",),
    )


def _context(
    *,
    checkpoint: CheckpointSpec | None = None,
    component_options: dict[str, object] | None = None,
) -> ComponentBuildContext:
    return ComponentBuildContext(
        model_id="wan2.1-t2v-1.3b",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(device=torch.device("cpu"), dtype=torch.float32),
        checkpoints={"weights": checkpoint or _checkpoint()},
        component_options=component_options or {},
    )


def test_benchmark_runner_config_executes_wan_peft_adapter_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_from_pretrained(
        cls: type[NativeDiffusionPipeline],
        model_id: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        del cls
        calls.append((model_id, dict(kwargs)))
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    adapter_path = tmp_path / "policy"
    config = WorldModelConfig(
        model_id="wan2.1-t2v-1.3b",
        runner="worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner",
        parameters={
            "pipeline_target": ("worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v:Wan2p1T2VPipeline"),
            "peft_adapter_path": str(adapter_path),
        },
        runtime={"device": "cpu"},
    )
    spec = build_pipeline_runner_spec(config)

    pipeline = call_pipeline_from_pretrained(
        Wan2p1T2VPipeline,
        model_path=spec.model_path,
        fallback_model_path=spec.fallback_model_path,
        required_components=spec.required_components,
        device=spec.device,
        model_id=spec.model_id,
    )

    assert isinstance(pipeline, Wan2p1T2VPipeline)
    assert len(calls) == 1
    loaded_model_id, loading_options = calls[0]
    assert loaded_model_id == "wan2.1-t2v-1.3b"
    assert loading_options["checkpoint_overrides"] is None
    assert loading_options["component_options"] == {"denoiser:main": {"peft_adapter_path": str(adapter_path)}}
    assert loading_options["policy"].device == torch.device("cpu")


def test_wan_pipeline_default_loading_does_not_inject_component_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(
        cls: type[NativeDiffusionPipeline],
        model_id: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        del cls
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    Wan2p1T2VPipeline.from_pretrained(device="cpu", model_id="wan2.1-t2v-1.3b")

    assert captured["component_options"] is None


def test_wan22_ti2v_exposes_distilled_lora_loading_boundary() -> None:
    assert Wan2p2Pipeline.PEFT_ADAPTER_COMPONENT == "denoiser:main"
    assert Wan2p2Pipeline._component_options(
        {"peft_adapter_path": "/models/wan22-four-step-lora"}
    ) == {
        "denoiser:main": {
            "peft_adapter_path": "/models/wan22-four-step-lora",
        }
    }


def test_wan_denoiser_rejects_unknown_component_options_before_loading() -> None:
    with pytest.raises(ValueError, match="unsupported Wan denoiser options"):
        wan_denoiser.build_wan21_t2v_1p3b_denoiser(_context(component_options={"peft_adapter_pat": "/tmp/adapter"}))


def _tiny_wan(seed: int) -> WanModel:
    torch.manual_seed(seed)
    return WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=1,
        has_image_input=False,
    )


def test_standard_training_peft_adapter_loads_and_forwards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_wan(23)
    application = apply_peft_lora(source, preset=WAN_ATTENTION, rank=2, alpha=2)
    with torch.no_grad():
        for name, parameter in application.model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.05)
    application.model.eval()
    torch.manual_seed(29)
    inputs = (
        torch.randn(1, 4, 1, 2, 2),
        torch.tensor([500.0]),
        torch.randn(1, 3, 16),
    )
    with torch.no_grad():
        expected = application.model(*inputs)
    adapter_path = tmp_path / "policy"
    application.model.save_pretrained(adapter_path, safe_serialization=True)
    assert not (adapter_path / "worldfoundry_adapter.json").exists()

    def load_with_post_hook(self, spec, checkpoint, policy):
        del self, checkpoint, policy
        loaded = _tiny_wan(23).eval()
        assert spec.post_load_hook is not None
        spec.post_load_hook(loaded)
        return loaded

    monkeypatch.setattr(wan_denoiser.NativeModuleLoader, "load", load_with_post_hook)
    context = _context(component_options={"peft_adapter_path": str(adapter_path)})
    denoiser = wan_denoiser.build_wan21_t2v_1p3b_denoiser(context)
    with torch.no_grad():
        actual = denoiser.model(*inputs)
    torch.testing.assert_close(actual, expected)

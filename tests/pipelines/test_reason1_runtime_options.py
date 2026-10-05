from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

_PIPELINES = (
    ("worldfoundry.pipelines.cosmos.pipeline_cosmos_predict2p5", "CosmosPredict2p5Pipeline"),
    ("worldfoundry.pipelines.cosmos.pipeline_cosmos_transfer2p5", "CosmosTransfer2p5Pipeline"),
    ("worldfoundry.pipelines.gamma_world.pipeline_gamma_world", "GammaWorldPipeline"),
)


@pytest.mark.parametrize("module_name,class_name", _PIPELINES)
def test_all_reason1_pipelines_forward_placement_cache_and_extensions(monkeypatch, module_name, class_name):
    module = importlib.import_module(module_name)
    calls = []

    def load(model_id, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(module.NativeDiffusionPipeline, "from_pretrained", load)
    extension = object()
    getattr(module, class_name).from_pretrained(
        device="cpu",
        text_encoder_run_on_cpu=True,
        text_embedding_cache_size=3,
        text_embedding_cache_max_bytes=1024,
        extensions=(extension,),
    )
    assert calls[0]["component_options"]["conditioner:main"] == {
        "run_on_cpu": True,
        "embedding_cache_size": 3,
        "embedding_cache_max_bytes": 1024,
    }
    assert calls[0]["extensions"] == (extension,)


@pytest.mark.parametrize("module_name,class_name", _PIPELINES)
@pytest.mark.parametrize(
    "options,error",
    [
        ({"text_encoder_run_on_cpu": "false"}, TypeError),
        ({"text_embedding_cache_size": True}, TypeError),
        ({"text_embedding_cache_size": -1}, ValueError),
        ({"text_embedding_cache_max_bytes": 1.5}, TypeError),
        ({"text_embedding_cache_max_bytes": -1}, ValueError),
    ],
)
def test_invalid_encoder_options_fail_before_loading_native_weights(
    monkeypatch, module_name, class_name, options, error
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(
        module.NativeDiffusionPipeline, "from_pretrained", lambda *a, **kw: pytest.fail("must not load weights")
    )
    with pytest.raises(error):
        getattr(module, class_name).from_pretrained(**options)


def test_gamma_explicit_component_options_keep_their_values_and_public_options_override(monkeypatch):
    from worldfoundry.pipelines.gamma_world import pipeline_gamma_world as module

    captured = []
    monkeypatch.setattr(
        module.NativeDiffusionPipeline,
        "from_pretrained",
        lambda model_id, **kw: captured.append(kw) or SimpleNamespace(model_id=model_id),
    )
    configured = {"conditioner:main": {"embedding_cache_size": 7, "sequence_length": 128}}
    module.GammaWorldPipeline.from_pretrained(component_options=configured)
    assert captured[0]["component_options"]["conditioner:main"]["embedding_cache_size"] == 7
    assert captured[0]["component_options"]["conditioner:main"]["sequence_length"] == 128
    module.GammaWorldPipeline.from_pretrained(component_options=configured, text_embedding_cache_size=2)
    assert captured[1]["component_options"]["conditioner:main"]["embedding_cache_size"] == 2
    assert configured["conditioner:main"]["embedding_cache_size"] == 7


def test_gamma_advanced_encoder_options_are_validated_before_weights(monkeypatch):
    from worldfoundry.pipelines.gamma_world import pipeline_gamma_world as module

    monkeypatch.setattr(module.NativeDiffusionPipeline, "from_pretrained", lambda *a, **kw: pytest.fail("weights"))
    with pytest.raises(ValueError, match="non-negative"):
        module.GammaWorldPipeline.from_pretrained(component_options={"conditioner:main": {"embedding_cache_size": -1}})

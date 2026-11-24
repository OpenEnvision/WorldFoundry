import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from worldfoundry.synthesis.action_generation.db_cogact.runtime import (
    _finalize_vision_tower,
    _policy_observation,
    _prepare_policy_observation,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("stats_source", ("norm_stats", "norm_stats_path", "normalization_stats_path", "checkpoint"))
def test_cached_weights_use_request_scoped_normalization(monkeypatch, tmp_path, stats_source) -> None:
    from worldfoundry.synthesis.action_generation.db_cogact import runtime

    construction = []
    tokenizer = object()
    architecture = ModuleType("worldfoundry.synthesis.action_generation.db_cogact.modeling.architecture")
    policy_module = ModuleType("worldfoundry.synthesis.action_generation.db_cogact.modeling.policy")
    transformers = ModuleType("transformers")
    tokenizer_calls = []
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path, **kwargs: tokenizer_calls.append((path, kwargs)) or tokenizer),
        raising=False,
    )
    modeling_utils = ModuleType("transformers.modeling_utils")
    monkeypatch.setattr(modeling_utils, "no_init_weights", nullcontext, raising=False)

    class Model:
        def __init__(self, config):
            construction.append(self)
            self.config = config
            self.model = SimpleNamespace(mm_vision_tower=SimpleNamespace())

        def to(self, **_kwargs):
            return self

        def eval(self):
            return self

    class Policy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.norm_stats = kwargs["norm_stats"]

        def select_action(self):
            return np.array([self.norm_stats["min"], self.norm_stats["max"]])

    monkeypatch.setattr(
        architecture, "CogActConfig", SimpleNamespace(from_pretrained=lambda *_, **__: SimpleNamespace()), raising=False
    )
    monkeypatch.setattr(architecture, "CogACTForCausalLM", Model, raising=False)
    monkeypatch.setattr(policy_module, "CogACTPolicy", Policy, raising=False)
    for module in (architecture, policy_module, transformers, modeling_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runtime, "_RUNTIME_CACHE", {})
    assignments = []
    monkeypatch.setattr(
        "worldfoundry.core.model_loading.checkpoints.load_safetensors_into_model_streaming",
        lambda *args, **_: assignments.append(args) or {},
    )
    monkeypatch.setattr("worldfoundry.core.vram.skip_model_initialization", nullcontext)
    monkeypatch.setattr(
        "worldfoundry.core.attention.resolve_transformers_attention_implementation", lambda **_: "eager"
    )
    location = str(tmp_path)
    stats_file = tmp_path / "norm_stats.json"

    def options_for(stats):
        if stats_source == "norm_stats":
            return {"norm_stats": stats}
        stats_file.write_text(json.dumps({"norm_stats": {"default": stats}}))
        return {} if stats_source == "checkpoint" else {stats_source: str(stats_file)}

    first = runtime._runtime_for(location, "cpu", options_for({"min": 1, "max": 3}))
    second = runtime._runtime_for(location, "cpu", {**options_for({"min": 10, "max": 30}), "camera_order": ["wrist"]})
    again = runtime._runtime_for(location, "cpu", options_for({"min": 1, "max": 3}))
    assert first is not second
    assert first.model is second.model is again.model
    assert first.tokenizer is second.tokenizer is tokenizer
    assert first.camera_order == again.camera_order == ["front", "left_wrist", "right_wrist"]
    assert second.camera_order == ["wrist"]
    assert len(construction) == len(assignments) == 1
    assert tokenizer_calls == [(location, {"trust_remote_code": False, "local_files_only": True})]
    np.testing.assert_array_equal(first.select_action(), [1, 3])
    np.testing.assert_array_equal(second.select_action(), [10, 30])
    np.testing.assert_array_equal(again.select_action(), [1, 3])


def _prepare_for_policy(policy, observation):
    payload = _policy_observation(observation, None, "test instruction", {})
    return _prepare_policy_observation(policy, payload)


def test_unused_state_is_not_sent_to_official_cogact_policy() -> None:
    payload = _prepare_for_policy(
        SimpleNamespace(state_used=False),
        {"image/1": "front.png", "state": [0.0] * 8},
    )

    assert "state" not in payload


def test_single_state_vector_is_not_mistaken_for_a_batch() -> None:
    payload = _prepare_for_policy(
        SimpleNamespace(state_used=True),
        {"image/1": "front.png", "state": [0.0] * 8},
    )

    assert isinstance(payload["state"], np.ndarray)
    assert payload["state"].shape == (8,)


def test_transformers_meta_vision_tower_is_finalized() -> None:
    class VisionTower:
        _meta_initialized = True
        is_loaded = False

        def load_model(self):
            self.is_loaded = True

    tower = VisionTower()
    model = SimpleNamespace(model=SimpleNamespace(mm_vision_tower=tower))

    _finalize_vision_tower(model)

    assert tower.is_loaded


@pytest.mark.parametrize("value", [True, "true", "1"])
@pytest.mark.parametrize("cached", [False, True])
def test_remote_tokenizer_code_is_rejected_before_cached_resource_lookup(
    monkeypatch: pytest.MonkeyPatch, value: bool | str, cached: bool
) -> None:
    from worldfoundry.synthesis.action_generation.db_cogact import runtime

    key = ("/checkpoint", "cpu", None, None, False)
    monkeypatch.setattr(runtime, "_RUNTIME_CACHE", {key: (object(), object())} if cached else {})
    with pytest.raises(ValueError, match="DB-CogACT requires trust_remote_code=False"):
        runtime._model_resources_for("/checkpoint", "cpu", {"trust_remote_code": value})


@pytest.mark.parametrize("value", [False, "false", None])
def test_native_tokenizer_setting_preserves_cached_resources(
    monkeypatch: pytest.MonkeyPatch, value: bool | str | None
) -> None:
    from worldfoundry.synthesis.action_generation.db_cogact import runtime

    resources = (object(), object())
    monkeypatch.setattr(runtime, "_RUNTIME_CACHE", {("/checkpoint", "cpu", None, None, False): resources})
    assert runtime._model_resources_for("/checkpoint", "cpu", {"trust_remote_code": value}) is resources

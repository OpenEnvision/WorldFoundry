from types import SimpleNamespace

import pytest

from worldfoundry.synthesis.action_generation.official_policy.runtime import (
    OfficialPolicyRuntime,
    build_runtime_config,
)


@pytest.mark.parametrize(
    "backend", ("custom_from_pretrained", "processor_select_action", "hf_auto_action_model", "lerobot_policy")
)
@pytest.mark.parametrize("unpack", (False, True))
@pytest.mark.parametrize("fails", (False, True))
def test_policy_dispatch_executes_once(backend, unpack, fails) -> None:
    runtime = OfficialPolicyRuntime(
        build_runtime_config(
            model_id="test-policy",
            profile_checkpoints=(),
            defaults={"backend": backend},
            options={},
            device="cpu",
        )
    )
    calls = []
    error = TypeError("policy tensor operation failed")

    def record(payload):
        calls.append(payload)
        if fails:
            raise error
        return [0.25]

    def predict_mapping(payload):
        return record(payload)

    def predict_keywords(**payload):
        return record(payload)

    model = SimpleNamespace(predict_action=predict_keywords if unpack else predict_mapping, device="cpu")
    if backend == "hf_auto_action_model":
        policy = (lambda **kwargs: {"input_ids": [1]}, model)
        expected = {"input_ids": [1]}
    elif backend in {"custom_from_pretrained", "processor_select_action"}:
        policy = (None, model)
        expected = {"state": [1], "task": ["move"], "prompt": "move"}
    else:
        policy = model
        expected = {"observation": {"state": [1]}, "task": "move", "action_context": []}

    def predict():
        return runtime._predict_with_policy(
            policy,
            instruction="move",
            image=None,
            observation={"state": [1]},
            action_context=[],
        )

    if fails:
        with pytest.raises(TypeError) as raised:
            predict()
        assert raised.value is error
    else:
        assert predict() == [0.25]
    assert calls == [expected]

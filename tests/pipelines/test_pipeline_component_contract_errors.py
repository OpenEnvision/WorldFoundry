"""Regression tests for PL-04: explicit component-contract errors.

``PipelineABC._call_component_pipeline`` dispatches
``processed["prompt"|"images"|"video"|"actions"]`` to the synthesis model.  A
custom operator or ``process()`` override that dropped one of those keys used
to surface as a bare ``KeyError`` deep inside the base class; it now raises a
``TypeError`` naming the pipeline, the missing keys, and the operator to
inspect.  Well-formed outputs keep working unchanged.
"""

from __future__ import annotations

import pytest

from worldfoundry.pipelines.pipeline_utils import PipelineABC


class _RecordingSynthesis:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return {"artifact_path": "artifact.mp4"}


class _RecordingMemory:
    def __init__(self, model_id=None):
        self.records = []

    def record(self, result, metadata=None):
        self.records.append((result, metadata))


class _StubOperator:
    """Minimal operator; only its class name matters for the error message."""


class _FixedProcessPipeline(PipelineABC):
    OPERATOR_CLS = _StubOperator
    MEMORY_CLS = _RecordingMemory
    SYNTHESIS_CLS = _RecordingSynthesis

    def __init__(self, processed):
        super().__init__(
            model_id="contract-test",
            operators=_StubOperator(),
            synthesis_model=_RecordingSynthesis(),
            memory_module=_RecordingMemory(),
        )
        self._processed = processed

    def process(self, *args, **kwargs):
        return self._processed


def _complete_processed(**overrides):
    processed = {
        "prompt": "a prompt",
        "images": None,
        "video": None,
        "actions": [],
        "extra_inputs": {},
    }
    processed.update(overrides)
    return processed


def test_complete_process_output_reaches_synthesis():
    pipeline = _FixedProcessPipeline(_complete_processed())
    result = pipeline(prompt="a prompt")
    assert result == "artifact.mp4"
    assert pipeline.synthesis_model.calls[0]["prompt"] == "a prompt"
    assert pipeline.memory_module.records


def test_missing_actions_key_raises_actionable_type_error():
    processed = _complete_processed()
    del processed["actions"]
    pipeline = _FixedProcessPipeline(processed)
    with pytest.raises(TypeError) as excinfo:
        pipeline(prompt="a prompt")
    message = str(excinfo.value)
    assert "_FixedProcessPipeline" in message
    assert "'actions'" in message
    assert "_StubOperator" in message


def test_multiple_missing_keys_are_all_reported():
    pipeline = _FixedProcessPipeline({"prompt": "p", "extra_inputs": {}})
    with pytest.raises(TypeError) as excinfo:
        pipeline(prompt="p")
    message = str(excinfo.value)
    for key in ("'images'", "'video'", "'actions'"):
        assert key in message


def test_non_mapping_process_output_raises_type_error():
    pipeline = _FixedProcessPipeline(["not", "a", "mapping"])
    with pytest.raises(TypeError) as excinfo:
        pipeline(prompt="p")
    message = str(excinfo.value)
    assert "must return a mapping" in message
    assert "list" in message

from __future__ import annotations

import json
import sys
import types

from worldfoundry.core.observability.torchprofile import TORCHPROFILE_SCHEMA_VERSION, profile_torch_module


class _Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _Module:
    def __init__(self) -> None:
        self.training = True
        self.child = types.SimpleNamespace(training=False)

    def modules(self):
        return (self, self.child)

    def eval(self) -> None:
        self.training = False
        self.child.training = False

    def parameters(self):
        return (_Parameter(3), _Parameter(5))


def _event(path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def test_torchprofile_summary_emits_canonical_event(tmp_path, monkeypatch):
    module = _Module()
    fake_torchprofile = types.SimpleNamespace(__version__="0.test", profile_macs=lambda model, inputs: 42)
    monkeypatch.setitem(sys.modules, "torchprofile", fake_torchprofile)
    event_path = tmp_path / "torchprofile.events.jsonl"

    result = profile_torch_module(
        module,
        inputs=("representative-input",),
        model_id="demo-model",
        input_spec={"batch_size": 1, "token": "must-redact"},
        event_log_path=event_path,
    )

    assert result.status == "completed"
    assert result.macs == 42
    assert result.flops_estimate == 84
    assert result.parameter_count == 8
    assert module.training is True
    assert module.child.training is False
    payload = _event(event_path)
    assert payload["schema_version"] == "worldfoundry.log.v1"
    assert payload["event"] == "torchprofile.completed"
    assert payload["model_id"] == "demo-model"
    assert payload["fields"]["profile_schema_version"] == TORCHPROFILE_SCHEMA_VERSION
    assert payload["fields"]["macs"] == 42
    assert payload["fields"]["input_spec"]["token"] == "[REDACTED]"


def test_torchprofile_failure_is_nonfatal_and_redacted(tmp_path, monkeypatch):
    module = _Module()

    def fail(*_args):
        raise RuntimeError("api_key=profile-secret")

    monkeypatch.setitem(sys.modules, "torchprofile", types.SimpleNamespace(__version__="0.test", profile_macs=fail))
    event_path = tmp_path / "torchprofile.events.jsonl"

    result = profile_torch_module(module, event_log_path=event_path)

    assert result.status == "failed"
    assert "profile-secret" in str(result.error)
    payload = _event(event_path)
    assert payload["event"] == "torchprofile.failed"
    assert "profile-secret" not in event_path.read_text(encoding="utf-8")
    assert "[REDACTED]" in payload["exception"]

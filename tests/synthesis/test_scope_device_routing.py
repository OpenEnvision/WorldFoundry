from __future__ import annotations

import pytest

from worldfoundry.synthesis.visual_generation.scope.worldfoundry_runtime import SCOPERuntime


def test_scope_subprocess_resolves_logical_device_against_visible_mask(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")

    first = SCOPERuntime(model_dir="/tmp/scope", device="cuda:0")._subprocess_env()
    second = SCOPERuntime(model_dir="/tmp/scope", device="cuda:1")._subprocess_env()

    assert first["CUDA_VISIBLE_DEVICES"] == "3"
    assert second["CUDA_VISIBLE_DEVICES"] == "1"
    assert first["SCOPE_DEVICE"] == second["SCOPE_DEVICE"] == "cuda"


def test_scope_subprocess_rejects_device_outside_visible_mask(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        SCOPERuntime(model_dir="/tmp/scope", device="cuda:1")._subprocess_env()


def test_scope_subprocess_selects_global_device_without_visible_mask(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env = SCOPERuntime(model_dir="/tmp/scope", device="cuda:2")._subprocess_env()

    assert env["CUDA_VISIBLE_DEVICES"] == "2"

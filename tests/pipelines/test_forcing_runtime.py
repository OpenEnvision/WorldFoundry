"""Weight-free contracts for the Forcing subprocess bridge."""

from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.forcing import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.forcing.runtime import RollingForcingRuntime


class _ValidatedRequest(Exception):
    """Stop predict after validation, before preparing or launching inference."""


@pytest.fixture
def validation_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    def forbid_execution(*args: object, **kwargs: object) -> None:
        raise pytest.fail.Exception("Validation must not load checkpoints, initialize CUDA, or launch inference")

    def stop_after_validation(self: object, root: Path) -> Path:
        raise _ValidatedRequest

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_execution)
    monkeypatch.setattr(torch, "load", forbid_execution)
    monkeypatch.setattr(runtime_module.subprocess, "run", forbid_execution)
    monkeypatch.setattr(runtime_module.RollingForcingRuntime, "_runtime_cwd", stop_after_validation)


@pytest.mark.parametrize(
    ("device", "inherited", "expected"),
    [
        ("cuda:1", "2,5", "5"),
        ("cuda", "2,5", "2,5"),
        ("cuda", "", ""),
        ("cuda:1", None, "1"),
        ("cuda", None, "0"),
    ],
)
def test_subprocess_env_maps_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
    validation_boundary: None,
    device: str,
    inherited: str | None,
    expected: str,
) -> None:
    if inherited is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    runtime = RollingForcingRuntime(device=device)

    assert runtime._subprocess_env()["CUDA_VISIBLE_DEVICES"] == expected
    if inherited is None:
        assert "CUDA_VISIBLE_DEVICES" not in runtime_module.os.environ
    else:
        assert runtime_module.os.environ["CUDA_VISIBLE_DEVICES"] == inherited


@pytest.mark.unit
@pytest.mark.parametrize("inherited", [None, "operator-owned-path"])
def test_subprocess_env_preserves_pythonpath(
    monkeypatch: pytest.MonkeyPatch, validation_boundary: None, inherited: str | None
) -> None:
    if inherited is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", inherited)
    runtime = RollingForcingRuntime()

    env = runtime._subprocess_env()

    assert env.get("PYTHONPATH") == inherited
    assert runtime.runtime_root == str(runtime_module.RUNTIME_DIR)


@pytest.mark.unit
@pytest.mark.parametrize("option", ["images", "i2v"])
def test_rolling_forcing_rejects_images_before_launch(validation_boundary: None, tmp_path: Path, option: str) -> None:
    runtime = RollingForcingRuntime()

    options: dict[str, Any] = {option: True}
    with pytest.raises(ValueError, match="text-to-video only"):
        runtime.predict(output_path=tmp_path / "video.mp4", **options)

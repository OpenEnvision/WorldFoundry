from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_source(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wan_video_logging_does_not_configure_root_logger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for dependency in ("imageio", "torch", "torchvision"):
        monkeypatch.setitem(sys.modules, dependency, types.ModuleType(dependency))
    module = _load_source(
        "worldfoundry/base_models/diffusion_model/models/networks/wan/media_geometry.py",
        "test_wan_video_geometry_logging",
    )

    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, "handlers", [])
    basic_config_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda *args, **kwargs: basic_config_calls.append((args, kwargs)),
    )

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(module.shutil, "move", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stderr=""),
    )
    module.merge_video_audio(str(video_path), str(audio_path))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=1, stderr="mux failed"),
    )
    with pytest.raises(RuntimeError, match="mux failed"):
        module.merge_video_audio(str(video_path), str(audio_path))

    class BrokenTensor:
        def clamp(self, *_args, **_kwargs):
            raise RuntimeError("fixture failure")

    module.save_video(BrokenTensor(), save_file=str(tmp_path / "output.mp4"))
    module.save_image(BrokenTensor(), str(tmp_path / "output.png"))

    assert root_logger.handlers == []
    assert basic_config_calls == []


def test_get_video_info_uses_configurable_bounded_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_source(
        "worldfoundry/base_models/llm_mllm_core/mllm/instance_anomaly_detector/split.py",
        "test_instance_anomaly_split_success",
    )
    calls: list[tuple[list[str], int]] = []

    def fake_run_bounded_command(command, *, timeout):
        calls.append((command, timeout))
        return {
            "stdout": json.dumps(
                {"streams": [{"duration": "12.5", "avg_frame_rate": "30000/1001"}]}
            ),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(module, "run_bounded_command", fake_run_bounded_command)

    duration, fps = module.get_video_info("sample.mp4", timeout_seconds=45)

    assert duration == 12.5
    assert fps == pytest.approx(30000 / 1001)
    assert calls == [
        (
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,duration",
                "-of",
                "json",
                "sample.mp4",
            ],
            45,
        )
    ]


def test_get_video_info_reports_nonzero_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_source(
        "worldfoundry/base_models/llm_mllm_core/mllm/instance_anomaly_detector/split.py",
        "test_instance_anomaly_split_nonzero",
    )
    monkeypatch.setattr(
        module,
        "run_bounded_command",
        lambda *_args, **_kwargs: {
            "stdout": "",
            "stderr": "bad container",
            "returncode": 2,
            "timed_out": False,
        },
    )

    with pytest.raises(RuntimeError, match=r"code 2: bad container"):
        module.get_video_info("broken.mp4")


def test_get_video_info_reports_probe_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_source(
        "worldfoundry/base_models/llm_mllm_core/mllm/instance_anomaly_detector/split.py",
        "test_instance_anomaly_split_timeout",
    )
    monkeypatch.setattr(
        module,
        "run_bounded_command",
        lambda *_args, **_kwargs: {
            "stdout": "",
            "stderr": "TimeoutExpired",
            "returncode": 124,
            "timed_out": True,
        },
    )

    with pytest.raises(TimeoutError, match=r"timed out after 30s for stuck.mp4"):
        module.get_video_info("stuck.mp4")


def test_get_video_info_rejects_nonpositive_timeout() -> None:
    module = _load_source(
        "worldfoundry/base_models/llm_mllm_core/mllm/instance_anomaly_detector/split.py",
        "test_instance_anomaly_split_timeout_validation",
    )

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        module.get_video_info("sample.mp4", timeout_seconds=0)

"""Current benchmark-zoo orchestration and removed script-zoo contracts."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import socket
import sys
import types
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.framework.io import (
    load_json,
    mean_numeric,
    normalize_unit_score,
    read_jsonl_objects,
    scalar_number,
    score_item,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
    run_benchmark_execution,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS_ROOT = REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "execution" / "runners"
REWARD_3D_SERVER = (
    RUNNERS_ROOT
    / "worldolympiad"
    / "runtime"
    / "worldolympiad"
    / "3d_metrics"
    / "serve_reward_3d.py"
)
_REMOVED_AUDIT_SCRIPTS = (
    "validate_integration",
    "env_check",
    "download_datasets",
    "materialize_benchmark_assets",
    "runtime_preflight",
    "manifest_cli",
)


def _load_reward_3d_server(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone server without importing its optional GPU stack."""

    bootstrap = types.ModuleType("_bootstrap")
    bootstrap.setup_paths = lambda: None
    model = types.ModuleType("model")
    model.__path__ = []
    vlm = types.ModuleType("model.vlm")
    vlm.resolve_model_name = lambda name, backend: name
    vlm.resolve_vlm_backend = lambda backend: backend
    monkeypatch.setitem(sys.modules, "_bootstrap", bootstrap)
    monkeypatch.setitem(sys.modules, "model", model)
    monkeypatch.setitem(sys.modules, "model.vlm", vlm)

    spec = importlib.util.spec_from_file_location("test_worldolympiad_reward_3d_server", REWARD_3D_SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_runner_io_score_helpers_cover_common_official_shapes() -> None:
    assert scalar_number({"score": "2.5"}) == 2.5
    assert scalar_number({"accuracy": "0.75"}, dict_keys=("accuracy",)) == 0.75
    assert scalar_number([1, "3", None], list_mode="mean") == 2.0
    assert scalar_number([False, True], list_mode="mean", allow_bool=True) == 0.5
    assert scalar_number(-1, reject_negative=True) is None
    assert mean_numeric([None, 0.25, 0.75]) == 0.5
    assert normalize_unit_score(75.0) == 0.75
    assert score_item(0.8, "field", 4) == {
        "raw_score": 0.8,
        "source": "field",
        "sample_count": 4,
    }


def test_runner_io_exposes_canonical_serialization_helpers(tmp_path: Path) -> None:
    json_path = tmp_path / "payload.json"
    jsonl_path = tmp_path / "rows.jsonl"

    write_json(json_path, {"value": 1})
    write_jsonl(jsonl_path, [{"row": 1}, {"row": 2}])

    assert load_json(json_path) == {"value": 1}
    assert read_jsonl_objects(jsonl_path) == [{"row": 1}, {"row": 2}]
    assert utc_now_iso().endswith("Z")


def test_script_zoo_official_runners_were_removed() -> None:
    script_dir = REPO_ROOT / "scripts" / "benchmark_zoo"
    leftover = {path.name for path in script_dir.glob("*.py")} if script_dir.is_dir() else set()
    assert leftover <= {"check_realtime_regression.py", "run_video_model_benchmark_suite.py"}
    assert not any(script_dir.glob("run_*_official_runner.py")) if script_dir.is_dir() else True
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldfoundry.evaluation.tasks.execution.framework.script_paths")
    for name in _REMOVED_AUDIT_SCRIPTS:
        assert importlib.util.find_spec(f"worldfoundry.evaluation.tasks.execution.framework.{name}") is None
        assert not (script_dir / f"{name}.py").exists()


def test_official_runners_live_as_python_modules() -> None:
    runners = sorted(RUNNERS_ROOT.rglob("run_*_official_runner.py"))
    assert len(runners) >= 20
    module = importlib.import_module(
        "worldfoundry.evaluation.tasks.execution.runners.iworldbench.run_iworldbench_official_runner"
    )
    assert callable(getattr(module, "main", None))
    assert callable(getattr(module, "parse_args", None))


def test_run_benchmark_execution_is_the_orchestration_entrypoint() -> None:
    signature = inspect.signature(run_benchmark_execution)
    assert "benchmark_id" in signature.parameters
    assert "output_dir" in signature.parameters
    assert "mode" in signature.parameters
    assert not hasattr(run_benchmark_execution, "build_parser")
    with pytest.raises(TypeError):
        run_benchmark_execution()


def test_create_tiny_video_lives_on_benchmark_data(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from worldfoundry.evaluation.tasks.execution.framework import benchmark_data

    output = tmp_path / "tiny.mp4"
    assert benchmark_data.main(["--output", str(output), "--frames", "4", "--size", "16"]) == 0
    assert output.is_file()
    assert output.stat().st_size > 0


def test_reward_3d_loopback_dns_requires_every_answer_to_be_local(monkeypatch) -> None:
    server = _load_reward_3d_server(monkeypatch)
    loopback = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
    public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", 0))

    monkeypatch.setattr(server.socket, "getaddrinfo", lambda *args, **kwargs: [loopback])
    assert server._is_loopback_host("reward.internal") is True
    monkeypatch.setattr(server.socket, "getaddrinfo", lambda *args, **kwargs: [loopback, public])
    assert server._is_loopback_host("reward.internal") is False
    assert server._is_loopback_host("0.0.0.0") is False


def test_reward_3d_bearer_auth_is_constant_time_and_required_when_configured(monkeypatch) -> None:
    server = _load_reward_3d_server(monkeypatch)
    token = "a-secure-token-with-entropy"

    assert server._is_authorized({}, None) is True
    assert server._is_authorized({}, token) is False
    assert server._is_authorized({"Authorization": f"Bearer {token}"}, token) is True
    assert server._is_authorized({"Authorization": "Bearer wrong"}, token) is False
    main_source = inspect.getsource(server.main)
    assert "REWARD_3D_ALLOW_NON_LOOPBACK" in main_source
    assert "REWARD_3D_AUTH_TOKEN" in main_source


def test_reward_3d_file_paths_stay_inside_configured_roots(tmp_path: Path, monkeypatch) -> None:
    server = _load_reward_3d_server(monkeypatch)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    outside_root = tmp_path / "outside"
    input_root.mkdir()
    output_root.mkdir()
    outside_root.mkdir()
    video = input_root / "sample.mp4"
    outside = outside_root / "secret.mp4"
    video.write_bytes(b"video")
    outside.write_bytes(b"secret")
    roots = (input_root.resolve(),)

    assert server._resolve_input_file(video, roots, field="video_path") == video.resolve()
    with pytest.raises(server.RequestError, match="outside") as read_error:
        server._resolve_input_file(outside, roots, field="video_path")
    assert read_error.value.status == 403
    with pytest.raises(server.RequestError, match="outside") as write_error:
        server._resolve_output_file(
            outside_root / "trajectory.json",
            default=output_root / "default.json",
            output_root=output_root.resolve(),
        )
    assert write_error.value.status == 403


def test_reward_3d_json_body_limit_is_checked_before_read(monkeypatch) -> None:
    server = _load_reward_3d_server(monkeypatch)
    handler = object.__new__(server.Reward3DHandler)
    handler.headers = {"Content-Type": "application/json", "Content-Length": "11"}
    handler.server = types.SimpleNamespace(max_body_bytes=10)
    handler.close_connection = False

    with pytest.raises(server.RequestError, match="too large") as error:
        handler._read_json_body()
    assert error.value.status == 413
    assert handler.close_connection is True

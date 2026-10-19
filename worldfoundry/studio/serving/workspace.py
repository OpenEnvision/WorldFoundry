from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from worldfoundry.core.io.paths import project_root
from worldfoundry.evaluation.tasks.catalog.workspace_registry import (
    validate_workspace_registry,
    workspace_benchmark_runtime_hint,
    workspace_benchmark_runtime_hints,
)
from worldfoundry.runtime.interactive_inference_catalog import (
    INTERACTIVE_INFERENCE_SPECS,
    interactive_task_for_variant,
)
from worldfoundry.studio.inference.catalog import (
    CatalogEntry,
    find_entry,
    template_id_hint,
)
from worldfoundry.studio.inference.dispatch import (
    dispatch_spec_for_inference,
    run_manager_payload_in_conda,
)
from worldfoundry.studio.inference.execution import (
    RunRecord,
    StudioManager,
    _is_gaussian_splat_ply,
    bind_run_preview_image,
)
from worldfoundry.studio.inference.requests import (
    DEFAULT_SETTINGS,
    InferenceRequestError,
    JobCreateRequest,
    _entry_inference_spec,
    _entry_runtime_options,
    _entry_workload,
    _resolve_inference_contract,
    _supports_attention_backend,
    _validate_inference_payload,
    _variant_load_kwargs,
    _variant_model_ref,
    build_inference_run_kwargs,
)
from worldfoundry.studio.serving import (
    bind_security_warning,
    path_allowed,
    request_token_valid,
    require_auth_token_for_host,
)
from worldfoundry.studio.serving.jobs import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    StudioJob,
    StudioJobStore,
    format_elapsed,
)
from worldfoundry.studio.ui.catalog import _studio_catalog
from worldfoundry.studio.ui.workspace import WORKSPACE_HTML
from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
from worldfoundry.studio.visualization.backends.viser import npz_has_supported_geometry, viser_orientation_defaults
from worldfoundry.studio.visualization.providers.run_record import first_geometry_point_candidate, first_splat_asset
from worldfoundry.studio.visualization.studio import prepare_run_record

logger = logging.getLogger(__name__)

REPO_ROOT = project_root(__file__)
WORKSPACE_MAX_JOBS = max(1, int(os.getenv("WORLDFOUNDRY_WORKSPACE_MAX_JOBS", "8") or "8"))
WORKSPACE_MAX_CACHED_PIPELINES = max(
    0,
    int(os.getenv("WORLDFOUNDRY_WORKSPACE_MAX_CACHED_PIPELINES", str(WORKSPACE_MAX_JOBS)) or "0"),
)
MANAGER = StudioManager(max_cached_pipelines=WORKSPACE_MAX_CACHED_PIPELINES, record_processor=prepare_run_record)


def _initial_studio_job_counter(workspace_root: str) -> int:
    max_counter = 0
    runtime_jobs_root = Path(workspace_root) / "runtime_jobs"
    for path in runtime_jobs_root.glob("studio-*"):
        suffix = path.name.removeprefix("studio-")
        if suffix.isdigit():
            max_counter = max(max_counter, int(suffix))
    return max_counter


JOBS = StudioJobStore(
    max_workers=WORKSPACE_MAX_JOBS,
    initial_counter=_initial_studio_job_counter(MANAGER.workspace_root),
)
_PACKAGED_OPENENVISION_LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "openenvision-logo.png"
OPENENVISION_LOGO_PATH = (
    _PACKAGED_OPENENVISION_LOGO_PATH
    if _PACKAGED_OPENENVISION_LOGO_PATH.is_file()
    else REPO_ROOT / "docs" / "fumadocs" / "public" / "openenvision-logo.png"
)
SUPPORTED_WORKSPACE_JOB_TYPES = {"inference", "evaluation"}
SETTING_CHOICES = {
    "backend": {"auto", "from_pretrained", "api_init"},
    "attention_backend": {"auto", "torch", "flash_attn_2", "flash_attn_3", "sage", "xformers"},
}
SETTINGS: dict[str, Any] = dict(DEFAULT_SETTINGS)
# FastAPI runs sync endpoints on a thread pool, so process-global mutable state
# must serialize its write paths (see also _VISUALIZER_LOCK below).
_SETTINGS_LOCK = threading.Lock()
MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4v": "video/mp4",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webm": "video/webm",
}
MEDIA_VISUALIZER_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}
SPARK_VISUALIZER_EXTS = {".spz", ".splat", ".ksplat", ".sog"}
GEOMETRY_VISUALIZER_EXTS = {".ply", ".pcd", ".xyz", ".glb", ".gltf", ".obj"}
VISUALIZER_LABELS = {
    "points": "Open in Viser",
    "spark": "Open in Spark",
    "rerun": "Open in Rerun",
    "media": "Open Media",
}


class SettingsUpdateRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class VisualizerLaunchRequest(BaseModel):
    model_id: str = ""
    asset_path: str = ""
    simulator_url: str = ""
    host: str = "127.0.0.1"
    port: int | None = None
    reuse: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ManagedVisualizer:
    mode: str
    title: str
    url: str
    health_url: str
    host: str
    port: int
    model_id: str
    asset_path: str
    command: list[str]
    log_path: Path | None
    started_at: float
    params: dict[str, Any] = field(default_factory=dict)
    process: subprocess.Popen[str] | None = None
    external: bool = False


DEFAULT_VISUALIZER_MODELS = {
    "world": "matrix-game-2",
    "spark": "vggt-omega",
    "points": "vggt-omega",
    "rerun": "vggt-omega",
    "media": "vggt-omega",
    "embodied": "openvla",
}
POINTS_VISUALIZER_PARAM_ENV = {
    "max_points": "WORLDFOUNDRY_STUDIO_VISER_MAX_POINTS",
    "point_size": "WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE",
    "point_shape": "WORLDFOUNDRY_STUDIO_VISER_POINT_SHAPE",
    "coordinate_preset": "WORLDFOUNDRY_STUDIO_VISER_COORDINATE_PRESET",
    "up_direction": "WORLDFOUNDRY_STUDIO_VISER_UP_DIRECTION",
    "alignment": "WORLDFOUNDRY_STUDIO_VISER_ALIGNMENT",
    "show_cameras": "WORLDFOUNDRY_STUDIO_VISER_SHOW_CAMERAS",
    "camera_size": "WORLDFOUNDRY_STUDIO_VISER_CAMERA_SIZE",
}
POINTS_VISUALIZER_DEFAULT_PARAMS = {
    "coordinate_preset": "asset-native",
    "up_direction": "+z",
    "alignment": "auto",
    "point_size": 0.02,
    "point_shape": "circle",
    "max_points": 400_000,
}
VISUALIZER_ASSET_REQUIRED = {"media", "points"}
VISUALIZER_URL_REQUIRED = {"embodied"}
VISUALIZER_MANAGED: dict[str, ManagedVisualizer] = {}
# Reentrant: _launch_visualizer stops/cleans existing viewers while holding it.
_VISUALIZER_LOCK = threading.RLock()


def _rerun_renderer() -> str:
    """Select the Rerun web renderer, preferring its modern WebGPU path."""

    value = os.getenv("WORLDFOUNDRY_STUDIO_RERUN_RENDERER", "webgpu").strip().lower()
    return value if value in {"webgpu", "webgl"} else "webgpu"


def _coerce_setting_value(key: str, value: Any) -> Any:
    if key not in SETTINGS:
        raise HTTPException(status_code=400, detail=f"unsupported setting: {key}")
    default = SETTINGS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"setting {key} must be an integer") from exc
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"setting {key} must be a number") from exc
    coerced = str(value)
    choices = SETTING_CHOICES.get(key)
    if choices is not None and coerced not in choices:
        raise HTTPException(status_code=400, detail=f"setting {key} must be one of: {', '.join(sorted(choices))}")
    return coerced


def _settings_file() -> Path | None:
    value = os.getenv("WORLDFOUNDRY_STUDIO_SETTINGS_FILE", "").strip()
    return Path(value).expanduser() if value else None


def _load_settings_from_disk() -> None:
    path = _settings_file()
    if path is None or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Ignoring unreadable Studio settings file %s; keeping default settings.",
            path,
            exc_info=True,
        )
        return
    if not isinstance(payload, dict):
        return
    with _SETTINGS_LOCK:
        for key, value in payload.items():
            if key in SETTINGS:
                SETTINGS[key] = _coerce_setting_value(key, value)


def _save_settings_to_disk() -> None:
    path = _settings_file()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _workspace_visualizer_dir() -> Path:
    path = Path(MANAGER.workspace_root).resolve() / "visualizers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _visualizer_public_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    return f"http://{browser_host}:{port}/"


def _visualizer_health_url(mode: str, url: str) -> str:
    if mode in {"world", "media", "spark"}:
        return url.rstrip("/") + "/healthz"
    return url


def _visualizer_process_alive(process: subprocess.Popen[str] | None) -> bool:
    return process is not None and process.poll() is None


def _visualizer_status(record: ManagedVisualizer) -> dict[str, Any]:
    running = record.external or _visualizer_process_alive(record.process)
    return {
        "mode": record.mode,
        "title": record.title,
        "url": record.url,
        "health_url": record.health_url,
        "host": record.host,
        "port": record.port,
        "model_id": record.model_id,
        "asset_path": record.asset_path,
        "params": dict(record.params),
        "command": record.command,
        "log_path": str(record.log_path) if record.log_path else "",
        "started_at": record.started_at,
        "running": running,
        "external": record.external,
        "returncode": record.process.poll() if record.process is not None else None,
    }


def _cleanup_finished_visualizer(mode: str) -> None:
    with _VISUALIZER_LOCK:
        record = VISUALIZER_MANAGED.get(mode)
        if record is None or record.external or _visualizer_process_alive(record.process):
            return
        VISUALIZER_MANAGED.pop(mode, None)


def _stop_visualizer(mode: str) -> bool:
    # Idempotent: the pop makes concurrent or repeated calls (stop endpoint,
    # shutdown event, atexit) observe no record and return without side effects.
    with _VISUALIZER_LOCK:
        record = VISUALIZER_MANAGED.pop(mode, None)
        if record is None or record.external or record.process is None:
            return record is not None
        process = record.process
        if process.poll() is not None:
            return True
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=4)
        return True


def _stop_all_visualizers() -> None:
    """Best-effort teardown of managed viewer subprocesses (shutdown + atexit)."""

    for mode in list(VISUALIZER_MANAGED):
        try:
            _stop_visualizer(mode)
        except Exception:
            logger.warning("Failed to stop %s visualizer during shutdown.", mode, exc_info=True)


_WORKSPACE_SHUTDOWN_LOCK = threading.Lock()
_WORKSPACE_SHUTDOWN_COMPLETE = False


def _shutdown_workspace(
    grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
) -> None:
    """Run process-global Studio teardown once from FastAPI or atexit."""

    global _WORKSPACE_SHUTDOWN_COMPLETE
    with _WORKSPACE_SHUTDOWN_LOCK:
        if _WORKSPACE_SHUTDOWN_COMPLETE:
            return
        _WORKSPACE_SHUTDOWN_COMPLETE = True
    try:
        if grace_seconds == DEFAULT_SHUTDOWN_GRACE_SECONDS:
            JOBS.shutdown()
        else:
            JOBS.shutdown(grace_seconds=grace_seconds)
    except Exception:
        logger.warning("Failed to shut down Studio jobs cleanly.", exc_info=True)
    finally:
        _stop_all_visualizers()


# uvicorn's normal exit path fires the FastAPI shutdown event, but abnormal
# exits (unhandled exceptions before serving, SystemExit) can skip it; atexit
# is the fallback so setsid-detached viewer children never outlive Studio.
atexit.register(_shutdown_workspace)


def _tcp_port_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
    except OSError:
        return False
    return True


def _visualizer_port(mode: str, host: str, requested_port: int | None) -> int:
    backend = STUDIO_VISUALIZATIONS.backend_for(mode)
    preferred = int(requested_port or backend.default_port)
    if requested_port is not None:
        if not _tcp_port_available(host, preferred):
            raise HTTPException(status_code=409, detail=f"port {preferred} is already in use")
        return preferred
    for offset in range(64):
        port = preferred + offset
        if _tcp_port_available(host, port):
            return port
    raise HTTPException(status_code=409, detail=f"no free port found near {preferred}")


def _wait_for_visualizer(url: str, *, timeout: float = 35.0) -> bool:
    deadline = time.time() + timeout
    request = urllib.request.Request(url, headers={"User-Agent": "WorldFoundry Workspace"})
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if 200 <= int(response.status) < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _visualizer_startup_timeout(mode: str) -> float:
    if mode == "world":
        return 120.0
    if mode in {"rerun", "points"}:
        return 60.0
    return 45.0


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=4)


def _visualizer_env_overrides(mode: str, params: Mapping[str, Any]) -> dict[str, str]:
    """Convert supported visualizer params to child-process environment values."""

    if mode != "points":
        return {}
    overrides: dict[str, str] = {}
    for param_key, env_key in POINTS_VISUALIZER_PARAM_ENV.items():
        value = params.get(param_key)
        if value is None:
            continue
        if isinstance(value, bool):
            text = "1" if value else "0"
        else:
            text = str(value).strip()
        if text:
            overrides[env_key] = text
    return overrides


def _resolved_visualizer_params(
    mode: str,
    *,
    model_id: str,
    asset_path: str,
    requested: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve automatic viewer orientation without overriding explicit choices."""

    if mode != "points":
        return dict(requested)
    params = dict(POINTS_VISUALIZER_DEFAULT_PARAMS)
    params.update(viser_orientation_defaults(model_id, asset_path))
    for key, value in requested.items():
        if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
            continue
        params[key] = value
    return params


def _visualizer_reusable(
    existing: ManagedVisualizer | None,
    *,
    model_id: str,
    asset_path: str,
    params: Mapping[str, Any],
) -> bool:
    """Return whether a managed viewer already serves this exact target."""

    return bool(
        existing is not None
        and existing.model_id == model_id
        and existing.asset_path == asset_path
        and existing.params == dict(params)
        and (existing.external or _visualizer_process_alive(existing.process))
    )


def _validate_visualizer_asset(mode: str, asset_path: str) -> str:
    value = (asset_path or "").strip()
    if not value:
        if mode in VISUALIZER_ASSET_REQUIRED:
            raise HTTPException(status_code=400, detail=f"{mode} requires an asset path")
        return ""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"asset does not exist: {path}")
    return str(path)


def _workspace_child_python() -> str:
    """Return the Python entrypoint child Studio frontends should reuse."""

    return (
        os.getenv("WORLDFOUNDRY_STUDIO_CHILD_PYTHON", "").strip() or os.getenv("PYTHON", "").strip() or sys.executable
    )


def _visualizer_launch_command(
    mode: str, payload: VisualizerLaunchRequest, host: str, port: int
) -> tuple[list[str], str, str]:
    STUDIO_VISUALIZATIONS.backend_for(mode)
    model_id = (payload.model_id or DEFAULT_VISUALIZER_MODELS.get(mode) or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail=f"{mode} requires a model id")
    try:
        find_entry(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    asset_path = _validate_visualizer_asset(mode, payload.asset_path)
    external_url = (payload.simulator_url or "").strip()
    if mode in VISUALIZER_URL_REQUIRED and not external_url:
        raise HTTPException(status_code=400, detail=f"{mode} requires a simulator URL")
    if mode in {"embodied", "rerun"} and external_url:
        return [], model_id, asset_path
    cmd = [
        _workspace_child_python(),
        "-m",
        "worldfoundry.studio.ui.launcher",
        model_id,
        "--frontend",
        mode,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if asset_path:
        cmd.extend(["--asset", asset_path])
    return cmd, model_id, asset_path


def _visualizer_mode_for_artifact(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix in MEDIA_VISUALIZER_EXTS:
        return "media"
    if suffix == ".rrd":
        return "rerun"
    if suffix in SPARK_VISUALIZER_EXTS:
        return "spark"
    if suffix == ".ply" and _is_gaussian_splat_ply(path):
        return "spark"
    if suffix == ".npz":
        return "points" if npz_has_supported_geometry(path) else ""
    if suffix in GEOMETRY_VISUALIZER_EXTS:
        return "points"
    return ""


def _artifact_visualization_action(
    name: str,
    path_text: str,
    *,
    model_id: str = "",
    output_dir: str = "",
) -> dict[str, Any] | None:
    mode = _visualizer_mode_for_artifact(path_text)
    if not mode:
        return None
    path = str(path_text)
    return {
        "name": name or Path(path).name,
        "path": path,
        "mode": mode,
        "label": VISUALIZER_LABELS.get(mode, f"Open in {mode}"),
        "model_id": model_id or DEFAULT_VISUALIZER_MODELS.get(mode, ""),
        "output_dir": output_dir,
    }


def _launch_visualizer(mode: str, payload: VisualizerLaunchRequest) -> dict[str, Any]:
    # Serialize launches: concurrent requests for one mode would otherwise both
    # pass the reuse check and race subprocess start/stop on VISUALIZER_MANAGED.
    with _VISUALIZER_LOCK:
        return _launch_visualizer_locked(mode, payload)


def _launch_visualizer_locked(mode: str, payload: VisualizerLaunchRequest) -> dict[str, Any]:
    if mode not in STUDIO_VISUALIZATIONS.modes:
        raise HTTPException(status_code=404, detail=f"unknown visualizer: {mode}")
    requested_model_id = (payload.model_id or DEFAULT_VISUALIZER_MODELS.get(mode) or "").strip()
    requested_asset_path = (payload.asset_path or "").strip()
    if requested_asset_path:
        requested_asset_path = str(Path(requested_asset_path).expanduser().resolve())
    params = _resolved_visualizer_params(
        mode,
        model_id=requested_model_id,
        asset_path=requested_asset_path,
        requested=dict(payload.params or {}),
    )
    _cleanup_finished_visualizer(mode)
    existing = VISUALIZER_MANAGED.get(mode)
    if payload.reuse and _visualizer_reusable(
        existing,
        model_id=requested_model_id,
        asset_path=requested_asset_path,
        params=params,
    ):
        return _visualizer_status(existing)

    if existing is not None:
        _stop_visualizer(mode)

    backend = STUDIO_VISUALIZATIONS.backend_for(mode)
    host = (payload.host or "127.0.0.1").strip() or "127.0.0.1"
    external_url = (payload.simulator_url or "").strip()
    port = int(payload.port or backend.default_port)
    if not (mode in {"embodied", "rerun"} and external_url):
        port = _visualizer_port(mode, host, payload.port)
    command, model_id, asset_path = _visualizer_launch_command(mode, payload, host, port)

    if mode in {"embodied", "rerun"} and external_url:
        record = ManagedVisualizer(
            mode=mode,
            title=backend.title,
            url=external_url,
            health_url=external_url,
            host=host,
            port=port,
            model_id=model_id,
            asset_path=asset_path,
            command=[],
            log_path=None,
            started_at=time.time(),
            params=params,
            process=None,
            external=True,
        )
        VISUALIZER_MANAGED[mode] = record
        return _visualizer_status(record)

    url = _visualizer_public_url(host, port)
    rerun_grpc_port: int | None = None
    rerun_ws_port: int | None = None
    if mode == "rerun":
        for candidate in range(port + 1, port + 65):
            if _tcp_port_available(host, candidate) and _tcp_port_available(host, candidate + 1):
                rerun_ws_port = candidate
                rerun_grpc_port = candidate + 1
                break
        if rerun_ws_port is None or rerun_grpc_port is None:
            raise HTTPException(status_code=409, detail=f"no free Rerun data ports found near {port + 1}")
        browser_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
        source_url = f"ws://{browser_host}:{rerun_ws_port}"
        url = url.rstrip("/") + "/?url=" + urllib.parse.quote(source_url, safe="") + f"&renderer={_rerun_renderer()}"
    health_url = _visualizer_health_url(mode, url)
    log_path = _workspace_visualizer_dir() / f"{mode}-{int(time.time())}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(item for item in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if item)
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    visualizer_env = _visualizer_env_overrides(mode, params)
    if rerun_grpc_port is not None:
        visualizer_env["WORLDFOUNDRY_STUDIO_RERUN_GRPC_PORT"] = str(rerun_grpc_port)
    if rerun_ws_port is not None:
        visualizer_env["WORLDFOUNDRY_STUDIO_RERUN_WS_PORT"] = str(rerun_ws_port)
    env.update(visualizer_env)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        if visualizer_env:
            log_file.write("# visualizer env " + json.dumps(visualizer_env, sort_keys=True) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            # start_new_session is the thread-safe equivalent of
            # CPython documents the legacy pre-exec callback as unsafe in
            # multi-threaded programs, and this Popen runs inside FastAPI's
            # threadpool (SA-8/PLW1509).  The child still becomes its own
            # process group leader, which _terminate_process_group relies on.
            start_new_session=hasattr(os, "setsid"),
        )
    ready = _wait_for_visualizer(health_url, timeout=_visualizer_startup_timeout(mode))
    if not ready:
        returncode = process.poll()
        if returncode is None:
            _terminate_process_group(process)
        details = ""
        with contextlib.suppress(OSError):
            details = log_path.read_text(encoding="utf-8")[-4000:]
        if returncode is None:
            raise HTTPException(
                status_code=504,
                detail=f"{mode} did not become ready at {health_url} within {_visualizer_startup_timeout(mode):.0f}s.\n{details}",
            )
        raise HTTPException(status_code=500, detail=f"{mode} exited before it was ready.\n{details}")

    record = ManagedVisualizer(
        mode=mode,
        title=backend.title,
        url=url,
        health_url=health_url,
        host=host,
        port=port,
        model_id=model_id,
        asset_path=asset_path,
        command=command,
        log_path=log_path,
        started_at=time.time(),
        params=params,
        process=process,
    )
    VISUALIZER_MANAGED[mode] = record
    return _visualizer_status(record)


def _catalog_url_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("http://", "https://")):
            return text
        return ""
    if isinstance(value, Mapping):
        for key in ("url", "href", "link"):
            text = str(value.get(key) or "").strip()
            if text.startswith(("http://", "https://")):
                return text
    return ""


def _catalog_link_keys(*values: str) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        candidates = {
            text,
            text.replace("_", "-"),
            text.replace("-", "_"),
        }
        for candidate in candidates:
            normalized = candidate.casefold()
            compact = normalized.replace("-", "").replace("_", "")
            for key in (normalized, compact):
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
    return tuple(keys)


def _merge_official_links(*link_rows: Mapping[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for row in link_rows:
        for key, value in row.items():
            text = str(value or "").strip()
            if text:
                merged[key] = text
    return merged


def _normalize_official_links(links: Mapping[str, str]) -> dict[str, str]:
    """Keep GitHub slots for github.com URLs; treat other official sites as project pages."""
    normalized = {key: value for key, value in links.items() if value}
    github = normalized.get("github") or ""
    if github and "github.com/" not in github.casefold():
        normalized.setdefault("project", github)
        normalized.pop("github", None)
    return normalized


_PAPER_SOURCE_KEYS = (
    "paper",
    "paper_url",
    "paper_plus_plus",
    "arxiv",
    "arxiv_url",
    "technical_report",
    "tech_report",
    "causal_paper",
    "publication",
    "pdf",
)
_PROJECT_SOURCE_KEYS = (
    "project_page",
    "project",
    "homepage",
    "website",
    "webpage",
    "project_url",
    "official_page",
    "demo_page",
    "worldarena_space",
)


def _first_catalog_url(*values: Any) -> str:
    for value in values:
        text = _catalog_url_value(value)
        if text:
            return text
    return ""


def _official_links_from_sources(sources: Mapping[str, Any]) -> dict[str, str]:
    links: dict[str, str] = {}
    github = _catalog_url_value(sources.get("github")) or _catalog_url_value(sources.get("inference_github"))
    if github:
        links["github"] = github
    paper = _first_catalog_url(*(sources.get(key) for key in _PAPER_SOURCE_KEYS))
    if paper:
        links["paper"] = paper
    project = _first_catalog_url(*(sources.get(key) for key in _PROJECT_SOURCE_KEYS))
    if project:
        links["project"] = project
    return links


def _official_links_from_catalog_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    from worldfoundry.evaluation.models.catalog.schema import _github_url_from_sources

    links: dict[str, str] = {}
    github = _github_url_from_sources(entry)
    if github:
        links["github"] = github

    official_sources = entry.get("official_sources")
    sources = official_sources if isinstance(official_sources, Mapping) else {}
    if not sources:
        raw_sources = entry.get("sources")
        if isinstance(raw_sources, Mapping):
            sources = raw_sources

    source_status = entry.get("source_status")
    if isinstance(source_status, Mapping):
        links = _merge_official_links(links, _official_links_from_sources(source_status))

    links = _merge_official_links(links, _official_links_from_sources(sources))

    paper = _first_catalog_url(*(entry.get(key) for key in _PAPER_SOURCE_KEYS))
    if paper:
        links.setdefault("paper", paper)
    project = _first_catalog_url(*(entry.get(key) for key in _PROJECT_SOURCE_KEYS))
    if project:
        links.setdefault("project", project)
    source = entry.get("source")
    if isinstance(source, Mapping):
        links = _merge_official_links(links, _official_links_from_sources(source))
        github = _catalog_url_value(source.get("official_repo_url"))
        if github:
            links.setdefault("github", github)
    return _normalize_official_links(links)


def _catalog_entry_link_keys(entry: Mapping[str, Any]) -> tuple[str, ...]:
    aliases = entry.get("aliases") or ()
    alias_rows = aliases if isinstance(aliases, (list, tuple)) else (aliases,)
    keys = _catalog_link_keys(
        str(entry.get("model_id") or entry.get("id") or ""),
        str(entry.get("pipeline_binding") or ""),
        *(str(alias) for alias in alias_rows),
    )
    variants = entry.get("variants") or ()
    if isinstance(variants, (list, tuple)):
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            keys += _catalog_link_keys(
                str(variant.get("id") or variant.get("variant_id") or ""),
                str(variant.get("pipeline_binding") or ""),
            )
    return keys


def _entry_link_keys(entry: CatalogEntry) -> tuple[str, ...]:
    return _catalog_link_keys(entry.model_id, *entry.aliases)


def _github_url_from_model_ref(model_ref: str) -> str:
    text = str(model_ref or "").strip()
    if "github.com/" in text.casefold():
        return text
    return ""


@lru_cache(maxsize=1)
def _model_catalog_links_index() -> dict[str, dict[str, str]]:
    from worldfoundry.evaluation.models.catalog.manifest import _catalog_paths, _iter_catalog_mappings

    index: dict[str, dict[str, str]] = {}
    for path in _catalog_paths():
        for entry in _iter_catalog_mappings(path):
            links = _official_links_from_catalog_entry(entry)
            if not links:
                continue
            for key in _catalog_entry_link_keys(entry):
                if not key:
                    continue
                existing = index.get(key)
                index[key] = _merge_official_links(existing or {}, links)
    return index


def _entry_official_links(entry: CatalogEntry) -> dict[str, str]:
    index = _model_catalog_links_index()
    links: dict[str, str] = {}
    for key in _entry_link_keys(entry):
        links = _merge_official_links(links, index.get(key, {}))
        if all(links.get(name) for name in ("github", "project", "paper")):
            break
    github_ref = _github_url_from_model_ref(entry.default_model_ref)
    if github_ref:
        links.setdefault("github", github_ref)
    return _normalize_official_links(links)


def _model_payload(entry: CatalogEntry) -> dict[str, Any]:
    template_id = template_id_hint(entry)
    infer_spec = _entry_inference_spec(entry)
    variant_payloads = []
    extra_by_id = {
        str(raw_variant.get("variant_id") or "").strip(): raw_variant for raw_variant in entry.extra_variants
    }
    for variant in infer_spec.variants:
        row = variant.to_dict()
        row["model_ref"] = _variant_model_ref(entry, variant)
        row["load_kwargs"] = _variant_load_kwargs(entry, variant)
        if entry.model_id in INTERACTIVE_INFERENCE_SPECS:
            row["tasks"] = [
                interactive_task_for_variant(infer_spec, variant, task).to_dict() for task in infer_spec.tasks
            ]
        extra = extra_by_id.get(variant.variant_id) or {}
        if extra.get("workload_type"):
            row["workload_type"] = extra["workload_type"]
        if extra.get("default_prompt"):
            row["default_prompt"] = extra["default_prompt"]
        if extra.get("default_input_path"):
            row["default_input_path"] = extra["default_input_path"]
        variant_payloads.append(row)
    return {
        "id": entry.model_id,
        "name": entry.display_name,
        "category": entry.category,
        "family": entry.family,
        "summary": entry.summary,
        "tags": list(entry.tags),
        "aliases": list(entry.aliases),
        "backend": entry.default_backend,
        "model_ref": entry.default_model_ref,
        "endpoint": entry.default_endpoint,
        "default_prompt": entry.default_prompt,
        "default_input_path": entry.default_input_path,
        "supports_stream": entry.supports_stream,
        "supports_from_pretrained": entry.supports_from_pretrained,
        "supports_api_init": entry.supports_api_init,
        "supports_attention_backend": _supports_attention_backend(entry),
        "template_id": template_id,
        "workload_type": _entry_workload(entry),
        "infer_spec": infer_spec.to_dict(),
        "variants": variant_payloads,
        "tasks": [task.to_dict() for task in infer_spec.tasks],
        "default_variant_id": infer_spec.default_variant_id,
        "default_task_id": infer_spec.default_task_id,
        "runtime_options": _entry_runtime_options(entry),
        "links": _entry_official_links(entry),
    }


_WORKSPACE_MODELS_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _workspace_models_cached() -> tuple[dict[str, Any], ...]:
    return tuple(_model_payload(entry) for entry in _studio_catalog())


def _workspace_models() -> tuple[dict[str, Any], ...]:
    """Return the catalog without duplicating an expensive cold build.

    ``functools.lru_cache`` is thread-safe for cache bookkeeping, but it may
    execute a cold miss more than once when requests arrive concurrently.
    Keep the cache lookup inside one lock so the first ``/api/models`` request
    performs the filesystem-heavy catalog build and every follower reuses it.
    """

    with _WORKSPACE_MODELS_LOCK:
        return _workspace_models_cached()


@lru_cache(maxsize=1)
def _workspace_model_ids() -> frozenset[str]:
    return frozenset(entry.model_id for entry in _studio_catalog())


def _workspace_job_output_dir(kind: str, job_id: str | None) -> str:
    identifier = job_id or "pending"
    return str(Path(MANAGER.workspace_root) / kind / identifier)


def _non_empty_list(values: Sequence[str] | None, default: Sequence[str] = ()) -> tuple[str, ...]:
    rows = tuple(str(item).strip() for item in (values or ()) if str(item).strip())
    return rows or tuple(default)


def _optional_path(value: str | Path | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_jsonable(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(item) for item in value]
    return value


@lru_cache(maxsize=1)
def _evaluation_catalog_payload() -> dict[str, Any]:
    try:
        from worldfoundry.evaluation.models.catalog.registry import discover_model_registry
        from worldfoundry.evaluation.tasks.catalog.specs import list_benchmark_zoo_cli_tasks
        from worldfoundry.evaluation.tasks.metrics.registry import list_metric_registry_entries

        benchmarks = list_benchmark_zoo_cli_tasks()
        models = [item.to_dict() for item in discover_model_registry().list()]
        metrics = [item.to_dict() for item in list_metric_registry_entries()]
        return {
            "ok": True,
            "modes": [
                "score-artifacts",
                "model-benchmark",
                "generate-and-score",
                "existing-results",
                "model",
            ],
            "benchmarks": benchmarks,
            "models": models,
            "metrics": metrics,
            "benchmark_runtime_hints": workspace_benchmark_runtime_hints(),
            "benchmark_runtime_issues": validate_workspace_registry(),
            "examples": _evaluation_examples_payload(),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - surfaced as catalog diagnostics in the UI.
        return {
            "ok": False,
            "modes": [
                "score-artifacts",
                "model-benchmark",
                "generate-and-score",
                "existing-results",
                "model",
            ],
            "benchmarks": [],
            "models": [],
            "metrics": [],
            "benchmark_runtime_hints": {},
            "benchmark_runtime_issues": [f"{type(exc).__name__}: {exc}"],
            "examples": _evaluation_examples_payload(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _evaluation_examples_payload() -> list[dict[str, Any]]:
    """Public installs use user-provided evaluation inputs."""
    return []


def _inference_backend_for_dispatch(entry: CatalogEntry, backend: str) -> str:
    selected = (backend or entry.default_backend or "auto").strip()
    if selected == "auto" and not entry.supports_from_pretrained and entry.supports_api_init:
        return "api_init"
    return selected


def _run_inference(payload: JobCreateRequest, job: StudioJob | None = None):
    entry, run_kwargs = build_inference_run_kwargs(payload, settings=SETTINGS, validate=False)
    backend = _inference_backend_for_dispatch(entry, str(run_kwargs.get("backend") or "auto"))
    spec = dispatch_spec_for_inference(entry.model_id, backend=backend, force_subprocess=True)
    if spec is not None:
        dispatch_root = Path(MANAGER.workspace_root) / "runtime_jobs" / (job.job_id if job else "direct")
        record = run_manager_payload_in_conda(
            model_id=entry.model_id,
            spec=spec,
            workspace_root=MANAGER.workspace_root,
            run_kwargs=run_kwargs,
            dispatch_root=dispatch_root,
            log_callback=job.append_log if job is not None else None,
            cancel_requested=(lambda: bool(job.cancel_requested)) if job is not None else None,
        )
        return prepare_run_record(record, entry)
    return MANAGER.run(**run_kwargs, progress_callback=None)


_PREPARED_EVALUATION_MODES = frozenset({"score-artifacts", "model-benchmark", "generate-and-score"})


def _prepared_evaluation_from_payload(payload: JobCreateRequest, output_dir: str | Path):
    """Compile a Workspace payload through the shared evaluation intent service."""

    from worldfoundry.evaluation.tasks.execution.orchestration.service import (
        GenerateAndScoreIntent,
        ModelBenchmarkIntent,
        ScoreArtifactsIntent,
        prepare_evaluation,
    )

    mode = (payload.eval_mode or "").strip().lower().replace("_", "-")
    model_parameters = dict(payload.call_kwargs or {})
    model_runtime = dict(payload.load_kwargs or {})
    if mode == "score-artifacts":
        if payload.resume or payload.metric_batch_size != 32:
            raise ValueError("resume/metric_batch_size apply to in-process scoring, not an official artifact protocol")
        intent = ScoreArtifactsIntent(
            output_dir=output_dir,
            benchmark_id=payload.benchmark_id,
            artifact_dir=_optional_path(payload.dataset_root) or _optional_path(payload.results_path) or "",
            dataset_id=_optional_path(payload.dataset_id),
            benchmark_env=model_runtime,
            benchmark_parameters=model_parameters,
            leaderboard_candidate=bool(payload.params.get("leaderboard_candidate")),
        )
    elif mode == "model-benchmark":
        intent = ModelBenchmarkIntent(
            output_dir=output_dir,
            model_id=payload.model_id,
            benchmark_id=payload.benchmark_id,
            model_variant_id=_optional_path(payload.model_variant_id or payload.variant_id),
            requests_path=_optional_path(payload.requests_path),
            dataset_root=_optional_path(payload.dataset_root),
            dataset_id=_optional_path(payload.dataset_id),
            num_samples=payload.limit,
            resume=payload.resume,
            metric_batch_size=payload.metric_batch_size,
            model_parameters=model_parameters,
            model_runtime=model_runtime,
            generation_cache_dir=_optional_path(payload.generation_cache_dir),
            generation_cache_mode=payload.generation_cache_mode or "read-write",
        )
    elif mode == "generate-and-score":
        intent = GenerateAndScoreIntent(
            output_dir=output_dir,
            model_id=payload.model_id,
            model_variant_id=_optional_path(payload.model_variant_id or payload.variant_id),
            dataset_manifest=payload.dataset_manifest,
            benchmark_id=_optional_path(payload.benchmark_id),
            metrics=_non_empty_list(payload.metrics),
            input_keys=_non_empty_list(payload.params.get("input_keys")),
            output_keys=_non_empty_list(payload.params.get("output_keys"), ("generated_video",)),
            required_artifacts=_non_empty_list(payload.required_artifacts),
            generation_defaults=dict(payload.params.get("generation_defaults") or {}),
            model_parameters=model_parameters,
            model_runtime=model_runtime,
            num_samples=payload.limit,
            resume=payload.resume,
            metric_batch_size=payload.metric_batch_size,
            generation_cache_dir=_optional_path(payload.generation_cache_dir),
            generation_cache_mode=payload.generation_cache_mode or "read-write",
        )
    else:
        raise ValueError(f"unsupported prepared evaluation mode: {mode}")
    return prepare_evaluation(intent)


def _run_evaluation(payload: JobCreateRequest, job: StudioJob | None = None) -> dict[str, Any]:
    from worldfoundry.evaluation.runner import EvaluateRunRequest, run_evaluate
    from worldfoundry.evaluation.tasks.execution.orchestration.plan import (
        evaluate_request_from_run_plan,
        load_run_plan,
        validate_run_plan,
    )

    output_dir = _optional_path(payload.output_dir) or _workspace_job_output_dir(
        "evaluations", job.job_id if job else None
    )
    intent_mode = (payload.eval_mode or "existing-results").strip().lower().replace("_", "-")
    if intent_mode in _PREPARED_EVALUATION_MODES:
        from worldfoundry.evaluation.tasks.execution.orchestration.service import execute_prepared_evaluation

        prepared = _prepared_evaluation_from_payload(payload, output_dir)
        if job is not None:
            job.append_log(
                "system",
                f"evaluation intent={prepared.intent_kind} ready={prepared.ready} "
                f"classification={prepared.classification}\n",
            )
            for issue in prepared.issues:
                job.append_log("system", f"{issue.severity}: {issue.code}: {issue.message}\n")
        result = execute_prepared_evaluation(prepared)
        result_payload = result.to_dict()
        result_payload["prepared_evaluation"] = prepared.to_dict()
        return result_payload
    if payload.run_plan_path:
        plan = load_run_plan(payload.run_plan_path)
        validation = validate_run_plan(plan)
        if job is not None:
            job.append_log("system", f"loaded run plan {payload.run_plan_path}\n")
            job.append_log("system", f"run plan fingerprint={validation.get('fingerprint')}\n")
        if not validation.get("ok"):
            issues = "; ".join(str(item) for item in validation.get("issues", ()))
            raise ValueError(f"Evaluation run plan is invalid: {issues}")
        request = replace(
            evaluate_request_from_run_plan(plan),
            resume=payload.resume,
            metric_batch_size=payload.metric_batch_size,
        )
        if payload.output_dir:
            request = replace(request, output_dir=output_dir)
    else:
        mode = (payload.eval_mode or "existing-results").strip().lower().replace("_", "-")
        request = EvaluateRunRequest(
            output_dir=output_dir,
            mode=mode,
            requests_path=_optional_path(payload.requests_path),
            results_path=_optional_path(payload.results_path),
            metrics=_non_empty_list(payload.metrics, ("artifact_count",)),
            required_artifacts=_non_empty_list(payload.required_artifacts),
            benchmark_id=_optional_path(payload.benchmark_id),
            model_id=_optional_path(payload.model_id),
            model_runner=_optional_path(payload.model_runner),
            model_zoo_manifest_dir=_optional_path(payload.model_zoo_manifest_dir),
            model_variant_id=_optional_path(payload.model_variant_id or payload.variant_id),
            model_parameters=dict(payload.call_kwargs or {}),
            model_runtime=dict(payload.load_kwargs or {}),
            model_config=payload.params.get("model_config") if isinstance(payload.params, dict) else None,
            dataset_id=_optional_path(payload.dataset_id),
            dataset=_safe_jsonable(
                {
                    "root": _optional_path(payload.dataset_root),
                    "manifest_path": _optional_path(payload.dataset_manifest),
                }
            ),
            resume=payload.resume,
            metric_batch_size=payload.metric_batch_size,
            fail_on_sample_error=bool(payload.fail_on_sample_error),
            write_artifacts_index=bool(payload.write_artifacts_index),
            generation_cache_dir=_optional_path(payload.generation_cache_dir),
            generation_cache_mode=payload.generation_cache_mode or "off",
            generation_cache_namespace=str(payload.params.get("generation_cache_namespace") or "workspace_evaluation"),
        )

    if job is not None:
        job.append_log(
            "system",
            f"evaluate mode={request.mode} output_dir={request.output_dir} metrics={','.join(request.metrics)}\n",
        )
    result = run_evaluate(request)
    payload_dict = result.to_dict()
    payload_dict["request"] = _safe_jsonable(asdict(request))
    return payload_dict


def _job_result_payload(result: Any) -> dict[str, Any] | None:
    if isinstance(result, RunRecord):
        return {
            "run_id": result.run_id,
            "output_dir": result.output_dir,
            "manifest_path": result.manifest_path,
            "preview_video": result.preview_video,
            "preview_image": result.preview_image,
            "preview_model": result.preview_model,
            "preview_splat": result.preview_splat,
            "gallery": result.gallery,
            "artifacts": result.artifacts,
            "metadata": result.metadata,
        }
    if isinstance(result, dict):
        return dict(result)
    if result is None:
        return None
    return {"value": str(result)}


def _result_output_dir(result: Any) -> str:
    if isinstance(result, RunRecord):
        return result.output_dir
    if isinstance(result, dict):
        return str(result.get("output_dir") or "")
    return ""


def _result_artifact_paths(result: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_path(name: str, path: Any) -> None:
        if not path:
            return
        text = str(path)
        if text in seen:
            return
        seen.add(text)
        rows.append((name or Path(text).name, text))

    if isinstance(result, RunRecord):
        for path in (
            result.manifest_path,
            result.preview_video,
            result.preview_image,
            result.preview_model,
            result.preview_splat,
            result.rrd_path,
            *list(result.artifacts or ()),
        ):
            add_path(Path(str(path)).name if path else "", path)
        return rows

    if isinstance(result, dict):
        for key, value in result.items():
            if not key.endswith(("_path", "_file")):
                continue
            if isinstance(value, (str, Path)) and str(value):
                add_path(key, value)
        output_dir = result.get("output_dir")
        if output_dir:
            add_path("output_dir", output_dir)
    return rows


def _resolve_under_output_dir(path_text: str, output_dir: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    if output_dir:
        candidate = Path(output_dir).expanduser() / path
        if candidate.exists():
            return str(candidate.resolve())
    if path.exists():
        return str(path.resolve())
    return str(path)


def _result_preview_fields(result: Any) -> tuple[str, str, str]:
    if isinstance(result, RunRecord):
        return (
            str(result.preview_model or ""),
            str(result.preview_splat or ""),
            str(result.rrd_path or ""),
        )
    if isinstance(result, dict):
        return (
            str(result.get("preview_model") or ""),
            str(result.get("preview_splat") or ""),
            str(result.get("rrd_path") or ""),
        )
    return ("", "", "")


def _primary_visualization_path_by_mode(result: Any) -> dict[str, str]:
    """Pick one canonical artifact per visualizer mode for gallery/detail actions."""

    artifact_paths = [path for _, path in _result_artifact_paths(result)]
    output_dir = _result_output_dir(result)
    preview_model, preview_splat, rrd_path = _result_preview_fields(result)
    selected: dict[str, str] = {}

    if rrd_path and _visualizer_mode_for_artifact(rrd_path) == "rerun":
        selected["rerun"] = rrd_path
    else:
        for path in artifact_paths:
            if _visualizer_mode_for_artifact(path) == "rerun":
                selected["rerun"] = path
                break

    splat_path = ""
    if preview_splat and _visualizer_mode_for_artifact(preview_splat) == "spark":
        splat_path = preview_splat
    if not splat_path:
        guessed_path, _ = first_splat_asset(artifact_paths, gs_ply_predicate=_is_gaussian_splat_ply)
        splat_path = guessed_path or ""
    if splat_path:
        selected["spark"] = splat_path

    points_path = ""
    if preview_model and _visualizer_mode_for_artifact(preview_model) == "points":
        points_path = preview_model
    if not points_path and output_dir:
        geometry_rel = first_geometry_point_candidate(
            artifact_paths,
            output_dir,
            gs_ply_predicate=_is_gaussian_splat_ply,
        )
        if geometry_rel:
            points_path = _resolve_under_output_dir(geometry_rel, output_dir)
    if not points_path:
        for path in artifact_paths:
            if _visualizer_mode_for_artifact(path) == "points":
                points_path = path
                break
    if points_path:
        selected["points"] = points_path

    return selected


def _result_visualization_actions(result: Any, *, model_id: str = "") -> list[dict[str, Any]]:
    output_dir = _result_output_dir(result)
    resolved_model_id = model_id
    if isinstance(result, RunRecord):
        resolved_model_id = result.model_id
    selected = _primary_visualization_path_by_mode(result)
    actions: list[dict[str, Any]] = []
    for mode in ("rerun", "points", "spark"):
        path = selected.get(mode)
        if not path:
            continue
        action = _artifact_visualization_action(
            Path(path).name,
            path,
            model_id=resolved_model_id,
            output_dir=output_dir,
        )
        if action is not None:
            actions.append(action)
    return actions


def _active_run_ids() -> set[str]:
    rows: set[str] = set()
    for job in JOBS.list():
        if isinstance(job.result, RunRecord):
            rows.add(job.result.run_id)
    return rows


def _recent_persisted_runs(limit: int = 100) -> list[RunRecord]:
    active = _active_run_ids()
    return [record for record in MANAGER.list_recent_runs(limit=limit) if record.run_id not in active]


_REGISTERED_ARTIFACT_CACHE_LOCK = threading.Lock()
_REGISTERED_ARTIFACT_CACHE_SIGNATURE: tuple[Any, ...] | None = None
_REGISTERED_ARTIFACT_CACHE_PATHS: frozenset[Path] = frozenset()


def _registered_artifact_paths() -> set[Path]:
    """Return the exact-path artifact allowlist, reusing an unchanged index."""

    jobs = JOBS.list()
    active_run_ids = {job.result.run_id for job in jobs if isinstance(job.result, RunRecord)}
    persisted_runs = [record for record in MANAGER.list_recent_runs(limit=100) if record.run_id not in active_run_ids]
    job_rows = [(job, _result_artifact_paths(job.result)) for job in jobs]
    persisted_rows = [(record, _result_artifact_paths(record)) for record in persisted_runs]
    signature: tuple[Any, ...] = (
        id(JOBS),
        id(MANAGER),
        tuple((job.job_id, job.status, id(job.result), tuple(path for _, path in rows)) for job, rows in job_rows),
        tuple((record.run_id, id(record), tuple(path for _, path in rows)) for record, rows in persisted_rows),
    )

    global _REGISTERED_ARTIFACT_CACHE_PATHS, _REGISTERED_ARTIFACT_CACHE_SIGNATURE
    with _REGISTERED_ARTIFACT_CACHE_LOCK:
        if signature == _REGISTERED_ARTIFACT_CACHE_SIGNATURE:
            return set(_REGISTERED_ARTIFACT_CACHE_PATHS)

        paths: set[Path] = set()
        for _, rows in job_rows:
            for _, path in rows:
                try:
                    paths.add(Path(path).expanduser().resolve())
                except (OSError, RuntimeError):
                    continue
        for _, rows in persisted_rows:
            for _, path in rows:
                try:
                    paths.add(Path(path).expanduser().resolve())
                except (OSError, RuntimeError):
                    continue
        _REGISTERED_ARTIFACT_CACHE_SIGNATURE = signature
        _REGISTERED_ARTIFACT_CACHE_PATHS = frozenset(paths)
        return set(paths)


def _gallery_poster_url(*, job_id: str = "", run_id: str = "", record: RunRecord) -> str:
    """Advertise a still for video cards even when the run never persisted preview_image."""
    if not (record.preview_video or record.preview_image or record.gallery):
        return ""
    if job_id:
        return f"/api/jobs/{job_id}/image"
    if run_id:
        return f"/api/runs/{run_id}/image"
    return ""


def _gallery_row_from_job(job: StudioJob) -> dict[str, Any] | None:
    if not isinstance(job.result, RunRecord):
        return None
    image_url = _gallery_poster_url(job_id=job.job_id, record=job.result)
    return {
        "job_id": job.job_id,
        "run_id": job.result.run_id,
        "title": job.title,
        "model_name": job.display_name,
        "model_id": job.model_id,
        "prompt": dict(job.metadata).get("prompt", ""),
        "video_url": f"/api/jobs/{job.job_id}/video" if job.result.preview_video else "",
        "image_url": image_url,
        "poster_url": image_url,
        "model_url": f"/api/jobs/{job.job_id}/model" if job.result.preview_model else "",
        "output_dir": job.result.output_dir,
        "visualization_actions": _result_visualization_actions(job.result, model_id=job.model_id),
    }


def _invalidate_registered_artifact_cache() -> None:
    global _REGISTERED_ARTIFACT_CACHE_SIGNATURE, _REGISTERED_ARTIFACT_CACHE_PATHS
    with _REGISTERED_ARTIFACT_CACHE_LOCK:
        _REGISTERED_ARTIFACT_CACHE_SIGNATURE = None
        _REGISTERED_ARTIFACT_CACHE_PATHS = frozenset()


def _delete_gallery_item(job_id: str = "", run_id: str = "") -> dict[str, Any]:
    """Remove a Gallery row and its persisted Studio run directory."""

    job_id = str(job_id or "").strip()
    run_id = str(run_id or "").strip()
    if not job_id and not run_id:
        raise HTTPException(status_code=400, detail="job_id or run_id is required")

    deleted_job_id = ""
    if job_id:
        job = JOBS.get(job_id)
        if job is None:
            if not run_id:
                raise HTTPException(status_code=404, detail="unknown job")
        else:
            if not job.terminal:
                raise HTTPException(status_code=409, detail=f"cannot delete a {job.status} job")
            if not run_id and isinstance(job.result, RunRecord):
                run_id = str(job.result.run_id or "").strip()
            try:
                JOBS.delete(job_id)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            deleted_job_id = job_id

    deleted_run_id = ""
    if run_id:
        try:
            MANAGER.delete_run(run_id)
            deleted_run_id = run_id
        except KeyError as exc:
            if not deleted_job_id:
                raise HTTPException(status_code=404, detail="run not found") from exc

    _invalidate_registered_artifact_cache()
    return {"ok": True, "job_id": deleted_job_id, "run_id": deleted_run_id}


def _gallery_row_from_run(record: RunRecord) -> dict[str, Any]:
    metadata = dict(record.metadata or {})
    request = metadata.get("request")
    prompt = request.get("prompt", "") if isinstance(request, dict) else ""
    image_url = _gallery_poster_url(run_id=record.run_id, record=record)
    return {
        "job_id": "",
        "run_id": record.run_id,
        "title": record.display_name or record.model_id or record.run_id,
        "model_name": record.display_name or record.model_id,
        "model_id": record.model_id,
        "prompt": prompt,
        "video_url": f"/api/runs/{record.run_id}/video" if record.preview_video else "",
        "image_url": image_url,
        "poster_url": image_url,
        "model_url": f"/api/runs/{record.run_id}/model" if record.preview_model else "",
        "output_dir": record.output_dir,
        "visualization_actions": _result_visualization_actions(record, model_id=record.model_id),
    }


def _job_payload(job: StudioJob, *, include_logs: bool = False) -> dict[str, Any]:
    return {
        "id": job.job_id,
        "job_id": job.job_id,
        "title": job.title,
        "job_type": job.job_type,
        "model_id": job.model_id,
        "model_name": job.display_name,
        "action": job.action,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "elapsed": format_elapsed(job),
        "error": job.error,
        "metadata": dict(job.metadata),
        "result": _job_result_payload(job.result),
        "visualization_actions": _result_visualization_actions(job.result, model_id=job.model_id),
        "output_dir": _result_output_dir(job.result) or str(dict(job.metadata).get("output_dir") or ""),
        "logs": job.logs[-200:] if include_logs else [],
    }


def _run_payload(record: RunRecord) -> dict[str, Any]:
    result = _job_result_payload(record) or {}
    return {
        "id": record.run_id,
        "run_id": record.run_id,
        "job_id": "",
        "title": record.display_name or record.model_id or record.run_id,
        "job_type": "inference",
        "model_id": record.model_id,
        "model_name": record.display_name or record.model_id,
        "status": "completed",
        "metadata": dict(record.metadata or {}),
        "result": result,
        "visualization_actions": _result_visualization_actions(record, model_id=record.model_id),
        "output_dir": record.output_dir,
    }


def _file_media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _iter_file_range(path: Path, start: int, end: int):
    chunk_size = 1024 * 1024
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _range_response(path: Path, range_header: str | None, media_type: str, headers: dict[str, str]) -> Response | None:
    if not range_header:
        return None
    unit, _, raw_range = range_header.partition("=")
    if unit.strip().lower() != "bytes" or "," in raw_range:
        return None
    file_size = path.stat().st_size
    start_text, _, end_text = raw_range.strip().partition("-")
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_size = int(end_text)
            start = max(file_size - suffix_size, 0)
            end = file_size - 1
    except ValueError:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{file_size}"})
    if start < 0 or end < start or start >= file_size:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{file_size}"})
    end = min(end, file_size - 1)
    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            **headers,
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{file_size}",
        },
    )


def _safe_file_response(path_text: str | None, request: Request | None = None) -> Response:
    if not path_text:
        raise HTTPException(status_code=404, detail="file not found")
    path = Path(path_text).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    workspace_root = Path(MANAGER.workspace_root).resolve()
    # Registered artifacts may legitimately live outside the workspace root
    # (session-scoped exact-path allowlist); everything else must resolve under
    # the workspace via the shared serving.path_allowed check.
    registered = path in _registered_artifact_paths()
    if not registered and not path_allowed(path, (workspace_root,)):
        raise HTTPException(status_code=403, detail="file is outside Studio workspace")
    media_type = _file_media_type(path)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=86400",
        "X-Content-Type-Options": "nosniff",
    }
    range_response = _range_response(path, request.headers.get("range") if request else None, media_type, headers)
    if range_response is not None:
        return range_response
    return FileResponse(path, media_type=media_type, headers=headers)


def create_app(auth_token: str = "") -> FastAPI:
    """Build the Workspace FastAPI app.

    This app is a single-user local tool: SETTINGS, VISUALIZER_MANAGED, JOBS,
    and MANAGER are process-global, so every connected client shares one
    configuration and one managed viewer per mode. Multi-user isolation is out
    of scope. ``auth_token``, when non-empty, is required on every non-static
    request (``Authorization: Bearer <token>`` or ``?token=<token>``); it is
    enforced by ``main()`` for non-loopback binds.
    """

    _load_settings_from_disk()
    app = FastAPI(title="OpenEnvision Workspace")
    app.router.add_event_handler("shutdown", _shutdown_workspace)

    if auth_token:
        static_paths = {"/", "/favicon.ico", "/assets/openenvision-logo.png"}

        @app.middleware("http")
        async def require_studio_token(request: Request, call_next):
            if request.url.path in static_paths:
                return await call_next(request)
            if not request_token_valid(
                auth_token,
                authorization_header=request.headers.get("authorization"),
                query_token=request.query_params.get("token"),
            ):
                return Response(
                    status_code=401,
                    content="Missing or invalid Studio auth token.",
                    media_type="text/plain",
                )
            return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return WORKSPACE_HTML

    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(OPENENVISION_LOGO_PATH, media_type="image/png")

    @app.get("/assets/openenvision-logo.png")
    def openenvision_logo() -> FileResponse:
        return FileResponse(OPENENVISION_LOGO_PATH, media_type="image/png")

    @app.exception_handler(InferenceRequestError)
    async def inference_request_error(request: Request, exc: InferenceRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return dict(SETTINGS)

    @app.post("/api/settings")
    def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
        coerced = {key: _coerce_setting_value(key, value) for key, value in payload.values.items()}
        with _SETTINGS_LOCK:
            SETTINGS.update(coerced)
            _save_settings_to_disk()
            return dict(SETTINGS)

    @app.get("/api/models")
    def list_models(workload_type: str | None = None) -> list[dict[str, Any]]:
        models = [dict(model) for model in _workspace_models()]
        if workload_type and workload_type != "all":
            models = [
                model for model in models if model["workload_type"] == workload_type or workload_type == "inference"
            ]
        return models

    @app.get("/api/evaluation/catalog")
    def evaluation_catalog() -> dict[str, Any]:
        return _evaluation_catalog_payload()

    @app.post("/api/evaluation/prepare")
    def prepare_evaluation_request(payload: JobCreateRequest) -> dict[str, Any]:
        output_dir = _optional_path(payload.output_dir) or Path(MANAGER.workspace_root) / "evaluations" / "prepared"
        try:
            return _prepared_evaluation_from_payload(payload, output_dir).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/evaluation/vbench/dimensions")
    def evaluation_vbench_dimensions() -> dict[str, Any]:
        return workspace_benchmark_runtime_hint("vbench")

    @app.get("/api/evaluation/benchmarks/{benchmark_id}/runtime")
    def evaluation_benchmark_runtime(benchmark_id: str) -> dict[str, Any]:
        hint = workspace_benchmark_runtime_hint(benchmark_id)
        if not hint:
            raise HTTPException(status_code=404, detail=f"no Workspace runtime hint for benchmark: {benchmark_id}")
        return hint

    @app.get("/api/visualizers")
    def list_visualizers() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mode in sorted(STUDIO_VISUALIZATIONS.modes):
            _cleanup_finished_visualizer(mode)
            backend = STUDIO_VISUALIZATIONS.backend_for(mode)
            running = VISUALIZER_MANAGED.get(mode)
            rows.append(
                {
                    "mode": backend.mode,
                    "title": backend.title,
                    "default_port": backend.default_port,
                    "default_model": DEFAULT_VISUALIZER_MODELS.get(mode, ""),
                    "aliases": list(backend.aliases),
                    "native": backend.native,
                    "capabilities": sorted(backend.capabilities.layer_kinds),
                    "requires_asset": mode in VISUALIZER_ASSET_REQUIRED,
                    "requires_url": mode in VISUALIZER_URL_REQUIRED,
                    "accepts_external_url": mode in {"embodied", "rerun"},
                    "status": _visualizer_status(running) if running else None,
                }
            )
        return rows

    @app.post("/api/visualizers/{mode}/launch")
    def launch_visualizer(mode: str, payload: VisualizerLaunchRequest) -> dict[str, Any]:
        return _launch_visualizer(mode.strip().lower(), payload)

    @app.post("/api/visualizers/{mode}/stop")
    def stop_visualizer(mode: str) -> dict[str, Any]:
        mode = mode.strip().lower()
        ok = _stop_visualizer(mode)
        return {"ok": ok, "mode": mode}

    @app.get("/api/jobs")
    def list_jobs(job_type: str | None = None) -> list[dict[str, Any]]:
        jobs = JOBS.list()
        if job_type and job_type != "all":
            jobs = [job for job in jobs if job.job_type == job_type]
        return [_job_payload(job) for job in jobs]

    @app.post("/api/jobs")
    def create_job(payload: JobCreateRequest) -> dict[str, Any]:
        payload.job_type = (payload.job_type or "inference").strip().lower()
        if payload.job_type not in SUPPORTED_WORKSPACE_JOB_TYPES:
            raise HTTPException(status_code=400, detail=f"unsupported job type: {payload.job_type}")

        if payload.job_type == "inference":
            try:
                entry = find_entry(payload.model_id)
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=f"unknown model id: {payload.model_id}") from exc
            if entry.model_id not in _workspace_model_ids():
                raise HTTPException(
                    status_code=400,
                    detail=f"{entry.model_id} is not available in the Workspace inference catalog",
                )
            variant, task, _, _, _, contract = _resolve_inference_contract(entry, payload)
            _validate_inference_payload(entry, task, payload)

            def run_callable(job: StudioJob) -> Any:
                job.append_log(
                    "system",
                    f"model={entry.model_id} variant={variant.variant_id} task={task.task_id} type={payload.job_type}\n",
                )
                result = _run_inference(payload, job)
                job.append_log("system", "job finished\n")
                return result

            title = f"{entry.display_name} {variant.label} {payload.job_type}"
            metadata = {
                "job_type": payload.job_type,
                "workload_type": payload.workload_type,
                "variant_id": variant.variant_id,
                "task_profile_id": task.task_id,
                "infer_contract": contract,
                "prompt": payload.prompt,
                "input_path": payload.input_path,
                "device": payload.device or SETTINGS.get("device"),
            }
            model_id = entry.model_id
            display_name = entry.display_name
        elif payload.job_type == "evaluation":
            mode = (payload.eval_mode or "existing-results").strip().lower().replace("_", "-")
            supported_modes = {"existing-results", "model"} | _PREPARED_EVALUATION_MODES
            if mode not in supported_modes:
                raise HTTPException(status_code=400, detail=f"unsupported evaluation mode: {payload.eval_mode}")
            if mode in _PREPARED_EVALUATION_MODES:
                output_dir = _optional_path(payload.output_dir) or _workspace_job_output_dir("evaluations", None)
                prepared = _prepared_evaluation_from_payload(payload, output_dir)
                if not prepared.ready:
                    messages = "; ".join(issue.message for issue in prepared.issues if issue.severity == "error")
                    raise HTTPException(status_code=400, detail=messages or "evaluation preflight failed")
            elif not payload.run_plan_path:
                if mode == "existing-results" and not payload.results_path:
                    raise HTTPException(status_code=400, detail="existing-results evaluation requires results_path")
                if mode == "model" and not payload.requests_path:
                    raise HTTPException(status_code=400, detail="model evaluation requires requests_path")

            def run_callable(job: StudioJob) -> Any:
                job.append_log("system", f"evaluation mode={mode} model={payload.model_id or 'materialized-results'}\n")
                result = _run_evaluation(payload, job)
                job.append_log("system", "job finished\n")
                return result

            title = f"Evaluation {payload.benchmark_id or mode}"
            metadata = {
                "job_type": payload.job_type,
                "eval_mode": mode,
                "benchmark_id": payload.benchmark_id,
                "model_id": payload.model_id,
                "requests_path": payload.requests_path,
                "results_path": payload.results_path,
                "output_dir": payload.output_dir,
                "metrics": list(payload.metrics),
            }
            model_id = payload.model_id or "materialized-results"
            display_name = payload.model_id or "Evaluation"

        job = JOBS.submit_run(
            title=title,
            model_id=model_id,
            display_name=display_name,
            action=payload.job_type,
            job_type=payload.job_type,
            metadata=metadata,
            run_callable=run_callable,
        )
        return _job_payload(job, include_logs=True)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return _job_payload(job, include_logs=True)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        ok, message = JOBS.cancel(job_id)
        job = JOBS.get(job_id)
        return {"ok": ok, "message": message, "job": _job_payload(job, include_logs=True) if job else None}

    @app.get("/api/jobs/{job_id}/logs")
    def get_job_logs(job_id: str, after: int = 0) -> dict[str, Any]:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        logs = job.logs[max(after, 0) :]
        return {"offset": len(job.logs), "logs": logs, "text": "".join(str(row.get("text") or "") for row in logs)}

    @app.get("/api/jobs/{job_id}/video")
    def get_job_video(job_id: str, request: Request) -> Response:
        job = JOBS.get(job_id)
        if job is None or not isinstance(job.result, RunRecord):
            raise HTTPException(status_code=404, detail="video not found")
        return _safe_file_response(job.result.preview_video, request=request)

    @app.get("/api/jobs/{job_id}/image")
    def get_job_image(job_id: str, request: Request) -> Response:
        job = JOBS.get(job_id)
        if job is None or not isinstance(job.result, RunRecord):
            raise HTTPException(status_code=404, detail="image not found")
        return _safe_file_response(bind_run_preview_image(job.result), request=request)

    @app.get("/api/jobs/{job_id}/model")
    def get_job_model(job_id: str, request: Request) -> Response:
        job = JOBS.get(job_id)
        if job is None or not isinstance(job.result, RunRecord):
            raise HTTPException(status_code=404, detail="model not found")
        return _safe_file_response(job.result.preview_model, request=request)

    @app.get("/api/runs/{run_id}/video")
    def get_run_video(run_id: str, request: Request) -> Response:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _safe_file_response(record.preview_video, request=request)

    @app.get("/api/runs/{run_id}/image")
    def get_run_image(run_id: str, request: Request) -> Response:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _safe_file_response(bind_run_preview_image(record), request=request)

    @app.get("/api/runs/{run_id}/model")
    def get_run_model(run_id: str, request: Request) -> Response:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _safe_file_response(record.preview_model, request=request)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            record = MANAGER.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _run_payload(record)

    @app.get("/api/gallery")
    def gallery() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in JOBS.list():
            row = _gallery_row_from_job(job)
            if row is not None:
                rows.append(row)
        rows.extend(_gallery_row_from_run(record) for record in _recent_persisted_runs())
        return rows

    @app.delete("/api/gallery")
    def delete_gallery(job_id: str = "", run_id: str = "") -> dict[str, Any]:
        return _delete_gallery_item(job_id=job_id, run_id=run_id)

    @app.get("/api/artifacts/file")
    def artifact_file(path: str, request: Request) -> Response:
        target = Path(path).expanduser().resolve()
        if target not in _registered_artifact_paths():
            raise HTTPException(status_code=404, detail="artifact is not registered in this workspace session")
        return _safe_file_response(str(target), request=request)

    @app.get("/api/artifacts")
    def artifacts() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in JOBS.list():
            for name, path in _result_artifact_paths(job.result):
                visualization = _artifact_visualization_action(
                    name,
                    path,
                    model_id=job.model_id,
                    output_dir=_result_output_dir(job.result),
                )
                rows.append(
                    {
                        "job_id": job.job_id,
                        "model_name": job.display_name,
                        "model_id": job.model_id,
                        "job_type": job.job_type,
                        "name": name,
                        "path": path,
                        "output_dir": _result_output_dir(job.result),
                        "visualizer_mode": visualization["mode"] if visualization else "",
                        "visualizer_label": visualization["label"] if visualization else "",
                    }
                )
        for record in _recent_persisted_runs():
            for name, path in _result_artifact_paths(record):
                visualization = _artifact_visualization_action(
                    name,
                    path,
                    model_id=record.model_id,
                    output_dir=record.output_dir,
                )
                rows.append(
                    {
                        "job_id": "",
                        "run_id": record.run_id,
                        "model_name": record.display_name,
                        "model_id": record.model_id,
                        "job_type": "inference",
                        "name": name,
                        "path": path,
                        "output_dir": record.output_dir,
                        "visualizer_mode": visualization["mode"] if visualization else "",
                        "visualizer_label": visualization["label"] if visualization else "",
                    }
                )
        return rows

    return app


def main(argv: Sequence[str] | None = None) -> None:
    from worldfoundry.cli.help import WorldFoundryArgumentParser

    parser = WorldFoundryArgumentParser(
        prog="worldfoundry-workspace", description="Launch the WorldFoundry Studio workspace."
    )
    parser.add_argument("--host", default=os.getenv("WORLDFOUNDRY_WORKSPACE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WORLDFOUNDRY_WORKSPACE_PORT", "7870") or "7870"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    auth_token = require_auth_token_for_host(args.host, server_name="OpenEnvision Workspace")
    warning = bind_security_warning(args.host)
    if warning:
        print(warning, flush=True)

    import uvicorn

    uvicorn.run(create_app(auth_token=auth_token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

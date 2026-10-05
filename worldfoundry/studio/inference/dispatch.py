"""Dispatch Studio inference jobs into isolated Conda runtime environments.

WorldFoundry Studio typically runs in a base Python environment, while individual
models require dedicated Conda envs (different CUDA/PyTorch stacks). This module:

1. Resolves which Conda env a model should use (``workspace_runtime_spec``).
2. Decides whether to spawn a child process (``dispatch_spec_for_inference``).
3. Remaps GPU indices and torchrun settings for the child (``_run_kwargs_for_child``).
4. Launches ``worldfoundry.studio.runtime_job`` inside that env and streams logs
   back to the Studio UI (``run_manager_payload_in_conda``).

The child process is marked with ``WORLDFOUNDRY_STUDIO_CONDA_CHILD=1`` so nested
calls do not re-dispatch indefinitely.
"""

from __future__ import annotations

import atexit
import codecs
import hashlib
import json
import math
import os
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from worldfoundry.core.execution.process import terminate_process_group
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path, worldfoundry_path_tokens
from worldfoundry.evaluation.reporting import is_sensitive_key, redact_secret_text
from worldfoundry.runtime.conda import (
    RuntimeCondaEnvSpec,
    apply_unified_env_override,
    load_runtime_conda_env_spec,
    runtime_env_is_usable,
)
from worldfoundry.runtime.device_pool import (
    CudaDeviceLease,
    CudaDeviceLeasePool,
    discover_cuda_device_tokens,
)
from worldfoundry.runtime.env import (
    apply_hf_endpoint_override,
    getenv_registered,
    resolve_ckpt_dir,
    resolve_hf_cache_dir,
    resolve_hfd_root,
)
from worldfoundry.studio.inference.config import (
    MATRIX_GAME3_MODEL_ID,
    resolve_lingbot_fast_num_procs,
    wmfactory_interactive_model_spec,
)
from worldfoundry.studio.inference.execution import TORCHRUN_DISTRIBUTED_ENV, RunRecord

# Environment variable set in child processes to prevent recursive dispatch.
STUDIO_CONDA_CHILD_ENV = "WORLDFOUNDRY_STUDIO_CONDA_CHILD"
# Comma-separated model IDs (or "all"/"*") that always run in a subprocess.
FORCE_SUBPROCESS_MODELS_ENV = "WORLDFOUNDRY_STUDIO_FORCE_SUBPROCESS_MODELS"
RESIDENT_WORKERS_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKERS"
RESIDENT_WORKER_MODELS_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_MODELS"
RESIDENT_WORKER_MAX_WORKERS_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_MAX_WORKERS"
RESIDENT_WORKER_IDLE_TTL_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_IDLE_TTL"
RESIDENT_WORKER_REQUEST_TIMEOUT_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_REQUEST_TIMEOUT"
INFERENCE_TIMEOUT_ENV = "WORLDFOUNDRY_STUDIO_INFERENCE_TIMEOUT"
DEFAULT_RESIDENT_WORKER_REQUEST_TIMEOUT_SECONDS = 6 * 60 * 60
AUTO_GPU_PLACEMENT_ENV = "WORLDFOUNDRY_STUDIO_AUTO_GPU_PLACEMENT"
DISPATCH_API_KEY_ENV = "WORLDFOUNDRY_STUDIO_DISPATCH_API_KEY"
DISPATCH_SECRET_ENV_PREFIX = "WORLDFOUNDRY_STUDIO_DISPATCH_SECRET"
CHILD_PYTHONPATH_PREPEND_ENV = "WORLDFOUNDRY_STUDIO_CHILD_PYTHONPATH_PREPEND"
# Placeholder key in serialized payload; actual secret is passed via env.
SECRET_ENV_REF_KEY = "__worldfoundry_secret_env__"
DEFAULT_FORCE_SUBPROCESS_MODELS = frozenset(
    {
        "cameractrl",
        "hunyuan-game-craft",
        "hunyuan-gamecraft",
        "hunyuan-world-voyager",
        "lingbot-world",
        "lingbot-world-v2",
        "longvie-2",
        "hunyuan-worldplay",
        "wan-2p2",
        "matrix-game-2",
        "matrix-game-3",
        "helios",
        "dreamx-world-5b-cam",
        "wan2.1-vace",
    }
)
LINGBOT_WORLD_MODEL_ID = "lingbot-world"
LINGBOT_WORLD_V2_MODEL_ID = "lingbot-world-v2"
TORCHRUN_LINGBOT_ENV = "WORLDFOUNDRY_STUDIO_TORCHRUN_LINGBOT_FAST"
TORCHRUN_MODEL_NPROC_KEYS: Mapping[str, tuple[str, ...]] = {
    "dreamx-world-5b-cam": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "helios": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "longvie-2": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "lingbot-world-v2": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    MATRIX_GAME3_MODEL_ID: (
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "ulysses_size",
        "world_size",
    ),
    "hunyuan-game-craft": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "hunyuan-gamecraft": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "hunyuan-world-voyager": (
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "ulysses_degree",
    ),
    "hunyuan-worldplay": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "wan2.1-vace": (
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "ulysses_size",
        "world_size",
    ),
}
TORCHRUN_DISTRIBUTED_MODELS = frozenset(
    {
        "hunyuan-game-craft",
        "hunyuan-gamecraft",
        "hunyuan-worldplay",
        "longvie-2",
        MATRIX_GAME3_MODEL_ID,
        "dreamx-world-5b-cam",
        "wan2.1-vace",
    }
)
DISPATCH_ONLY_LOAD_KWARGS = frozenset(
    {
        "cuda_visible_devices",
        "visible_devices",
        "cuda_devices",
        "gpu_ids",
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "world_size",
    }
)
DISPATCH_ONLY_CALL_KWARGS = frozenset(
    {
        # In a dispatched conda child, nested official entrypoints must run with
        # the child env's interpreter instead of the parent Studio default.
        "python_executable",
    }
)

LogCallback = Callable[[str, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class _ResidentWorker:
    key: tuple[str, ...]
    base_key: tuple[str, ...]
    model_id: str
    process: subprocess.Popen[Any]
    lock: threading.RLock
    decoder: codecs.IncrementalDecoder
    command: list[str]
    created_at: float
    last_used_at: float
    in_use: int = 0
    device_lease: CudaDeviceLease | None = None
    shutdown_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    shutdown_complete: bool = False


class _ResidentWorkerUnavailable(RuntimeError):
    """Raised when resident-worker startup fails before a request can run."""


class _ResidentWorkerReady(Exception):
    """A matching resident appeared while its GPU was being allocated."""


@dataclass(frozen=True)
class _ResidentRunContext:
    child_run_kwargs: dict[str, Any]
    payload_run_kwargs: dict[str, Any]
    env: dict[str, str]
    key: tuple[str, ...]
    automatic_cuda_device_count: int = 0
    requested_cuda_tokens: tuple[str, ...] | None = None


_RESIDENT_WORKERS: dict[tuple[str, ...], _ResidentWorker] = {}
_RESIDENT_WORKERS_LOCK = threading.RLock()
_RESIDENT_WORKERS_LIFECYCLE_LOCK = threading.Lock()
_RESIDENT_WORKERS_REAPER_LOCK = threading.Lock()
_RESIDENT_WORKERS_REAPER_STOP = threading.Event()
_RESIDENT_WORKERS_REAPER_THREAD: threading.Thread | None = None
_AUTO_GPU_POOL_LOCK = threading.Lock()
_AUTO_GPU_POOL: CudaDeviceLeasePool | None = None
_AUTO_GPU_POOL_DEVICES: tuple[str, ...] = ()


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy string."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def in_conda_child_process() -> bool:
    """Return True when running inside a Studio-dispatched Conda child process."""
    return _env_flag(STUDIO_CONDA_CHILD_ENV)


def _automatic_gpu_placement_enabled() -> bool:
    """Return whether bare ``device=cuda`` jobs should receive exclusive devices."""

    value = os.getenv(AUTO_GPU_PLACEMENT_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off", "never"}


def _automatic_gpu_pool() -> CudaDeviceLeasePool:
    """Reserve GPUs across controllers on the same host using inherited file locks."""

    global _AUTO_GPU_POOL, _AUTO_GPU_POOL_DEVICES
    devices = discover_cuda_device_tokens()
    if not devices:
        raise RuntimeError("device=cuda was requested, but no visible CUDA devices were discovered")
    with _AUTO_GPU_POOL_LOCK:
        if _AUTO_GPU_POOL is None or _AUTO_GPU_POOL_DEVICES != devices:
            lock_keys: dict[str, str] = {}
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=False,
                )
                if result.returncode == 0:
                    for row in result.stdout.splitlines():
                        index, uuid = (item.strip() for item in row.split(",", 1))
                        lock_keys[index] = uuid
                        lock_keys[uuid] = uuid
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            for token in devices:
                if token.isdecimal() and str(int(token)) in lock_keys:
                    lock_keys[token] = lock_keys[str(int(token))]
                elif token.startswith("GPU-") and token not in lock_keys:
                    matches = {value for value in lock_keys.values() if value.startswith(token)}
                    if len(matches) == 1:
                        lock_keys[token] = matches.pop()
            lock_dir = Path(os.getenv("WORLDFOUNDRY_GPU_LEASE_DIR") or
                            str(Path(tempfile.gettempdir()) / f"worldfoundry-gpu-leases-{getattr(os, 'getuid', lambda: 0)()}"))
            _AUTO_GPU_POOL = CudaDeviceLeasePool(devices, lock_dir=lock_dir, lock_keys=lock_keys)
            _AUTO_GPU_POOL_DEVICES = devices
        return _AUTO_GPU_POOL


def _force_subprocess_for_model(model_id: str) -> bool:
    """Return True when the model must run in a subprocess even if Python paths match."""
    raw = os.getenv(FORCE_SUBPROCESS_MODELS_ENV)
    if raw is None:
        candidates = DEFAULT_FORCE_SUBPROCESS_MODELS
    else:
        candidates = frozenset(item.strip() for item in raw.split(",") if item.strip())
    return "*" in candidates or "all" in candidates or model_id in candidates


def _resident_workers_enabled_for_model(model_id: str) -> bool:
    """Return True when eligible conda jobs may reuse a long-lived worker."""
    value = os.getenv(RESIDENT_WORKERS_ENV, "auto").strip().lower()
    if value in {"0", "false", "no", "off", "never"}:
        return False
    raw_models = os.getenv(RESIDENT_WORKER_MODELS_ENV, "").strip()
    if not raw_models:
        return True
    candidates = {item.strip() for item in raw_models.split(",") if item.strip()}
    if not candidates:
        return True
    return "*" in candidates or "all" in candidates or model_id in candidates


def _resident_worker_request_timeout() -> float:
    try:
        default = str(DEFAULT_RESIDENT_WORKER_REQUEST_TIMEOUT_SECONDS)
        value = float(os.getenv(RESIDENT_WORKER_REQUEST_TIMEOUT_ENV, default) or default)
        return max(value, 0.0) if math.isfinite(value) else float(default)
    except ValueError:
        return float(DEFAULT_RESIDENT_WORKER_REQUEST_TIMEOUT_SECONDS)


def _inference_request_timeout() -> float:
    """One wall-clock budget for allocation, dispatch, execution and output drain."""
    default = _resident_worker_request_timeout()
    try:
        value = float(os.getenv(INFERENCE_TIMEOUT_ENV, str(default)))
        return max(value, 0.0) if math.isfinite(value) else default
    except ValueError:
        return default


def _check_dispatch_interrupt(cancel_requested: CancelCallback | None, deadline: float | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("inference dispatch cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("inference request timed out during allocation or execution")


@contextmanager
def _resident_lifecycle_lock(cancel_requested: CancelCallback | None, deadline: float | None):
    while True:
        _check_dispatch_interrupt(cancel_requested, deadline)
        if _RESIDENT_WORKERS_LIFECYCLE_LOCK.acquire(timeout=0.05):
            break
    try:
        _check_dispatch_interrupt(cancel_requested, deadline)
        yield
    finally:
        _RESIDENT_WORKERS_LIFECYCLE_LOCK.release()


def _resident_worker_max_workers() -> int:
    # Keep the resident pool aligned with the Workspace scheduler unless an
    # operator explicitly chooses a different cap.  A hard-coded default of 2
    # made jobs 3..N fall back to one-shot processes even when Workspace was
    # launched with (for example) ``--max-jobs 8``.  Large checkpoints then
    # paid their multi-minute cold-load cost again and could never reuse the
    # pipeline that had just been loaded.
    raw_value = os.getenv(RESIDENT_WORKER_MAX_WORKERS_ENV)
    if raw_value is None:
        raw_value = os.getenv("WORLDFOUNDRY_WORKSPACE_MAX_JOBS", "2")
    try:
        return max(int(raw_value or "2"), 0)
    except ValueError:
        return 2


def _resident_worker_idle_ttl() -> float:
    try:
        return max(float(os.getenv(RESIDENT_WORKER_IDLE_TTL_ENV, "900") or "900"), 0.0)
    except ValueError:
        return 900.0


def _same_executable(left: str | Path, right: str | Path) -> bool:
    """Compare two Python interpreter paths after resolving symlinks."""
    try:
        return Path(left).expanduser().absolute() == Path(right).expanduser().absolute()
    except OSError:
        return str(left) == str(right)


def _dedupe_paths(values: list[Path]) -> tuple[Path, ...]:
    rows: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path not in rows:
            rows.append(path)
    return tuple(rows)


def _dedupe_text(values: tuple[str, ...]) -> tuple[str, ...]:
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in rows:
            rows.append(text)
    return tuple(rows)


def _candidate_env_roots(configured: Path) -> tuple[Path, ...]:
    """Collect likely Conda env root directories to search for a model runtime."""
    tokens = worldfoundry_path_tokens()
    repo = _repo_root()
    conda_root = Path(tokens["WORLDFOUNDRY_CONDA_ROOT"]).expanduser()
    conda_envs_root = Path(tokens["WORLDFOUNDRY_CONDA_ENVS_ROOT"]).expanduser()
    candidates = [
        configured,
        conda_envs_root,
        conda_root / "envs",
        conda_root,
        Path(tokens["WORLDFOUNDRY_HOME"]).expanduser() / "conda_envs",
        repo.parent / "conda" / "envs",
        repo.parent / "conda",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser()
        candidates.extend([prefix.parent, prefix.parent.parent / "envs"])
    if configured.name == "conda":
        candidates.append(configured / "envs")
    elif configured.name == "envs":
        candidates.append(configured.parent)
    return _dedupe_paths(candidates)


def _candidate_env_names(model_id: str, env_name: str) -> tuple[str, ...]:
    """Generate common Conda env name variants for a model (hyphen/underscore aliases)."""
    return _dedupe_text((env_name, model_id, model_id.replace("-", "_"), model_id.replace("_", "-")))


def _discover_existing_env_spec(spec: RuntimeCondaEnvSpec) -> RuntimeCondaEnvSpec | None:
    """Locate an on-disk Conda env when the configured path does not exist yet."""
    if spec.exists:
        return spec
    for root in _candidate_env_roots(spec.env_root):
        for name in _candidate_env_names(spec.model_id, spec.env_name):
            candidate = root / name
            if runtime_env_is_usable(candidate):
                return replace(
                    spec,
                    env_name=name,
                    env_root=root,
                    notes=spec.notes + (f"studio_env_discovery:{candidate}",),
                )
    return None


def _current_explicit_unified_env_spec(
    spec: RuntimeCondaEnvSpec,
) -> RuntimeCondaEnvSpec | None:
    """Use the active unified interpreter when the user selected it explicitly.

    Static dependency policy intentionally keeps profiles with upper version
    bounds out of an automatically selected shared environment.  An explicit
    ``WORLDFOUNDRY_USE_UNIFIED_ENV=1`` request is different: if Codex/Studio is
    already running from a real ``worldfoundry-unified-*`` prefix, that exact
    interpreter must remain discoverable so force-subprocess and torchrun
    models do not silently fall back to same-process, single-GPU execution.
    Model runtimes still enforce their package-version contracts on load.
    """

    enabled = os.getenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    env_prefix = Path(sys.executable).expanduser().absolute().parent.parent
    if not env_prefix.name.startswith("worldfoundry-unified-"):
        return None
    if not runtime_env_is_usable(env_prefix):
        return None
    return replace(
        spec,
        env_name=env_prefix.name,
        env_root=env_prefix.parent,
        notes=spec.notes + (f"studio_explicit_active_unified_env:{env_prefix}",),
    )


@lru_cache(maxsize=None)
def workspace_runtime_spec(model_id: str) -> RuntimeCondaEnvSpec | None:
    """Resolve the Conda runtime spec for a model, with discovery and unified-env fallback."""
    spec = load_runtime_conda_env_spec(model_id)
    if spec is not None:
        discovered = _discover_existing_env_spec(spec)
        if discovered is not None:
            return discovered

    fallback = RuntimeCondaEnvSpec(model_id=model_id, env_name=model_id)
    unified_fallback = apply_unified_env_override(fallback)
    if unified_fallback != fallback:
        discovered = _discover_existing_env_spec(unified_fallback)
        if discovered is not None:
            return discovered
    current_unified = _current_explicit_unified_env_spec(spec or fallback)
    if current_unified is not None:
        return current_unified
    return _discover_existing_env_spec(fallback)


@dataclass(frozen=True)
class _LocalRuntimeSpec(RuntimeCondaEnvSpec):
    @property
    def python_executable(self) -> Path:
        return Path(sys.executable)


def dispatch_spec_for_inference(
    model_id: str, *, backend: str = "auto", force_subprocess: bool = False,
) -> RuntimeCondaEnvSpec | None:
    """Return a Conda spec when inference should run in a separate env, else None.

    Dispatch is skipped when already in a child process, when using ``api_init``,
    when no runtime env exists, or when the current interpreter already matches
    the target env (unless the model is force-subprocess listed).
    """
    if in_conda_child_process():
        return None
    if str(backend or "auto").strip() == "api_init":
        return None
    spec = workspace_runtime_spec(model_id)
    if spec is None:
        if not force_subprocess:
            return None
        prefix = Path(sys.prefix)
        return _LocalRuntimeSpec(model_id=model_id, env_name=prefix.name, env_root=prefix.parent)
    if (not force_subprocess and _same_executable(sys.executable, spec.python_executable)
            and not _force_subprocess_for_model(model_id)):
        return None
    return spec


def _visible_device_items(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _cuda_device_index(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text.startswith("cuda:"):
        return None
    index = text.split(":", maxsplit=1)[1].strip()
    return index if index.isdigit() else None


def _model_cuda_device_indices(run_kwargs: Mapping[str, Any]) -> list[str]:
    indices: list[str] = []
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    for source in (load_kwargs, call_kwargs):
        for key, value in source.items():
            if key != "device" and not str(key).endswith("_device"):
                continue
            index = _cuda_device_index(value)
            if index is not None and index not in indices:
                indices.append(index)
    return indices


def _expand_visible_devices_for_model_devices(
    visible_devices: str,
    run_kwargs: Mapping[str, Any],
) -> str:
    visible = _visible_device_items(visible_devices)
    if not visible:
        return visible_devices
    model_indices = _model_cuda_device_indices(run_kwargs)
    if not model_indices or not any(index in visible for index in model_indices):
        return visible_devices
    expanded = list(visible)
    for index in model_indices:
        if index not in expanded:
            expanded.append(index)
    return ",".join(expanded)


def _rewrite_cuda_devices_for_child(
    values: Mapping[str, Any],
    visible_devices: str,
) -> dict[str, Any]:
    """Remap ``cuda:N`` device strings to child-local indices under CUDA_VISIBLE_DEVICES."""
    visible = _visible_device_items(visible_devices)
    if not visible:
        return dict(values)
    rewritten = dict(values)
    for key, value in values.items():
        if key != "device" and not str(key).endswith("_device"):
            continue
        index = _cuda_device_index(value)
        if index is not None and index in visible:
            rewritten[key] = f"cuda:{visible.index(index)}"
    return rewritten


def _run_kwargs_for_child(
    run_kwargs: Mapping[str, Any],
    *,
    cuda_visible_devices: str = "",
) -> dict[str, Any]:
    """Prepare run kwargs for a dispatched child: remap devices and strip dispatch-only keys."""
    child_kwargs = dict(run_kwargs)
    device = str(child_kwargs.get("device") or "")
    if device.startswith("cuda:"):
        # _runtime_env pins CUDA_VISIBLE_DEVICES to the selected physical GPU.
        # Inside that child process the selected GPU is exposed as logical cuda:0.
        child_kwargs["device"] = "cuda"
    load_kwargs = _json_object_from_text(child_kwargs.get("load_kwargs_text"))
    if load_kwargs:
        original_load_kwargs = dict(load_kwargs)
        load_kwargs = _rewrite_cuda_devices_for_child(load_kwargs, cuda_visible_devices)
        cleaned = {key: value for key, value in load_kwargs.items() if key not in DISPATCH_ONLY_LOAD_KWARGS}
        if cleaned != original_load_kwargs:
            child_kwargs["load_kwargs_text"] = json.dumps(cleaned)
    call_kwargs = _json_object_from_text(child_kwargs.get("call_kwargs_text"))
    if call_kwargs:
        original_call_kwargs = dict(call_kwargs)
        call_kwargs = _rewrite_cuda_devices_for_child(call_kwargs, cuda_visible_devices)
        cleaned = {key: value for key, value in call_kwargs.items() if key not in DISPATCH_ONLY_CALL_KWARGS}
        if cleaned != original_call_kwargs:
            child_kwargs["call_kwargs_text"] = json.dumps(cleaned)
    return child_kwargs


def _with_matrix_game3_parallelism(
    model_id: str,
    run_kwargs: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Keep Matrix-Game 3 Ulysses configuration aligned with torchrun WORLD_SIZE."""

    updated = dict(run_kwargs)
    if model_id != MATRIX_GAME3_MODEL_ID:
        return updated
    normalized_world_size = _validate_model_torchrun_nproc(
        model_id,
        max(int(world_size or 1), 1),
    )
    load_kwargs = _json_object_from_text(updated.get("load_kwargs_text"))
    load_kwargs["ulysses_size"] = normalized_world_size
    # WORLD_SIZE is owned by torchrun. Do not forward a second, potentially
    # conflicting loader argument when a user supplied `world_size` as input.
    load_kwargs.pop("world_size", None)
    updated["load_kwargs_text"] = json.dumps(load_kwargs)
    call_kwargs = _json_object_from_text(updated.get("call_kwargs_text"))
    if call_kwargs:
        if "ulysses_size" in call_kwargs:
            call_kwargs["ulysses_size"] = normalized_world_size
        for key in (
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "nproc_per_node",
            "world_size",
        ):
            call_kwargs.pop(key, None)
        updated["call_kwargs_text"] = json.dumps(call_kwargs)
    return updated


def _with_lingbot_world_parallelism(
    model_id: str,
    run_kwargs: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Keep LingBot runtime options identical to the outer torchrun group.

    V1 consumes the selected topology while loading the model via
    ``ulysses_size``.  V2 consumes it at prediction time via
    ``nproc_per_node``.  In particular, leaving V2's catalog default of eight
    in the child request makes an otherwise valid four-rank launch fail before
    checkpoint loading.
    """

    updated = dict(run_kwargs)
    if model_id not in {LINGBOT_WORLD_MODEL_ID, LINGBOT_WORLD_V2_MODEL_ID}:
        return updated
    normalized_world_size = _validate_model_torchrun_nproc(
        model_id,
        max(int(world_size or 1), 1),
    )
    if model_id == LINGBOT_WORLD_V2_MODEL_ID:
        call_kwargs = _json_object_from_text(updated.get("call_kwargs_text"))
        call_kwargs["nproc_per_node"] = normalized_world_size
        for key in ("torchrun_nproc_per_node", "torchrun_nproc", "world_size"):
            call_kwargs.pop(key, None)
        updated["call_kwargs_text"] = json.dumps(call_kwargs)
        return updated

    load_kwargs = _json_object_from_text(updated.get("load_kwargs_text"))
    # The official eight-rank recipe shards both T5 and the two DiTs.  On the
    # four-rank Workspace topology, checkpoint-backed validation showed that
    # keeping T5 sharded while replicating/offloading the DiTs is the stable
    # full-quality configuration: it leaves enough room for FP32 VAE
    # conditioning while preserving Ulysses sequence parallelism.
    # Restrict the rewrite to the exact catalog-default topology so explicit
    # user choices remain untouched.
    catalog_default_fsdp = (
        normalized_world_size == 4
        and _truthy_value(load_kwargs.get("t5_fsdp"))
        and _truthy_value(load_kwargs.get("dit_fsdp"))
        and not _truthy_value(load_kwargs.get("t5_cpu"))
    )
    if catalog_default_fsdp:
        load_kwargs["dit_fsdp"] = False
        call_kwargs = _json_object_from_text(updated.get("call_kwargs_text"))
        call_kwargs["offload_model"] = True
        updated["call_kwargs_text"] = json.dumps(call_kwargs)
    load_kwargs["ulysses_size"] = normalized_world_size
    for key in (
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "world_size",
    ):
        load_kwargs.pop(key, None)
    updated["load_kwargs_text"] = json.dumps(load_kwargs)
    return updated


def _with_longvie2_parallelism(
    model_id: str,
    run_kwargs: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Keep LongVie 2 USP settings aligned with its supported process count."""

    updated = dict(run_kwargs)
    if model_id != "longvie-2":
        return updated
    normalized_world_size = _validate_model_torchrun_nproc(
        model_id,
        max(int(world_size or 1), 1),
    )
    load_kwargs = _json_object_from_text(updated.get("load_kwargs_text"))
    load_kwargs.update(
        use_usp=normalized_world_size == 4,
        ring_degree=1,
        ulysses_degree=normalized_world_size,
    )
    for key in (
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "world_size",
    ):
        load_kwargs.pop(key, None)
    updated["load_kwargs_text"] = json.dumps(load_kwargs)
    return updated


def _with_wan_vace_parallelism(
    model_id: str,
    run_kwargs: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Keep native VACE Ulysses settings aligned with its outer torchrun job.

    VACE runs through the native diffusion pipeline, so Workspace must launch
    every rank and the loader must explicitly install sequence parallelism.
    The in-tree native implementation currently supports pure Ulysses only
    (ring size one), and the 14B checkpoint has 40 attention heads.
    """

    updated = dict(run_kwargs)
    if model_id != "wan2.1-vace":
        return updated
    normalized_world_size = int(world_size or 0)
    if normalized_world_size <= 1:
        raise ValueError(
            "wan2.1-vace requires nproc_per_node > 1 because its native 14B "
            "pipeline is configured for distributed sequence parallelism."
        )
    call_kwargs = _json_object_from_text(updated.get("call_kwargs_text"))
    load_kwargs = _json_object_from_text(updated.get("load_kwargs_text"))
    raw_ring_size = call_kwargs.get("ring_size", load_kwargs.get("ring_size", 1))
    try:
        ring_size = int(raw_ring_size or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"wan2.1-vace ring_size must be an integer, got {raw_ring_size!r}."
        ) from exc
    if ring_size != 1:
        raise ValueError(
            "wan2.1-vace native sequence parallelism currently requires "
            f"ring_size=1; got ring_size={ring_size}."
        )
    ulysses_size = normalized_world_size
    if 40 % ulysses_size:
        raise ValueError(
            "wan2.1-vace uses 40 attention heads, which must be divisible by "
            f"ulysses_size={ulysses_size} (nproc={normalized_world_size}, ring={ring_size})."
        )
    call_kwargs.update(
        nproc_per_node=normalized_world_size,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
    )
    load_kwargs["sequence_parallel"] = ulysses_size
    load_kwargs.pop("sp_degree", None)
    updated["load_kwargs_text"] = json.dumps(load_kwargs)
    updated["call_kwargs_text"] = json.dumps(call_kwargs)
    return updated


def _json_object_from_text(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _truthy_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _official_video_internal_torchrun_nproc(
    model_id: str,
    *,
    call_kwargs: Mapping[str, Any],
    load_kwargs: Mapping[str, Any],
) -> int:
    """Return nproc for official video configs whose command launches torchrun."""
    config_path = _package_root() / "data" / "models" / "runtime" / "configs" / "video_official" / f"{model_id}.yaml"
    if not config_path.is_file():
        return 0
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    if "{torchrun}" not in text:
        return 0
    requested = (
        call_kwargs.get("nproc_per_node")
        or call_kwargs.get("torchrun_nproc_per_node")
        or call_kwargs.get("torchrun_nproc")
        or load_kwargs.get("nproc_per_node")
        or load_kwargs.get("torchrun_nproc_per_node")
        or load_kwargs.get("torchrun_nproc")
    )
    if requested in {None, ""}:
        match = re.search(r"(?m)^\s+nproc_per_node:\s*([0-9]+)\s*$", text)
        requested = match.group(1) if match else 0
    try:
        return int(requested or 0)
    except Exception:
        return 0


def _lingbot_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Resolve torchrun process count for LingBot-World (FSDP / ulysses fast path)."""
    if model_id != LINGBOT_WORLD_MODEL_ID:
        return 0
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    requested = (
        load_kwargs.get("nproc_per_node")
        or load_kwargs.get("torchrun_nproc_per_node")
        or load_kwargs.get("torchrun_nproc")
        or load_kwargs.get("ulysses_size")
    )
    try:
        nproc = int(requested or 0)
    except Exception:
        nproc = 0
    if nproc > 1:
        return _validate_model_torchrun_nproc(model_id, nproc)
    if _truthy_value(load_kwargs.get("dit_fsdp")) or _truthy_value(load_kwargs.get("t5_fsdp")):
        visible_count = _cuda_visible_device_count(os.getenv("CUDA_VISIBLE_DEVICES"))
        resolved = resolve_lingbot_fast_num_procs(visible_count=visible_count or None)
        if resolved < 4:
            raise RuntimeError(
                "LingBot-World base-cam defaults to native multi-GPU FSDP/Ulysses and requires "
                f"at least four visible GPUs; detected {resolved}. Explicitly disable FSDP and use "
                "ulysses_size=1 only for a deliberate single-GPU fallback."
            )
        return _validate_model_torchrun_nproc(
            model_id,
            resolved,
        )
    return 0


def _explicit_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Return explicit multi-GPU torchrun count from load/call kwargs, or 0."""
    requested = _requested_torchrun_nproc(model_id, run_kwargs)
    return requested if requested is not None and requested > 1 else 0


def _requested_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int | None:
    for env_name in (
        "WORLDFOUNDRY_REALTIME_NPROC_PER_NODE",
        "WORLDFOUNDRY_REALTIME_NPROC",
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            requested = int(raw)
        except ValueError as exc:
            if model_id in {MATRIX_GAME3_MODEL_ID, "dreamx-world-5b-cam"}:
                raise ValueError(f"{env_name} must be an integer, got {raw!r}.") from exc
            return None
        if requested <= 0 and model_id != MATRIX_GAME3_MODEL_ID:
            return None
        return _validate_model_torchrun_nproc(model_id, requested)
    keys = TORCHRUN_MODEL_NPROC_KEYS.get(model_id)
    if not keys:
        return None
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    for source in (load_kwargs, call_kwargs):
        for key in keys:
            if key not in source or source.get(key) in {None, ""}:
                continue
            try:
                requested = int(source.get(key) or 0)
            except Exception as exc:
                if model_id in {MATRIX_GAME3_MODEL_ID, "dreamx-world-5b-cam"}:
                    raise ValueError(
                        f"{model_id} {key} must be an integer, got {source.get(key)!r}."
                    ) from exc
                return None
            return _validate_model_torchrun_nproc(model_id, requested)
    return None


def _validate_model_torchrun_nproc(model_id: str, requested: int) -> int:
    """Validate model-specific distributed process-count constraints."""

    if model_id in {LINGBOT_WORLD_MODEL_ID, LINGBOT_WORLD_V2_MODEL_ID}:
        supported = (1, 2, 4, 8)
        if requested not in supported:
            choices = ", ".join(str(value) for value in supported)
            raise ValueError(
                f"{model_id} Workspace integration supports nproc values {choices}; got {requested}. "
                "The 8-GPU topology is the official recipe; 2 and 4 GPUs are compact inference topologies."
            )
        return requested
    if model_id == "longvie-2":
        if requested not in {1, 4}:
            raise ValueError(
                "longvie-2 supports nproc values 1 or 4 (official USP topology); "
                f"got {requested}."
            )
        return requested
    if model_id == "dreamx-world-5b-cam":
        supported = (1, 2, 3, 4, 6, 8)
        if requested not in supported:
            choices = ", ".join(str(value) for value in supported)
            raise ValueError(
                f"dreamx-world-5b-cam supports nproc values {choices}; got {requested}."
            )
        return requested
    if model_id in {"hunyuan-game-craft", "hunyuan-gamecraft"}:
        supported = (1, 2, 3, 4, 6, 8)
        if requested not in supported:
            choices = ", ".join(str(value) for value in supported)
            raise ValueError(
                f"{model_id} supports nproc values {choices}; got {requested}. "
                "Its Ulysses all-to-all path requires the process count to divide 24 attention heads."
            )
        return requested
    if model_id == "hunyuan-worldplay":
        supported = (1, 2, 3, 4, 6, 8)
        if requested not in supported:
            choices = ", ".join(str(value) for value in supported)
            raise ValueError(
                f"hunyuan-worldplay supports nproc values {choices}; got {requested}. "
                "Its Ulysses all-to-all path requires the process count to divide 24 attention heads."
            )
        return requested
    if model_id != MATRIX_GAME3_MODEL_ID:
        return requested
    spec = wmfactory_interactive_model_spec(model_id)
    supported = tuple(spec.supported_process_counts) if spec is not None else (1, 2, 4, 8)
    if requested not in supported:
        choices = ", ".join(str(value) for value in supported)
        raise ValueError(
            f"{MATRIX_GAME3_MODEL_ID} supports nproc/ulysses_size values {choices}; got {requested}."
        )
    return requested


def _matrix_game3_default_torchrun_nproc(run_kwargs: Mapping[str, Any]) -> int:
    """Choose the largest supported Matrix-Game 3 size up to its four-GPU default."""

    if str(run_kwargs.get("device") or "").startswith("cuda:"):
        return 1
    wmfactory_override = (
        getenv_registered("WORLDFOUNDRY_MATRIXGAME3_CUDA_VISIBLE_DEVICES", "") or ""
    ).strip()
    explicit_visible = _cuda_visible_devices_from_kwargs(run_kwargs) or wmfactory_override
    visible_count = _cuda_visible_device_count(
        explicit_visible or os.getenv("CUDA_VISIBLE_DEVICES")
    )
    if not visible_count:
        try:
            visible_count = len(_parse_nvidia_smi_rows())
        except (OSError, subprocess.SubprocessError, ValueError):
            visible_count = 0
    spec = wmfactory_interactive_model_spec(MATRIX_GAME3_MODEL_ID)
    preferred = int(spec.preferred_visible_devices or 4) if spec is not None else 4
    supported = tuple(spec.supported_process_counts) if spec is not None else (1, 2, 4, 8)
    upper_bound = min(preferred, visible_count) if visible_count else preferred
    eligible = [value for value in supported if value <= upper_bound]
    return max(eligible) if eligible else 1


def _default_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Apply model-specific default torchrun sizing when the user did not specify nproc."""
    if _requested_torchrun_nproc(model_id, run_kwargs) is not None:
        return 0
    if model_id in {"hunyuan-game-craft", "hunyuan-gamecraft"}:
        if str(run_kwargs.get("device") or "").startswith("cuda:"):
            return 1
        requested = int(os.getenv("WORLDFOUNDRY_STUDIO_GAMECRAFT_TORCHRUN_NPROC", "8") or "8")
        requested = _validate_model_torchrun_nproc(model_id, requested)
        visible_count = _cuda_visible_device_count(os.getenv("CUDA_VISIBLE_DEVICES"))
        if not visible_count:
            try:
                visible_count = len(_parse_nvidia_smi_rows())
            except (OSError, subprocess.SubprocessError, ValueError):
                visible_count = 0
        upper_bound = min(requested, visible_count) if visible_count else requested
        eligible = [value for value in (1, 2, 3, 4, 6, 8) if value <= upper_bound]
        return max(eligible) if eligible else 1
    if model_id == MATRIX_GAME3_MODEL_ID:
        return _matrix_game3_default_torchrun_nproc(run_kwargs)
    if model_id == "dreamx-world-5b-cam":
        if str(run_kwargs.get("device") or "").startswith("cuda:"):
            return 1
        explicit_visible = _cuda_visible_devices_from_kwargs(run_kwargs)
        visible_count = _cuda_visible_device_count(
            explicit_visible or os.getenv("CUDA_VISIBLE_DEVICES")
        )
        if not visible_count:
            try:
                visible_count = len(_parse_nvidia_smi_rows())
            except (OSError, subprocess.SubprocessError, ValueError):
                visible_count = 0
        upper_bound = min(8, visible_count) if visible_count else 8
        eligible = [value for value in (1, 2, 3, 4, 6, 8) if value <= upper_bound]
        return max(eligible) if eligible else 1
    if model_id == "longvie-2":
        if str(run_kwargs.get("device") or "").startswith("cuda:"):
            return 1
        visible_count = _cuda_visible_device_count(os.getenv("CUDA_VISIBLE_DEVICES"))
        if not visible_count:
            try:
                visible_count = len(_parse_nvidia_smi_rows())
            except (OSError, subprocess.SubprocessError, ValueError):
                visible_count = 0
        return min(4, visible_count) if visible_count else 4
    if model_id != "hunyuan-worldplay":
        return 0
    if str(run_kwargs.get("device") or "").startswith("cuda:"):
        return 1
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    wm_spec = wmfactory_interactive_model_spec(model_id, load_kwargs=load_kwargs)
    preferred = int(wm_spec.preferred_visible_devices or 8) if wm_spec is not None else 8
    supported = tuple(wm_spec.supported_process_counts) if wm_spec is not None else (1, 2, 3, 4, 6, 8)
    explicit_visible = _cuda_visible_devices_from_kwargs(run_kwargs)
    visible_count = _cuda_visible_device_count(
        explicit_visible or os.getenv("CUDA_VISIBLE_DEVICES")
    )
    if not visible_count:
        try:
            visible_count = len(_parse_nvidia_smi_rows())
        except (OSError, subprocess.SubprocessError, ValueError):
            visible_count = 0
    upper_bound = min(preferred, visible_count) if visible_count else preferred
    eligible = [value for value in supported if value <= upper_bound]
    return max(eligible) if eligible else 1


def _internal_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Return torchrun nproc for models that launch distributed jobs inside their own code."""
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    if model_id == "bernini":
        requested = (
            call_kwargs.get("nproc_per_node")
            or call_kwargs.get("ulysses_size")
            or load_kwargs.get("nproc_per_node")
            or load_kwargs.get("ulysses_size")
            or 1
        )
    elif model_id == "kairos-sensenova":
        requested = (
            call_kwargs.get("nproc_per_node")
            or call_kwargs.get("torchrun_nproc_per_node")
            or call_kwargs.get("torchrun_nproc")
            or load_kwargs.get("nproc_per_node")
            or load_kwargs.get("torchrun_nproc_per_node")
            or load_kwargs.get("torchrun_nproc")
            or 1
        )
    else:
        nproc = _official_video_internal_torchrun_nproc(
            model_id,
            call_kwargs=call_kwargs,
            load_kwargs=load_kwargs,
        )
        return nproc
    try:
        nproc = int(requested or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_id} nproc must be an integer, got {requested!r}.") from exc
    if model_id == "bernini":
        if nproc < 1:
            raise ValueError(f"bernini nproc_per_node must be positive; got {nproc}.")
        ulysses_size = int(call_kwargs.get("ulysses_size") or nproc)
        if ulysses_size != nproc:
            raise ValueError(
                "Bernini uses one Ulysses group per request, so ulysses_size must equal "
                f"nproc_per_node; got ulysses_size={ulysses_size}, nproc_per_node={nproc}."
            )
    if model_id == "kairos-sensenova":
        if nproc < 1 or nproc > 20:
            raise ValueError(
                "kairos-sensenova nproc_per_node must be between 1 and its 20 attention heads; "
                f"got {nproc}."
            )
        height = int(call_kwargs.get("height", 480) or 480)
        width = int(call_kwargs.get("width", 640) or 640)
        num_frames = int(call_kwargs.get("num_frames", 81) or 81)
        if height % 16 or width % 16 or (num_frames - 1) % 4:
            raise ValueError(
                "kairos-sensenova requires height/width divisible by 16 and num_frames=4n+1 "
                f"for the vendored VAE/DiT patching; got {height}x{width}, frames={num_frames}."
            )
        sequence_tokens = ((num_frames - 1) // 4 + 1) * (height // 16) * (width // 16)
        if sequence_tokens % nproc:
            raise ValueError(
                "kairos-sensenova sequence tokens must divide evenly across TP/SP ranks; "
                f"got tokens={sequence_tokens}, nproc={nproc}."
            )
    return nproc


def _cuda_visible_device_count(value: str | None) -> int:
    if not value:
        return 0
    return len([item for item in value.split(",") if item.strip()])


def _automatic_cuda_device_count(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Return the exclusive GPU count for an unpinned Workspace CUDA request."""

    if in_conda_child_process() or not _automatic_gpu_placement_enabled():
        return 0
    if str(run_kwargs.get("device") or "").strip().lower() != "cuda":
        return 0
    if _cuda_visible_devices_from_kwargs(run_kwargs):
        return 0
    requested = max(
        _lingbot_torchrun_nproc(model_id, run_kwargs),
        _explicit_torchrun_nproc(model_id, run_kwargs),
        _default_torchrun_nproc(model_id, run_kwargs),
        _internal_torchrun_nproc(model_id, run_kwargs),
        1,
    )
    return requested


def _cuda_lease_request(
    model_id: str, run_kwargs: Mapping[str, Any],
) -> tuple[int, tuple[str, ...] | None]:
    """Apply admission to explicitly pinned GPUs as well as automatic requests."""
    if in_conda_child_process():
        return 0, None
    visible = (_cuda_visible_devices_from_kwargs(run_kwargs) or _cuda_visible_devices_from_device(run_kwargs)
               or ",".join(_model_cuda_device_indices(run_kwargs)))
    if (not visible and str(run_kwargs.get("device") or "").strip().lower() == "cuda"
            and not _automatic_gpu_placement_enabled()):
        # Disabling automatic placement preserves visibility, not an admission
        # bypass. Reserve the whole visible set before launching an unpinned job.
        visible = ",".join(discover_cuda_device_tokens())
        if not visible:
            raise RuntimeError("device=cuda was requested, but no visible CUDA devices were discovered")
    if visible:
        visible = _expand_visible_devices_for_model_devices(visible, run_kwargs)
        tokens = tuple(_visible_device_items(visible))
        requested = max(_lingbot_torchrun_nproc(model_id, run_kwargs),
                        _explicit_torchrun_nproc(model_id, run_kwargs),
                        _default_torchrun_nproc(model_id, run_kwargs),
                        _internal_torchrun_nproc(model_id, run_kwargs), 1)
        if len(tokens) < requested:
            raise ValueError(f"{model_id} needs {requested} CUDA devices but only {len(tokens)} were pinned")
        return len(tokens), tokens
    return _automatic_cuda_device_count(model_id, run_kwargs), None


def _run_kwargs_with_cuda_lease(
    run_kwargs: Mapping[str, Any],
    lease: CudaDeviceLease,
) -> dict[str, Any]:
    """Pin a dispatched job to a lease without leaking allocator kwargs to the model."""

    updated = dict(run_kwargs)
    load_kwargs = _json_object_from_text(updated.get("load_kwargs_text"))
    load_kwargs["cuda_visible_devices"] = lease.visible_devices
    updated["load_kwargs_text"] = json.dumps(load_kwargs)
    return updated


def _parse_nvidia_smi_rows() -> list[dict[str, float]]:
    """Query GPU memory/utilization stats via ``nvidia-smi`` for auto device selection."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    rows: list[dict[str, float]] = []
    for raw_line in proc.stdout.strip().splitlines():
        parts = [part.strip().replace("%", "") for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2]),
                "utilization": float(parts[3]),
            }
        )
    return rows


def _select_wmfactory_visible_devices(num_devices: int | None = None) -> str:
    """Pick idle GPUs for WMFactory-style models using nvidia-smi heuristics."""
    requested = None if num_devices is None else max(int(num_devices or 1), 1)
    if (getenv_registered("WORLDFOUNDRY_AUTO_CUDA_VISIBLE_DEVICES", "1") or "1") != "1":
        return ""

    try:
        max_mem_fraction = float(
            getenv_registered("WORLDFOUNDRY_AUTO_GPU_MAX_MEMORY_FRACTION", "0.5") or "0.5"
        )
        max_util_fraction = float(
            getenv_registered("WORLDFOUNDRY_AUTO_GPU_MAX_UTILIZATION_FRACTION", "0.5") or "0.5"
        )
    except ValueError:
        max_mem_fraction = 0.5
        max_util_fraction = 0.5

    try:
        rows = _parse_nvidia_smi_rows()
    except Exception:
        rows = []
    if rows:
        def _rank_key(row: Mapping[str, float]) -> tuple[float, float, float]:
            return (
                0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"],
                row["utilization"],
                row["index"],
            )

        ranked = sorted(
            rows,
            key=_rank_key,
        )

        eligible = []
        for row in rows:
            mem_fraction = 0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"]
            util_fraction = row["utilization"] / 100.0
            if mem_fraction < max_mem_fraction and util_fraction < max_util_fraction:
                eligible.append(row)

        if requested is None:
            picked = eligible or ranked[:1]
        else:
            ranked_eligible = sorted(eligible, key=_rank_key)
            picked = ranked_eligible[:requested] if len(ranked_eligible) >= requested else ranked[:requested]
        if picked:
            return ",".join(str(int(row["index"])) for row in picked)

    visible = _normalize_cuda_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    if visible:
        if requested is None:
            return visible
        parts = [part.strip() for part in visible.split(",") if part.strip()]
        return ",".join(parts[:requested])
    return ""


def _wmfactory_visible_devices_for_run(model_id: str, run_kwargs: Mapping[str, Any]) -> str:
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    spec = wmfactory_interactive_model_spec(model_id, load_kwargs=load_kwargs)
    if spec is None:
        return ""
    override = os.getenv(f"WM_{spec.env_prefix}_CUDA_VISIBLE_DEVICES")
    if override is not None:
        return _normalize_cuda_visible_devices(override)
    preferred_visible_devices = spec.preferred_visible_devices
    if spec.supported_process_counts:
        requested_nproc = _requested_torchrun_nproc(model_id, run_kwargs)
        if requested_nproc is not None:
            preferred_visible_devices = requested_nproc
    return _select_wmfactory_visible_devices(preferred_visible_devices)


def _apply_wmfactory_env_hints(model_id: str, run_kwargs: Mapping[str, Any], env: dict[str, str]) -> None:
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    spec = wmfactory_interactive_model_spec(model_id, load_kwargs=load_kwargs)
    if spec is None:
        return
    visible = [part.strip() for part in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if part.strip()]
    if spec.use_dual_device_hint:
        env.setdefault(f"WM_{spec.env_prefix}_GEN_DEVICE", "cuda:0")
        env.setdefault(f"WM_{spec.env_prefix}_DECODE_DEVICE", "cuda:1" if len(visible) >= 2 else "cuda:0")


def _normalize_cuda_visible_devices(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return _normalize_cuda_visible_devices(parsed)
    return text


def _cuda_visible_devices_from_kwargs(run_kwargs: Mapping[str, Any]) -> str:
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    for source in (load_kwargs, call_kwargs):
        for key in ("cuda_visible_devices", "visible_devices", "cuda_devices", "gpu_ids"):
            value = _normalize_cuda_visible_devices(source.get(key))
            if value:
                return value
    return ""


def _cuda_visible_devices_from_device(run_kwargs: Mapping[str, Any]) -> str:
    device = str(run_kwargs.get("device") or "").strip()
    if not device.startswith("cuda:"):
        return ""
    gpu_idx = device.split(":", maxsplit=1)[1].strip()
    if not gpu_idx.isdigit():
        return ""
    parent_visible = _visible_device_items(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    if parent_visible:
        logical_index = int(gpu_idx)
        if 0 <= logical_index < len(parent_visible):
            return parent_visible[logical_index]
    return gpu_idx








def _shutdown_resident_worker(worker: _ResidentWorker, *, force: bool = False) -> None:
    with worker.shutdown_lock:
        if worker.shutdown_complete:
            return
        process = worker.process
        if process.poll() is None and not force:
            try:
                process.stdin.write(b'{"command":"shutdown"}\n')
                process.stdin.flush()
            except (AttributeError, BrokenPipeError, OSError, ValueError):
                pass
            try:
                process.wait(timeout=2)
            except (subprocess.TimeoutExpired, ChildProcessError, OSError):
                pass
        # Parent exit alone does not release CUDA: descendants may still hold it.
        terminate_process_group(process, grace_seconds=0 if force else 2)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        if worker.device_lease is not None:
            worker.device_lease.release()
            worker.device_lease = None

        worker.shutdown_complete = True


def _reap_resident_workers_once(*, now: float | None = None) -> int:
    """Remove dead and expired-idle workers, then reap them without registry locks."""
    retired_workers: list[_ResidentWorker] = []
    current_time = time.monotonic() if now is None else now
    idle_ttl = _resident_worker_idle_ttl()
    # Hold the lifecycle lock until retired processes are actually reaped so a
    # concurrent allocator cannot replace them above the configured hard cap.
    with _RESIDENT_WORKERS_LIFECYCLE_LOCK:
        with _RESIDENT_WORKERS_LOCK:
            for key, candidate in list(_RESIDENT_WORKERS.items()):
                is_dead = candidate.process.poll() is not None
                is_expired = (
                    candidate.in_use == 0
                    and idle_ttl > 0
                    and current_time - candidate.last_used_at >= idle_ttl
                )
                if candidate.in_use == 0 and (is_dead or is_expired):
                    _RESIDENT_WORKERS.pop(key, None)
                    retired_workers.append(candidate)
        for worker in retired_workers:
            _shutdown_resident_worker(worker, force=False)
    return len(retired_workers)


def _resident_worker_reaper_loop() -> None:
    while True:
        idle_ttl = _resident_worker_idle_ttl()
        interval = min(30.0, max(0.1, idle_ttl / 2.0)) if idle_ttl > 0 else 30.0
        if _RESIDENT_WORKERS_REAPER_STOP.wait(interval):
            return
        try:
            _reap_resident_workers_once()
        except Exception:
            # Reaping is best effort and must not permanently disable cleanup
            # because one process object behaved unexpectedly.
            continue


def _ensure_resident_worker_reaper() -> None:
    global _RESIDENT_WORKERS_REAPER_THREAD
    with _RESIDENT_WORKERS_REAPER_LOCK:
        thread = _RESIDENT_WORKERS_REAPER_THREAD
        if thread is not None and thread.is_alive():
            return
        _RESIDENT_WORKERS_REAPER_STOP.clear()
        thread = threading.Thread(
            target=_resident_worker_reaper_loop,
            name="worldfoundry-resident-worker-reaper",
            daemon=True,
        )
        _RESIDENT_WORKERS_REAPER_THREAD = thread
        thread.start()


def _shutdown_all_resident_workers() -> None:
    global _RESIDENT_WORKERS_REAPER_THREAD
    _RESIDENT_WORKERS_REAPER_STOP.set()
    with _RESIDENT_WORKERS_REAPER_LOCK:
        reaper_thread = _RESIDENT_WORKERS_REAPER_THREAD
    if reaper_thread is not None and reaper_thread is not threading.current_thread():
        reaper_thread.join(timeout=1)
    with _RESIDENT_WORKERS_REAPER_LOCK:
        if _RESIDENT_WORKERS_REAPER_THREAD is reaper_thread:
            _RESIDENT_WORKERS_REAPER_THREAD = None
    with _RESIDENT_WORKERS_LOCK:
        workers = list(_RESIDENT_WORKERS.values())
        _RESIDENT_WORKERS.clear()
    for worker in workers:
        _shutdown_resident_worker(worker, force=False)


atexit.register(_shutdown_all_resident_workers)


def _drain_process_output(
    *,
    process: subprocess.Popen[Any],
    decoder: codecs.IncrementalDecoder,
    log_callback: LogCallback | None,
) -> bool:
    """Drain currently available child output without waiting for a newline."""
    if process.stdout is None:
        return True
    fd = process.stdout.fileno()
    reached_eof = False
    # Yield to deadline/cancellation checks even under continuous log output.
    for _ in range(16):
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        except InterruptedError:
            continue
        except OSError:
            reached_eof = True
            break
        if not chunk:
            reached_eof = True
            break
        text = decoder.decode(chunk)
        if text:
            _append_log(log_callback, "stdout", text)
    return reached_eof


def run_record_from_manifest(payload: Mapping[str, Any]) -> RunRecord:
    output_dir = str(payload.get("output_dir") or "")
    return RunRecord(
        run_id=str(payload.get("run_id") or Path(output_dir).name),
        model_id=str(payload.get("model_id") or ""),
        display_name=str(payload.get("display_name") or payload.get("model_id") or ""),
        mode=str(payload.get("mode") or ""),
        status=str(payload.get("status") or ""),
        output_dir=output_dir,
        manifest_path=str(payload.get("manifest_path") or (Path(output_dir) / "manifest.json")),
        preview_video=payload.get("preview_video") or None,
        preview_image=payload.get("preview_image") or None,
        preview_splat=payload.get("preview_splat") or None,
        preview_model=payload.get("preview_model") or None,
        gallery=[str(item) for item in payload.get("gallery") or ()],
        rrd_path=payload.get("rrd_path") or None,
        artifacts=[str(item) for item in payload.get("artifacts") or ()],
        metadata=dict(payload.get("metadata") or {}),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _payload_run_kwargs_with_secret_refs(run_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Move payload credentials into child-only environment references.

    Studio accepts JSON-encoded ``load_kwargs_text`` and ``call_kwargs_text``
    in addition to ordinary nested mappings. If either JSON blob contains a
    credential, the whole original blob is carried through the environment so
    the serialized dispatch payload never contains the secret while the child
    still receives the exact input string.
    """

    secret_env: dict[str, str] = {}
    env_name_by_secret: dict[str, str] = {}
    next_secret_index = 1

    api_key = run_kwargs.get("api_key")
    if isinstance(api_key, str) and api_key:
        secret_env[DISPATCH_API_KEY_ENV] = api_key
        env_name_by_secret[api_key] = DISPATCH_API_KEY_ENV

    def secret_ref(secret: str) -> dict[str, str]:
        nonlocal next_secret_index
        env_name = env_name_by_secret.get(secret)
        if env_name is None:
            while True:
                env_name = f"{DISPATCH_SECRET_ENV_PREFIX}_{next_secret_index}"
                next_secret_index += 1
                if env_name not in secret_env:
                    break
            env_name_by_secret[secret] = env_name
            secret_env[env_name] = secret
        return {SECRET_ENV_REF_KEY: env_name}

    def contains_secret(value: Any, *, sensitive_context: bool = False) -> bool:
        if isinstance(value, Mapping):
            if set(value) == {SECRET_ENV_REF_KEY}:
                return False
            return any(
                contains_secret(
                    item,
                    sensitive_context=sensitive_context or is_sensitive_key(str(key)),
                )
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_secret(item, sensitive_context=sensitive_context) for item in value)
        return isinstance(value, str) and bool(value) and (
            sensitive_context or redact_secret_text(value) != value
        )

    def replace_secret_values(value: Any, *, sensitive_context: bool = False) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {SECRET_ENV_REF_KEY}:
                return dict(value)
            replaced: dict[str, Any] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                item_is_sensitive = sensitive_context or is_sensitive_key(key)
                if key in {"load_kwargs_text", "call_kwargs_text"} and isinstance(item, str):
                    try:
                        parsed_kwargs = json.loads(item)
                    except (TypeError, json.JSONDecodeError):
                        parsed_kwargs = None
                    if parsed_kwargs is not None and contains_secret(parsed_kwargs):
                        replaced[key] = secret_ref(item)
                        continue
                replaced[key] = replace_secret_values(
                    item,
                    sensitive_context=item_is_sensitive,
                )
            return replaced
        if isinstance(value, list):
            return [replace_secret_values(item, sensitive_context=sensitive_context) for item in value]
        if isinstance(value, tuple):
            return tuple(replace_secret_values(item, sensitive_context=sensitive_context) for item in value)
        if isinstance(value, str) and value and (
            sensitive_context or redact_secret_text(value) != value
        ):
            return secret_ref(value)
        return value

    payload_kwargs = replace_secret_values(dict(run_kwargs))
    return dict(payload_kwargs), secret_env


def _repo_root() -> Path:
    return project_root(Path(__file__).resolve())


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_pythonpath(spec: RuntimeCondaEnvSpec, env: Mapping[str, str]) -> str:
    repo = _repo_root()
    paths: list[str] = []
    staged_prepend = os.getenv(CHILD_PYTHONPATH_PREPEND_ENV, "").strip()
    if staged_prepend:
        paths.append(staged_prepend)
    src_root = repo / "src"
    if (src_root / "worldfoundry").is_dir():
        paths.append(str(src_root))
    if (repo / "worldfoundry").is_dir():
        paths.append(str(repo))
    for item in spec.pythonpath_dirs:
        resolved = resolve_worldfoundry_path(str(item), env)
        if not resolved.is_absolute():
            resolved = repo / resolved
        paths.append(str(resolved))
    existing = env.get("PYTHONPATH")
    if existing:
        for item in existing.split(os.pathsep):
            candidate = Path(item).expanduser()
            is_package_dir = any(part in {"site-packages", "dist-packages"} for part in candidate.parts)
            try:
                belongs_to_child = candidate.is_relative_to(spec.env_prefix)
            except ValueError:
                belongs_to_child = False
            if is_package_dir and not belongs_to_child:
                continue
            paths.append(item)
    deduped: list[str] = []
    for item in paths:
        if item and item not in deduped:
            deduped.append(item)
    return os.pathsep.join(deduped)


def _venv_base_prefix(env_prefix: Path) -> Path | None:
    pyvenv_cfg = env_prefix / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        return None
    try:
        lines = pyvenv_cfg.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().lower() == "home" and value.strip():
            home = Path(value.strip())
            return home.parent if home.name == "bin" else home
    return None


def _runtime_library_dirs(spec: RuntimeCondaEnvSpec, env: Mapping[str, str]) -> str:
    prefixes: list[Path] = [spec.env_prefix]
    base_prefix = _venv_base_prefix(spec.env_prefix)
    if base_prefix is not None:
        prefixes.append(base_prefix)

    dirs: list[str] = []
    for prefix in prefixes:
        candidates = [prefix / "lib"]
        candidates.extend(prefix.glob("lib/python*/site-packages/torch/lib"))
        candidates.extend(prefix.glob("lib/python*/site-packages/nvidia/*/lib"))
        for candidate in candidates:
            if candidate.is_dir():
                item = str(candidate)
                if item not in dirs:
                    dirs.append(item)

    shim_dirs = _runtime_library_shim_dirs(spec, dirs)
    dirs = shim_dirs + dirs

    existing = env.get("LD_LIBRARY_PATH", "")
    for item in existing.split(os.pathsep):
        if item and item not in dirs:
            dirs.append(item)
    return os.pathsep.join(dirs)


def _runtime_library_shim_dirs(spec: RuntimeCondaEnvSpec, library_dirs: list[str]) -> list[str]:
    """Create transient soname shims required by some NVIDIA Python wheels."""
    digest = hashlib.sha1(str(spec.env_prefix).encode("utf-8")).hexdigest()[:12]
    root = Path(
        os.getenv(
            "WORLDFOUNDRY_RUNTIME_LIB_SHIM_DIR",
            str(Path(tempfile.gettempdir()) / "worldfoundry-runtime-lib-shims"),
        )
    )
    shim_dir = root / digest
    created = False
    for library_name in ("libcudnn.so",):
        if any((Path(item) / library_name).exists() for item in library_dirs):
            continue
        target: Path | None = None
        for item in library_dirs:
            candidates = sorted(Path(item).glob(f"{library_name}.*"))
            if candidates:
                target = candidates[0]
                break
        if target is None:
            continue
        try:
            shim_dir.mkdir(parents=True, exist_ok=True)
            link_path = shim_dir / library_name
            if link_path.is_symlink() or link_path.exists():
                try:
                    if link_path.resolve() == target.resolve():
                        created = True
                        continue
                except OSError:
                    pass
                if link_path.is_symlink():
                    link_path.unlink()
                else:
                    continue
            link_path.symlink_to(target)
            created = True
        except OSError:
            continue
    return [str(shim_dir)] if created else []


def _runtime_cache_dir(spec: RuntimeCondaEnvSpec, name: str) -> str:
    root = Path(
        os.getenv(
            "WORLDFOUNDRY_RUNTIME_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "worldfoundry-runtime-cache"),
        )
    )
    path = root / spec.model_id / name
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(path)


def _runtime_env(spec: RuntimeCondaEnvSpec, device: str | None = None) -> dict[str, str]:
    """Build the child-process environment: Conda paths, PYTHONPATH, and optional GPU pin."""
    env = os.environ.copy()
    # A dedicated Conda runtime must not import packages from the launching
    # account's ``~/.local`` tree. Besides violating the selected runtime
    # contract, a partially installed user package can make an otherwise
    # complete model environment fail during import (for example, boto3
    # without its jmespath dependency while Transformers imports Accelerate).
    env["PYTHONNOUSERSITE"] = "1"
    # Preserve an explicitly offline parent Workspace. Unset variables still
    # retain the normal online behavior, while strict local-checkpoint runs do
    # not unexpectedly download from a child Conda process.
    env[STUDIO_CONDA_CHILD_ENV] = "1"
    env["WORLDFOUNDRY_REPO_ROOT"] = str(_repo_root())
    ckpt_dir = resolve_ckpt_dir(env)
    hfd_root = resolve_hfd_root(env)
    env.setdefault("WORLDFOUNDRY_CKPT_DIR", str(ckpt_dir))
    env.setdefault("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))
    apply_hf_endpoint_override(env)
    local_hf_home = ckpt_dir / "huggingface"
    if local_hf_home.is_dir():
        env.setdefault("HF_HOME", str(local_hf_home))
        env.setdefault("HF_HUB_CACHE", str(local_hf_home / "hub"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(local_hf_home / "hub"))
    else:
        env.setdefault("HF_HUB_CACHE", str(resolve_hf_cache_dir(env)))
    env["CONDA_PREFIX"] = str(spec.env_prefix)
    env["WORLDFOUNDRY_CONDA_ENVS_ROOT"] = str(spec.env_prefix.parent)
    env["WORLDFOUNDRY_CONDA_ENV_ROOT"] = str(spec.env_prefix.parent)
    if spec.env_prefix.parent.name == "envs":
        env["WORLDFOUNDRY_CONDA_ROOT"] = str(spec.env_prefix.parent.parent)
    env["PATH"] = os.pathsep.join([str(spec.env_prefix / "bin"), env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = _runtime_library_dirs(spec, env)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("TERMINFO_DIRS", "/usr/share/terminfo:/lib/terminfo")
    env.pop("TERMINFO", None)
    if spec.model_id == "gen3c":
        cuda_home_override = os.getenv("WORLDFOUNDRY_CUDA_HOME_OVERRIDE", "").strip()
        if cuda_home_override.lower() in {"0", "none", "unset"}:
            env.pop("CUDA_HOME", None)
        else:
            env["CUDA_HOME"] = cuda_home_override or str(spec.env_prefix)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("WORLDFOUNDRY_GEN3C_IMPORT_STAGGER_SECONDS", "1")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", _runtime_cache_dir(spec, "torchinductor"))
        env.setdefault("TRITON_CACHE_DIR", _runtime_cache_dir(spec, "triton"))
        env.setdefault("CUDA_CACHE_PATH", _runtime_cache_dir(spec, "cuda"))
        env.setdefault("XDG_CACHE_HOME", _runtime_cache_dir(spec, "xdg"))
    env["PYTHONPATH"] = _runtime_pythonpath(spec, env)
    # Map device string like "cuda:3" to CUDA_VISIBLE_DEVICES=3
    # so each job can run on a specific GPU instead of all defaulting to GPU 0.
    if device and device.startswith("cuda"):
        gpu_idx = device.split(":")[1] if ":" in device else None
        if gpu_idx is not None and gpu_idx.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    return env


def _prepare_resident_run_context(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
) -> _ResidentRunContext | None:
    if not _resident_workers_enabled_for_model(model_id):
        return None

    requested_torchrun_nproc = (
        _lingbot_torchrun_nproc(model_id, run_kwargs)
        or _explicit_torchrun_nproc(model_id, run_kwargs)
        or _default_torchrun_nproc(model_id, run_kwargs)
    )
    requested_internal_torchrun_nproc = _internal_torchrun_nproc(model_id, run_kwargs)
    if requested_torchrun_nproc > 1 or requested_internal_torchrun_nproc > 1:
        return None

    kwargs_cuda_visible_devices = _cuda_visible_devices_from_kwargs(run_kwargs)
    device_cuda_visible_devices = _cuda_visible_devices_from_device(run_kwargs)
    parent_cuda_visible_devices = _normalize_cuda_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    lease_count, requested_cuda_tokens = _cuda_lease_request(model_id, run_kwargs)
    automatic_cuda_device_count = lease_count if requested_cuda_tokens is None else 0
    explicit_cuda_visible_devices = _expand_visible_devices_for_model_devices(
        kwargs_cuda_visible_devices
        or device_cuda_visible_devices
        or ("" if automatic_cuda_device_count else parent_cuda_visible_devices),
        run_kwargs,
    )
    child_run_kwargs = _run_kwargs_for_child(
        run_kwargs,
        cuda_visible_devices=explicit_cuda_visible_devices,
    )
    payload_run_kwargs, secret_env = _payload_run_kwargs_with_secret_refs(child_run_kwargs)
    if secret_env:
        return None

    wmfactory_visible_devices = (
        ""
        if explicit_cuda_visible_devices or automatic_cuda_device_count
        else _wmfactory_visible_devices_for_run(model_id, run_kwargs)
    )
    env = _runtime_env(spec, device=None if wmfactory_visible_devices else str(run_kwargs.get("device", "")))
    if explicit_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = explicit_cuda_visible_devices
    elif wmfactory_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = wmfactory_visible_devices
    _apply_wmfactory_env_hints(model_id, run_kwargs, env)

    key = (
        model_id,
        str(spec.python_executable),
        str(spec.env_prefix),
        str(Path(workspace_root).expanduser()),
        env.get("CUDA_VISIBLE_DEVICES", ""),
        env.get("WORLDFOUNDRY_CKPT_DIR", ""),
        env.get("WORLDFOUNDRY_HFD_ROOT", ""),
        env.get("PYTHONPATH", ""),
    )
    return _ResidentRunContext(
        child_run_kwargs=child_run_kwargs,
        payload_run_kwargs=payload_run_kwargs,
        env=env,
        key=key,
        automatic_cuda_device_count=automatic_cuda_device_count,
        requested_cuda_tokens=requested_cuda_tokens,
    )


def _start_resident_worker(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    context: _ResidentRunContext,
    log_callback: LogCallback | None,
    base_key: tuple[str, ...] | None = None,
    device_lease: CudaDeviceLease | None = None,
) -> _ResidentWorker:
    command = [
        str(spec.python_executable),
        "-m",
        "worldfoundry.studio.runtime_job",
        "run-manager-worker",
        "--workspace-root",
        workspace_root,
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            env=context.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=sys.platform != "win32",
            **({"pass_fds": device_lease.file_descriptors} if os.name == "posix" and device_lease else {}),
        )
    except OSError as exc:
        raise _ResidentWorkerUnavailable(
            f"could not start resident worker for {model_id}: {exc}"
        ) from exc
    try:
        if device_lease is not None:
            device_lease.process_group_id = process.pid
        if process.stdin is None or process.stdout is None:
            raise _ResidentWorkerUnavailable("resident worker pipes were not created")
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
    except BaseException:
        terminate_process_group(process, grace_seconds=0)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()
        raise
    worker = _ResidentWorker(
        key=context.key,
        base_key=base_key or context.key,
        model_id=model_id,
        process=process,
        lock=threading.RLock(),
        decoder=codecs.getincrementaldecoder("utf-8")("replace"),
        command=command,
        created_at=time.monotonic(),
        last_used_at=time.monotonic(),
        device_lease=device_lease,
    )
    return worker


def _resident_worker_for(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    context: _ResidentRunContext,
    log_callback: LogCallback | None,
    cancel_requested: CancelCallback | None = None,
    deadline: float | None = None,
) -> _ResidentWorker:
    options = dict(model_id=model_id, spec=spec, workspace_root=workspace_root,
                   context=context, log_callback=log_callback,
                   cancel_requested=cancel_requested, deadline=deadline)
    worker = _resident_worker_for_locked(**options, allow_start=False)
    if worker is not None:
        return worker
    count = context.automatic_cuda_device_count or len(context.requested_cuda_tokens or ())
    lease = None
    try:
        if count:
            _retire_idle_resident_workers_for_cuda_request(
                count, tokens=context.requested_cuda_tokens,
                cancel_requested=cancel_requested, deadline=deadline,
            )
            # Never wait for a GPU while holding the lifecycle lock: a finishing
            # request must be able to retire its worker and return its lease.
            def allocation_interrupt() -> bool:
                _check_dispatch_interrupt(cancel_requested, deadline)
                # The pool invokes this under its own lock. Avoid reversing
                # the registry -> pool lock order used by worker release.
                ready = False
                if _RESIDENT_WORKERS_LOCK.acquire(blocking=False):
                    try:
                        ready = any(candidate.base_key == context.key and candidate.process.poll() is None
                                    for candidate in _RESIDENT_WORKERS.values())
                    finally:
                        _RESIDENT_WORKERS_LOCK.release()
                if ready:
                    raise _ResidentWorkerReady
                return False

            while True:
                try:
                    lease = _automatic_gpu_pool().acquire(
                        count, tokens=context.requested_cuda_tokens,
                        cancel_requested=allocation_interrupt, deadline=deadline,
                    )
                    break
                except _ResidentWorkerReady:
                    worker = _resident_worker_for_locked(**options, allow_start=False)
                    if worker is not None:
                        return worker
        worker = _resident_worker_for_locked(**options, device_lease=lease)
        assert worker is not None
        return worker
    finally:
        if lease is not None and (worker is None or worker.device_lease is not lease):
            lease.release()


def _resident_worker_for_locked(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    context: _ResidentRunContext,
    log_callback: LogCallback | None,
    device_lease: CudaDeviceLease | None = None,
    allow_start: bool = True,
    cancel_requested: CancelCallback | None = None,
    deadline: float | None = None,
) -> _ResidentWorker | None:
    retired_workers: list[_ResidentWorker] = []
    worker: _ResidentWorker | None = None
    started_worker = False
    base_key = context.key
    with _resident_lifecycle_lock(cancel_requested, deadline):
        try:
            max_workers = _resident_worker_max_workers()
            with _RESIDENT_WORKERS_LOCK:
                now = time.monotonic()
                idle_ttl = _resident_worker_idle_ttl()
                for key, candidate in list(_RESIDENT_WORKERS.items()):
                    is_dead = candidate.process.poll() is not None
                    is_expired = idle_ttl > 0 and now - candidate.last_used_at >= idle_ttl
                    if is_dead:
                        _RESIDENT_WORKERS.pop(key, None)
                        # An existing waiter may still be draining the dead
                        # process. Do not close its pipes from this thread.
                        if candidate.in_use == 0:
                            retired_workers.append(candidate)
                    elif candidate.in_use == 0 and is_expired:
                        _RESIDENT_WORKERS.pop(key, None)
                        retired_workers.append(candidate)

                if max_workers == 0:
                    for key, candidate in list(_RESIDENT_WORKERS.items()):
                        if candidate.in_use == 0:
                            _RESIDENT_WORKERS.pop(key, None)
                            retired_workers.append(candidate)
                    raise _ResidentWorkerUnavailable("resident worker capacity is disabled")

                # Enforce a lowered cap before reusing an existing worker. Prefer
                # keeping the requested key when another idle worker can retire.
                while len(_RESIDENT_WORKERS) > max_workers:
                    idle_workers = [candidate for candidate in _RESIDENT_WORKERS.values() if candidate.in_use == 0]
                    if not idle_workers:
                        raise _ResidentWorkerUnavailable(
                            f"resident worker pool exceeds the {max_workers}-worker cap and all slots are busy"
                        )
                    victim = min(
                        idle_workers,
                        key=lambda candidate: (candidate.key == context.key, candidate.last_used_at),
                    )
                    _RESIDENT_WORKERS.pop(victim.key, None)
                    retired_workers.append(victim)

                existing = next(
                    (candidate for candidate in _RESIDENT_WORKERS.values() if candidate.base_key == base_key),
                    None,
                )
                if existing is not None:
                    existing.in_use += 1
                    worker = existing

                while len(_RESIDENT_WORKERS) >= max_workers:
                    if worker is not None:
                        break
                    idle_workers = [candidate for candidate in _RESIDENT_WORKERS.values() if candidate.in_use == 0]
                    if not idle_workers:
                        raise _ResidentWorkerUnavailable(
                            f"all {max_workers} resident worker slots are busy"
                        )
                    victim = min(idle_workers, key=lambda candidate: candidate.last_used_at)
                    _RESIDENT_WORKERS.pop(victim.key, None)
                    retired_workers.append(victim)

        finally:
            # Never wait for process exit under the registry lock. Keep the
            # lifecycle lock until exit so replacements cannot exceed the cap.
            for retired_worker in retired_workers:
                _shutdown_resident_worker(retired_worker, force=False)

        if worker is None:
            if not allow_start:
                return None
            _check_dispatch_interrupt(cancel_requested, deadline)
            if device_lease is not None:
                allocated_env = dict(context.env)
                allocated_env["CUDA_VISIBLE_DEVICES"] = device_lease.visible_devices
                context = replace(
                    context,
                    env=allocated_env,
                    key=(*base_key, f"auto-cuda:{device_lease.visible_devices}"),
                )
            worker = _start_resident_worker(
                    model_id=model_id,
                    spec=spec,
                    workspace_root=workspace_root,
                    context=context,
                    log_callback=log_callback,
                    base_key=base_key,
                    device_lease=device_lease,
            )
            started_worker = True
            with _RESIDENT_WORKERS_LOCK:
                worker.in_use = 1
                _RESIDENT_WORKERS[worker.key] = worker

    if worker is None:  # pragma: no cover - all paths above assign or raise
        raise _ResidentWorkerUnavailable("resident worker allocation failed")
    try:
        _ensure_resident_worker_reaper()
        if started_worker:
            device_suffix = (
                f" (CUDA_VISIBLE_DEVICES={worker.device_lease.visible_devices})"
                if worker.device_lease is not None
                else ""
            )
            _append_log(
                log_callback,
                "system",
                f"started resident conda worker for {model_id} in {spec.resolved_env_name}: "
                f"{spec.python_executable}{device_suffix}\n",
            )
    except BaseException:
        if started_worker:
            _drop_resident_worker(worker, force=True)
        _release_resident_worker(worker)
        raise
    return worker


def _release_resident_worker(worker: _ResidentWorker) -> None:
    retire_for_waiter = False
    with _RESIDENT_WORKERS_LOCK:
        if worker.in_use > 0:
            worker.in_use -= 1
        if worker.in_use == 0:
            worker.last_used_at = time.monotonic()
            retire_for_waiter = bool(
                worker.device_lease is not None and worker.device_lease.allocation_waiting
            )
    if retire_for_waiter:
        _drop_resident_worker(worker, force=False, only_if_idle=True)


def _retire_idle_resident_workers_for_cuda_request(
    requested_count: int, *, tokens: tuple[str, ...] | None = None,
    cancel_requested: CancelCallback | None = None, deadline: float | None = None,
) -> int:
    """Release enough idle resident leases for a larger queued CUDA request.

    A worker that became idle before a multi-GPU allocator started waiting will
    never observe ``allocation_waiting`` in ``_release_resident_worker``.  That
    left the large request blocked even though every retained worker was idle.
    Proactively retire the oldest idle lease holders before entering the
    allocator wait.
    """

    requested = max(int(requested_count or 0), 0)
    if requested <= 0:
        return 0
    pool = _automatic_gpu_pool()
    missing = requested - pool.available_count
    if missing <= 0 and tokens is None:
        return 0

    retired_workers: list[_ResidentWorker] = []
    released_capacity = 0
    with _resident_lifecycle_lock(cancel_requested, deadline):
        with _RESIDENT_WORKERS_LOCK:
            candidates = sorted(
                (
                    worker
                    for worker in _RESIDENT_WORKERS.values()
                    if worker.in_use == 0 and worker.device_lease is not None
                    and (tokens is None or set(tokens).intersection(worker.device_lease.tokens))
                ),
                key=lambda worker: worker.last_used_at,
            )
            for worker in candidates:
                if _RESIDENT_WORKERS.get(worker.key) is not worker:
                    continue
                _RESIDENT_WORKERS.pop(worker.key, None)
                retired_workers.append(worker)
                released_capacity += len(worker.device_lease.tokens)
                if tokens is None and released_capacity >= missing:
                    break
        for worker in retired_workers:
            _shutdown_resident_worker(worker, force=False)
    return len(retired_workers)


def _retire_idle_resident_workers_for_new_resident_key(
    base_key: tuple[str, ...],
    requested_count: int,
) -> int:
    """Make CUDA capacity for a new resident key without retiring a reusable worker."""

    with _RESIDENT_WORKERS_LOCK:
        has_reusable_worker = any(
            worker.base_key == base_key and worker.process.poll() is None
            for worker in _RESIDENT_WORKERS.values()
        )
    if has_reusable_worker:
        return 0
    return _retire_idle_resident_workers_for_cuda_request(requested_count)


def _drop_resident_worker(worker: _ResidentWorker, *, force: bool = False, only_if_idle: bool = False) -> None:
    with _RESIDENT_WORKERS_LIFECYCLE_LOCK:
        with _RESIDENT_WORKERS_LOCK:
            if only_if_idle and worker.in_use != 0:
                return
            if _RESIDENT_WORKERS.get(worker.key) is worker:
                _RESIDENT_WORKERS.pop(worker.key, None)
        _shutdown_resident_worker(worker, force=force)


def _send_resident_worker_request(
    worker: _ResidentWorker,
    *,
    request_payload: Mapping[str, Any],
    cancel_requested: CancelCallback | None = None,
    deadline: float | None = None,
) -> None:
    if worker.process.poll() is not None:
        raise _ResidentWorkerUnavailable(f"resident worker exited before request: {worker.process.returncode}")
    if worker.process.stdin is None:
        raise _ResidentWorkerUnavailable("resident worker stdin is closed")
    data = json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    try:
        fd = worker.process.stdin.fileno()
        offset = 0
        while offset < len(data):
            _check_dispatch_interrupt(cancel_requested, deadline)
            try:
                offset += os.write(fd, data[offset:])
            except BlockingIOError:
                select.select([], [fd], [], 0.05)
    except (BrokenPipeError, OSError, ValueError) as exc:
        if isinstance(exc, TimeoutError):
            raise
        raise _ResidentWorkerUnavailable("resident worker pipe closed before request") from exc


def _acquire_resident_worker_request_lock(
    worker: _ResidentWorker,
    *,
    deadline: float | None,
    request_timeout: float,
    cancel_requested: CancelCallback | None,
) -> None:
    """Acquire a worker serially while continuing to honor timeout and cancellation."""
    while True:
        if cancel_requested is not None and cancel_requested():
            raise RuntimeError("resident inference worker cancelled while waiting for an available worker")
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise TimeoutError(f"resident worker request timed out after {request_timeout:.1f}s")
        poll_interval = 0.05
        if deadline is not None:
            poll_interval = min(poll_interval, max(deadline - now, 0.0))
        if worker.lock.acquire(timeout=poll_interval):
            return


def _wait_for_resident_worker_result(
    worker: _ResidentWorker,
    *,
    result_path: Path,
    error_path: Path,
    log_callback: LogCallback | None,
    cancel_requested: CancelCallback | None,
    deadline: float | None = None,
    request_timeout: float | None = None,
) -> RunRecord:
    timeout = _resident_worker_request_timeout() if request_timeout is None else max(request_timeout, 0.0)
    if deadline is None and timeout > 0:
        deadline = time.monotonic() + timeout
    stdout_eof = False
    process = worker.process
    if process.stdout is None:
        raise _ResidentWorkerUnavailable("resident worker stdout is closed")
    stdout_fd = process.stdout.fileno()

    while True:
        if cancel_requested is not None and cancel_requested():
            try:
                _append_log(log_callback, "system", "termination requested for resident inference worker\n")
            finally:
                _drop_resident_worker(worker, force=True)
            raise RuntimeError("resident inference worker cancelled")

        now = time.monotonic()
        if deadline is not None and now >= deadline:
            _drop_resident_worker(worker, force=True)
            raise TimeoutError(f"resident worker request timed out after {timeout:.1f}s")

        if result_path.is_file():
            worker.last_used_at = time.monotonic()
            _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
            return run_record_from_manifest(json.loads(result_path.read_text(encoding="utf-8")))

        if error_path.is_file():
            worker.last_used_at = time.monotonic()
            _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
            payload = json.loads(error_path.read_text(encoding="utf-8"))
            detail = payload.get("traceback") or payload.get("error") or f"see {error_path}"
            # A failed resident request can leave partially constructed model
            # state, CUDA allocations, and imported source modules in the
            # worker. Reusing that process makes the next request both less
            # deterministic and, during development, liable to execute stale
            # code. Retire it before surfacing the original error so the next
            # request always starts from a clean interpreter.
            _drop_resident_worker(worker, force=True)
            raise RuntimeError(str(detail))

        if process.poll() is not None:
            terminate_process_group(process, grace_seconds=0.5)
            while not stdout_eof:
                _check_dispatch_interrupt(cancel_requested, deadline)
                stdout_eof = _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
                if not stdout_eof:
                    time.sleep(0.05)
            _drop_resident_worker(worker, force=False)
            raise RuntimeError(f"resident worker for {worker.model_id} exited with status {process.returncode}")

        if not stdout_eof:
            select_timeout = 0.5
            if deadline is not None:
                select_timeout = min(select_timeout, max(deadline - time.monotonic(), 0.0))
            readable, _, _ = select.select([stdout_fd], [], [], select_timeout)
            if readable:
                stdout_eof = _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
        else:
            sleep_interval = 0.05
            if deadline is not None:
                sleep_interval = min(sleep_interval, max(deadline - time.monotonic(), 0.0))
            if sleep_interval > 0:
                time.sleep(sleep_interval)


def _run_manager_payload_in_resident_conda(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
    dispatch_root: str | Path,
    log_callback: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    deadline: float | None = None,
    request_timeout: float | None = None,
) -> RunRecord | None:
    dispatch_started_at = time.monotonic()
    request_timeout = _inference_request_timeout() if request_timeout is None else request_timeout
    if deadline is None and request_timeout > 0:
        deadline = dispatch_started_at + request_timeout
    _check_dispatch_interrupt(cancel_requested, deadline)
    context = _prepare_resident_run_context(
        model_id=model_id,
        spec=spec,
        workspace_root=workspace_root,
        run_kwargs=run_kwargs,
    )
    if context is None:
        return None

    dispatch_dir = Path(dispatch_root)
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    payload_path = dispatch_dir / "payload.json"
    result_path = dispatch_dir / "result.json"
    error_path = dispatch_dir / "error.json"
    for stale_path in (result_path, error_path):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
    payload_path.write_text(
        json.dumps(
            {
                "workspace_root": workspace_root,
                "run_kwargs": _json_safe(context.payload_run_kwargs),
                "resident_worker": True,
                "request_id": request_id,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    worker = _resident_worker_for(
        model_id=model_id,
        spec=spec,
        workspace_root=workspace_root,
        context=context,
        log_callback=log_callback,
        cancel_requested=cancel_requested,
        deadline=deadline,
    )
    lock_acquired = False
    try:
        _acquire_resident_worker_request_lock(
            worker,
            deadline=deadline,
            request_timeout=request_timeout,
            cancel_requested=cancel_requested,
        )
        lock_acquired = True
        _append_log(log_callback, "system", f"dispatching {model_id} to resident conda worker\n")
        try:
            _send_resident_worker_request(
                worker,
                cancel_requested=cancel_requested,
                deadline=deadline,
                request_payload={
                    "request_id": request_id,
                    "run_kwargs": _json_safe(context.payload_run_kwargs),
                    "result_path": str(result_path),
                    "error_path": str(error_path),
                },
            )
        except _ResidentWorkerUnavailable:
            _drop_resident_worker(worker, force=True)
            raise
        return _wait_for_resident_worker_result(
            worker,
            result_path=result_path,
            error_path=error_path,
            log_callback=log_callback,
            cancel_requested=cancel_requested,
            deadline=deadline,
            request_timeout=request_timeout,
        )
    except BaseException:
        # Once a request owns the worker, an observer/pipe failure must not
        # leave model execution running in the background. Waiting requests
        # must not tear down the worker of another request.
        if lock_acquired:
            _drop_resident_worker(worker, force=True)
        raise
    finally:
        if lock_acquired:
            worker.lock.release()
        _release_resident_worker(worker)


def _append_log(log_callback: LogCallback | None, stream: str, text: str) -> None:
    if log_callback is not None:
        log_callback(stream, text)
    elif stream == "stderr":
        sys.stderr.write(text)
    else:
        sys.stdout.write(text)


def _run_manager_payload_one_shot(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
    dispatch_root: str | Path,
    log_callback: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    deadline: float | None = None,
    device_lease: CudaDeviceLease | None = None,
) -> RunRecord:
    """Run a Studio manager job in an isolated Conda env and return the child RunRecord.

    Writes ``payload.json``, spawns ``runtime_job run-manager-payload`` (optionally
    via ``torchrun``), streams stdout to ``log_callback``, and reads ``result.json``.
    """
    dispatch_dir = Path(dispatch_root)
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload_path = dispatch_dir / "payload.json"
    result_path = dispatch_dir / "result.json"
    for stale_log in ("stdout.log", "stderr.log"):
        stale_log_path = dispatch_dir / stale_log
        if stale_log_path.exists():
            stale_log_path.unlink()
    kwargs_cuda_visible_devices = _cuda_visible_devices_from_kwargs(run_kwargs)
    device_cuda_visible_devices = _cuda_visible_devices_from_device(run_kwargs)
    parent_cuda_visible_devices = _normalize_cuda_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    explicit_cuda_visible_devices = _expand_visible_devices_for_model_devices(
        kwargs_cuda_visible_devices
        or device_cuda_visible_devices
        or parent_cuda_visible_devices,
        run_kwargs,
    )
    torchrun_nproc = _lingbot_torchrun_nproc(model_id, run_kwargs)
    if torchrun_nproc <= 1:
        torchrun_nproc = _explicit_torchrun_nproc(model_id, run_kwargs)
    if torchrun_nproc <= 1:
        torchrun_nproc = _default_torchrun_nproc(model_id, run_kwargs)
    internal_torchrun_nproc = _internal_torchrun_nproc(model_id, run_kwargs)
    dispatch_run_kwargs = _with_matrix_game3_parallelism(
        model_id,
        run_kwargs,
        world_size=max(torchrun_nproc, 1),
    )
    dispatch_run_kwargs = _with_lingbot_world_parallelism(
        model_id,
        dispatch_run_kwargs,
        world_size=max(torchrun_nproc, 1),
    )
    dispatch_run_kwargs = _with_wan_vace_parallelism(
        model_id,
        dispatch_run_kwargs,
        world_size=max(torchrun_nproc, 1),
    )
    dispatch_run_kwargs = _with_longvie2_parallelism(
        model_id,
        dispatch_run_kwargs,
        world_size=max(torchrun_nproc, 1),
    )
    # Strip parent-only kwargs and remap cuda:N indices for the child process.
    child_run_kwargs = _run_kwargs_for_child(
        dispatch_run_kwargs,
        cuda_visible_devices=explicit_cuda_visible_devices,
    )
    payload_run_kwargs, secret_env = _payload_run_kwargs_with_secret_refs(child_run_kwargs)
    payload_path.write_text(
        json.dumps(
            {
                "workspace_root": workspace_root,
                "run_kwargs": _json_safe(payload_run_kwargs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if result_path.exists():
        result_path.unlink()

    internal_torchrun_nproc = _internal_torchrun_nproc(model_id, child_run_kwargs)
    wmfactory_visible_devices = "" if explicit_cuda_visible_devices else _wmfactory_visible_devices_for_run(model_id, run_kwargs)
    runtime_args = [
        "-m",
        "worldfoundry.studio.runtime_job",
        "run-manager-payload",
        "--payload-path",
        str(payload_path),
        "--result-path",
        str(result_path),
    ]
    # Multi-GPU models launch via torchrun; single-GPU uses the env's python directly.
    if torchrun_nproc > 1:
        command = [
            str(spec.python_executable),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={torchrun_nproc}",
            *runtime_args,
        ]
    else:
        command = [
            str(spec.python_executable),
            *runtime_args,
        ]
    _append_log(
        log_callback,
        "system",
        (
            f"dispatching {model_id} to conda env {spec.resolved_env_name} with torchrun "
            f"nproc={torchrun_nproc}: {spec.python_executable}\n"
            if torchrun_nproc > 1
            else f"dispatching {model_id} to conda env {spec.resolved_env_name}: {spec.python_executable}\n"
        ),
    )
    env = _runtime_env(
        spec,
        device=(
            None
            if torchrun_nproc > 1 or internal_torchrun_nproc > 1 or wmfactory_visible_devices
            else str(run_kwargs.get("device", ""))
        ),
    )
    if explicit_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = explicit_cuda_visible_devices
    elif wmfactory_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = wmfactory_visible_devices
    env.update(secret_env)
    _apply_wmfactory_env_hints(model_id, run_kwargs, env)
    if torchrun_nproc > 1:
        if model_id in {LINGBOT_WORLD_MODEL_ID, LINGBOT_WORLD_V2_MODEL_ID}:
            env[TORCHRUN_LINGBOT_ENV] = "1"
        if model_id in TORCHRUN_DISTRIBUTED_MODELS:
            env[TORCHRUN_DISTRIBUTED_ENV] = "1"
    if torchrun_nproc > 1 or internal_torchrun_nproc > 1:
        requested_nproc = max(torchrun_nproc, internal_torchrun_nproc)
        visible_count = _cuda_visible_device_count(env.get("CUDA_VISIBLE_DEVICES"))
        if (
            explicit_cuda_visible_devices
            or wmfactory_visible_devices
            or model_id == MATRIX_GAME3_MODEL_ID
        ) and visible_count < requested_nproc:
            selected_devices = explicit_cuda_visible_devices or env.get("CUDA_VISIBLE_DEVICES", "")
            raise RuntimeError(
                f"{model_id} requested nproc={requested_nproc} but CUDA_VISIBLE_DEVICES="
                f"{selected_devices!r} exposes only {visible_count} device(s)"
            )
        # Let torchrun see all GPUs when the pinned set is too small for nproc.
        if visible_count < requested_nproc:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        env.setdefault("OMP_NUM_THREADS", "1")
    _check_dispatch_interrupt(cancel_requested, deadline)
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=sys.platform != "win32",
        **({"pass_fds": device_lease.file_descriptors} if os.name == "posix" and device_lease else {}),
    )
    try:
        if device_lease is not None:
            device_lease.process_group_id = process.pid
        assert process.stdout is not None
        stdout_fd = process.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stdout_eof = False
        group_cleaned = False
        while True:
            _check_dispatch_interrupt(cancel_requested, deadline)
            if process.poll() is not None and not group_cleaned:
                # Descendants can keep stdout open after the leader exits.
                # Stop them before draining, otherwise EOF may never arrive.
                terminate_process_group(process, grace_seconds=0.5)
                group_cleaned = True
            if not stdout_eof:
                wait = 0.1 if deadline is None else min(0.1, max(deadline - time.monotonic(), 0.0))
                readable, _, _ = select.select([stdout_fd], [], [], wait)
                if readable:
                    stdout_eof = _drain_process_output(
                        process=process, decoder=decoder, log_callback=log_callback,
                    )
            elif not group_cleaned:
                time.sleep(0.05)
            if group_cleaned and stdout_eof:
                remainder = decoder.decode(b"", final=True)
                if remainder:
                    _append_log(log_callback, "stdout", remainder)
                break
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(f"{model_id} conda runtime exited with status {exit_code}: {spec.python_executable}")
        if not result_path.is_file():
            raise RuntimeError(f"{model_id} conda runtime did not write result payload: {result_path}")
        return run_record_from_manifest(json.loads(result_path.read_text(encoding="utf-8")))
    finally:
        # Callbacks, select/read failures, cancellation and normal completion all
        # stop the owned group before its lease can be returned to the pool.
        try:
            terminate_process_group(process, grace_seconds=0.5)
        finally:
            if process.stdout is not None:
                process.stdout.close()


def _run_manager_payload_in_conda_impl(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
    dispatch_root: str | Path,
    log_callback: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    deadline: float | None = None,
    request_timeout: float | None = None,
) -> RunRecord:
    _check_dispatch_interrupt(cancel_requested, deadline)
    try:
        record = _run_manager_payload_in_resident_conda(
            model_id=model_id, spec=spec, workspace_root=workspace_root,
            run_kwargs=run_kwargs, dispatch_root=dispatch_root,
            log_callback=log_callback, cancel_requested=cancel_requested,
            deadline=deadline, request_timeout=request_timeout,
        )
    except _ResidentWorkerUnavailable as exc:
        _append_log(log_callback, "system", f"resident worker unavailable; falling back to one-shot subprocess: {exc}\n")
        record = None
    if record is not None:
        return record

    count, tokens = _cuda_lease_request(model_id, run_kwargs)
    lease = None
    try:
        if count:
            _retire_idle_resident_workers_for_cuda_request(
                count, tokens=tokens, cancel_requested=cancel_requested, deadline=deadline,
            )
            lease = _automatic_gpu_pool().acquire(
                count, tokens=tokens, cancel_requested=cancel_requested, deadline=deadline,
            )
            _append_log(log_callback, "system", f"assigned CUDA_VISIBLE_DEVICES={lease.visible_devices} for {model_id}\n")
            run_kwargs = _run_kwargs_with_cuda_lease(run_kwargs, lease)
        return _run_manager_payload_one_shot(
            model_id=model_id, spec=spec, workspace_root=workspace_root,
            run_kwargs=run_kwargs, dispatch_root=dispatch_root,
            log_callback=log_callback, cancel_requested=cancel_requested,
            deadline=deadline, device_lease=lease,
        )
    finally:
        if lease is not None:
            lease.release()


def run_manager_payload_in_conda(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
    dispatch_root: str | Path,
    log_callback: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
) -> RunRecord:
    """Run managed inference with GPU admission and one end-to-end timeout."""
    timeout = _inference_request_timeout()
    deadline = time.monotonic() + timeout if timeout > 0 else None
    return _run_manager_payload_in_conda_impl(
        model_id=model_id, spec=spec, workspace_root=workspace_root,
        run_kwargs=run_kwargs, dispatch_root=dispatch_root,
        log_callback=log_callback, cancel_requested=cancel_requested,
        deadline=deadline, request_timeout=timeout,
    )

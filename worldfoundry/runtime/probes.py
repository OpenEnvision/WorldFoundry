"""GPU and environment probe utilities for WorldFoundry preflight checks.

Provides subprocess-based probes for Python module importability, Torch+CUDA
functionality, and low-level CUDA driver memory allocation, plus helpers that
assemble aggregate reports for diagnostic dashboards and preflight manifests.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

# This file lives at <repo>/worldfoundry/runtime/probes.py, so the repository
# root is two levels above the containing package (parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]
STRICT_IMPORT_MODULES = frozenset({"pkg_resources", "droid_backends", "groundingdino._C", "sam2._C"})
TORCH_FIRST_STRICT_IMPORT_MODULES = frozenset({"droid_backends", "groundingdino._C", "sam2._C"})


def python_module_probe(python: Path, modules: tuple[str, ...], *, pythonpath: list[Path], timeout: int) -> dict[str, Any]:
    """Probe module importability in an isolated subprocess under a given Python.

    Args:
        python: Path to the Python executable to probe.
        modules: Module names to check via ``importlib.util.find_spec``.
        pythonpath: Paths to prepend to ``PYTHONPATH`` for the probe subprocess.
        timeout: Maximum seconds to wait before treating the probe as timed out.

    Returns:
        A dict with ``python``, ``ok``, ``modules``, ``module_errors``, and
        ``stderr`` keys describing the importability status.
    """
    if not python.is_file():
        return {"python": str(python), "ok": False, "error": "python_not_found", "modules": {module: False for module in modules}}
    code = (
        "import importlib,importlib.util,json;"
        f"mods={json.dumps(list(modules))};"
        f"strict={json.dumps(sorted(STRICT_IMPORT_MODULES))};"
        f"torch_first={json.dumps(sorted(TORCH_FIRST_STRICT_IMPORT_MODULES))};"
        "found={};errors={};imported=[];"
        "\nfor m in mods:\n"
        "    try:\n"
        "        found[m]=importlib.util.find_spec(m) is not None\n"
        "    except Exception as exc:\n"
        "        found[m]=False;errors[m]=type(exc).__name__+': '+str(exc)\n"
        "    if found[m] and m in strict:\n"
        "        try:\n"
        "            importlib.import_module('torch') if m in torch_first else None\n"
        "            importlib.import_module(m);imported.append(m)\n"
        "        except Exception as exc:\n"
        "            found[m]=False;errors[m]=type(exc).__name__+': '+str(exc)\n"
        "print(json.dumps({'modules': found, 'errors': errors, 'imported': imported}, sort_keys=True))"
    )
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"python": str(python), "ok": False, "error": "timeout", "modules": {module: False for module in modules}}
    modules_result: dict[str, bool]
    module_errors: dict[str, str]
    try:
        parsed = json.loads((completed.stdout.strip().splitlines() or ["{}"])[-1])
        modules_result = parsed.get("modules", {}) if isinstance(parsed, dict) else {}
        module_errors = parsed.get("errors", {}) if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        modules_result = {module: False for module in modules}
        module_errors = {"__probe__": "json_decode_error"}
    return {
        "python": str(python),
        "ok": completed.returncode == 0 and all(modules_result.get(module) for module in modules),
        "probe": "find_spec+strict_import",
        "returncode": completed.returncode,
        "modules": modules_result,
        "strict_import_modules": sorted(STRICT_IMPORT_MODULES.intersection(modules)),
        "module_errors": module_errors,
        "stderr": completed.stderr.strip(),
    }


def torch_cuda_probe(python: Path, *, timeout: int, cuda_visible_devices: str = "0") -> dict[str, Any]:
    """Verify Torch CUDA functionality by running a minimal tensor computation.

    Args:
        python: Path to the Python executable with Torch installed.
        timeout: Maximum seconds before treating the probe as timed out.
        cuda_visible_devices: Value to set ``CUDA_VISIBLE_DEVICES`` for the probe.

    Returns:
        A dict with ``ok``, ``returncode``, ``timed_out``, ``payload``, and
        ``duration_seconds`` keys describing the CUDA probe outcome.
    """
    if not python.is_file():
        return {
            "python": str(python),
            "ok": False,
            "error": "python_not_found",
            "cuda_visible_devices": cuda_visible_devices,
        }

    code = r"""
import json
import torch

payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count_visible": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    tensor = torch.ones((1,), device="cuda") + 1
    torch.cuda.synchronize()
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["tensor_sum"] = float(tensor.detach().cpu().sum())
    payload["memory_allocated"] = torch.cuda.memory_allocated(0)
print(json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    start = time.monotonic()
    process = subprocess.Popen(
        [str(python), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    timed_out = False
    kill_stuck = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            kill_stuck = True
            stdout, stderr = "", ""
    duration_seconds = time.monotonic() - start
    payload: dict[str, Any] = {}
    if stdout.strip():
        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = {}
    tensor_sum = payload.get("tensor_sum")
    return {
        "python": str(python),
        "ok": (
            not timed_out
            and not kill_stuck
            and process.returncode == 0
            and payload.get("cuda_available") is True
            and tensor_sum == 2.0
        ),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "kill_stuck": kill_stuck,
        "duration_seconds": duration_seconds,
        "cuda_visible_devices": cuda_visible_devices,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "payload": payload,
    }


def cuda_driver_probe(*, device_index: int = 0, allocation_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    """Allocate and free GPU memory through libcuda without importing framework runtimes."""

    libcuda = ctypes.util.find_library("cuda") or "libcuda.so"
    try:
        cuda = ctypes.CDLL(libcuda)
    except OSError as exc:
        return {
            "ok": False,
            "probe": "cuda_driver",
            "error": f"{type(exc).__name__}: {exc}",
            "libcuda": libcuda,
            "device_index": device_index,
            "allocation_bytes": allocation_bytes,
        }

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cuMemAlloc_v2.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_void_p]
    cuda.cuMemFree_v2.restype = ctypes.c_int

    def call(name: str, *args: Any) -> int:
        return int(getattr(cuda, name)(*args))

    started = time.monotonic()
    context = ctypes.c_void_p()
    pointer = ctypes.c_void_p()
    failed: dict[str, Any] | None = None
    device_count = ctypes.c_int()
    device = ctypes.c_int()
    try:
        steps = (
            ("cuInit", lambda: call("cuInit", 0)),
            ("cuDeviceGetCount", lambda: call("cuDeviceGetCount", ctypes.byref(device_count))),
            ("cuDeviceGet", lambda: call("cuDeviceGet", ctypes.byref(device), device_index)),
            ("cuCtxCreate", lambda: call("cuCtxCreate_v2", ctypes.byref(context), 0, device.value)),
            ("cuMemAlloc", lambda: call("cuMemAlloc_v2", ctypes.byref(pointer), allocation_bytes)),
        )
        for step, func in steps:
            code = func()
            if code != 0:
                failed = {"step": step, "cuda_code": code}
                break
    finally:
        if pointer.value:
            cuda.cuMemFree_v2(pointer)
        if context.value:
            cuda.cuCtxDestroy_v2(context)

    return {
        "ok": failed is None,
        "probe": "cuda_driver",
        "libcuda": libcuda,
        "device_index": device_index,
        "device_count": device_count.value,
        "allocation_bytes": allocation_bytes,
        "duration_seconds": time.monotonic() - started,
        "failure": failed,
    }


def build_gpu_report(conda_root: Path, conda_envs_root: Path, timeout: int, *, probe_all: bool = False) -> dict[str, Any]:
    """Run Torch CUDA probes across known benchmark environments.

    Args:
        conda_root: Root directory for legacy benchmark conda environments.
        conda_envs_root: Root directory for WorldFoundry-managed conda environments.
        timeout: Per-probe timeout in seconds.
        probe_all: Whether to probe all candidates instead of stopping at the first success.

    Returns:
        A dict with ``ok``, ``selected_probe``, ``driver_ok``, ``benchmark_env_ok``, and
        per-environment ``probes`` entries.
    """
    candidates = {
        "benchmark_cu113": conda_envs_root / "worldfoundry-zeroscope-cu113" / "bin" / "python",
        "benchmark_worldplay": conda_root / "worldplay" / "bin" / "python",
        "benchmark_worldscore": conda_root / "worldscore" / "bin" / "python",
        "benchmark_cu113_animatediff": conda_envs_root / "worldfoundry-animatediff-official-cu113" / "bin" / "python",
        "benchmark_cu113_zeroscope": conda_envs_root / "worldfoundry-zeroscope-cu113" / "bin" / "python",
    }
    probes: dict[str, Any] = {}
    selected: str | None = None
    benchmark_env_ok = False
    for probe_id, python in candidates.items():
        if selected is not None and not probe_all:
            probes[probe_id] = {
                "python": str(python),
                "ok": None,
                "skipped": True,
                "reason": f"gpu probe already passed with {selected}",
            }
            continue
        probe = torch_cuda_probe(python, timeout=timeout)
        probes[probe_id] = probe
        if probe["ok"]:
            if selected is None:
                selected = probe_id
            if probe_id.startswith("benchmark_"):
                benchmark_env_ok = True
    driver_probe = cuda_driver_probe()
    return {
        "selected_probe": selected,
        "ok": selected is not None,
        "driver_ok": driver_probe["ok"],
        "driver_probe": driver_probe,
        "benchmark_env_ok": benchmark_env_ok,
        "probe_all": probe_all,
        "probes": probes,
    }


def build_environment_report(conda_root: Path, conda_envs_root: Path, model_root: Path, timeout: int) -> dict[str, Any]:
    """Build a module-importability report for all benchmark conda environments.

    Args:
        conda_root: Root directory for legacy benchmark conda environments.
        conda_envs_root: Root directory for WorldFoundry-managed conda environments.
        model_root: Root directory for model source repos (used in ``PYTHONPATH``).
        timeout: Per-probe timeout in seconds.

    Returns:
        A dict with ``conda_root``, ``conda_envs_root``, and per-environment
        ``envs`` entries describing module import status.
    """
    vbench_root = model_root / "VBench"
    specs = {
        "base_current": {
            "python": Path(sys.executable),
            "modules": ("numpy", "torch", "cv2", "decord"),
            "pythonpath": [REPO_ROOT],
        },
        "worldplay_vbench": {
            "python": conda_root / "worldplay" / "bin" / "python",
            "modules": ("torch", "torchvision", "numpy", "cv2", "decord", "sentencepiece", "vbench", "pkg_resources", "clip"),
            "pythonpath": [vbench_root, REPO_ROOT],
        },
        "worldscore": {
            "python": conda_root / "worldscore" / "bin" / "python",
            "modules": (
                "numpy",
                "torch",
                "cv2",
                "decord",
                "fire",
                "mmengine",
                "omegaconf",
                "structlog",
                "submitit",
                "pyiqa",
                "pytorch_lightning",
                "lietorch",
                "spacy",
                "cvxpy",
                "clip",
                "transformers",
                "huggingface_hub",
                "h5py",
                "pycocotools",
                "supervision",
                "yacs",
                "loguru",
                "einops",
                "timm",
                "imageio",
                "iopath",
                "torchmetrics",
                "mamba_ssm",
                "causal_conv1d",
                "droid_backends",
                "groundingdino._C",
                "sam2._C",
            ),
            "pythonpath": [model_root / "WorldScore", REPO_ROOT],
        },
        "worldplay_worldscore": {
            "python": conda_root / "worldplay" / "bin" / "python",
            "modules": (
                "numpy",
                "torch",
                "cv2",
                "decord",
                "fire",
                "mmengine",
                "omegaconf",
                "structlog",
                "submitit",
                "pyiqa",
                "pytorch_lightning",
                "lietorch",
                "spacy",
                "cvxpy",
                "clip",
                "transformers",
                "huggingface_hub",
                "h5py",
                "pycocotools",
                "supervision",
                "yacs",
                "loguru",
                "einops",
                "timm",
                "imageio",
                "iopath",
                "torchmetrics",
                "mamba_ssm",
                "causal_conv1d",
                "droid_backends",
                "groundingdino._C",
                "sam2._C",
            ),
            "pythonpath": [model_root / "WorldScore", REPO_ROOT],
        },
        "benchmark_cu113": {
            "python": conda_envs_root / "worldfoundry-zeroscope-cu113" / "bin" / "python",
            "modules": ("torch", "torchvision", "numpy", "cv2", "decord", "sentencepiece", "vbench", "pkg_resources", "clip", "fire", "mmengine", "omegaconf", "structlog", "submitit"),
            "pythonpath": [vbench_root, model_root / "WorldScore", REPO_ROOT],
        },
    }
    envs: dict[str, Any] = {}
    for env_id, spec in specs.items():
        envs[env_id] = python_module_probe(
            Path(spec["python"]),
            tuple(spec["modules"]),
            pythonpath=list(spec["pythonpath"]),
            timeout=timeout,
        )
    return {
        "conda_root": str(conda_root),
        "conda_envs_root": str(conda_envs_root),
        "envs": envs,
    }

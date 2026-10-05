from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from packaging.requirements import InvalidRequirement, Requirement
from PIL import Image

from worldfoundry.core.io.formats.serialization import write_json as _core_write_json
from worldfoundry.core.utils.python.import_guard import third_party_lazy_import_guard
from worldfoundry.runtime.compile_cache import configure_persistent_compile_cache
from worldfoundry.runtime.conda import RuntimeCondaEnvSpec, load_runtime_conda_env_specs_with_overrides
from worldfoundry.studio.inference.catalog import CatalogEntry, find_entry
from worldfoundry.studio.inference.config import lingbot_fast_sequence_parallel_enabled
from worldfoundry.studio.inference.paths import studio_workspace_root

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - imageio is optional for JSON-only action runs
    imageio = None

torch: Any | None = None

NAVIGATION_DELTAS: Dict[str, list[float]] = {
    "forward": [0.0, 0.0, -0.5, 0.0, 0.0],
    "backward": [0.0, 0.0, 0.5, 0.0, 0.0],
    "left": [-0.5, 0.0, 0.0, 0.0, 0.0],
    "right": [0.5, 0.0, 0.0, 0.0, 0.0],
    "forward_left": [-0.35, 0.0, -0.35, 0.0, 0.0],
    "forward_right": [0.35, 0.0, -0.35, 0.0, 0.0],
    "backward_left": [-0.35, 0.0, 0.35, 0.0, 0.0],
    "backward_right": [0.35, 0.0, 0.35, 0.0, 0.0],
    "camera_up": [0.0, 0.0, 0.0, -0.15, 0.0],
    "camera_down": [0.0, 0.0, 0.0, 0.15, 0.0],
    "camera_l": [0.0, 0.0, 0.0, 0.0, -0.15],
    "camera_r": [0.0, 0.0, 0.0, 0.0, 0.15],
    "camera_ul": [0.0, 0.0, 0.0, -0.1, -0.1],
    "camera_ur": [0.0, 0.0, 0.0, -0.1, 0.1],
    "camera_dl": [0.0, 0.0, 0.0, 0.1, -0.1],
    "camera_dr": [0.0, 0.0, 0.0, 0.1, 0.1],
    "camera_zoom_in": [0.0, 0.0, -0.3, 0.0, 0.0],
    "camera_zoom_out": [0.0, 0.0, 0.3, 0.0, 0.0],
}


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
SPLAT_EXTS = {".spz", ".splat", ".ksplat", ".sog"}
MESH_EXTS = {".obj", ".glb", ".gltf", ".stl"}
MODEL_EXTS = {".ply", *MESH_EXTS}
LINGBOT_WORLD_MODEL_ID = "lingbot-world"
LINGBOT_WORLD_V2_MODEL_ID = "lingbot-world-v2"
MATRIX_GAME3_MODEL_ID = "matrix-game-3"
LONGVIE2_MODEL_ID = "longvie-2"
HELIOS_MODEL_ID = "helios"
DREAMX_WORLD_MODEL_ID = "dreamx-world-5b-cam"
LINGBOT_VARIANT_FAST = "fast"
TORCHRUN_LINGBOT_FAST_ENV = "WORLDFOUNDRY_STUDIO_TORCHRUN_LINGBOT_FAST"
TORCHRUN_DISTRIBUTED_ENV = "WORLDFOUNDRY_STUDIO_TORCHRUN_DISTRIBUTED"
STUDIO_CONDA_CHILD_ENV = "WORLDFOUNDRY_STUDIO_CONDA_CHILD"
TORCH_COMPILE_ENV_MODELS = {"matrix-game-2"}
RUNTIME_CHECKS_ENV = "WORLDFOUNDRY_STUDIO_RUNTIME_CHECKS"
# Lazy torchrun command-channel group. PyTorch group creation/destruction may
# block in collectives, so the condition only protects lifecycle transitions;
# those calls themselves must happen without holding its lock.
_TORCHRUN_CONTROL_GROUP: Any = None
_TORCHRUN_CONTROL_GROUP_CONDITION = threading.Condition()
_TORCHRUN_CONTROL_GROUP_STATE = "open"
_TORCHRUN_CONTROL_GROUP_GENERATION = 0
_TORCHRUN_CONTROL_GROUP_CREATING = False
# Model families whose input_path should never be bound as video even if the
# file extension is a video format. CameraCtrl, for example, uses input_path
# as a reference for camera trajectory extraction — its predict() raises
# ValueError if `video` is passed.
_NO_VIDEO_BIND_FAMILIES = {"cameractrl"}
PREFERRED_PREVIEW_PREFIXES = (
    "preview",
    "first_frame",
    "stream_",
    "render_",
    "orbit_",
    "trajectory_",
    "generated_",
    "output_",
)
ARTIFACT_SCAN_MODE_ENV = "WORLDFOUNDRY_STUDIO_ARTIFACT_SCAN_MODE"
PREVIEW_VIDEO_VALIDATE_ENV = "WORLDFOUNDRY_STUDIO_VALIDATE_PREVIEW_VIDEO"
VIDEO_PREVIEW_IMAGE_ENV = "WORLDFOUNDRY_STUDIO_EXTRACT_VIDEO_PREVIEW_IMAGE"
RERUN_PREVIEW_ENV = "WORLDFOUNDRY_STUDIO_BUILD_RERUN_PREVIEW"
PREVIEW_ASSET_EXTS = {*IMAGE_EXTS, *VIDEO_EXTS, *SPLAT_EXTS, *MODEL_EXTS, ".rrd"}
VIEWPORT_ARTIFACT_EXTS = {
    *PREVIEW_ASSET_EXTS,
    ".json",
    ".jsonl",
    ".npz",
    ".npy",
    ".pkl",
    ".yaml",
    ".yml",
    ".toml",
}


def _output_suffix_from_infer_metadata(entry: CatalogEntry, infer_metadata: Optional[Dict[str, Any]]) -> str:
    task = infer_metadata.get("task") if isinstance(infer_metadata, Mapping) else None
    outputs = task.get("outputs") if isinstance(task, Mapping) else None
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes, bytearray)):
        kinds = {
            str(item.get("kind") or item.get("artifact_id") or "").strip().lower()
            for item in outputs
            if isinstance(item, Mapping)
        }
        if kinds & {"image", "generated_image"}:
            return ".png"
        if kinds & {"video", "generated_video"}:
            return ".mp4"
        if entry.model_id in {"lagernvs", "stable-virtual-camera", "wonderjourney"}:
            return ".mp4"
        if entry.model_id == "wonderworld":
            return ".splat"
        if kinds & {"splat", "gaussian_splat"}:
            return ".splat"
        if kinds & {"model", "mesh", "3d", "3d_asset", "generated_3d_asset", "scene"}:
            return ".ply"
        if kinds & {"geometry", "depth", "point_cloud", "pointcloud"}:
            return ".npz"
        if kinds & {"action_trace", "action_tokens"}:
            return ".json"
    return ".json" if entry.category in {"Embodied Action", "Visual Action"} else ".mp4"


def _input_path_should_bind_as_video(input_path: str | None, names: set[str]) -> bool:
    if not input_path:
        return False
    suffix = Path(str(input_path)).suffix.lower()
    if suffix in VIDEO_EXTS:
        return True
    if suffix in IMAGE_EXTS:
        return False
    try:
        if Path(str(input_path)).expanduser().is_dir():
            return False
    except (OSError, ValueError):
        pass
    image_like_params = {"images", "image_path", "input_data"}
    video_like_params = {"videos", "video", "video_path"}
    return not (names & image_like_params) and bool(names & video_like_params)
MODEL_RUNTIME_EXTRA_HINTS = {
    "vggt": "repair the resolved VGGT conda env from its runtime manifest",
    "vggt-omega": "repair the resolved VGGT-Omega conda env from runtime/environments/three_d/vggt-omega.yaml",
}
_PACKAGE_IMPORT_OVERRIDES = {
    "controlnet-aux": "controlnet_aux",
    "hydra-core": "hydra",
    "json-numpy": "json_numpy",
    "nvidia-ml-py": "pynvml",
    "open-clip-torch": "open_clip",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "pytorch-lightning": "pytorch_lightning",
}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _studio_runtime_checks_enabled() -> bool:
    value = os.getenv(RUNTIME_CHECKS_ENV, "skip").strip().lower()
    return value in {"1", "true", "yes", "on", "strict", "preflight"}


def _torch_module() -> Any | None:
    global torch
    if torch is not None:
        return torch
    try:
        import torch as loaded_torch
    except Exception:  # pragma: no cover - torch import is optional for light tests
        return None
    torch = loaded_torch
    return torch


class _temporary_env:
    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = dict(values)
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for key, previous in self.previous.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _torch_dist() -> Any:
    torch = _torch_module()
    if torch is None:
        return None
    try:
        return torch.distributed
    except Exception:
        return None


def _torchrun_world_size() -> int:
    try:
        return max(int(os.getenv("WORLD_SIZE", "1") or "1"), 1)
    except Exception:
        return 1


def _torchrun_rank() -> int:
    try:
        return max(int(os.getenv("RANK", "0") or "0"), 0)
    except Exception:
        return 0




def _torchrun_cuda_device_index(torch_module: Any) -> int:
    """Map LOCAL_RANK without silently collapsing several ranks onto one GPU."""

    device_count = int(torch_module.cuda.device_count())
    if device_count < 1:
        raise RuntimeError("torchrun requested CUDA execution but no CUDA device is visible")
    raw_local_rank = os.getenv("LOCAL_RANK", str(_torchrun_rank()))
    try:
        local_rank = int(raw_local_rank or "0")
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be an integer, got {raw_local_rank!r}.") from exc
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}.")
    if local_rank < device_count:
        return local_rank

    # Some schedulers expose one already-remapped GPU to each independently
    # launched process while retaining a node-local rank greater than zero.
    # That layout is safe.  A torch.distributed.run/elastic parent, however,
    # gives every child the same visibility mask and must fail closed instead
    # of mapping ranks 1..N to cuda:0.
    try:
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1") or "1")
    except ValueError:
        local_world_size = 1
    elastic_launch = bool(os.getenv("TORCHELASTIC_RUN_ID", "").strip())
    if device_count == 1 and not elastic_launch:
        return 0
    raise RuntimeError(
        f"LOCAL_RANK={local_rank} is invalid for {device_count} visible CUDA device(s) "
        f"with LOCAL_WORLD_SIZE={local_world_size} (elastic_launch={elastic_launch})."
    )


def _torchrun_lingbot_fast_enabled() -> bool:
    return (
        _env_flag(TORCHRUN_LINGBOT_FAST_ENV)
        and _torchrun_world_size() > 1
    )


@lru_cache(maxsize=1)
def _runtime_conda_env_specs() -> Mapping[str, RuntimeCondaEnvSpec]:
    return load_runtime_conda_env_specs_with_overrides()


def _runtime_conda_env_spec(model_id: str) -> RuntimeCondaEnvSpec | None:
    return _runtime_conda_env_specs().get(model_id)


def _same_executable(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return str(left) == str(right)


def _should_check_runtime_imports_in_current_process(spec: RuntimeCondaEnvSpec | None) -> bool:
    if spec is None:
        return False
    if _env_flag(STUDIO_CONDA_CHILD_ENV):
        return True
    if not spec.exists:
        return True
    return _same_executable(sys.executable, spec.python_executable)


def _normalise_package_key(value: str) -> str:
    return str(value).strip().replace("_", "-").lower()


def _requirement_import_name(requirement: Requirement) -> str:
    package_key = _normalise_package_key(requirement.name)
    return _PACKAGE_IMPORT_OVERRIDES.get(package_key, package_key.replace("-", "_"))


def _direct_import_name(value: str) -> str:
    return str(value).strip().split(";", maxsplit=1)[0].strip()


def _validation_import_missing(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    direct_name = _direct_import_name(text)
    if direct_name and direct_name == text:
        try:
            if importlib.util.find_spec(direct_name) is not None:
                return None
        except (ImportError, AttributeError, ValueError):
            pass
    try:
        requirement = Requirement(text)
    except InvalidRequirement:
        import_name = _direct_import_name(text)
        try:
            return None if importlib.util.find_spec(import_name) is not None else text
        except (ImportError, AttributeError, ValueError):
            return text

    import_name = _requirement_import_name(requirement)
    try:
        importable = importlib.util.find_spec(import_name) is not None
    except (ImportError, AttributeError, ValueError):
        importable = False
    if not importable:
        return text

    if requirement.specifier:
        try:
            installed_version = importlib_metadata.version(requirement.name)
        except importlib_metadata.PackageNotFoundError:
            try:
                installed_version = importlib_metadata.version(import_name)
            except importlib_metadata.PackageNotFoundError:
                return text
        if installed_version not in requirement.specifier:
            return f"{text} (installed {installed_version})"
    return None


def _missing_runtime_validation_imports(entry: CatalogEntry) -> tuple[str, ...]:
    spec = _runtime_conda_env_spec(entry.model_id)
    if not _should_check_runtime_imports_in_current_process(spec):
        return ()
    missing: list[str] = []
    for item in spec.validation_imports:
        missing_item = _validation_import_missing(str(item))
        if missing_item and missing_item not in missing:
            missing.append(missing_item)
    return tuple(missing)


def _runtime_dependency_error(entry: CatalogEntry, missing: Sequence[str]) -> RuntimeError:
    spec = _runtime_conda_env_spec(entry.model_id)
    install_hint = MODEL_RUNTIME_EXTRA_HINTS.get(
        entry.model_id,
        "repair the resolved per-model conda env from its runtime manifest",
    )
    details = [
        f"{entry.display_name} runtime is missing Python modules in the active runtime process: "
        f"{', '.join(missing)}.",
        f"Configure the model conda environment instead of bypassing imports: {install_hint}.",
    ]
    if spec is not None:
        details.append(
            "Configured runtime env: "
            f"{spec.env_name} ({spec.python_executable}, exists={str(spec.exists).lower()})."
        )
    return RuntimeError(" ".join(details))


def _request_model_ref(entry: CatalogEntry, request: "PreparedInputs") -> str:
    return (request.model_ref or entry.default_model_ref or "").strip()


def _is_explicit_local_ref(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "://" in text:
        return False
    if text.startswith(("~", ".")):
        return True
    return Path(text).expanduser().is_absolute()


def _missing_local_model_ref_error(entry: CatalogEntry, model_ref: str) -> RuntimeError:
    env_key = "WORLDFOUNDRY_STUDIO_MODEL_REF_" + "".join(
        char if char.isalnum() else "_"
        for char in entry.model_id.upper()
    ).strip("_")
    return RuntimeError(
        f"{entry.display_name} default model_ref points to a missing local path: {model_ref}. "
        f"Stage the checkpoint there, pass a valid model_ref in the job, or set {env_key} to a checkpoint file/directory."
    )


def ensure_torchrun_lingbot_fast_runtime() -> bool:
    dist = _torch_dist()
    if not _torchrun_lingbot_fast_enabled() or dist is None or not dist.is_available():
        return False

    torch = _torch_module()
    if torch is not None and torch.cuda.is_available():
        local_rank = _torchrun_cuda_device_index(torch)
        try:
            torch.cuda.set_device(local_rank)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to select cuda:{local_rank} for LOCAL_RANK={os.getenv('LOCAL_RANK', '0')}."
            ) from exc

    if not dist.is_initialized():
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("LingBot torchrun requires CUDA before initializing the NCCL process group.")
        dist.init_process_group(backend="nccl", init_method="env://")
    actual_world_size = int(dist.get_world_size())
    actual_rank = int(dist.get_rank())
    if actual_world_size != _torchrun_world_size() or actual_rank != _torchrun_rank():
        raise RuntimeError(
            "LingBot command bridge environment does not match the initialized process group: "
            f"env RANK/WORLD_SIZE={_torchrun_rank()}/{_torchrun_world_size()}, "
            f"group rank/world_size={actual_rank}/{actual_world_size}."
        )
    return dist.is_initialized()


def _torchrun_control_group() -> Any:
    """CPU process group for command messages while model tensors stay on NCCL.

    Nonzero resident ranks intentionally block in a broadcast while the UI is
    idle. Gloo's 30-minute default timeout treats that healthy idle state as a
    failed collective, so give the command channel a service-lifetime timeout.
    Model collectives keep their normal NCCL timeout and failure detection.
    """
    global _TORCHRUN_CONTROL_GROUP, _TORCHRUN_CONTROL_GROUP_CREATING

    if not _torchrun_lingbot_fast_enabled() or _torchrun_world_size() <= 1:
        return None

    with _TORCHRUN_CONTROL_GROUP_CONDITION:
        while True:
            if _TORCHRUN_CONTROL_GROUP_STATE != "open":
                return None
            if _TORCHRUN_CONTROL_GROUP is not None:
                return _TORCHRUN_CONTROL_GROUP
            if not _TORCHRUN_CONTROL_GROUP_CREATING:
                _TORCHRUN_CONTROL_GROUP_CREATING = True
                generation = _TORCHRUN_CONTROL_GROUP_GENERATION
                break
            _TORCHRUN_CONTROL_GROUP_CONDITION.wait()

    created_group = None
    try:
        # Re-check after claiming creation. Shutdown waits for this claim to be
        # released, while no lifecycle lock is held around PyTorch callbacks.
        dist = _torch_dist()
        if (
            dist is None
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return None
        created_group = dist.new_group(
            ranks=list(range(_torchrun_world_size())),
            backend="gloo",
            timeout=timedelta(days=365),
        )
    finally:
        with _TORCHRUN_CONTROL_GROUP_CONDITION:
            _TORCHRUN_CONTROL_GROUP_CREATING = False
            if created_group is not None:
                # A shutdown that advanced the generation owns this stale
                # result. Publish it only so shutdown can destroy it before the
                # default process group, never so a caller can use it.
                _TORCHRUN_CONTROL_GROUP = created_group
            publish = (
                created_group is not None
                and _TORCHRUN_CONTROL_GROUP_STATE == "open"
                and _TORCHRUN_CONTROL_GROUP_GENERATION == generation
            )
            _TORCHRUN_CONTROL_GROUP_CONDITION.notify_all()
    return created_group if publish else None


def ensure_torchrun_lingbot_fast_control_group() -> bool:
    if not ensure_torchrun_lingbot_fast_runtime():
        return False
    return _torchrun_control_group() is not None


@lru_cache(maxsize=1)
def _torchrun_min_gpu_vram_gib() -> float | None:
    """Return the minimum visible GPU memory across all torchrun ranks."""

    torch = _torch_module()
    if torch is None or not torch.cuda.is_available():
        return None
    local_bytes = 0
    try:
        device_index = _torchrun_cuda_device_index(torch)
        local_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Preserve collective ordering: a failed rank contributes a zero
        # sentinel so every rank selects the conservative FSDP policy.
        local_bytes = 0
    dist = _torch_dist()
    control_group = _torchrun_control_group()
    if (
        dist is not None
        and dist.is_initialized()
        and control_group is not None
        and _torchrun_world_size() > 1
    ):
        memory = torch.tensor(local_bytes, dtype=torch.int64, device="cpu")
        dist.all_reduce(memory, op=dist.ReduceOp.MIN, group=control_group)
        local_bytes = int(memory.item())
    return local_bytes / (1024**3) if local_bytes > 0 else None


def shutdown_torchrun_lingbot_fast_runtime() -> None:
    global _TORCHRUN_CONTROL_GROUP
    global _TORCHRUN_CONTROL_GROUP_GENERATION, _TORCHRUN_CONTROL_GROUP_STATE

    _torchrun_min_gpu_vram_gib.cache_clear()
    with _TORCHRUN_CONTROL_GROUP_CONDITION:
        if _TORCHRUN_CONTROL_GROUP_STATE == "closing":
            closing_generation = _TORCHRUN_CONTROL_GROUP_GENERATION
            _TORCHRUN_CONTROL_GROUP_CONDITION.wait_for(
                lambda: (
                    _TORCHRUN_CONTROL_GROUP_STATE != "closing"
                    or _TORCHRUN_CONTROL_GROUP_GENERATION != closing_generation
                )
            )
            return
        _TORCHRUN_CONTROL_GROUP_STATE = "closing"
        _TORCHRUN_CONTROL_GROUP_GENERATION += 1
        _TORCHRUN_CONTROL_GROUP_CONDITION.notify_all()
        _TORCHRUN_CONTROL_GROUP_CONDITION.wait_for(
            lambda: not _TORCHRUN_CONTROL_GROUP_CREATING
        )
        control_group = _TORCHRUN_CONTROL_GROUP
        _TORCHRUN_CONTROL_GROUP = None
    try:
        dist = _torch_dist()
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return
        if control_group is not None:
            try:
                dist.destroy_process_group(control_group)
            except Exception:
                pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    finally:
        # Do not reopen creation until both the control and default groups have
        # finished destruction. A future ensure call may then initialize a new
        # runtime generation safely.
        with _TORCHRUN_CONTROL_GROUP_CONDITION:
            _TORCHRUN_CONTROL_GROUP_STATE = "open"
            _TORCHRUN_CONTROL_GROUP_CONDITION.notify_all()


def _is_gaussian_splat_ply(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(16384).decode("latin-1", errors="ignore")
    except Exception:
        return False

    marker = "end_header"
    if marker in header:
        header = header.split(marker, maxsplit=1)[0]
    lower = header.lower()
    if not lower.lstrip().startswith("ply"):
        return False
    return (
        "property float opacity" in lower
        and "property float scale_0" in lower
        and "property float f_dc_0" in lower
        and "property float rot_0" in lower
    )


def parse_jsonish(text: str, default: Any = None) -> Any:
    if text is None:
        return default
    stripped = str(text).strip()
    if not stripped:
        return default
    return json.loads(stripped)


def _split_interaction_tokens(text: str) -> list[str]:
    tokens = []
    for line in str(text).splitlines():
        for part in line.split(","):
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens


def parse_interactions(text: str) -> Any:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, str):
            return _split_interaction_tokens(parsed) or None
        return parsed
    except json.JSONDecodeError:
        pass
    return _split_interaction_tokens(stripped) or None


def _interaction_tokens_summary(interactions: Any, *, max_tokens: int = 14) -> str:
    """Fold arbitrary interactions payloads into a compact token-ish comma phrase."""
    if interactions is None:
        return "—"
    if isinstance(interactions, (list, tuple)):
        flattened: List[str] = []
        for item in interactions:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                flattened.append(text)
        if not flattened:
            return "—"
        if len(flattened) <= max_tokens:
            return ", ".join(flattened)
        head = flattened[:max_tokens]
        return ", ".join(head) + f" (+{len(flattened) - max_tokens} more)"
    if isinstance(interactions, dict):
        payload = _safe_json(interactions)
        dumped = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(dumped) > 240:
            return dumped[:237] + "…"
        return dumped or "—"
    text = str(interactions).strip()
    return text if text else "—"


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "__dict__"):
        return _safe_json(vars(value))
    return repr(value)


def _redact_manifest_secrets(value: Any) -> Any:
    """Redact nested credentials without adding evaluation to Studio import time."""

    from worldfoundry.evaluation.reporting.run_manifest import redact_secrets

    return redact_secrets(_safe_json(value))


def _status_from_model_result(result: Any) -> str:
    if isinstance(result, Mapping):
        status = str(result.get("status") or "").strip().lower()
        if status in {"blocked", "cancelled", "failed"}:
            return status
        if status == "error":
            return "failed"
    return "succeeded"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")


def _hash_payload(payload: Dict[str, Any]) -> str:
    serial = json.dumps(_safe_json(payload), sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serial.encode("utf-8")).hexdigest()[:12]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_file(src: str | Path, dst: Path) -> str:
    src_path = Path(src).expanduser().resolve()
    _ensure_dir(dst.parent)
    shutil.copy2(src_path, dst)
    return str(dst)


def _save_pil(image: Optional[Image.Image], dst: Path) -> Optional[str]:
    if image is None:
        return None
    _ensure_dir(dst.parent)
    image.save(dst)
    return str(dst)


def _extract_file_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "path", "video", "file"):
            if key in value and isinstance(value[key], str):
                return value[key]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return None


def _extract_multi_paths(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        return [str(values)]
    if isinstance(values, dict):
        path = _extract_file_path(values)
        return [path] if path else []
    paths = []
    for item in values:
        path = _extract_file_path(item)
        if path:
            paths.append(path)
    return paths


def build_worldfm_pose_sequence(interactions: Sequence[str]) -> list[list[list[float]]]:
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    for token in interactions:
        delta = NAVIGATION_DELTAS.get(str(token))
        if delta is None:
            raise ValueError(
                f"WorldFM can convert only navigation tokens into poses. Unsupported token: {token}"
            )
        current = apply_camera_delta(current, delta)
        poses.append(current.copy())
    return [pose.tolist() for pose in poses]


def apply_camera_delta(c2w: np.ndarray, delta: Sequence[float]) -> np.ndarray:
    dx, dy, dz, theta_x, theta_z = [float(item) for item in delta]
    result = np.array(c2w, dtype=np.float64).copy()

    rx = np.eye(4, dtype=np.float64)
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    rx[1, 1], rx[1, 2] = cx, -sx
    rx[2, 1], rx[2, 2] = sx, cx

    rz = np.eye(4, dtype=np.float64)
    cz, sz = math.cos(theta_z), math.sin(theta_z)
    rz[0, 0], rz[0, 1] = cz, -sz
    rz[1, 0], rz[1, 1] = sz, cz

    result = result @ rx @ rz
    result[0, 3] += dx
    result[1, 3] += dy
    result[2, 3] += dz
    return result


def guess_intrinsics(image_path: str) -> list[list[float]]:
    with Image.open(image_path) as image:
        width, height = image.size
    focal = float(max(width, height))
    return [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ]


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Move a tensor to CPU and bridge dtypes unsupported by NumPy."""
    tensor = tensor.detach().cpu()
    try:
        return tensor.numpy()
    except TypeError:
        if tensor.is_floating_point():
            return tensor.float().numpy()
        raise


def _to_uint8_rgb(frame: Any) -> np.ndarray:
    torch = _torch_module()
    if isinstance(frame, Image.Image):
        arr = np.array(frame.convert("RGB"))
    elif torch is not None and isinstance(frame, torch.Tensor):
        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 0)
        elif tensor.ndim == 4 and tensor.shape[1] in {1, 3, 4}:
            tensor = tensor.permute(0, 2, 3, 1)
        arr = _tensor_to_numpy(tensor)
    else:
        arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        min_value = float(arr.min()) if arr.size else 0.0
        max_value = float(arr.max()) if arr.size else 0.0
        if min_value < 0.0 and max_value <= 1.0:
            arr = (arr + 1.0) * 127.5
        elif max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _normalize_frame_list(value: Any) -> Optional[list[np.ndarray]]:
    if value is None:
        return None
    if hasattr(value, "videos"):
        sr_videos = getattr(value, "sr_videos", None)
        return _normalize_frame_list(sr_videos if sr_videos is not None else getattr(value, "videos"))
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        try:
            torch = _torch_module()
            arrays = [
                _tensor_to_numpy(item) if torch is not None and isinstance(item, torch.Tensor) else np.asarray(item)
                for item in value
            ]
            shapes = {tuple(arr.shape) for arr in arrays}
            if (
                len(value) in {1, 3, 4}
                and len(shapes) == 1
                and all(arr.ndim == 3 and arr.shape[0] not in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4} for arr in arrays)
            ):
                stacked = np.stack(arrays, axis=-1)
                return [_to_uint8_rgb(frame) for frame in stacked]
            return [_to_uint8_rgb(item) for item in value]
        except Exception:
            return None
    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4:
            # Prefer an explicit RGB tail unless dimension 1 is also RGB.
            # The latter is the ambiguous TCHW case where W happens to be 3.
            # Checking the leading dimension first misclassifies common THWC
            # chunks containing exactly 1, 3, or 4 frames.
            if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
                pass
            elif tensor.shape[1] in {1, 3, 4}:
                tensor = tensor.permute(0, 2, 3, 1)
            elif tensor.shape[0] in {1, 3, 4}:
                tensor = tensor.permute(1, 2, 3, 0)
            elif tensor.shape[-1] not in {1, 3, 4}:
                return None
            # Layout is explicitly frame-major HWC at this point. Bridge each
            # frame to NumPy first so small heights such as 3 or 4 are not
            # reinterpreted as CHW by the generic single-frame helper.
            return [_to_uint8_rgb(_tensor_to_numpy(frame)) for frame in tensor]
        return None
    if isinstance(value, np.ndarray) and value.ndim == 5:
        value = value[0]
    if isinstance(value, np.ndarray) and value.ndim == 4:
        if value.shape[-1] in {1, 3, 4}:
            return [_to_uint8_rgb(frame) for frame in value]
        if value.shape[1] in {1, 3, 4}:
            return [_to_uint8_rgb(frame.transpose(1, 2, 0)) for frame in value]
    return None


def _save_preview_image_sequence(
    value: Any,
    output_dir: str,
    stem: str,
    *,
    max_images: int = 8,
) -> list[str]:
    if value is None:
        return []
    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        value = _tensor_to_numpy(value)

    frames: list[Any]
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            frames = [value[index] for index in range(min(value.shape[0], max_images))]
        elif value.ndim in {2, 3}:
            frames = [value]
        else:
            return []
    elif isinstance(value, (list, tuple)):
        frames = list(value[:max_images])
    else:
        return []

    saved: list[str] = []
    root = _ensure_dir(Path(output_dir))
    for index, frame in enumerate(frames):
        try:
            image = Image.fromarray(_to_uint8_rgb(frame))
        except Exception:
            continue
        path = root / f"{stem}_{index:03d}.png"
        image.save(path)
        saved.append(str(path))
    return saved


def _coerce_video_chunk(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray) and value.ndim == 4:
        return value
    frames = _normalize_frame_list(value)
    if not frames:
        return None
    return np.stack([frame.astype(np.float32) / 255.0 for frame in frames], axis=0)


def _supports_memory_stream(pipeline: Any) -> bool:
    memory = getattr(pipeline, "memory_module", None)
    stream = getattr(pipeline, "stream", None)
    return (
        memory is not None
        and stream is not None
        and hasattr(memory, "manage")
        and hasattr(memory, "record")
        and "images" in _signature_names(stream)
    )


def _pipeline_stream_state_ready(pipeline: Any) -> bool:
    ready = getattr(pipeline, "stream_state_ready", None)
    if callable(ready):
        try:
            return bool(ready())
        except Exception:
            return False
    return False


def export_frames_to_video(frames: Sequence[Any], output_path: str, fps: int = 16) -> str:
    if imageio is None:
        raise RuntimeError("imageio is required to export video previews.")
    frame_list = [_to_uint8_rgb(frame) for frame in frames]
    if not frame_list:
        raise RuntimeError("No frames available for video export.")
    output = Path(output_path)
    if output.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif"}:
        output = output.with_suffix(".mp4")
    _ensure_dir(output.parent)
    if len(frame_list) == 1:
        frame_list = [frame_list[0], frame_list[0]]
    imageio.mimsave(output, frame_list, fps=fps)
    return str(output)


def collect_artifact_paths(output_dir: str) -> list[str]:
    root = Path(output_dir)
    if not root.exists():
        return []
    artifacts = []
    for path in root.rglob("*"):
        if path.is_file():
            artifacts.append(str(path))
    return artifacts


def _existing_artifact_paths(paths: Sequence[str | Path | None], *, check_files: bool = True) -> list[str]:
    seen: set[str] = set()
    artifacts: list[str] = []
    for raw_path in paths:
        if not raw_path:
            continue
        try:
            path = Path(raw_path)
        except (TypeError, ValueError):
            continue
        if check_files:
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
        key = path.expanduser().as_posix()
        if key in seen:
            continue
        seen.add(key)
        artifacts.append(str(path))
    return artifacts


def _persist_video_artifact_in_run(
    path: str | Path,
    *,
    output_dir: str | Path,
    output_path: str | Path,
) -> str:
    """Copy an encoded video returned from a temporary directory into its Studio run."""
    source = Path(path).expanduser()
    if source.suffix.lower() not in VIDEO_EXTS or not source.is_file():
        return str(source)

    source_resolved = source.resolve()
    run_root = Path(output_dir).expanduser().resolve()
    if source_resolved.is_relative_to(run_root):
        return str(source)

    target = Path(output_path).expanduser()
    if target.suffix.lower() not in VIDEO_EXTS:
        target = target.with_suffix(source.suffix or ".mp4")
    if source_resolved == target.resolve():
        return str(target)
    return _copy_file(source_resolved, target)


def _artifact_scan_mode() -> str:
    mode = os.getenv(ARTIFACT_SCAN_MODE_ENV, "missing").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return "off"
    if mode in {"1", "true", "yes", "on", "all"}:
        return "always"
    if mode in {"always", "missing"}:
        return mode
    return "missing"


def _should_scan_artifacts(
    *,
    request: "PreparedInputs",
    saved_artifacts: Sequence[str],
    previews: Mapping[str, Any],
) -> bool:
    mode = _artifact_scan_mode()
    if mode == "always":
        return True
    if mode == "off":
        return False
    if _missing_required_inference_outputs(request, previews):
        return True
    meaningful = [
        path
        for path in saved_artifacts
        if Path(path).name not in {"result_metadata.json", "manifest.json"}
    ]
    return not meaningful


def _load_json_payload(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _has_action_trace_payload(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        if str(payload.get("artifact_kind") or "").strip().lower() == "action_trace":
            return True
        if any(key in payload for key in ("action", "actions", "prediction", "latent_action_tokens")):
            return True
    if isinstance(payload, list):
        return bool(payload)
    return False


def _write_canonical_action_trace(output_dir: str, result: Mapping[str, Any]) -> str | None:
    target = Path(output_dir) / "actions.json"
    if target.exists() and _has_action_trace_payload(_load_json_payload(target)):
        return str(target)

    payload: Any | None = None
    artifact_path = result.get("artifact_path")
    if isinstance(artifact_path, str):
        source = Path(artifact_path)
        if source.exists() and source.suffix.lower() == ".json":
            payload = _load_json_payload(source)
    if not _has_action_trace_payload(payload):
        payload = result
    if not _has_action_trace_payload(payload):
        return None

    _core_write_json(target, _safe_json(payload), atomic=True)
    return str(target)


def _artifact_is_input(path: str | Path | None) -> bool:
    if not path:
        return False
    return any(part.lower() == "inputs" for part in Path(path).parts)


def _artifact_priority(path: str) -> tuple[int, int, int, int, str]:
    artifact_path = Path(path)
    name = artifact_path.name.lower()
    ext = artifact_path.suffix.lower()
    is_input = 1 if _artifact_is_input(artifact_path) else 0
    media_rank = 0 if ext in VIDEO_EXTS else 1
    try:
        video_size_rank = -artifact_path.stat().st_size if ext in VIDEO_EXTS else 0
    except OSError:
        video_size_rank = 0
    prefix_rank = next(
        (index for index, prefix in enumerate(PREFERRED_PREVIEW_PREFIXES) if name.startswith(prefix)),
        len(PREFERRED_PREVIEW_PREFIXES),
    )
    return (
        is_input,
        media_rank,
        video_size_rank,
        prefix_rank,
        name,
    )


def _ordered_artifacts_for_preview(artifacts: Sequence[str]) -> list[str]:
    seen = set()
    unique = []
    for path in artifacts:
        if not path:
            continue
        normalized = str(path)
        if os.path.splitext(normalized)[1].lower() not in PREVIEW_ASSET_EXTS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return sorted(unique, key=_artifact_priority)


def _viewport_artifact_paths(artifacts: Sequence[str]) -> list[str]:
    return [
        path
        for path in artifacts
        if os.path.splitext(path)[1].lower() in VIEWPORT_ARTIFACT_EXTS
    ]


def _is_decodable_video(path: str) -> bool:
    video_path = Path(path)
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        return False
    if imageio is None:
        return True
    try:
        reader = imageio.get_reader(str(video_path))
        try:
            reader.get_data(0)
        finally:
            reader.close()
        return True
    except Exception:
        return False


def pick_preview_assets(artifacts: Sequence[str], *, validate_videos: bool = True) -> Dict[str, Any]:
    preview_video = None
    preview_image = None
    preview_splat = None
    preview_model = None
    gallery: list[str] = []
    rrd_path = None
    gaussian_ply_candidate = None
    for path in _ordered_artifacts_for_preview(artifacts):
        ext = Path(path).suffix.lower()
        if ext in VIDEO_EXTS and preview_video is None:
            if not validate_videos or _is_decodable_video(path):
                preview_video = path
        elif ext in IMAGE_EXTS:
            gallery.append(path)
            if preview_image is None:
                preview_image = path
        elif preview_splat is None and ext in SPLAT_EXTS:
            preview_splat = path
        elif gaussian_ply_candidate is None and ext == ".ply" and _is_gaussian_splat_ply(path):
            gaussian_ply_candidate = path
        elif ext in MODEL_EXTS and preview_model is None:
            preview_model = path
        elif ext == ".rrd" and rrd_path is None:
            rrd_path = path
    if preview_splat is None:
        preview_splat = gaussian_ply_candidate
    return {
        "preview_video": preview_video,
        "preview_image": preview_image,
        "preview_splat": preview_splat,
        "preview_model": preview_model,
        "gallery": gallery[:18],
        "rrd_path": rrd_path,
    }


def maybe_extract_video_preview_image(
    video_path: Optional[str],
    output_dir: str,
    *,
    frame_position: str = "first",
    output_name: str = "preview.png",
) -> Optional[str]:
    if not video_path:
        return None
    if imageio is None:
        return None

    try:
        reader = imageio.get_reader(video_path)
        try:
            if frame_position == "last":
                frame = None
                frame_count = None
                try:
                    frame_count = int(reader.count_frames())
                except Exception:
                    try:
                        length = reader.get_length()
                        if isinstance(length, (int, float)) and math.isfinite(length):
                            frame_count = int(length)
                    except Exception:
                        frame_count = None
                if frame_count and frame_count > 0:
                    frame = reader.get_data(frame_count - 1)
                else:
                    for candidate in reader:
                        frame = candidate
                if frame is None:
                    return None
            else:
                frame = reader.get_data(0)
        finally:
            reader.close()
    except Exception:
        return None

    filename = Path(output_name).name or "preview.png"
    preview_path = Path(output_dir) / filename
    Image.fromarray(_to_uint8_rgb(frame)).save(preview_path)
    return str(preview_path)


_FIRST_FRAME_FILENAMES = (
    "first_frame.png",
    "first_frame.jpg",
    "first_frame.jpeg",
    "first_frame.webp",
    "preview.png",
    "preview.jpg",
    "preview.jpeg",
    "preview.webp",
)


def _usable_preview_image(path: str | Path | None) -> str | None:
    """Return a generated still path; skip missing files and conditioning inputs."""
    if not path:
        return None
    candidate = Path(path)
    try:
        if not candidate.is_file():
            return None
    except OSError:
        return None
    if candidate.suffix.lower() not in IMAGE_EXTS:
        return None
    if _artifact_is_input(candidate):
        return None
    return str(candidate)


def ensure_run_preview_image(record: RunRecord) -> str | None:
    """Return a first-frame still for Gallery, extracting it from the video if needed."""
    output_dir = str(record.output_dir or "")
    if output_dir:
        for name in _FIRST_FRAME_FILENAMES:
            usable = _usable_preview_image(Path(output_dir) / name)
            if usable:
                return usable

    usable = _usable_preview_image(record.preview_image)
    if usable:
        return usable

    for candidate in record.gallery or ():
        usable = _usable_preview_image(candidate)
        if usable:
            return usable

    video_path = record.preview_video
    if not video_path:
        return None
    try:
        if not Path(video_path).is_file():
            return None
    except OSError:
        return None

    extract_dir = output_dir or str(Path(video_path).parent)
    extracted = maybe_extract_video_preview_image(
        video_path,
        extract_dir,
        frame_position="first",
        output_name="first_frame.png",
    )
    return _usable_preview_image(extracted)


def bind_run_preview_image(record: RunRecord) -> str | None:
    """Resolve a first-frame still and cache the path on the in-memory record."""
    path = ensure_run_preview_image(record)
    if path:
        record.preview_image = path
    return path


@dataclass
class PreparedInputs:
    prompt: str
    input_path: str
    image: Optional[Image.Image]
    image_path: Optional[str]
    video_path: Optional[str]
    last_frame: Optional[Image.Image]
    last_frame_path: Optional[str]
    reference_images: list[Image.Image]
    reference_image_paths: list[str]
    interactions: Any
    camera_view: Any
    task_type: str
    intrinsics: Any
    meta_path: str
    panorama_path: str
    scene_name: str
    fps: int
    num_frames: int
    output_dir: str
    output_path: str
    call_kwargs: Dict[str, Any]
    load_kwargs: Dict[str, Any]
    model_ref: str
    backend: str
    endpoint: str
    api_key: str
    device: str
    infer_metadata: Dict[str, Any] = field(default_factory=dict)


def _infer_metadata_variant_id(infer_metadata: Mapping[str, Any] | None) -> str:
    payload = infer_metadata or {}
    variant = payload.get("variant")
    variant_id = payload.get("variant_id")
    if not variant_id and isinstance(variant, Mapping):
        variant_id = variant.get("variant_id")
    return str(variant_id or "").strip()


def _entry_extra_variant_ids(entry: CatalogEntry) -> set[str]:
    return {str(raw_variant.get("variant_id") or "").strip() for raw_variant in entry.extra_variants}


def _prepared_inputs_payload(request: PreparedInputs) -> Dict[str, Any]:
    return {
        "prompt": request.prompt,
        "input_path": request.input_path,
        "image_path": request.image_path,
        "video_path": request.video_path,
        "last_frame_path": request.last_frame_path,
        "reference_image_paths": list(request.reference_image_paths),
        "interactions": _safe_json(request.interactions),
        "camera_view": _safe_json(request.camera_view),
        "task_type": request.task_type,
        "intrinsics": _safe_json(request.intrinsics),
        "meta_path": request.meta_path,
        "panorama_path": request.panorama_path,
        "scene_name": request.scene_name,
        "fps": int(request.fps),
        "num_frames": int(request.num_frames),
        "output_dir": request.output_dir,
        "output_path": request.output_path,
        "call_kwargs": _safe_json(request.call_kwargs),
        "load_kwargs": _safe_json(request.load_kwargs),
        "model_ref": request.model_ref,
        "backend": request.backend,
        "endpoint": request.endpoint,
        "api_key": request.api_key,
        "device": request.device,
        "infer_metadata": _safe_json(request.infer_metadata),
    }


def _prepared_inputs_from_payload(payload: Dict[str, Any]) -> PreparedInputs:
    return PreparedInputs(
        prompt=str(payload.get("prompt", "")),
        input_path=str(payload.get("input_path", "")),
        image=None,
        image_path=payload.get("image_path"),
        video_path=payload.get("video_path"),
        last_frame=None,
        last_frame_path=payload.get("last_frame_path"),
        reference_images=[],
        reference_image_paths=list(payload.get("reference_image_paths", [])),
        interactions=payload.get("interactions"),
        camera_view=payload.get("camera_view"),
        task_type=str(payload.get("task_type", "")),
        intrinsics=payload.get("intrinsics"),
        meta_path=str(payload.get("meta_path", "")),
        panorama_path=str(payload.get("panorama_path", "")),
        scene_name=str(payload.get("scene_name", "")),
        fps=int(payload.get("fps", 16) or 16),
        num_frames=int(payload.get("num_frames", 0) or 0),
        output_dir=str(payload.get("output_dir", "")),
        output_path=str(payload.get("output_path", "")),
        call_kwargs=dict(payload.get("call_kwargs", {}) or {}),
        load_kwargs=dict(payload.get("load_kwargs", {}) or {}),
        model_ref=str(payload.get("model_ref", "")),
        backend=str(payload.get("backend", "auto")),
        endpoint=str(payload.get("endpoint", "")),
        api_key=str(payload.get("api_key", "")),
        device=str(payload.get("device", "cuda")),
        infer_metadata=dict(payload.get("infer_metadata", {}) or {}),
    )


def _required_inference_output_kinds(infer_metadata: Mapping[str, Any]) -> list[tuple[str, str]]:
    task = infer_metadata.get("task") if isinstance(infer_metadata, Mapping) else None
    outputs = task.get("outputs") if isinstance(task, Mapping) else None
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        return []

    required: list[tuple[str, str]] = []
    for output in outputs:
        if not isinstance(output, Mapping) or not output.get("required"):
            continue
        artifact_id = str(output.get("artifact_id") or output.get("id") or output.get("kind") or "").strip()
        kind = str(output.get("kind") or artifact_id).strip().lower().replace("-", "_")
        if kind:
            required.append((artifact_id or kind, kind))
    return required


def _missing_required_inference_outputs(
    request: PreparedInputs,
    previews: Mapping[str, Any],
) -> list[str]:
    missing: list[str] = []
    for artifact_id, kind in _required_inference_output_kinds(request.infer_metadata):
        if kind in {"manifest", "metadata", "camera_controls"}:
            continue
        if kind in {"video", "generated_video"}:
            ok = bool(previews.get("preview_video"))
        elif kind in {"image", "generated_image"}:
            ok = bool(previews.get("preview_image"))
        elif kind in {"model", "mesh", "point_cloud", "pointcloud", "splat", "gaussian_splat", "3d", "3d_asset", "generated_3d_asset"}:
            ok = bool(previews.get("preview_model") or previews.get("preview_splat"))
        else:
            continue
        if not ok:
            missing.append(f"{artifact_id}:{kind}")
    return missing


@dataclass
class RunRecord:
    run_id: str
    model_id: str
    display_name: str
    mode: str
    status: str
    output_dir: str
    manifest_path: str
    preview_video: Optional[str] = None
    preview_image: Optional[str] = None
    preview_splat: Optional[str] = None
    preview_model: Optional[str] = None
    gallery: list[str] = field(default_factory=list)
    rrd_path: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> Dict[str, Any]:
        return dict(
            _redact_manifest_secrets(
                {
                    "run_id": self.run_id,
                    "model_id": self.model_id,
                    "display_name": self.display_name,
                    "mode": self.mode,
                    "status": self.status,
                    "output_dir": self.output_dir,
                    "manifest_path": self.manifest_path,
                    "preview_video": self.preview_video,
                    "preview_image": self.preview_image,
                    "preview_splat": self.preview_splat,
                    "preview_model": self.preview_model,
                    "gallery": self.gallery,
                    "rrd_path": self.rrd_path,
                    "artifacts": self.artifacts,
                    "metadata": self.metadata,
                }
            )
        )


def _run_record_from_manifest(manifest_path: Path, *, run_id: str | None = None) -> RunRecord:
    """Load one Studio run without consulting the workspace-wide run index."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Studio manifest must contain a JSON object: {manifest_path}")

    # A pipeline may return an artifact from a temporary directory and Studio
    # subsequently copies it into the persistent run directory. Prefer the
    # persistent copy when an older manifest still advertises the dead path.
    persisted_artifacts = _existing_artifact_paths(
        [
            *list(payload.get("artifacts") or []),
            *collect_artifact_paths(str(manifest_path.parent)),
        ]
    )
    recovered_previews = pick_preview_assets(persisted_artifacts, validate_videos=False)

    def persisted_preview(key: str) -> Any:
        configured = payload.get(key)
        if configured:
            try:
                if Path(configured).is_file():
                    return configured
            except (OSError, TypeError, ValueError):
                pass
        return recovered_previews.get(key)

    return RunRecord(
        # The directory name is authoritative for direct lookups. This keeps a
        # stale or malformed payload from aliasing another run identifier.
        run_id=run_id or str(payload.get("run_id") or manifest_path.parent.name),
        model_id=str(payload.get("model_id") or ""),
        display_name=str(payload.get("display_name") or payload.get("model_id") or ""),
        mode=str(payload.get("mode") or ""),
        status=str(payload.get("status") or ""),
        output_dir=str(payload.get("output_dir") or manifest_path.parent),
        manifest_path=str(manifest_path),
        preview_video=persisted_preview("preview_video"),
        preview_image=persisted_preview("preview_image"),
        preview_splat=persisted_preview("preview_splat"),
        preview_model=persisted_preview("preview_model"),
        gallery=list(recovered_previews.get("gallery") or payload.get("gallery") or []),
        rrd_path=persisted_preview("rrd_path"),
        artifacts=list(payload.get("artifacts") or []),
        metadata=dict(payload.get("metadata") or {}),
    )


def _persist_studio_performance_metadata(record: RunRecord, timings: Mapping[str, float]) -> None:
    """Persist normalized wall-time telemetry into manifests for Studio diagnostics."""
    cleaned = {key: round(float(value), 3) for key, value in timings.items()}
    existing = record.metadata.get("studio_performance")
    if isinstance(existing, Mapping):
        cleaned = {**dict(existing), **cleaned}
    record.metadata["studio_performance"] = cleaned
    _core_write_json(
        Path(record.manifest_path),
        record.to_manifest(),
        atomic=True,
    )


@dataclass
class PipelineContext:
    entry: CatalogEntry
    pipeline: Any
    cache_key: str
    backend: str
    model_ref: str
    endpoint: str
    load_kwargs: Dict[str, Any]
    device: str
    state: Dict[str, Any] = field(default_factory=dict)
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    active_leases: int = 0
    dispose_when_idle: bool = False


class _LeasedPipelineIterator:
    """Pin and serialize a pipeline for the lifetime of a lazy result."""

    def __init__(self, manager: "StudioManager", context: PipelineContext, iterator: Iterator[Any]) -> None:
        self._manager = manager
        self._context = context
        self._iterator: Iterator[Any] = iterator
        self._state_lock = threading.Lock()
        self._next_active = False
        self._close_requested = False
        self._closed = False

    def __iter__(self) -> "_LeasedPipelineIterator":
        return self

    def __next__(self) -> Any:
        with self._state_lock:
            if self._closed or self._close_requested:
                raise StopIteration
            if self._next_active:
                raise RuntimeError("concurrent next() calls on a pipeline iterator are not supported")
            self._next_active = True
        try:
            # The context lifecycle lock was transferred to this wrapper when
            # it was created and remains held across the complete iterator.
            value = next(self._iterator)
        except BaseException:
            self._finish_next(terminal=True)
            raise
        self._finish_next(terminal=False)
        return value

    def close(self) -> None:
        finalize = False
        with self._state_lock:
            if self._closed:
                return
            self._close_requested = True
            # generator.close() cannot run while another thread is executing
            # next(). That thread owns finalization when it leaves next().
            if not self._next_active:
                self._closed = True
                finalize = True
        if finalize:
            self._finalize(suppress_errors=False)

    def _finish_next(self, *, terminal: bool) -> None:
        finalize = False
        with self._state_lock:
            self._next_active = False
            if terminal:
                self._close_requested = True
            if self._close_requested and not self._closed:
                self._closed = True
                finalize = True
        if finalize:
            # If close was requested by another thread, any close error must be
            # suppressed here so it does not replace the result/exception from
            # the in-flight next() call.
            self._finalize(suppress_errors=True)

    def _finalize(self, *, suppress_errors: bool) -> None:
        context = self._context
        manager = self._manager
        iterator = self._iterator
        # Drop our reference before releasing the eviction pin. A closed
        # custom iterator may itself own the pipeline or CUDA tensors.
        self._iterator = iter(())
        close = getattr(iterator, "close", None)
        close_error: BaseException | None = None
        try:
            if callable(close):
                close()
        except BaseException as exc:  # pragma: no cover - defensive custom iterator cleanup
            close_error = exc
        finally:
            # A primitive Lock is deliberately used here: the frontend may
            # consume/close the iterator on a different thread from creation.
            del close
            del iterator
            context.lifecycle_lock.release()
            manager._release_pipeline_lease(context)
            # A caller may keep an exhausted wrapper around indefinitely; it
            # must not retain an otherwise uncached PipelineContext.
            self._context = None  # type: ignore[assignment]
            self._manager = None  # type: ignore[assignment]
        if close_error is not None and not suppress_errors:
            raise close_error

    def __del__(self) -> None:  # pragma: no cover - nondeterministic GC safety net
        try:
            self.close()
        except Exception:
            pass


class BaseRuntimeDriver:
    kind = "default"

    def load_pipeline(
        self,
        manager: "StudioManager",
        entry: CatalogEntry,
        request: PreparedInputs,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> PipelineContext:
        backend = request.backend or entry.default_backend
        if backend == "auto":
            if entry.supports_from_pretrained:
                backend = "from_pretrained"
            elif entry.supports_api_init:
                backend = "api_init"
        payload = {
            "model_id": entry.model_id,
            "backend": request.backend or entry.default_backend,
            "model_ref": request.model_ref,
            "endpoint": request.endpoint,
            "device": request.device,
            "load_kwargs": request.load_kwargs,
        }
        cache_key = _hash_payload(payload)
        # API clients capture credentials, including provider environment
        # defaults. Keep them request-scoped; only model weights are reusable.
        cached = manager.pipeline_cache.get(cache_key) if backend != "api_init" else None
        if cached is not None:
            if cached.dispose_when_idle:
                raise RuntimeError(f"{entry.display_name} is pending unload")
            if progress_callback is not None:
                progress_callback(0.46, "Reusing cached pipeline")
            manager.pipeline_cache.move_to_end(cache_key)
            return cached

        model_ref = _request_model_ref(entry, request)
        if _studio_runtime_checks_enabled():
            missing_imports = _missing_runtime_validation_imports(entry)
            if missing_imports:
                raise _runtime_dependency_error(entry, missing_imports)
            if _is_explicit_local_ref(model_ref) and not Path(model_ref).expanduser().exists():
                raise _missing_local_model_ref_error(entry, model_ref)

        pipeline_class = manager.import_pipeline_class(entry)

        if backend == "api_init":
            load_method = getattr(pipeline_class, "api_init")
            call_payload = dict(request.load_kwargs)
            if request.endpoint:
                call_payload.setdefault("endpoint", request.endpoint)
            if request.api_key:
                call_payload.setdefault("api_key", request.api_key)
        else:
            load_method = getattr(pipeline_class, "from_pretrained")
            call_payload = dict(request.load_kwargs)
            model_ref = request.model_ref or entry.default_model_ref
            if model_ref:
                if "model_path" in _signature_names(load_method):
                    call_payload.setdefault("model_path", model_ref)
                elif "pretrained_model_path" in _signature_names(load_method):
                    call_payload.setdefault("pretrained_model_path", model_ref)
                elif "representation_path" in _signature_names(load_method):
                    call_payload.setdefault("representation_path", model_ref)
            if request.device:
                call_payload.setdefault("device", request.device)

        if progress_callback is not None:
            progress_callback(0.36, f"Loading {entry.display_name} pipeline")
        env_updates: dict[str, str] = {}
        if entry.model_id in TORCH_COMPILE_ENV_MODELS and call_payload.pop("torch_compile", False):
            env_updates["WORLDFOUNDRY_ENABLE_TORCH_COMPILE"] = "1"
            configure_persistent_compile_cache(namespace=entry.model_id)

        # Reserve cache capacity before constructing the next pipeline. Loading
        # first and evicting afterwards transiently requires N+1 model copies,
        # which is exactly when large CUDA pipelines tend to OOM.
        if backend != "api_init":
            manager._reserve_pipeline_cache_slot()
        with _temporary_env(env_updates):
            pipeline = _call_with_supported_kwargs(load_method, call_payload)
        if progress_callback is not None:
            progress_callback(0.62, "Pipeline loaded")
        context = PipelineContext(
            entry=entry,
            pipeline=pipeline,
            cache_key=cache_key,
            backend=backend,
            model_ref=request.model_ref,
            endpoint=request.endpoint,
            load_kwargs=request.load_kwargs,
            device=request.device,
        )
        if manager.max_cached_pipelines > 0 and backend != "api_init":
            manager.pipeline_cache[cache_key] = context
            manager.pipeline_cache.move_to_end(cache_key)
            manager._enforce_cache_limit()
        return context

    def run_fresh(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        result = self._invoke(ctx, request, mode="run")
        self._prime_memory_after_fresh(ctx, result)
        return manager.materialize_run(ctx, request, result=result, mode="run")

    def run_init(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        if not _supports_memory_stream(ctx.pipeline):
            raise RuntimeError(
                f"{ctx.entry.display_name} does not support state-only INIT in Studio. "
                "Provide an initial interaction for the first generated step."
            )

        seed_image = self._seed_stream_state(ctx, request, force_reset=True)
        if seed_image is None:
            raise ValueError(f"{ctx.entry.display_name} expects an image or tray selection before INIT.")

        preview_image = request.image_path
        if not preview_image:
            preview_path = Path(request.output_dir) / "preview.png"
            preview_image = _save_pil(seed_image, preview_path)

        return manager.materialize_run(
            ctx,
            request,
            result={
                "preview_image": preview_image,
                "message": f"Interactive state initialized for {ctx.entry.display_name}.",
            },
            mode="init",
            extra_metadata={"message": "interactive state initialized"},
        )

    def run_continue(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        if not ctx.entry.supports_stream or not hasattr(ctx.pipeline, "stream"):
            if ctx.entry.category in {"Embodied Action", "Visual Action"}:
                result = self._invoke(ctx, request, mode="run")
                return manager.materialize_run(ctx, request, result=result, mode="stream")
            if not ctx.entry.supports_stream:
                raise RuntimeError(f"{ctx.entry.display_name} does not expose stream().")
            raise RuntimeError(f"{ctx.entry.display_name} is stream-capable but the loaded pipeline does not expose stream().")
        self._seed_stream_state(ctx, request, force_reset=False)
        stream_request = self._prepare_stream_request(ctx, request)
        result = self._invoke(ctx, stream_request, mode="stream")
        if _supports_memory_stream(ctx.pipeline) or _pipeline_stream_state_ready(ctx.pipeline):
            ctx.state["studio_stream_initialized"] = True
        return manager.materialize_run(ctx, request, result=result, mode="stream")

    def reset(self, manager: "StudioManager", ctx: PipelineContext) -> str:
        pipeline = ctx.pipeline
        memory = getattr(pipeline, "memory_module", None)
        if memory is not None and hasattr(memory, "manage"):
            memory.manage(action="reset")
        else:
            reset_memory = getattr(pipeline, "reset_memory", None)
            if callable(reset_memory):
                reset_memory()
        ctx.state.clear()
        return f"Reset interactive state for {ctx.entry.display_name}."

    def can_init(self, ctx: PipelineContext, request: PreparedInputs) -> bool:
        return _supports_memory_stream(ctx.pipeline) and self._resolve_init_image(request) is not None

    def _resolve_init_image(self, request: PreparedInputs) -> Optional[Image.Image]:
        if request.image is not None:
            return request.image

        candidate_path = request.image_path or request.input_path
        if not candidate_path:
            return None

        path = Path(candidate_path).expanduser()
        if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
            return None

        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except Exception:
            return None

    def _seed_stream_state(
        self,
        ctx: PipelineContext,
        request: PreparedInputs,
        *,
        force_reset: bool,
    ) -> Optional[Image.Image]:
        if not _supports_memory_stream(ctx.pipeline):
            return None
        if ctx.state.get("studio_stream_initialized") and not force_reset:
            return None

        seed_image = self._resolve_init_image(request)
        if seed_image is None:
            return None

        memory = ctx.pipeline.memory_module
        memory.manage(action="reset")
        memory.record(seed_image, metadata={"prompt": request.prompt, "mode": "init"})
        ctx.state["studio_stream_initialized"] = True
        return seed_image

    def _prepare_stream_request(self, ctx: PipelineContext, request: PreparedInputs) -> PreparedInputs:
        if not _supports_memory_stream(ctx.pipeline):
            return request
        if not ctx.state.get("studio_stream_initialized"):
            return request
        # The resident memory already owns the seed.  Clear every alias that
        # `_invoke` could bind back to `images`; retaining `input_path` would
        # silently re-seed/reset pipelines such as LingBot on every chunk.
        return replace(request, input_path="", image=None, image_path=None)

    def _prime_memory_after_fresh(self, ctx: PipelineContext, result: Any) -> None:
        pipeline = ctx.pipeline
        if not _supports_memory_stream(pipeline):
            if _pipeline_stream_state_ready(pipeline):
                ctx.state["studio_stream_initialized"] = True
            return

        video_chunk = _coerce_video_chunk(result)
        if video_chunk is None:
            return

        memory = pipeline.memory_module
        memory.manage(action="reset")
        memory.record(video_chunk, type="video_chunk")
        ctx.state["studio_stream_initialized"] = True

    def _invoke(
        self,
        ctx: PipelineContext,
        request: PreparedInputs,
        mode: str,
        *,
        materialize_outputs: bool = True,
    ) -> Any:
        method = getattr(ctx.pipeline, "stream" if mode == "stream" else "__call__")
        base_kwargs = dict(request.call_kwargs)
        names = _signature_names(method)
        accepts_var_kwargs = _accepts_var_kwargs(method)
        # These adapters use **kwargs for declared model controls, not for
        # arbitrary Studio aliases (image_path, task_type, output_dir, ...).
        from worldfoundry.runtime.interactive_inference_catalog import INTERACTIVE_INFERENCE_SPECS

        if ctx.entry.model_id in INTERACTIVE_INFERENCE_SPECS:
            names = set(ctx.entry.call_params)
            accepts_var_kwargs = False
        declares_visual_modalities = hasattr(ctx.pipeline, "ACCEPTS_IMAGES") or hasattr(
            ctx.pipeline, "ACCEPTS_VIDEO"
        )
        accepts_images = getattr(ctx.pipeline, "ACCEPTS_IMAGES", True) is not False
        accepts_video = getattr(ctx.pipeline, "ACCEPTS_VIDEO", True) is not False
        accepts_generic_input = not declares_visual_modalities or accepts_images or accepts_video
        input_path_as_video = _input_path_should_bind_as_video(request.input_path, names)
        if input_path_as_video and ctx.entry.family in _NO_VIDEO_BIND_FAMILIES:
            input_path_as_video = False

        if request.prompt and ("prompt" in names or accepts_var_kwargs):
            base_kwargs.setdefault("prompt", request.prompt)
        if accepts_images and request.image is not None and ("images" in names or accepts_var_kwargs):
            base_kwargs.setdefault("images", request.image)
        elif accepts_images and request.image_path and ("images" in names or accepts_var_kwargs) and request.image is None:
            base_kwargs.setdefault("images", request.image_path)
        elif (
            accepts_images
            and
            request.video_path
            and "images" in names
            and request.image is None
            and not accepts_var_kwargs
            and not ({"video", "videos", "video_path"} & names)
        ):
            base_kwargs.setdefault("images", request.video_path)
        elif accepts_images and request.input_path and not input_path_as_video and ("images" in names or accepts_var_kwargs) and request.image is None:
            base_kwargs.setdefault("images", request.input_path)
        if accepts_images and request.image_path and ("image_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("image_path", request.image_path)
        elif accepts_images and request.input_path and not input_path_as_video and ("image_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("image_path", request.input_path)
        if accepts_video and request.video_path and ("videos" in names or accepts_var_kwargs):
            base_kwargs.setdefault("videos", request.video_path)
        elif accepts_video and request.input_path and input_path_as_video and ("videos" in names or accepts_var_kwargs):
            base_kwargs.setdefault("videos", request.input_path)
        if accepts_video and request.video_path and ("video" in names or accepts_var_kwargs):
            base_kwargs.setdefault("video", request.video_path)
        elif accepts_video and request.input_path and input_path_as_video and ("video" in names or accepts_var_kwargs):
            base_kwargs.setdefault("video", request.input_path)
        if accepts_video and request.video_path and ("video_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("video_path", request.video_path)
        elif accepts_video and request.input_path and input_path_as_video and ("video_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("video_path", request.input_path)
        if accepts_generic_input and request.input_path and ("input_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("input_path", request.input_path)
        if "data_path" in names:
            data_path = request.input_path or request.image_path or request.video_path
            if data_path:
                base_kwargs.setdefault("data_path", data_path)
        if request.interactions is not None and ("interactions" in names or accepts_var_kwargs):
            base_kwargs["interactions"] = request.interactions
        if request.interactions is not None and ("interaction_signal" in names or accepts_var_kwargs):
            base_kwargs["interaction_signal"] = request.interactions
        if request.camera_view is not None and ("camera_view" in names or accepts_var_kwargs):
            base_kwargs.setdefault("camera_view", request.camera_view)
        if request.task_type and ("task_type" in names or accepts_var_kwargs):
            base_kwargs.setdefault("task_type", request.task_type)
        if request.intrinsics is not None and ("K" in names or accepts_var_kwargs):
            base_kwargs.setdefault("K", request.intrinsics)
        if request.meta_path and ("meta_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("meta_path", request.meta_path)
        if request.panorama_path and ("panorama_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("panorama_path", request.panorama_path)
        if request.scene_name and ("scene_name" in names or accepts_var_kwargs):
            base_kwargs.setdefault("scene_name", request.scene_name)
        if request.last_frame is not None and ("last_frame" in names or accepts_var_kwargs):
            base_kwargs.setdefault("last_frame", request.last_frame)
        if request.reference_images and ("reference_images" in names or accepts_var_kwargs):
            base_kwargs.setdefault("reference_images", request.reference_images)
        # A few official runtimes expose both ``frame_num`` (their native
        # option) and ``num_frames`` (the generic Studio alias).  Do not let
        # Studio's generic fallback silently override an explicit native
        # value that is already present in ``call_kwargs``.
        frame_aliases = ("num_frames", "frame_num", "frames", "video_length")
        if (
            request.num_frames
            and "num_frames" in names
            and not any(alias in base_kwargs for alias in frame_aliases)
        ):
            base_kwargs.setdefault("num_frames", request.num_frames)
        if request.fps and "fps" in names:
            base_kwargs.setdefault("fps", request.fps)
        if mode == "run" and ("return_dict" in names or accepts_var_kwargs):
            base_kwargs.setdefault("return_dict", True)
        if mode == "stream" and "images" in names and "images" not in base_kwargs:
            # Some stream() entry points require the images argument even after
            # Studio has already seeded memory from the previous turn.
            base_kwargs["images"] = None
        if materialize_outputs and ("output_dir" in names or accepts_var_kwargs):
            base_kwargs.setdefault("output_dir", request.output_dir)
        if materialize_outputs and ("output_path" in names or accepts_var_kwargs):
            base_kwargs.setdefault("output_path", request.output_path)

        result = _call_with_supported_kwargs(method, base_kwargs)
        if materialize_outputs and hasattr(result, "__iter__") and not isinstance(
            result,
            (dict, list, tuple, str, bytes, Image.Image, np.ndarray),
        ):
            result = list(result)
        return result


class TwoStage3DGSRuntimeDriver(BaseRuntimeDriver):
    kind = "two_stage_3dgs"

    def run_fresh(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        pipeline = ctx.pipeline
        image_input = request.image_path or request.input_path
        if not image_input:
            raise ValueError(f"{ctx.entry.display_name} expects an image path or image upload.")
        task_type = str(request.task_type or request.call_kwargs.get("task_type") or "").strip()
        if task_type in {"cut3r_official_export", "cut3r_official", "official"}:
            call_kwargs = dict(request.call_kwargs)
            call_kwargs.pop("task_type", None)
            result = pipeline(
                image_path=image_input,
                task_type=task_type,
                output_dir=request.output_dir,
                **call_kwargs,
            )
            return manager.materialize_run(
                ctx,
                request,
                result=result,
                mode="run",
                extra_metadata={"task_type": task_type},
            )
        recon_kwargs = {}
        if ctx.entry.model_id in {"vggt", "vggt-omega"}:
            recon_kwargs["point_conf_threshold"] = request.call_kwargs.get("point_conf_threshold", 0.2)
            recon_kwargs["resolution"] = request.call_kwargs.get("resolution", 518)
            recon_kwargs["preprocess_mode"] = request.call_kwargs.get("preprocess_mode", "crop")
        else:
            recon_kwargs["size"] = request.call_kwargs.get("size")
            recon_kwargs["vis_threshold"] = request.call_kwargs.get("vis_threshold", 1.5)

        recon_info = pipeline.reconstruct_ply(image_input, ply_path=request.output_dir, **recon_kwargs)
        ctx.state["recon_info"] = recon_info
        frames = self._render_frames(ctx, request)
        preview_video = export_frames_to_video(frames, request.output_path, fps=request.fps)
        first_frame_path = str(Path(request.output_dir) / "first_frame.png")
        Image.fromarray(_to_uint8_rgb(frames[0])).save(first_frame_path)
        metadata = {"recon_info": recon_info, "preview_video": preview_video}
        return manager.materialize_run(
            ctx,
            request,
            result={"preview_video": preview_video, "recon_info": recon_info, "first_frame": first_frame_path},
            mode="run",
            extra_metadata=metadata,
        )

    def run_continue(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        if "recon_info" not in ctx.state:
            raise RuntimeError(f"{ctx.entry.display_name} has no cached reconstruction yet.")
        frames = self._render_frames(ctx, request)
        preview_video = export_frames_to_video(frames, request.output_path, fps=request.fps)
        first_frame_path = str(Path(request.output_dir) / "first_frame.png")
        Image.fromarray(_to_uint8_rgb(frames[0])).save(first_frame_path)
        return manager.materialize_run(
            ctx,
            request,
            result={"preview_video": preview_video, "recon_info": ctx.state["recon_info"], "first_frame": first_frame_path},
            mode="stream",
        )

    def _render_frames(self, ctx: PipelineContext, request: PreparedInputs) -> list[Any]:
        pipeline = ctx.pipeline
        recon_info = ctx.state["recon_info"]
        camera_range = recon_info["camera_range"]
        base_camera = dict(recon_info["default_camera"])
        base_camera["radius"] = request.call_kwargs.get("camera_radius", base_camera.get("radius", 4.0))
        base_camera["yaw"] = request.call_kwargs.get("camera_yaw", base_camera.get("yaw", 0.0))
        base_camera["pitch"] = request.call_kwargs.get("camera_pitch", base_camera.get("pitch", 0.0))
        if request.camera_view is not None and ctx.entry.model_id == "vggt":
            base_camera = pipeline._apply_camera_view_to_camera_cfg(  # type: ignore[attr-defined]
                camera_cfg=base_camera,
                camera_view=request.camera_view,
                camera_range=camera_range,
            )

        interactions = request.interactions or []
        if isinstance(interactions, str):
            interactions = [interactions]
        if interactions:
            repeats = max(int(request.call_kwargs.get("frames_per_interaction", 10)), 1)
            supports_interpolated_actions = ctx.entry.model_id in {"vggt", "vggt-omega"}
            sequence = list(interactions) if supports_interpolated_actions else [
                token for token in interactions for _ in range(repeats)
            ]
            render_kwargs = {"frames_per_interaction": repeats} if supports_interpolated_actions else {}
            return pipeline.render_interaction_video_with_3dgs(
                ply_path=recon_info["ply_path"],
                camera_range=camera_range,
                base_camera_config=base_camera,
                interaction_sequence=sequence,
                image_width=int(request.call_kwargs.get("image_width", 704)),
                image_height=int(request.call_kwargs.get("image_height", 480)),
                fps=request.fps,
                **render_kwargs,
            )
        return pipeline.render_orbit_video_with_3dgs(
            ply_path=recon_info["ply_path"],
            base_camera_config=base_camera,
            num_frames=int(request.call_kwargs.get("num_orbit_frames", 24)),
            yaw_step=float(request.call_kwargs.get("yaw_step", 6.0 if ctx.entry.model_id == "vggt" else 5.0)),
            image_width=int(request.call_kwargs.get("image_width", 704)),
            image_height=int(request.call_kwargs.get("image_height", 480)),
            fps=request.fps,
        )


class PointCloudNavRuntimeDriver(BaseRuntimeDriver):
    kind = "pointcloud_nav"

    def run_fresh(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        pipeline = ctx.pipeline
        data_input = request.input_path or request.image_path or request.video_path
        if not data_input:
            raise ValueError(f"{ctx.entry.display_name} expects image(s), video, or an input path.")
        result = pipeline(
            images=data_input if request.video_path is None else None,
            videos=request.video_path,
            task_type="reconstruction",
            **request.call_kwargs,
        )
        ctx.state["reconstructed"] = True
        saved_paths = []
        if hasattr(result, "save"):
            saved_paths = result.save(request.output_dir)
        default_render = pipeline(task_type="render_view", camera_view=0)
        default_render_path = str(Path(request.output_dir) / "render_default.png")
        default_render.save(default_render_path)
        saved_paths.append(default_render_path)
        metadata = {"saved_paths": saved_paths, "camera_range": getattr(result, "camera_range", {})}
        return manager.materialize_run(ctx, request, result=result, mode="run", extra_metadata=metadata)

    def run_continue(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        if not ctx.state.get("reconstructed"):
            raise RuntimeError(f"{ctx.entry.display_name} has no cached reconstruction yet.")
        pipeline = ctx.pipeline
        task_type = request.task_type or "render_view"
        if task_type == "render_trajectory":
            frames = pipeline(task_type="render_trajectory", **request.call_kwargs)
            preview_video = export_frames_to_video(frames, request.output_path, fps=request.fps)
            result = {"preview_video": preview_video}
        else:
            interactions = request.interactions
            if interactions is None:
                image = pipeline(
                    task_type="render_view",
                    camera_view=request.camera_view,
                    **request.call_kwargs,
                )
                image_path = str(Path(request.output_dir) / "render_view.png")
                image.save(image_path)
                result = {"preview_image": image_path}
            elif isinstance(interactions, list) and len(interactions) > 1:
                frames = pipeline(task_type="render_view", interactions=interactions, **request.call_kwargs)
                preview_video = export_frames_to_video(frames, request.output_path, fps=request.fps)
                result = {"preview_video": preview_video}
            else:
                token = interactions[0] if isinstance(interactions, list) and interactions else interactions
                stream_kwargs = dict(request.call_kwargs)
                stream_params = _signature_names(pipeline.stream)
                if request.camera_view is not None and "camera_view" in stream_params:
                    stream_kwargs["camera_view"] = request.camera_view
                if "interaction_signal" in stream_params:
                    stream_kwargs["interaction_signal"] = token
                else:
                    stream_kwargs["interactions"] = token
                image = _call_with_supported_kwargs(pipeline.stream, stream_kwargs)
                image_path = str(Path(request.output_dir) / "stream_view.png")
                image.save(image_path)
                result = {"preview_image": image_path}
        return manager.materialize_run(ctx, request, result=result, mode="stream")


class WorldFMRuntimeDriver(BaseRuntimeDriver):
    kind = "worldfm"

    def _resolve_interactions(self, request: PreparedInputs, fallback: Any = None) -> Any:
        interactions = request.interactions if request.interactions is not None else fallback
        if isinstance(interactions, list) and interactions and isinstance(interactions[0], str):
            return build_worldfm_pose_sequence(interactions)
        return interactions

    def _resolve_intrinsics(self, request: PreparedInputs) -> Any:
        if request.intrinsics is not None:
            return request.intrinsics
        if request.image_path:
            return guess_intrinsics(request.image_path)
        return None

    def run_fresh(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        pipeline = ctx.pipeline
        call_kwargs = dict(request.call_kwargs)
        call_interactions = call_kwargs.pop("interactions", None)
        panorama_path = request.panorama_path or call_kwargs.pop("panorama_path", None)
        scene_name = request.scene_name or call_kwargs.pop("scene_name", None) or "worldfm_scene"
        call_kwargs["return_dict"] = True
        result = pipeline(
            images=request.image,
            panorama_path=panorama_path or None,
            prompt=request.prompt,
            K=self._resolve_intrinsics(request),
            interactions=self._resolve_interactions(request, call_interactions),
            scene_name=scene_name,
            output_dir=request.output_dir,
            **call_kwargs,
        )
        return manager.materialize_run(ctx, request, result=result, mode="run")

    def run_continue(self, manager: "StudioManager", ctx: PipelineContext, request: PreparedInputs) -> RunRecord:
        pipeline = ctx.pipeline
        call_kwargs = dict(request.call_kwargs)
        call_interactions = call_kwargs.pop("interactions", None)
        panorama_path = request.panorama_path or call_kwargs.pop("panorama_path", None)
        scene_name = request.scene_name or call_kwargs.pop("scene_name", None) or "worldfm_scene"
        call_kwargs["return_dict"] = True
        result = pipeline.stream(
            images=request.image,
            panorama_path=panorama_path or None,
            prompt=request.prompt,
            K=self._resolve_intrinsics(request),
            interactions=self._resolve_interactions(request, call_interactions),
            scene_name=scene_name,
            output_dir=request.output_dir,
            **call_kwargs,
        )
        return manager.materialize_run(ctx, request, result=result, mode="stream")


def _signature_names(callable_obj: Any) -> set[str]:
    import inspect

    try:
        return set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return set()


def _accepts_var_kwargs(callable_obj: Any) -> bool:
    import inspect

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _call_with_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Any:
    import inspect

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(**kwargs)

    accepts_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kw:
        cleaned = {}
        for key, value in kwargs.items():
            if key in signature.parameters:
                cleaned[key] = value
            elif value is not None:
                cleaned[key] = value
    else:
        cleaned = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return callable_obj(**cleaned)


RUNTIME_DRIVERS: Dict[str, BaseRuntimeDriver] = {
    "default": BaseRuntimeDriver(),
    "two_stage_3dgs": TwoStage3DGSRuntimeDriver(),
    "pointcloud_nav": PointCloudNavRuntimeDriver(),
    "worldfm": WorldFMRuntimeDriver(),
}


class StudioManager:
    def __init__(
        self, workspace_root: str | None = None, max_cached_pipelines: int = 2, *,
        record_processor: Callable[[RunRecord, CatalogEntry], RunRecord] | None = None,
    ) -> None:
        # Optional presentation adapter supplied by a UI, never imported here.
        self.record_processor = record_processor
        resolved_root = Path(workspace_root) if workspace_root else studio_workspace_root()
        self.workspace_root = str(_ensure_dir(resolved_root))
        self.runs_root = str(_ensure_dir(Path(self.workspace_root) / "runs"))
        self.pipeline_cache: "OrderedDict[str, PipelineContext]" = OrderedDict()
        self.max_cached_pipelines = max(0, int(max_cached_pipelines))
        # Iterator finalizers may release a pipeline lease while manager code
        # already owns this lock in the same thread.
        self.lock = threading.RLock()
        self.torchrun_command_lock = threading.Lock()
        self._recent_runs_lock = threading.RLock()
        self._recent_runs_root_signature: tuple[int, int, int] | None = None
        self._recent_run_manifests: tuple[Path, ...] = ()
        self._recent_run_manifest_cursor = 0
        self._recent_runs_cache: tuple[RunRecord, ...] = ()

    def _invalidate_recent_runs_cache(self) -> None:
        """Invalidate the manifest index after a manager-owned run write."""

        with self._recent_runs_lock:
            self._recent_runs_root_signature = None
            self._recent_run_manifests = ()
            self._recent_run_manifest_cursor = 0
            self._recent_runs_cache = ()

    def _persist_performance_metadata(self, record: RunRecord, timings: Mapping[str, float]) -> None:
        _persist_studio_performance_metadata(record, timings)
        self._invalidate_recent_runs_cache()

    def _torchrun_dist(self) -> Any:
        dist = _torch_dist()
        if dist is None or not _torchrun_lingbot_fast_enabled():
            return None
        if not ensure_torchrun_lingbot_fast_runtime():
            return None
        return dist

    def _torchrun_worker_request(
        self,
        entry: CatalogEntry,
        request: PreparedInputs,
    ) -> PreparedInputs:
        if not _torchrun_lingbot_fast_enabled():
            return request
        load_kwargs = dict(request.load_kwargs)
        load_kwargs["rank"] = _torchrun_rank()
        torch = _torch_module()

        device = request.device
        if torch is not None and torch.cuda.is_available():
            device = f"cuda:{_torchrun_cuda_device_index(torch)}"

        if entry.model_id == MATRIX_GAME3_MODEL_ID:
            # Matrix-Game 3 owns its Ulysses/FSDP policy. Do not leak LingBot's
            # replicated-vs-sharded heuristic or offload flags into this model.
            load_kwargs["ulysses_size"] = _torchrun_world_size()
            load_kwargs.pop("world_size", None)
            return replace(request, load_kwargs=load_kwargs, device=device)

        if entry.model_id in {HELIOS_MODEL_ID, LONGVIE2_MODEL_ID, DREAMX_WORLD_MODEL_ID}:
            # These runtimes own their context/sequence parallel topology. The
            # Studio command bridge maps each rank to local CUDA without
            # leaking LingBot FSDP/offload policy into their model loaders.
            if entry.model_id == LONGVIE2_MODEL_ID:
                world_size = _torchrun_world_size()
                if world_size != 4:
                    raise RuntimeError(
                        "LongVie 2 distributed Studio requires exactly four torchrun ranks."
                    )
                load_kwargs.update(
                    use_usp=True,
                    ring_degree=1,
                    ulysses_degree=4,
                )
            return replace(request, load_kwargs=load_kwargs, device=device)

        replicate_model = False
        total_gib = _torchrun_min_gpu_vram_gib()
        if total_gib is not None:
            try:
                min_gib = float(os.getenv("WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB", "72") or "72")
                replicate_model = total_gib >= max(min_gib, 0.0)
            except ValueError:
                replicate_model = False
        # Replication avoids per-block FSDP all-gathers on 80GB-class GPUs.
        # Lower-memory and unknown devices default to the conservative sharded
        # path; explicit user choices always win.
        load_kwargs.setdefault("t5_fsdp", not replicate_model)
        load_kwargs.setdefault("dit_fsdp", not replicate_model)
        load_kwargs.setdefault(
            "ulysses_size",
            _torchrun_world_size() if lingbot_fast_sequence_parallel_enabled() else 1
        )
        load_kwargs.setdefault("t5_cpu", False)

        # LingBot v1 keeps ``offload_model`` in call kwargs, while its runtime
        # constructor also uses that value as the default prediction policy.
        # Propagate the effective request policy into the loader so the
        # catalog's distributed default (DiT FSDP + no offload) is not
        # accidentally validated against the constructor's legacy
        # ``offload_model=True`` default.  Call kwargs are authoritative for
        # the actual prediction, so synchronize that effective policy before
        # the loader performs its FSDP/offload preflight.
        load_kwargs["offload_model"] = bool(
            request.call_kwargs.get("offload_model", load_kwargs.get("offload_model", False))
        )

        call_kwargs = dict(request.call_kwargs)

        if entry.model_id == LINGBOT_WORLD_V2_MODEL_ID:
            # The active Workspace process group is authoritative.  Catalog
            # defaults describe the official eight-rank recipe, but a valid
            # compact four-rank launch must not carry a stale eight into the
            # runtime's preflight validation.
            call_kwargs["nproc_per_node"] = _torchrun_world_size()
        call_kwargs.setdefault("offload_model", False)

        return replace(
            request,
            load_kwargs=load_kwargs,
            call_kwargs=call_kwargs,
            device=device,
        )

    def _should_use_torchrun_lingbot_fast(self, entry: CatalogEntry, request: PreparedInputs) -> bool:
        if entry.model_id not in {
            LINGBOT_WORLD_MODEL_ID,
            LINGBOT_WORLD_V2_MODEL_ID,
            MATRIX_GAME3_MODEL_ID,
            LONGVIE2_MODEL_ID,
            HELIOS_MODEL_ID,
            DREAMX_WORLD_MODEL_ID,
        }:
            return False
        if request.backend == "api_init":
            return False
        if not _torchrun_lingbot_fast_enabled():
            return False
        if entry.model_id == LONGVIE2_MODEL_ID:
            world_size = _torchrun_world_size()
            if world_size not in {1, 4}:
                raise RuntimeError(
                    "LongVie 2 supports only one GPU or exactly four USP ranks; "
                    f"got WORLD_SIZE={world_size}."
                )
            return world_size == 4
        if entry.model_id in {MATRIX_GAME3_MODEL_ID, HELIOS_MODEL_ID, DREAMX_WORLD_MODEL_ID}:
            return _torchrun_world_size() > 1
        if entry.model_id == LINGBOT_WORLD_V2_MODEL_ID:
            return True
        runtime_variant = str(request.load_kwargs.get("runtime_variant", "") or "").strip().lower()
        if runtime_variant == "fast":
            return True
        try:
            if int(request.load_kwargs.get("ulysses_size") or 1) > 1:
                return True
        except Exception:
            pass
        return bool(request.load_kwargs.get("dit_fsdp") or request.load_kwargs.get("t5_fsdp"))

    def _broadcast_torchrun_command(self, command: Dict[str, Any]) -> None:
        dist = self._torchrun_dist()
        if dist is None:
            return
        control_group = _torchrun_control_group()
        if control_group is None:
            raise RuntimeError("Torchrun LingBot fast control group is not initialized.")
        payload = [command]
        dist.broadcast_object_list(payload, src=0, group=control_group)

    def _run_torchrun_command(self, command: Dict[str, Any]) -> Any:
        # Gloo/NCCL collectives must run in exactly the same order on every rank.
        # The standalone world frontend serves requests concurrently, so serialize
        # the full command transaction rather than only the local model execution.
        with self.torchrun_command_lock:
            self._broadcast_torchrun_command(command)
            return self._execute_torchrun_command(command)

    def shutdown_torchrun_workers(self) -> None:
        with self.torchrun_command_lock:
            self._broadcast_torchrun_command({"kind": "shutdown"})

    def _run_local_prepared_request(
        self,
        *,
        entry: CatalogEntry,
        action: str,
        request: PreparedInputs,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        perf_segments: Optional[Dict[str, float]] = None,
        materialize: bool = True,
    ) -> Optional[RunRecord]:
        driver = self.runtime_driver_for(entry)
        with self.lock:
            if progress_callback is not None:
                progress_callback(0.24, "Resolving runtime")
            t_load = time.perf_counter()
            ctx = driver.load_pipeline(self, entry, request, progress_callback=progress_callback)
            self._pin_pipeline_context(ctx)
            if perf_segments is not None:
                perf_segments["load_pipeline_ms"] = (time.perf_counter() - t_load) * 1000.0

        t_exec = time.perf_counter()
        try:
            with ctx.lifecycle_lock:
                if action == "init":
                    if progress_callback is not None:
                        progress_callback(0.72, "Initializing interactive state")
                    record = driver.run_init(self, ctx, request)
                elif action == "run":
                    if progress_callback is not None:
                        progress_callback(0.72, "Running fresh inference")
                    if materialize:
                        record = driver.run_fresh(self, ctx, request)
                    else:
                        result = driver._invoke(ctx, request, mode="run")
                        driver._prime_memory_after_fresh(ctx, result)
                        record = None
                elif action == "stream":
                    if progress_callback is not None:
                        progress_callback(0.72, "Running stream continuation")
                    if not materialize:
                        raise RuntimeError(
                            "Non-materializing torchrun Studio execution only supports fresh run actions."
                        )
                    record = driver.run_continue(self, ctx, request)
                elif action == "reset":
                    if progress_callback is not None:
                        progress_callback(0.9, "Resetting interactive state")
                    if not materialize:
                        raise RuntimeError(
                            "Non-materializing torchrun Studio execution only supports fresh run actions."
                        )
                    message = driver.reset(self, ctx)
                    record = self._make_message_record(entry, request, message)
                elif not materialize:
                    raise RuntimeError(
                        "Non-materializing torchrun Studio execution only supports fresh run actions."
                    )
                else:
                    raise ValueError(f"Unsupported Studio action: {action}")
                if perf_segments is not None:
                    perf_segments["execute_ms"] = (time.perf_counter() - t_exec) * 1000.0
                return record
        finally:
            self._release_pipeline_lease(ctx)

    def _run_local_torchrun_lingbot_fast(
        self,
        *,
        entry: CatalogEntry,
        action: str,
        request: PreparedInputs,
        materialize: bool,
        realtime: bool = False,
    ) -> Any:
        driver = self.runtime_driver_for(entry)
        request = self._torchrun_worker_request(entry, request)
        with self.lock:
            ctx = driver.load_pipeline(self, entry, request, progress_callback=None)
            self._pin_pipeline_context(ctx)

        try:
            with ctx.lifecycle_lock:
                if realtime:
                    handled, realtime_result = self._run_native_realtime_pipeline_action(
                        driver=driver,
                        ctx=ctx,
                        request=request,
                        action=action,
                    )
                    if handled:
                        return list(realtime_result) if isinstance(realtime_result, Iterator) else realtime_result

                if action in {"configure", "init"}:
                    if action == "configure" and not driver.can_init(ctx, request):
                        return None
                    if materialize:
                        return driver.run_init(self, ctx, request)
                    seed_image = driver._seed_stream_state(ctx, request, force_reset=True)
                    if seed_image is None:
                        raise ValueError(f"{ctx.entry.display_name} expects an image or tray selection before INIT.")
                    return seed_image

                if action == "run":
                    result = driver._invoke(
                        ctx,
                        request,
                        mode="run",
                        # All ranks must execute the same model API/kwargs.
                        # ``materialize`` controls rank-0 artifact I/O only.
                        materialize_outputs=False,
                    )
                    if isinstance(result, Iterator):
                        result = list(result)
                    driver._prime_memory_after_fresh(ctx, result)
                    if materialize:
                        return self.materialize_run(ctx, request, result=result, mode="run")
                    return result

                if action == "stream":
                    if not ctx.entry.supports_stream:
                        if ctx.entry.category in {"Embodied Action", "Visual Action"}:
                            result = driver._invoke(
                                ctx,
                                request,
                                mode="run",
                                materialize_outputs=False,
                            )
                            if isinstance(result, Iterator):
                                result = list(result)
                            if materialize:
                                return self.materialize_run(ctx, request, result=result, mode="stream")
                            return result
                        raise RuntimeError(f"{ctx.entry.display_name} does not expose stream().")
                    driver._seed_stream_state(ctx, request, force_reset=False)
                    stream_request = driver._prepare_stream_request(ctx, request)
                    result = driver._invoke(
                        ctx,
                        stream_request,
                        mode="stream",
                        materialize_outputs=False,
                    )
                    if isinstance(result, Iterator):
                        result = list(result)
                    if _supports_memory_stream(ctx.pipeline) or _pipeline_stream_state_ready(ctx.pipeline):
                        ctx.state["studio_stream_initialized"] = True
                    if materialize:
                        return self.materialize_run(ctx, request, result=result, mode="stream")
                    return result

                if action == "reset":
                    message = driver.reset(self, ctx)
                    if materialize:
                        return self._make_message_record(entry, request, message)
                    return None

                raise ValueError(f"Unsupported Studio action: {action}")
        finally:
            self._release_pipeline_lease(ctx)

    def _run_native_realtime_pipeline_action(
        self,
        *,
        driver: BaseRuntimeDriver,
        ctx: PipelineContext,
        request: PreparedInputs,
        action: str,
    ) -> tuple[bool, Any]:
        """Dispatch to a model-owned resident session when one is exposed."""

        pipeline = ctx.pipeline
        configure = getattr(pipeline, "configure_realtime", None)
        stream = getattr(pipeline, "stream_realtime", None)
        if not callable(configure) or not callable(stream):
            return False, None

        if action in {"configure", "init"}:
            seed_image = driver._resolve_init_image(request)
            prompt_only_configure = (
                "prompt-scheduled" in ctx.entry.tags and bool(str(request.prompt or "").strip())
            )
            if seed_image is None and not prompt_only_configure:
                prepare = getattr(pipeline, "prepare_realtime", None)
                if callable(prepare):
                    return True, prepare()
                return True, None
            kwargs = dict(request.call_kwargs)
            for key in ("images", "image", "prompt", "interactions", "fps"):
                kwargs.pop(key, None)
            result = configure(
                images=seed_image,
                prompt=request.prompt,
                fps=request.fps,
                **kwargs,
            )
            ctx.state["studio_stream_initialized"] = True
            return True, result

        if action == "stream":
            if not ctx.state.get("studio_stream_initialized"):
                raise RuntimeError(
                    f"{ctx.entry.display_name} realtime stream has not been configured."
                )
            kwargs = dict(request.call_kwargs)
            for key in ("images", "image", "prompt", "interactions"):
                kwargs.pop(key, None)
            result = stream(
                prompt=request.prompt,
                interactions=list(request.interactions or []),
                **kwargs,
            )
            return True, result

        if action == "reset":
            reset = getattr(pipeline, "reset_realtime", None)
            if callable(reset):
                reset()
            ctx.state.clear()
            return True, f"Reset realtime state for {ctx.entry.display_name}."

        return False, None

    def _reset_cached_model_local(self, model_id: str) -> str:
        with self.lock:
            matching_contexts = [
                context
                for context in self.pipeline_cache.values()
                if context.entry.model_id == model_id and not context.dispose_when_idle
            ]
            if not matching_contexts:
                try:
                    entry = find_entry(model_id)
                    return f"No cached interactive state for {entry.display_name}."
                except Exception:
                    return f"No cached interactive state for {model_id}."
            for context in matching_contexts:
                self._pin_pipeline_context(context)

        messages: list[str] = []
        try:
            for context in matching_contexts:
                driver = self.runtime_driver_for(context.entry)
                with context.lifecycle_lock:
                    messages.append(driver.reset(self, context))
        finally:
            for context in matching_contexts:
                self._release_pipeline_lease(context)

        deduped_messages = list(dict.fromkeys(messages))
        if len(deduped_messages) == 1:
            return deduped_messages[0]
        return " ".join(deduped_messages)

    def _unload_local(self, model_id: Optional[str] = None) -> str:
        with self.lock:
            removed: list[str] = []
            scheduled: list[str] = []
            for key, context in list(self.pipeline_cache.items()):
                if model_id and context.entry.model_id != model_id:
                    continue
                if context.active_leases > 0:
                    context.dispose_when_idle = True
                    scheduled.append(context.entry.display_name)
                    continue
                self.pipeline_cache.pop(key, None)
                removed.append(context.entry.display_name)
                self._dispose_pipeline_context(context)
            if removed and scheduled:
                return (
                    f"Unloaded: {', '.join(removed)}. "
                    f"Scheduled after active stream: {', '.join(scheduled)}."
                )
            if removed:
                return f"Unloaded: {', '.join(removed)}."
            if scheduled:
                return f"Scheduled after active stream: {', '.join(scheduled)}."
            return "No cached pipelines were unloaded."

    def close(self) -> None:
        """Release every idle cached pipeline and schedule active streams for disposal."""

        with self.lock:
            for key, context in list(self.pipeline_cache.items()):
                if context.active_leases > 0:
                    context.dispose_when_idle = True
                    continue
                self.pipeline_cache.pop(key, None)
                self._dispose_pipeline_context(context)

    def _execute_torchrun_command(self, command: Dict[str, Any]) -> Any:
        dist = self._torchrun_dist()
        if dist is None:
            raise RuntimeError("Torchrun LingBot fast runtime is not initialized.")

        kind = str(command.get("kind", "") or "")
        local_error = ""
        local_result: Any = None
        rank = _torchrun_rank()
        try:
            if kind == "run_action":
                entry = find_entry(str(command["model_id"]))
                request = _prepared_inputs_from_payload(dict(command["request"]))
                local_result = self._run_local_torchrun_lingbot_fast(
                    entry=entry,
                    action=str(command["action"]),
                    request=request,
                    materialize=rank == 0,
                    realtime=False,
                )
            elif kind == "realtime_action":
                entry = find_entry(str(command["model_id"]))
                request = _prepared_inputs_from_payload(dict(command["request"]))
                local_result = self._run_local_torchrun_lingbot_fast(
                    entry=entry,
                    action=str(command["action"]),
                    request=request,
                    materialize=False,
                    realtime=True,
                )
            elif kind == "reset_model":
                local_result = self._reset_cached_model_local(str(command["model_id"]))
            elif kind == "unload_model":
                model_id = command.get("model_id") or None
                local_result = self._unload_local(str(model_id) if model_id else None)
            else:
                raise ValueError(f"Unsupported torchrun Studio command: {kind}")
            if isinstance(local_result, Iterator):
                # Every rank must advance a distributed generator so its
                # collectives execute. Returning a rank-0-only lazy iterator
                # would leave the remaining ranks waiting for the next command.
                local_result = list(local_result)
        except Exception as exc:
            print(
                f"[studio][torchrun][rank {rank}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            local_error = f"{type(exc).__name__}: {exc}"

        statuses: list[Dict[str, Any] | None] = [None] * _torchrun_world_size()
        dist.all_gather_object(
            statuses,
            {"rank": rank, "error": local_error},
            group=_torchrun_control_group(),
        )
        failures = [status for status in statuses if status and status.get("error")]
        if failures:
            if rank == 0:
                joined = "; ".join(
                    f"rank {status['rank']}: {status['error']}"
                    for status in failures
                )
                raise RuntimeError(f"Distributed Studio command failed: {joined}")
            return None
        return local_result if rank == 0 else None

    def run_torchrun_worker_loop(self) -> None:
        dist = self._torchrun_dist()
        rank = _torchrun_rank()
        if dist is None or rank == 0:
            return
        control_group = _torchrun_control_group()
        if control_group is None:
            raise RuntimeError("Torchrun LingBot fast control group is not initialized.")
        print(f"[studio][torchrun][rank {rank}] worker loop starting", file=sys.stderr, flush=True)
        try:
            while True:
                payload: list[Any] = [None]
                try:
                    dist.broadcast_object_list(payload, src=0, group=control_group)
                except Exception as exc:
                    print(
                        f"[studio][torchrun][rank {rank}] command broadcast failed: "
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                command = payload[0]
                if not isinstance(command, dict):
                    continue
                kind = str(command.get("kind", "") or "")
                print(
                    f"[studio][torchrun][rank {rank}] received command: {kind or 'unknown'}",
                    file=sys.stderr,
                    flush=True,
                )
                if kind == "shutdown":
                    return
                self._execute_torchrun_command(command)
        finally:
            print(f"[studio][torchrun][rank {rank}] worker loop exiting", file=sys.stderr, flush=True)

    def _enforce_cache_limit(self) -> None:
        while len(self.pipeline_cache) > self.max_cached_pipelines:
            self._evict_oldest_pipeline()

    def _reserve_pipeline_cache_slot(self) -> None:
        """Evict before loading so a cache miss never requires N+1 pipelines."""

        if self.max_cached_pipelines <= 0:
            while self.pipeline_cache:
                self._evict_oldest_pipeline()
            return
        while len(self.pipeline_cache) >= self.max_cached_pipelines:
            self._evict_oldest_pipeline()

    def _evict_oldest_pipeline(self) -> None:
        if not self.pipeline_cache:
            return
        for key, context in list(self.pipeline_cache.items()):
            if context.active_leases > 0:
                continue
            self.pipeline_cache.pop(key, None)
            self._dispose_pipeline_context(context)
            return
        raise RuntimeError(
            "All cached pipelines are serving active lazy streams; "
            "cannot evict one safely for a new model load."
        )

    def _lease_lazy_pipeline_result(
        self,
        context: PipelineContext,
        result: Any,
        *,
        lifecycle_lock_held: bool = False,
        lease_held: bool = False,
    ) -> Any:
        """Pin a context when model execution escapes as a lazy iterator."""

        if not isinstance(result, Iterator):
            return result
        if context.dispose_when_idle:
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise RuntimeError(f"{context.entry.display_name} is pending unload")
        if not lease_held:
            self._pin_pipeline_context(context)
        if not lifecycle_lock_held:
            context.lifecycle_lock.acquire()
        return _LeasedPipelineIterator(self, context, result)

    def _pin_pipeline_context(self, context: PipelineContext) -> None:
        with self.lock:
            if context.dispose_when_idle:
                raise RuntimeError(f"{context.entry.display_name} is pending unload")
            context.active_leases += 1

    def _release_pipeline_lease(self, context: PipelineContext) -> None:
        with self.lock:
            if context.active_leases > 0:
                context.active_leases -= 1
            if context.active_leases != 0:
                return
            cached = self.pipeline_cache.get(context.cache_key) is context
            if not context.dispose_when_idle and cached:
                return
            if cached:
                self.pipeline_cache.pop(context.cache_key, None)
            self._dispose_pipeline_context(context)

    def _dispose_pipeline_context(self, context: PipelineContext) -> None:
        """Run a pipeline's cleanup hook before dropping its final cached reference."""

        pipeline = context.pipeline
        context.pipeline = None
        context.state.clear()
        try:
            cleanup = getattr(pipeline, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception as exc:
            warnings.warn(
                f"Studio pipeline cleanup failed for {context.entry.model_id}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            del pipeline
            self._collect_device_memory()

    def _dispose_pipeline(self, pipeline: Any) -> None:
        """Compatibility helper for detached, caller-owned pipeline values."""

        if pipeline is None:
            return
        del pipeline
        self._collect_device_memory()

    def _collect_device_memory(self) -> None:
        gc.collect()
        torch = _torch_module()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def import_pipeline_class(self, entry: CatalogEntry) -> Any:
        # Pipeline modules frequently resolve Transformers/Diffusers lazy
        # exports at import time.  A different job may be doing the same from
        # its lazy runtime constructor, so protect the complete module import
        # rather than only each individual importlib call.
        with third_party_lazy_import_guard():
            module = importlib.import_module(entry.module_path)
            return getattr(module, entry.class_name)

    def runtime_driver_for(self, entry: CatalogEntry) -> BaseRuntimeDriver:
        return RUNTIME_DRIVERS.get(entry.runtime_kind, RUNTIME_DRIVERS["default"])

    def prepare_inputs(
        self,
        *,
        entry: CatalogEntry,
        prompt: str,
        input_path: str,
        image: Optional[Image.Image],
        video: Any,
        last_frame: Optional[Image.Image],
        reference_files: Any,
        interactions_text: str,
        camera_view_text: str,
        task_type: str,
        intrinsics_text: str,
        meta_path: str,
        panorama_path: str,
        scene_name: str,
        fps: int,
        num_frames: int,
        call_kwargs_text: str,
        load_kwargs_text: str,
        model_ref: str,
        backend: str,
        endpoint: str,
        api_key: str,
        device: str,
        infer_metadata: Optional[Dict[str, Any]] = None,
    ) -> PreparedInputs:
        run_seed = f"{_timestamp()}-{_slugify(entry.model_id)}"
        output_dir = str(_ensure_dir(Path(self.runs_root) / run_seed))
        output_suffix = _output_suffix_from_infer_metadata(entry, infer_metadata)
        output_path = str(Path(output_dir) / f"{_slugify(entry.model_id)}{output_suffix}")

        inputs_dir = _ensure_dir(Path(output_dir) / "inputs")
        image_path = _save_pil(image, inputs_dir / "main_image.png")
        video_path = None
        extracted_video = _extract_file_path(video)
        if extracted_video:
            video_path = _copy_file(extracted_video, inputs_dir / Path(extracted_video).name)
        raw_input_path = (input_path or "").strip()
        if raw_input_path and not image_path and not video_path:
            source = Path(raw_input_path).expanduser()
            if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
                image_path = _copy_file(str(source), inputs_dir / source.name)
            elif source.is_file() and source.suffix.lower() in VIDEO_EXTS:
                video_path = _copy_file(str(source), inputs_dir / source.name)
        last_frame_path = _save_pil(last_frame, inputs_dir / "last_frame.png")

        reference_paths = []
        reference_images: list[Image.Image] = []
        for index, raw_path in enumerate(_extract_multi_paths(reference_files)):
            copied = _copy_file(raw_path, inputs_dir / f"reference_{index:02d}{Path(raw_path).suffix}")
            reference_paths.append(copied)
            with Image.open(copied) as ref_image:
                reference_images.append(ref_image.convert("RGB"))

        variant_id = _infer_metadata_variant_id(infer_metadata)
        call_kwargs = {} if variant_id in _entry_extra_variant_ids(entry) else entry.default_call_kwargs.copy()
        call_kwargs.update(parse_jsonish(call_kwargs_text, default={}) or {})
        if raw_input_path:
            default_input_kwarg = entry.default_call_kwargs.get("input_path")
            current_input_kwarg = call_kwargs.get("input_path")
            current_missing = current_input_kwarg is None or (
                isinstance(current_input_kwarg, str) and current_input_kwarg == ""
            )
            current_is_default = (
                default_input_kwarg is not None
                and default_input_kwarg != ""
                and str(current_input_kwarg) == str(default_input_kwarg)
            )
            supports_input_kwarg = (
                "input_path" in set(entry.call_params)
                or "input_path" in set(entry.stream_params)
                or "input_path" in call_kwargs
            )
            if supports_input_kwarg and (current_missing or current_is_default):
                call_kwargs["input_path"] = raw_input_path
        load_kwargs = entry.default_load_kwargs.copy()
        load_kwargs.update(parse_jsonish(load_kwargs_text, default={}) or {})

        if entry.category in {"Embodied Action", "Visual Action"}:
            call_kwargs.setdefault("return_dict", True)
            call_kwargs.setdefault("run_dir", output_dir)

        interactions = parse_interactions(interactions_text)
        camera_view = parse_jsonish(camera_view_text, default=None)
        intrinsics = parse_jsonish(intrinsics_text, default=None)

        return PreparedInputs(
            prompt=(prompt or entry.default_prompt or "").strip(),
            input_path=raw_input_path,
            image=image.convert("RGB") if image is not None else None,
            image_path=image_path,
            video_path=video_path,
            last_frame=last_frame.convert("RGB") if last_frame is not None else None,
            last_frame_path=last_frame_path,
            reference_images=reference_images,
            reference_image_paths=reference_paths,
            interactions=interactions,
            camera_view=camera_view,
            task_type=(task_type or entry.default_task_type or "").strip(),
            intrinsics=intrinsics,
            meta_path=(meta_path or "").strip(),
            panorama_path=(panorama_path or "").strip(),
            scene_name=(scene_name or "").strip(),
            fps=int(fps or 16),
            num_frames=int(num_frames or 0),
            output_dir=output_dir,
            output_path=output_path,
            call_kwargs=call_kwargs,
            load_kwargs=load_kwargs,
            model_ref=(model_ref or entry.default_model_ref or "").strip(),
            backend=(backend or entry.default_backend or "auto").strip(),
            endpoint=(endpoint or entry.default_endpoint or "").strip(),
            api_key=(api_key or "").strip(),
            device=(device or "cuda").strip(),
            infer_metadata=dict(infer_metadata or {}),
        )

    def _prepare_run_request(
        self,
        *,
        model_id: str,
        prompt: str,
        input_path: str,
        image: Optional[Image.Image],
        video: Any,
        last_frame: Optional[Image.Image],
        reference_files: Any,
        interactions_text: str,
        camera_view_text: str,
        task_type: str,
        intrinsics_text: str,
        meta_path: str,
        panorama_path: str,
        scene_name: str,
        fps: int,
        num_frames: int,
        call_kwargs_text: str,
        load_kwargs_text: str,
        model_ref: str,
        backend: str,
        endpoint: str,
        api_key: str,
        device: str,
        infer_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> tuple[CatalogEntry, PreparedInputs, Dict[str, float], float]:
        entry = find_entry(model_id)
        perf_segments: Dict[str, float] = {}
        wall_t0 = time.perf_counter()
        if progress_callback is not None:
            progress_callback(0.08, "Normalizing inputs")
        t_prepare = time.perf_counter()
        request = self.prepare_inputs(
            entry=entry,
            prompt=prompt,
            input_path=input_path,
            image=image,
            video=video,
            last_frame=last_frame,
            reference_files=reference_files,
            interactions_text=interactions_text,
            camera_view_text=camera_view_text,
            task_type=task_type,
            intrinsics_text=intrinsics_text,
            meta_path=meta_path,
            panorama_path=panorama_path,
            scene_name=scene_name,
            fps=fps,
            num_frames=num_frames,
            call_kwargs_text=call_kwargs_text,
            load_kwargs_text=load_kwargs_text,
            model_ref=model_ref,
            backend=backend,
            endpoint=endpoint,
            api_key=api_key,
            device=device,
            infer_metadata=infer_metadata,
        )
        perf_segments["prepare_inputs_ms"] = (time.perf_counter() - t_prepare) * 1000.0
        return entry, request, perf_segments, wall_t0

    def run(
        self,
        *,
        model_id: str,
        action: str,
        prompt: str,
        input_path: str,
        image: Optional[Image.Image],
        video: Any,
        last_frame: Optional[Image.Image],
        reference_files: Any,
        interactions_text: str,
        camera_view_text: str,
        task_type: str,
        intrinsics_text: str,
        meta_path: str,
        panorama_path: str,
        scene_name: str,
        fps: int,
        num_frames: int,
        call_kwargs_text: str,
        load_kwargs_text: str,
        model_ref: str,
        backend: str,
        endpoint: str,
        api_key: str,
        device: str,
        infer_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        materialize: bool = True,
    ) -> Optional[RunRecord]:
        entry, request, perf_segments, wall_t0 = self._prepare_run_request(
            model_id=model_id,
            prompt=prompt,
            input_path=input_path,
            image=image,
            video=video,
            last_frame=last_frame,
            reference_files=reference_files,
            interactions_text=interactions_text,
            camera_view_text=camera_view_text,
            task_type=task_type,
            intrinsics_text=intrinsics_text,
            meta_path=meta_path,
            panorama_path=panorama_path,
            scene_name=scene_name,
            fps=fps,
            num_frames=num_frames,
            call_kwargs_text=call_kwargs_text,
            load_kwargs_text=load_kwargs_text,
            model_ref=model_ref,
            backend=backend,
            endpoint=endpoint,
            api_key=api_key,
            device=device,
            infer_metadata=infer_metadata,
        )

        if self._should_use_torchrun_lingbot_fast(entry, request):
            command = {
                "kind": "run_action",
                "model_id": entry.model_id,
                "action": action,
                "request": _prepared_inputs_payload(request),
            }
            t_cmd = time.perf_counter()
            result = self._run_torchrun_command(command)
            perf_segments["torchrun_execute_ms"] = (time.perf_counter() - t_cmd) * 1000.0
            perf_segments["total_client_ms"] = (time.perf_counter() - wall_t0) * 1000.0
            if not isinstance(result, RunRecord):
                raise RuntimeError("Torchrun LingBot fast command did not return a Studio run record.")
            self._persist_performance_metadata(result, perf_segments)
            return result

        record = self._run_local_prepared_request(
            entry=entry,
            action=action,
            request=request,
            progress_callback=progress_callback,
            perf_segments=perf_segments,
            materialize=materialize,
        )
        if record is None:
            return None
        perf_segments["total_client_ms"] = (time.perf_counter() - wall_t0) * 1000.0
        self._persist_performance_metadata(record, perf_segments)
        return record

    def run_realtime(
        self,
        *,
        entry: CatalogEntry,
        request: PreparedInputs,
        action: str,
    ) -> Any:
        """Run one resident interactive action without creating artifacts.

        Realtime frontends keep one prepared request and one cached pipeline for
        the lifetime of a session.  This path deliberately bypasses
        :meth:`materialize_run`: no MP4 encoding, artifact scan, preview
        selection, or manifest write is allowed on the control-to-frame hot
        path.
        """

        if self._should_use_torchrun_lingbot_fast(entry, request):
            return self._run_torchrun_command(
                {
                    "kind": "realtime_action",
                    "model_id": entry.model_id,
                    "action": action,
                    "request": _prepared_inputs_payload(request),
                }
            )

        driver = self.runtime_driver_for(entry)
        with self.lock:
            ctx = driver.load_pipeline(self, entry, request, progress_callback=None)
            self._pin_pipeline_context(ctx)

        ctx.lifecycle_lock.acquire()
        lifecycle_transferred = False

        def finish_realtime_result(value: Any) -> Any:
            nonlocal lifecycle_transferred
            wrapped = self._lease_lazy_pipeline_result(
                ctx,
                value,
                lifecycle_lock_held=True,
                lease_held=True,
            )
            lifecycle_transferred = isinstance(wrapped, _LeasedPipelineIterator)
            return wrapped

        try:
            if ctx.dispose_when_idle:
                raise RuntimeError(f"{ctx.entry.display_name} is pending unload")
            handled, realtime_result = self._run_native_realtime_pipeline_action(
                driver=driver,
                ctx=ctx,
                request=request,
                action=action,
            )
            if handled:
                return finish_realtime_result(realtime_result)
            if action in {"configure", "init"}:
                if action == "configure" and not driver.can_init(ctx, request):
                    return None
                seed_image = driver._seed_stream_state(ctx, request, force_reset=True)
                if seed_image is None:
                    raise ValueError(
                        f"{ctx.entry.display_name} expects an image before realtime INIT."
                    )
                return seed_image
            if action == "run":
                result = driver._invoke(
                    ctx,
                    request,
                    mode="run",
                    materialize_outputs=False,
                )
                driver._prime_memory_after_fresh(ctx, result)
                return finish_realtime_result(result)
            if action == "stream":
                if not ctx.entry.supports_stream or not hasattr(ctx.pipeline, "stream"):
                    raise RuntimeError(
                        f"{ctx.entry.display_name} does not expose realtime stream controls."
                    )
                driver._seed_stream_state(ctx, request, force_reset=False)
                stream_request = driver._prepare_stream_request(ctx, request)
                result = driver._invoke(
                    ctx,
                    stream_request,
                    mode="stream",
                    materialize_outputs=False,
                )
                if _supports_memory_stream(ctx.pipeline) or _pipeline_stream_state_ready(ctx.pipeline):
                    ctx.state["studio_stream_initialized"] = True
                return finish_realtime_result(result)
            if action == "reset":
                return driver.reset(self, ctx)
            raise ValueError(f"Unsupported realtime action: {action}")
        finally:
            if not lifecycle_transferred:
                ctx.lifecycle_lock.release()
                self._release_pipeline_lease(ctx)

    def _new_message_output_dir(self, model_id: str) -> str:
        run_seed = f"{_timestamp()}-{_slugify(model_id)}"
        return str(_ensure_dir(Path(self.runs_root) / run_seed))

    def make_message_record(
        self,
        entry: CatalogEntry,
        message: str,
        *,
        mode: str = "message",
        status: str = "ok",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        output_dir = self._new_message_output_dir(entry.model_id)
        metadata = {"message": message}
        if extra_metadata:
            metadata.update(extra_metadata)
        manifest_path = str(
            _core_write_json(
                Path(output_dir) / "manifest.json",
                _redact_manifest_secrets(
                    {
                        "message": message,
                        "mode": mode,
                        "status": status,
                        "model_id": entry.model_id,
                        "display_name": entry.display_name,
                        "output_dir": output_dir,
                        "metadata": metadata,
                    }
                ),
                atomic=True,
            )
        )
        self._invalidate_recent_runs_cache()
        return RunRecord(
            run_id=Path(output_dir).name,
            model_id=entry.model_id,
            display_name=entry.display_name,
            mode=mode,
            status=status,
            output_dir=output_dir,
            manifest_path=manifest_path,
            metadata=metadata,
        )

    def _make_message_record(self, entry: CatalogEntry, request: PreparedInputs, message: str) -> RunRecord:
        record = self.make_message_record(entry, message)
        if record.output_dir != request.output_dir:
            _ensure_dir(Path(request.output_dir))
        return record

    def reset_cached_model(self, model_id: str) -> str:
        if _torchrun_lingbot_fast_enabled() and model_id in {
            LINGBOT_WORLD_MODEL_ID,
            LINGBOT_WORLD_V2_MODEL_ID,
            MATRIX_GAME3_MODEL_ID,
            LONGVIE2_MODEL_ID,
            HELIOS_MODEL_ID,
            DREAMX_WORLD_MODEL_ID,
        }:
            command = {"kind": "reset_model", "model_id": model_id}
            result = self._run_torchrun_command(command)
            return str(result or "")
        return self._reset_cached_model_local(model_id)

    def materialize_run(
        self,
        ctx: PipelineContext,
        request: PreparedInputs,
        *,
        result: Any,
        mode: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        materialize_t0 = time.perf_counter()
        materialize_timings: Dict[str, float] = {}

        def record_materialize_timing(key: str, start: float) -> None:
            materialize_timings[key] = (time.perf_counter() - start) * 1000.0

        t_result = time.perf_counter()
        output_dir = request.output_dir
        metadata = {"result_type": type(result).__name__}
        if extra_metadata:
            metadata.update(extra_metadata)

        saved_artifacts: list[str] = []
        result_metadata_path = str(
            _core_write_json(
                Path(output_dir) / "result_metadata.json",
                _redact_manifest_secrets(metadata),
                atomic=True,
            )
        )
        saved_artifacts.append(result_metadata_path)

        if hasattr(result, "save") and not isinstance(result, (dict, str, bytes)):
            try:
                saved_artifacts.extend(result.save(output_dir))
            except Exception:
                pass

        if isinstance(result, dict):
            if "scene_params" in result and hasattr(ctx.pipeline, "save_results"):
                exported = ctx.pipeline.save_results(
                    result,
                    output_dir,
                    video_path=request.output_path,
                )
                if isinstance(exported, Mapping):
                    saved_artifacts.extend(
                        str(path)
                        for path in exported.values()
                        if isinstance(path, str) and Path(path).exists()
                    )
            preview_images = _save_preview_image_sequence(
                result.get("depth_visualizations"),
                output_dir,
                "preview_depth",
            )
            if not preview_images:
                preview_images = _save_preview_image_sequence(
                    result.get("processed_images"),
                    output_dir,
                    "preview_image",
                    max_images=4,
                )
            saved_artifacts.extend(preview_images)
            for key in ("generated_video_path", "output_path", "preview_video", "video_path"):
                value = result.get(key)
                if isinstance(value, str) and Path(value).exists():
                    value = _persist_video_artifact_in_run(
                        value,
                        output_dir=output_dir,
                        output_path=request.output_path,
                    )
                    result[key] = value
                    saved_artifacts.append(value)
            for key in (
                "artifact_path",
                "model_path",
                "mesh_path",
                "point_cloud_path",
                "pointcloud_path",
                "splat_path",
                "gaussian_splat_path",
                "scene_path",
                "glb_path",
                "ply_path",
                "rrd_path",
                "visualization_artifact_path",
            ):
                value = result.get(key)
                if isinstance(value, str) and Path(value).exists():
                    value = _persist_video_artifact_in_run(
                        value,
                        output_dir=output_dir,
                        output_path=request.output_path,
                    )
                    result[key] = value
                    saved_artifacts.append(value)
            for key in ("artifact_paths", "artifacts"):
                values = result.get(key)
                if isinstance(values, (list, tuple)):
                    saved_artifacts.extend(
                        str(path) for path in values if isinstance(path, (str, Path)) and Path(path).exists()
                    )
            for key in ("preview_image", "first_frame"):
                value = result.get(key)
                if isinstance(value, str) and Path(value).exists():
                    saved_artifacts.append(value)
            # Pipelines that return both an already encoded video path and the
            # source tensor have finished materializing their output. Re-encoding
            # that tensor can overwrite the valid artifact at the same path, and
            # is especially unsafe for model-specific sample ranges/layouts.
            has_encoded_video = any(
                Path(path).suffix.lower() in VIDEO_EXTS and Path(path).is_file()
                for path in saved_artifacts
            )
            if not has_encoded_video:
                for key in ("sr_videos", "videos", "frames", "video"):
                    if key not in result:
                        continue
                    frames = _normalize_frame_list(result[key])
                    if frames:
                        saved_artifacts.append(
                            export_frames_to_video(frames, request.output_path, fps=request.fps)
                        )
                        break
            if "recon_info" in result:
                recon_path = str(
                    _core_write_json(
                        Path(output_dir) / "recon_info.json",
                        _redact_manifest_secrets(result["recon_info"]),
                        atomic=True,
                    )
                )
                saved_artifacts.append(recon_path)
            if ctx.entry.category in {"Embodied Action", "Visual Action"}:
                action_trace_path = _write_canonical_action_trace(output_dir, result)
                if action_trace_path:
                    saved_artifacts.append(action_trace_path)
            manifest_data = _safe_json(result)
        else:
            frames = _normalize_frame_list(result)
            if frames:
                saved_artifacts.append(export_frames_to_video(frames, request.output_path, fps=request.fps))
                first_frame_path = str(Path(output_dir) / "first_frame.png")
                Image.fromarray(_to_uint8_rgb(frames[0])).save(first_frame_path)
                saved_artifacts.append(first_frame_path)
            elif isinstance(result, Image.Image):
                image_path = str(Path(output_dir) / "preview.png")
                result.save(image_path)
                saved_artifacts.append(image_path)
            elif isinstance(result, str) and Path(result).exists():
                saved_artifacts.append(result)
            manifest_data = _safe_json(result)

        record_materialize_timing("materialize_result_ms", t_result)

        saved_artifacts = _existing_artifact_paths(saved_artifacts)
        validate_preview_videos = _env_flag(PREVIEW_VIDEO_VALIDATE_ENV)
        t_preview = time.perf_counter()
        previews = pick_preview_assets(saved_artifacts, validate_videos=validate_preview_videos)
        record_materialize_timing("materialize_preview_select_ms", t_preview)

        if _should_scan_artifacts(request=request, saved_artifacts=saved_artifacts, previews=previews):
            t_scan = time.perf_counter()
            artifacts = collect_artifact_paths(output_dir)
            record_materialize_timing("materialize_artifact_scan_ms", t_scan)
            saved_artifacts = _existing_artifact_paths([*saved_artifacts, *artifacts], check_files=False)
            t_preview = time.perf_counter()
            previews = pick_preview_assets(saved_artifacts, validate_videos=validate_preview_videos)
            materialize_timings["materialize_preview_select_ms"] += (time.perf_counter() - t_preview) * 1000.0
        else:
            materialize_timings["materialize_artifact_scan_ms"] = 0.0

        t_video_frame = time.perf_counter()
        interactive_video_state = bool(ctx.entry.supports_stream and mode in {"run", "stream"})
        should_extract_video_frame = _env_flag(VIDEO_PREVIEW_IMAGE_ENV)
        if should_extract_video_frame and previews["preview_video"] and (
            interactive_video_state
            or previews["preview_image"] is None
            or _artifact_is_input(previews["preview_image"])
        ):
            frame_position = "last" if interactive_video_state else "first"
            preview_image = maybe_extract_video_preview_image(
                previews["preview_video"],
                output_dir,
                frame_position=frame_position,
            )
            if preview_image and preview_image not in saved_artifacts:
                saved_artifacts.append(preview_image)
                saved_artifacts = _existing_artifact_paths(saved_artifacts)
                previews = pick_preview_assets(saved_artifacts, validate_videos=validate_preview_videos)
        record_materialize_timing("materialize_video_frame_extract_ms", t_video_frame)

        # Inference persists model outputs. Viewer conversion and viewport
        # manifests belong to the optional presentation adapter.
        preview_model = previews["preview_model"]
        rrd_path = previews["rrd_path"]
        materialize_timings["materialize_preview_convert_ms"] = 0.0

        record_status = _status_from_model_result(result)
        missing_required = _missing_required_inference_outputs(
            request,
            {
                **previews,
                "preview_model": preview_model,
                "rrd_path": rrd_path,
            },
        )
        if missing_required and record_status in {"blocked", "failed"} and isinstance(result, Mapping):
            manifest_path = result.get("plan_path") or result.get("artifact_path") or result.get("metadata", {}).get("log_path")
            if isinstance(manifest_path, str) and manifest_path and Path(manifest_path).is_file():
                saved_artifacts.append(manifest_path)
        if missing_required and record_status not in {"blocked", "cancelled", "failed"}:
            raise RuntimeError(
                f"{ctx.entry.display_name} did not materialize required inference output(s): "
                f"{', '.join(missing_required)}."
            )

        metadata_payload: Dict[str, Any] = {
            "entry": {
                "model_id": ctx.entry.model_id,
                "display_name": ctx.entry.display_name,
                "module_path": ctx.entry.module_path,
                "class_name": ctx.entry.class_name,
            },
            "request": {
                "prompt": request.prompt,
                "task_type": request.task_type,
                "interactions": _safe_json(request.interactions),
                "camera_view": _safe_json(request.camera_view),
                "call_kwargs": _safe_json(request.call_kwargs),
                "load_kwargs": _safe_json(request.load_kwargs),
                "model_ref": request.model_ref,
                "backend": request.backend,
                "endpoint": request.endpoint,
                "device": request.device,
                "infer_contract": _safe_json(request.infer_metadata),
            },
            "result": manifest_data,
            **metadata,
        }
        if ctx.entry.supports_stream:
            metadata_payload["studio_interactive"] = {
                "streaming_model": True,
                "memory_ready_after_step": bool(ctx.state.get("studio_stream_initialized")),
                "interaction_tokens_preview": _interaction_tokens_summary(request.interactions),
                "step_kind": mode,
            }

        artifact_list = sorted(saved_artifacts)
        materialize_timings["materialize_viewports_ms"] = 0.0
        materialize_timings["materialize_total_ms"] = (time.perf_counter() - materialize_t0) * 1000.0
        metadata_payload["studio_performance"] = {
            key: round(float(value), 3)
            for key, value in materialize_timings.items()
        }

        record = RunRecord(
            run_id=Path(output_dir).name,
            model_id=ctx.entry.model_id,
            display_name=ctx.entry.display_name,
            mode=mode,
            status=record_status,
            output_dir=output_dir,
            manifest_path=str(Path(output_dir) / "manifest.json"),
            preview_video=previews["preview_video"],
            preview_image=previews["preview_image"],
            preview_splat=previews["preview_splat"],
            preview_model=preview_model,
            gallery=previews["gallery"],
            rrd_path=rrd_path,
            artifacts=artifact_list,
            metadata=metadata_payload,
        )
        if self.record_processor is not None:
            record = self.record_processor(record, ctx.entry)
        _core_write_json(
            Path(record.manifest_path),
            record.to_manifest(),
            atomic=True,
        )
        self._invalidate_recent_runs_cache()
        return record

    def list_recent_runs(self, limit: int = 24) -> list[RunRecord]:
        requested = max(0, int(limit))
        if requested == 0:
            return []

        runs_root = Path(self.runs_root)
        try:
            root_stat = runs_root.stat()
            root_signature = (root_stat.st_dev, root_stat.st_ino, root_stat.st_mtime_ns)
        except OSError:
            return []

        # A few historical tests call this method with a SimpleNamespace. Keep
        # that lightweight usage working while real managers use the cache.
        cache_lock = getattr(self, "_recent_runs_lock", None)
        if cache_lock is None:
            cache_lock = threading.RLock()
            self._recent_runs_lock = cache_lock
            self._recent_runs_root_signature = None
            self._recent_run_manifests = ()
            self._recent_run_manifest_cursor = 0
            self._recent_runs_cache = ()

        with cache_lock:
            cache_current = self._recent_runs_root_signature == root_signature
            if cache_current and (
                len(self._recent_runs_cache) >= requested
                or self._recent_run_manifest_cursor >= len(self._recent_run_manifests)
            ):
                return list(self._recent_runs_cache[:requested])

            if not cache_current:
                # Studio is the writer for this tree: manager-owned manifest
                # replacements explicitly invalidate above, while directory
                # mtime catches additions/removals. Direct load_run bypasses the
                # cache, so exact lookups always observe their current file.
                manifests: list[tuple[int, Path]] = []
                for manifest_path in runs_root.glob("*/manifest.json"):
                    try:
                        manifests.append((manifest_path.stat().st_mtime_ns, manifest_path))
                    except OSError:
                        continue
                manifests.sort(key=lambda item: item[0], reverse=True)

                # Directory additions/removals update runs_root mtime. If it
                # moved during the scan, leave the index invalid so the next
                # call retries rather than treating a partial index as stable.
                try:
                    final_stat = runs_root.stat()
                    final_signature = (final_stat.st_dev, final_stat.st_ino, final_stat.st_mtime_ns)
                except OSError:
                    final_signature = None
                self._recent_runs_root_signature = (
                    final_signature if final_signature == root_signature else None
                )
                self._recent_run_manifests = tuple(path for _, path in manifests)
                self._recent_run_manifest_cursor = 0
                self._recent_runs_cache = ()

            records = list(self._recent_runs_cache)
            while (
                len(records) < requested
                and self._recent_run_manifest_cursor < len(self._recent_run_manifests)
            ):
                manifest_path = self._recent_run_manifests[self._recent_run_manifest_cursor]
                self._recent_run_manifest_cursor += 1
                try:
                    records.append(_run_record_from_manifest(manifest_path))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            self._recent_runs_cache = tuple(records)
            return list(self._recent_runs_cache[:requested])

    def _resolved_run_dir(self, run_id: str) -> Path:
        value = str(run_id or "")
        if not value or value in {".", ".."} or "/" in value or "\\" in value or Path(value).name != value:
            raise KeyError(f"Unknown Studio run id: {run_id}")

        runs_root = Path(self.runs_root).resolve()
        run_dir = runs_root / value
        try:
            resolved_run_dir = run_dir.resolve()
        except (OSError, RuntimeError):
            raise KeyError(f"Unknown Studio run id: {run_id}") from None
        if resolved_run_dir.parent != runs_root:
            raise KeyError(f"Unknown Studio run id: {run_id}")
        return resolved_run_dir

    def delete_run(self, run_id: str) -> Path:
        """Remove one persisted Studio run directory from ``runs_root``."""

        resolved_run_dir = self._resolved_run_dir(run_id)
        if not resolved_run_dir.is_dir():
            raise KeyError(f"Unknown Studio run id: {run_id}")
        shutil.rmtree(resolved_run_dir)
        self._invalidate_recent_runs_cache()
        return resolved_run_dir

    def load_run(self, run_id: str) -> RunRecord:
        resolved_run_dir = self._resolved_run_dir(run_id)
        resolved_manifest = resolved_run_dir / "manifest.json"
        try:
            resolved_manifest = resolved_manifest.resolve()
        except (OSError, RuntimeError):
            raise KeyError(f"Unknown Studio run id: {run_id}") from None
        if resolved_manifest.parent != resolved_run_dir:
            raise KeyError(f"Unknown Studio run id: {run_id}")
        try:
            return _run_record_from_manifest(resolved_manifest, run_id=str(run_id))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raise KeyError(f"Unknown Studio run id: {run_id}") from None

    def unload(self, model_id: Optional[str] = None) -> str:
        if _torchrun_lingbot_fast_enabled() and model_id in {
            None,
            LINGBOT_WORLD_MODEL_ID,
            LINGBOT_WORLD_V2_MODEL_ID,
            MATRIX_GAME3_MODEL_ID,
            LONGVIE2_MODEL_ID,
            HELIOS_MODEL_ID,
            DREAMX_WORLD_MODEL_ID,
        }:
            command = {"kind": "unload_model", "model_id": model_id}
            result = self._run_torchrun_command(command)
            return str(result or "")
        message = self._unload_local(model_id)
        return message


def recent_runs_table(records: Sequence[RunRecord]) -> list[list[str]]:
    rows = []
    for record in records:
        rows.append(
            [
                record.run_id,
                record.display_name,
                record.mode,
                record.status,
                Path(record.output_dir).name,
            ]
        )
    return rows

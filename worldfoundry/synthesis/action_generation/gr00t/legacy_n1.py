"""Isolated adapter for NVIDIA's original GR00T-N1 checkpoint format.

The in-tree GR00T policy implements N1.7. Original N1 uses a different Eagle2
backbone and modality transform, so its pinned official source runs in a
separate Python environment instead of sharing the N1.7 import graph.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io import file_sha256


N1_RELEASE_COMMIT = "755876a9afdb41ca6eb6383b36f4a0adb085c73f"


def describe_n1_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Inspect the released single-file N1 checkpoint without N1.7 assumptions."""
    root = Path(checkpoint_dir).expanduser().resolve()
    for name in ("config.json", "model.safetensors", "experiment_cfg/metadata.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"GR00T-N1 checkpoint file is missing: {root / name}")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "gr00t_n1":
        raise ValueError(f"Not an original GR00T-N1 checkpoint: {root}")
    metadata = json.loads((root / "experiment_cfg/metadata.json").read_text())
    return {
        "model_type": "gr00t_n1",
        "architectures": config.get("architectures", []),
        "backend": "official_n1_release_subprocess",
        "embodiments": sorted(metadata),
        "action_horizon": config.get("action_horizon"),
        "action_dim": config.get("action_dim"),
        "source_revision": N1_RELEASE_COMMIT,
    }


def _official_source(path: str | Path | None) -> Path:
    chosen = path or os.environ.get("WORLDFOUNDRY_GR00T_N1_SOURCE")
    if not chosen:
        raise FileNotFoundError(
            "Original GR00T-N1 requires the NVIDIA Isaac-GR00T n1-release checkout; "
            "pass n1_source_dir or set WORLDFOUNDRY_GR00T_N1_SOURCE."
        )
    root = Path(chosen).expanduser().resolve()
    for name in ("gr00t/model/gr00t_n1.py", "gr00t/model/policy.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"GR00T-N1 official source file is missing: {root / name}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    revision = result.stdout.strip()
    if revision != N1_RELEASE_COMMIT:
        raise ValueError(
            f"GR00T-N1 source revision is {revision}; expected n1-release {N1_RELEASE_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "gr00t"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"GR00T-N1 official source has local changes under {root / 'gr00t'}")
    return root


def _input_state(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise ValueError("Original GR00T-N1 requires a gr00t_observation state mapping.")
    state = observation.get("state")
    if state is None:
        state = {key: value for key, value in observation.items() if str(key).startswith("state.")}
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Original GR00T-N1 requires named GR1 state arrays.")
    result = {}
    for key, value in state.items():
        full_key = str(key) if str(key).startswith("state.") else f"state.{key}"
        result[full_key] = value.tolist() if hasattr(value, "tolist") else value
    return result


def _input_images(image: Any, observation: Mapping[str, Any] | None) -> Any:
    if isinstance(observation, Mapping):
        camera_views = observation.get("camera_views")
        if camera_views is None:
            camera_views = observation.get("video")
        if camera_views is not None:
            image = camera_views
    if isinstance(image, Mapping):
        if not image:
            raise ValueError("Original GR00T-N1 needs at least one named camera view.")
        views = {}
        for key, value in image.items():
            name = str(key)
            name = name if name.startswith("video.") else f"video.{name}"
            if name in views:
                raise ValueError(f"Duplicate original GR00T-N1 camera view: {name}")
            views[name] = value
        return views
    return image


def predict_n1_action(
    *,
    checkpoint_dir: str | Path,
    source_dir: str | Path | None,
    python_executable: str | Path | None,
    data_config_name: str,
    embodiment_tag: str,
    device: str,
    seed: int,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any] | None,
    output_path: str | Path,
    run_dir: str | Path,
    extra_metadata: Mapping[str, Any] | None = None,
    timeout_seconds: int = 21600,
) -> dict[str, Any]:
    """Run official N1 in a version-pinned subprocess and return its action trace."""
    import numpy as np

    from worldfoundry.synthesis.action_generation.gr00t.runtime import _load_image_array

    source = _official_source(source_dir)
    python = python_executable or os.environ.get("WORLDFOUNDRY_GR00T_N1_PYTHON")
    if not python:
        raise ValueError(
            "Original GR00T-N1 requires an isolated Python with transformers==4.45.2; "
            "pass n1_python or set WORLDFOUNDRY_GR00T_N1_PYTHON."
        )
    # Keep the venv entrypoint path: resolving its symlink would select the
    # base interpreter and silently lose the pinned Transformers installation.
    python = Path(python).expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f"GR00T-N1 Python executable is missing: {python}")
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    describe_n1_checkpoint(checkpoint)
    image_inputs = _input_images(image, observation)
    state = _input_state(observation)
    work_parent = Path(run_dir).expanduser().resolve()
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="n1-official-", dir=work_parent))
    image_paths = {}
    image_sources = {}
    views = image_inputs if isinstance(image_inputs, Mapping) else {"__single__": image_inputs}
    for index, (key, value) in enumerate(views.items()):
        image_array, image_origin = _load_image_array(value)
        image_path = work / f"image-{index}.npy"
        np.save(image_path, image_array, allow_pickle=False)
        image_paths[key] = str(image_path)
        image_sources[key] = image_origin
    image_source = image_sources["__single__"] if "__single__" in image_sources else json.dumps(image_sources)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "checkpoint_dir": str(checkpoint),
        "output_path": str(target),
        "image_paths": image_paths,
        "image_source": image_source,
        "state": state,
        "instruction": instruction,
        "data_config_name": data_config_name,
        "embodiment_tag": embodiment_tag,
        "device": device,
        "seed": seed,
        "extra_metadata": dict(extra_metadata or {}),
        "source_revision": N1_RELEASE_COMMIT,
    }
    request_path = work / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, default=str, indent=2) + "\n")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(source), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    worker = Path(__file__).with_name("legacy_n1_worker.py")
    started = time.monotonic()
    with (work / "worker.log").open("w") as log:
        completed = subprocess.run(
            [str(python), str(worker), str(request_path)],
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        worker_status = work / "status.json"
        details = json.loads(worker_status.read_text()) if worker_status.is_file() else {}
        tail = "\n".join((work / "worker.log").read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(
            f"Official GR00T-N1 worker failed (exit {completed.returncode}): "
            f"{details.get('error', 'no status')}\n{tail}"
        )
    if not target.is_file():
        raise FileNotFoundError(f"Official GR00T-N1 worker did not write {target}")
    payload = json.loads(target.read_text())
    if payload.get("status") != "success":
        raise ValueError(f"Official GR00T-N1 worker returned {payload.get('status')}")
    return {
        "status": "success",
        "model_id": "gr00t",
        "artifact_kind": "action_trace",
        "artifact_path": str(target),
        "artifact_sha256": file_sha256(target),
        "backend": "worldfoundry.gr00t.official_n1_release_subprocess",
        "backend_quality": "official_checkpoint_wrapper",
        "duration_seconds": round(time.monotonic() - started, 3),
        "worker_dir": str(work),
    }

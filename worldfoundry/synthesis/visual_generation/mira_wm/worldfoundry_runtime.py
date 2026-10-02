"""Workspace command adapter for the vendored MIRA inference entrypoint."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from .infer import _load_json, _normalize_actions, _resolve_checkpoint

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
RUNTIME_SRC = RUNTIME_DIR / "src"
BLOCKED_REASON = ""


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dataset_index(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path / "index.json" if path.is_dir() else path


def _actions_payload(options: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions_file = _option(options, "actions_file", "action_path")
    if actions_file:
        return _normalize_actions(_load_json(Path(str(actions_file))))
    raw_actions = options.get("actions")
    if isinstance(raw_actions, str):
        raw_actions = json.loads(raw_actions) if raw_actions.strip() else None
    return _normalize_actions(raw_actions)


def _actions_path(options: Mapping[str, Any], output_dir: Path) -> Path | None:
    actions_file = _option(options, "actions_file", "action_path")
    if actions_file:
        return Path(str(actions_file)).expanduser().resolve()
    actions = _actions_payload(options)
    if not actions:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mira.actions.json"
    path.write_text(json.dumps(actions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    """Return actionable preflight failures before Workspace starts MIRA."""
    del runtime_root, profile
    options = dict(options or {})
    missing: list[dict[str, str]] = []

    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append(
            {"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "MIRA infer.py is missing"}
        )

    checkpoint = _option(options, "checkpoint", "checkpoint_path", "checkpoint_dir", "model_path")
    if not checkpoint:
        missing.append(
            {
                "kind": "checkpoint",
                "path": "checkpoint_path",
                "reason": "MIRA requires a local checkpoint file or run directory",
            }
        )
    else:
        try:
            _resolve_checkpoint(str(checkpoint))
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            missing.append({"kind": "checkpoint", "path": str(checkpoint), "reason": str(exc)})

    dataset = _option(options, "dataset", "dataset_path", "data_path")
    if not dataset:
        missing.append(
            {
                "kind": "dataset",
                "path": "dataset_path",
                "reason": "MIRA requires a local Rocket Science split",
            }
        )
    else:
        index_path = _dataset_index(str(dataset))
        if not index_path.is_file():
            missing.append(
                {
                    "kind": "dataset",
                    "path": str(index_path),
                    "reason": "Rocket Science index.json does not exist",
                }
            )

    try:
        _actions_payload(options)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        actions_path = str(_option(options, "actions_file", "action_path", default="actions"))
        missing.append({"kind": "input", "path": actions_path, "reason": f"Invalid MIRA actions: {exc}"})

    for module_name in ("torch", "numpy", "einops", "pydantic", "omegaconf", "PIL", "tqdm", "imageio"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required MIRA runtime package {module_name!r} is not importable",
                }
            )

    decoder_available = False
    torchcodec_error = "not installed"
    if importlib.util.find_spec("torchcodec") is not None:
        try:
            importlib.import_module("torchcodec")
            decoder_available = True
        except Exception as exc:  # noqa: BLE001 - native-loader errors must be surfaced in the plan.
            torchcodec_error = str(exc).splitlines()[0] or type(exc).__name__
    if not decoder_available and importlib.util.find_spec("av") is not None:
        try:
            importlib.import_module("av")
            decoder_available = True
        except Exception:  # noqa: BLE001 - report both decoder routes together below.
            pass
    if not decoder_available:
        missing.append(
            {
                "kind": "python_module",
                "path": "torchcodec|av",
                "reason": f"MIRA requires TorchCodec or PyAV; TorchCodec failed with: {torchcodec_error}",
            }
        )

    device = str(_option(options, "device", default="cuda"))
    if device.startswith("cuda") and importlib.util.find_spec("torch") is not None:
        try:
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - the import failure is reported as an unavailable device.
            cuda_available = False
        if not cuda_available:
            missing.append(
                {
                    "kind": "device",
                    "path": device,
                    "reason": "MIRA requires an available NVIDIA CUDA device",
                }
            )
    return missing


def build_command(context):
    """Build the standalone ``infer.py`` command used by Workspace."""
    options = dict(context.get("options") or {})
    checkpoint = _option(options, "checkpoint", "checkpoint_path", "checkpoint_dir", "model_path")
    dataset = _option(options, "dataset", "dataset_path", "data_path")
    if not checkpoint or not dataset:
        raise ValueError("MIRA Workspace execution requires checkpoint_path and dataset_path")

    command = [
        str(context["python"]),
        str(context["entrypoint"]),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--output",
        str(context["output_path"]),
        "--device",
        str(_option(options, "device", default=context.get("device") or "cuda")),
        "--seed",
        str(_option(options, "seed", default=42)),
        "--clip-index",
        str(_option(options, "clip_index", default=0)),
        "--n-context-frames",
        str(_option(options, "n_context_frames", default=38)),
        "--num-unrolled-frames",
        str(_option(options, "num_unrolled_frames", default=20)),
        "--n-diffusion-steps",
        str(_option(options, "n_diffusion_steps", default=10)),
        "--schedule-type",
        str(_option(options, "schedule_type", default="linear")),
        "--noise-level",
        str(_option(options, "noise_level", default=0.0)),
    ]
    actions_path = _actions_path(options, Path(str(context["output_dir"])))
    if actions_path is not None:
        command.extend(["--actions-file", str(actions_path)])
    if _as_bool(options.get("compile")):
        command.append("--compile")
    if not _as_bool(options.get("overlay_actions"), default=True):
        command.append("--no-overlay-actions")
    if _as_bool(options.get("generated_only")):
        command.append("--generated-only")
    return command


def pythonpath_entries(*, runtime_root, options, profile):
    del runtime_root, options, profile
    return (RUNTIME_SRC,)


__all__ = [
    "BLOCKED_REASON",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "RUNTIME_SRC",
    "build_command",
    "missing_requirements",
    "pythonpath_entries",
]

"""Subprocess adapters for the two independent projects in JoyAI-Echo.

The upstream repository intentionally gives Echo-LongVideo and Echo-WM separate
Python environments.  These adapters preserve that boundary: WorldFoundry owns
request normalization, preflight, launch planning, logs, and artifact discovery,
while the selected upstream interpreter executes the official entrypoint.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from worldfoundry.core.io import resolve_data_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.runtime.env import resolve_hfd_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_REVISION = "0b931fd42f5ef51410019ae41b261eb3b8936410"
WM_UPSTREAM_REVISION = "a08f1274573d7cd6ec66719f204951f4227a5173"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "cache" / "generative_taxonomy" / "jd-opensource--JoyAI-Echo"
DEFAULT_HFD_ROOT = resolve_hfd_root()

LONGVIDEO_CHECKPOINT_NAMES = {
    "bf16": "echo15_full_dmd",
    "fp8": "echo15_fp8",
    "fp4": "echo15_fp4",
}
WM_CHECKPOINT_NAMES = {
    "base": "echo-wm-base.safetensors",
    "flash": "echo-wm-flash.safetensors",
}


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    expanded = os.path.expandvars(str(value))
    return Path(expand_worldfoundry_path(expanded)).expanduser().resolve()


def _python_path(value: str | Path | None) -> Path:
    if value is None or not str(value).strip():
        return Path(sys.executable).absolute()
    text = os.path.expandvars(str(value))
    if Path(text).is_absolute() or os.sep in text:
        return Path(text).expanduser().absolute()
    discovered = shutil.which(text)
    return Path(discovered).absolute() if discovered else Path(text).expanduser()


def _first_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and value:
        return _first_path(value[0])
    return None


def _append_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        command.extend((flag, str(value)))


def _append_bool(command: list[str], flag: str, value: Any) -> None:
    if bool(value):
        command.append(flag)


def _safe_slug(value: Any, fallback: str) -> str:
    text = "".join(char if str(char).isalnum() or char in "-_" else "-" for char in str(value or ""))
    return text.strip("-") or fallback


def _missing_paths(paths: Sequence[Path]) -> tuple[str, ...]:
    return tuple(str(path) for path in paths if not path.exists())


def _resolved_resource(value: Any, *, field_name: str) -> str | None:
    text = _first_path(value)
    if text is None or not text.strip():
        return None
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https", "data", "file"}:
        return text
    if parsed.scheme:
        raise ValueError(f"{field_name} uses unsupported URL scheme {parsed.scheme!r}.")
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} not found: {path}")
    return str(path)


def _cuda_visible_devices(device: str) -> str | None:
    """Resolve explicit CUDA indices without importing torch during catalog use."""
    text = str(device).strip().lower()
    if not text.startswith("cuda:"):
        return None
    visible = text.split(":", 1)[1].strip()
    if not visible:
        return None
    indices = [item.strip() for item in visible.split(",")]
    return ",".join(indices) if all(item.isdigit() for item in indices) else None


def _runtime_env(device: str, source_root: Path) -> dict[str, str]:
    python_path = os.pathsep.join(value for value in (str(source_root), os.environ.get("PYTHONPATH")) if value)
    env = {
        "PYTHONPATH": python_path,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # The upstream overlay otherwise defaults to /usr/bin/ffprobe, which
    # need not exist in conda and user-local installations.
    for tool in ("ffmpeg", "ffprobe"):
        variable = f"{tool.upper()}_BIN"
        resolved = shutil.which(os.environ.get(variable) or tool)
        if resolved:
            env[variable] = resolved
    visible_devices = _cuda_visible_devices(device)
    if visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    return env


@dataclass(frozen=True)
class JoyAIEchoRuntimePlan:
    """Serializable launch plan for one official JoyAI-Echo subprocess."""

    project: str
    variant: str
    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    output_path: str
    output_root: str
    request_path: str | None = None
    upstream_revision: str = UPSTREAM_REVISION

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "variant": self.variant,
            "command": list(self.command),
            "env": dict(self.env),
            "workdir": self.workdir,
            "output_path": self.output_path,
            "output_root": self.output_root,
            "request_path": self.request_path,
            "upstream_revision": self.upstream_revision,
        }


class _JoyAIEchoSubprocessRuntime:
    """Shared official-subprocess execution and artifact handling."""

    project_name = "joyai_echo"

    def _artifact_candidates(self, plan: JoyAIEchoRuntimePlan, started_at: float) -> list[Path]:
        requested = Path(plan.output_path)
        if plan.project == "echo_wm":
            candidates = [requested]
        else:
            root = Path(plan.output_root)
            candidates = sorted(root.rglob("result.mp4")) if root.is_dir() else []
        return [
            path
            for path in candidates
            if path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime >= started_at - 2.0
        ]

    def run_plan(
        self,
        plan: JoyAIEchoRuntimePlan,
        *,
        timeout_seconds: int = 7200,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        target = Path(plan.output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target_log_dir = Path(log_dir or target.parent).expanduser().resolve()
        target_log_dir.mkdir(parents=True, exist_ok=True)
        log_prefix = "joyai_echo_longvideo" if plan.project == "echo_longvideo" else "joyai_echo_wm"
        stdout_path = target_log_dir / f"{log_prefix}_stdout.log"
        stderr_path = target_log_dir / f"{log_prefix}_stderr.log"
        started = time.monotonic()
        started_at = time.time()
        returncode = -1
        error: str | None = None
        env = os.environ.copy()
        env.update(plan.env)
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    list(plan.command),
                    cwd=plan.workdir,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            returncode = completed.returncode
        except Exception as exc:  # pragma: no cover - deployment/runtime dependent
            error = str(exc)

        candidates = self._artifact_candidates(plan, started_at)
        generated_files: list[str] = []
        if returncode == 0 and candidates:
            source = max(candidates, key=lambda path: path.stat().st_mtime)
            if source.resolve() != target:
                shutil.copy2(source, target)
            generated_files.append(str(target))
            if plan.project == "echo_wm":
                overlay = source.with_name(f"{source.stem}_action{source.suffix}")
                if overlay.is_file() and overlay.stat().st_size > 0:
                    generated_files.append(str(overlay.resolve()))

        ok = returncode == 0 and bool(generated_files)
        if returncode != 0 and error is None:
            error = f"JoyAI-Echo {plan.variant} runner exited with code {returncode}."
        elif returncode == 0 and not generated_files and error is None:
            error = f"JoyAI-Echo {plan.variant} completed without producing a fresh artifact."
        return {
            "ok": ok,
            "status": "success" if ok else "failed",
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_files": generated_files,
            "error": error,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "runtime_plan": plan.to_dict(),
            "run_dir": plan.output_root,
        }


class JoyAIEchoLongVideoRuntime(_JoyAIEchoSubprocessRuntime):
    """Adapter for the official Echo-LongVideo reference-to-video CLI."""

    project_name = "echo_longvideo"

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        gemma_path: str | Path | None = None,
        source_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        config_path: str | Path | None = None,
        precision: str = "bf16",
        consumer: bool = False,
        low_vram: bool = False,
        device: str = "cuda",
    ) -> None:
        normalized_precision = str(precision).lower()
        if normalized_precision not in LONGVIDEO_CHECKPOINT_NAMES:
            raise ValueError("Echo-LongVideo precision must be one of: bf16, fp8, fp4.")
        if consumer and low_vram:
            raise ValueError("Echo-LongVideo consumer and low_vram profiles are mutually exclusive.")
        source_candidate = _expand_path(source_root or os.getenv("JOYAI_ECHO_ROOT")) or DEFAULT_SOURCE_ROOT
        if source_candidate.name == "echo_longvideo":
            self.runtime_root = source_candidate
            self.source_root = source_candidate.parent
        else:
            self.source_root = source_candidate
            self.runtime_root = source_candidate / "echo_longvideo"
        self.precision = normalized_precision
        self.consumer = bool(consumer)
        self.low_vram = bool(low_vram)
        self.device = str(device)
        self.python_executable = _python_path(python_executable)
        checkpoint_root = _expand_path(checkpoint_dir) or DEFAULT_HFD_ROOT / "jdopensource--JoyAI-Echo"
        checkpoint_name = LONGVIDEO_CHECKPOINT_NAMES[self.precision]
        checkpoint_is_variant = (
            checkpoint_root.name == checkpoint_name or (checkpoint_root / "checkpoint.json").is_file()
        )
        self.checkpoint_dir = checkpoint_root if checkpoint_is_variant else checkpoint_root / checkpoint_name
        self.gemma_path = _expand_path(gemma_path) or DEFAULT_HFD_ROOT / "google--gemma-3-12b-it"
        self.config_path = _expand_path(config_path) or self.runtime_root / "configs" / self._config_name()

    def _config_name(self) -> str:
        if self.consumer:
            return f"inference.consumer.{self.precision}.yaml"
        if self.low_vram:
            return f"inference.{self.precision}.low_vram.yaml"
        return f"inference.{self.precision}.yaml"

    def missing_checkpoint_files(self) -> tuple[str, ...]:
        required = [self.checkpoint_dir / "checkpoint.json"]
        if self.precision == "fp4":
            required.extend(
                (self.checkpoint_dir / "components.safetensors", self.checkpoint_dir / "transformer_modelopt.pt")
            )
        else:
            required.append(self.checkpoint_dir / "model.safetensors")
        required.append(self.gemma_path / "config.json")
        return _missing_paths(required)

    def missing_runtime_files(self) -> tuple[str, ...]:
        return _missing_paths(
            (
                self.runtime_root / "inference.py",
                self.runtime_root / "r2v_schema.py",
                self.config_path,
                self.python_executable,
            )
        )

    def preflight(self) -> dict[str, Any]:
        missing_checkpoint = self.missing_checkpoint_files()
        missing_runtime = self.missing_runtime_files()
        ffmpeg_path = shutil.which("ffmpeg")
        device_ready = self.device.lower().startswith("cuda")
        ready = not missing_checkpoint and not missing_runtime and ffmpeg_path is not None and device_ready
        return {
            "status": "ready" if ready else "blocked",
            "project": self.project_name,
            "variant": self.precision,
            "source_root": str(self.source_root),
            "runtime_root": str(self.runtime_root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "gemma_path": str(self.gemma_path),
            "config_path": str(self.config_path),
            "python_executable": str(self.python_executable),
            "ffmpeg": ffmpeg_path,
            "missing_checkpoint_files": list(missing_checkpoint),
            "missing_runtime_files": list(missing_runtime),
            "missing_system_tools": [] if ffmpeg_path else ["ffmpeg"],
            "device": self.device,
            "device_ready": device_ready,
            "blocked_reasons": [] if device_ready else ["Echo-LongVideo inference requires CUDA."],
            "license": "LTX-2 Community License; upstream marks JoyAI-Echo academic/non-commercial",
            "upstream_revision": UPSTREAM_REVISION,
        }

    def _materialize_request(self, request: Mapping[str, Any], output_path: Path) -> Path:
        supplied = _expand_path(request.get("request_path") or request.get("request"))
        if supplied is not None:
            if not supplied.is_file():
                raise FileNotFoundError(f"Echo-LongVideo R2V request not found: {supplied}")
            return supplied
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Echo-LongVideo requires prompt text or request_path.")
        memory_slots = request.get("memory_slots") or []
        if not isinstance(memory_slots, list):
            raise TypeError("Echo-LongVideo memory_slots must be a list.")
        if len(memory_slots) > 7:
            raise ValueError("Echo-LongVideo supports at most seven memory slots.")
        normalized_slots: list[dict[str, Any]] = []
        for index, slot in enumerate(memory_slots):
            if not isinstance(slot, Mapping):
                raise TypeError(f"Echo-LongVideo memory_slots[{index}] must be a mapping.")
            normalized = dict(slot)
            for key in ("image_url", "audio_url"):
                if normalized.get(key) is not None:
                    normalized[key] = _resolved_resource(
                        normalized[key],
                        field_name=f"memory_slots[{index}].{key}",
                    )
            normalized_slots.append(normalized)
        condition_img = _resolved_resource(
            request.get("condition_img") or request.get("images") or request.get("image"),
            field_name="condition_img",
        )
        payload: dict[str, Any] = {
            "work_id": _safe_slug(request.get("work_id"), "worldfoundry"),
            "shot_id": _safe_slug(request.get("shot_id") or request.get("sample_id"), output_path.stem),
            "prompt": prompt,
            "condition_img": condition_img,
            "memory_slots": normalized_slots,
            "num_frames": int(request.get("num_frames") or 241),
            "width": int(request.get("width") or 1280),
            "height": int(request.get("height") or 736),
            "seed": int(request.get("seed") if request.get("seed") is not None else 42),
        }
        if request.get("duration_sec") is not None:
            payload["duration_sec"] = float(request["duration_sec"])
        request_path = output_path.with_suffix(".request.json")
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return request_path

    def build_plan(
        self,
        *,
        request: Mapping[str, Any],
        output_path: str | Path,
        output_root: str | Path | None = None,
    ) -> JoyAIEchoRuntimePlan:
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() not in {".mp4", ".mov", ".webm"}:
            output = output.with_suffix(".mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        request_path = self._materialize_request(request, output)
        run_root = _expand_path(output_root or request.get("runtime_output_root"))
        if run_root is None:
            run_root = output.parent / f"{output.stem}_echo_longvideo"
        run_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python_executable),
            str(self.runtime_root / "inference.py"),
            "--config",
            str(self.config_path),
            "--request",
            str(request_path),
            "--checkpoint",
            str(self.checkpoint_dir),
            "--gemma-path",
            str(self.gemma_path),
            "--output-root",
            str(run_root),
        ]
        value_flags = {
            "seed": "--seed",
            "num_frames": "--num-frames",
            "height": "--video-height",
            "width": "--video-width",
            "fps": "--video-fps",
            "conditioning_cache_dir": "--conditioning-cache-dir",
            "video_vae_decode_mode": "--video-vae-decode-mode",
            "video_vae_tile_size_frames": "--video-vae-tile-size-frames",
            "video_vae_tile_overlap_frames": "--video-vae-tile-overlap-frames",
            "video_vae_tile_size_pixels": "--video-vae-tile-size-pixels",
            "video_vae_tile_overlap_pixels": "--video-vae-tile-overlap-pixels",
            "text_batch_size": "--text-batch-size",
            "image_batch_size": "--image-batch-size",
            "audio_batch_size": "--audio-batch-size",
            "memory_max_size": "--memory-max-size",
        }
        for key, flag in value_flags.items():
            _append_value(command, flag, request.get(key))
        for key, flag in {
            "dit_layerwise_offload": "--dit-layerwise-offload",
            "dit_pin_memory": "--dit-pin-memory",
        }.items():
            if request.get(key) is not None:
                _append_value(command, flag, str(bool(request[key])).lower())
        _append_bool(command, "--overwrite-condition-cache", request.get("overwrite_condition_cache"))
        return JoyAIEchoRuntimePlan(
            project=self.project_name,
            variant=self.precision,
            command=tuple(command),
            env=_runtime_env(self.device, self.runtime_root),
            workdir=str(self.runtime_root),
            output_path=str(output),
            output_root=str(run_root),
            request_path=str(request_path),
        )


class JoyAIEchoWMRuntime(_JoyAIEchoSubprocessRuntime):
    """Adapter for Echo-WM Base and Echo-WM Flash Preview inference."""

    project_name = "echo_wm"

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        gemma_path: str | Path | None = None,
        source_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        config_path: str | Path | None = None,
        variant: str = "base",
        device: str = "cuda",
    ) -> None:
        normalized_variant = str(variant).lower()
        if normalized_variant in {"causal", "flash-preview", "flash_preview"}:
            normalized_variant = "flash"
        if normalized_variant not in WM_CHECKPOINT_NAMES:
            raise ValueError("Echo-WM variant must be base or flash.")
        self.variant = normalized_variant
        source_candidate = _expand_path(source_root or os.getenv("JOYAI_ECHO_ROOT")) or DEFAULT_SOURCE_ROOT
        if source_candidate.name == "echo_wm":
            self.runtime_root = source_candidate
            self.source_root = source_candidate.parent
        else:
            self.source_root = source_candidate
            self.runtime_root = source_candidate / "echo_wm"
        self.device = str(device)
        self.python_executable = _python_path(python_executable)
        checkpoint_root = _expand_path(checkpoint_dir) or DEFAULT_HFD_ROOT / "Echo-Team--Echo-WM"
        checkpoint_name = WM_CHECKPOINT_NAMES[self.variant]
        checkpoint_is_file = checkpoint_root.suffix == ".safetensors" or checkpoint_root.name == checkpoint_name
        self.checkpoint_path = checkpoint_root if checkpoint_is_file else checkpoint_root / checkpoint_name
        self.gemma_path = _expand_path(gemma_path) or DEFAULT_HFD_ROOT / "google--gemma-3-12b-it-qat-q4_0-unquantized"
        default_config = "inference_wm.yaml" if self.variant == "base" else "inference_wm_causal.yaml"
        self.config_path = _expand_path(config_path) or Path(
            resolve_data_path(f"models/runtime/configs/joyai-echo-wm/{default_config}")
        )

    @property
    def script_path(self) -> Path:
        name = "inference_wm.py" if self.variant == "base" else "inference_wm_causal.py"
        return self.runtime_root / name

    def missing_checkpoint_files(self) -> tuple[str, ...]:
        return _missing_paths((self.checkpoint_path, self.gemma_path / "config.json"))

    def missing_runtime_files(self) -> tuple[str, ...]:
        return _missing_paths((self.script_path, self.config_path, self.python_executable))

    def preflight(self) -> dict[str, Any]:
        missing_checkpoint = self.missing_checkpoint_files()
        missing_runtime = self.missing_runtime_files()
        ffmpeg_path = shutil.which(os.environ.get("FFMPEG_BIN") or "ffmpeg")
        ffprobe_path = shutil.which(os.environ.get("FFPROBE_BIN") or "ffprobe")
        missing_tools = [name for name, path in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)) if not path]
        device_ready = self.device.lower().startswith("cuda")
        ready = not missing_checkpoint and not missing_runtime and not missing_tools and device_ready
        return {
            "status": "ready" if ready else "blocked",
            "project": self.project_name,
            "variant": self.variant,
            "source_root": str(self.source_root),
            "runtime_root": str(self.runtime_root),
            "checkpoint_path": str(self.checkpoint_path),
            "gemma_path": str(self.gemma_path),
            "config_path": str(self.config_path),
            "python_executable": str(self.python_executable),
            "ffmpeg": ffmpeg_path,
            "ffprobe": ffprobe_path,
            "missing_checkpoint_files": list(missing_checkpoint),
            "missing_runtime_files": list(missing_runtime),
            "missing_system_tools": missing_tools,
            "device": self.device,
            "device_ready": device_ready,
            "blocked_reasons": [] if device_ready else ["Echo-WM inference requires CUDA."],
            "license": "LTX-2 Community License; upstream marks JoyAI-Echo academic/non-commercial",
            "upstream_revision": WM_UPSTREAM_REVISION,
        }

    @staticmethod
    def _action_string(request: Mapping[str, Any]) -> str:
        value = request.get("action_str") or request.get("actions") or request.get("interactions")
        if isinstance(value, Mapping):
            value = value.get("action_str") or value.get("actions") or value.get("action")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value = ",".join(str(item) for item in value)
        text = str(value or "").strip()
        if not text:
            raise ValueError("Echo-WM requires an Action DSL string, for example 'w-60,a-60'.")
        if not all(re.fullmatch(r"(?:none|[wasdijkl]+)-[1-9][0-9]*", segment.strip()) for segment in text.split(",")):
            raise ValueError("Invalid Echo-WM Action DSL; use segments such as w-60,none-60,wj-60.")
        return text

    @staticmethod
    def _image_path(request: Mapping[str, Any], output: Path) -> Path:
        value = request.get("image")
        if value is None:
            value = request.get("images")
        path_text = _first_path(value)
        if path_text is not None:
            path = Path(path_text).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Echo-WM first-frame image not found: {path}")
            return path
        image = value[0] if isinstance(value, Sequence) and value else value
        if hasattr(image, "save"):
            target = output.with_suffix(".input.png")
            image.save(target)
            return target
        raise ValueError("Echo-WM requires a first-frame image path or PIL-compatible image.")

    def build_plan(
        self,
        *,
        request: Mapping[str, Any],
        output_path: str | Path,
    ) -> JoyAIEchoRuntimePlan:
        request = dict(request)
        reference = request.pop("ref_image_path", None)
        if request.get("images") is None and request.get("image") is None and reference is not None:
            request["images"] = reference

        common = {
            "prompt",
            "images",
            "image",
            "interactions",
            "actions",
            "action_str",
            "width",
            "height",
            "num_frames",
            "fps",
            "seed",
            "fov_deg",
            "translation_speed",
            "rotation_speed_deg",
            "pitch_limit_deg",
            "auto_fov",
            "no_audio",
            "audio",
            "action_overlay",
            "sample_id",
        }
        variant_keys = (
            {"steps", "guidance_scale", "video_cfg", "audio_cfg", "negative_prompt", "stg_scale", "stg_blocks"}
            if self.variant == "base"
            else {"video_local_attn_size", "video_sink_size", "video_chunk_size", "timesteps"}
        )
        unknown = request.keys() - common - variant_keys
        if unknown:
            raise ValueError(f"Unsupported Echo-WM {self.variant} options: {sorted(unknown)}")
        for name in ("steps", "video_local_attn_size", "video_sink_size"):
            value = request.get(name)
            minimum = 0 if name == "video_sink_size" else 1
            if value is not None and (isinstance(value, bool) or int(value) != float(value) or int(value) < minimum):
                raise ValueError(f"Echo-WM {name} must be an integer >= {minimum}")
        timesteps = request.get("timesteps")
        if timesteps is not None:
            if not isinstance(timesteps, (list, tuple)) or not timesteps:
                raise ValueError("Echo-WM timesteps must be a non-empty descending list")
            values = [float(value) for value in timesteps]
            if not all(math.isfinite(v) and v == int(v) and 0 < v <= 1000 for v in values) or any(
                a <= b for a, b in zip(values, values[1:])
            ):
                raise ValueError("Echo-WM timesteps must be integers in [1, 1000] and descending")
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Echo-WM requires a prompt.")
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() not in {".mp4", ".mov", ".webm"}:
            output = output.with_suffix(".mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        image_path = self._image_path(request, output)
        action_str = self._action_string(request)
        import yaml

        defaults = yaml.safe_load(self.config_path.read_text())
        video = defaults.get("video", {})
        causal = defaults.get("causal", {})
        num_frames = request.get("num_frames", video.get("num_frames", 241))
        chunk_size = request.get("video_chunk_size", causal.get("video_chunk_size", 3))
        for name, value in (("num_frames", num_frames), ("video_chunk_size", chunk_size)):
            if isinstance(value, bool) or int(value) != float(value) or int(value) <= 0:
                raise ValueError(f"Echo-WM {name} must be a positive integer")
        stride = 8 * int(chunk_size) if self.variant == "flash" else 8
        if int(num_frames) <= 1 or (int(num_frames) - 1) % stride:
            raise ValueError(f"Echo-WM {self.variant} num_frames must follow 1 + {stride}*N with N >= 1.")
        for name in ("width", "height"):
            value = request.get(name, video.get(name))
            if value is not None and (int(value) <= 0 or int(value) != float(value) or int(value) % 32):
                raise ValueError(f"Echo-WM {name} must be a positive multiple of 32")
        for name in ("fps", "fov_deg"):
            value = request.get(name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"Echo-WM {name} must be finite and positive")
        if request.get("fov_deg") is not None and float(request["fov_deg"]) >= 180:
            raise ValueError("Echo-WM fov_deg must be below 180")
        for name in (
            "width",
            "height",
            "num_frames",
            "video_chunk_size",
            "steps",
            "seed",
            "video_local_attn_size",
            "video_sink_size",
        ):
            if request.get(name) is not None:
                value = request[name]
                if isinstance(value, bool) or int(value) != float(value):
                    raise ValueError(f"Echo-WM {name} must be an integer")
                request[name] = int(value)
        if timesteps is not None:
            request["timesteps"] = [int(value) for value in timesteps]
        command = [
            str(self.python_executable),
            str(self.script_path),
            "--config",
            str(self.config_path),
            "--image",
            str(image_path),
            "--prompt",
            prompt,
            "--action-str",
            action_str,
            "--checkpoint",
            str(self.checkpoint_path),
            "--gemma-path",
            str(self.gemma_path),
            "--output",
            str(output),
        ]
        common_flags = {
            "width": "--width",
            "height": "--height",
            "num_frames": "--num-frames",
            "fps": "--fps",
            "seed": "--seed",
            "fov_deg": "--fov-deg",
            "translation_speed": "--translation-speed",
            "rotation_speed_deg": "--rotation-speed-deg",
            "pitch_limit_deg": "--pitch-limit-deg",
        }
        for key, flag in common_flags.items():
            _append_value(command, flag, request.get(key))
        if self.variant == "base":
            base_flags = {
                "steps": "--steps",
                "guidance_scale": "--guidance-scale",
                "video_cfg": "--video-cfg",
                "audio_cfg": "--audio-cfg",
                "negative_prompt": "--negative-prompt",
                "stg_scale": "--stg-scale",
            }
            for key, flag in base_flags.items():
                _append_value(command, flag, request.get(key))
            stg_blocks = request.get("stg_blocks")
            if isinstance(stg_blocks, Sequence) and not isinstance(stg_blocks, (str, bytes, bytearray)):
                command.append("--stg-blocks")
                command.extend(str(item) for item in stg_blocks)
        else:
            causal_flags = {
                "video_local_attn_size": "--video_local_attn_size",
                "video_sink_size": "--video_sink_size",
                "video_chunk_size": "--video_chunk_size",
            }
            for key, flag in causal_flags.items():
                _append_value(command, flag, request.get(key))
            timesteps = request.get("timesteps")
            if isinstance(timesteps, Sequence) and not isinstance(timesteps, (str, bytes, bytearray)):
                command.append("--timesteps")
                command.extend(str(item) for item in timesteps)
        _append_bool(command, "--auto-fov", request.get("auto_fov"))
        _append_bool(command, "--no-audio", request.get("no_audio") or request.get("audio") is False)
        if request.get("action_overlay") is False:
            command.append("--no-action-overlay")
        return JoyAIEchoRuntimePlan(
            project=self.project_name,
            variant=self.variant,
            upstream_revision=WM_UPSTREAM_REVISION,
            command=tuple(command),
            env=_runtime_env(self.device, self.runtime_root),
            workdir=str(self.runtime_root),
            output_path=str(output),
            output_root=str(output.parent),
        )


__all__ = [
    "JoyAIEchoLongVideoRuntime",
    "JoyAIEchoRuntimePlan",
    "JoyAIEchoWMRuntime",
]

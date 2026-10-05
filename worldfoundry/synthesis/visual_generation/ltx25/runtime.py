"""Official-subprocess runtime for LTX-2.5 distilled inference.

LTX-2.5 uses split component checkpoints and Gemma 4, so it cannot be loaded
through WorldFoundry's older LTX-2/2.3 native checkpoint recipe.  This adapter
keeps the version-pinned official source and its environment external while
giving WorldFoundry a deterministic preflight, command plan, logs, and artifact
contract.  Prediction never clones source or downloads weights.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.runtime.assets import expand_worldfoundry_path, local_file_is_ready
from worldfoundry.runtime.env import resolve_hfd_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_REVISION = "a95ab856bf29407b6b066ede0abe1846050db56c"
CHECKPOINT_REVISION = "bf86adedf518142442575d1ce2e767b7d01c8c76"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "cache" / "generative_taxonomy" / "Lightricks--LTX-2"
DEFAULT_CHECKPOINT_ROOT = resolve_hfd_root() / "Lightricks--LTX-2.5"

DEFAULT_COMPONENTS = {
    "transformer_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae_path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler_path": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    ),
    "duration_head_path": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
}


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    expanded = os.path.expandvars(str(value))
    return Path(expand_worldfoundry_path(expanded)).expanduser().resolve()


def _python_path(value: str | Path | None) -> Path:
    if value is None or not str(value).strip():
        return Path(os.path.abspath(sys.executable))
    text = os.path.expandvars(str(value))
    if Path(text).is_absolute() or os.sep in text:
        # venv Python is often a symlink to its base interpreter. Resolving it
        # loses the venv's site-packages when launched from another workdir.
        return Path(os.path.abspath(os.path.expanduser(text)))
    discovered = shutil.which(text)
    return Path(os.path.abspath(discovered)) if discovered else Path(text).expanduser()


def _append_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        command.extend((flag, str(value)))


def _cuda_visible_devices(device: str) -> str | None:
    text = str(device).strip().lower()
    if text == "cpu":
        return ""
    if not text.startswith("cuda:"):
        return None
    indices = [item.strip() for item in text.split(":", 1)[1].split(",")]
    return ",".join(indices) if indices and all(item.isdigit() for item in indices) else None


def _conditioning_images(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike, Mapping)):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError("LTX-2.5 images must be a local path, mapping, or sequence of those values.")

    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            path_value = item.get("path") or item.get("image") or item.get("image_path")
            frame_index = int(item.get("frame_index", item.get("frame_idx", 0)))
            strength = float(item.get("strength", item.get("image_strength", 1.0)))
            crf = item.get("crf")
        else:
            path_value = item
            frame_index = 0
            strength = 1.0
            crf = None
        path = _expand_path(path_value)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"LTX-2.5 conditioning image {index} not found: {path}")
        if frame_index < 0:
            raise ValueError("LTX-2.5 conditioning frame_index must be non-negative.")
        if strength < 0:
            raise ValueError("LTX-2.5 conditioning strength must be non-negative.")
        resolved.append({"path": str(path), "frame_index": frame_index, "strength": strength, "crf": crf})
    return resolved


@dataclass(frozen=True)
class LTX25RuntimePlan:
    """Serializable launch plan for one official LTX-2.5 inference process."""

    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    output_path: str
    task: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "env": dict(self.env),
            "workdir": self.workdir,
            "output_path": self.output_path,
            "task": self.task,
            "pipeline": "ltx_pipelines.distilled",
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }


class LTX25DistilledRuntime:
    """Launch the official LTX-2.5 ``DistilledPipeline`` for T2V or I2V."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        source_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        transformer_path: str | Path | None = None,
        text_encoder_path: str | Path | None = None,
        video_vae_path: str | Path | None = None,
        audio_vae_path: str | Path | None = None,
        spatial_upsampler_path: str | Path | None = None,
        duration_head_path: str | Path | None = None,
        prompt_enhancer_gemma_root: str | Path | None = None,
        offload_mode: str = "cpu",
        quantization: str | None = None,
        diffvae_optimization: str = "chunked_eager",
        device: str = "cuda",
    ) -> None:
        self.source_root = _expand_path(source_root or os.getenv("LTX2_ROOT")) or DEFAULT_SOURCE_ROOT
        self.checkpoint_dir = _expand_path(checkpoint_dir) or DEFAULT_CHECKPOINT_ROOT
        self.python_executable = _python_path(python_executable)
        self.device = str(device)
        self.offload_mode = str(offload_mode).strip().lower()
        self.quantization = str(quantization).strip().lower() if quantization else None
        self.diffvae_optimization = str(diffvae_optimization).strip().lower()
        if self.offload_mode not in {"none", "cpu", "disk"}:
            raise ValueError("LTX-2.5 offload_mode must be one of: none, cpu, disk.")
        if self.quantization not in {None, "fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"}:
            raise ValueError("Unsupported LTX-2.5 quantization policy.")

        overrides = {
            "transformer_path": transformer_path,
            "text_encoder_path": text_encoder_path,
            "video_vae_path": video_vae_path,
            "audio_vae_path": audio_vae_path,
            "spatial_upsampler_path": spatial_upsampler_path,
            "duration_head_path": duration_head_path,
        }
        for name, relative in DEFAULT_COMPONENTS.items():
            setattr(self, name, _expand_path(overrides[name]) or self.checkpoint_dir / relative)
        self.prompt_enhancer_gemma_root = _expand_path(prompt_enhancer_gemma_root)

    @property
    def _runtime_files(self) -> tuple[Path, ...]:
        return (
            self.source_root / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "distilled.py",
            self.source_root / "packages" / "ltx-core" / "src" / "ltx_core" / "__init__.py",
            self.python_executable,
        )

    @property
    def _required_checkpoint_files(self) -> tuple[Path, ...]:
        return (
            self.transformer_path,
            self.text_encoder_path,
            self.video_vae_path,
            self.audio_vae_path,
            self.spatial_upsampler_path,
        )

    def _runtime_env(self) -> dict[str, str]:
        roots = (
            self.source_root / "packages" / "ltx-pipelines" / "src",
            self.source_root / "packages" / "ltx-core" / "src",
        )
        python_path = os.pathsep.join(
            [*(str(path) for path in roots), *(value for value in (os.environ.get("PYTHONPATH"),) if value)]
        )
        env = {
            "PYTHONPATH": python_path,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        visible_devices = _cuda_visible_devices(self.device)
        if visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        return env

    def preflight(self) -> dict[str, Any]:
        missing_checkpoint = [str(path) for path in self._required_checkpoint_files if not path.is_file()]
        incomplete_checkpoint = [
            str(path) for path in self._required_checkpoint_files
            if path.is_file() and not local_file_is_ready(path)
        ]
        missing_runtime = [str(path) for path in self._runtime_files if not path.is_file()]
        ffmpeg = shutil.which("ffmpeg")
        device_ready = self.device.lower().startswith(("cuda", "mps"))
        ready = not missing_checkpoint and not incomplete_checkpoint and not missing_runtime and ffmpeg is not None and device_ready
        blocked_reasons = [] if device_ready else ["LTX-2.5 inference requires a CUDA or MPS accelerator."]
        blocked_reasons.extend(f"Checkpoint download is incomplete or empty: {path}" for path in incomplete_checkpoint)
        return {
            "status": "ready" if ready else "blocked",
            "model_id": "ltx-2.5",
            "pipeline": "ltx_pipelines.distilled",
            "source_root": str(self.source_root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "python_executable": str(self.python_executable),
            "component_paths": {
                name: str(getattr(self, name)) for name in DEFAULT_COMPONENTS
            },
            "prompt_enhancer_gemma_root": (
                str(self.prompt_enhancer_gemma_root) if self.prompt_enhancer_gemma_root else None
            ),
            "missing_checkpoint_files": missing_checkpoint,
            "incomplete_checkpoint_files": incomplete_checkpoint,
            "missing_runtime_files": missing_runtime,
            "missing_system_tools": [] if ffmpeg else ["ffmpeg"],
            "blocked_reasons": blocked_reasons,
            "device": self.device,
            "device_ready": device_ready,
            "ffmpeg": ffmpeg,
            "license": "LTX-2.x Community License Agreement (August 11, 2026)",
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }

    def build_plan(self, *, request: Mapping[str, Any], output_path: str | Path) -> LTX25RuntimePlan:
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("LTX-2.5 inference requires a non-empty prompt.")
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            output = output.with_suffix(".mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        images = _conditioning_images(request.get("images") or request.get("image"))
        height = int(request.get("height", 512))
        width = int(request.get("width", 768))
        frame_rate = float(request.get("fps", request.get("frame_rate", 24.0)))
        offload_mode = str(request.get("offload_mode", self.offload_mode)).strip().lower()
        quantization = request.get("quantization", self.quantization)
        quantization = str(quantization).strip().lower() if quantization else None
        if height <= 0 or width <= 0 or height % 64 or width % 64:
            raise ValueError("Two-stage LTX-2.5 height and width must be positive multiples of 64.")
        if frame_rate <= 0:
            raise ValueError("LTX-2.5 frame_rate must be positive.")
        if offload_mode not in {"none", "cpu", "disk"}:
            raise ValueError("LTX-2.5 offload_mode must be one of: none, cpu, disk.")
        if quantization not in {None, "fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"}:
            raise ValueError("Unsupported LTX-2.5 quantization policy.")

        command = [
            str(self.python_executable),
            "-m",
            "ltx_pipelines.distilled",
            "--transformer-path",
            str(self.transformer_path),
            "--text-encoder-path",
            str(self.text_encoder_path),
            "--video-vae-path",
            str(self.video_vae_path),
            "--audio-vae-path",
            str(self.audio_vae_path),
            "--spatial-upsampler-path",
            str(self.spatial_upsampler_path),
            "--prompt",
            prompt,
            "--output-path",
            str(output),
            "--seed",
            str(int(request.get("seed", 42))),
            "--height",
            str(height),
            "--width",
            str(width),
            "--frame-rate",
            str(frame_rate),
            "--offload",
            offload_mode,
            "--diffvae-optimization",
            str(request.get("diffvae_optimization", self.diffvae_optimization)),
        ]
        auto_duration = request.get("auto_duration")
        if request.get("num_frames") is not None or auto_duration is None:
            num_frames = int(request.get("num_frames", 121))
            if num_frames < 1 or (num_frames - 1) % 8:
                raise ValueError("LTX-2.5 num_frames must satisfy num_frames = 8 * k + 1.")
            _append_value(command, "--num-frames", num_frames)
        else:
            if not self.duration_head_path.is_file():
                raise FileNotFoundError(
                    "LTX-2.5 auto_duration requires duration_head_path: " f"{self.duration_head_path}"
                )
            if not isinstance(auto_duration, Sequence) or len(auto_duration) != 2:
                raise ValueError("LTX-2.5 auto_duration must contain [min_seconds, max_seconds].")
            command.extend(("--auto-duration", str(float(auto_duration[0])), str(float(auto_duration[1]))))
        if self.duration_head_path.is_file():
            _append_value(command, "--duration-head-path", self.duration_head_path)

        for image in images:
            values = [
                "--image",
                image["path"],
                str(image["frame_index"]),
                str(image["strength"]),
            ]
            if image["crf"] is not None:
                values.append(str(int(image["crf"])))
            command.extend(values)

        _append_value(command, "--quantization", quantization)
        if bool(request.get("enhance_prompt", False)):
            if self.prompt_enhancer_gemma_root is None or not self.prompt_enhancer_gemma_root.is_dir():
                raise FileNotFoundError(
                    "LTX-2.5 enhance_prompt requires a local prompt_enhancer_gemma_root."
                )
            command.extend(("--enhance-prompt", "--prompt-enhancer-gemma-root", str(self.prompt_enhancer_gemma_root)))
        generated_keyframes = int(request.get("num_generated_keyframes", 0))
        if generated_keyframes:
            _append_value(command, "--num-generated-keyframes", generated_keyframes)

        return LTX25RuntimePlan(
            command=tuple(command),
            env=self._runtime_env(),
            workdir=str(self.source_root),
            output_path=str(output),
            task="image-to-video" if images else "text-to-video",
        )

    def run_plan(
        self,
        plan: LTX25RuntimePlan,
        *,
        timeout_seconds: int = 7200,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        output = Path(plan.output_path)
        target_log_dir = Path(log_dir or output.parent).expanduser().resolve()
        target_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = target_log_dir / "ltx_2_5_stdout.log"
        stderr_path = target_log_dir / "ltx_2_5_stderr.log"
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

        fresh_artifact = (
            output.is_file()
            and output.stat().st_size > 0
            and output.stat().st_mtime >= started_at - 2.0
        )
        ok = returncode == 0 and fresh_artifact
        if returncode != 0 and error is None:
            error = f"LTX-2.5 official runner exited with code {returncode}."
        elif returncode == 0 and not fresh_artifact and error is None:
            error = "LTX-2.5 completed without producing a fresh non-empty MP4 artifact."
        return {
            "ok": ok,
            "status": "success" if ok else "failed",
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_files": [str(output)] if ok else [],
            "error": error,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "runtime_plan": plan.to_dict(),
        }


__all__ = [
    "CHECKPOINT_REVISION",
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_COMPONENTS",
    "DEFAULT_SOURCE_ROOT",
    "LTX25DistilledRuntime",
    "LTX25RuntimePlan",
    "UPSTREAM_REVISION",
]

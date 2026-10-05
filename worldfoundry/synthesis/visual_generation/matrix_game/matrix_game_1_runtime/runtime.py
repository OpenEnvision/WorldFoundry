from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core import jsonable
from worldfoundry.core.io.paths import conda_envs_root_path, conda_root_path, project_root, resolve_worldfoundry_path
from worldfoundry.evaluation.utils import worldfoundry_data_path
from worldfoundry.runtime.env import resolve_ckpt_dir, resolve_hfd_root


RUNTIME_ROOT = Path(__file__).resolve().parent
REPO_ROOT = project_root(RUNTIME_ROOT)
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "WORLDFOUNDRY_MATRIX_GAME_1_CHECKPOINT_DIR",
        str(Path(resolve_ckpt_dir()) / "Matrix-Game"),
    )
)
DEFAULT_CONDA_DIR = os.environ.get(
    "WORLDFOUNDRY_MATRIX_GAME_1_CONDA_DIR",
    str(conda_envs_root_path() / "matrix-game-1.0"),
)
DEFAULT_RUNTIME_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "matrix_game_1")
EXPECTED_DIT_FILES = tuple(
    f"dit/diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
    for index in range(1, 8)
)
EXPECTED_TEXT_ENCODER_FILES = tuple(
    f"text_encoder_i2v/model-{index:05d}-of-00004.safetensors"
    for index in range(1, 5)
)
EXPECTED_VAE_FILES = ("vae/pytorch_model.pt",)
EXPECTED_CHECKPOINT_FILES = EXPECTED_DIT_FILES + EXPECTED_TEXT_ENCODER_FILES + EXPECTED_VAE_FILES
REQUIRED_RUNTIME_CONFIG_FILES = ("environment.yml",)
REQUIRED_RUNTIME_FILES = (
    "inference_bench.py",
    "teacache_forward.py",
    "condtions.py",
    "tools/visualize.py",
    "matrixgame/sample/pipeline_matrixgame.py",
    "matrixgame/vae_variants/matrixgame_vae.py",
    "matrixgame/encoder_variants/matrixgame_i2v.py",
)
OPTIONAL_CHECKPOINT_FILES = (
    "assets/mouse.png",
    "dit/config.json",
    "text_encoder_i2v/config.json",
    "text_encoder_i2v/tokenizer.json",
    "vae/config.json",
)
VIDEO_OUTPUT_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _expand_runtime_path(value: str | Path | None) -> Path | None:
    """Expand WorldFoundry runtime tokens used in model metadata.

    Args:
        value: Optional path with environment-token placeholders.
    """

    if value is None:
        return None
    env = os.environ.copy()
    env.setdefault("WORLDFOUNDRY_BENCH_ROOT", str(REPO_ROOT))
    env.setdefault("WORLDFOUNDRY_REPO_ROOT", str(REPO_ROOT))
    env.setdefault("WORLDFOUNDRY_CONDA_ROOT", str(conda_root_path()))
    env.setdefault("WORLDFOUNDRY_CONDA_ENVS_ROOT", str(conda_envs_root_path()))
    env.setdefault("WORLDFOUNDRY_CKPT_DIR", str(resolve_ckpt_dir()))
    env.setdefault("WORLDFOUNDRY_HFD_ROOT", str(resolve_hfd_root()))
    return resolve_worldfoundry_path(value, env)


def _gpu_id_from_device(device: str | None) -> str | None:
    if not device:
        return None
    text = str(device).strip()
    if not text or text == "cuda":
        return None
    if text.startswith("cuda:"):
        return text.split(":", 1)[1].strip() or None
    if text.isdigit() or "," in text:
        return text
    return None


def _is_video_output_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_OUTPUT_SUFFIXES


def _first_filesystem_path(*values: Any) -> str | None:
    """Return the first string/path value usable by the official image runner."""

    for value in values:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value
    return None


@dataclass(frozen=True)
class MatrixGame1RuntimePlan:
    """Matrix-Game-1 execution command and resolved resource paths.

    Args:
        command: Python command for the vendored official runner.
        env: Environment variables required by the runner.
        workdir: Directory from which the command should be launched.
        checkpoint_dir: Matrix-Game-1 checkpoint directory.
        output_dir: Directory where official runner writes generated artifacts.
    """

    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    checkpoint_dir: str
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the runtime plan.

        Args:
            None.
        """
        return {
            "command": list(self.command),
            "env": dict(self.env),
            "workdir": self.workdir,
            "checkpoint_dir": self.checkpoint_dir,
            "output_dir": self.output_dir,
        }


class MatrixGame1Runtime:
    """Lightweight facade around the vendored Matrix-Game-1 official runtime."""

    MODEL_ID = "matrix-game-1"
    DISPLAY_NAME = "Matrix-Game-1"

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_id: str = MODEL_ID,
        checkpoint_dir: str | Path | None = None,
        conda_dir: str | Path | None = None,
        runtime_root: str | Path | None = None,
    ) -> None:
        """Initialize runtime paths without importing heavy ML dependencies.

        Args:
            checkpoint_dir: Directory containing Skywork/Matrix-Game assets.
            conda_dir: Optional Matrix-Game-1 conda prefix.
            runtime_root: Optional vendored runtime root override.
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "image_to_video"
        self.device = device
        selected_checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.checkpoint_dir = _expand_runtime_path(selected_checkpoint_dir) or Path(selected_checkpoint_dir).expanduser()
        self.conda_dir = _expand_runtime_path(conda_dir) if conda_dir is not None else None
        if self.conda_dir is None and DEFAULT_CONDA_DIR is not None:
            self.conda_dir = _expand_runtime_path(DEFAULT_CONDA_DIR)
        self.runtime_root = _expand_runtime_path(runtime_root) if runtime_root is not None else RUNTIME_ROOT

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "MatrixGame1Runtime":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            checkpoint_dir=options.get("checkpoint_dir") or options.get("model_dir"),
            conda_dir=options.get("conda_dir") or options.get("python_env_dir"),
            runtime_root=options.get("runtime_root"),
        )

    @property
    def python_executable(self) -> Path | None:
        """Return the configured Matrix-Game-1 Python executable when present.

        Args:
            None.
        """
        if self.conda_dir is None:
            return None
        return self.conda_dir / "bin" / "python"

    def missing_checkpoint_files(self) -> tuple[str, ...]:
        """Return missing required Matrix-Game-1 checkpoint files.

        Args:
            None.
        """
        return tuple(
            str(self.checkpoint_dir / relative_path)
            for relative_path in EXPECTED_CHECKPOINT_FILES
            if not (self.checkpoint_dir / relative_path).is_file()
        )

    def missing_runtime_files(self) -> tuple[str, ...]:
        """Return missing vendored runtime files.

        Args:
            None.
        """
        missing = [
            str(self.runtime_root / relative_path)
            for relative_path in REQUIRED_RUNTIME_FILES
            if not (self.runtime_root / relative_path).is_file()
        ]
        python_executable = self.python_executable
        if python_executable is not None and not python_executable.is_file():
            missing.append(str(python_executable))
        return tuple(missing)

    def missing_runtime_config_files(self) -> tuple[str, ...]:
        """Return missing external Matrix-Game-1 runtime config files.

        Args:
            None.
        """
        return tuple(
            str(DEFAULT_RUNTIME_CONFIG_ROOT / relative_path)
            for relative_path in REQUIRED_RUNTIME_CONFIG_FILES
            if not (DEFAULT_RUNTIME_CONFIG_ROOT / relative_path).is_file()
        )

    def optional_checkpoint_files(self) -> dict[str, bool]:
        """Return optional Matrix-Game-1 asset presence.

        Args:
            None.
        """
        return {
            relative_path: (self.checkpoint_dir / relative_path).is_file()
            for relative_path in OPTIONAL_CHECKPOINT_FILES
        }

    def preflight(self) -> dict[str, Any]:
        """Return complete Matrix-Game-1 runtime and checkpoint readiness.

        Args:
            None.
        """
        missing_checkpoint = self.missing_checkpoint_files()
        missing_runtime = self.missing_runtime_files()
        missing_runtime_config = self.missing_runtime_config_files()
        code_ready = not any(
            item for item in missing_runtime if not item.endswith("/bin/python")
        ) and not missing_runtime_config
        checkpoint_ready = len(missing_checkpoint) == 0
        env_ready = self.python_executable is None or self.python_executable.is_file()
        return {
            "status": "ready" if code_ready and checkpoint_ready and env_ready else "blocked",
            "code_ready": code_ready,
            "checkpoint_ready": checkpoint_ready,
            "env_ready": env_ready,
            "runtime_root": str(self.runtime_root),
            "runtime_config_root": str(DEFAULT_RUNTIME_CONFIG_ROOT),
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_dir_exists": self.checkpoint_dir.is_dir(),
            "conda_dir": "" if self.conda_dir is None else str(self.conda_dir),
            "conda_dir_exists": False if self.conda_dir is None else self.conda_dir.is_dir(),
            "python_executable": "" if self.python_executable is None else str(self.python_executable),
            "expected_checkpoint_files": list(EXPECTED_CHECKPOINT_FILES),
            "required_runtime_files": list(REQUIRED_RUNTIME_FILES),
            "required_runtime_config_files": list(REQUIRED_RUNTIME_CONFIG_FILES),
            "missing_checkpoint_files": list(missing_checkpoint),
            "missing_runtime_files": list(missing_runtime),
            "missing_runtime_config_files": list(missing_runtime_config),
            "optional_checkpoint_files": self.optional_checkpoint_files(),
            "requires": [
                "complete Skywork/Matrix-Game checkpoint shards",
                "vae/pytorch_model.pt",
                "vendored Matrix-Game-1 runtime files",
                "external Matrix-Game-1 runtime configs under worldfoundry/data/models/runtime/configs",
                "Matrix-Game-1 Python environment for heavy official runner",
            ],
        }

    def _blocked_reasons(self) -> list[str]:
        preflight = self.preflight()
        reasons = [
            "Matrix-Game-1 is a distinct Skywork/Matrix-Game model and must not route through Matrix-Game-2.",
            "Matrix-Game-1 official Python runtime is vendored in-tree and import-light preflight is code-ready.",
        ]
        if preflight["checkpoint_ready"]:
            reasons.append("Matrix-Game-1 checkpoint files are complete.")
        elif preflight["checkpoint_dir_exists"]:
            reasons.append(
                "Local Skywork--Matrix-Game metadata/index files exist, but required weight shards are incomplete."
            )
        else:
            reasons.append("Local Skywork--Matrix-Game checkpoint directory is missing.")
        if preflight["env_ready"]:
            reasons.append("Matrix-Game-1 Python environment is executable.")
        elif preflight["conda_dir_exists"]:
            reasons.append("Local matrix-game-1.0 conda directory exists, but bin/python is missing.")
        else:
            reasons.append("Local matrix-game-1.0 conda directory is missing.")
        return reasons

    def build_plan(
        self,
        *,
        image_path: str,
        output_dir: str | Path,
        inference_steps: int = 50,
        fps: int = 16,
        max_images: int = 1,
        max_conditions: int = 1,
        bfloat16: bool = True,
        resolution: Sequence[int] | None = None,
        video_length: int | None = None,
        num_pre_frames: int | None = None,
        i2v_type: str | None = None,
        interactions: Sequence[str] = (),
        gpu_id: str | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> MatrixGame1RuntimePlan:
        """Build an official Matrix-Game-1 runner command.

        Args:
            image_path: Input image or image directory for the official runner.
            output_dir: Output directory for generated artifacts.
            inference_steps: Number of diffusion steps.
            fps: Output FPS.
            max_images: Maximum input images to process.
            max_conditions: Maximum action conditions to process.
            bfloat16: Whether to run the official runner in bfloat16 mode.
            resolution: Optional output resolution as ``(width, height)``.
            video_length: Optional number of generated video frames.
            num_pre_frames: Optional number of repeated pre-frames.
            i2v_type: Optional official I2V conditioning type.
            interactions: Ordered keyboard/camera command segments.
            extra_env: Additional environment variables.
        """
        python_executable = self.python_executable or Path("python")
        resolved_image_path = _expand_runtime_path(image_path) or Path(image_path).expanduser()
        if not resolved_image_path.is_absolute():
            resolved_image_path = resolved_image_path.resolve()
        runner_image_path = resolved_image_path
        resolved_output_dir = Path(output_dir).expanduser()
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = resolved_output_dir.resolve()
        command = [
            str(python_executable),
            str(self.runtime_root / "inference_bench.py"),
            "--dit_path",
            str(self.checkpoint_dir / "dit"),
            "--textenc_path",
            str(self.checkpoint_dir),
            "--vae_path",
            str(self.checkpoint_dir / "vae"),
            "--mouse_icon_path",
            str(self.checkpoint_dir / "assets" / "mouse.png"),
            "--image_path",
            str(runner_image_path),
            "--output_path",
            str(resolved_output_dir),
            "--inference_steps",
            str(inference_steps),
            "--fps",
            str(fps),
            "--max_images",
            str(max_images),
            "--max_conditions",
            str(max_conditions),
        ]
        if gpu_id:
            command.extend(["--gpu_id", str(gpu_id)])
        if bfloat16:
            command.append("--bfloat16")
        if resolution is not None:
            width, height = tuple(resolution)
            command.extend(["--resolution", str(width), str(height)])
        if video_length is not None:
            command.extend(["--video_length", str(video_length)])
        if num_pre_frames is not None:
            command.extend(["--num_pre_frames", str(num_pre_frames)])
        if i2v_type is not None:
            command.extend(["--i2v_type", str(i2v_type)])
        if interactions:
            from .condtions import interaction_condition

            interaction_condition(interactions, video_length or 65)
            command.extend(["--interactions", *interactions])
        extra_env_dict = dict(extra_env or {})
        existing_pythonpath = extra_env_dict.pop("PYTHONPATH", None) or os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [str(REPO_ROOT), str(self.runtime_root)]
        if existing_pythonpath:
            pythonpath_entries.append(str(existing_pythonpath))
        env = {
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            **extra_env_dict,
        }
        return MatrixGame1RuntimePlan(
            command=tuple(command),
            env=env,
            workdir=str(self.runtime_root),
            checkpoint_dir=str(self.checkpoint_dir),
            output_dir=str(resolved_output_dir),
        )

    def run_plan(
        self,
        plan: MatrixGame1RuntimePlan,
        *,
        timeout_seconds: int = 3600,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute a Matrix-Game-1 runtime plan and collect generated videos.

        Args:
            plan: Command plan produced by :meth:`build_plan`.
            timeout_seconds: Maximum wall-clock seconds for the subprocess.
            log_dir: Optional directory for stdout/stderr logs.
        """
        started = time.monotonic()
        target_log_dir = Path(log_dir or plan.output_dir).expanduser().resolve()
        target_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = target_log_dir / "matrix_game_1_stdout.log"
        stderr_path = target_log_dir / "matrix_game_1_stderr.log"
        env = os.environ.copy()
        env.update(plan.env)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
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
        generated_files = sorted(str(path) for path in Path(plan.output_dir).expanduser().resolve().rglob("*.mp4"))
        ok = completed.returncode == 0 and bool(generated_files)
        return {
            "ok": ok,
            "status": "success" if ok else "failed",
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_count": len(generated_files),
            "generated_files": generated_files,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "runtime_plan": plan.to_dict(),
        }

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute Matrix-Game-1 or fail clearly when required assets are missing.

        Args:
            prompt: Optional text prompt.
            images: Optional conditioning image.
            video: Optional conditioning video.
            interactions: Optional keyboard or mouse interaction tokens.
            output_path: Optional execution report JSON path.
            fps: Optional requested output FPS.
            **kwargs: Runtime plan and execution metadata.
        """
        target = Path(output_path) if output_path is not None else Path.cwd() / "matrix_game_1_result.json"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        report_path = target.with_suffix(".json") if _is_video_output_path(target) else target
        preflight = self.preflight()
        output_dir = kwargs.get("output_dir") or target.parent
        image_path = _first_filesystem_path(
            kwargs.get("image_path"),
            kwargs.get("input_path"),
            kwargs.get("source_image_path"),
            images,
        )
        runtime_plan = None
        if isinstance(image_path, str) and preflight["code_ready"]:
            runtime_plan = self.build_plan(
                image_path=image_path,
                output_dir=output_dir,
                inference_steps=int(kwargs.get("inference_steps") or 50),
                fps=int(fps or kwargs.get("runtime_fps") or 16),
                max_images=int(kwargs.get("max_images") or 1),
                max_conditions=int(kwargs.get("max_conditions") or 1),
                bfloat16=bool(kwargs.get("bfloat16", True)),
                resolution=kwargs.get("resolution"),
                video_length=kwargs.get("video_length"),
                num_pre_frames=kwargs.get("num_pre_frames"),
                i2v_type=kwargs.get("i2v_type"),
                interactions=interactions,
                gpu_id=str(kwargs.get("gpu_id") or _gpu_id_from_device(self.device) or ""),
            )
        if kwargs.get("execute") is True and runtime_plan is not None and preflight["status"] == "ready":
            run_result = self.run_plan(
                runtime_plan,
                timeout_seconds=int(kwargs.get("timeout_seconds") or 3600),
                log_dir=target.parent,
            )
            generated_files = list(run_result["generated_files"])
            artifact_path = generated_files[0] if generated_files else str(target)
            if _is_video_output_path(target) and generated_files:
                target_resolved = target.resolve()
                video_candidates = [
                    Path(path)
                    for path in generated_files
                    if _is_video_output_path(Path(path))
                    and Path(path).resolve() != target_resolved
                    and Path(path).is_file()
                    and Path(path).stat().st_size > 0
                ]
                if video_candidates:
                    source_video = video_candidates[0]
                    shutil.copyfile(source_video, target)
                    artifact_path = str(target)
                    generated_files = [str(target), *[path for path in generated_files if str(Path(path)) != str(target)]]
                    run_result = {
                        **run_result,
                        "generated_files": generated_files,
                        "generated_count": len(generated_files),
                        "studio_output_path": str(target),
                        "studio_output_source": str(source_video),
                    }
            payload = {
                "status": run_result["status"],
                "ok": run_result["ok"],
                "model_id": self.model_id,
                "artifact_kind": "generated_video",
                "artifact_path": artifact_path,
                "artifact_files": run_result["generated_files"],
                "report_path": str(report_path),
                "backend": "worldfoundry.matrix_game_1.in_tree_runtime",
                "backend_quality": "public_runner_official_runner",
                "preflight": preflight,
                "request": {
                    "runtime_root": str(self.runtime_root),
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "conda_dir": "" if self.conda_dir is None else str(self.conda_dir),
                    "device": self.device,
                    "prompt": prompt,
                    "image_path": image_path,
                    "fps": fps,
                    "extra_kwargs": kwargs,
                },
                "runtime_result": run_result,
            }
            payload = jsonable(payload)
            report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return payload
        next_steps = [
            "Stage the full Skywork/Matrix-Game checkpoint shards under the configured checkpoint_dir.",
            "Install or repair the Matrix-Game-1 environment so conda_dir/bin/python exists.",
            "Pass a filesystem image_path for the official in-tree runner command plan.",
        ]
        reasons = [*self._blocked_reasons(), *next_steps]
        if runtime_plan is None:
            reasons.append("Matrix-Game-1 requires a filesystem image_path to run the official in-tree runner.")
            reasons.append(
                "Received path fields: "
                f"image_path={kwargs.get('image_path')!r}, "
                f"input_path={kwargs.get('input_path')!r}, "
                f"source_image_path={kwargs.get('source_image_path')!r}, "
                f"images_type={type(images).__name__}, "
                f"extra_keys={sorted(str(key) for key in kwargs)}."
            )
        if kwargs.get("execute") is not True:
            reasons.append("Matrix-Game-1 requires execute=True; preflight artifacts are no longer emitted.")
        raise RuntimeError("Matrix-Game-1 cannot run: " + " ".join(reasons))


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_CONDA_DIR",
    "EXPECTED_CHECKPOINT_FILES",
    "MatrixGame1Runtime",
    "MatrixGame1RuntimePlan",
    "OPTIONAL_CHECKPOINT_FILES",
    "REQUIRED_RUNTIME_FILES",
    "RUNTIME_ROOT",
]

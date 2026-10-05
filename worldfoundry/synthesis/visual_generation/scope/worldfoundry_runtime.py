from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core import jsonable
from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import package_module_root as package_root

DEFAULT_MODEL_DIR_CANDIDATES = (
    "${WORLDFOUNDRY_HFD_ROOT}/SCOPE",
    "${WORLDFOUNDRY_HFD_ROOT}/zizhaotong--SCOPE",
)
DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR_CANDIDATES[0]


def _registered_python_executable() -> str | None:
    try:
        from worldfoundry.runtime.conda import resolve_conda_executable
    except Exception:
        return None
    return resolve_conda_executable("scope", "python")


def runtime_root() -> Path:
    """Return the in-tree SCOPE runtime source root."""

    return package_root("worldfoundry.synthesis.visual_generation.scope.scope_runtime")


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[6]


def _repo_src_root() -> Path:
    project = _project_root()
    src_root = project / "src"
    if (src_root / "worldfoundry").is_dir():
        return src_root
    return project


def _expand_path_tokens(value: str | Path | None) -> str | None:
    if value is None:
        return None
    from worldfoundry.runtime.assets import expand_worldfoundry_path

    return str(expand_worldfoundry_path(os.path.expandvars(str(value))))


def _default_model_dir() -> str | None:
    for candidate in DEFAULT_MODEL_DIR_CANDIDATES:
        expanded = _expand_path_tokens(candidate)
        if expanded and Path(expanded).exists():
            return expanded
    return _expand_path_tokens(DEFAULT_MODEL_DIR)


class SCOPERuntime:
    """WorldFoundry adapter around the vendored SCOPE inference script."""

    MODEL_ID = "scope"
    DISPLAY_NAME = "SCOPE"

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        device: str = "cuda",
        python_executable: str | Path | None = None,
        default_work_dir: str | Path | None = None,
    ) -> None:
        self.model_id = self.MODEL_ID
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "action_conditioned_i2v"
        self.model_dir = _expand_path_tokens(model_dir or os.getenv("SCOPE_MODEL_DIR") or _default_model_dir())
        self.device = str(device)
        self.python_executable = str(python_executable or _registered_python_executable() or sys.executable)
        self.default_work_dir = (
            Path(default_work_dir).expanduser()
            if default_work_dir is not None
            else _project_root() / "tmp" / "scope"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "SCOPERuntime":
        """Create a lazy SCOPE runtime without importing torch or pandas."""

        del args, model_id
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["model_dir"] = str(pretrained_model_path)
        options.update(kwargs)
        return cls(
            model_dir=(
                options.get("model_dir")
                or options.get("checkpoint_root")
                or options.get("checkpoint_dir")
                or options.get("model_path")
            ),
            device=str(device or options.get("device") or "cuda"),
            python_executable=options.get("python_executable") or options.get("python"),
            default_work_dir=options.get("work_dir") or options.get("default_work_dir"),
        )

    @staticmethod
    def _target_path(output_path: str | Path | None) -> Path:
        if output_path is None:
            return (_project_root() / "tmp" / "scope" / "scope.mp4").resolve()
        target = Path(output_path).expanduser()
        if not target.suffix:
            target = target / "scope.mp4"
        return target.resolve()

    @staticmethod
    def _work_dir(output_path: str | Path | None) -> Path:
        if output_path is None:
            return (_project_root() / "tmp" / "scope" / ".scope_work").resolve()
        base = Path(output_path).expanduser()
        if base.suffix:
            return (base.parent / f".{base.stem}_scope_work").resolve()
        return (base / ".scope_work").resolve()

    @staticmethod
    def _materialize_image(images: Any, work_dir: Path) -> Path:
        if images is None:
            raise ValueError("SCOPE requires an input image.")
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
            if not images:
                raise ValueError("SCOPE received an empty image sequence.")
            images = images[0]
        if isinstance(images, (str, os.PathLike)):
            path = Path(images).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"SCOPE input image not found: {path}")
            return path.resolve()

        import numpy as np
        from PIL import Image

        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / "input_image.png"
        if isinstance(images, Image.Image):
            image = images.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(images)).convert("RGB")
        image.save(target)
        return target

    @staticmethod
    def _resolve_action_path(action_path: Any = None, interactions: Any = None) -> Path:
        candidate = action_path
        if candidate is None:
            if isinstance(interactions, Mapping):
                candidate = interactions.get("action_path") or interactions.get("actions_path")
            elif isinstance(interactions, Sequence) and not isinstance(interactions, (str, bytes, bytearray)):
                if len(interactions) == 1:
                    candidate = interactions[0]
            else:
                candidate = interactions
        if candidate is None:
            raise ValueError("SCOPE requires action_path or interactions pointing to an action parquet file.")
        path = Path(str(candidate)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SCOPE action parquet not found: {path}")
        return path.resolve()

    def _subprocess_env(self) -> dict[str, str]:
        pythonpath_parts = [str(runtime_root()), str(_repo_src_root())]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            pythonpath_parts.append(existing)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.device.startswith("cuda:"):
            cuda_index = self.device.split(":", 1)[1].strip()
            if cuda_index.isdigit():
                # CUDA device indices are relative to the caller's visible set.
                # The subprocess uses its own cuda:0 after narrowing that set.
                visible = env.get("CUDA_VISIBLE_DEVICES")
                if visible is None:
                    selected = cuda_index
                else:
                    devices = [value.strip() for value in visible.split(",") if value.strip()]
                    index = int(cuda_index)
                    if index >= len(devices):
                        raise ValueError(
                            f"SCOPE device {self.device!r} is outside "
                            f"CUDA_VISIBLE_DEVICES={visible!r}"
                        )
                    selected = devices[index]
                env["CUDA_VISIBLE_DEVICES"] = selected
                env["SCOPE_DEVICE"] = "cuda"
            else:
                env["SCOPE_DEVICE"] = self.device
        elif self.device and self.device != "cuda":
            env["SCOPE_DEVICE"] = self.device
        return env

    def _argv(
        self,
        *,
        image_path: Path,
        action_path: Path,
        output_dir: Path,
        prompt: str,
        height: int,
        width: int,
        max_frames: int,
        num_inference_steps: int,
        seed: int,
    ) -> list[str]:
        return [
            self.python_executable,
            str(runtime_root() / "inference.py"),
            "--model_dir",
            str(self.model_dir),
            "--input_image",
            str(image_path),
            "--action_path",
            str(action_path),
            "--prompt",
            prompt,
            "--output_dir",
            str(output_dir),
            "--height",
            str(int(height)),
            "--width",
            str(int(width)),
            "--max_frames",
            str(int(max_frames)),
            "--num_inference_steps",
            str(int(num_inference_steps)),
            "--seed",
            str(int(seed)),
        ]

    @staticmethod
    def _sha256(path: Path) -> str:
        return file_sha256(path)

    def predict(
        self,
        prompt: str,
        images: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        execute: bool = False,
        action_path: Any = None,
        height: int = 480,
        width: int = 832,
        max_frames: int = 81,
        num_inference_steps: int = 30,
        seed: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = prompt or ""
        generation = {
            "height": height,
            "width": width,
            "max_frames": max_frames,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "fps": fps or 20,
            **kwargs,
        }
        if not execute:
            raise RuntimeError("SCOPE requires execute=True; request-plan artifacts are no longer emitted.")
        if not self.device.startswith("cuda"):
            raise RuntimeError(
                "SCOPE execute=True requires a CUDA device because upstream inference uses CUDA; "
                f"got {self.device!r}."
            )

        target = self._target_path(output_path)
        work_dir = self._work_dir(output_path)
        output_dir = work_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._materialize_image(images, work_dir)
        resolved_action_path = self._resolve_action_path(action_path=action_path, interactions=interactions)
        argv = self._argv(
            image_path=image_path,
            action_path=resolved_action_path,
            output_dir=output_dir,
            prompt=prompt,
            height=height,
            width=width,
            max_frames=max_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        completed = subprocess.run(
            argv,
            cwd=str(runtime_root()),
            env=self._subprocess_env(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path = target.with_suffix(target.suffix + ".scope.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"SCOPE runtime failed with code {completed.returncode}; see {log_path}")

        generated = sorted(output_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not generated:
            raise FileNotFoundError(f"SCOPE runtime did not produce an mp4 under {output_dir}; see {log_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if generated[0].resolve() != target:
            shutil.copyfile(generated[0], target)

        return {
            "status": "success",
            "model_id": self.model_id,
            "model_name": self.model_name,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "generated_video_path": str(target),
            "artifact_sha256": self._sha256(target),
            "runtime": "worldfoundry.synthesis.visual_generation.scope.scope_runtime.inference",
            "backend_quality": "in_tree_scope_runtime",
            "run_dir": str(work_dir),
            "log_path": str(log_path),
            "metadata": {
                "prompt": prompt,
                "model_dir": self.model_dir,
                "image_path": str(image_path),
                "action_path": str(resolved_action_path),
                "generation": jsonable(generation),
            },
        }


__all__ = [
    "DEFAULT_MODEL_DIR",
    "SCOPERuntime",
    "runtime_root",
]

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    resolve_data_path,
)

DEFAULT_PROMPT = "A cinematic video with coherent motion, rich detail, and realistic lighting."
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "rolling_forcing"
MODEL_ALIASES = {
    "rolling": "rolling-forcing",
    "rolling-forcing": "rolling-forcing",
    "rollingforcing": "rolling-forcing",
}


def _path_values(*values: str | None) -> tuple[Path, ...]:
    return tuple(Path(value).expanduser() for value in values if value)


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _hf_snapshot_file_candidates(repo_id: str, relative_path: str | Path) -> tuple[Path, ...]:
    """Return local Hub-cache candidates without contacting Hugging Face."""

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    cache_roots = tuple(
        dict.fromkeys(
            path.expanduser()
            for path in (
                Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else hf_home / "hub",
                hf_home / "hub",
            )
        )
    )
    relative = Path(relative_path)
    candidates: list[Path] = []
    namespace = repo_id.replace("/", "--")
    for cache_root in cache_roots:
        repo_root = cache_root / f"models--{namespace}"
        snapshots = repo_root / "snapshots"
        main_ref = repo_root / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots / revision / relative)
        if snapshots.is_dir():
            candidates.extend(
                snapshot / relative for snapshot in sorted(snapshots.iterdir(), reverse=True) if snapshot.is_dir()
            )
    return tuple(dict.fromkeys(candidates))


def _canonical_model_id(value: Any) -> str:
    text = str(value or "rolling-forcing").strip().lower().replace("_", "-")
    if text in MODEL_ALIASES:
        return MODEL_ALIASES[text]
    if "rolling" in text:
        return "rolling-forcing"
    raise ValueError(f"Unsupported forcing model id: {value!r}. Expected rolling-forcing.")


def _ckpt_root_candidates() -> tuple[Path, ...]:
    return (
        checkpoint_root_path("RollingForcing", specific_env="WORLDFOUNDRY_ROLLING_FORCING_CKPT_ROOT"),
        checkpoint_root_path("TencentARC--RollingForcing"),
        hfd_root_path("TencentARC--RollingForcing"),
    )


def _wan_root_candidates() -> tuple[Path, ...]:
    roots = [
        os.environ.get("WORLDFOUNDRY_WAN_MODELS_ROOT"),
        str(checkpoint_root_path()),
    ]
    return _path_values(*roots)


def _default_checkpoint_path() -> str:
    candidates = tuple(
        path
        for root in _ckpt_root_candidates()
        for path in (
            root / "checkpoints" / "rolling_forcing_dmd.pt",
            root / "rolling_forcing_dmd.pt",
        )
    ) + _hf_snapshot_file_candidates(
        "TencentARC/RollingForcing",
        "checkpoints/rolling_forcing_dmd.pt",
    )
    return str(_first_existing(candidates) or candidates[0])


def _default_config_path() -> str:
    name = "rolling_forcing_dmd.yaml"
    candidates = (
        resolve_data_path("models", "runtime", "configs", "rolling_forcing", name),
        RUNTIME_DIR / "configs" / name,
    )
    return str(_first_existing(candidates) or candidates[0])


def _default_wan_models_root() -> str:
    local = _first_existing(_wan_root_candidates())
    return str(local or _wan_root_candidates()[0])


def _wan_model_dir(root: str | Path, model_name: str) -> Path:
    """Resolve canonical and flat-HFD Wan checkpoint directory layouts."""

    root = Path(root).expanduser()
    if root.name in {model_name, f"Wan-AI--{model_name}"}:
        return root
    canonical = root / model_name
    if canonical.exists():
        return canonical
    hfd_flat = root / f"Wan-AI--{model_name}"
    if hfd_flat.exists():
        return hfd_flat
    return canonical


def _load_video_frames(video_path: str | Path) -> np.ndarray:
    import imageio.v3 as iio

    frames = [np.asarray(frame).astype(np.uint8) for frame in iio.imiter(str(video_path))]
    if not frames:
        raise ValueError(f"No frames found in generated forcing video: {video_path}")
    return np.stack(frames, axis=0)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _symlink_dir(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    if not target.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=True)


def _forcing_prompt_root() -> Path:
    override = os.getenv("WORLDFOUNDRY_FORCING_PROMPTS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return resolve_data_path("test_cases", "forcing", "prompts").resolve()


def _latest_mp4(directory: Path) -> Path:
    candidates = sorted(
        [path for path in directory.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No generated mp4 found under {directory}")
    return candidates[0]


class RollingForcingRuntime:
    """In-tree runtime bridge for the official RollingForcing runner."""

    MODEL_ID = "rolling-forcing"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        wan_models_root: str | Path | None = None,
        device: str = "cuda",
        python_executable: str | None = None,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_model_id = _canonical_model_id(model_id or self.MODEL_ID)
        self.model_id = resolved_model_id
        self.model_name = "RollingForcing"
        self.generation_type = "text_to_video"
        self.runtime_root = str(RUNTIME_DIR)
        configured_checkpoint = Path(checkpoint_path or _default_checkpoint_path()).expanduser()
        if not configured_checkpoint.is_file() and "Self-Forcing" in configured_checkpoint.parts:
            staged_checkpoint = Path(_default_checkpoint_path())
            if staged_checkpoint.is_file():
                configured_checkpoint = staged_checkpoint
        self.checkpoint_path = str(configured_checkpoint)
        self.config_path = str(config_path or _default_config_path())
        self.wan_models_root = str(Path(wan_models_root or _default_wan_models_root()).expanduser())
        self.device = device
        self.python_executable = python_executable or sys.executable
        self.defaults = {
            "num_output_frames": 126,
            "seed": 0,
            "num_samples": 1,
            "fps": 16,
            "use_ema": True,
            "save_with_index": False,
            "report_timing": False,
        }
        if defaults:
            self.defaults.update(dict(defaults))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> RollingForcingRuntime:
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = _canonical_model_id(
            options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID
        )
        defaults = {
            key: options.pop(key)
            for key in tuple(options)
            if key
            in {
                "num_output_frames",
                "seed",
                "num_samples",
                "fps",
                "use_ema",
                "save_with_index",
                "report_timing",
                "extended_prompt",
            }
        }
        return cls(
            model_id=resolved_model_id,
            checkpoint_path=options.get("checkpoint_path") or options.get("ckpt_path"),
            config_path=options.get("config_path"),
            wan_models_root=options.get("wan_models_root") or options.get("wan_root"),
            device=str(device or options.get("device") or "cuda"),
            python_executable=options.get("python_executable"),
            defaults=defaults,
        )

    def runtime_plan(self, **overrides: Any) -> dict[str, Any]:
        options = {**self.defaults, **overrides}
        return {
            "status": "prepared",
            "backend": f"official_{self.model_id}_in_tree_runtime",
            "backend_quality": "official_in_tree_runtime",
            "model_id": self.model_id,
            "runtime_root": self.runtime_root,
            "runtime_root_exists": Path(self.runtime_root).is_dir(),
            "config_path": self.config_path,
            "config_exists": Path(self.config_path).is_file(),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_exists": Path(self.checkpoint_path).is_file(),
            "wan_models_root": self.wan_models_root,
            "wan_1_3b_exists": _wan_model_dir(self.wan_models_root, "Wan2.1-T2V-1.3B").is_dir(),
            "wan_14b_exists": _wan_model_dir(self.wan_models_root, "Wan2.1-T2V-14B").is_dir(),
            "python_executable": self.python_executable,
            "i2v": bool(options.get("i2v", False)),
            "license": "RollingForcing academic-only; commercial and production use prohibited",
        }

    def _runtime_cwd(self, root: Path) -> Path:
        cwd = root / "runtime_cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        config_parent = Path(self.config_path).expanduser().resolve().parent
        _symlink_dir(config_parent, cwd / "configs")
        _symlink_dir(_forcing_prompt_root(), cwd / "prompts")
        wan_dir = cwd / "wan_models"
        for name in ("Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"):
            _symlink_dir(_wan_model_dir(self.wan_models_root, name), wan_dir / name)
        return cwd

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["WORLDFOUNDRY_WAN_MODELS_ROOT"] = str(Path(self.wan_models_root).expanduser().resolve())
        env["WORLDFOUNDRY_FORCING_VARIANT"] = self.model_id
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.device.startswith("cuda:") or (self.device == "cuda" and "CUDA_VISIBLE_DEVICES" not in env):
            from worldfoundry.core.platform.device import cuda_visible_devices_from_device

            visible_devices = cuda_visible_devices_from_device(self.device, inherited=env.get("CUDA_VISIBLE_DEVICES"))
            if visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = visible_devices
        return env

    def _write_prompt_file(self, root: Path, prompt: str, extended_prompt: str | None) -> tuple[Path, Path | None]:
        prompt_path = root / "prompts.txt"
        prompt_path.write_text((prompt or DEFAULT_PROMPT).replace("\n", " ").strip() + "\n", encoding="utf-8")
        if extended_prompt is None:
            return prompt_path, None
        extended_path = root / "extended_prompts.txt"
        extended_path.write_text(extended_prompt.replace("\n", " ").strip() + "\n", encoding="utf-8")
        return prompt_path, extended_path

    def _command(
        self,
        *,
        data_path: Path,
        output_dir: Path,
        options: Mapping[str, Any],
        extended_prompt_path: Path | None,
    ) -> list[str]:
        command = [
            "--config_path",
            str(Path(self.config_path).expanduser().resolve()),
            "--checkpoint_path",
            str(Path(self.checkpoint_path).expanduser().resolve()),
            "--data_path",
            str(data_path),
            "--output_folder",
            str(output_dir),
            "--num_output_frames",
            str(int(options.get("num_output_frames", 21))),
            "--seed",
            str(int(options.get("seed", 0))),
        ]
        command.extend(["--num_samples", str(int(options.get("num_samples", 1)))])
        command.extend(["--wan_models_root", str(Path(self.wan_models_root).expanduser().resolve())])
        if extended_prompt_path is not None:
            command.extend(["--extended_prompt_path", str(extended_prompt_path)])
        if _as_bool(options.get("save_with_index", True)):
            command.append("--save_with_index")
        if _as_bool(options.get("use_ema", False)):
            command.append("--use_ema")
        return package_module_command(
            "worldfoundry.synthesis.visual_generation.rolling_forcing.inference",
            command,
            python_executable=self.python_executable,
        )

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | np.ndarray:
        del video, fps, return_dict, show_progress
        options = {**self.defaults, **kwargs}
        if interactions and "seed" not in options:
            first = list(interactions)[0]  # noqa: RUF015 - Keeps eager list construction and IndexError.
            if isinstance(first, int):
                options["seed"] = first
        output_target = Path(output_path).expanduser() if output_path is not None else None
        if output_target is None:
            output_target = Path(tempfile.mkdtemp(prefix=f"{self.model_id}_")) / f"{self.model_id}.mp4"
        output_target = output_target.resolve()
        prompt_text = prompt or DEFAULT_PROMPT
        i2v = images is not None or _as_bool(options.get("i2v", False))
        if i2v:
            raise ValueError(
                "RollingForcing's released checkpoint and official inference route are text-to-video only."
            )
        with tempfile.TemporaryDirectory(prefix=f"worldfoundry_{self.model_id}_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            runtime_cwd = self._runtime_cwd(temp_dir)
            output_dir = temp_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            data_path, extended_prompt_path = self._write_prompt_file(
                temp_dir,
                prompt_text,
                options.get("extended_prompt"),
            )
            command = self._command(
                data_path=data_path,
                output_dir=output_dir,
                options=options,
                extended_prompt_path=extended_prompt_path,
            )
            completed = subprocess.run(
                command,
                check=False,
                cwd=str(runtime_cwd),
                env=self._subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                tail = "\n".join((completed.stdout or "").splitlines()[-40:])
                raise RuntimeError(f"{self.model_name} command failed with exit code {completed.returncode}.\n{tail}")
            generated = _latest_mp4(output_dir)
            output_target.parent.mkdir(parents=True, exist_ok=True)
            if generated.resolve() != output_target:
                shutil.copy2(generated, output_target)

        frames = _load_video_frames(output_target)
        return {
            "artifact_path": str(output_target),
            "video": frames,
            "prompt": prompt_text,
            "model_id": self.model_id,
            "runtime_plan": self.runtime_plan(i2v=i2v),
            "command": command,
        }


__all__ = ["RollingForcingRuntime"]

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    resolve_data_path,
)
from worldfoundry.core.io.paths import (
    package_module_root as package_root,
)

DEFAULT_PROMPT = "A cinematic video with coherent motion, rich detail, and realistic lighting."
TORCHVISION_VIDEO_COMPAT_DIR = Path(__file__).resolve().parent / "torchvision_video_compat"
FORCING_RUNTIME_DIR = Path(__file__).resolve().parent
MODEL_ALIASES = {
    "self": "self-forcing",
    "self-forcing": "self-forcing",
    "selfforcing": "self-forcing",
    "causal": "causal-forcing",
    "causal-forcing": "causal-forcing",
    "causalforcing": "causal-forcing",
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
                snapshot / relative
                for snapshot in sorted(snapshots.iterdir(), reverse=True)
                if snapshot.is_dir()
            )
    return tuple(dict.fromkeys(candidates))


def _canonical_model_id(value: Any) -> str:
    text = str(value or "self-forcing").strip().lower().replace("_", "-")
    if text in MODEL_ALIASES:
        return MODEL_ALIASES[text]
    if "rolling" in text:
        return "rolling-forcing"
    if "causal" in text:
        return "causal-forcing"
    if "self" in text:
        return "self-forcing"
    raise ValueError(
        f"Unsupported forcing model id: {value!r}. "
        "Expected self-forcing, causal-forcing, or rolling-forcing."
    )


def _local_repo_candidates(model_id: str) -> tuple[Path, ...]:
    if model_id == "rolling-forcing":
        # RollingForcing is deliberately source-closed inside WorldFoundry: no
        # checkout of the upstream repository is consulted at runtime.
        return (FORCING_RUNTIME_DIR.parent / "rolling_forcing",)
    env_name = (
        "WORLDFOUNDRY_CAUSAL_FORCING_REPO_ROOT"
        if model_id == "causal-forcing"
        else "WORLDFOUNDRY_SELF_FORCING_REPO_ROOT"
    )
    in_tree = FORCING_RUNTIME_DIR / (
        "causal_forcing_runtime" if model_id == "causal-forcing" else "self_forcing_runtime"
    )
    if os.environ.get(env_name):
        return (in_tree, Path(os.environ[env_name]).expanduser())
    return (in_tree,)


def _ckpt_root_candidates(model_id: str) -> tuple[Path, ...]:
    if model_id == "rolling-forcing":
        return (
            checkpoint_root_path("RollingForcing", specific_env="WORLDFOUNDRY_ROLLING_FORCING_CKPT_ROOT"),
            checkpoint_root_path("TencentARC--RollingForcing"),
            hfd_root_path("TencentARC--RollingForcing"),
        )
    ckpt_name = "Causal-Forcing" if model_id == "causal-forcing" else "Self-Forcing"
    env_name = (
        "WORLDFOUNDRY_CAUSAL_FORCING_CKPT_ROOT"
        if model_id == "causal-forcing"
        else "WORLDFOUNDRY_SELF_FORCING_CKPT_ROOT"
    )
    owner_layout = (
        "zhuhz22--Causal-Forcing"
        if model_id == "causal-forcing"
        else "gdhe17--Self-Forcing"
    )
    return (
        checkpoint_root_path(ckpt_name, specific_env=env_name),
        checkpoint_root_path(owner_layout),
    )


def _wan_root_candidates() -> tuple[Path, ...]:
    roots = [
        os.environ.get("WORLDFOUNDRY_WAN_MODELS_ROOT"),
        str(checkpoint_root_path()),
    ]
    return _path_values(*roots)


def _default_repo_root(model_id: str) -> str:
    local = _first_existing(_local_repo_candidates(model_id))
    return str(local or _local_repo_candidates(model_id)[0])


def _default_ckpt_root(model_id: str) -> Path:
    local = _first_existing(_ckpt_root_candidates(model_id))
    return local or _ckpt_root_candidates(model_id)[0]


def _default_checkpoint_path(model_id: str) -> str:
    root = _default_ckpt_root(model_id)
    if model_id == "rolling-forcing":
        roots = _ckpt_root_candidates(model_id)
        candidates = tuple(
            path
            for candidate_root in roots
            for path in (
                candidate_root / "checkpoints" / "rolling_forcing_dmd.pt",
                candidate_root / "rolling_forcing_dmd.pt",
            )
        ) + _hf_snapshot_file_candidates(
            "TencentARC/RollingForcing",
            "checkpoints/rolling_forcing_dmd.pt",
        )
    elif model_id == "causal-forcing":
        candidates = (
            root / "chunkwise" / "causal_forcing.pt",
            root / "framewise" / "causal_forcing.pt",
            root / "causal-forcing++" / "framewise-2step.pt",
            root / "causal-forcing++" / "framewise-1step.pt",
        )
    else:
        candidates = (
            root / "checkpoints" / "self_forcing_dmd.pt",
            root / "self_forcing_dmd.pt",
            root / "checkpoints" / "self_forcing_sid.pt",
            checkpoint_root_path("LongSANA_2B_480p_self_forcing")
            / "checkpoints"
            / "LongSANA_2B_480p_self_forcing.pt",
        )
    return str(_first_existing(candidates) or candidates[0])


def _default_config_path(model_id: str, runtime_root: str | Path) -> str:
    if model_id == "rolling-forcing":
        name = "rolling_forcing_dmd.yaml"
        family = "rolling_forcing"
    elif model_id == "causal-forcing":
        name = "causal_forcing_dmd_chunkwise.yaml"
        family = "causal_forcing"
    else:
        name = "self_forcing_dmd.yaml"
        family = "self_forcing"
    candidates = (
        resolve_data_path("models", "runtime", "configs", family, name),
        Path(runtime_root).expanduser() / "configs" / name,
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


def _ensure_runtime_prompts(runtime_root: Path) -> None:
    _symlink_dir(_forcing_prompt_root(), runtime_root / "prompts")


def _latest_mp4(directory: Path) -> Path:
    candidates = sorted(
        [path for path in directory.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No generated mp4 found under {directory}")
    return candidates[0]


def _image_source(images: Any) -> Any:
    if images is None:
        return None
    if isinstance(images, (list, tuple)) and images:
        return images[0]
    return images


def _write_torchvision_video_compat(root: Path) -> Path:
    compat_dir = root / "torchvision_video_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        '''"""Runtime-only compatibility for official forcing scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy_video(video_array: Any) -> np.ndarray:
    if hasattr(video_array, "detach"):
        video_array = video_array.detach().cpu().numpy()
    array = np.asarray(video_array)
    if array.ndim != 4:
        raise ValueError(f"write_video expects a 4D video array, got shape {array.shape!r}")
    if array.shape[1] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 1, -1)
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _install_write_video() -> None:
    try:
        import torchvision.io as tv_io
    except Exception:
        return

    def write_video(
        filename: str,
        video_array: Any,
        fps: float,
        video_codec: str = "libx264",
        options: dict[str, Any] | None = None,
        audio_array: Any | None = None,
        audio_fps: float | None = None,
        audio_codec: str | None = None,
    ) -> None:
        del audio_array, audio_fps, audio_codec
        import imageio.v2 as imageio

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer_options = dict(options or {})
        if video_codec:
            writer_options.setdefault("codec", video_codec)
        with imageio.get_writer(str(path), fps=fps, **writer_options) as writer:
            for frame in _to_numpy_video(video_array):
                writer.append_data(frame)

    tv_io.write_video = write_video


_install_write_video()
''',
        encoding="utf-8",
    )
    return compat_dir


class _ForcingRuntime:
    """Shared in-tree runtime bridge for forcing-family official runners."""

    MODEL_ID: str | None = None
    DISPLAY_NAMES = {
        "self-forcing": "Self-Forcing",
        "causal-forcing": "Causal-Forcing",
        "rolling-forcing": "RollingForcing",
    }

    def __init__(
        self,
        *,
        model_id: str | None = None,
        runtime_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        wan_models_root: str | Path | None = None,
        device: str = "cuda",
        python_executable: str | None = None,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_model_id = _canonical_model_id(model_id or self.MODEL_ID or "self-forcing")
        if self.MODEL_ID is not None and resolved_model_id != self.MODEL_ID:
            raise ValueError(f"{self.__class__.__name__} cannot load {resolved_model_id!r}.")
        resolved_runtime_root = str(Path(runtime_root or _default_repo_root(resolved_model_id)).expanduser())
        self.model_id = resolved_model_id
        self.model_name = self.DISPLAY_NAMES[resolved_model_id]
        self.generation_type = (
            "text_to_video" if resolved_model_id == "rolling-forcing" else "text_or_image_to_video"
        )
        self.runtime_root = resolved_runtime_root
        configured_checkpoint = Path(
            checkpoint_path or _default_checkpoint_path(resolved_model_id)
        ).expanduser()
        stale_directory = (
            "Causal-Forcing"
            if resolved_model_id == "causal-forcing"
            else "Self-Forcing"
        )
        if not configured_checkpoint.is_file() and stale_directory in configured_checkpoint.parts:
            staged_checkpoint = Path(_default_checkpoint_path(resolved_model_id))
            if staged_checkpoint.is_file():
                configured_checkpoint = staged_checkpoint
        self.checkpoint_path = str(configured_checkpoint)
        self.config_path = str(config_path or _default_config_path(resolved_model_id, resolved_runtime_root))
        self.wan_models_root = str(Path(wan_models_root or _default_wan_models_root()).expanduser())
        self.device = device
        self.python_executable = python_executable or sys.executable
        self.defaults = {
            "num_output_frames": 126 if resolved_model_id == "rolling-forcing" else 21,
            "seed": 0,
            "num_samples": 1,
            "fps": 16,
            "use_ema": resolved_model_id in {"self-forcing", "rolling-forcing"},
            "save_with_index": resolved_model_id == "self-forcing",
            "report_timing": False,
        }
        if defaults:
            self.defaults.update(dict(defaults))
        _ensure_runtime_prompts(Path(self.runtime_root))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "_ForcingRuntime":
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = _canonical_model_id(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
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
            runtime_root=options.get("runtime_root") or options.get("repo_root"),
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
            "wan_1_3b_exists": _wan_model_dir(
                self.wan_models_root, "Wan2.1-T2V-1.3B"
            ).is_dir(),
            "wan_14b_exists": _wan_model_dir(
                self.wan_models_root, "Wan2.1-T2V-14B"
            ).is_dir(),
            "python_executable": self.python_executable,
            "i2v": bool(options.get("i2v", False)),
            "license": (
                "RollingForcing academic-only; commercial and production use prohibited"
                if self.model_id == "rolling-forcing"
                else None
            ),
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

    def _subprocess_env(self, *, compat_dir: Path | None = None) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = [
            str(compat_dir) if compat_dir is not None else "",
            str(package_root("worldfoundry").parent),
            self.runtime_root,
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
        env["WORLDFOUNDRY_WAN_MODELS_ROOT"] = str(
            Path(self.wan_models_root).expanduser().resolve()
        )
        env["WORLDFOUNDRY_FORCING_VARIANT"] = self.model_id
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.device.startswith("cuda:"):
            env.setdefault("CUDA_VISIBLE_DEVICES", self.device.split(":", 1)[1])
        return env

    def _write_prompt_file(self, root: Path, prompt: str, extended_prompt: str | None) -> tuple[Path, Path | None]:
        prompt_path = root / "prompts.txt"
        prompt_path.write_text((prompt or DEFAULT_PROMPT).replace("\n", " ").strip() + "\n", encoding="utf-8")
        if extended_prompt is None:
            return prompt_path, None
        extended_path = root / "extended_prompts.txt"
        extended_path.write_text(extended_prompt.replace("\n", " ").strip() + "\n", encoding="utf-8")
        return prompt_path, extended_path

    def _write_i2v_dataset(self, root: Path, prompt: str, images: Any) -> Path:
        source = _image_source(images)
        if source is None:
            raise ValueError("I2V generation requires an image path or image-like object.")
        from PIL import Image

        data_dir = root / "i2v_data"
        aspect = "26-15"
        image_dir = data_dir / aspect
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "000000.png"
        if isinstance(source, (str, os.PathLike)):
            image = Image.open(Path(source).expanduser()).convert("RGB")
        elif hasattr(source, "convert"):
            image = source.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(source).astype(np.uint8)).convert("RGB")
        image.save(image_path)
        width, height = image.size
        metadata = [
            {
                "file_name": image_path.name,
                "caption": (prompt or DEFAULT_PROMPT).replace("\n", " ").strip(),
                "type": "worldfoundry_i2v",
                "origin_width": width,
                "origin_height": height,
                "target_crop": {
                    "target_bbox": [0, 0, width, height],
                    "target_ratio": aspect,
                },
            }
        ]
        (data_dir / f"target_crop_info_{aspect}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return data_dir

    def _command(
        self,
        *,
        data_path: Path,
        output_dir: Path,
        options: Mapping[str, Any],
        extended_prompt_path: Path | None,
        i2v: bool,
    ) -> list[str]:
        command = [
            self.python_executable,
            str((Path(self.runtime_root) / "inference.py").resolve()),
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
        if self.model_id in {"self-forcing", "rolling-forcing"}:
            command.extend(["--num_samples", str(int(options.get("num_samples", 1)))])
            if self.model_id == "rolling-forcing":
                command.extend(
                    ["--wan_models_root", str(Path(self.wan_models_root).expanduser().resolve())]
                )
            if extended_prompt_path is not None:
                command.extend(["--extended_prompt_path", str(extended_prompt_path)])
            if _as_bool(options.get("save_with_index", True)):
                command.append("--save_with_index")
        if i2v:
            command.append("--i2v")
        if _as_bool(options.get("use_ema", False)):
            command.append("--use_ema")
        if self.model_id == "causal-forcing" and _as_bool(options.get("report_timing", False)):
            command.append("--report_timing")
        return command

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
            first = list(interactions)[0]
            if isinstance(first, int):
                options["seed"] = first
        output_target = Path(output_path).expanduser() if output_path is not None else None
        if output_target is None:
            output_target = Path(tempfile.mkdtemp(prefix=f"{self.model_id}_")) / f"{self.model_id}.mp4"
        output_target = output_target.resolve()
        prompt_text = prompt or DEFAULT_PROMPT
        i2v = images is not None or _as_bool(options.get("i2v", False))
        if self.model_id == "rolling-forcing" and i2v:
            raise ValueError(
                "RollingForcing's released checkpoint and official inference route are text-to-video only."
            )
        if self.model_id in {"self-forcing", "causal-forcing"}:
            num_output_frames = int(options.get("num_output_frames", 21))
            generated_latent_frames = num_output_frames - 1 if i2v else num_output_frames
            if generated_latent_frames <= 0 or generated_latent_frames % 3:
                mode = "I2V" if i2v else "T2V"
                rule = "1 + 3k" if i2v else "3k"
                raise ValueError(
                    f"{self.model_name} {mode} num_output_frames must follow {rule}; "
                    f"got {num_output_frames}. The released causal checkpoint generates "
                    "three latent frames per block."
                )

        with tempfile.TemporaryDirectory(prefix=f"worldfoundry_{self.model_id}_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            runtime_cwd = self._runtime_cwd(temp_dir)
            compat_dir = TORCHVISION_VIDEO_COMPAT_DIR
            output_dir = temp_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            if i2v:
                data_path = self._write_i2v_dataset(temp_dir, prompt_text, images)
                extended_prompt_path = None
            else:
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
                i2v=i2v,
            )
            completed = subprocess.run(
                command,
                check=False,
                cwd=str(runtime_cwd),
                env=self._subprocess_env(compat_dir=compat_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                tail = "\n".join((completed.stdout or "").splitlines()[-40:])
                raise RuntimeError(
                    f"{self.model_name} command failed with exit code {completed.returncode}.\n{tail}"
                )
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


class SelfForcingRuntime(_ForcingRuntime):
    """In-tree runtime bridge for the official Self-Forcing runner."""

    MODEL_ID = "self-forcing"


class CausalForcingRuntime(_ForcingRuntime):
    """In-tree runtime bridge for the official Causal-Forcing runner."""

    MODEL_ID = "causal-forcing"


class RollingForcingRuntime(_ForcingRuntime):
    """In-tree runtime bridge for the official RollingForcing runner."""

    MODEL_ID = "rolling-forcing"


__all__ = ["CausalForcingRuntime", "RollingForcingRuntime", "SelfForcingRuntime"]

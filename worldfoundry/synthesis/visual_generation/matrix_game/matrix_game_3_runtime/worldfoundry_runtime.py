from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

from worldfoundry.core import cuda_visible_devices_from_device, load_pil_image, load_video_frames
from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    package_module_root as package_root,
    project_root,
)
from worldfoundry.evaluation.utils import worldfoundry_data_path

import numpy as np
import torch
from worldfoundry.runtime.env import resolve_hfd_root


DEFAULT_MATRIX_GAME3_ALIASES = {
    "",
    "matrix-game-3",
    "matrix-game3",
    "matrix_game_3",
    "Matrix-Game-3",
    "Skywork/Matrix-Game-3.0",
}


DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR = resolve_hfd_root() / "Skywork--Matrix-Game-3.0"
DEFAULT_MATRIX_GAME3_ASSET_ROOT = worldfoundry_data_path(
    "models",
    "runtime", "configs",
    "matrix_game_3",
    "assets",
)
DEFAULT_MATRIX_GAME3_MOUSE_ICON = DEFAULT_MATRIX_GAME3_ASSET_ROOT / "images" / "mouse.png"


def _hf_snapshot_dirs(root: Path) -> list[Path]:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def _checkpoint_layout_complete(checkpoint_path: Path) -> bool:
    required = [
        checkpoint_path / "models_t5_umt5-xxl-enc-bf16.pth",
        checkpoint_path / "google" / "umt5-xxl",
        checkpoint_path / "Wan2.2_VAE.pth",
    ]
    optional_model_dirs = [
        checkpoint_path / "base_distilled_model",
        checkpoint_path / "base_model",
    ]
    return all(path.exists() for path in required) and any(path.exists() for path in optional_model_dirs)


def _matrix_game3_checkpoint_candidates(primary: str | Path | None = None) -> list[Path]:
    raw: list[Path] = []
    if primary not in {None, ""}:
        raw.append(Path(str(primary)).expanduser())
    raw.extend(
        [
            checkpoint_root_path("Matrix-Game-3.0"),
            hfd_root_path("Skywork--Matrix-Game-3.0"),
            hfd_root_path("custom--Matrix-Game-3.0"),
            DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR,
            checkpoint_root_path("huggingface", "hub", "models--Skywork--Matrix-Game-3.0"),
        ]
    )

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in raw:
        if root.is_file():
            root = root.parent
        expanded = [root, *_hf_snapshot_dirs(root)]
        for item in expanded:
            key = str(item)
            if key not in seen:
                candidates.append(item)
                seen.add(key)
    return candidates


def runtime_root() -> Path:
    return package_root("worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime")


def _validate_runtime_root(path_value: Path) -> Path:
    sentinels = [("generate.py",), ("pipeline", "inference_pipeline.py")]
    missing = ["/".join(parts) for parts in sentinels if not path_value.joinpath(*parts).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Matrix-Game-3 in-tree runtime is incomplete at {path_value}: missing {joined}")
    return path_value.resolve()


def resolve_checkpoint_dir(path_value: Optional[str]) -> str:
    requested = "" if path_value is None else str(path_value).strip()
    primary: str | Path | None
    if requested in DEFAULT_MATRIX_GAME3_ALIASES:
        primary = None
    else:
        candidate = Path(requested).expanduser()
        primary = candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()

    candidates = _matrix_game3_checkpoint_candidates(primary)
    for candidate in candidates:
        if _checkpoint_layout_complete(candidate):
            return str(candidate.resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR)


def _distributed_rank_from_env() -> int:
    for key in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(key)
        if value not in (None, ""):
            return int(value)
    return 0


def _wait_for_video_readable(video_path: Path, timeout: float = 180.0, interval: float = 1.0) -> None:
    import imageio

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    last_size = -1
    while time.monotonic() < deadline:
        if video_path.is_file():
            size = video_path.stat().st_size
            if size > 0 and size == last_size:
                try:
                    reader = imageio.get_reader(str(video_path))
                    reader.get_meta_data()
                    reader.close()
                    return
                except Exception as exc:  # imageio reports half-written mp4 files as backend-specific errors.
                    last_error = exc
            last_size = size
        time.sleep(interval)
    if last_error is not None:
        raise RuntimeError(f"Matrix-Game-3 output video is not readable yet: {video_path}") from last_error
    raise FileNotFoundError(f"Matrix-Game-3 output video not found: {video_path}")


def build_subprocess_env(runtime_path: str, device: str | None = None) -> dict:
    env = os.environ.copy()
    pythonpath_entries = [str(project_root() / "src"), runtime_path]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    visible_device = cuda_visible_devices_from_device(device)
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_device
    env["WORLDFOUNDRY_MATRIX_GAME3_ASSET_ROOT"] = str(DEFAULT_MATRIX_GAME3_ASSET_ROOT)
    return env


class MatrixGame3Runtime:
    """Subprocess wrapper around Matrix-Game-3 official inference code."""

    def __init__(
        self,
        runtime_path: str,
        checkpoint_dir: str,
        device: str = "cuda",
        defaults: Optional[dict] = None,
    ):
        self.runtime_root = runtime_path
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.defaults = defaults or {}
        self._resident_pipeline = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        args=None,
        device=None,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ) -> "MatrixGame3Runtime":
        del args
        runtime_path = str(_validate_runtime_root(runtime_root()))
        checkpoint_source = checkpoint_dir
        if checkpoint_source is None and pretrained_model_path is not None:
            model_ref = str(pretrained_model_path).strip()
            if model_ref not in DEFAULT_MATRIX_GAME3_ALIASES:
                checkpoint_source = model_ref
        resolved_checkpoint_dir = resolve_checkpoint_dir(checkpoint_source)
        return cls(
            runtime_path=runtime_path,
            checkpoint_dir=resolved_checkpoint_dir,
            device=device or "cuda",
            defaults=kwargs,
        )

    def predict(
        self,
        image,
        prompt: str,
        keyboard_condition,
        mouse_condition,
        num_iterations: int,
        output_dir: Optional[str] = None,
        save_name: str = "matrix_game_3",
        size: str | Sequence[int] = "704*1280",
        fps: int = 17,
        seed: int = 42,
        num_inference_steps: Optional[int] = None,
        sample_shift: Optional[float] = None,
        sample_guide_scale: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        vae_type: Optional[str] = None,
        lightvae_pruning_rate: Optional[float] = None,
        fa_version: Optional[str] = None,
        visualize_ops: bool = True,
        use_base_model: Optional[bool] = None,
        use_int8: Optional[bool] = None,
        verify_quant: Optional[bool] = None,
        use_async_vae: Optional[bool] = None,
        async_vae_warmup_iters: Optional[int] = None,
        compile_vae: Optional[bool] = None,
        ulysses_size: Optional[int] = None,
        t5_fsdp: Optional[bool] = None,
        t5_cpu: Optional[bool] = None,
        dit_fsdp: Optional[bool] = None,
        convert_model_dtype: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ):
        resolved_checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
        self._ensure_checkpoint_layout(resolved_checkpoint_dir)

        output_dir_path = Path(output_dir or tempfile.mkdtemp(prefix="matrix_game3_")).expanduser().resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        image_path = self._materialize_image(image, output_dir_path)
        actions_path = output_dir_path / f"{save_name}_actions.json"
        with actions_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "keyboard_condition": self._tensor_to_list(keyboard_condition),
                    "mouse_condition": self._tensor_to_list(mouse_condition),
                },
                file,
            )

        size_value = self._normalize_size(size)
        command = [
            sys.executable,
            "-m",
            "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime.worldfoundry_runner",
            "--runtime_root",
            self.runtime_root,
            "--ckpt_dir",
            resolved_checkpoint_dir,
            "--actions_json",
            str(actions_path),
            "--image_path",
            str(image_path),
            "--prompt",
            prompt or "",
            "--output_dir",
            str(output_dir_path),
            "--save_name",
            save_name,
            "--size",
            size_value,
            "--fps",
            str(int(fps)),
            "--seed",
            str(int(seed)),
            "--num_iterations",
            str(int(num_iterations)),
            "--num_inference_steps",
            str(int(num_inference_steps if num_inference_steps is not None else self.defaults.get("num_inference_steps", 3))),
            "--sample_guide_scale",
            str(float(sample_guide_scale if sample_guide_scale is not None else self.defaults.get("sample_guide_scale", 5.0))),
            "--ulysses_size",
            str(int(ulysses_size if ulysses_size is not None else self.defaults.get("ulysses_size", 1))),
            "--vae_type",
            str(vae_type or self.defaults.get("vae_type", "wan")),
        ]
        resolved_fa_version = fa_version if fa_version is not None else self.defaults.get("fa_version")
        if resolved_fa_version is not None:
            command.extend(["--fa_version", str(resolved_fa_version)])
        if sample_shift is not None or self.defaults.get("sample_shift") is not None:
            command.extend(["--sample_shift", str(float(sample_shift if sample_shift is not None else self.defaults["sample_shift"]))])
        if lightvae_pruning_rate is not None:
            command.extend(["--lightvae_pruning_rate", str(float(lightvae_pruning_rate))])
        elif self.defaults.get("lightvae_pruning_rate") is not None:
            command.extend(["--lightvae_pruning_rate", str(float(self.defaults["lightvae_pruning_rate"]))])

        for flag_name, flag_value in [
            ("--visualize_ops", visualize_ops),
            ("--use_base_model", self.defaults.get("use_base_model", False) if use_base_model is None else use_base_model),
            ("--use_int8", self.defaults.get("use_int8", False) if use_int8 is None else use_int8),
            ("--verify_quant", self.defaults.get("verify_quant", False) if verify_quant is None else verify_quant),
            ("--use_async_vae", self.defaults.get("use_async_vae", False) if use_async_vae is None else use_async_vae),
            ("--compile_vae", self.defaults.get("compile_vae", False) if compile_vae is None else compile_vae),
            ("--t5_fsdp", self.defaults.get("t5_fsdp", False) if t5_fsdp is None else t5_fsdp),
            ("--t5_cpu", self.defaults.get("t5_cpu", False) if t5_cpu is None else t5_cpu),
            ("--dit_fsdp", self.defaults.get("dit_fsdp", False) if dit_fsdp is None else dit_fsdp),
            ("--convert_model_dtype", self.defaults.get("convert_model_dtype", False) if convert_model_dtype is None else convert_model_dtype),
        ]:
            if flag_value:
                command.append(flag_name)

        async_vae_warmup_iters = (
            self.defaults.get("async_vae_warmup_iters", 0)
            if async_vae_warmup_iters is None
            else async_vae_warmup_iters
        )
        if int(async_vae_warmup_iters) > 0:
            command.extend(["--async_vae_warmup_iters", str(int(async_vae_warmup_iters))])

        stdout = None if show_progress else subprocess.DEVNULL
        stderr = None if show_progress else subprocess.STDOUT
        env = build_subprocess_env(self.runtime_root, self.device)
        subprocess.run(
            command,
            check=True,
            cwd=project_root(),
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        video_path = output_dir_path / f"{save_name}.mp4"
        if _distributed_rank_from_env() == 0:
            _wait_for_video_readable(video_path)
            video = load_video_frames(str(video_path))
        else:
            # Only rank 0 owns a VAE and writes the artifact.  Distributed
            # Studio workers intentionally use rank-local temporary directories,
            # so requiring this rank's path to exist races (or always fails)
            # after an otherwise successful sequence-parallel generation.
            video = None
        return {
            "video": video,
            "generated_video_path": str(video_path),
            "output_dir": str(output_dir_path),
            "fps": int(fps),
            "num_iterations": int(num_iterations),
            "size": size_value,
        }

    def ensure_resident_pipeline(self):
        """Load the native model once for in-memory interactive rollout."""

        if self._resident_pipeline is not None:
            return self._resident_pipeline
        self._ensure_checkpoint_layout(self.checkpoint_dir)
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if world_size > 1 and not distributed:
            raise RuntimeError(
                "Matrix-Game 3 multi-GPU realtime requires an initialized torch.distributed "
                "process group on every rank."
            )
        if distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        if world_size not in {1, 2, 4, 8}:
            raise ValueError(
                "Matrix-Game 3 realtime supports 1, 2, 4, or 8 sequence-parallel "
                f"processes; got world_size={world_size}."
            )

        runtime_path = str(Path(self.runtime_root).expanduser().resolve())
        if runtime_path in sys.path:
            sys.path.remove(runtime_path)
        sys.path.insert(0, runtime_path)
        from .worldfoundry_runner import patch_torch_dynamo_config_aliases

        patch_torch_dynamo_config_aliases()
        from worldfoundry.base_models.diffusion_model.recipes.wan_configs.action_22 import (
            WAN_CONFIGS,
        )
        import pipeline.inference_interactive_pipeline as native_runtime

        device = torch.device(self.device)
        device_id = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else int(device.index or os.environ.get("LOCAL_RANK", "0") or "0")
        )
        defaults = self.defaults
        requested_parallelism = int(defaults.get("ulysses_size", world_size) or world_size)
        if requested_parallelism != world_size:
            raise ValueError(
                "Matrix-Game 3 realtime ulysses_size must equal the torchrun world size; "
                f"got ulysses_size={requested_parallelism}, world_size={world_size}."
            )
        runtime_args = SimpleNamespace(
            ckpt_dir=self.checkpoint_dir,
            size=self._normalize_size(defaults.get("size", "704*1280")),
            save_name=str(defaults.get("save_name", "matrix_game_3")),
            seed=int(defaults.get("seed", 42)),
            num_iterations=1,
            output_dir=None,
            sample_shift=defaults.get("sample_shift"),
            sample_guide_scale=float(defaults.get("sample_guide_scale", 5.0)),
            num_inference_steps=int(defaults.get("num_inference_steps", 3)),
            ulysses_size=world_size,
            t5_fsdp=False,
            t5_cpu=bool(defaults.get("t5_cpu", False)),
            dit_fsdp=False,
            convert_model_dtype=bool(defaults.get("convert_model_dtype", False)),
            use_base_model=bool(defaults.get("use_base_model", False)),
            use_int8=bool(defaults.get("use_int8", True)),
            verify_quant=bool(defaults.get("verify_quant", False)),
            vae_type=str(defaults.get("vae_type", "mg_lightvae")),
            lightvae_pruning_rate=defaults.get("lightvae_pruning_rate", 0.5),
            use_async_vae=False,
            async_vae_warmup_iters=0,
            compile_vae=bool(defaults.get("compile_vae", True)),
            fa_version=defaults.get("fa_version", "3"),
        )
        self._resident_pipeline = native_runtime.MatrixGame3Pipeline(
            config=WAN_CONFIGS["matrix_game3"],
            checkpoint_dir=self.checkpoint_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=world_size > 1,
            t5_cpu=runtime_args.t5_cpu,
            convert_model_dtype=runtime_args.convert_model_dtype,
            args=runtime_args,
            fa_version=runtime_args.fa_version,
            use_base_model=runtime_args.use_base_model,
        )
        return self._resident_pipeline

    def _materialize_image(self, image, output_dir: Path) -> Path:
        if isinstance(image, str):
            candidate = Path(image).expanduser()
            if candidate.exists():
                return candidate.resolve()
        image_path = output_dir / "input.png"
        load_pil_image(image).save(image_path)
        return image_path

    @staticmethod
    def _tensor_to_list(value) -> list:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return list(value)

    @staticmethod
    def _normalize_size(size: str | Sequence[int]) -> str:
        if isinstance(size, str):
            return size
        if len(size) != 2:
            raise ValueError(f"Expected size with 2 elements, got {size}")
        return f"{int(size[0])}*{int(size[1])}"

    @staticmethod
    def _ensure_checkpoint_layout(checkpoint_dir: str):
        checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Matrix-Game-3 checkpoint_dir not found: {checkpoint_dir}")

        required = [
            checkpoint_path / "models_t5_umt5-xxl-enc-bf16.pth",
            checkpoint_path / "google" / "umt5-xxl",
            checkpoint_path / "Wan2.2_VAE.pth",
        ]
        optional_groups = [
            [checkpoint_path / "base_distilled_model", checkpoint_path / "base_model"],
        ]
        for path_value in required:
            if not path_value.exists():
                raise FileNotFoundError(f"Matrix-Game-3 required checkpoint asset not found: {path_value}")
        for group in optional_groups:
            if not any(path.exists() for path in group):
                joined = ", ".join(str(path) for path in group)
                raise FileNotFoundError(f"Matrix-Game-3 expected one of the following model directories: {joined}")


__all__ = [
    "DEFAULT_MATRIX_GAME3_ALIASES",
    "DEFAULT_MATRIX_GAME3_ASSET_ROOT",
    "DEFAULT_MATRIX_GAME3_CHECKPOINT_DIR",
    "DEFAULT_MATRIX_GAME3_MOUSE_ICON",
    "MatrixGame3Runtime",
    "build_subprocess_env",
    "load_pil_image",
    "load_video_frames",
    "project_root",
    "resolve_checkpoint_dir",
    "runtime_root",
]

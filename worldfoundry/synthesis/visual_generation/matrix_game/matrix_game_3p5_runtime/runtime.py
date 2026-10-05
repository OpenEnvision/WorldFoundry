"""WorldFoundry-native adapter for Matrix-Game 3.5 inference."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.filesystem.file_utils import file_sha256
from worldfoundry.core.io.paths import (
    resolve_local_checkpoint_file,
    resolve_local_hf_model_path,
)

from .config_paths import MATRIX_GAME_35_CONFIG_ROOT, matrix_game_35_infer_config_path
from .specs import MatrixGame35ModelSpec, get_matrix_game_35_model_spec

SOURCE_REPOSITORY = "https://github.com/Riemann-Dynamics/Matrix-Game-3.5"
SOURCE_REVISION = "6c94dd787659aa19fac22581cd8bea54e65d813f"
MATRIX_GAME_35_CHECKPOINT_REPO = "RiemannDynamics/Matrix-Game-3.5-Base"
WAN_22_TI2V_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DA3_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT.parents[4]
INFERENCE_ENTRYPOINT = RUNTIME_ROOT / "entrypoint.py"
NATIVE_DIFFUSION_ROOT = PROJECT_ROOT / "worldfoundry/base_models/diffusion_model"
SHARED_DA3_ROOT = PROJECT_ROOT / "worldfoundry/base_models/three_dimensions/depth/depth_anything/depth_anything_v3"

WAN_REQUIRED_FILES = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
)
TOKENIZER_REQUIRED_FILES = ("tokenizer_config.json",)
DA3_REQUIRED_FILES = ("config.json", "model.safetensors")
REQUIRED_RUNTIME_FILES = (
    "entrypoint.py",
    "run_inference.py",
    "config_paths.py",
    "data/build_da3_video_index.py",
    "mosaic/main.py",
    "frustum/frustum_handler.py",
)
REQUIRED_DATA_FILES = tuple(
    matrix_game_35_infer_config_path(profile) for profile in ("common", "first_person", "third_person")
)
REQUIRED_SHARED_FILES = (
    NATIVE_DIFFUSION_ROOT / "pipeline.py",
    NATIVE_DIFFUSION_ROOT / "recipes/matrix_game.py",
    NATIVE_DIFFUSION_ROOT / "models/autoencoders/wan/model.py",
    NATIVE_DIFFUSION_ROOT / "models/encoders/wan/model.py",
    NATIVE_DIFFUSION_ROOT / "models/networks/matrix_game_3p5/model.py",
    NATIVE_DIFFUSION_ROOT / "models/denoisers/matrix_game_3p5.py",
    NATIVE_DIFFUSION_ROOT / "models/networks/wan/model.py",
    SHARED_DA3_ROOT / "api.py",
)

_RESERVED_PIPELINE_ARGS = frozenset(
    {
        "--config",
        "--dataset_cache_dir",
        "--dataset_index_path",
        "--log_dir_name",
        "--model_id_with_origin_paths",
        "--memory_latent_cache_dir",
        "--num_inference_batches",
        "--num_inference_blocks",
        "--num_inference_steps",
        "--output_path",
        "--tokenizer_path",
        "--trained_dit",
        "--guidance_scale",
        "--inference_sample_offset",
        "--inference_seed",
    }
)
_SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "DA3_MODEL_PATH",
    "HF_HUB_OFFLINE",
    "PYTHONPATH",
    "TRANSFORMERS_OFFLINE",
    "WORLDFOUNDRY_MATRIX_GAME_3P5_CACHE_DIR",
    "WORLDFOUNDRY_MATRIX_GAME_3P5_PYTHON",
    "WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD",
)


@dataclass(frozen=True)
class MatrixGame35Assets:
    """Resolved local assets consumed by one inference run."""

    checkpoint: Path
    wan_dir: Path
    tokenizer_dir: Path
    da3_dir: Path
    python_executable: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "checkpoint": str(self.checkpoint),
            "wan_dir": str(self.wan_dir),
            "tokenizer_dir": str(self.tokenizer_dir),
            "da3_dir": str(self.da3_dir),
            "python_executable": str(self.python_executable),
        }


@dataclass(frozen=True)
class MatrixGame35RuntimePlan:
    """Fully resolved, shell-free inference command."""

    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: Path
    output_dir: Path
    result_path: Path
    workspace_path: Path
    assets: MatrixGame35Assets

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "environment": {key: self.env[key] for key in _SAFE_ENV_KEYS if key in self.env},
            "workdir": str(self.workdir),
            "output_dir": str(self.output_dir),
            "result_path": str(self.result_path),
            "workspace_path": str(self.workspace_path),
            "assets": self.assets.to_dict(),
        }


def inspect_camera_npz(camera_path: str | os.PathLike[str], *, num_blocks: int = 1) -> dict[str, Any]:
    """Validate the Matrix-Game camera schema without importing model weights."""

    import numpy as np

    path = Path(camera_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Matrix-Game 3.5 camera trajectory not found: {path}")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be positive")

    with np.load(path, allow_pickle=False) as payload:
        extrinsics_key = "extrinsics_c2w" if "extrinsics_c2w" in payload else "extrinsics"
        if extrinsics_key not in payload:
            raise ValueError(f"{path} must contain 'extrinsics_c2w' (preferred) or 'extrinsics'")
        if "intrinsics" not in payload:
            raise ValueError(f"{path} must contain 'intrinsics'")
        extrinsics = np.asarray(payload[extrinsics_key])
        intrinsics = np.asarray(payload["intrinsics"])

    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4) or extrinsics.shape[0] == 0:
        raise ValueError(f"extrinsics must have shape (N,4,4) with N>0; got {extrinsics.shape}")
    valid_intrinsics = (
        (intrinsics.ndim == 1 and intrinsics.shape == (4,))
        or (intrinsics.ndim == 2 and intrinsics.shape == (3, 3))
        or (intrinsics.ndim == 2 and intrinsics.shape[1:] == (4,) and intrinsics.shape[0] > 0)
        or (intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3) and intrinsics.shape[0] > 0)
    )
    if not valid_intrinsics:
        raise ValueError(f"intrinsics must have shape (4,), (N,4), (3,3), or (N,3,3); got {intrinsics.shape}")
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f"camera trajectory contains NaN or infinite values: {path}")

    required_poses = max(86, 1 + 84 * int(num_blocks))
    return {
        "path": str(path),
        "extrinsics_key": extrinsics_key,
        "extrinsics_shape": list(extrinsics.shape),
        "intrinsics_shape": list(intrinsics.shape),
        "pose_count": int(extrinsics.shape[0]),
        "minimum_pose_count": required_poses,
        "tail_will_be_held": bool(extrinsics.shape[0] < required_poses),
    }


def _resolve_python(value: str | os.PathLike[str] | None) -> Path:
    raw = str(value or os.environ.get("WORLDFOUNDRY_MATRIX_GAME_3P5_PYTHON") or sys.executable)
    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        candidate = candidate / "bin" / "python"
    if candidate.is_file():
        # Keep conda/uv wrapper scripts intact. Resolving their symlink or
        # launcher target can lose the environment's site-packages in child
        # processes.
        return Path(os.path.abspath(candidate))
    resolved = shutil.which(raw)
    if resolved:
        return Path(os.path.abspath(resolved))
    raise FileNotFoundError(f"Matrix-Game 3.5 Python executable not found: {raw}")


def _device_visibility(device: str | None) -> str | None:
    value = str(device or "").strip()
    if not value or value == "cuda":
        return None
    if value.startswith("cuda:"):
        return value.split(":", 1)[1] or None
    if value.isdigit() or re.fullmatch(r"\d+(,\d+)*", value):
        return value
    return None


def _validate_extra_args(extra_args: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(str(item) for item in (extra_args or ()))
    for value in values:
        option = value.split("=", 1)[0]
        if option in _RESERVED_PIPELINE_ARGS:
            raise ValueError(f"extra_args cannot override model identity or required inference wiring: {option}")
    return values


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _log_tail(path: Path, *, lines: int = 60) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _artifact_uri(value: Any) -> Any:
    """Unwrap WorldFoundry ArtifactRef-like local inputs without importing eval contracts."""

    uri = getattr(value, "uri", None)
    return uri if isinstance(uri, str) and uri else value


class MatrixGame35Runtime:
    """Run one immutable Matrix-Game 3.5 model through the native runtime."""

    SOURCE_REPOSITORY = SOURCE_REPOSITORY
    SOURCE_REVISION = SOURCE_REVISION
    CHECKPOINT_REPOSITORY = MATRIX_GAME_35_CHECKPOINT_REPO

    def __init__(
        self,
        *,
        model_id: str,
        checkpoint_path: str | os.PathLike[str] | None = None,
        wan_dir: str | os.PathLike[str] | None = None,
        tokenizer_dir: str | os.PathLike[str] | None = None,
        da3_dir: str | os.PathLike[str] | None = None,
        python_executable: str | os.PathLike[str] | None = None,
        device: str = "cuda",
        num_blocks: int = 1,
        steps: int = 25,
        cfg_scale: float = 5.0,
        seed: int = 3407,
        camera_convention: str = "c2w",
        keep_workspace: bool = False,
        timeout_seconds: float | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> None:
        self.spec: MatrixGame35ModelSpec = get_matrix_game_35_model_spec(model_id)
        self.model_id = self.spec.model_id
        self.checkpoint_ref = checkpoint_path
        self.wan_ref = wan_dir
        self.tokenizer_ref = tokenizer_dir
        self.da3_ref = da3_dir
        self.python_ref = python_executable
        self.device = str(device)
        self.num_blocks = int(num_blocks)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.seed = int(seed)
        self.camera_convention = str(camera_convention)
        self.keep_workspace = bool(keep_workspace)
        self.timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self.extra_args = _validate_extra_args(extra_args)
        self._validate_generation_options(
            num_blocks=self.num_blocks,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            camera_convention=self.camera_convention,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "MatrixGame35Runtime":
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = pretrained_model_path
        options.update(kwargs)
        return cls(**options)

    @staticmethod
    def _validate_generation_options(
        *,
        num_blocks: int,
        steps: int,
        cfg_scale: float,
        camera_convention: str,
    ) -> None:
        if int(num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if int(steps) <= 0:
            raise ValueError("steps must be positive")
        if float(cfg_scale) <= 0:
            raise ValueError("cfg_scale must be positive")
        if camera_convention not in {"c2w", "w2c"}:
            raise ValueError("camera_convention must be 'c2w' or 'w2c'")

    def _resolve_checkpoint(self) -> Path:
        reference = (
            self.checkpoint_ref
            or os.environ.get("CKPT_FIRST_PERSON" if self.spec.person == "first" else "CKPT_THIRD_PERSON")
            or MATRIX_GAME_35_CHECKPOINT_REPO
        )
        return resolve_local_checkpoint_file(reference, self.spec.checkpoint_filename)

    def _resolve_wan(self) -> Path:
        reference = self.wan_ref or os.environ.get("WAN22_TI2V_5B_DIR") or WAN_22_TI2V_REPO
        return resolve_local_hf_model_path(reference, required_files=WAN_REQUIRED_FILES)

    def _resolve_tokenizer(self, wan_dir: Path) -> Path:
        reference = self.tokenizer_ref or os.environ.get("UMT5_TOKENIZER_DIR")
        if reference is None:
            tokenizer = wan_dir / "google" / "umt5-xxl"
        else:
            direct = Path(reference).expanduser()
            if direct.is_dir():
                tokenizer = direct.resolve()
            else:
                tokenizer = resolve_local_hf_model_path(
                    reference,
                    required_files=TOKENIZER_REQUIRED_FILES,
                )
        missing = [name for name in TOKENIZER_REQUIRED_FILES if not (tokenizer / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Matrix-Game 3.5 tokenizer is incomplete at {tokenizer}: missing {missing}")
        return tokenizer

    def _resolve_da3(self) -> Path:
        reference = self.da3_ref or os.environ.get("DA3_MODEL_PATH") or DA3_REPO
        return resolve_local_hf_model_path(reference, required_files=DA3_REQUIRED_FILES)

    def resolve_assets(self) -> MatrixGame35Assets:
        wan_dir = self._resolve_wan()
        return MatrixGame35Assets(
            checkpoint=self._resolve_checkpoint(),
            wan_dir=wan_dir,
            tokenizer_dir=self._resolve_tokenizer(wan_dir),
            da3_dir=self._resolve_da3(),
            python_executable=_resolve_python(self.python_ref),
        )

    def preflight(self) -> dict[str, Any]:
        missing_runtime = [
            str(RUNTIME_ROOT / relative)
            for relative in REQUIRED_RUNTIME_FILES
            if not (RUNTIME_ROOT / relative).is_file()
        ]
        missing_runtime.extend(
            str(path) for path in (*REQUIRED_DATA_FILES, *REQUIRED_SHARED_FILES) if not path.is_file()
        )
        assets: dict[str, str] = {}
        asset_errors: dict[str, str] = {}

        resolvers = {
            "checkpoint": self._resolve_checkpoint,
            "wan_dir": self._resolve_wan,
            "da3_dir": self._resolve_da3,
            "python_executable": lambda: _resolve_python(self.python_ref),
        }
        for name, resolver in resolvers.items():
            try:
                assets[name] = str(resolver())
            except (FileNotFoundError, OSError, ValueError) as exc:
                asset_errors[name] = str(exc)
        if "wan_dir" in assets:
            try:
                assets["tokenizer_dir"] = str(self._resolve_tokenizer(Path(assets["wan_dir"])))
            except (FileNotFoundError, OSError, ValueError) as exc:
                asset_errors["tokenizer_dir"] = str(exc)

        code_ready = not missing_runtime
        assets_ready = not asset_errors
        device_ready = self.device.startswith("cuda") or bool(_device_visibility(self.device))
        return {
            "status": "ready" if code_ready and assets_ready and device_ready else "blocked",
            "model_id": self.model_id,
            "person": self.spec.person,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "runtime_root": str(RUNTIME_ROOT),
            "config_root": str(MATRIX_GAME_35_CONFIG_ROOT),
            "project_root": str(PROJECT_ROOT),
            "native_diffusion_root": str(NATIVE_DIFFUSION_ROOT),
            "shared_da3_root": str(SHARED_DA3_ROOT),
            "code_ready": code_ready,
            "assets_ready": assets_ready,
            "device_ready": device_ready,
            "assets": assets,
            "asset_errors": asset_errors,
            "missing_runtime_files": missing_runtime,
        }

    @staticmethod
    def _materialize_image(image: Any, input_dir: Path) -> Path:
        image = _artifact_uri(image)
        if isinstance(image, (str, os.PathLike)):
            path = Path(image).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Matrix-Game 3.5 anchor image not found: {path}")
            return path
        from worldfoundry.core import materialize_image_input

        return Path(materialize_image_input(image, input_dir, filename="anchor.png"))

    @staticmethod
    def _materialize_camera(camera: Any, input_dir: Path) -> Path:
        camera = _artifact_uri(camera)
        if isinstance(camera, (str, os.PathLike)):
            return Path(camera).expanduser().resolve()
        if not isinstance(camera, Mapping):
            raise TypeError("camera must be an NPZ path or a mapping with extrinsics_c2w/extrinsics and intrinsics")
        import numpy as np

        extrinsics_key = "extrinsics_c2w" if "extrinsics_c2w" in camera else "extrinsics"
        if extrinsics_key not in camera or "intrinsics" not in camera:
            raise ValueError("camera mapping requires extrinsics_c2w (or extrinsics) and intrinsics")
        path = input_dir / "camera.npz"
        np.savez_compressed(
            path,
            **{
                extrinsics_key: np.asarray(camera[extrinsics_key]),
                "intrinsics": np.asarray(camera["intrinsics"]),
            },
        )
        return path

    @staticmethod
    def _materialize_refs(refs: Any, input_dir: Path) -> Path | None:
        refs = _artifact_uri(refs)
        if refs is None or (isinstance(refs, str) and refs == ""):
            return None
        if isinstance(refs, (str, os.PathLike)):
            path = Path(refs).expanduser().resolve()
            if path.is_dir():
                return path
            if not path.is_file():
                raise FileNotFoundError(f"Matrix-Game 3.5 subject reference not found: {path}")
            values: Sequence[Any] = (path,)
        elif isinstance(refs, Sequence) and not isinstance(refs, (bytes, bytearray)):
            values = refs
        else:
            values = (refs,)
        if not values:
            return None
        if len(values) > 2:
            raise ValueError("Matrix-Game 3.5 third-person accepts at most two subject crops")

        from worldfoundry.core import materialize_image_input

        refs_dir = input_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(values):
            value = _artifact_uri(value)
            if isinstance(value, (str, os.PathLike)) and Path(value).expanduser().is_file():
                source = Path(value).expanduser().resolve()
                suffix = source.suffix.lower() if source.suffix else ".png"
                shutil.copy2(source, refs_dir / f"ref_{index:02d}{suffix}")
            else:
                materialize_image_input(value, refs_dir, filename=f"ref_{index:02d}.png")
        return refs_dir

    def build_plan(
        self,
        *,
        image_path: str | os.PathLike[str],
        camera_path: str | os.PathLike[str],
        prompt: str,
        caption_path: str | os.PathLike[str] | None,
        refs_dir: str | os.PathLike[str] | None,
        output_root: str | os.PathLike[str],
        cache_root: str | os.PathLike[str],
        run_name: str,
        num_blocks: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        camera_convention: str,
        keep_workspace: bool,
        extra_args: Sequence[str] | None = None,
    ) -> MatrixGame35RuntimePlan:
        self._validate_generation_options(
            num_blocks=num_blocks,
            steps=steps,
            cfg_scale=cfg_scale,
            camera_convention=camera_convention,
        )
        assets = self.resolve_assets()
        output_root_path = Path(output_root).expanduser().resolve()
        cache_root_path = Path(cache_root).expanduser().resolve()
        inference_output_dir = output_root_path / self.spec.output_namespace / run_name
        workspace_path = cache_root_path / "infer_runs" / f"{self.spec.output_namespace}_{run_name}"

        command = [
            str(assets.python_executable),
            str(INFERENCE_ENTRYPOINT),
            "--person",
            self.spec.person,
            "--image",
            str(Path(image_path).expanduser().resolve()),
            "--camera",
            str(Path(camera_path).expanduser().resolve()),
            "--prompt",
            str(prompt or ""),
            "--num-blocks",
            str(int(num_blocks)),
            "--steps",
            str(int(steps)),
            "--cfg-scale",
            str(float(cfg_scale)),
            "--seed",
            str(int(seed)),
            "--output",
            str(output_root_path),
            "--name",
            run_name,
            "--camera-convention",
            camera_convention,
            "--ckpt",
            str(assets.checkpoint),
            "--wan-dir",
            str(assets.wan_dir),
            "--tokenizer-dir",
            str(assets.tokenizer_dir),
            "--da3-dir",
            str(assets.da3_dir),
        ]
        if caption_path is not None:
            command.extend(["--caption", str(Path(caption_path).expanduser().resolve())])
        if refs_dir is not None:
            command.extend(["--refs", str(Path(refs_dir).expanduser().resolve())])
        if keep_workspace:
            command.append("--keep-workspace")
        command.extend(_validate_extra_args(extra_args))

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(entry for entry in (str(PROJECT_ROOT), env.get("PYTHONPATH", "")) if entry)
        env["DA3_MODEL_PATH"] = str(assets.da3_dir)
        env["WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD"] = "true"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["WORLDFOUNDRY_MATRIX_GAME_3P5_CACHE_DIR"] = str(cache_root_path)
        env["WORLDFOUNDRY_MATRIX_GAME_3P5_PYTHON"] = str(assets.python_executable)
        visible_devices = _device_visibility(self.device)
        if visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices

        return MatrixGame35RuntimePlan(
            command=tuple(command),
            env=env,
            workdir=RUNTIME_ROOT,
            output_dir=inference_output_dir,
            result_path=inference_output_dir / "result.mp4",
            workspace_path=workspace_path,
            assets=assets,
        )

    def predict(
        self,
        *,
        prompt: str = "",
        images: Any = None,
        image: Any = None,
        camera_path: Any = None,
        camera: Any = None,
        caption_path: str | os.PathLike[str] | None = None,
        refs: Any = None,
        output_path: str | os.PathLike[str] | None = None,
        fps: int | None = None,
        num_blocks: int | None = None,
        blocks: int | None = None,
        steps: int | None = None,
        num_inference_steps: int | None = None,
        cfg_scale: float | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        base_seed: int | None = None,
        camera_convention: str | None = None,
        keep_workspace: bool | None = None,
        timeout_seconds: float | None = None,
        extra_args: Sequence[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not self.device.startswith("cuda") and _device_visibility(self.device) is None:
            raise ValueError("Matrix-Game 3.5 inference requires a CUDA device")
        if fps not in (None, 16):
            raise ValueError("Matrix-Game 3.5 inference writes at 16 FPS; fps cannot be overridden")
        anchor = images if images is not None else image
        if anchor is None:
            raise ValueError(f"{self.model_id} requires an anchor image")
        camera_input = camera_path if camera_path is not None else camera
        if camera_input is None:
            raise ValueError(f"{self.model_id} requires camera_path or a camera mapping")
        if not prompt and caption_path is None:
            raise ValueError(f"{self.model_id} requires prompt or caption_path")
        if refs is not None and not (isinstance(refs, str) and refs == "") and not self.spec.supports_subject_refs:
            raise ValueError("Subject references are only supported by matrix-game-3.5-third-person")

        block_count = num_blocks if num_blocks is not None else blocks
        block_count = self.num_blocks if block_count is None else int(block_count)
        requested_steps = steps if steps is not None else num_inference_steps
        inference_steps = self.steps if requested_steps is None else int(requested_steps)
        requested_guidance = cfg_scale if cfg_scale is not None else guidance_scale
        guidance = self.cfg_scale if requested_guidance is None else float(requested_guidance)
        requested_seed = seed if seed is not None else base_seed
        generation_seed = self.seed if requested_seed is None else int(requested_seed)
        convention = self.camera_convention if camera_convention is None else str(camera_convention)
        preserve_workspace = self.keep_workspace if keep_workspace is None else bool(keep_workspace)
        timeout = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        pipeline_args = self.extra_args if extra_args is None else _validate_extra_args(extra_args)
        self._validate_generation_options(
            num_blocks=block_count,
            steps=inference_steps,
            cfg_scale=guidance,
            camera_convention=convention,
        )

        if output_path is None:
            persistent_output = Path(tempfile.mkdtemp(prefix=f"{self.model_id}_")) / f"{self.model_id}.mp4"
        else:
            persistent_output = Path(output_path).expanduser().resolve()
            if persistent_output.suffix and persistent_output.suffix.lower() != ".mp4":
                raise ValueError("Matrix-Game 3.5 output_path must be an .mp4 path or a directory")
            if not persistent_output.suffix:
                persistent_output = persistent_output / f"{self.model_id}.mp4"
        persistent_output.parent.mkdir(parents=True, exist_ok=True)

        run_root = Path(tempfile.mkdtemp(prefix=f"worldfoundry_{self.model_id}_"))
        input_dir = run_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_root / "infer.log"
        run_name = f"wf_{uuid.uuid4().hex}"
        succeeded = False
        launched = False
        plan: MatrixGame35RuntimePlan | None = None
        try:
            image_path = self._materialize_image(anchor, input_dir)
            materialized_camera = self._materialize_camera(camera_input, input_dir)
            camera_info = inspect_camera_npz(materialized_camera, num_blocks=block_count)
            refs_dir = self._materialize_refs(refs, input_dir)
            resolved_caption = None
            if caption_path is not None:
                resolved_caption = Path(_artifact_uri(caption_path)).expanduser().resolve()
                if not resolved_caption.is_file():
                    raise FileNotFoundError(f"Matrix-Game 3.5 caption JSON not found: {resolved_caption}")

            plan = self.build_plan(
                image_path=image_path,
                camera_path=materialized_camera,
                prompt=str(prompt or ""),
                caption_path=resolved_caption,
                refs_dir=refs_dir,
                output_root=run_root / "outputs",
                cache_root=run_root / "cache",
                run_name=run_name,
                num_blocks=block_count,
                steps=inference_steps,
                cfg_scale=guidance,
                seed=generation_seed,
                camera_convention=convention,
                keep_workspace=preserve_workspace,
                extra_args=pipeline_args,
            )
            with log_path.open("w", encoding="utf-8") as log_handle:
                launched = True
                completed = subprocess.run(
                    plan.command,
                    cwd=plan.workdir,
                    env=dict(plan.env),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            if completed.returncode != 0:
                tail = _log_tail(log_path)
                raise RuntimeError(
                    f"Matrix-Game 3.5 inference failed with exit code {completed.returncode}. "
                    f"Preserved run directory: {run_root}\n{tail}"
                )
            if not plan.result_path.is_file() or plan.result_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"Matrix-Game 3.5 completed without a readable result.mp4; preserved run directory: {run_root}"
                )

            _atomic_copy(plan.result_path, persistent_output)
            succeeded = True
            return {
                "status": "succeeded",
                "artifact_kind": "generated_world",
                "artifact_path": str(persistent_output),
                "generated_video_path": str(persistent_output),
                "artifact_sha256": file_sha256(persistent_output),
                "model_id": self.model_id,
                "runtime": "matrix_game_3p5_native_infer",
                "backend_quality": (
                    "official_base_25_step" if inference_steps == 25 else "official_base_checkpoint_custom_step_count"
                ),
                "metadata": {
                    "person": self.spec.person,
                    "source_revision": SOURCE_REVISION,
                    "num_blocks": block_count,
                    "expected_frames": 1 + 84 * block_count,
                    "generated_frames_per_block": 84,
                    "uses_official_step_count": inference_steps == 25,
                    "fps": 16,
                    "steps": inference_steps,
                    "cfg_scale": guidance,
                    "seed": generation_seed,
                    "camera_convention": convention,
                    "camera": camera_info,
                    "workspace_path": str(plan.workspace_path) if preserve_workspace else None,
                },
            }
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Matrix-Game 3.5 inference timed out after {timeout} seconds. Preserved run directory: {run_root}"
            ) from exc
        finally:
            if not launched or (succeeded and not preserve_workspace):
                shutil.rmtree(run_root, ignore_errors=True)


__all__ = [
    "DA3_REPO",
    "MATRIX_GAME_35_CHECKPOINT_REPO",
    "INFERENCE_ENTRYPOINT",
    "PROJECT_ROOT",
    "SHARED_DA3_ROOT",
    "NATIVE_DIFFUSION_ROOT",
    "RUNTIME_ROOT",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "WAN_22_TI2V_REPO",
    "MatrixGame35Assets",
    "MatrixGame35Runtime",
    "MatrixGame35RuntimePlan",
    "inspect_camera_npz",
]

"""Independent in-tree runtime for MagicWorld."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_candidates, resolve_data_path
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_local_checkpoint_alias(value: Any) -> Path:
    """Resolve hfd/<repo> and flat <repo> checkpoint mirror layouts."""

    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    for root in checkpoint_root_candidates():
        for candidate in (root / path.name, root / "hfd" / path.name):
            if candidate.exists():
                return candidate.resolve()
    return path


def _resolve_depth_pro_checkpoint() -> Path:
    configured = os.environ.get("DEPTH_PRO_CHECKPOINT", "").strip()
    if configured:
        return require_path(configured, "DepthPro checkpoint", kind="file")
    for root in checkpoint_root_candidates():
        for relative in (
            Path("apple--DepthPro") / "depth_pro.pt",
            Path("DepthPro") / "depth_pro.pt",
        ):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        "MagicWorld requires a local DepthPro checkpoint at "
        "$WORLDFOUNDRY_CKPT_DIR/apple--DepthPro/depth_pro.pt"
    )


def _resolve_magic_checkpoint(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = (
        root / "MagicWorld-Fast" / "model.pt",
        root / "MagicWorld" / "MagicWorld-Fast" / "model.pt",
        root / "model.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"MagicWorld-Fast model.pt not found under {root}")


def _resolve_magic_base(root: Path) -> Path:
    candidates = (root / "MagicWorld-Base", root / "MagicWorld" / "MagicWorld-Base", root)
    for path in candidates:
        transformer = path / "transformer"
        if path.is_dir() and (
            transformer.is_dir()
            or ((path / "config.json").is_file() and (path / "diffusion_pytorch_model.safetensors").is_file())
        ):
            return path
    raise FileNotFoundError(
        f"MagicWorld-Base config.json and diffusion_pytorch_model.safetensors not found under {root}"
    )


def _coerce_native_rows(value: Any) -> list[list[float]]:
    if isinstance(value, (str, os.PathLike)):
        path = require_path(value, "MagicWorld camera rows", kind="file")
        raw_rows: list[Any] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.strip().split()
            if not fields:
                continue
            try:
                raw_rows.append([float(field) for field in fields])
            except ValueError:
                if raw_rows:
                    raise ValueError(f"MagicWorld camera row {line_number} is not numeric: {path}")
                continue
    elif isinstance(value, (list, tuple)):
        raw_rows = list(value)
    else:
        raise TypeError("MagicWorld native_rows must be a numeric row sequence or a camera txt path")

    rows: list[list[float]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"MagicWorld camera row {index} must be a sequence")
        converted = [float(item) for item in row]
        if len(converted) not in (19, 25):
            raise ValueError(
                f"MagicWorld camera row {index} has {len(converted)} values; expected 19 or 25"
            )
        rows.append(converted)
    if not rows:
        raise ValueError("MagicWorld camera rows are empty")
    return rows


def _magicworld_latent_frames(num_frames: int) -> int:
    """Validate the public frame count against MagicWorld-Fast's three-frame blocks."""

    requested = int(num_frames)
    if requested <= 0:
        raise ValueError("MagicWorld num_frames must be positive")
    latent_frames = max(1, (requested - 1) // 4 + 1)
    if (requested - 1) % 4 != 0 or latent_frames % 3 != 0:
        raise ValueError(
            f"MagicWorld num_frames={requested} maps to {latent_frames} latent frames; "
            "MagicWorld-Fast requires a multiple of 3 latent frames "
            "(valid output counts include 9, 21, 33, 45, 57, 69, and 81)."
        )
    return latent_frames


class MagicWorldRuntime:
    """Run the bundled MagicWorld-Fast camera-control path."""

    SOURCE_REVISION = "a378d67d1b803db4268340fcc130c98a243ad9a8"

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT", "cache/hfd"))
        self.checkpoint_path = _resolve_local_checkpoint_alias(
            checkpoint_path or hfd / "LuckyLiGY--MagicWorld"
        )
        self.base_model_path = _resolve_local_checkpoint_alias(
            base_model_path or hfd / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP"
        )
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "magic_world_runtime"

    @staticmethod
    def video_x_fun_root() -> Path:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return worldfoundry_root / "synthesis" / "visual_generation" / "video_x_fun" / "video_x_fun_runtime"

    @staticmethod
    def uni3c_root() -> Path:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return worldfoundry_root / "base_models" / "three_dimensions" / "general_3d" / "uni3c"

    @staticmethod
    def depth_pro_root() -> Path:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return worldfoundry_root / "base_models" / "three_dimensions" / "depth" / "depth_pro"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "MagicWorldRuntime":
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": "magicworld",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def predict(
        self,
        *,
        prompt: str,
        image_path: Any,
        native_rows: Any,
        output_path: Any,
        num_frames: int = 81,
        seed: int = 42,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        image = require_path(image_path, "MagicWorld first-frame image", kind="file")
        checkpoint_root = require_path(self.checkpoint_path, "MagicWorld checkpoint")
        checkpoint = _resolve_magic_checkpoint(checkpoint_root)
        magic_base = _resolve_magic_base(checkpoint_root if checkpoint_root.is_dir() else checkpoint_root.parent)
        wan_root = require_path(self.base_model_path, "Wan2.1-Fun-V1.1-1.3B-InP", kind="dir")
        camera_rows = _coerce_native_rows(native_rows)
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_magicworld_inputs"
        image_dir = input_dir / "images"
        native_output = input_dir / "outputs"
        image_dir.mkdir(parents=True, exist_ok=True)
        staged_image = image_dir / "input.png"
        shutil.copy2(image, staged_image)
        prompts = input_dir / "prompts.json"
        prompts.write_text(
            json.dumps([{"name": staged_image.name, "describe": prompt}], indent=2),
            encoding="utf-8",
        )
        camera = input_dir / "camera.txt"
        camera.write_text(
            "frame fx fy cx cy unused0 unused1 w2c_00 w2c_01 w2c_02 w2c_03 "
            "w2c_10 w2c_11 w2c_12 w2c_13 w2c_20 w2c_21 w2c_22 w2c_23\n"
            + "\n".join(" ".join(f"{value:.8f}" for value in row) for row in camera_rows)
            + "\n",
            encoding="utf-8",
        )
        latent_frames = _magicworld_latent_frames(num_frames)
        required_camera_rows = (latent_frames - 1) * 4 + 1
        if len(camera_rows) < required_camera_rows:
            raise ValueError(
                f"MagicWorld camera rows are too short: got {len(camera_rows)}, "
                f"need at least {required_camera_rows} for {num_frames} output frames"
            )
        env = {
            **self.env,
            "DEPTH_PRO_CHECKPOINT": str(_resolve_depth_pro_checkpoint()),
            "WORLDFOUNDRY_MAGICWORLD_WAN_ROOT": str(wan_root),
            "WORLDFOUNDRY_MAGICWORLD_BASE_ROOT": str(magic_base),
            "WORLDFOUNDRY_MAGICWORLD_CONFIG": str(
                resolve_data_path(
                    "models", "runtime", "configs", "video_x_fun", "wan2.1", "wan_civitai.yaml"
                )
            ),
            "WORLDFOUNDRY_MAGICWORLD_DEFAULT_CONFIG": str(
                resolve_data_path(
                    "models", "runtime", "configs", "magic_world", "default_config.yaml"
                )
            ),
        }
        command = [
            self.python_executable,
            "inference/inference_magicworld_fast.py",
            "--config_path",
            resolve_data_path(
                "models", "runtime", "configs", "magic_world", "reward_forcing_switch.yaml"
            ),
            "--checkpoint_path",
            checkpoint,
            "--data_path",
            image_dir,
            "--extended_prompt_path",
            prompts,
            "--output_folder",
            native_output,
            "--control_camera_txt",
            camera,
            "--num_output_frames",
            str(latent_frames),
            "--seed",
            str(seed),
            "--i2v",
        ]
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            search_roots=(native_output,),
            env=env,
            python_paths=(
                _PROJECT_ROOT,
                self.repo_root,
                self.video_x_fun_root(),
                self.uni3c_root().parent,
                self.depth_pro_root().parent,
            ),
        )
        return result if return_dict else result.get("video")


__all__ = ["MagicWorldRuntime"]

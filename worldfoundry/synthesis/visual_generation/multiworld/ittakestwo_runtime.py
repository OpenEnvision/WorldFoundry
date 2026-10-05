"""WorldFoundry-native single-sample inference for MultiWorld ItTakesTwo."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.core import load_pil_image, load_video_frames, save_video_frames
from .native_pipeline import load_multiworld_config, load_multiworld_pipeline
from .runtime_env import (
    resolve_checkpoint_path,
    resolve_config_path,
    resolve_runtime_root,
    resolve_wan_ti2v_root,
)


def save_image_input(image_input: Any, output_path: str | Path) -> str:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_pil_image(image_input).save(output_path)
    return str(output_path)


def _numeric_array_from_sequence(value: Sequence[Any]) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0 or array.dtype == object:
        return None
    return array if array.dtype.kind in {"b", "i", "u", "f", "c"} else None


def _to_numpy_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_numpy_tree(child) for key, child in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        array = _numeric_array_from_sequence(value)
        return array if array is not None else [_to_numpy_tree(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
        return value
    return np.asarray(value)


def dump_tree(value: Any, output_path: str | Path) -> str:
    """Persist an action/environment tree for the standalone inference CLI."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(_to_numpy_tree(value), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return str(output_path)


def load_tree(input_path: str | Path) -> Any:
    with Path(input_path).expanduser().resolve().open("rb") as handle:
        return pickle.load(handle)


def load_ittakestwo_action_csv(
    action_path: str | Path,
    *,
    num_frames: int = 81,
    stick_threshold: float = 0.3,
) -> dict[str, Any]:
    """Load the official CSV schema into MultiWorld's two-player action tensors."""

    resolved = Path(action_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MultiWorld ItTakesTwo action CSV not found: {resolved}")

    def as_int(row: Mapping[str, str], key: str) -> int:
        try:
            return int(float(row.get(key) or 0))
        except ValueError:
            return 0

    def as_float(row: Mapping[str, str], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except ValueError:
            return 0.0

    discrete: list[list[list[int]]] = []
    continuous: list[list[list[float]]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left_discrete = [
                as_int(row, key)
                for key in ("w", "a", "s", "d", "space", "shift", "ctrl", "e", "q", "f")
            ]
            left_continuous = [as_float(row, "norm_dx"), as_float(row, "norm_dy")]
            right_discrete = [0] * 10
            right_continuous = [as_float(row, "axis_2"), as_float(row, "axis_3")]
            button_mapping = {0: 4, 1: 6, 3: 9, 4: 8, 5: 7}
            for button, action_index in button_mapping.items():
                if as_int(row, f"button_{button}"):
                    right_discrete[action_index] = 1
            if as_int(row, "button_2") or as_int(row, "button_8"):
                right_discrete[5] = 1
            axis_0, axis_1 = as_float(row, "axis_0"), as_float(row, "axis_1")
            if axis_0 < -stick_threshold:
                right_discrete[1] = 1
            elif axis_0 > stick_threshold:
                right_discrete[3] = 1
            if axis_1 < -stick_threshold:
                right_discrete[0] = 1
            elif axis_1 > stick_threshold:
                right_discrete[2] = 1
            discrete.append([left_discrete, right_discrete])
            continuous.append([left_continuous, right_continuous])
            if len(discrete) >= num_frames:
                break
    if len(discrete) < num_frames:
        raise ValueError(
            f"MultiWorld action CSV contains {len(discrete)} frames, but {num_frames} are required"
        )
    return {"discrete_action": [discrete], "continuous_action": [continuous]}


def _derive_env_obv_from_image(input_image_path: str | Path) -> torch.Tensor:
    from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.environment_observation import load_and_preprocess_images

    left = load_and_preprocess_images([input_image_path], mode="pad", return_view="left")
    right = load_and_preprocess_images([input_image_path], mode="pad", return_view="right")
    return torch.cat([left[None, None, ...], right[None, None, ...]], dim=2)


def _resolve_num_frames(action: Mapping[str, Any], fallback: int) -> int:
    for key in ("discrete_action", "continuous_action"):
        value = action.get(key)
        shape = getattr(value, "shape", None)
        if shape is None:
            try:
                shape = np.asarray(value).shape
            except (TypeError, ValueError):
                continue
        if len(shape) >= 2:
            return int(shape[1])
    return int(fallback)


class MultiWorldItTakesTwoRuntime:
    """Lazy native runtime that keeps one shared-role pipeline per process."""

    def __init__(
        self,
        runtime_root: str | None,
        config_path: str,
        checkpoint_path: str,
        *,
        base_model_root: str | None = None,
        vggt_root: str | None = None,
        python_executable: str | None = None,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime_root = resolve_runtime_root(runtime_root)
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.base_model_root = resolve_wan_ti2v_root(base_model_root)
        self.vggt_root = vggt_root
        self.python_executable = python_executable or sys.executable
        self.device = device
        self.weight_dtype = weight_dtype
        self.defaults = {
            "derive_env_obv_from_image": True,
            "num_inference_steps": 35,
            "inference_seed": 0,
            "fps": None,
        }
        self.defaults.update(dict(defaults or {}))
        self.config = load_multiworld_config(self.config_path)
        self._pipeline = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | None = None,
        args=None,
        device: str | None = None,
        runtime_root: str | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        base_model_root: str | None = None,
        vggt_root: str | None = None,
        python_executable: str | None = None,
        derive_env_obv_from_image: bool = True,
        num_inference_steps: int = 35,
        inference_seed: int = 0,
        fps: int | None = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> "MultiWorldItTakesTwoRuntime":
        del args
        if kwargs:
            raise ValueError(f"Unsupported MultiWorld kwargs: {', '.join(sorted(kwargs))}")
        resolved_runtime = resolve_runtime_root(runtime_root)
        return cls(
            runtime_root=resolved_runtime,
            config_path=resolve_config_path(config_path, resolved_runtime),
            checkpoint_path=resolve_checkpoint_path(
                checkpoint_path or pretrained_model_path,
                resolved_runtime,
            ),
            base_model_root=base_model_root,
            vggt_root=vggt_root,
            python_executable=python_executable,
            device=device or "cuda",
            weight_dtype=weight_dtype,
            defaults={
                "derive_env_obv_from_image": bool(derive_env_obv_from_image),
                "num_inference_steps": int(num_inference_steps),
                "inference_seed": int(inference_seed),
                "fps": None if fps is None else int(fps),
            },
        )

    def _load_pipeline(self):
        if self._pipeline is None:
            self._pipeline = load_multiworld_pipeline(
                base_model_root=self.base_model_root,
                checkpoint_path=self.checkpoint_path,
                config_path=self.config_path,
                vggt_root=self.vggt_root,
                device=self.device,
                torch_dtype=self.weight_dtype,
            )
        return self._pipeline

    def predict(
        self,
        image: Any,
        action: Mapping[str, Any],
        env_obv: Any = None,
        output_dir: str | None = None,
        save_name: str = "multiworld_ittakestwo",
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        num_inference_steps: int | None = None,
        inference_seed: int | None = None,
        derive_env_obv_from_image: bool | None = None,
        return_dict: bool = False,
        show_progress: bool = True,
    ):
        if not isinstance(action, Mapping):
            raise TypeError("MultiWorld ItTakesTwo expects an action mapping")
        output_root = Path(output_dir or tempfile.mkdtemp(prefix="multiworld_ittakestwo_"))
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        input_image = load_pil_image(image).convert("RGB")

        derive_environment = (
            self.defaults["derive_env_obv_from_image"]
            if derive_env_obv_from_image is None
            else bool(derive_env_obv_from_image)
        )
        derived_environment = env_obv is None and derive_environment
        if derived_environment:
            image_path = save_image_input(input_image, output_root / "input.png")
            env_obv = _derive_env_obv_from_image(image_path)
        if env_obv is None:
            raise ValueError("MultiWorld requires env_obv or derive_env_obv_from_image=True")

        dataset = self.config["eval_dataset_config"]["params"]
        video_params = dataset["video_params"]
        resolved_frames = int(
            num_frames
            or _resolve_num_frames(action, int(video_params.get("num_frames", 81)))
        )
        resolved_height = int(height or video_params.get("height", 480))
        default_width = int(video_params.get("width", 960))
        if str(dataset.get("return_view", "both")) != "both":
            default_width //= 2
        resolved_width = int(width or default_width)
        resolved_fps = int(
            fps
            or self.defaults["fps"]
            or max(1, 60 // int(video_params.get("frame_skip", 1)))
        )
        steps = int(num_inference_steps or self.defaults["num_inference_steps"])
        seed = int(
            self.defaults["inference_seed"]
            if inference_seed is None
            else inference_seed
        )

        frames = self._load_pipeline()(
            input_image=input_image,
            action=action,
            env_obv=env_obv,
            seed=seed,
            height=resolved_height,
            width=resolved_width,
            num_frames=resolved_frames,
            num_inference_steps=steps,
            show_progress=show_progress,
        )
        video_path = output_root / f"{save_name}.mp4"
        save_video_frames(frames, video_path, fps=resolved_fps)
        metadata = {
            "generated_video_path": str(video_path),
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "num_frames": resolved_frames,
            "height": resolved_height,
            "width": resolved_width,
            "fps": resolved_fps,
            "num_inference_steps": steps,
            "inference_seed": seed,
            "derived_env_obv_from_image": derived_environment,
        }
        (output_root / f"{save_name}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result = {
            "video": load_video_frames(video_path),
            "frames": frames,
            "output_dir": str(output_root),
            **metadata,
        }
        return result if return_dict else result["video"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native single-sample MultiWorld runner")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--runtime_root", default=None)
    parser.add_argument("--base_model_root", default=None)
    parser.add_argument("--vggt_root", default=None)
    parser.add_argument("--input_image_path", required=True)
    parser.add_argument("--action_path", required=True)
    parser.add_argument("--env_obv_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_name", default="multiworld_ittakestwo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--inference_seed", type=int, default=0)
    parser.add_argument("--derive_env_obv_from_image", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = MultiWorldItTakesTwoRuntime.from_pretrained(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        runtime_root=args.runtime_root,
        base_model_root=args.base_model_root,
        vggt_root=args.vggt_root,
        device=args.device,
    )
    runtime.predict(
        image=args.input_image_path,
        action=load_tree(args.action_path),
        env_obv=None if args.env_obv_path is None else load_tree(args.env_obv_path),
        output_dir=args.output_dir,
        save_name=args.save_name,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        inference_seed=args.inference_seed,
        derive_env_obv_from_image=args.derive_env_obv_from_image,
        show_progress=True,
    )
    return 0


__all__ = [
    "MultiWorldItTakesTwoRuntime",
    "dump_tree",
    "load_ittakestwo_action_csv",
    "load_tree",
    "main",
    "save_image_input",
]


if __name__ == "__main__":
    raise SystemExit(main())

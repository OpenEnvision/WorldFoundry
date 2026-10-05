"""Checkpoint-gated in-tree inference for TinyVLA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import json
from pathlib import Path
import pickle
from typing import Any

from worldfoundry.core.io.paths import resolve_data_path, resolve_worldfoundry_path
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class TinyVLARuntimeConfig:
    checkpoint_location: str
    base_model_location: str | None
    config_location: str | None
    stats_location: str
    device: str
    torch_dtype: str
    attention_backend: str
    compile_action_head: bool
    compile_mode: str
    allow_tf32: bool
    num_inference_steps: int
    seed: int
    camera_keys: tuple[str, ...]
    image_width: int
    image_height: int


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"configuration must contain a mapping: {path}")
    return dict(payload)


def _runtime_defaults() -> dict[str, Any]:
    path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", "tinyvla.yaml")
    return _load_yaml(path)


def _strict_local_path(location: str | None, label: str, *, required: bool = True) -> Path | None:
    if location in (None, "", "null", "None"):
        if required:
            raise ValueError(f"{label} must be configured")
        return None
    value = str(location)
    if "://" in value:
        raise ValueError(f"{label} must be a staged local path, not a URI: {value}")
    path = resolve_worldfoundry_path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


class _RestrictedStatsUnpickler(pickle.Unpickler):
    """Load the upstream numeric stats pickle without arbitrary globals."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("collections", "OrderedDict"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"disallowed class in TinyVLA statistics: {module}.{name}")
        return super().find_class(module, name)


def _load_stats(path: Path) -> dict[str, Any]:
    import numpy as np

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
    else:
        payload = _RestrictedStatsUnpickler(io.BytesIO(path.read_bytes())).load()
    if not isinstance(payload, Mapping):
        raise TypeError(f"TinyVLA statistics must contain a mapping: {path}")
    result: dict[str, Any] = {}
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "action_min", "action_max"):
        if key not in payload:
            continue
        value = payload[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(array).all():
            raise ValueError(f"TinyVLA statistic {key!r} contains non-finite values")
        result[key] = array
    return result


def _state_vector(observation: Mapping[str, Any], dimension: int) -> Any:
    import numpy as np

    value = first_present(
        observation,
        "state",
        "proprio",
        "robot_state",
        "joint_positions",
        "joint_action",
        "qpos",
    )
    while isinstance(value, Mapping):
        value = first_present(value, "vector", "value", "state", "proprio", "joint_positions", "qpos")
    if value is None:
        raise ValueError(f"TinyVLA requires a {dimension}-D robot state")
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.size != dimension:
        raise ValueError(f"TinyVLA requires a {dimension}-D robot state, got {state.size}")
    if not np.isfinite(state).all():
        raise ValueError("TinyVLA robot state contains non-finite values")
    return state


def _rgb_chw(value: Any, *, width: int, height: int) -> Any:
    import cv2
    import numpy as np
    import torch

    if hasattr(value, "convert"):
        array = np.asarray(value.convert("RGB"))
    else:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu").numpy()
        array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"TinyVLA camera must be unbatched, got {array.shape}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"TinyVLA camera must be HWC or CHW RGB, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("TinyVLA camera contains non-finite values")
        low = float(array.min(initial=0.0))
        high = float(array.max(initial=0.0))
        if low < 0.0:
            if low < -1.0 or high > 1.0:
                raise ValueError(f"unsupported TinyVLA camera range [{low}, {high}]")
            array = (array + 1.0) * 127.5
        elif high <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[:2] != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0))).to(torch.float32) / 255.0


def _process_camera(image: Any, processor: Any, *, device: str, dtype: Any) -> Any:
    import numpy as np
    import torch

    batch = image.unsqueeze(0)
    _, channels, height, width = batch.shape
    side = max(height, width)
    background = np.asarray(processor.image_mean, dtype=np.float32)
    expanded = np.broadcast_to(background, (1, side, side, channels)).copy()
    image_hwc = batch.permute(0, 2, 3, 1).cpu().numpy()
    y = (side - height) // 2
    x = (side - width) // 2
    expanded[:, y : y + height, x : x + width, :] = image_hwc
    pixels = processor.preprocess(
        torch.from_numpy(expanded),
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"]
    return pixels.to(device=device, dtype=dtype)


class TinyVLARuntime:
    """Persistent local TinyVLA VLM/action-head runtime."""

    def __init__(self, config: TinyVLARuntimeConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._stats: dict[str, Any] = {}
        self._checkpoint = ""
        self._device = ""
        self._dtype: Any = None
        self._checkpoint_kind = ""

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .loader import load_local_policy

        checkpoint = _strict_local_path(self.config.checkpoint_location, "TinyVLA task checkpoint")
        base_model = _strict_local_path(
            self.config.base_model_location,
            "TinyVLA base VLM",
            required=False,
        )
        config_dir = _strict_local_path(
            self.config.config_location,
            "TinyVLA task config directory",
            required=False,
        )
        stats_path = _strict_local_path(self.config.stats_location, "TinyVLA dataset statistics")
        assert checkpoint is not None and stats_path is not None
        device = resolve_inference_device(self.config.device, allow_cpu_fallback=True)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        if self.config.attention_backend not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(f"unsupported TinyVLA attention backend: {self.config.attention_backend}")
        tokenizer, model, processor, checkpoint_kind = load_local_policy(
            checkpoint,
            base_model=base_model,
            config_dir=config_dir,
            device=device,
            dtype=dtype,
            attention_backend=self.config.attention_backend,
        )
        if hasattr(model, "num_inference_timesteps"):
            model.num_inference_timesteps = self.config.num_inference_steps
        if self.config.compile_action_head and device.startswith("cuda"):
            model.embed_out = torch.compile(model.embed_out, mode=self.config.compile_mode)
        if device.startswith("cuda"):
            index = torch.device(device).index or 0
            major, _ = torch.cuda.get_device_capability(index)
            torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32 and major >= 8
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32 and major >= 8
        torch.manual_seed(self.config.seed)
        self._tokenizer = tokenizer
        self._model = model
        self._processor = processor
        self._stats = _load_stats(stats_path)
        self._checkpoint = str(checkpoint)
        self._device = device
        self._dtype = dtype
        self._checkpoint_kind = checkpoint_kind

    def _validate_stats(self, *, state_dim: int, action_dim: int, head_type: str) -> None:
        required = ["qpos_mean", "qpos_std"]
        required += ["action_min", "action_max"] if head_type == "droid_diffusion" else ["action_mean", "action_std"]
        missing = [key for key in required if key not in self._stats]
        if missing:
            raise ValueError(f"TinyVLA dataset statistics are missing {missing}")
        for key in ("qpos_mean", "qpos_std"):
            if self._stats[key].size != state_dim:
                raise ValueError(
                    f"TinyVLA statistic {key!r} has dimension "
                    f"{self._stats[key].size}, expected {state_dim}"
                )
        for key in required[2:]:
            if self._stats[key].size != action_dim:
                raise ValueError(
                    f"TinyVLA statistic {key!r} has dimension "
                    f"{self._stats[key].size}, expected {action_dim}"
                )
        if (self._stats["qpos_std"] <= 0).any():
            raise ValueError("TinyVLA qpos_std must be positive")

    def predict_action(self, *, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        from .prompt import build_prompt, tokenizer_image_token

        self._load()
        model = self._model
        config = model.config
        state_dim = int(config.state_dim)
        action_dim = int(config.action_dim)
        chunk_size = int(config.chunk_size)
        head_type = str(config.action_head_type)
        self._validate_stats(state_dim=state_dim, action_dim=action_dim, head_type=head_type)

        views = collect_images(observation, image, self.config.camera_keys)
        if len(views) < 2:
            raise ValueError("TinyVLA requires at least left and right RGB camera views")
        cameras = [
            _rgb_chw(view, width=self.config.image_width, height=self.config.image_height)
            for view in views[:3]
        ]
        processed = [
            _process_camera(camera, self._processor, device=self._device, dtype=self._dtype)
            for camera in cameras
        ]
        state = _state_vector(observation, state_dim)
        normalized_state = (state - self._stats["qpos_mean"]) / self._stats["qpos_std"]
        states = torch.from_numpy(normalized_state).unsqueeze(0).to(device=self._device, dtype=self._dtype)
        prompt = build_prompt(
            instruction,
            use_image_start_end=bool(getattr(config, "mm_use_im_start_end", False)),
        )
        input_ids = tokenizer_image_token(prompt, self._tokenizer).unsqueeze(0).to(self._device)
        attention_mask = input_ids.ne(self._tokenizer.pad_token_id)
        torch.manual_seed(self.config.seed)
        with torch.inference_mode():
            actions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=processed[0],
                images_r=processed[1],
                images_top=processed[2] if len(processed) > 2 else None,
                states=states,
                eval=True,
            )
        actions = actions[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        if head_type == "droid_diffusion":
            actions = (actions + 1.0) * 0.5 * (
                self._stats["action_max"] - self._stats["action_min"]
            ) + self._stats["action_min"]
        else:
            actions = actions * self._stats["action_std"] + self._stats["action_mean"]
        actions = np.asarray(actions, dtype=np.float32)
        expected = (chunk_size, action_dim)
        if actions.shape != expected:
            raise RuntimeError(f"TinyVLA returned action shape {actions.shape}, expected {expected}")
        if not np.isfinite(actions).all():
            raise FloatingPointError("TinyVLA produced non-finite actions")
        return completed_action_result(
            model_id="tinyvla",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=self._checkpoint,
            device=self._device,
            runtime="worldfoundry.tinyvla.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "dtype": str(self._dtype),
                "checkpoint_kind": self._checkpoint_kind,
                "action_head": head_type,
                "attention_backend": self.config.attention_backend,
                "camera_count": len(processed),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], TinyVLARuntime] = {}


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """WorldFoundry callable entrypoint for local TinyVLA inference."""

    del action_context
    options = {**_runtime_defaults(), **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    base_model = options.get("base_model_path")
    config_path = options.get("config_path")
    config = TinyVLARuntimeConfig(
        checkpoint_location=checkpoint,
        base_model_location=None if base_model in (None, "", "null", "None") else str(base_model),
        config_location=None if config_path in (None, "", "null", "None") else str(config_path),
        stats_location=str(options.get("stats_path") or ""),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        attention_backend=str(options.get("attention_backend") or "sdpa"),
        compile_action_head=option_bool(options.get("compile_action_head"), True),
        compile_mode=str(options.get("compile_mode") or "reduce-overhead"),
        allow_tf32=option_bool(options.get("allow_tf32"), True),
        num_inference_steps=int(options.get("num_inference_steps") or 10),
        seed=int(options.get("seed") or 42),
        camera_keys=tuple(str(item) for item in options["camera_keys"]),
        image_width=int(options.get("image_width") or 640),
        image_height=int(options.get("image_height") or 480),
    )
    if min(config.num_inference_steps, config.image_width, config.image_height) <= 0:
        raise ValueError("TinyVLA inference steps and image dimensions must be positive")
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = TinyVLARuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["TinyVLARuntime", "TinyVLARuntimeConfig", "predict_action"]

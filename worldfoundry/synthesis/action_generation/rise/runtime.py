"""Checkpoint-gated in-tree inference for the RISE action policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
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
class RiseRuntimeConfig:
    checkpoint_location: str
    norm_stats_location: str
    paligemma_tokenizer_location: str
    fast_tokenizer_location: str
    device: str
    torch_dtype: str
    compile_policy: bool
    compile_mode: str
    num_inference_steps: int
    seed: int
    camera_keys: tuple[str, ...]
    image_width: int
    image_height: int
    action_advantage: float
    advantage_bins: int


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"configuration must contain a mapping: {path}")
    return dict(payload)


def _runtime_defaults() -> dict[str, Any]:
    path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", "rise.yaml")
    return _load_yaml(path)


def _strict_local_path(location: str, label: str) -> Path:
    if not location:
        raise ValueError(f"{label} must be configured")
    if "://" in location:
        raise ValueError(f"{label} must be a staged local path, not a URI: {location}")
    return resolve_worldfoundry_path(location).expanduser().resolve()


def _checkpoint_directory(location: str) -> Path:
    root = _strict_local_path(location, "RISE policy checkpoint")
    if not root.is_dir():
        raise FileNotFoundError(f"RISE policy checkpoint directory is missing: {root}")
    pytorch_weights = root / "model.safetensors"
    jax_weights = root / "params"
    if not pytorch_weights.is_file() and not jax_weights.is_dir():
        raise FileNotFoundError(
            f"RISE requires a trained action-policy checkpoint at {pytorch_weights} or {jax_weights}"
        )
    return root


def _tokenizer_asset(location: str) -> Path:
    root = _strict_local_path(location, "PaliGemma tokenizer")
    candidates = (root,) if root.is_file() else (root / "paligemma_tokenizer.model", root / "tokenizer.model")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no local PaliGemma SentencePiece model was found under {root}")


def _norm_stats(location: str) -> dict[str, Any]:
    from worldfoundry.synthesis.action_generation.openpi import normalize

    path = _strict_local_path(location, "RISE normalization statistics")
    if path.is_dir():
        return normalize.load(path)
    if not path.is_file():
        raise FileNotFoundError(f"RISE normalization statistics are missing: {path}")
    return normalize.deserialize_json(path.read_text(encoding="utf-8"))


def _state_vector(observation: Mapping[str, Any]) -> Any:
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
        raise ValueError("RISE requires a 14-D robot state")
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.size != 14:
        raise ValueError(f"RISE requires a 14-D robot state, got {state.size}")
    if not np.isfinite(state).all():
        raise ValueError("RISE robot state contains non-finite values")
    return state


def _chw_rgb(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "convert"):
        array = np.asarray(value.convert("RGB"))
    else:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu").numpy()
        array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"RISE camera must be unbatched, got {array.shape}")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"RISE camera must be an RGB image, got {array.shape}")
    if array.shape[-1] == 3:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 3:
        raise ValueError(f"RISE camera must be CHW or HWC RGB, got {array.shape}")
    return np.ascontiguousarray(array)


def _standardize_camera(value: Any, *, width: int, height: int) -> Any:
    """Convert one view to RGB uint8 and match the policy's 320x240 source grid."""

    import cv2
    import numpy as np

    chw = _chw_rgb(value)
    hwc = np.moveaxis(chw, 0, -1)
    if np.issubdtype(hwc.dtype, np.floating):
        if not np.isfinite(hwc).all():
            raise ValueError("RISE camera contains non-finite values")
        low = float(hwc.min(initial=0.0))
        high = float(hwc.max(initial=0.0))
        if low < 0.0:
            if low < -1.0 or high > 1.0:
                raise ValueError(f"unsupported RISE camera range [{low}, {high}]")
            hwc = (hwc + 1.0) * 127.5
        elif high <= 1.0:
            hwc = hwc * 255.0
        hwc = np.clip(hwc, 0.0, 255.0).astype(np.uint8)
    elif hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    if hwc.shape[:2] != (height, width):
        hwc = cv2.resize(hwc, (width, height), interpolation=cv2.INTER_AREA)
    if hwc.shape != (height, width, 3):
        raise RuntimeError(f"RISE camera standardization produced {hwc.shape}, expected {(height, width, 3)}")
    return np.ascontiguousarray(np.moveaxis(hwc, -1, 0))


@dataclasses.dataclass(frozen=True)
class _RiseDataConfigFactory:
    """Materialize the official deployment-only AgileX Pi0.5 transforms."""

    action_advantage: float = 1.0
    advantage_bins: int = 10

    def create(self, assets_dirs: Path, model_config: Any) -> Any:
        del assets_dirs
        from worldfoundry.synthesis.action_generation.openpi import config as openpi_config
        from worldfoundry.synthesis.action_generation.openpi import transforms as openpi_transforms

        from .transforms import AdvantagePaligemmaTokenizer, AgilexInputs, AgilexOutputs

        model_transforms = openpi_transforms.Group(
            inputs=[
                openpi_transforms.ResizeImages(224, 224),
                openpi_transforms.TokenizePrompt(
                    AdvantagePaligemmaTokenizer(
                        model_config.max_token_len,
                        advantage=self.action_advantage,
                        advantage_bins=self.advantage_bins,
                    ),
                    discrete_state_input=True,
                ),
                openpi_transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )

        return openpi_config.DataConfig(
            repo_id="rise",
            asset_id="rise",
            data_transforms=openpi_transforms.Group(
                inputs=[AgilexInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[AgilexOutputs()],
            ),
            model_transforms=model_transforms,
            use_quantile_norm=True,
        )


class RiseRuntime:
    """Persistent Pi0.5 policy runtime using only local checkpoint assets."""

    def __init__(self, config: RiseRuntimeConfig) -> None:
        self.config = config
        self._policy: Any = None
        self._checkpoint = ""
        self._device = ""
        self._dtype: Any = None

    def _load(self) -> Any:
        if self._policy is not None:
            return self._policy

        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.synthesis.action_generation.openpi import config as openpi_config
        from worldfoundry.synthesis.action_generation.openpi import policy_loader
        from worldfoundry.synthesis.action_generation.openpi.modeling.pi0_config import Pi0Config

        checkpoint = _checkpoint_directory(self.config.checkpoint_location)
        tokenizer = _tokenizer_asset(self.config.paligemma_tokenizer_location)
        stats = _norm_stats(self.config.norm_stats_location)
        device = resolve_inference_device(self.config.device, allow_cpu_fallback=True)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        compile_mode = self.config.compile_mode if self.config.compile_policy and device.startswith("cuda") else None
        model_config = Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            pytorch_compile_mode=compile_mode,
        )
        runtime_config = openpi_config.RuntimeConfig(
            name="rise-policy-offline-release",
            model=model_config,
            data=_RiseDataConfigFactory(
                action_advantage=self.config.action_advantage,
                advantage_bins=self.config.advantage_bins,
            ),
            assets_base_dir=str(checkpoint / "assets"),
            seed=self.config.seed,
        )
        openpi_config.configure_local_tokenizers(
            paligemma=str(tokenizer),
            fast=self.config.fast_tokenizer_location,
        )
        torch.manual_seed(self.config.seed)
        policy = policy_loader.create_trained_policy(
            runtime_config,
            checkpoint,
            norm_stats=stats,
            sample_kwargs={"num_steps": self.config.num_inference_steps},
            pytorch_device=device,
            inference_dtype=str(dtype).removeprefix("torch."),
        )
        self._policy = policy
        self._checkpoint = str(checkpoint)
        self._device = device
        self._dtype = dtype
        return policy

    def predict_action(self, *, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        policy = self._load()
        views = collect_images(observation, image, self.config.camera_keys)
        if len(views) < 3:
            raise ValueError("RISE requires three RGB views ordered as head, left wrist, and right wrist")
        request = {
            "images": {
                "top_head": _standardize_camera(
                    views[0], width=self.config.image_width, height=self.config.image_height
                ),
                "hand_left": _standardize_camera(
                    views[1], width=self.config.image_width, height=self.config.image_height
                ),
                "hand_right": _standardize_camera(
                    views[2], width=self.config.image_width, height=self.config.image_height
                ),
            },
            "state": _state_vector(observation),
            "prompt": instruction,
        }
        torch.manual_seed(self.config.seed)
        with torch.inference_mode():
            result = policy.infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected = (50, 14)
        if actions.shape != expected:
            raise RuntimeError(f"RISE returned action shape {actions.shape}, expected {expected}")
        if not np.isfinite(actions).all():
            raise FloatingPointError("RISE produced non-finite actions")
        from .transforms import discretize_advantage

        return completed_action_result(
            model_id="rise",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=self._checkpoint,
            device=self._device,
            runtime="worldfoundry.rise.in_tree_openpi_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "dtype": str(self._dtype),
                "architecture": "pi0.5-flow-matching-policy",
                "num_inference_steps": self.config.num_inference_steps,
                "action_advantage": self.config.action_advantage,
                "advantage_bin": discretize_advantage(
                    self.config.action_advantage,
                    self.config.advantage_bins,
                ),
                "source_image_size": [self.config.image_width, self.config.image_height],
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], RiseRuntime] = {}


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
    """WorldFoundry callable entrypoint for local RISE action inference."""

    del action_context
    options = {**_runtime_defaults(), **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    config = RiseRuntimeConfig(
        checkpoint_location=checkpoint,
        norm_stats_location=str(options.get("norm_stats_path") or ""),
        paligemma_tokenizer_location=str(options.get("paligemma_tokenizer_path") or ""),
        fast_tokenizer_location=str(options.get("fast_tokenizer_path") or ""),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        compile_policy=option_bool(options.get("compile_policy"), True),
        compile_mode=str(options.get("compile_mode") or "reduce-overhead"),
        num_inference_steps=int(options.get("num_inference_steps") or 10),
        seed=int(options.get("seed") or 42),
        camera_keys=tuple(str(item) for item in options["camera_keys"]),
        image_width=int(options.get("image_width") or 320),
        image_height=int(options.get("image_height") or 240),
        action_advantage=float(
            options.get("action_advantage")
            if options.get("action_advantage") is not None
            else 1.0
        ),
        advantage_bins=int(options.get("advantage_bins") or 10),
    )
    if config.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if config.image_width <= 0 or config.image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if config.advantage_bins <= 0:
        raise ValueError("advantage_bins must be positive")
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = RiseRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["RiseRuntime", "RiseRuntimeConfig", "predict_action"]

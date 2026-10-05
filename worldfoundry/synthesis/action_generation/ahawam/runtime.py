"""Stateful, local-only AHA-WAM action inference runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import (
    resolve_data_path,
    resolve_local_hf_model_path,
    resolve_worldfoundry_path,
)
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class AHAWAMRuntimeConfig:
    checkpoint_location: str
    checkpoint_filename: str
    dataset_stats_location: str
    vae_location: str
    vae_filename: str
    text_encoder_location: str
    text_encoder_filename: str
    tokenizer_location: str
    architecture_config_location: str
    device: str
    torch_dtype: str
    vae_device: str
    vae_dtype: str
    text_encoder_device: str
    text_encoder_dtype: str
    auto_vae_gpu_min_gib: float
    auto_text_gpu_min_gib: float
    chunks_per_video_prefill: int
    num_inference_steps: int
    sigma_shift: float | None
    seed: int | None
    rand_device: str
    image_width: int
    image_height: int
    camera_keys: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"configuration must contain a mapping: {path}")
    return dict(payload)


def _runtime_defaults() -> dict[str, Any]:
    return _load_yaml(resolve_data_path("models", "runtime", "configs", "vla_va_wam", "ahawam.yaml"))


def _policy_checkpoint(location: str, filename: str) -> Path:
    direct = resolve_worldfoundry_path(location)
    if direct.is_file():
        return direct.resolve()
    if direct.is_dir():
        root = direct.resolve()
    else:
        root = resolve_local_hf_model_path(location, required_files=(filename,))
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"AHA-WAM policy checkpoint is missing: {path}")
    return path.resolve()


def _asset_file(location: str, expected_name: str) -> Path:
    direct = resolve_worldfoundry_path(location)
    if direct.is_file():
        return direct.resolve()
    if direct.is_dir() and (direct / expected_name).is_file():
        return (direct / expected_name).resolve()
    root = resolve_local_hf_model_path(location, required_files=(expected_name,))
    return (root / expected_name).resolve()


def _tokenizer_directory(location: str) -> Path:
    direct = resolve_worldfoundry_path(location)
    root = direct.resolve() if direct.is_dir() else resolve_local_hf_model_path(location)
    markers = ("tokenizer.json", "tokenizer_config.json", "spiece.model")
    if not any((root / marker).is_file() for marker in markers):
        raise FileNotFoundError(f"local UMT5 tokenizer assets are missing under {root}")
    return root


def _component_device(requested: str, policy_device: str, minimum_gib: float) -> str:
    import torch

    from worldfoundry.core.execution.device import resolve_inference_device

    choice = str(requested).strip().lower()
    if choice in {"policy", "same"}:
        return policy_device
    if choice != "auto":
        return resolve_inference_device(choice)
    parsed = torch.device(policy_device)
    if parsed.type != "cuda":
        return "cpu"
    index = torch.cuda.current_device() if parsed.index is None else parsed.index
    total_gib = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    return policy_device if total_gib >= float(minimum_gib) else "cpu"


def _vector(value: Any, dimension: int) -> Any:
    import numpy as np

    while isinstance(value, Mapping):
        nested = first_present(value, "vector", "value", "state", "proprio", "joint_positions", "qpos")
        if nested is None:
            break
        value = nested
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != dimension:
        raise ValueError(f"AHA-WAM requires a {dimension}-D proprioceptive state, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError("proprioceptive state contains non-finite values")
    return array


def _resize(value: Any, width: int, height: int) -> Any:
    import numpy as np
    from PIL import Image

    from worldfoundry.core.media.processing.image_utils import load_pil_image

    image = load_pil_image(value, first_sequence_item=False)
    return np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def _camera_tensor(
    observation: Mapping[str, Any],
    image: Any,
    *,
    width: int,
    height: int,
    camera_keys: Sequence[str],
    device: str,
    dtype: Any,
) -> Any:
    import numpy as np
    import torch

    views = collect_images(observation, image, camera_keys)
    if not views:
        raise ValueError("AHA-WAM requires at least one RGB camera image")
    if len(views) >= 3:
        lower_height = height // 3
        upper_height = height - lower_height
        left_width = width // 2
        head = _resize(views[0], width, upper_height)
        left = _resize(views[1], left_width, lower_height)
        right = _resize(views[2], width - left_width, lower_height)
        rgb = np.concatenate((head, np.concatenate((left, right), axis=1)), axis=0)
    else:
        rgb = _resize(views[0], width, height)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


def _statistics_entry(statistics: Mapping[str, Any], kind: str, dimension: int) -> tuple[Any, Any]:
    import numpy as np

    group = statistics.get(kind)
    if not isinstance(group, Mapping):
        raise KeyError(f"dataset statistics are missing {kind!r}")
    entry: Any = group.get("default", group)
    if isinstance(entry, Mapping) and "global_mean" not in entry and "mean" not in entry:
        candidates = [value for value in entry.values() if isinstance(value, Mapping)]
        if len(candidates) == 1:
            entry = candidates[0]
    if not isinstance(entry, Mapping):
        raise TypeError(f"dataset statistics {kind!r} entry must be a mapping")
    mean = np.asarray(entry.get("global_mean", entry.get("mean")), dtype=np.float32).reshape(-1)
    std = np.asarray(entry.get("global_std", entry.get("std")), dtype=np.float32).reshape(-1)
    if mean.size != dimension or std.size != dimension:
        raise ValueError(f"dataset {kind} statistics must be {dimension}-D, got {mean.size}/{std.size}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"dataset {kind} statistics contain invalid mean/std values")
    return mean, std


class AHAWAMRuntime:
    """Persistent two-phase runtime matching the released deployment policy."""

    def __init__(self, config: AHAWAMRuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._checkpoint = ""
        self._device = ""
        self._dtype: Any = None
        self._vae_device = ""
        self._vae_dtype: Any = None
        self._text_device = ""
        self._text_dtype: Any = None
        self._state_mean: Any = None
        self._state_std: Any = None
        self._action_mean: Any = None
        self._action_std: Any = None
        self._state_dim = 0
        self._action_dim = 0
        self._prompt_template = ""
        self._episode_id: str | None = None
        self._instruction: str | None = None
        self._prefilled = False
        self._chunks_since_prefill = 0

    def reset(self, episode_id: Any = None) -> None:
        if self._model is not None:
            self._model.reset_history()
            self._model._inference_state = None
        self._episode_id = None if episode_id is None else str(episode_id)
        self._instruction = None
        self._prefilled = False
        self._chunks_since_prefill = 0

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import restore_ahawam_model

        checkpoint = _policy_checkpoint(self.config.checkpoint_location, self.config.checkpoint_filename)
        stats_path = resolve_worldfoundry_path(self.config.dataset_stats_location)
        if not stats_path.is_file():
            raise FileNotFoundError(f"AHA-WAM dataset_stats.json is missing: {stats_path}")
        statistics = json.loads(stats_path.read_text(encoding="utf-8"))
        if not isinstance(statistics, Mapping):
            raise TypeError("AHA-WAM dataset_stats.json must contain an object")

        architecture_path = resolve_worldfoundry_path(self.config.architecture_config_location)
        if not architecture_path.is_file():
            raise FileNotFoundError(architecture_path)
        architecture_payload = _load_yaml(architecture_path)
        architecture = architecture_payload.get("architecture", architecture_payload)
        if not isinstance(architecture, Mapping):
            raise TypeError("AHA-WAM architecture config must contain an architecture mapping")

        device_name = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device_name, self.config.torch_dtype)
        vae_device = _component_device(self.config.vae_device, device_name, self.config.auto_vae_gpu_min_gib)
        vae_dtype = resolve_inference_dtype(vae_device, self.config.vae_dtype)
        text_device = _component_device(
            self.config.text_encoder_device,
            device_name,
            self.config.auto_text_gpu_min_gib,
        )
        text_dtype = resolve_inference_dtype(text_device, self.config.text_encoder_dtype)
        model = restore_ahawam_model(
            policy_checkpoint=checkpoint,
            vae_checkpoint=_asset_file(self.config.vae_location, self.config.vae_filename),
            text_encoder_checkpoint=_asset_file(
                self.config.text_encoder_location,
                self.config.text_encoder_filename,
            ),
            tokenizer_path=_tokenizer_directory(self.config.tokenizer_location),
            architecture=architecture,
            policy_device=torch.device(device_name),
            policy_dtype=dtype,
            vae_device=torch.device(vae_device),
            vae_dtype=vae_dtype,
            text_encoder_device=torch.device(text_device),
            text_encoder_dtype=text_dtype,
        )
        policy_config = architecture.get("policy")
        action_config = architecture.get("action_expert")
        text_config = architecture.get("text")
        if not isinstance(policy_config, Mapping) or not isinstance(action_config, Mapping):
            raise TypeError("AHA-WAM architecture requires policy and action_expert mappings")
        if not isinstance(text_config, Mapping):
            raise TypeError("AHA-WAM architecture requires a text mapping")
        self._state_dim = int(policy_config["proprio_dim"])
        self._action_dim = int(action_config["action_dim"])
        self._prompt_template = str(text_config["prompt_template"])
        if "{task}" not in self._prompt_template:
            raise ValueError("AHA-WAM prompt_template must contain {task}")
        self._state_mean, self._state_std = _statistics_entry(statistics, "state", self._state_dim)
        self._action_mean, self._action_std = _statistics_entry(statistics, "action", self._action_dim)
        self._model = model
        self._checkpoint = str(checkpoint)
        self._device = device_name
        self._dtype = dtype
        self._vae_device = vae_device
        self._vae_dtype = vae_dtype
        self._text_device = text_device
        self._text_dtype = text_dtype
        return model

    def _episode(self, observation: Mapping[str, Any]) -> None:
        episode = first_present(observation, "episode_id", "episode", "trajectory_id")
        if option_bool(observation.get("reset"), False):
            self.reset(episode)
        elif episode is not None and str(episode) != self._episode_id:
            self.reset(episode)

    def _soft_reprefill(self) -> None:
        self._model._inference_state = None
        self._prefilled = False
        self._chunks_since_prefill = 0

    def predict_action(self, *, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        model = self._load()
        self._episode(observation)
        if self._instruction is not None and instruction != self._instruction:
            self._soft_reprefill()
        self._instruction = instruction

        chunks_per_prefill = int(self.config.chunks_per_video_prefill)
        maximum_chunks = int(model.action_horizon) // int(model.action_chunk_size)
        if chunks_per_prefill <= 0 or chunks_per_prefill > maximum_chunks:
            raise ValueError(f"chunks_per_video_prefill must be in [1, {maximum_chunks}]")
        state_value = first_present(
            observation,
            "state",
            "proprio",
            "robot_state",
            "joint_positions",
            "joint_action",
            "qpos",
        )
        if state_value is None:
            raise ValueError(f"AHA-WAM requires a {self._state_dim}-D state/proprio observation")
        state = _vector(state_value, self._state_dim)
        normalized_state = (state - self._state_mean) / self._state_std
        proprio = torch.from_numpy(normalized_state).unsqueeze(0)
        current_image = _camera_tensor(
            observation,
            image,
            width=int(self.config.image_width),
            height=int(self.config.image_height),
            camera_keys=self.config.camera_keys,
            device=self._device,
            dtype=self._dtype,
        )

        next_chunk = 0
        if getattr(model, "_inference_state", None) is not None:
            next_chunk = int(model._inference_state.get("next_chunk_index", 0))
        if next_chunk >= chunks_per_prefill or self._chunks_since_prefill >= chunks_per_prefill:
            self._soft_reprefill()

        prompt = self._prompt_template.format(task=instruction)
        prefilled_now = False
        with torch.inference_mode():
            if not self._prefilled:
                model.infer_action(
                    prompt=prompt,
                    input_image=current_image,
                    action_horizon=chunks_per_prefill * int(model.action_chunk_size),
                    seed=self.config.seed,
                    rand_device=self.config.rand_device,
                    phase="video",
                )
                self._prefilled = True
                self._chunks_since_prefill = 0
                prefilled_now = True
            result = model.infer_action(
                chunk_obs_image=current_image,
                chunk_proprio=proprio,
                num_inference_steps=int(self.config.num_inference_steps),
                sigma_shift=self.config.sigma_shift,
                phase="action",
            )
        self._chunks_since_prefill += 1
        normalized_action = result["action_chunk"].numpy()
        action = normalized_action * self._action_std + self._action_mean
        if action.shape != (int(model.action_chunk_size), self._action_dim):
            raise RuntimeError(f"AHA-WAM returned an unexpected action shape: {action.shape}")
        if not np.isfinite(action).all():
            raise FloatingPointError("AHA-WAM produced non-finite actions")
        return completed_action_result(
            model_id="ahawam",
            instruction=instruction,
            actions=action.tolist(),
            checkpoint_path=self._checkpoint,
            device=self._device,
            runtime="worldfoundry.ahawam.in_tree_runtime",
            raw_output={"normalized_actions": normalized_action.tolist()},
            metadata={
                "action_shape": list(action.shape),
                "chunk_index": int(result["chunk_index"]),
                "video_prefilled": prefilled_now,
                "chunks_per_video_prefill": chunks_per_prefill,
                "dtype": str(self._dtype),
                "vae_device": self._vae_device,
                "vae_dtype": str(self._vae_dtype),
                "text_encoder_device": self._text_device,
                "text_encoder_dtype": str(self._text_dtype),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], AHAWAMRuntime] = {}


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
    """WorldFoundry callable entrypoint for the local two-phase policy."""

    del action_context
    defaults = _runtime_defaults()
    options = {**defaults, **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    if not checkpoint:
        raise ValueError("AHA-WAM requires a staged local checkpoint_path")
    sigma_shift = options.get("sigma_shift")
    seed = options.get("seed")
    config = AHAWAMRuntimeConfig(
        checkpoint_location=checkpoint,
        checkpoint_filename=str(options["checkpoint_filename"]),
        dataset_stats_location=str(options.get("dataset_stats_path") or ""),
        vae_location=str(options["vae_path"]),
        vae_filename=str(options["vae_filename"]),
        text_encoder_location=str(options["text_encoder_path"]),
        text_encoder_filename=str(options["text_encoder_filename"]),
        tokenizer_location=str(options["tokenizer_path"]),
        architecture_config_location=str(
            options.get("architecture_config_path")
            or resolve_data_path("models", "runtime", "configs", "vla_va_wam", "ahawam-architecture.yaml")
        ),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        vae_device=str(options.get("vae_device") or "auto"),
        vae_dtype=str(options.get("vae_dtype") or "auto"),
        text_encoder_device=str(options.get("text_encoder_device") or "auto"),
        text_encoder_dtype=str(options.get("text_encoder_dtype") or "auto"),
        auto_vae_gpu_min_gib=float(options["auto_vae_gpu_min_gib"]),
        auto_text_gpu_min_gib=float(options["auto_text_gpu_min_gib"]),
        chunks_per_video_prefill=int(options["chunks_per_video_prefill"]),
        num_inference_steps=int(options["num_inference_steps"]),
        sigma_shift=None if sigma_shift in (None, "", "null") else float(sigma_shift),
        seed=None if seed in (None, "", "null") else int(seed),
        rand_device=str(options["rand_device"]),
        image_width=int(options["image_width"]),
        image_height=int(options["image_height"]),
        camera_keys=tuple(str(value) for value in options["camera_keys"]),
    )
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = AHAWAMRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["AHAWAMRuntime", "AHAWAMRuntimeConfig", "predict_action"]

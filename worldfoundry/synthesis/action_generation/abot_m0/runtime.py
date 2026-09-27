"""Local-checkpoint inference runtime for ABot-M0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_data_path, resolve_local_hf_model_path, resolve_worldfoundry_path
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)

MODEL_ID = "abot-m0"


def _runtime_defaults() -> dict[str, Any]:
    import yaml

    path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", "abot-m0.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"ABot-M0 runtime config must be a mapping: {path}")
    return payload


def _resolve_local_directory(value: str, required_files: Sequence[str] = ()) -> Path:
    direct = resolve_worldfoundry_path(value)
    if direct.is_dir() and all((direct / name).is_file() for name in required_files):
        return direct.resolve()
    return resolve_local_hf_model_path(value, required_files=tuple(required_files))


def _checkpoint_root(location: str) -> tuple[Path, Path]:
    direct = resolve_worldfoundry_path(location)
    if direct.is_file():
        checkpoint = direct.resolve()
        root = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
        return root, checkpoint
    root = _resolve_local_directory(location, required_files=("config.yaml", "dataset_statistics.json"))
    patterns = (
        "checkpoints/steps_*_pytorch_model.pt",
        "steps_*_pytorch_model.pt",
        "checkpoints/pytorch_model.pt",
        "pytorch_model.pt",
        "model.safetensors",
        "model.pt",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = sorted(set(path.resolve() for path in candidates if path.is_file()))
    if not candidates:
        raise FileNotFoundError(f"ABot-M0 policy weights are missing under {root}")
    # Released step checkpoints sort lexicographically; compare the numeric step.
    def rank(path: Path) -> tuple[int, str]:
        import re

        match = re.search(r"steps_(\d+)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return root, max(candidates, key=rank)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_policy_config(root: Path, filename: str) -> Mapping[str, Any]:
    import yaml

    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"ABot-M0 model config is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, "ABot-M0 model config")


def _load_state_dict(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload: Any = load_file(str(path), device="cpu")
    else:
        from worldfoundry.core.model_loading import load_torch_checkpoint

        payload = load_torch_checkpoint(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(payload, Mapping) and len(payload) == 1:
        for key in ("state_dict", "model", "module", "model_state"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise TypeError(f"ABot-M0 checkpoint must contain a state dict: {path}")
    if payload and all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


def _align_qwen_vocabulary(model: Any, state_dict: Mapping[str, Any]) -> None:
    """Match checkpoints trained with the released action-token extension.

    Action tokens are not emitted during policy inference, so the ordinary
    local Qwen processor is sufficient.  The embedding table must nonetheless
    retain the checkpoint's expanded row count for strict weight loading.
    """

    embedding_rows: int | None = None
    for key, value in state_dict.items():
        if str(key).endswith("embed_tokens.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            embedding_rows = int(value.shape[0])
            break
    if embedding_rows is None:
        return
    qwen = model.qwen_vl_interface.model
    current_rows = int(qwen.get_input_embeddings().weight.shape[0])
    if current_rows != embedding_rows:
        qwen.resize_token_embeddings(embedding_rows)


def _statistics(root: Path, filename: str, key: str | None) -> Mapping[str, Any]:
    from worldfoundry.core.action_normalization import select_modality_statistics

    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"ABot-M0 dataset statistics are missing: {path}")
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    _selected_key, action = select_modality_statistics(
        _mapping(payload, "ABot-M0 statistics"),
        modality="action",
        key=key,
    )
    return action


def _dual_arm_to_training(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.shape[-1] != 14:
        raise ValueError(f"dual_arm_14 layout requires 14 values, got {array.shape}")
    return np.concatenate((array[..., :6], array[..., 7:13], array[..., 6:7], array[..., 13:14]), axis=-1)


def _training_to_dual_arm(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.shape[-1] != 14:
        raise ValueError(f"dual_arm_14 layout requires 14 values, got {array.shape}")
    return np.concatenate((array[..., :6], array[..., 12:13], array[..., 6:12], array[..., 13:14]), axis=-1)


@dataclass(frozen=True)
class ABotM0RuntimeConfig:
    checkpoint_location: str
    qwen_location: str
    processor_location: str | None
    config_file: str
    statistics_file: str
    statistics_key: str | None
    camera_keys: tuple[str, ...]
    image_size: tuple[int, int]
    include_state: bool
    action_layout: str
    binary_action_indices: tuple[int, ...]
    device: str
    torch_dtype: str
    attention_backend: str
    num_inference_steps: int | None
    tokenizer_max_length: int | None
    seed: int
    device_map: Any
    device_ids: tuple[int, ...]
    allow_cpu_offload: bool
    spatial_device: str | None
    compile_action_expert: bool
    compile_mode: str | None


class ABotM0Runtime:
    def __init__(self, config: ABotM0RuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._root: Path | None = None
        self._checkpoint: Path | None = None
        self._action_stats: Mapping[str, Any] | None = None
        self._resolved_device = ""
        self._resolved_dtype = ""
        self._device_map: Mapping[str, Any] | None = None

    def _place_qwen_multi_gpu(self, qwen: Any, dtype: Any) -> Mapping[str, Any]:
        import torch
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory

        selected = self.config.device_ids or tuple(range(torch.cuda.device_count()))
        max_memory: dict[Any, Any] = {}
        for index in selected:
            free_bytes, _total = torch.cuda.mem_get_info(index)
            max_memory[index] = int(free_bytes * 0.85)
        if self.config.allow_cpu_offload:
            max_memory["cpu"] = "64GiB"
        no_split = (
            "Qwen3VLDecoderLayer",
            "Qwen3VLVisionBlock",
            "Qwen3VLVisionTransformerBlock",
        )
        balanced = get_balanced_memory(
            qwen,
            max_memory=max_memory,
            no_split_module_classes=list(no_split),
            dtype=dtype,
            low_zero=str(self.config.device_map) == "balanced_low_0",
        )
        device_map = infer_auto_device_map(
            qwen,
            max_memory=balanced,
            no_split_module_classes=list(no_split),
            dtype=dtype,
        )
        offloaded = {str(value) for value in device_map.values()} & {"cpu", "disk"}
        if offloaded and not self.config.allow_cpu_offload:
            raise RuntimeError(f"ABot-M0 does not fit selected GPUs; inferred offload targets: {sorted(offloaded)}")
        dispatch_model(qwen, device_map=device_map)
        return device_map

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.core.execution.inference import compile_module_if_enabled

        from .action_head import ActionHeadConfig
        from .modeling import ABotM0Model

        root, checkpoint = _checkpoint_root(self.config.checkpoint_location)
        policy_config = _load_policy_config(root, self.config.config_file)
        framework = _mapping(policy_config.get("framework"), "ABot-M0 framework config")
        action_mapping = _mapping(framework.get("action_model"), "ABot-M0 action model config")
        state_dict = _load_state_dict(checkpoint)
        # The released LIBERO and RoboTwin2 configs predate the ``use_vggt``
        # field even though their checkpoints contain the spatial branch.  The
        # checkpoint is authoritative when the field is absent; an explicit
        # config value remains strict and any disagreement is caught below by
        # state-dict validation.
        has_spatial_weights = any(str(key).startswith("spatial_model.") for key in state_dict)
        use_vggt = option_bool(framework.get("use_vggt"), has_spatial_weights)
        action_config = ActionHeadConfig.from_mapping(action_mapping)

        qwen_root = _resolve_local_directory(self.config.qwen_location, required_files=("config.json",))
        processor_root = (
            _resolve_local_directory(self.config.processor_location, required_files=("tokenizer_config.json",))
            if self.config.processor_location
            else qwen_root
        )
        primary_request = f"cuda:{self.config.device_ids[0]}" if self.config.device_ids else self.config.device
        primary = resolve_inference_device(primary_request)
        dtype = resolve_inference_dtype(primary, self.config.torch_dtype)
        model = ABotM0Model(
            qwen_path=str(qwen_root),
            processor_path=str(processor_root),
            action_config=action_config,
            dtype=dtype,
            attention_backend=self.config.attention_backend,
            use_vggt=use_vggt,
        )
        _align_qwen_vocabulary(model, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "ABot-M0 checkpoint architecture mismatch: "
                f"missing={list(missing)[:20]} ({len(missing)} total), "
                f"unexpected={list(unexpected)[:20]} ({len(unexpected)} total)"
            )
        del state_dict
        model.requires_grad_(False).eval()

        multi_gpu = str(self.config.device_map or "").lower() in {"auto", "balanced", "balanced_low_0"}
        multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        if multi_gpu:
            self._device_map = self._place_qwen_multi_gpu(model.qwen_vl_interface.model, dtype)
        else:
            model.qwen_vl_interface.model.to(device=primary, dtype=dtype)
        model.action_model.to(device=primary, dtype=dtype)
        if model.spatial_model is not None:
            spatial = resolve_inference_device(self.config.spatial_device or primary)
            model.spatial_model.to(device=spatial, dtype=dtype)
            model.spatial_projector.to(device=spatial, dtype=dtype)
            model.fuser.to(device=spatial, dtype=dtype)
        model.action_model.model = compile_module_if_enabled(
            model.action_model.model,
            enabled=self.config.compile_action_expert,
            label="abot_m0_action_expert",
            mode=self.config.compile_mode,
            dynamic=False,
        )
        self._model = model
        self._root = root
        self._checkpoint = checkpoint
        self._action_stats = _statistics(root, self.config.statistics_file, self.config.statistics_key)
        self._resolved_device = primary
        self._resolved_dtype = str(dtype)
        return model

    def _prepare_state(self, observation: Mapping[str, Any], state_dim: int) -> Any:
        import numpy as np

        if not self.config.include_state:
            return None
        state = first_present(observation, "state", "proprio", "proprio_state", "robot_state")
        if state is None:
            raise ValueError("ABot-M0 checkpoint is configured to use robot state")
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.config.action_layout == "dual_arm_14":
            values = _dual_arm_to_training(values)
        if values.shape != (state_dim,):
            raise ValueError(f"ABot-M0 requires {state_dim} state values, got {values.shape}")
        return values[None, None, :]

    def _unnormalize(self, normalized: Any) -> Any:
        import numpy as np

        from worldfoundry.core.action_normalization import unnormalize_action_values

        assert self._action_stats is not None
        stats = dict(self._action_stats)
        if "min" in stats and "max" in stats:
            mode = "min_max"
            low_name, high_name = "min", "max"
        elif "q01" in stats and "q99" in stats:
            mode = "q99"
            low_name, high_name = "q01", "q99"
        else:
            raise KeyError("ABot-M0 action statistics require min/max or q01/q99")
        low = np.asarray(stats[low_name], dtype=np.float32)
        high = np.asarray(stats[high_name], dtype=np.float32)
        values = np.clip(np.asarray(normalized, dtype=np.float32), -1.0, 1.0)
        statistics_width = int(low.shape[-1])
        if high.shape[-1] != statistics_width:
            raise ValueError("ABot-M0 action statistics have inconsistent widths")
        if values.shape[-1] < statistics_width:
            raise ValueError(
                f"ABot-M0 model emitted {values.shape[-1]} actions but statistics require {statistics_width}"
            )
        padded_action_head = values.shape[-1] > statistics_width
        # The official LIBERO adapter takes ``normalized_actions[:, -7:]``:
        # its released 14-wide shared head left-pads the seven Franka actions.
        if padded_action_head:
            values = values[..., -statistics_width:]
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        action_layout = self.config.action_layout
        if action_layout == "auto":
            # RoboTwin2 trains in [left joints, right joints, left gripper,
            # right gripper] order and its official adapter restores the
            # interleaved per-arm environment layout after unnormalization.
            action_layout = (
                "dual_arm_14"
                if statistics_width == 14 and mask.shape == (14,) and not mask[-2:].any()
                else "identity"
            )
        if action_layout == "dual_arm_14" and low.shape[-1] == 14:
            # Some deployment statistics are stored in interleaved modality order.
            if abs(float(low[6])) < 1e-6 and abs(float(low[12])) > 0.05:
                low, high, mask = _dual_arm_to_training(low), _dual_arm_to_training(high), _dual_arm_to_training(mask)
        stats[low_name] = low
        stats[high_name] = high
        stats["mask"] = mask
        binary_action_indices = self.config.binary_action_indices
        if not binary_action_indices and padded_action_head and statistics_width == 7:
            # LIBERO's official adapter thresholds its final gripper value.
            binary_action_indices = (6,)
        for index in binary_action_indices:
            if 0 <= index < values.shape[-1]:
                values[..., index] = (values[..., index] >= 0.5).astype(np.float32)
        actions = unnormalize_action_values(values, stats, mode=mode)
        return _training_to_dual_arm(actions) if action_layout == "dual_arm_14" else actions

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch

        from worldfoundry.core.execution.inference import worldfoundry_inference_context

        model = self._load()
        views = collect_images(observation, image, self.config.camera_keys)
        if len(views) != len(self.config.camera_keys):
            raise ValueError(
                f"ABot-M0 requires {len(self.config.camera_keys)} ordered RGB views "
                f"{list(self.config.camera_keys)}, got {len(views)}"
            )
        state = self._prepare_state(observation, model.action_config.state_dim)
        state_tensor = torch.from_numpy(state) if state is not None else None
        with worldfoundry_inference_context():
            normalized = model.predict_action(
                images=[views],
                instructions=[instruction],
                state=state_tensor,
                image_size=self.config.image_size,
                seed=self.config.seed,
                num_inference_steps=self.config.num_inference_steps,
                tokenizer_max_length=self.config.tokenizer_max_length,
            )[0].float().cpu().numpy()
        actions = self._unnormalize(normalized)
        if not np.isfinite(actions).all():
            raise FloatingPointError("ABot-M0 produced non-finite actions")
        assert self._checkpoint is not None
        return completed_action_result(
            model_id=MODEL_ID,
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=str(self._checkpoint),
            device=self._resolved_device,
            runtime="worldfoundry.abot_m0.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "camera_views": len(views),
                "dtype": self._resolved_dtype,
                "device_map": dict(self._device_map or {}),
                "num_inference_steps": self.config.num_inference_steps or model.action_config.num_inference_timesteps,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], ABotM0Runtime] = {}


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
    """WorldFoundry callable entrypoint."""

    del action_context
    options = {**_runtime_defaults(), **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    if not checkpoint:
        raise ValueError("ABot-M0 requires a staged local checkpoint")
    cameras = tuple(str(item) for item in options.get("camera_keys") or ("image_0",))
    raw_size = options.get("image_size") or (224, 224)
    image_size = (int(raw_size), int(raw_size)) if isinstance(raw_size, int) else (int(raw_size[0]), int(raw_size[1]))
    raw_indices = options.get("binary_action_indices") or ()
    raw_devices = options.get("device_ids") or ()
    config = ABotM0RuntimeConfig(
        checkpoint_location=checkpoint,
        qwen_location=str(options["qwen_path"]),
        processor_location=str(options["processor_path"]) if options.get("processor_path") else None,
        config_file=str(options.get("config_file") or "config.yaml"),
        statistics_file=str(options.get("statistics_file") or "dataset_statistics.json"),
        statistics_key=str(options["statistics_key"]) if options.get("statistics_key") else None,
        camera_keys=cameras,
        image_size=image_size,
        include_state=option_bool(options.get("include_state"), False),
        action_layout=str(options.get("action_layout") or "identity"),
        binary_action_indices=tuple(int(item) for item in raw_indices),
        device=str(options.get("device") or device),
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        attention_backend=str(options.get("attention_backend") or "auto"),
        num_inference_steps=(option_int(options.get("num_inference_steps"), 4) if options.get("num_inference_steps") is not None else None),
        tokenizer_max_length=(option_int(options.get("tokenizer_max_length"), 256) if options.get("tokenizer_max_length") is not None else None),
        seed=option_int(options.get("seed"), 0),
        device_map=options.get("device_map"),
        device_ids=tuple(int(item) for item in raw_devices),
        allow_cpu_offload=option_bool(options.get("allow_cpu_offload"), False),
        spatial_device=str(options["spatial_device"]) if options.get("spatial_device") else None,
        compile_action_expert=option_bool(options.get("compile_action_expert"), False),
        compile_mode=str(options["compile_mode"]) if options.get("compile_mode") else None,
    )
    if config.action_layout not in {"auto", "identity", "dual_arm_14"}:
        raise ValueError(f"unsupported ABot-M0 action_layout {config.action_layout!r}")
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = ABotM0Runtime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["ABotM0Runtime", "ABotM0RuntimeConfig", "MODEL_ID", "predict_action"]

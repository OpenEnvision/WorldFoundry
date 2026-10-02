"""Local-only checkpoint loader and action inference runtime for A1."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)

from .configuration import A1Config
from .preprocessing import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    A1Processor,
    load_local_tokenizer,
)

log = logging.getLogger(__name__)


def _strip_uniform_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(state)
    for prefix in ("module.", "model.", "_orig_mod.", "_fsdp_wrapped_module."):
        if output and all(str(key).startswith(prefix) for key in output):
            output = {str(key)[len(prefix) :]: value for key, value in output.items()}
    return output


def _unwrap_state(value: Any) -> dict[str, torch.Tensor]:
    while isinstance(value, Mapping):
        tensor_items = {str(key): item for key, item in value.items() if isinstance(item, torch.Tensor)}
        if tensor_items and len(tensor_items) == len(value):
            return _strip_uniform_prefix(tensor_items)
        nested = None
        for key in ("model_state_dict", "state_dict", "model", "module"):
            if isinstance(value.get(key), Mapping):
                nested = value[key]
                break
        if nested is None:
            if tensor_items:
                return _strip_uniform_prefix(tensor_items)
            break
        value = nested
    raise TypeError("A1 checkpoint does not contain a tensor state dictionary")


def _safetensor_files(root: Path) -> list[Path]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") or {}
        return [root / name for name in sorted(set(weight_map.values()))]
    direct = root / "model.safetensors"
    if direct.is_file():
        return [direct]
    return sorted(root.glob("*.safetensors"))


def checkpoint_shapes(root: str | Path) -> dict[str, tuple[int, ...]]:
    directory = Path(root).expanduser().resolve()
    safe_files = _safetensor_files(directory)
    if safe_files:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("A1 safetensors checkpoints require safetensors") from error
        shapes: dict[str, tuple[int, ...]] = {}
        for path in safe_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                shapes.update({key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()})
        normalized = _strip_uniform_prefix(shapes)
        return {key: tuple(value) for key, value in normalized.items()}

    model_file = directory / "model.pt"
    if not model_file.is_file():
        candidates = sorted(directory.glob("*.pt")) + sorted(directory.glob("*.pth"))
        if len(candidates) == 1:
            model_file = candidates[0]
        else:
            raise FileNotFoundError(
                f"A1 checkpoint requires model.pt or model.safetensors under {directory}"
            )
    from worldfoundry.core.model_loading.checkpoints.file import load_torch_checkpoint

    try:
        payload = load_torch_checkpoint(model_file, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        payload = load_torch_checkpoint(model_file, map_location="cpu", weights_only=True)
    state = _unwrap_state(payload)
    return {key: tuple(value.shape) for key, value in state.items()}


def _load_checkpoint(model: torch.nn.Module, root: Path, *, strict: bool) -> None:
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    safe_files = _safetensor_files(root)
    if safe_files:
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("A1 safetensors checkpoints require safetensors") from error
        for path in safe_files:
            shard = _strip_uniform_prefix(load_file(str(path), device="cpu"))
            result = model.load_state_dict(shard, strict=False, assign=True)
            loaded.update(key for key in shard if key in expected)
            unexpected.update(result.unexpected_keys)
            del shard
    else:
        model_file = root / "model.pt"
        if not model_file.is_file():
            candidates = sorted(root.glob("*.pt")) + sorted(root.glob("*.pth"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"No unambiguous A1 PyTorch checkpoint under {root}")
            model_file = candidates[0]
        from worldfoundry.core.model_loading.checkpoints.file import load_torch_checkpoint

        try:
            payload = load_torch_checkpoint(
                model_file, map_location="cpu", weights_only=True, mmap=True
            )
        except (TypeError, RuntimeError):
            payload = load_torch_checkpoint(model_file, map_location="cpu", weights_only=True)
        state = _unwrap_state(payload)
        result = model.load_state_dict(state, strict=False, assign=True)
        loaded.update(key for key in state if key in expected)
        unexpected.update(result.unexpected_keys)
        del state, payload

    missing = expected - loaded
    if strict and (missing or unexpected):
        missing_preview = sorted(missing)[:24]
        unexpected_preview = sorted(unexpected)[:24]
        raise RuntimeError(
            "A1 checkpoint architecture mismatch; "
            f"missing={missing_preview} ({len(missing)} total), "
            f"unexpected={unexpected_preview} ({len(unexpected)} total)"
        )
    if missing or unexpected:
        log.warning(
            "A1 non-strict checkpoint load: %d missing and %d unexpected tensors",
            len(missing),
            len(unexpected),
        )


def _load_data_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"A1 statistics root must be a mapping: {path}")
    return dict(payload)


def _nested_stats(payload: Mapping[str, Any], key: str | None) -> dict[str, Any]:
    selected: Any = payload
    if key:
        for part in str(key).split("."):
            if not isinstance(selected, Mapping) or part not in selected:
                raise KeyError(f"A1 statistics key {key!r} was not found")
            selected = selected[part]
    for wrapper in ("norm_stats", "statistics", "stats"):
        if isinstance(selected, Mapping) and isinstance(selected.get(wrapper), Mapping):
            selected = selected[wrapper]
    if not isinstance(selected, Mapping):
        raise TypeError("A1 selected statistics must be a mapping")
    return dict(selected)


def _feature_stats(stats: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any] | None:
    for name in names:
        value = stats.get(name)
        if isinstance(value, Mapping):
            return dict(value)
    observation = stats.get("observation")
    if isinstance(observation, Mapping):
        for name in names:
            value = observation.get(name)
            if isinstance(value, Mapping):
                return dict(value)
    return None


def _normalize(values: np.ndarray, stats: Mapping[str, Any] | None, mode: str) -> np.ndarray:
    if stats is None or mode == "none":
        return values.astype(np.float32, copy=True)
    source = values.astype(np.float32, copy=False)
    if mode in {"bounds_q99", "q99"}:
        low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    elif mode in {"bounds", "minmax"}:
        low, high = np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)
    elif mode in {"normal", "mean_std"}:
        mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        width = min(source.shape[-1], mean.shape[-1], std.shape[-1])
        output = source.copy()
        mask = np.asarray(stats.get("mask", np.ones(width, bool)), dtype=bool)[:width]
        output[..., :width] = np.where(
            mask,
            (source[..., :width] - mean[:width]) / (std[:width] + 1e-8),
            source[..., :width],
        )
        return output
    else:
        raise ValueError(f"Unsupported A1 normalization mode: {mode!r}")
    width = min(source.shape[-1], low.shape[-1], high.shape[-1])
    output = source.copy()
    mask = np.asarray(stats.get("mask", np.ones(width, bool)), dtype=bool)[:width]
    normalized = 2.0 * (source[..., :width] - low[:width]) / (high[:width] - low[:width] + 1e-8) - 1.0
    output[..., :width] = np.where(mask, normalized, source[..., :width])
    output[..., :width] = np.clip(output[..., :width], -1.0, 1.0)
    return output


def _unnormalize(values: np.ndarray, stats: Mapping[str, Any] | None, mode: str) -> np.ndarray:
    if stats is None or mode == "none":
        return values.astype(np.float32, copy=True)
    source = values.astype(np.float32, copy=False)
    if mode in {"bounds_q99", "q99"}:
        low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
        transform = lambda item, index: (item + 1.0) * 0.5 * (high[index] - low[index]) + low[index]
    elif mode in {"bounds", "minmax"}:
        low, high = np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)
        transform = lambda item, index: (item + 1.0) * 0.5 * (high[index] - low[index]) + low[index]
    elif mode in {"normal", "mean_std"}:
        mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        transform = lambda item, index: item * std[index] + mean[index]
        low = high = mean
    else:
        raise ValueError(f"Unsupported A1 normalization mode: {mode!r}")
    width = min(source.shape[-1], len(low), len(high))
    output = source.copy()
    mask = np.asarray(stats.get("mask", np.ones(width, bool)), dtype=bool)[:width]
    restored = transform(source[..., :width], slice(0, width))
    output[..., :width] = np.where(mask, restored, source[..., :width])
    return output


def _signed_mask(parts: Sequence[int], width: int) -> np.ndarray:
    output: list[bool] = []
    for part in parts:
        output.extend([part > 0] * abs(int(part)))
    if len(output) < width:
        output.extend([False] * (width - len(output)))
    return np.asarray(output[:width], dtype=bool)


@dataclass(frozen=True)
class A1RuntimeConfig:
    checkpoint_location: str
    tokenizer_location: str | None = None
    device: str = "cuda"
    torch_dtype: str = "auto"
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    strict_checkpoint: bool = True
    seed: int = 0
    camera_keys: tuple[str, ...] = ("front_camera", "wrist_camera", "side_camera")
    statistics_file: str | None = None
    statistics_key: str | None = None
    normalization: str = "bounds_q99"
    allow_missing_statistics: bool = False
    output_action_dim: int | None = None
    action_mode: str = "delta"
    delta_mask: tuple[int, ...] = (6, -1, 6, -1)


class A1Runtime:
    def __init__(self, config: A1RuntimeConfig) -> None:
        self.runtime_config = config
        self.checkpoint_root = Path(config.checkpoint_location).expanduser().resolve()
        if not self.checkpoint_root.is_dir():
            raise FileNotFoundError(
                "A1 requires an hfd-staged local checkpoint directory: "
                f"{self.checkpoint_root}"
            )
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        self.device = torch.device(resolve_inference_device(config.device))
        self.dtype = resolve_inference_dtype(self.device, config.torch_dtype)
        shapes = checkpoint_shapes(self.checkpoint_root)
        self.model_config = A1Config.from_checkpoint(self.checkpoint_root).with_weight_shapes(shapes)

        tokenizer_root = self._resolve_tokenizer_root(config.tokenizer_location)
        tokenizer, special_ids = load_local_tokenizer(tokenizer_root, self.model_config)
        self.special_ids = special_ids
        self.processor = A1Processor(self.model_config, tokenizer, special_ids)
        self.statistics, self.state_stats, self.action_stats = self._load_statistics()

        from .modeling import AffordVLA

        if self.model_config.action_head == "flow_matching":
            # Transformers lazily imports Qwen2 (and torchao) on first access.
            # Import it before the meta-device context: torchao constructs CPU
            # lookup tables during import and cannot materialize meta tensors.
            from transformers import Qwen2ForCausalLM  # noqa: F401

        with torch.device("meta"):
            self.model = AffordVLA(self.model_config, device="meta")
        _load_checkpoint(self.model, self.checkpoint_root, strict=config.strict_checkpoint)
        # Transformers keeps RoPE frequencies as non-persistent buffers, so
        # they are intentionally absent from the checkpoint state dict.  A
        # meta-initialized Qwen2 expert therefore needs those deterministic
        # buffers rebuilt before the whole model can move to CUDA.
        qwen2 = getattr(self.model.action_head, "qwen2", None)
        rotary = getattr(getattr(qwen2, "model", None), "rotary_emb", None)
        if rotary is not None and any(buffer.is_meta for buffer in rotary.buffers()):
            qwen2.model.rotary_emb = type(rotary)(rotary.config, device="cpu")
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if config.compile_model:
            self.model.apply_compile(mode=config.compile_mode)

    def _resolve_tokenizer_root(self, configured: str | None) -> Path:
        if configured:
            from worldfoundry.core.io.paths import resolve_worldfoundry_path

            path = resolve_worldfoundry_path(configured).resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"A1 tokenizer directory does not exist: {path}")
            return path
        if any(
            (self.checkpoint_root / name).is_file()
            for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        ):
            return self.checkpoint_root
        raise FileNotFoundError(
            "A1 checkpoint does not bundle tokenizer assets; stage Qwen/Qwen2-7B with hfd "
            "and set tokenizer_path"
        )

    def _load_statistics(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        if self.runtime_config.normalization == "none":
            return None, None, None
        candidates = []
        if self.runtime_config.statistics_file:
            candidate = Path(self.runtime_config.statistics_file)
            if not candidate.is_absolute():
                candidate = self.checkpoint_root / candidate
            candidates.append(candidate)
        candidates.extend(
            self.checkpoint_root / name
            for name in (
                "dataset_statistics.json",
                "norm_stats.json",
                "statistics.json",
                "stats.json",
                "normalization.yaml",
            )
        )
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            if self.runtime_config.allow_missing_statistics:
                log.warning("A1 normalization statistics were not found; using checkpoint-space values")
                return None, None, None
            raise FileNotFoundError(
                "A1 normalization statistics were not found in the local checkpoint; "
                "provide statistics_file or explicitly set normalization=none"
            )
        selected = _nested_stats(
            _load_data_file(path), self.runtime_config.statistics_key
        )
        state_stats = _feature_stats(selected, ("state", "proprio", "robot_state"))
        action_stats = _feature_stats(selected, ("actions", "action"))
        if state_stats is None or action_stats is None:
            raise KeyError(
                f"A1 statistics {path} must contain state/proprio and action/actions entries"
            )
        return selected, state_stats, action_stats

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        instruction: str,
        images: Sequence[Any],
        state: Any,
    ) -> dict[str, Any]:
        raw_state = np.asarray(state, dtype=np.float32).reshape(-1)
        normalized_state = _normalize(
            raw_state, self.state_stats, self.runtime_config.normalization
        )
        prepared = self.processor.prepare(instruction, images, normalized_state)
        batch = {
            key: value.to(
                device=self.device,
                dtype=self.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in prepared.tensors.items()
        }
        generator = torch.Generator(device=self.device).manual_seed(self.runtime_config.seed)
        actions = self.model.predict_actions(
            batch,
            action_start_token_id=self.special_ids[ACTION_START_TOKEN],
            action_end_token_id=self.special_ids[ACTION_END_TOKEN],
            generator=generator,
        ).float().cpu().numpy()[0]
        output_dim = self.runtime_config.output_action_dim or self.model_config.action_dim
        output_dim = min(int(output_dim), actions.shape[-1])
        actions = actions[..., :output_dim]
        actions = _unnormalize(
            actions, self.action_stats, self.runtime_config.normalization
        )
        if self.runtime_config.action_mode == "delta":
            width = min(actions.shape[-1], raw_state.shape[-1])
            mask = _signed_mask(self.runtime_config.delta_mask, width)
            actions[..., :width] += np.where(mask, raw_state[:width], 0.0)
        elif self.runtime_config.action_mode != "absolute":
            raise ValueError(
                f"Unsupported A1 action_mode: {self.runtime_config.action_mode!r}"
            )
        return completed_action_result(
            model_id="a1",
            instruction=instruction,
            actions=actions.tolist(),
            raw_output=actions.tolist(),
            checkpoint_path=str(self.checkpoint_root),
            device=str(self.device),
            runtime="worldfoundry.a1.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "action_head": self.model_config.action_head,
                "camera_count": prepared.camera_count,
                "dtype": str(self.dtype),
                "normalization": self.runtime_config.normalization,
                "action_mode": self.runtime_config.action_mode,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], A1Runtime] = {}


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


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
    del action_context
    options = dict(runtime_options or {})
    checkpoint = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not checkpoint:
        raise ValueError("A1 requires a local checkpoint_path")
    camera_keys = tuple(
        str(item)
        for item in options.get("camera_keys")
        or ("front_camera", "wrist_camera", "side_camera")
    )
    images = collect_images(observation, image, camera_keys)
    if not images:
        raise ValueError("A1 requires one or more configured camera images")
    state = first_present(observation, "state", "proprio", "robot_state", "joint_state", "qpos")
    if state is None:
        raise ValueError("A1 requires state/proprio in the observation")

    config = A1RuntimeConfig(
        checkpoint_location=checkpoint,
        tokenizer_location=str(options.get("tokenizer_path") or "") or None,
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        compile_model=option_bool(options.get("compile_model"), False),
        compile_mode=str(options.get("compile_mode") or "max-autotune"),
        strict_checkpoint=option_bool(options.get("strict_checkpoint"), True),
        seed=option_int(options.get("seed"), 0),
        camera_keys=camera_keys,
        statistics_file=str(options.get("statistics_file") or "") or None,
        statistics_key=str(options.get("statistics_key") or "") or None,
        normalization=str(options.get("normalization") or "bounds_q99").lower(),
        allow_missing_statistics=option_bool(
            options.get("allow_missing_statistics"), False
        ),
        output_action_dim=(
            option_int(options.get("output_action_dim"), 0)
            if options.get("output_action_dim") not in (None, "")
            else None
        ),
        action_mode=str(options.get("action_mode") or "delta").lower(),
        delta_mask=tuple(int(item) for item in options.get("delta_mask") or (6, -1, 6, -1)),
    )
    key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = A1Runtime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, images=images, state=state)


__all__ = [
    "A1Runtime",
    "A1RuntimeConfig",
    "checkpoint_shapes",
    "clear_runtime_cache",
    "predict_action",
]

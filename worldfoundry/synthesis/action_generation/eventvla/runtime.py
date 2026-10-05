"""Local checkpoint runtime for EventVLA action and event prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    option_float,
    option_int,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class EventVLARuntimeConfig:
    checkpoint_location: str
    checkpoint_filename: str
    config_filename: str
    statistics_filename: str
    base_vlm_location: str
    device: str
    torch_dtype: str
    attention_implementation: str
    unnorm_key: str | None
    camera_keys: tuple[str, ...]
    temporal_absolute_indices: tuple[int, ...]
    temporal_delta_indices: tuple[int, ...]
    history_capacity: int
    action_token: str
    prompt_template: str
    temporal_role_text: str
    memory_role_text: str
    supported_memory_mode: str
    required_keyframe_image_position: str
    required_memory_write_policy: str
    supported_action_modes: tuple[str, ...]
    event_future_min_offset: int
    event_commit_threshold: float
    keyframe_nms_window: int
    keyframe_cooldown_steps: int
    max_keyframe_images: int
    use_image_role_text: bool


def _runtime_defaults() -> dict[str, Any]:
    return _load_yaml(resolve_data_path("models", "runtime", "configs", "vla_va_wam", "eventvla.yaml"))


def _checkpoint_root(
    location: str,
    *,
    checkpoint_filename: str,
    config_filename: str,
    statistics_filename: str,
) -> Path:
    direct = resolve_worldfoundry_path(location)
    if direct.is_file():
        if direct.parent.name in {"final_model", "checkpoints"}:
            return direct.parent.parent.resolve()
        return direct.parent.resolve()
    return resolve_local_hf_model_path(
        location,
        required_files=(checkpoint_filename, config_filename, statistics_filename),
    )


def _checkpoint_file(location: str, root: Path, filename: str) -> Path:
    direct = resolve_worldfoundry_path(location)
    if direct.is_file():
        if direct.suffix not in {".pt", ".safetensors"}:
            raise ValueError(f"unsupported EventVLA checkpoint file: {direct}")
        return direct.resolve()
    checkpoint = root / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EventVLA policy checkpoint is missing: {checkpoint}")
    return checkpoint.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"EventVLA config must be a mapping: {path}")
    return payload


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


_MISSING = object()


def _required_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value = _nested(mapping, *keys, default=_MISSING)
    if value is _MISSING:
        raise ValueError(f"EventVLA checkpoint config is missing {'.'.join(keys)}")
    return value


def _flatten_images(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        flattened: list[Any] = []
        for item in value.values():
            flattened.extend(_flatten_images(item))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if hasattr(value, "shape"):
            return [value]
        flattened = []
        for item in value:
            flattened.extend(_flatten_images(item))
        return flattened
    return [value]


class EventVLARuntime:
    """Persistent runtime with temporal anchors and delayed raw-image memory."""

    def __init__(self, config: EventVLARuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._checkpoint = ""
        self._device = ""
        self._dtype: Any = None
        self._attention = ""
        self._model_config: dict[str, Any] = {}
        self._statistics: dict[str, Any] = {}
        self._action_dim = 0
        self._action_horizon = 0
        self._action_mode = ""
        self._image_size = (0, 0)
        self._history: list[tuple[int, list[Any]]] = []
        self._memory: list[tuple[int, Any, float]] = []
        self._memory_mode = ""
        self._max_keyframe_images = 0
        self._pending_event: tuple[int, float] | None = None
        self._last_committed_step: int | None = None
        self._episode_id: str | None = None
        self._implicit_step = 0

    def reset(self, episode_id: Any = None) -> None:
        self._history.clear()
        self._memory.clear()
        self._pending_event = None
        self._last_committed_step = None
        self._episode_id = None if episode_id is None else str(episode_id)
        self._implicit_step = 0

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        import torch

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import restore_eventvla_policy

        root = _checkpoint_root(
            self.config.checkpoint_location,
            checkpoint_filename=self.config.checkpoint_filename,
            config_filename=self.config.config_filename,
            statistics_filename=self.config.statistics_filename,
        )
        checkpoint = _checkpoint_file(
            self.config.checkpoint_location,
            root,
            self.config.checkpoint_filename,
        )
        config_path = root / self.config.config_filename
        statistics_path = root / self.config.statistics_filename
        if not config_path.is_file() or not statistics_path.is_file():
            raise FileNotFoundError(
                f"EventVLA requires {self.config.config_filename} and "
                f"{self.config.statistics_filename} beside {checkpoint}"
            )
        model_config = _load_yaml(config_path)
        memory_config = _required_nested(model_config, "framework", "memory_buffer")
        if not isinstance(memory_config, Mapping):
            raise ValueError("EventVLA config framework.memory_buffer must be a mapping")
        if bool(_required_nested(model_config, "framework", "memory_buffer", "enable")):
            raise ValueError("EventVLA released checkpoints require model-side memory writes to be disabled")
        if not bool(
            _required_nested(
                model_config,
                "framework",
                "memory_buffer",
                "qwen_memory_injection",
                "enabled",
            )
        ):
            raise ValueError("EventVLA released checkpoints require raw keyframe-image injection")
        injection_mode = (
            str(
                _required_nested(
                    model_config,
                    "framework",
                    "memory_buffer",
                    "qwen_memory_injection",
                    "mode",
                )
            )
            .strip()
            .lower()
        )
        raw_memory_mode = _nested(model_config, "framework", "memory_ablation_mode")
        # The RMBench release predates ``memory_ablation_mode`` and records its
        # exact mode only in qwen_memory_injection.mode.  RoboTwin-MeM carries
        # both fields.  Accept those two released, checkpoint-authored modes
        # while still rejecting an ambiguous or inconsistent configuration.
        memory_mode = (
            injection_mode
            if raw_memory_mode is None
            else str(raw_memory_mode).strip().lower()
        )
        released_modes = {self.config.supported_memory_mode, "raw_anchors_only"}
        if memory_mode not in released_modes or injection_mode != memory_mode:
            raise ValueError(
                f"unsupported EventVLA memory mode {memory_mode!r}; "
                f"released modes={sorted(released_modes)}"
            )
        keyframe_position = (
            str(
                _required_nested(
                    model_config,
                    "framework",
                    "memory_buffer",
                    "qwen_memory_injection",
                    "keyframe_image_position",
                )
            )
            .strip()
            .lower()
        )
        if keyframe_position != self.config.required_keyframe_image_position:
            raise ValueError(
                f"unsupported EventVLA keyframe_image_position={keyframe_position!r}; "
                f"expected {self.config.required_keyframe_image_position!r}"
            )
        raw_write_policy = _nested(
            model_config,
            "framework",
            "memory_buffer",
            "memory_write_policy",
        )
        if raw_write_policy is None:
            legacy_keyframe_memory = bool(
                _required_nested(
                    model_config,
                    "datasets",
                    "vla_data",
                    "keyframe_image_memory",
                    "enabled",
                )
            )
            if memory_mode != "raw_anchors_only" or legacy_keyframe_memory:
                raise ValueError("EventVLA checkpoint config is missing framework.memory_buffer.memory_write_policy")
            memory_write_policy = "disabled"
        else:
            memory_write_policy = str(raw_write_policy).strip().lower()
        if memory_write_policy != self.config.required_memory_write_policy:
            raise ValueError(
                f"unsupported EventVLA memory_write_policy={memory_write_policy!r}; "
                f"expected {self.config.required_memory_write_policy!r}"
            )
        trained_max_keyframes = int(
            _required_nested(
                model_config,
                "framework",
                "memory_buffer",
                "qwen_memory_injection",
                "max_keyframe_images",
            )
        )
        if self.config.max_keyframe_images < 0 or self.config.max_keyframe_images > trained_max_keyframes:
            raise ValueError(
                f"max_keyframe_images must be in [0, {trained_max_keyframes}] for this EventVLA checkpoint"
            )
        trained_role_text = bool(
            _required_nested(
                model_config,
                "framework",
                "memory_buffer",
                "qwen_memory_injection",
                "use_image_role_text",
            )
        )
        if self.config.use_image_role_text != trained_role_text:
            raise ValueError("use_image_role_text must match the EventVLA checkpoint prompt configuration")

        action_config = _required_nested(model_config, "framework", "action_model")
        if not isinstance(action_config, Mapping):
            raise ValueError("EventVLA config is missing framework.action_model")
        action_dim = int(_required_nested(action_config, "action_dim"))
        past = int(_required_nested(action_config, "past_action_window_size"))
        future = int(_required_nested(action_config, "future_action_window_size"))
        action_horizon = int(_required_nested(action_config, "action_horizon"))
        if action_dim <= 0 or past < 0 or future < 0 or action_horizon <= 0:
            raise ValueError("EventVLA action dimensions and windows must be non-negative and non-empty")
        if action_horizon != past + 1 + future:
            raise ValueError("EventVLA checkpoint action-window fields are inconsistent")
        raw_image_size = _required_nested(model_config, "datasets", "vla_data", "image_size")
        if not isinstance(raw_image_size, Sequence) or len(raw_image_size) != 2:
            raise ValueError("EventVLA image_size must contain height and width")
        image_size = (int(raw_image_size[1]), int(raw_image_size[0]))
        action_mode = str(_required_nested(model_config, "datasets", "vla_data", "action_mode")).strip().lower()
        if action_mode not in self.config.supported_action_modes:
            raise ValueError(
                f"unsupported EventVLA action_mode={action_mode!r}; "
                f"supported={list(self.config.supported_action_modes)}"
            )
        temporal_image_config = _required_nested(
            model_config,
            "datasets",
            "vla_data",
            "temporal",
            "image",
        )
        if not isinstance(temporal_image_config, Mapping):
            raise ValueError("EventVLA datasets.vla_data.temporal.image must be a mapping")
        trained_absolute = tuple(int(value) for value in _required_nested(temporal_image_config, "absolute_indices"))
        trained_delta = tuple(int(value) for value in _required_nested(temporal_image_config, "delta_indices"))
        if trained_absolute != self.config.temporal_absolute_indices:
            raise ValueError(
                "temporal_absolute_indices must match the EventVLA checkpoint: "
                f"configured={self.config.temporal_absolute_indices}, trained={trained_absolute}"
            )
        if trained_delta != self.config.temporal_delta_indices:
            raise ValueError(
                "temporal_delta_indices must match the EventVLA checkpoint: "
                f"configured={self.config.temporal_delta_indices}, trained={trained_delta}"
            )

        base_vlm = resolve_local_hf_model_path(
            self.config.base_vlm_location,
            required_files=("config.json", "preprocessor_config.json"),
        )
        device_name = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device_name, self.config.torch_dtype)
        attention = resolve_transformers_attention_implementation(
            self.config.attention_implementation,
            device_name,
        )
        device = torch.device(device_name)
        effective_max_keyframes = self.config.max_keyframe_images if memory_mode == self.config.supported_memory_mode else 0
        model = restore_eventvla_policy(
            checkpoint,
            base_vlm=base_vlm,
            action_dim=action_dim,
            action_horizon=action_horizon,
            image_size=image_size,
            max_keyframe_images=effective_max_keyframes,
            use_image_role_text=self.config.use_image_role_text,
            action_token=self.config.action_token,
            prompt_template=self.config.prompt_template,
            temporal_role_text=self.config.temporal_role_text,
            memory_role_text=self.config.memory_role_text,
            device=device,
            dtype=dtype,
            attention_implementation=attention,
        )
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        if not isinstance(statistics, dict):
            raise TypeError("EventVLA dataset_statistics.json must contain an object")

        self._model = model
        self._checkpoint = str(checkpoint)
        self._device = device_name
        self._dtype = dtype
        self._attention = attention
        self._model_config = model_config
        self._statistics = statistics
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._action_mode = action_mode
        self._image_size = image_size
        self._memory_mode = memory_mode
        self._max_keyframe_images = effective_max_keyframes
        return model

    def _step_and_episode(self, observation: Mapping[str, Any]) -> int:
        episode = first_present(observation, "episode_id", "episode", "trajectory_id")
        if option_bool(observation.get("reset"), False):
            self.reset(episode)
        elif episode is not None and str(episode) != self._episode_id:
            self.reset(episode)
        raw_step = first_present(observation, "timestep", "step", "frame_index")
        step = int(raw_step) if raw_step is not None else self._implicit_step
        self._implicit_step = step + self._action_horizon
        return step

    def _current_views(self, observation: Mapping[str, Any], image: Any) -> list[Any]:
        views = collect_images(observation, image, self.config.camera_keys)
        if not views:
            raise ValueError("EventVLA requires at least one current RGB camera view")
        return views

    def _record_history(self, step: int, views: list[Any]) -> None:
        if self._history and self._history[-1][0] == step:
            self._history[-1] = (step, list(views))
        else:
            self._history.append((step, list(views)))
        self._history = self._history[-self.config.history_capacity :]

    def _views_at_or_before(self, target: int) -> list[Any]:
        for step, views in reversed(self._history):
            if step <= target:
                return list(views)
        return list(self._history[0][1])

    def _temporal_anchors(
        self,
        observation: Mapping[str, Any],
        step: int,
    ) -> list[Any]:
        explicit = first_present(observation, "temporal_images", "anchor_images")
        if explicit is not None:
            anchors = _flatten_images(explicit)
            if anchors:
                return anchors
        anchors: list[Any] = []
        first_step = self._history[0][0]
        for absolute_index in self.config.temporal_absolute_indices:
            anchors.extend(self._views_at_or_before(first_step + absolute_index))
        for delta in self.config.temporal_delta_indices:
            anchors.extend(self._views_at_or_before(step + delta))
        return anchors

    def _commit_matured_event(self, step: int, current_views: list[Any]) -> bool:
        if self._pending_event is None or step < self._pending_event[0]:
            return False
        target, confidence = self._pending_event
        self._pending_event = None
        if (
            self._last_committed_step is not None
            and abs(target - self._last_committed_step) <= self.config.keyframe_cooldown_steps
        ):
            return False
        if self._max_keyframe_images > 0:
            self._memory.append((target, current_views[0], confidence))
            self._memory = self._memory[-self._max_keyframe_images :]
        else:
            self._memory.clear()
        self._last_committed_step = target
        return True

    def _memory_images(self, observation: Mapping[str, Any]) -> list[Any]:
        explicit = first_present(
            observation,
            "memory_keyframe_images",
            "keyframe_images",
        )
        if explicit is not None:
            if self._max_keyframe_images <= 0:
                return []
            return _flatten_images(explicit)[-self._max_keyframe_images :]
        return [image for _step, image, _confidence in self._memory]

    def _select_event(self, probabilities: Any, step: int) -> dict[str, Any]:
        import numpy as np

        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        start = min(max(0, self.config.event_future_min_offset), len(values))
        candidates = [
            (index, float(values[index]))
            for index in range(start, len(values))
            if float(values[index]) >= self.config.event_commit_threshold
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        suppressed: set[int] = set()
        nms_candidates: list[tuple[int, float]] = []
        for offset, confidence in candidates:
            if offset in suppressed:
                continue
            nms_candidates.append((offset, confidence))
            begin = max(start, offset - self.config.keyframe_nms_window)
            end = min(len(values), offset + self.config.keyframe_nms_window + 1)
            suppressed.update(range(begin, end))

        raw_offset, raw_confidence = (-1, 0.0)
        if start < len(values):
            raw_offset = int(start + values[start:].argmax())
            raw_confidence = float(values[raw_offset])
        selected_offset, selected_confidence = (-1, 0.0)
        for offset, confidence in nms_candidates:
            candidate_step = step + offset
            if (
                self._last_committed_step is not None
                and abs(candidate_step - self._last_committed_step) <= self.config.keyframe_cooldown_steps
            ):
                continue
            if self._pending_event is not None and (
                abs(candidate_step - self._pending_event[0]) <= self.config.keyframe_cooldown_steps
            ):
                pending_step, pending_confidence = self._pending_event
                replaces_pending = confidence > pending_confidence or (
                    abs(confidence - pending_confidence) <= 1e-8 and candidate_step >= pending_step
                )
                if not replaces_pending:
                    continue
            selected_offset, selected_confidence = offset, confidence
            self._pending_event = (candidate_step, confidence)
            break
        return {
            "chunk_keyframe_prob": values.tolist(),
            "pred_event_offset": selected_offset,
            "pred_event_confidence": selected_confidence,
            "should_trigger_event": selected_offset >= 0,
            "raw_pred_event_offset": raw_offset,
            "raw_pred_event_confidence": raw_confidence,
            "raw_should_trigger_event": raw_confidence >= self.config.event_commit_threshold,
            "keyframe_event_suppressed_by_filter": raw_confidence >= self.config.event_commit_threshold
            and raw_offset != selected_offset,
        }

    def _unnormalize(self, normalized_actions: Any) -> tuple[Any, str | None]:
        import numpy as np

        from worldfoundry.core.action_normalization import unnormalize_action_values

        actions = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1.0, 1.0)
        if actions.shape[-1] != self._action_dim:
            raise RuntimeError(f"EventVLA returned action width {actions.shape[-1]}, expected {self._action_dim}")
        available = tuple(self._statistics)
        selected_key = self.config.unnorm_key
        if selected_key is None:
            if len(available) != 1:
                raise ValueError(f"EventVLA has multiple normalization keys {list(available)}; select unnorm_key")
            selected_key = str(available[0])
        if selected_key not in self._statistics:
            raise KeyError(f"EventVLA normalization key {selected_key!r} is unavailable; choices={list(available)}")
        statistics: Any = self._statistics[selected_key]
        if isinstance(statistics, Mapping) and self._action_mode in statistics:
            statistics = statistics[self._action_mode]
        if isinstance(statistics, Mapping) and isinstance(statistics.get("action"), Mapping):
            statistics = statistics["action"]
        if not isinstance(statistics, Mapping):
            raise TypeError("EventVLA action normalization statistics must be a mapping")
        mode = "q99" if "q01" in statistics and "q99" in statistics else "min_max"
        return unnormalize_action_values(actions, statistics, mode=mode), selected_key

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np

        model = self._load()
        step = self._step_and_episode(observation)
        current_views = self._current_views(observation, image)
        self._record_history(step, current_views)
        memory_committed = self._commit_matured_event(step, current_views)
        anchors = self._temporal_anchors(observation, step)
        memory_images = self._memory_images(observation)
        outputs = model.predict_action(
            anchor_images=anchors,
            memory_images=memory_images,
            instruction=instruction,
        )
        normalized = outputs["normalized_actions"].detach().float().cpu().numpy()[0]
        probabilities = outputs["chunk_keyframe_prob"].detach().float().cpu().numpy()[0]
        actions, selected_key = self._unnormalize(normalized)
        event = self._select_event(probabilities, step)
        if actions.shape != (self._action_horizon, self._action_dim):
            raise RuntimeError(f"EventVLA returned unexpected action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise FloatingPointError("EventVLA produced non-finite actions")
        return completed_action_result(
            model_id="eventvla",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=self._checkpoint,
            device=self._device,
            runtime="worldfoundry.eventvla.in_tree_runtime",
            raw_output={
                "normalized_actions": normalized.tolist(),
                **event,
            },
            metadata={
                "action_shape": list(actions.shape),
                "temporal_anchor_images": len(anchors),
                "memory_keyframe_images": len(memory_images),
                "memory_committed": memory_committed,
                "timestep": step,
                "normalization_key": selected_key,
                "attention_implementation": self._attention,
                "dtype": str(self._dtype),
                **{
                    key: event[key]
                    for key in (
                        "pred_event_offset",
                        "pred_event_confidence",
                        "should_trigger_event",
                    )
                },
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], EventVLARuntime] = {}


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
    """WorldFoundry callable policy entrypoint."""

    del action_context
    options = {**_runtime_defaults(), **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    base_vlm = str(options.get("base_vlm_path") or options.get("base_vlm") or "")
    if not checkpoint:
        raise ValueError("EventVLA requires a local checkpoint_path")
    if not base_vlm:
        raise ValueError("EventVLA requires a staged local base_vlm_path")
    config = EventVLARuntimeConfig(
        checkpoint_location=checkpoint,
        checkpoint_filename=str(options["checkpoint_filename"]),
        config_filename=str(options["config_filename"]),
        statistics_filename=str(options["statistics_filename"]),
        base_vlm_location=base_vlm,
        device=str(options.get("device") or device),
        torch_dtype=str(options["torch_dtype"]),
        attention_implementation=str(options.get("attn_implementation") or options["attention_implementation"]),
        unnorm_key=str(options["unnorm_key"]) if options.get("unnorm_key") else None,
        camera_keys=tuple(str(value) for value in options["camera_keys"]),
        temporal_absolute_indices=tuple(int(value) for value in options["temporal_absolute_indices"]),
        temporal_delta_indices=tuple(int(value) for value in options["temporal_delta_indices"]),
        history_capacity=option_int(options["history_capacity"], 0),
        action_token=str(options["action_token"]),
        prompt_template=str(options["prompt_template"]),
        temporal_role_text=str(options["temporal_role_text"]),
        memory_role_text=str(options["memory_role_text"]),
        supported_memory_mode=str(options["supported_memory_mode"]).strip().lower(),
        required_keyframe_image_position=str(options["required_keyframe_image_position"]).strip().lower(),
        required_memory_write_policy=str(options["required_memory_write_policy"]).strip().lower(),
        supported_action_modes=tuple(str(value).strip().lower() for value in options["supported_action_modes"]),
        event_future_min_offset=option_int(options["event_future_min_offset"], 0),
        event_commit_threshold=option_float(options["event_commit_threshold"], 0.0),
        keyframe_nms_window=option_int(options["keyframe_nms_window"], 0),
        keyframe_cooldown_steps=option_int(options["keyframe_cooldown_steps"], 0),
        max_keyframe_images=option_int(options["max_keyframe_images"], 0),
        use_image_role_text=option_bool(options["use_image_role_text"], False),
    )
    if not config.camera_keys:
        raise ValueError("EventVLA camera_keys must not be empty")
    if not config.temporal_absolute_indices and not config.temporal_delta_indices:
        raise ValueError("EventVLA requires at least one temporal image index")
    if config.history_capacity <= 0:
        raise ValueError("EventVLA history_capacity must be positive")
    if "{instruction}" not in config.prompt_template or "{action_tokens}" not in config.prompt_template:
        raise ValueError("EventVLA prompt_template must contain {instruction} and {action_tokens}")
    if not config.action_token:
        raise ValueError("EventVLA action_token must not be empty")
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = EventVLARuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["EventVLARuntime", "EventVLARuntimeConfig", "predict_action"]

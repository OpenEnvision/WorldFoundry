"""Local-only inference runtime for the Mem-0 execution policy."""

from __future__ import annotations

import bisect
import gc
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class Mem0RuntimeConfig:
    checkpoint_location: str
    backbone_location: str
    device: str
    torch_dtype: str
    attention_backend: str
    checkpoint_file: str
    checkpoint_parts_glob: str
    stats_file: str
    strict_checkpoint: bool
    seed: int
    tokenizer_max_length: int
    image_size: tuple[int, int]
    state_keys: tuple[str, ...]
    state_value_keys: tuple[str, ...]
    camera_aliases: tuple[str, ...]
    image_value_keys: tuple[str, ...]
    reset_keys: tuple[str, ...]
    state_layout: str
    model_state_dim: int
    short_state_mapping: tuple[int, ...]
    state_input_indices: tuple[int, ...]
    action_output_indices: tuple[int, ...]
    normalization_mode: str
    normalization_dims: int
    memory_config: Mapping[str, Any]
    action_config: Mapping[str, Any]
    classifier_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tokenizer_max_length < 1:
            raise ValueError("Mem-0 tokenizer_max_length must be positive")
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("Mem-0 image_size must contain two positive values")
        if self.state_layout not in {"environment", "model"}:
            raise ValueError("Mem-0 state_layout must be 'environment' or 'model'")
        if len(self.state_input_indices) != self.model_state_dim:
            raise ValueError("Mem-0 state_input_indices must match model_state_dim")
        action_dim = int(self.action_config["action_dim"])
        if not self.action_output_indices:
            raise ValueError("Mem-0 action_output_indices cannot be empty")
        if len(set(self.action_output_indices)) != len(self.action_output_indices):
            raise ValueError("Mem-0 action_output_indices must be unique")
        if min(self.action_output_indices) < 0 or max(self.action_output_indices) >= action_dim:
            raise ValueError("Mem-0 action_output_indices are outside the model action vector")
        if self.normalization_dims < 1 or self.normalization_dims > min(
            self.model_state_dim,
            action_dim,
        ):
            raise ValueError("Mem-0 normalization_dims is outside the state/action vectors")


class _SplitCheckpointReader(io.RawIOBase):
    """Seekable view over byte-split checkpoint parts without a second 15 GB copy."""

    def __init__(self, parts: Sequence[Path]) -> None:
        super().__init__()
        self._handles = [part.open("rb") for part in parts]
        self._sizes = [part.stat().st_size for part in parts]
        self._offsets = [0]
        for size in self._sizes:
            self._offsets.append(self._offsets[-1] + size)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._offsets[-1] + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise OSError("Cannot seek before the start of a split checkpoint")
        self._position = min(position, self._offsets[-1])
        return self._position

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        total_read = 0
        while total_read < len(view) and self._position < self._offsets[-1]:
            part_index = bisect.bisect_right(self._offsets, self._position) - 1
            part_offset = self._position - self._offsets[part_index]
            remaining = self._sizes[part_index] - part_offset
            count = min(len(view) - total_read, remaining)
            handle = self._handles[part_index]
            handle.seek(part_offset)
            read = handle.readinto(view[total_read : total_read + count])
            if not read:
                break
            total_read += read
            self._position += read
        return total_read

    def close(self) -> None:
        for handle in self._handles:
            handle.close()
        super().close()


def _required_option(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"Mem-0 runtime config requires {key!r}")
    return value


def _string_tuple(value: Any, *, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Mem-0 {key} must be a sequence")
    return tuple(str(item) for item in value)


def _int_tuple(value: Any, *, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Mem-0 {key} must be a sequence")
    return tuple(int(item) for item in value)


def _mapping(value: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Mem-0 {key} must be a mapping")
    return dict(value)


def _torch_load_checkpoint(root: Path, config: Mem0RuntimeConfig) -> tuple[Any, str]:
    import numpy as np
    import torch

    # The publisher's training payload stores the scheduler learning rate as
    # a NumPy float64 scalar.  Keep weights_only=True and allow only the three
    # inert NumPy constructors that are statically present in data.pkl.
    safe_numpy_globals = (
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    )
    checkpoint = root / config.checkpoint_file
    if checkpoint.is_file():
        with torch.serialization.safe_globals(safe_numpy_globals):
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        return payload, str(checkpoint)
    parts = sorted(root.glob(config.checkpoint_parts_glob))
    if not parts:
        raise FileNotFoundError(
            f"Mem-0 checkpoint is missing {checkpoint} and split parts "
            f"matching {config.checkpoint_parts_glob!r}"
        )
    expected = [f".part{index:02d}" for index in range(len(parts))]
    suffixes = [part.name[-7:] for part in parts]
    if suffixes != expected:
        raise ValueError(f"Mem-0 checkpoint parts are not contiguous: {suffixes}")
    raw = _SplitCheckpointReader(parts)
    try:
        with io.BufferedReader(raw, buffer_size=8 * 1024 * 1024) as reader:
            with torch.serialization.safe_globals(safe_numpy_globals):
                payload = torch.load(reader, map_location="cpu", weights_only=True)
    finally:
        raw.close()
    return payload, f"{parts[0]}..{parts[-1].name}"


class Mem0Runtime:
    """Persistent episodic policy runtime."""

    def __init__(self, config: Mem0RuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._stats: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._checkpoint_root: Path | None = None
        self._checkpoint_source: str | None = None
        self._backbone_path: str | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        import torch

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import Mem0Policy
        from .normalization import load_stats

        device_name = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device_name, self.config.torch_dtype)
        device = torch.device(device_name)
        checkpoint_root = resolve_local_hf_model_path(
            self.config.checkpoint_location,
            required_files=("README.md",),
        )
        backbone = str(
            resolve_local_hf_model_path(
                self.config.backbone_location,
                required_files=("config.json", "preprocessor_config.json"),
            )
        )
        stats_path = checkpoint_root / self.config.stats_file
        if not stats_path.is_file():
            raise FileNotFoundError(f"Mem-0 checkpoint statistics are missing: {stats_path}")
        attention = resolve_transformers_attention_implementation(
            self.config.attention_backend,
            device_name,
        )
        model = Mem0Policy(
            backbone_path=backbone,
            device=device,
            dtype=dtype,
            attention_implementation=attention,
            memory_config=self.config.memory_config,
            action_config=self.config.action_config,
            classifier_config=self.config.classifier_config,
            tokenizer_max_length=self.config.tokenizer_max_length,
        ).eval()
        payload, checkpoint_source = _torch_load_checkpoint(checkpoint_root, self.config)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(state_dict, Mapping):
            raise TypeError("Mem-0 checkpoint does not contain a model state dictionary")
        if not state_dict or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise TypeError("Mem-0 checkpoint state dictionary must contain tensor values only")
        model.load_state_dict(state_dict, strict=self.config.strict_checkpoint)
        del state_dict, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self._model = model
        self._stats = load_stats(stats_path)
        self._device = device_name
        self._dtype = dtype
        self._checkpoint_root = checkpoint_root
        self._checkpoint_source = checkpoint_source
        self._backbone_path = backbone
        return model

    def _state(self, observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, *self.config.state_keys)
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(nested, *self.config.state_keys)
        if isinstance(state, Mapping):
            state = first_present(state, *self.config.state_value_keys)
        return state

    def _image(self, observation: Mapping[str, Any], image: Any) -> Any:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Mapping):
                return first_present(value, *self.config.image_value_keys)
            return value

        images = observation.get("images")
        if isinstance(images, Mapping):
            selected = first_present(images, *self.config.camera_aliases)
            if selected is not None:
                return unwrap(selected)
        selected = first_present(observation, *self.config.camera_aliases)
        if selected is not None:
            return unwrap(selected)
        nested = observation.get("observation")
        if isinstance(nested, Mapping):
            nested_images = nested.get("images")
            if isinstance(nested_images, Mapping):
                selected = first_present(nested_images, *self.config.camera_aliases)
            else:
                selected = first_present(nested, *self.config.camera_aliases)
            if selected is not None:
                return unwrap(selected)
        if isinstance(image, Mapping):
            selected = first_present(image, *self.config.camera_aliases)
            return unwrap(selected if selected is not None else image)
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)):
            return image[0] if image else None
        return image

    def _model_state(self, state: Any, observation: Mapping[str, Any]) -> Any:
        import numpy as np
        import torch

        from .normalization import normalize

        if isinstance(state, torch.Tensor):
            values = state.detach().to(dtype=torch.float32, device="cpu").numpy().reshape(-1)
        else:
            values = np.asarray(state, dtype=np.float32).reshape(-1)
        layout = str(observation.get("state_layout", self.config.state_layout)).lower()
        if layout == "model":
            if values.size != self.config.model_state_dim:
                raise ValueError(
                    f"Mem-0 model-layout state requires {self.config.model_state_dim} values"
                )
            model_values = values
        else:
            if values.size == self.config.model_state_dim:
                environment_values = values
            elif values.size == len(self.config.short_state_mapping):
                environment_values = np.zeros(self.config.model_state_dim, dtype=np.float32)
                environment_values[list(self.config.short_state_mapping)] = values
            else:
                raise ValueError(
                    f"Mem-0 state requires {self.config.model_state_dim} values or "
                    f"the configured {len(self.config.short_state_mapping)}-value compact layout"
                )
            model_values = environment_values[list(self.config.state_input_indices)]
        model_values = normalize(
            model_values,
            prefix="state",
            mode=self.config.normalization_mode,
            dimensions=self.config.normalization_dims,
            stats=self._stats,
            inverse=False,
        )
        return torch.from_numpy(model_values).view(1, 1, -1).to(
            device=self._device,
            dtype=self._dtype,
        )

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch

        from worldfoundry.core.media.processing.image_utils import load_pil_image
        from worldfoundry.core.utils.tensors.torch import set_seed_everywhere

        from .normalization import normalize

        model = self._load()
        episode_id = observation.get("episode_id", 0)
        if any(bool(observation.get(key, False)) for key in self.config.reset_keys):
            model.reset(episode_id)
        state = self._state(observation)
        if state is None:
            raise ValueError("Mem-0 requires a robot state vector")
        state_tensor = self._model_state(state, observation)
        image_value = self._image(observation, image)
        if image_value is None:
            raise ValueError("Mem-0 requires the configured head-camera image")
        pil_image = load_pil_image(image_value, first_sequence_item=False).convert("RGB")
        pil_image = pil_image.resize(self.config.image_size)
        set_seed_everywhere(self.config.seed)
        with torch.inference_mode():
            normalized_actions, subtask_ended = model.predict_action(
                image=pil_image,
                instruction=instruction,
                state=state_tensor,
                episode_id=episode_id,
            )
        action_values = normalized_actions.to(dtype=torch.float32).cpu().numpy()
        action_values = normalize(
            action_values,
            prefix="action",
            mode=self.config.normalization_mode,
            dimensions=self.config.normalization_dims,
            stats=self._stats,
            inverse=True,
        )
        action_values = action_values[..., list(self.config.action_output_indices)]
        actions = torch.from_numpy(np.asarray(action_values, dtype=np.float32))
        return completed_action_result(
            model_id="mem-0",
            instruction=instruction,
            actions=actions,
            checkpoint_path=str(self._checkpoint_root),
            device=str(self._device),
            runtime="worldfoundry.mem0.in_tree",
            metadata={
                "backbone_path": self._backbone_path,
                "checkpoint_source": self._checkpoint_source,
                "episode_id": episode_id,
                "memory_size": model.memory_bank.get_memory_size(episode_id),
                "subtask_ended": subtask_ended,
                "seed": self.config.seed,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str, str], Mem0Runtime] = {}


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
    """Run the official M1 execution checkpoint through the shared contract."""

    del action_context
    options = dict(runtime_options or {})
    config = Mem0RuntimeConfig(
        checkpoint_location=checkpoint_path,
        backbone_location=str(_required_option(options, "backbone_path")),
        device=device,
        torch_dtype=str(_required_option(options, "torch_dtype")),
        attention_backend=str(_required_option(options, "attention_backend")),
        checkpoint_file=str(_required_option(options, "checkpoint_file")),
        checkpoint_parts_glob=str(_required_option(options, "checkpoint_parts_glob")),
        stats_file=str(_required_option(options, "stats_file")),
        strict_checkpoint=option_bool(_required_option(options, "strict_checkpoint")),
        seed=option_int(_required_option(options, "seed"), 0),
        tokenizer_max_length=option_int(_required_option(options, "tokenizer_max_length"), 0),
        image_size=_int_tuple(_required_option(options, "image_size"), key="image_size"),
        state_keys=_string_tuple(_required_option(options, "state_keys"), key="state_keys"),
        state_value_keys=_string_tuple(
            _required_option(options, "state_value_keys"),
            key="state_value_keys",
        ),
        camera_aliases=_string_tuple(
            _required_option(options, "camera_aliases"),
            key="camera_aliases",
        ),
        image_value_keys=_string_tuple(
            _required_option(options, "image_value_keys"),
            key="image_value_keys",
        ),
        reset_keys=_string_tuple(_required_option(options, "reset_keys"), key="reset_keys"),
        state_layout=str(_required_option(options, "state_layout")),
        model_state_dim=option_int(_required_option(options, "model_state_dim"), 0),
        short_state_mapping=_int_tuple(
            _required_option(options, "short_state_mapping"),
            key="short_state_mapping",
        ),
        state_input_indices=_int_tuple(
            _required_option(options, "state_input_indices"),
            key="state_input_indices",
        ),
        action_output_indices=_int_tuple(
            _required_option(options, "action_output_indices"),
            key="action_output_indices",
        ),
        normalization_mode=str(_required_option(options, "normalization_mode")),
        normalization_dims=option_int(_required_option(options, "normalization_dims"), 0),
        memory_config=_mapping(_required_option(options, "memory_config"), key="memory_config"),
        action_config=_mapping(_required_option(options, "action_config"), key="action_config"),
        classifier_config=_mapping(
            _required_option(options, "classifier_config"),
            key="classifier_config",
        ),
    )
    cache_key = (checkpoint_path, device, runtime_options_cache_key(options))
    runtime = _RUNTIME_CACHE.setdefault(cache_key, Mem0Runtime(config))
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["Mem0Runtime", "Mem0RuntimeConfig", "predict_action"]

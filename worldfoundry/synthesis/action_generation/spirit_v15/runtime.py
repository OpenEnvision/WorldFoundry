"""Native inference runtime for Spirit-v1.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_int,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class SpiritV15RuntimeConfig:
    checkpoint_location: str
    backbone_location: str
    device: str
    torch_dtype: str
    attention_backend: str
    robot_type: str
    seed: int
    state_keys: tuple[str, ...]
    camera_keys: tuple[str, ...]
    camera_aliases: tuple[tuple[str, ...], ...]
    state_mask_indices: tuple[int, ...]
    user_prompt_template: str

    def __post_init__(self) -> None:
        if not self.state_keys:
            raise ValueError("Spirit-v1.5 state_keys cannot be empty")
        if not self.camera_keys or len(self.camera_keys) != len(self.camera_aliases):
            raise ValueError("Spirit-v1.5 camera_keys must match camera_aliases")
        if not self.user_prompt_template:
            raise ValueError("Spirit-v1.5 user_prompt_template cannot be empty")


class SpiritV15Runtime:
    """Persistent official-checkpoint Spirit-v1.5 policy runtime."""

    def __init__(self, config: SpiritV15RuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._checkpoint_location: str | None = None
        self._backbone_location: str | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import SpiritVLAPolicy

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        checkpoint = str(
            resolve_local_hf_model_path(
                self.config.checkpoint_location,
                required_files=("config.json", "model.safetensors"),
            )
        )
        backbone = str(
            resolve_local_hf_model_path(
                self.config.backbone_location,
                required_files=("config.json",),
            )
        )
        attention = resolve_transformers_attention_implementation(
            self.config.attention_backend,
            device,
        )
        model = SpiritVLAPolicy.from_pretrained(
            checkpoint,
            strict=True,
            backbone_path=backbone,
            local_files_only=True,
            attention_implementation=attention,
            torch_dtype=dtype,
            config_overrides={
                "camera_keys": self.config.camera_keys,
                "state_mask_indices": self.config.state_mask_indices,
                "user_prompt_template": self.config.user_prompt_template,
            },
        )
        model = model.to(device=device, dtype=dtype).eval()
        self._model = model
        self._device = device
        self._dtype = dtype
        self._checkpoint_location = checkpoint
        self._backbone_location = backbone
        return model

    def _state(self, observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, *self.config.state_keys)
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(nested, *self.config.state_keys)
        return state

    def _images(self, observation: Mapping[str, Any], image: Any) -> list[Any | None]:
        def from_mapping(mapping: Mapping[str, Any]) -> list[Any]:
            selected = []
            for target, aliases in zip(
                self.config.camera_keys,
                self.config.camera_aliases,
                strict=True,
            ):
                value = first_present(mapping, target, *aliases)
                # Keep the configured camera slot even when it is absent.  The
                # official preprocessing replaces an absent slot with a masked
                # zero image; compacting this list would silently relabel later
                # wrist cameras as earlier cameras.
                selected.append(value)
            return selected

        nested_images = observation.get("images")
        if isinstance(nested_images, Mapping):
            selected = from_mapping(nested_images)
            if any(value is not None for value in selected):
                return selected
        selected = from_mapping(observation)
        if any(value is not None for value in selected):
            return selected
        if isinstance(image, Mapping):
            return from_mapping(image)
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)):
            return list(image)
        return [image] if image is not None else []

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

        model = self._load()
        images = self._images(observation, image)
        if not any(value is not None for value in images):
            raise ValueError("Spirit-v1.5 requires at least one RGB image")
        image_tensors: list[tuple[str, Any]] = []
        for key, value in zip(
            self.config.camera_keys,
            images[: len(self.config.camera_keys)],
            strict=False,
        ):
            if value is None:
                continue
            pil_image = load_pil_image(value, first_sequence_item=False)
            array = np.asarray(pil_image.convert("RGB"), dtype=np.float32).copy() / 255.0
            image_tensors.append((key, torch.from_numpy(array).permute(2, 0, 1)))

        state = self._state(observation)
        if state is None:
            raise ValueError("Spirit-v1.5 requires a robot state vector")
        if isinstance(state, torch.Tensor):
            state_tensor = state.detach().to(dtype=self._dtype)
        else:
            state_tensor = torch.as_tensor(np.asarray(state), dtype=self._dtype)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        expected_dim = model.config.input_features["observation.state"].shape[0]
        if state_tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"Spirit-v1.5 checkpoint expects {expected_dim} state values, "
                f"received {state_tensor.shape[-1]}"
            )

        batch: dict[str, Any] = {
            "observation.state": state_tensor.to(self._device),
            "task": [instruction],
            "robot_type": [self.config.robot_type],
        }
        for key, tensor in image_tensors:
            batch[key] = tensor.unsqueeze(0).to(device=self._device, dtype=self._dtype)

        set_seed_everywhere(self.config.seed)
        with torch.inference_mode():
            actions = model.select_action(batch).to(dtype=torch.float32).cpu()
        return completed_action_result(
            model_id="spirit-v1.5",
            instruction=instruction,
            actions=actions,
            checkpoint_path=str(self._checkpoint_location),
            device=str(self._device),
            runtime="worldfoundry.spirit_v15.in_tree",
            metadata={
                "backbone_path": self._backbone_location,
                "robot_type": self.config.robot_type,
                "image_views": len(image_tensors),
                "seed": self.config.seed,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str, str], SpiritV15Runtime] = {}


def _required_option(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"Spirit-v1.5 runtime config requires {key!r}")
    return value


def _string_tuple(value: Any, *, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Spirit-v1.5 {key} must be a sequence")
    return tuple(str(item) for item in value)


def _nested_string_tuple(value: Any, *, key: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Spirit-v1.5 {key} must be a sequence")
    return tuple(_string_tuple(item, key=key) for item in value)


def _int_tuple(value: Any, *, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Spirit-v1.5 {key} must be a sequence")
    return tuple(int(item) for item in value)


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
    """Run the native policy through the shared action-runtime contract."""

    del action_context
    options = dict(runtime_options or {})
    config = SpiritV15RuntimeConfig(
        checkpoint_location=checkpoint_path,
        backbone_location=str(_required_option(options, "backbone_path")),
        device=device,
        torch_dtype=str(_required_option(options, "torch_dtype")),
        attention_backend=str(_required_option(options, "attention_backend")),
        robot_type=str(_required_option(options, "robot_type")),
        seed=option_int(_required_option(options, "seed"), 0),
        state_keys=_string_tuple(_required_option(options, "state_keys"), key="state_keys"),
        camera_keys=_string_tuple(_required_option(options, "camera_keys"), key="camera_keys"),
        camera_aliases=_nested_string_tuple(
            _required_option(options, "camera_aliases"),
            key="camera_aliases",
        ),
        state_mask_indices=_int_tuple(
            _required_option(options, "state_mask_indices"),
            key="state_mask_indices",
        ),
        user_prompt_template=str(_required_option(options, "user_prompt_template")),
    )
    # Call metadata such as task_instruction changes per request. Cache the
    # parsed model configuration instead, so a new prompt does not reload the
    # Qwen backbone and Spirit policy weights on every action prediction.
    cache_key = (config.checkpoint_location, config.device, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.setdefault(cache_key, SpiritV15Runtime(config))
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["SpiritV15Runtime", "SpiritV15RuntimeConfig", "predict_action"]

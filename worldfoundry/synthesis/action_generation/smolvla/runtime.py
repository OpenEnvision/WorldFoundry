"""Native in-tree inference runtime for SmolVLA."""

from __future__ import annotations

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
class SmolVLARuntimeConfig:
    checkpoint_location: str
    backbone_location: str
    device: str
    torch_dtype: str
    seed: int
    compile_model: bool
    denoising_steps: int
    state_keys: tuple[str, ...]
    camera_aliases: tuple[tuple[str, ...], ...]
    prompt_suffix: str

    def __post_init__(self) -> None:
        if self.denoising_steps < 1:
            raise ValueError("SmolVLA denoising_steps must be positive")
        if not self.state_keys:
            raise ValueError("SmolVLA state_keys cannot be empty")
        if not self.camera_aliases:
            raise ValueError("SmolVLA camera_aliases cannot be empty")


def _find_stats_file(directory: Path, pipeline: str) -> Path:
    candidates = sorted(directory.glob(f"policy_{pipeline}_step_*.safetensors"))
    normalizer = [path for path in candidates if "normalizer" in path.name]
    selected = normalizer or candidates
    if not selected:
        raise FileNotFoundError(f"SmolVLA checkpoint is missing {pipeline} normalization statistics")
    if len(selected) != 1:
        names = ", ".join(path.name for path in selected)
        raise ValueError(f"SmolVLA checkpoint has ambiguous {pipeline} statistics: {names}")
    return selected[0]


def _apply_normalization(
    tensor: Any,
    *,
    feature: str,
    feature_type: str,
    mode_map: Mapping[str, str],
    stats: Mapping[str, Any],
    inverse: bool,
) -> Any:
    import torch

    mode = str(mode_map.get(feature_type.upper(), "IDENTITY")).upper()
    if mode == "IDENTITY":
        return tensor

    def stat(name: str) -> Any:
        key = f"{feature}.{name}"
        if key not in stats:
            raise KeyError(f"SmolVLA normalization statistics are missing {key!r}")
        return stats[key].to(device=tensor.device, dtype=tensor.dtype)

    if mode == "MEAN_STD":
        mean, std = stat("mean"), stat("std")
        return tensor * std + mean if inverse else (tensor - mean) / (std + 1e-8)
    if mode == "MIN_MAX":
        lower, upper = stat("min"), stat("max")
    elif mode == "QUANTILES":
        lower, upper = stat("q01"), stat("q99")
    elif mode == "QUANTILE10":
        lower, upper = stat("q10"), stat("q90")
    else:
        raise ValueError(f"Unsupported SmolVLA normalization mode: {mode}")
    width = torch.where(upper == lower, torch.full_like(upper, 1e-8), upper - lower)
    return (tensor + 1.0) * width / 2.0 + lower if inverse else 2.0 * (tensor - lower) / width - 1.0


class SmolVLARuntime:
    """Persistent policy runtime with local-only backbone and strict weights."""

    def __init__(self, config: SmolVLARuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._policy_config: Any = None
        self._pre_stats: Mapping[str, Any] | None = None
        self._post_stats: Mapping[str, Any] | None = None
        self._device: str | None = None
        self._dtype: Any = None
        self._checkpoint_location: str | None = None
        self._backbone_location: str | None = None
        self._weight_report: Mapping[str, int] | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        from worldfoundry.core.model_loading.checkpoints import load_safetensors_into_model_streaming
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .configuration import SmolVLAConfig
        from .modeling import SmolVLAPolicy

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        checkpoint = Path(
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
        policy_config = SmolVLAConfig.from_json(checkpoint / "config.json")
        policy_config.device = device
        policy_config.vlm_model_name = backbone
        # The policy checkpoint contains the complete VLM and action expert.
        # Only use the backbone directory for its local config/processor so we
        # do not read and then immediately overwrite another 2 GB weight file.
        policy_config.load_vlm_weights = False
        policy_config.compile_model = self.config.compile_model
        policy_config.runtime_torch_dtype = dtype
        policy_config.num_steps = self.config.denoising_steps

        # Materialize checkpoint tensors directly at the requested dtype and
        # device.  This avoids both random CPU parameter initialization and a
        # second full model copy during strict restoration.
        # Keep non-persistent rotary/cache buffers concretely initialized;
        # they are intentionally absent from the checkpoint state dict.
        with init_empty_weights(include_buffers=False):
            model = SmolVLAPolicy(policy_config)
        weight_report = load_safetensors_into_model_streaming(
            model,
            checkpoint,
            strict=True,
            device=device,
            dtype=dtype,
        )
        meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
        meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                "SmolVLA checkpoint left tensors on the meta device: "
                f"parameters={meta_parameters}, buffers={meta_buffers}"
            )
        model.requires_grad_(False)
        # Parameters already reside on the target device/dtype. This cheap
        # final move places the concrete non-persistent buffers alongside them.
        model = model.to(device=device, dtype=dtype).eval()

        self._model = model
        self._policy_config = policy_config
        self._pre_stats = load_file(
            str(_find_stats_file(checkpoint, "preprocessor")),
            device="cpu",
        )
        self._post_stats = load_file(
            str(_find_stats_file(checkpoint, "postprocessor")),
            device="cpu",
        )
        self._device = device
        self._dtype = dtype
        self._checkpoint_location = str(checkpoint)
        self._backbone_location = backbone
        self._weight_report = weight_report
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
            for aliases in self.config.camera_aliases:
                value = first_present(mapping, *aliases)
                if isinstance(value, Mapping):
                    value = None  # A multi-camera container is not one image.
                # Missing cameras are meaningful: the official policy creates
                # masked empty-camera tokens for them.  Preserve each semantic
                # slot so a later wrist camera is never assigned to camera1.
                selected.append(value)
            return selected

        nested_observation = observation.get("observation")
        mappings: list[Mapping[str, Any]] = []
        for container in (observation, nested_observation):
            if isinstance(container, Mapping):
                for key in ("images", "image"):
                    views = container.get(key)
                    if isinstance(views, Mapping):
                        mappings.append(views)
        mappings.append(observation)
        if isinstance(nested_observation, Mapping):
            mappings.append(nested_observation)
        for mapping in mappings:
            selected = from_mapping(mapping)
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
        config = self._policy_config
        state = self._state(observation)
        if state is None:
            raise ValueError("SmolVLA requires a robot state vector")
        if isinstance(state, torch.Tensor):
            state_tensor = state.detach().to(device=self._device, dtype=self._dtype)
        else:
            state_tensor = torch.as_tensor(np.asarray(state), device=self._device, dtype=self._dtype)
        state_tensor = state_tensor.flatten().unsqueeze(0)
        expected_state_dim = config.state_feature.shape[0]
        if state_tensor.shape[-1] != expected_state_dim:
            raise ValueError(
                f"SmolVLA checkpoint expects {expected_state_dim} state values, "
                f"received {state_tensor.shape[-1]}"
            )
        state_tensor = _apply_normalization(
            state_tensor,
            feature="observation.state",
            feature_type="STATE",
            mode_map=config.normalization_mapping,
            stats=self._pre_stats,
            inverse=False,
        )

        images = self._images(observation, image)
        if not any(value is not None for value in images):
            raise ValueError("SmolVLA requires at least one RGB image")
        image_keys = tuple(config.image_features)
        batch: dict[str, Any] = {"observation.state": state_tensor}
        for key, value in zip(image_keys, images, strict=False):
            if value is None:
                continue
            pil_image = load_pil_image(value, first_sequence_item=False).convert("RGB")
            array = np.asarray(pil_image, dtype=np.float32).copy() / 255.0
            batch[key] = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=self._device, dtype=self._dtype)
            )

        tokenizer = model.model.vlm_with_expert.processor.tokenizer
        tokenizer.padding_side = "right"
        prompt = (
            instruction
            if not self.config.prompt_suffix or instruction.endswith(self.config.prompt_suffix)
            else f"{instruction}{self.config.prompt_suffix}"
        )
        tokens = tokenizer(
            [prompt],
            max_length=config.tokenizer_max_length,
            truncation=True,
            padding=config.pad_language_to,
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tokens["input_ids"].to(self._device)
        batch["observation.language.attention_mask"] = tokens["attention_mask"].to(
            device=self._device,
            dtype=torch.bool,
        )

        set_seed_everywhere(self.config.seed)
        with torch.inference_mode():
            actions = model.predict_action_chunk(batch).to(dtype=torch.float32)
        actions = _apply_normalization(
            actions,
            feature="action",
            feature_type="ACTION",
            mode_map=config.normalization_mapping,
            stats=self._post_stats,
            inverse=True,
        ).cpu()
        return completed_action_result(
            model_id="smolvla",
            instruction=instruction,
            actions=actions,
            checkpoint_path=str(self._checkpoint_location),
            device=str(self._device),
            runtime="worldfoundry.smolvla.in_tree",
            metadata={
                "backbone_path": self._backbone_location,
                "image_views": sum(value is not None for value in images[: len(image_keys)]),
                "seed": self.config.seed,
                "denoising_steps": config.num_steps,
                "weight_report": self._weight_report,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str, str], SmolVLARuntime] = {}


def _required_option(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"SmolVLA runtime config requires {key!r}")
    return value


def _string_tuple(value: Any, *, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"SmolVLA {key} must be a sequence")
    return tuple(str(item) for item in value)


def _nested_string_tuple(value: Any, *, key: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"SmolVLA {key} must be a sequence")
    return tuple(_string_tuple(item, key=key) for item in value)


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
    """Run SmolVLA through the shared action-runtime contract."""

    del action_context
    options = dict(runtime_options or {})
    config = SmolVLARuntimeConfig(
        checkpoint_location=checkpoint_path,
        backbone_location=str(_required_option(options, "backbone_path")),
        device=device,
        torch_dtype=str(_required_option(options, "torch_dtype")),
        seed=option_int(_required_option(options, "seed"), 0),
        compile_model=option_bool(_required_option(options, "compile_model")),
        denoising_steps=option_int(_required_option(options, "denoising_steps"), 0),
        state_keys=_string_tuple(_required_option(options, "state_keys"), key="state_keys"),
        camera_aliases=_nested_string_tuple(
            _required_option(options, "camera_aliases"),
            key="camera_aliases",
        ),
        prompt_suffix=str(_required_option(options, "prompt_suffix")),
    )
    cache_key = (checkpoint_path, device, runtime_options_cache_key(options))
    runtime = _RUNTIME_CACHE.setdefault(cache_key, SmolVLARuntime(config))
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["SmolVLARuntime", "SmolVLARuntimeConfig", "predict_action"]

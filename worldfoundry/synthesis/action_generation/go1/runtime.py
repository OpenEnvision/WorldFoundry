# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Local-only checkpoint runtime for the official GO-1 policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class GO1RuntimeConfig:
    checkpoint_location: str
    camera_keys: Sequence[str]
    camera_aliases: Mapping[str, Sequence[str]]
    normalization_mean: Sequence[float]
    normalization_std: Sequence[float]
    prompt_template: Mapping[str, str]
    variant: str
    control_frequency: int
    max_sequence_length: int
    normalization_epsilon: float
    device: str = "cuda"
    torch_dtype: str = "auto"
    local_files_only: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    attention_backend: str = "auto"
    dynamic_image_size: bool | None = None
    use_thumbnail: bool | None = None
    min_dynamic_patches: int | None = None
    max_dynamic_patches: int | None = None
    pad_to_square: bool | None = None
    action_horizon: int | None = None
    seed: int = 0
    statistics_path: str | None = None
    statistics_file: str | None = None
    statistics_key: str | None = None


class GO1Runtime:
    """Persistent in-process GO-1 model and tokenizer."""

    def __init__(self, config: GO1RuntimeConfig) -> None:
        if not config.local_files_only:
            raise ValueError("GO-1 requires local_files_only=true; runtime downloads are disabled")
        if not config.camera_keys:
            raise ValueError("GO-1 camera_keys cannot be empty")
        if len(config.normalization_mean) != 3 or len(config.normalization_std) != 3:
            raise ValueError("GO-1 image normalization requires three mean/std values")
        if config.control_frequency <= 0 or config.max_sequence_length <= 0:
            raise ValueError("GO-1 control frequency and max sequence length must be positive")
        self.config = config
        self._snapshot: Path | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._attention_backend: str | None = None
        self._statistics: Mapping[str, Any] | None = None

    @staticmethod
    def _existing_path(value: str | None) -> Path | None:
        if not value:
            return None
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        path = resolve_worldfoundry_path(value)
        return path.resolve() if path.exists() else None

    def _resolve_snapshot(self) -> Path:
        if self._snapshot is not None:
            return self._snapshot
        from worldfoundry.core.io.hf import materialize_hf_snapshot

        direct = self._existing_path(self.config.checkpoint_location)
        location = str(direct) if direct is not None else self.config.checkpoint_location
        snapshot = materialize_hf_snapshot(
            location,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            required_files=("config.json",),
            local_files_only=self.config.local_files_only,
        )
        if not snapshot.is_dir():
            raise FileNotFoundError(
                f"GO-1 checkpoint must be staged before inference: {self.config.checkpoint_location}"
            )
        safe_weights = list(snapshot.glob("*.safetensors"))
        if not safe_weights:
            raise FileNotFoundError(
                f"GO-1 staged checkpoint has no safetensors weights: {snapshot}"
            )
        self._snapshot = snapshot
        return snapshot

    def _select_attention_backend(self, device: str, dtype: Any) -> str:
        import torch

        requested = self.config.attention_backend.strip().lower().replace("-", "_")
        if requested not in {"auto", "eager", "flash", "flash_attention_2"}:
            raise ValueError(f"Unsupported GO-1 attention backend: {self.config.attention_backend}")
        from .flash_attn_utils import load_flash_attn

        flash_ready = False
        if torch.device(device).type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            index = torch.device(device).index
            major, _ = torch.cuda.get_device_capability(index or 0)
            flash_ready = major in {8, 9} and bool(load_flash_attn())
        if requested in {"flash", "flash_attention_2"} and not flash_ready:
            raise RuntimeError(
                "GO-1 Flash Attention was explicitly requested but no compatible local FA2 build is available"
            )
        return "flash_attention_2" if flash_ready and requested != "eager" else "eager"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .configuration_go1 import GO1ModelConfig
        from .modeling_go1 import GO1Model
        from .tokenization_internlm2 import InternLM2Tokenizer

        snapshot = self._resolve_snapshot()
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        config = GO1ModelConfig.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        backend = self._select_attention_backend(device, dtype)
        config.llm_config.attn_implementation = backend
        config.action_config.attn_implementation = backend
        config.latent_planner_config.attn_implementation = backend
        config.vision_config.use_flash_attn = backend == "flash_attention_2"
        model = GO1Model.from_pretrained(
            snapshot,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(device=device, dtype=dtype).eval()
        model.torch_dtype = dtype
        model.time_embedder.dtype = dtype
        model.freq_embedder.dtype = dtype
        if getattr(model, "enable_lam", False):
            model.latent_planner.torch_dtype = dtype
        tokenizer = InternLM2Tokenizer.from_pretrained(
            snapshot,
            model_max_length=int(self.config.max_sequence_length),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("GO-1 staged tokenizer must define pad_token_id")
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._dtype = dtype
        self._attention_backend = backend
        return model

    def _statistics_payload(self) -> Mapping[str, Any]:
        if self._statistics is not None:
            return self._statistics
        path = self._existing_path(self.config.statistics_path)
        if path is None and self.config.statistics_file:
            candidate = self._resolve_snapshot() / self.config.statistics_file
            path = candidate if candidate.is_file() else None
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "This GO-1 checkpoint enables normalization, but its dataset statistics were not staged"
            )
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if self.config.statistics_key:
            payload = payload[self.config.statistics_key]
        if not isinstance(payload, Mapping):
            raise ValueError("GO-1 statistics must contain a JSON object")
        self._statistics = payload
        return payload

    def _normalize_state(self, state: Any) -> Any:
        import numpy as np

        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if not bool(self._model.config.norm):
            return values
        stats = self._statistics_payload().get("state")
        if not isinstance(stats, Mapping):
            raise ValueError("GO-1 statistics have no 'state' entry")
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
        if values.shape != mean.shape or values.shape != std.shape:
            raise ValueError(
                f"GO-1 state shape {values.shape} does not match statistics {mean.shape}/{std.shape}"
            )
        return (values - mean) / (std + float(self.config.normalization_epsilon))

    def _denormalize_actions(self, actions: Any) -> Any:
        import numpy as np

        values = np.asarray(actions, dtype=np.float32)
        if not bool(self._model.config.norm):
            return values
        stats = self._statistics_payload().get("action")
        if not isinstance(stats, Mapping):
            raise ValueError("GO-1 statistics have no 'action' entry")
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
        if values.shape[-1] != mean.shape[0] or mean.shape != std.shape:
            raise ValueError(
                f"GO-1 action width {values.shape[-1]} does not match statistics {mean.shape}/{std.shape}"
            )
        return values * (std + float(self.config.normalization_epsilon)) + mean

    @staticmethod
    def _state(observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, "state", "proprio", "agent_pos", "joint_state", "robot_state")
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(nested, "state", "proprio", "agent_pos", "joint_state", "robot_state")
        return state

    def _camera_values(self, observation: Mapping[str, Any], image: Any) -> list[Any]:
        nested = observation.get("images")
        wrapped = observation.get("observation")
        containers = [nested] if isinstance(nested, Mapping) else []
        if isinstance(wrapped, Mapping):
            wrapped_images = wrapped.get("images")
            if isinstance(wrapped_images, Mapping):
                containers.append(wrapped_images)
            containers.append(wrapped)
        containers.append(observation)
        if isinstance(image, Mapping):
            containers.append(image)
        values: list[Any] = []
        for index, key in enumerate(self.config.camera_keys):
            candidates = (key, *tuple(self.config.camera_aliases.get(key, ())))
            value = None
            for container in containers:
                candidate = first_present(container, *candidates)
                if isinstance(candidate, Mapping):
                    candidate = first_present(candidate, *candidates)
                value = candidate
                if value is not None:
                    break
            if value is None and index == 0 and image is not None and not isinstance(image, Mapping):
                value = image
            if value is None:
                raise ValueError(f"GO-1 requires camera {key!r}; accepted aliases are {candidates}")
            values.append(value)
        return values

    @staticmethod
    def _config_bool(override: bool | None, value: Any) -> bool:
        return bool(value) if override is None else bool(override)

    @staticmethod
    def _config_int(override: int | None, value: Any) -> int:
        return int(value) if override is None else int(override)

    def _prepare_inputs(self, observation: Mapping[str, Any], image: Any, instruction: str) -> dict[str, Any]:
        from .preprocessing import prepare_inputs

        model_config = self._model.config
        image_size = int(model_config.force_image_size or model_config.vision_config.image_size)
        num_image_tokens = int(
            (image_size // int(model_config.vision_config.patch_size)) ** 2
            * float(model_config.downsample_ratio) ** 2
        )
        return prepare_inputs(
            images=self._camera_values(observation, image),
            instruction=instruction,
            tokenizer=self._tokenizer,
            num_image_tokens=num_image_tokens,
            image_size=image_size,
            dynamic_image_size=self._config_bool(
                self.config.dynamic_image_size,
                model_config.dynamic_image_size,
            ),
            use_thumbnail=self._config_bool(self.config.use_thumbnail, model_config.use_thumbnail),
            min_dynamic_patches=self._config_int(
                self.config.min_dynamic_patches,
                model_config.min_dynamic_patch,
            ),
            max_dynamic_patches=self._config_int(
                self.config.max_dynamic_patches,
                model_config.max_dynamic_patch,
            ),
            pad_to_square=self._config_bool(self.config.pad_to_square, model_config.pad2square),
            normalization_mean=self.config.normalization_mean,
            normalization_std=self.config.normalization_std,
            max_sequence_length=int(self.config.max_sequence_length),
            prompt_template=self.config.prompt_template,
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

        model = self._load_model()
        raw_state = self._state(observation)
        if raw_state is None:
            raise ValueError("GO-1 requires a state/proprio vector")
        state = self._normalize_state(raw_state)
        expected_state_dim = int(model.config.action_config.state_dim)
        if state.shape[0] != expected_state_dim:
            raise ValueError(f"GO-1 expects {expected_state_dim} state values, got {state.shape[0]}")
        prepared = self._prepare_inputs(observation, image, instruction)
        frequency = int(first_present(observation, "control_frequency", "ctrl_freq") or self.config.control_frequency)
        devices = []
        if torch.device(self._device).type == "cuda":
            devices = [torch.device(self._device).index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config.seed))
            with torch.inference_mode():
                output = model(
                    pixel_values=prepared["pixel_values"].to(device=self._device, dtype=self._dtype),
                    input_ids=prepared["input_ids"].to(self._device).unsqueeze(0),
                    attention_mask=prepared["attention_mask"].to(self._device).unsqueeze(0),
                    position_ids=prepared["position_ids"].to(self._device).unsqueeze(0),
                    image_flags=prepared["image_flags"].to(self._device),
                    state=torch.as_tensor(state, device=self._device, dtype=self._dtype).view(1, 1, -1),
                    ctrl_freqs=torch.tensor([[frequency]], device=self._device, dtype=self._dtype),
                    return_dict=True,
                )
        action_tensor = getattr(output, "action_logits", None)
        if action_tensor is None and isinstance(output, Sequence) and len(output) > 1:
            action_tensor = output[1]
        if action_tensor is None:
            raise RuntimeError("GO-1 model returned no action logits")
        actions = self._denormalize_actions(action_tensor[0].float().cpu().numpy())
        horizon = min(int(self.config.action_horizon or actions.shape[0]), actions.shape[0])
        actions = np.asarray(actions[:horizon], dtype=np.float32)
        return completed_action_result(
            model_id="go1",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=str(self._snapshot),
            device=str(self._device),
            runtime="worldfoundry.go1.in_tree_runtime",
            metadata={
                "variant": self.config.variant,
                "action_shape": list(actions.shape),
                "camera_keys": list(self.config.camera_keys),
                "tiles_per_image": list(prepared["tiles_per_image"]),
                "control_frequency": frequency,
                "attention_backend": self._attention_backend,
                "dtype": str(self._dtype),
                "seed": int(self.config.seed),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], GO1Runtime] = {}


def _required(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"GO-1 runtime option {key!r} is required; load its data runtime config")
    return value


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
    """Callable entrypoint used by the shared policy runtime."""

    del action_context
    options = dict(runtime_options or {})
    checkpoint = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not checkpoint:
        raise ValueError("GO-1 checkpoint_path or checkpoint_ref is required")
    config = GO1RuntimeConfig(
        checkpoint_location=checkpoint,
        camera_keys=tuple(_required(options, "camera_keys")),
        camera_aliases={key: tuple(value) for key, value in dict(_required(options, "camera_aliases")).items()},
        normalization_mean=tuple(_required(options, "normalization_mean")),
        normalization_std=tuple(_required(options, "normalization_std")),
        prompt_template=dict(_required(options, "prompt_template")),
        variant=str(_required(options, "variant")),
        control_frequency=option_int(_required(options, "control_frequency"), 0),
        max_sequence_length=option_int(_required(options, "max_sequence_length"), 0),
        normalization_epsilon=float(_required(options, "normalization_epsilon")),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        local_files_only=option_bool(options.get("local_files_only"), True),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
        revision=str(options["revision"]) if options.get("revision") else None,
        attention_backend=str(options.get("attention_backend") or "auto"),
        dynamic_image_size=(
            option_bool(options.get("dynamic_image_size"))
            if options.get("dynamic_image_size") is not None
            else None
        ),
        use_thumbnail=(
            option_bool(options.get("use_thumbnail"))
            if options.get("use_thumbnail") is not None
            else None
        ),
        min_dynamic_patches=(
            option_int(options.get("min_dynamic_patches"), 1)
            if options.get("min_dynamic_patches") is not None
            else None
        ),
        max_dynamic_patches=(
            option_int(options.get("max_dynamic_patches"), 6)
            if options.get("max_dynamic_patches") is not None
            else None
        ),
        pad_to_square=(
            option_bool(options.get("pad_to_square"))
            if options.get("pad_to_square") is not None
            else None
        ),
        action_horizon=(
            option_int(options.get("action_horizon"), 0)
            if options.get("action_horizon") is not None
            else None
        ),
        seed=option_int(options.get("seed"), 0),
        statistics_path=str(options["statistics_path"]) if options.get("statistics_path") else None,
        statistics_file=str(options["statistics_file"]) if options.get("statistics_file") else None,
        statistics_key=str(options["statistics_key"]) if options.get("statistics_key") else None,
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = GO1Runtime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["GO1Runtime", "GO1RuntimeConfig", "predict_action"]

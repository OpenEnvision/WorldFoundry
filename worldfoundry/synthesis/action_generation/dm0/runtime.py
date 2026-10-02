"""Local, checkpoint-backed DM0 action inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_data_path, resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class DM0RuntimeConfig:
    checkpoint_location: str
    device: str
    torch_dtype: str
    max_language_tokens: int
    diffusion_steps: int
    model_action_dim: int
    action_dim: int
    num_images: int
    camera_keys: tuple[str, ...]
    non_delta_indices: tuple[int, ...]
    system_prompt: str
    norm_stats_filename: str
    require_norm_stats: bool
    seed: int
    compile_model: bool


def _runtime_defaults() -> dict[str, Any]:
    import yaml

    path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", "dm0.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"DM0 runtime config must contain a mapping: {path}")
    return dict(payload)


class DM0Runtime:
    """Persistent DM0 policy runtime with in-tree preprocessing and sampling."""

    def __init__(self, config: DM0RuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._checkpoint = ""
        self._statistics: dict[str, Any] = {}

    @staticmethod
    def _weight_present(checkpoint: Path) -> bool:
        return any(
            (checkpoint / name).is_file()
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoTokenizer

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import DM0Config, DM0ForCausalLM

        checkpoint = resolve_local_hf_model_path(
            self.config.checkpoint_location,
            required_files=("config.json", "tokenizer_config.json"),
        )
        if not self._weight_present(checkpoint):
            raise FileNotFoundError(f"DM0 weights are missing under {checkpoint}")
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        common = {"local_files_only": True, "trust_remote_code": False}
        model_config = DM0Config.from_pretrained(checkpoint, **common)
        model_config._attn_implementation = "eager"
        if int(model_config.action_dim) != self.config.model_action_dim:
            raise ValueError(
                f"unsupported DM0 checkpoint action_dim={model_config.action_dim}; "
                f"expected {self.config.model_action_dim}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=False,
            **common,
        )
        model = DM0ForCausalLM.from_pretrained(
            checkpoint,
            config=model_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **common,
        ).to(device=device)
        model.requires_grad_(False).eval()
        if self.config.compile_model and hasattr(torch, "compile"):
            model.model.action_out_proj = torch.compile(
                model.model.action_out_proj,
                dynamic=False,
            )
        stats_path = checkpoint / self.config.norm_stats_filename
        if stats_path.is_file():
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            statistics = payload.get("norm_stats", payload)
            if not isinstance(statistics, dict):
                raise ValueError(f"DM0 {self.config.norm_stats_filename} must contain an object")
            self._statistics = statistics
        elif self.config.require_norm_stats:
            raise FileNotFoundError(f"DM0 normalization statistics are missing: {stats_path}")
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = dtype
        self._checkpoint = str(checkpoint)
        return tokenizer, model

    def _tokenize(self, instruction: str) -> tuple[Any, Any]:
        import torch

        tokenizer = self._tokenizer
        segments = (
            f"{self.config.system_prompt} ",
            "USER: ",
            f"{instruction.strip().replace(chr(10), ' ')} ",
        )
        token_ids: list[int] = []
        for segment in segments:
            token_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
        maximum = int(self.config.max_language_tokens)
        if maximum <= 0:
            raise ValueError("max_language_tokens must be positive")
        token_ids = token_ids[:maximum]
        attention = [True] * len(token_ids)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("DM0 tokenizer is missing pad_token_id")
        padding = maximum - len(token_ids)
        token_ids.extend([pad_id] * padding)
        attention.extend([False] * padding)
        return (
            torch.tensor([token_ids], device=self._device, dtype=torch.long),
            torch.tensor([attention], device=self._device, dtype=torch.bool),
        )

    @staticmethod
    def _state_from_observation(observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, "state", "proprio", "robot_state")
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(nested, "state", "proprio", "robot_state")
        return state

    @staticmethod
    def _bounds(entry: Mapping[str, Any], dimension: int) -> tuple[Any, Any]:
        import numpy as np

        low = first_present(entry, "min", "q01")
        high = first_present(entry, "max", "q99")
        if low is None or high is None:
            raise KeyError("normalization statistics require min/max or q01/q99")
        low = np.asarray(low, dtype=np.float32).reshape(-1)
        high = np.asarray(high, dtype=np.float32).reshape(-1)
        if low.size == 1:
            low = np.repeat(low, dimension)
            high = np.repeat(high, dimension)
        if low.size < dimension:
            low = np.pad(low, (0, dimension - low.size), constant_values=-1.0)
            high = np.pad(high, (0, dimension - high.size), constant_values=1.0)
        if low.size != dimension or high.size != dimension:
            raise ValueError(f"normalization statistics have {low.size}/{high.size} values, expected {dimension}")
        return low, high

    def _normalize_state(self, state: Any) -> tuple[Any, Any]:
        import numpy as np

        original = np.asarray(state, dtype=np.float32).reshape(-1)
        dimension = self.config.model_action_dim
        if original.size > dimension:
            raise ValueError(f"DM0 accepts at most {dimension} state values, got {original.size}")
        padded = np.pad(original, (0, dimension - original.size))
        entry = self._statistics.get("state")
        if not isinstance(entry, Mapping):
            return padded, padded.copy()
        low, high = self._bounds(entry, dimension)
        normalized = (padded - low) / (high - low + 1e-6) * 2.0 - 1.0
        return normalized.astype(np.float32), padded

    def _denormalize_actions(self, actions: Any) -> Any:
        import numpy as np

        actions = np.asarray(actions, dtype=np.float32)
        entry = self._statistics.get("action")
        if not isinstance(entry, Mapping):
            return actions
        low, high = self._bounds(entry, self.config.model_action_dim)
        return (actions + 1.0) * 0.5 * (high - low + 1e-6) + low

    def _absolute_actions(self, actions: Any, state: Any) -> Any:
        actions = actions + state[None, :]
        for index in self.config.non_delta_indices:
            if 0 <= index < actions.shape[-1]:
                actions[:, index] -= state[index]
        return actions

    def _images(self, observation: Mapping[str, Any], image: Any, model: Any) -> tuple[Any, Any]:
        import torch

        from worldfoundry.core.media.processing.image_utils import load_pil_image

        values = collect_images(observation, image, self.config.camera_keys)
        if not values:
            raise ValueError("DM0 requires at least one RGB image")
        values = values[: self.config.num_images]
        pil_images = [load_pil_image(value, first_sequence_item=False) for value in values]
        pixels = model.process_images(pil_images).to(device=self._device, dtype=self._dtype)
        valid = int(pixels.shape[0])
        if valid < self.config.num_images:
            padding = torch.zeros_like(pixels[:1]).expand(self.config.num_images - valid, -1, -1, -1)
            pixels = torch.cat((pixels, padding), dim=0)
        pixels = pixels[: self.config.num_images][None]
        mask = torch.zeros(1, self.config.num_images, device=self._device, dtype=torch.bool)
        mask[:, :valid] = True
        return pixels, mask

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch

        tokenizer, model = self._load()
        del tokenizer
        state_value = self._state_from_observation(observation)
        if state_value is None:
            raise ValueError("DM0 requires a state/proprio vector")
        normalized_state, original_state = self._normalize_state(state_value)
        input_ids, attention_mask = self._tokenize(instruction)
        images, image_masks = self._images(observation, image, model)
        states = torch.as_tensor(
            normalized_state[None],
            device=self._device,
            dtype=self._dtype,
        )
        generator = torch.Generator(device=self._device).manual_seed(int(self.config.seed))
        with torch.inference_mode():
            actions = (
                model.inference_action(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    states=states,
                    images=images,
                    image_masks=image_masks,
                    diffusion_steps=self.config.diffusion_steps,
                    generator=generator,
                )[0]
                .float()
                .cpu()
                .numpy()
            )
        actions = self._absolute_actions(self._denormalize_actions(actions), original_state)
        actions = np.asarray(actions[:, : self.config.action_dim], dtype=np.float32)
        if actions.shape != (int(model.config.chunk_size), self.config.action_dim):
            raise RuntimeError(f"DM0 returned unexpected action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise FloatingPointError("DM0 produced non-finite actions")
        return completed_action_result(
            model_id="dm0",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=self._checkpoint,
            device=str(self._device),
            runtime="worldfoundry.dm0.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "diffusion_steps": self.config.diffusion_steps,
                "input_views": int(image_masks.sum().item()),
                "dtype": str(self._dtype),
                "normalized_with_statistics": "state" in self._statistics,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], DM0Runtime] = {}


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
    if not checkpoint:
        raise ValueError("DM0 requires checkpoint_path")
    raw_indices = options["non_delta_indices"]
    if isinstance(raw_indices, str):
        raw_indices = tuple(int(value.strip()) for value in raw_indices.split(",") if value.strip())
    config = DM0RuntimeConfig(
        checkpoint_location=checkpoint,
        device=str(options.get("device") or device),
        torch_dtype=str(options["torch_dtype"]),
        max_language_tokens=option_int(options["max_language_tokens"], 0),
        diffusion_steps=option_int(options["diffusion_steps"], 0),
        model_action_dim=option_int(options["model_action_dim"], 0),
        action_dim=option_int(options["action_dim"], 0),
        num_images=option_int(options["num_images"], 0),
        camera_keys=tuple(str(value) for value in options["camera_keys"]),
        non_delta_indices=tuple(int(value) for value in raw_indices),
        system_prompt=str(options["system_prompt"]),
        norm_stats_filename=str(options["norm_stats_filename"]),
        require_norm_stats=option_bool(options["require_norm_stats"], True),
        seed=option_int(options["seed"], 0),
        compile_model=option_bool(options["compile_model"], False),
    )
    if config.model_action_dim <= 0:
        raise ValueError("DM0 model_action_dim must be positive")
    if config.action_dim <= 0 or config.action_dim > config.model_action_dim:
        raise ValueError(f"DM0 action_dim must be in [1, {config.model_action_dim}]")
    if config.num_images <= 0 or config.num_images > 3:
        raise ValueError("DM0 num_images must be in [1, 3]")
    if len(config.camera_keys) < config.num_images:
        raise ValueError("DM0 camera_keys must provide at least num_images entries")
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = DM0Runtime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )

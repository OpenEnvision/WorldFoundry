"""Local-only checkpoint runtime for InternVLA-A1-3B."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)


def _transformers_no_init_weights() -> Any:
    """Return the no-init context across Transformers 4.x and 5.x."""

    try:
        from transformers.initialization import no_init_weights

        return no_init_weights()
    except ImportError:
        from transformers.modeling_utils import no_init_weights

        return no_init_weights()


@dataclass(frozen=True)
class InternVLAA1RuntimeConfig:
    checkpoint_location: str
    processor_location: str
    cosmos_encoder_path: str
    cosmos_decoder_path: str
    model_defaults: Mapping[str, Any]
    camera_keys: Sequence[str]
    statistics_file: str
    device: str = "cuda"
    torch_dtype: str = "auto"
    local_files_only: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    processor_revision: str | None = None
    image_size: int = 224
    tokenizer_max_length: int = 48
    image_history_interval: int = 15
    action_horizon: int | None = None
    num_inference_steps: int | None = None
    seed: int = 0
    statistics_path: str | None = None
    statistics_key: str | None = None
    normalization: str = "mean_std"
    action_mode: str = "absolute"
    delta_zero_state_indices: Sequence[int] = ()


class InternVLAA1Runtime:
    """Persistent InternVLA-A1 model, processor, tokenizer, and camera history."""

    def __init__(self, config: InternVLAA1RuntimeConfig) -> None:
        if not config.local_files_only:
            raise ValueError(
                "InternVLA-A1 requires local_files_only=true; runtime downloads are disabled"
            )
        if len(config.camera_keys) != 3:
            raise ValueError("InternVLA-A1 requires exactly three ordered camera keys")
        if config.normalization not in {"mean_std", "identity"}:
            raise ValueError(f"Unsupported InternVLA-A1 normalization: {config.normalization}")
        if config.action_mode not in {"absolute", "delta"}:
            raise ValueError(f"Unsupported InternVLA-A1 action_mode: {config.action_mode}")
        self.config = config
        self._checkpoint_root: Path | None = None
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._policy_config: Any = None
        self._statistics: Mapping[str, Any] | None = None
        self._action_dim: int | None = None
        self._histories = [
            deque(maxlen=max(2, int(config.image_history_interval) + 1))
            for _ in range(3)
        ]

    @staticmethod
    def _existing_path(value: str | None) -> Path | None:
        if not value:
            return None
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        path = resolve_worldfoundry_path(value)
        return path.resolve() if path.exists() else None

    def _resolve_checkpoint(self) -> Path:
        if self._checkpoint_root is not None:
            return self._checkpoint_root
        from worldfoundry.core.io.hf import materialize_hf_snapshot

        direct = self._existing_path(self.config.checkpoint_location)
        location = str(direct) if direct is not None else self.config.checkpoint_location
        root = materialize_hf_snapshot(
            location,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            required_files=("config.json", "model.safetensors", self.config.statistics_file),
            local_files_only=self.config.local_files_only,
        )
        self._checkpoint_root = root
        return root

    def _resolve_processor(self) -> Path:
        from worldfoundry.core.io.hf import materialize_hf_snapshot

        direct = self._existing_path(self.config.processor_location)
        location = str(direct) if direct is not None else self.config.processor_location
        return materialize_hf_snapshot(
            location,
            revision=self.config.processor_revision,
            cache_dir=self.config.cache_dir,
            required_files=("tokenizer_config.json", "preprocessor_config.json"),
            local_files_only=self.config.local_files_only,
        )

    def _resolve_cosmos_encoder(self) -> Path:
        path = self._existing_path(self.config.cosmos_encoder_path)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"InternVLA-A1 Cosmos encoder must be staged before inference: "
                f"{self.config.cosmos_encoder_path}"
            )
        return path

    def _resolve_cosmos_decoder(self) -> Path:
        path = self._existing_path(self.config.cosmos_decoder_path)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"InternVLA-A1 Cosmos decoder must be staged before inference: "
                f"{self.config.cosmos_decoder_path}"
            )
        return path

    @staticmethod
    def _config_payload(root: Path) -> Mapping[str, Any]:
        payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("InternVLA-A1 config.json must contain an object")
        return payload

    @staticmethod
    def _action_dim_from_config(payload: Mapping[str, Any], fallback: int) -> int:
        output_features = payload.get("output_features")
        if isinstance(output_features, Mapping):
            action = output_features.get("action")
            if isinstance(action, Mapping):
                shape = action.get("shape")
                if isinstance(shape, Sequence) and shape:
                    return int(shape[0])
        return fallback

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from torch import nn
        from safetensors.torch import load_model as load_safetensors_model

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .configuration import InternVLAA1Config
        from .cosmos import CosmosImageTokenizer
        from .modeling import QwenA1

        root = self._resolve_checkpoint()
        payload = self._config_payload(root)
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        construction_dtype = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }.get(dtype)
        if construction_dtype is None:
            raise ValueError(f"InternVLA-A1 does not support inference dtype {dtype}")
        policy_config = InternVLAA1Config.from_mapping(
            payload,
            defaults=self.config.model_defaults,
            dtype=construction_dtype,
            num_inference_steps=self.config.num_inference_steps,
        )
        cosmos = CosmosImageTokenizer(
            self._resolve_cosmos_encoder(),
            self._resolve_cosmos_decoder(),
            device="cpu",
            dtype={
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[construction_dtype],
        )
        with _transformers_no_init_weights():
            model = QwenA1(policy_config, cosmos=cosmos)
        # Transformers 5's no-init context also suppresses ``init_weights``,
        # which is where tied input/output embeddings are connected.  The
        # released safetensors intentionally stores only ``lm_head.weight``;
        # restore the shared-parameter relationship before strict loading so
        # safetensors can resolve the omitted duplicate embedding key.
        model.qwen3_vl_with_expert.und_expert.tie_weights()
        # LeRobot saves a wrapper whose sole trainable child is named `model`.
        # Loading through the same wrapper preserves tied Qwen tensors and the
        # frozen Cosmos module's exact state-key contract.
        wrapper = nn.Module()
        wrapper.add_module("model", model)
        missing, unexpected = load_safetensors_model(
            wrapper,
            str(root / "model.safetensors"),
            strict=True,
            device="cpu",
        )
        if missing or unexpected:
            raise RuntimeError(
                f"InternVLA-A1 checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
            )
        model = model.to(device=device, dtype=dtype).eval()
        model.cosmos = model.cosmos.to(device=device, dtype=dtype).eval()
        # ``dtype`` is an execution hint used by the TorchScript wrapper and
        # is not a Tensor/buffer, so ``Module.to`` cannot update it for us.
        model.cosmos.dtype = dtype
        model.action_out_proj = model.action_out_proj.to(device=device, dtype=torch.float32)

        self._model = model
        self._device = device
        self._dtype = dtype
        self._policy_config = policy_config
        self._action_dim = self._action_dim_from_config(payload, policy_config.max_action_dim)
        return model

    def _load_processor(self) -> Any:
        if self._processor is not None:
            return self._processor
        from transformers.models.qwen3_vl import Qwen3VLProcessor

        root = self._resolve_processor()
        self._processor = Qwen3VLProcessor.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        return self._processor

    def _statistics_payload(self) -> Mapping[str, Any]:
        if self._statistics is not None:
            return self._statistics
        path = self._existing_path(self.config.statistics_path)
        if path is None:
            path = self._resolve_checkpoint() / self.config.statistics_file
        if not path.is_file():
            raise FileNotFoundError(f"InternVLA-A1 statistics are missing: {path}")
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if self.config.statistics_key:
            payload = payload[self.config.statistics_key]
        if not isinstance(payload, Mapping):
            raise ValueError("InternVLA-A1 statistics must contain an object")
        self._statistics = payload
        return payload

    def _stats(self, kind: str) -> Mapping[str, Any]:
        payload = self._statistics_payload()
        selected = payload.get(kind)
        if not isinstance(selected, Mapping):
            raise ValueError(f"InternVLA-A1 statistics have no {kind!r} entry")
        return selected

    def _normalize_state(self, values: Any) -> Any:
        import numpy as np

        state = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.config.normalization == "identity":
            return state
        stats = self._stats("observation.state")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if state.shape != mean.shape:
            raise ValueError(
                f"InternVLA-A1 state shape {state.shape} does not match statistics {mean.shape}"
            )
        return (state - mean) / (std + 1.0e-6)

    def _denormalize_action(self, values: Any) -> Any:
        import numpy as np

        actions = np.asarray(values, dtype=np.float32)
        if self.config.normalization == "identity":
            return actions
        stats = self._stats("action")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return actions[..., : mean.shape[0]] * (std + 1.0e-6) + mean

    @staticmethod
    def _state(observation: Mapping[str, Any]) -> Any:
        value = first_present(
            observation,
            "state",
            "proprio",
            "agent_pos",
            "joint_state",
            "robot_state",
        )
        nested = observation.get("observation")
        if value is None and isinstance(nested, Mapping):
            value = first_present(nested, "state", "proprio", "agent_pos", "joint_state")
        return value

    @staticmethod
    def _resize_with_pad(image: Any, size: int) -> Any:
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, Path)):
            image_path = Path(image).expanduser()
            if not image_path.is_file():
                raise FileNotFoundError(f"InternVLA-A1 camera image not found: {image_path}")
            with Image.open(image_path) as loaded:
                source = loaded.convert("RGB")
        elif isinstance(image, Image.Image):
            source = image
        else:
            source = Image.fromarray(np.asarray(image).astype("uint8"))
        source = source.convert("RGB")
        scale = min(size / source.width, size / source.height)
        resized = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
        canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
        return np.asarray(canvas, dtype=np.uint8)

    def _camera_values(self, observation: Mapping[str, Any], image: Any) -> list[Any]:
        values = collect_images(observation, image, tuple(self.config.camera_keys))
        if len(values) != 3:
            raise ValueError(
                f"InternVLA-A1 requires three camera views {tuple(self.config.camera_keys)}, "
                f"got {len(values)}"
            )
        return values

    def _image_history(self, observation: Mapping[str, Any], image: Any) -> Any:
        import numpy as np
        import torch

        if option_bool(observation.get("reset"), False):
            for history in self._histories:
                history.clear()
        current = [
            self._resize_with_pad(value, int(self.config.image_size))
            for value in self._camera_values(observation, image)
        ]
        pairs = []
        for history, frame in zip(self._histories, current):
            history.append(frame)
            past_index = max(len(history) - int(self.config.image_history_interval) - 1, 0)
            pair = np.stack([history[past_index], history[-1]], axis=0)
            pairs.append(torch.from_numpy(pair).permute(0, 3, 1, 2).float().div_(255.0))
        return torch.stack(pairs, dim=0)

    def _language_and_pixels(self, images: Any, instruction: str) -> tuple[Any, Any, Any, Any]:
        import torch

        processor = self._load_processor()
        input_ids: list[int] = []
        attention_mask: list[int] = []
        pixel_values = []
        image_grid_thw = []
        spatial_merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        for view in images:
            result = processor.image_processor(
                view[-1],
                do_rescale=False,
                return_tensors="pt",
            )
            pixels = result.pixel_values
            grid = result.image_grid_thw
            pixel_values.append(pixels)
            image_grid_thw.append(grid)
            image_tokens = int(torch.prod(grid).item()) // (spatial_merge_size**2)
            input_ids.extend(
                [int(processor.vision_start_token_id)]
                + [int(processor.image_token_id)] * image_tokens
                + [int(processor.vision_end_token_id)]
            )
            attention_mask.extend([1] * (image_tokens + 2))
        language = processor.tokenizer(
            instruction,
            max_length=int(self.config.tokenizer_max_length),
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        input_ids.extend(language.input_ids)
        attention_mask.extend(language.attention_mask)
        return (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(attention_mask, dtype=torch.bool).unsqueeze(0),
            torch.cat(pixel_values, dim=0).unsqueeze(0),
            torch.cat(image_grid_thw, dim=0).unsqueeze(0),
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
        state_raw = self._state(observation)
        if state_raw is None:
            raise ValueError("InternVLA-A1 requires a state/proprio vector")
        raw_state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        normalized_state = self._normalize_state(raw_state)
        if normalized_state.shape[0] > self._policy_config.max_state_dim:
            raise ValueError("InternVLA-A1 state exceeds the checkpoint max_state_dim")
        padded_state = np.pad(
            normalized_state,
            (0, self._policy_config.max_state_dim - normalized_state.shape[0]),
        )
        images = self._image_history(observation, image)
        lang_tokens, lang_mask, pixels, grid = self._language_and_pixels(images, instruction)
        images = images.unsqueeze(0).to(device=self._device, dtype=self._dtype)
        image_mask = torch.ones((1, 3), dtype=torch.bool, device=self._device)
        generator = torch.Generator(device=self._device).manual_seed(int(self.config.seed))
        noise = torch.randn(
            (1, self._policy_config.chunk_size, self._policy_config.max_action_dim),
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            actions, _ = model.sample_actions(
                images,
                image_mask,
                pixels.to(device=self._device, dtype=self._dtype),
                grid.to(device=self._device),
                lang_tokens.to(device=self._device),
                lang_mask.to(device=self._device),
                torch.as_tensor(padded_state, device=self._device, dtype=self._dtype).unsqueeze(0),
                noise=noise,
                num_steps=self._policy_config.num_inference_steps,
                decode_image=False,
            )
        action_dim = int(self._action_dim or raw_state.shape[0])
        actions = self._denormalize_action(actions[0, :, :action_dim].float().cpu().numpy())
        if self.config.action_mode == "delta":
            base = raw_state.copy()
            for index in self.config.delta_zero_state_indices:
                if 0 <= int(index) < base.shape[0]:
                    base[int(index)] = 0.0
            actions = actions + base[: actions.shape[-1]]
        horizon = int(self.config.action_horizon or actions.shape[0])
        actions = actions[:horizon]
        return completed_action_result(
            model_id="internvla-a1",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=str(self._checkpoint_root or self.config.checkpoint_location),
            device=str(self._device),
            runtime="worldfoundry.internvla_a1.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "flow_steps": int(self._policy_config.num_inference_steps),
                "history_interval": int(self.config.image_history_interval),
                "camera_keys": list(self.config.camera_keys),
                "dtype": str(self._dtype),
                "seed": int(self.config.seed),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], InternVLAA1Runtime] = {}


def _required(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(
            f"InternVLA-A1 runtime option {key!r} is required; load its data runtime config"
        )
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
        raise ValueError("InternVLA-A1 checkpoint_path or checkpoint_ref is required")
    config = InternVLAA1RuntimeConfig(
        checkpoint_location=checkpoint,
        processor_location=str(_required(options, "processor_path")),
        cosmos_encoder_path=str(_required(options, "cosmos_encoder_path")),
        cosmos_decoder_path=str(_required(options, "cosmos_decoder_path")),
        model_defaults=dict(_required(options, "model_defaults")),
        camera_keys=tuple(_required(options, "camera_keys")),
        statistics_file=str(_required(options, "statistics_file")),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        local_files_only=option_bool(options.get("local_files_only"), True),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
        revision=str(options["revision"]) if options.get("revision") else None,
        processor_revision=(
            str(options["processor_revision"]) if options.get("processor_revision") else None
        ),
        image_size=option_int(options.get("image_size"), 224),
        tokenizer_max_length=option_int(options.get("tokenizer_max_length"), 48),
        image_history_interval=option_int(options.get("image_history_interval"), 15),
        action_horizon=(
            option_int(options.get("action_horizon"), 30)
            if options.get("action_horizon") is not None
            else None
        ),
        num_inference_steps=(
            option_int(options.get("num_inference_steps"), 10)
            if options.get("num_inference_steps") is not None
            else None
        ),
        seed=option_int(options.get("seed"), 0),
        statistics_path=(
            str(options["statistics_path"]) if options.get("statistics_path") else None
        ),
        statistics_key=(
            str(options["statistics_key"]) if options.get("statistics_key") else None
        ),
        normalization=str(options.get("normalization") or "mean_std"),
        action_mode=str(options.get("action_mode") or "absolute"),
        delta_zero_state_indices=tuple(options.get("delta_zero_state_indices") or ()),
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = InternVLAA1Runtime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["InternVLAA1Runtime", "InternVLAA1RuntimeConfig", "predict_action"]

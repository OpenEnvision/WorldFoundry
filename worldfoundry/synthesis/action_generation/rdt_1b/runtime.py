"""Native inference runtime for RDT-1B."""

from __future__ import annotations

import json
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
class RDT1BRuntimeConfig:
    checkpoint_location: str
    vision_encoder_location: str
    text_encoder_location: str
    device: str
    torch_dtype: str
    control_frequency: int
    denoising_steps: int
    seed: int
    release_text_encoder: bool
    embodiment: str
    state_indices: tuple[int, ...] | None
    state_scale: tuple[float, ...] | None
    action_scale: tuple[float, ...] | None
    camera_views: tuple[tuple[str, ...], ...]
    history_frames: int
    image_width: int
    image_height: int
    jpeg_roundtrip: bool

    def __post_init__(self) -> None:
        if self.control_frequency < 1:
            raise ValueError("RDT-1B control_frequency must be positive")
        if self.denoising_steps < 1:
            raise ValueError("RDT-1B denoising_steps must be positive")
        if self.history_frames < 1:
            raise ValueError("RDT-1B history_frames must be positive")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("RDT-1B image_width and image_height must be positive")


class RDT1BRuntime:
    """Persistent RDT/SigLIP runtime with prompt-embedding cache."""

    def __init__(self, config: RDT1BRuntimeConfig) -> None:
        self.config = config
        self._policy: Any = None
        self._vision_model: Any = None
        self._image_processor: Any = None
        self._tokenizer: Any = None
        self._text_model: Any = None
        self._text_cache: dict[str, tuple[Any, Any]] = {}
        self._device: str | None = None
        self._dtype: Any = None
        self._checkpoint_location: str | None = None
        self._vision_location: str | None = None
        self._text_location: str | None = None
        self._max_language_tokens: int | None = None
        self._action_dim: int | None = None
        self._frame_history: dict[str, list[list[Any | None]]] = {}

    def _load(self) -> tuple[Any, Any, Any]:
        if self._policy is not None:
            return self._policy, self._image_processor, self._vision_model

        import torch
        from transformers import SiglipImageProcessor, SiglipVisionModel

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .runner import RDTRunner

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        checkpoint = str(
            resolve_local_hf_model_path(
                self.config.checkpoint_location,
                required_files=("config.json", "pytorch_model.bin"),
            )
        )
        vision_location = str(
            resolve_local_hf_model_path(
                self.config.vision_encoder_location,
                required_files=("config.json", "preprocessor_config.json"),
            )
        )
        config_path = Path(checkpoint) / "config.json"
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        policy = RDTRunner(
            action_dim=int(model_config["action_dim"]),
            pred_horizon=int(model_config["pred_horizon"]),
            config=model_config,
            lang_token_dim=int(model_config["lang_token_dim"]),
            img_token_dim=int(model_config["img_token_dim"]),
            state_token_dim=int(model_config["state_token_dim"]),
            max_lang_cond_len=int(model_config["max_lang_cond_len"]),
            img_cond_len=int(model_config["img_cond_len"]),
            lang_pos_embed_config=model_config.get("lang_pos_embed_config"),
            img_pos_embed_config=model_config.get("img_pos_embed_config"),
            dtype=dtype,
        )
        weights = torch.load(
            Path(checkpoint) / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(weights, Mapping) and "state_dict" in weights:
            weights = weights["state_dict"]
        if not isinstance(weights, Mapping) or not weights or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in weights.items()
        ):
            raise TypeError(
                f"RDT-1B checkpoint must contain a non-empty string-to-tensor state dict: {checkpoint}"
            )
        policy.load_state_dict(weights, strict=True)
        policy.num_inference_timesteps = self.config.denoising_steps
        policy = policy.to(device=device, dtype=dtype).eval()

        image_processor = SiglipImageProcessor.from_pretrained(
            vision_location,
            local_files_only=True,
            trust_remote_code=False,
        )
        vision_model = SiglipVisionModel.from_pretrained(
            vision_location,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device=device, dtype=dtype).eval()

        self._policy = policy
        self._image_processor = image_processor
        self._vision_model = vision_model
        self._device = device
        self._dtype = dtype
        self._checkpoint_location = checkpoint
        self._vision_location = vision_location
        self._max_language_tokens = int(model_config["max_lang_cond_len"])
        self._action_dim = int(model_config["action_dim"])
        return policy, image_processor, vision_model

    def _encode_text(self, instruction: str) -> tuple[Any, Any]:
        cached = self._text_cache.get(instruction)
        if cached is not None:
            return cached[0].to(self._device), cached[1].to(self._device)

        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        if self._tokenizer is None or self._text_model is None:
            text_location = str(
                resolve_local_hf_model_path(
                    self.config.text_encoder_location,
                    required_files=("config.json",),
                )
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                text_location,
                model_max_length=self._max_language_tokens,
                local_files_only=True,
                trust_remote_code=False,
            )
            self._text_model = T5EncoderModel.from_pretrained(
                text_location,
                dtype=self._dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=False,
            ).to(device=self._device, dtype=self._dtype).eval()
            self._text_location = text_location

        tokens = self._tokenizer(
            instruction,
            max_length=self._max_language_tokens,
            padding=False,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self._device)
        attention_mask = tokens["attention_mask"].to(self._device).bool()
        with torch.inference_mode():
            embeddings = self._text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state.detach()
        self._text_cache[instruction] = (embeddings.cpu(), attention_mask.cpu())
        if self.config.release_text_encoder:
            self._text_model = None
            if str(self._device).startswith("cuda"):
                torch.cuda.empty_cache()
        return embeddings, attention_mask

    @staticmethod
    def _state(observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, "state", "proprio", "robot_state", "joint_state")
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(nested, "state", "proprio", "robot_state", "joint_state")
        return state

    def _camera_frame(self, value: Any) -> list[Any | None]:
        if isinstance(value, Mapping):
            result: list[Any | None] = []
            for aliases in self.config.camera_views:
                selected = next((value[key] for key in aliases if value.get(key) is not None), None)
                result.append(selected)
            return result if any(item is not None for item in result) else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        return [value] if value is not None else []

    def _images(self, image: Any, observation: Mapping[str, Any]) -> list[Any | None]:
        history = first_present(observation, "image_history", "images_history")
        frames: list[list[Any | None]] = []
        explicit_history = isinstance(history, Sequence) and not isinstance(history, (str, bytes, bytearray))
        if explicit_history:
            for value in history[-self.config.history_frames :]:
                frame = self._camera_frame(value)
                if frame:
                    frames.append(frame)
        current = first_present(observation, "images")
        if current is None:
            current = image
        current_frame = self._camera_frame(current)
        if not current_frame:
            current_frame = self._camera_frame(observation)
        history_key = str(first_present(observation, "env_idx", "environment_index") or 0)
        if not explicit_history:
            frames.extend(self._frame_history.get(history_key, []))
        if current_frame:
            frames.append(current_frame)
        if not frames:
            raise ValueError("RDT-1B requires RGB observations")
        if current_frame:
            retained_frames = self.config.history_frames - 1
            self._frame_history[history_key] = frames[-retained_frames:] if retained_frames else []
        while len(frames) < self.config.history_frames:
            frames.insert(0, [None] * len(self.config.camera_views))
        normalized_frames: list[Any | None] = []
        expected_views = len(self.config.camera_views)
        for frame in frames[-self.config.history_frames :]:
            if len(frame) > expected_views:
                raise ValueError(
                    f"RDT-1B expects {expected_views} camera views per history frame"
                )
            frame = [*frame, *([None] * (expected_views - len(frame)))]
            normalized_frames.extend(frame)
        return normalized_frames

    def _format_state(self, state: Any, action_mask: Any = None) -> tuple[Any, Any, tuple[int, ...] | None]:
        import numpy as np
        import torch

        if isinstance(state, torch.Tensor):
            values = state.detach().to(dtype=self._dtype, device=self._device).flatten()
        else:
            values = torch.as_tensor(np.asarray(state), dtype=self._dtype, device=self._device).flatten()
        configured = self.config.state_indices
        action_dim = int(self._action_dim or 0)
        if values.numel() == action_dim:
            universal = values
            indices = configured
            if action_mask is None:
                mask = torch.ones(action_dim, dtype=self._dtype, device=self._device)
            else:
                mask = torch.as_tensor(action_mask, dtype=self._dtype, device=self._device).flatten()
        else:
            indices = configured
            if indices is None or len(indices) != values.numel():
                raise ValueError(
                    "RDT-1B non-universal states require state_indices with one index per value"
                )
            if max(indices, default=-1) >= action_dim or min(indices, default=0) < 0:
                raise ValueError(f"RDT-1B state_indices must be within [0, {action_dim - 1}]")
            if self.config.state_scale is not None:
                if len(self.config.state_scale) != values.numel():
                    raise ValueError("RDT-1B state_scale must match the robot state dimension")
                scale = torch.tensor(self.config.state_scale, dtype=self._dtype, device=self._device)
                values = values / scale
            universal = torch.zeros(action_dim, dtype=self._dtype, device=self._device)
            universal[list(indices)] = values
            mask = torch.zeros(action_dim, dtype=self._dtype, device=self._device)
            mask[list(indices)] = 1
        if mask.numel() != action_dim:
            raise ValueError(f"RDT-1B action_mask must have {action_dim} values")
        return universal.view(1, 1, action_dim), mask.view(1, 1, action_dim), indices

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        from worldfoundry.core.media.processing.image_utils import load_pil_image
        from worldfoundry.core.utils.tensors.torch import set_seed_everywhere

        policy, image_processor, vision_model = self._load()
        state = self._state(observation)
        if state is None:
            raise ValueError("RDT-1B requires a proprioceptive state")
        state_tokens, action_mask, indices = self._format_state(
            state,
            first_present(observation, "action_mask"),
        )

        image_values = self._images(image, observation)
        background = tuple(int(channel * 255) for channel in image_processor.image_mean)
        padded_images = []
        valid_image_views = 0
        for value in image_values:
            if value is None:
                size = getattr(image_processor, "size", {})
                crop_size = getattr(image_processor, "crop_size", {})
                width = int(size.get("width") or crop_size.get("width") or 384)
                height = int(size.get("height") or crop_size.get("height") or 384)
                pil_image = Image.new("RGB", (width, height), background)
            else:
                pil_image = load_pil_image(value, first_sequence_item=False).convert("RGB")
                valid_image_views += 1
                if pil_image.size != (self.config.image_width, self.config.image_height) or self.config.jpeg_roundtrip:
                    import cv2

                    rgb = np.asarray(pil_image)
                    if rgb.shape[:2] != (self.config.image_height, self.config.image_width):
                        rgb = cv2.resize(
                            rgb,
                            (self.config.image_width, self.config.image_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    if self.config.jpeg_roundtrip:
                        encoded_ok, encoded = cv2.imencode(".jpg", rgb)
                        if not encoded_ok:
                            raise RuntimeError("RDT-1B failed to JPEG-standardize an input camera")
                        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        if decoded is None:
                            raise RuntimeError("RDT-1B failed to decode its standardized camera")
                        rgb = decoded
                    pil_image = Image.fromarray(rgb)
            width, height = pil_image.size
            if width != height:
                edge = max(width, height)
                padded = Image.new("RGB", (edge, edge), background)
                padded.paste(pil_image, ((edge - width) // 2, (edge - height) // 2))
                pil_image = padded
            padded_images.append(pil_image)
        pixels = image_processor(images=padded_images, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            image_embeds = vision_model(pixel_values=pixels).last_hidden_state.detach()
        image_embeds = image_embeds.reshape(1, -1, image_embeds.shape[-1])

        supplied_embeddings = first_present(observation, "text_embeddings", "language_embeddings")
        if supplied_embeddings is None:
            text_embeds, text_mask = self._encode_text(instruction)
        else:
            text_embeds = torch.as_tensor(
                np.asarray(supplied_embeddings),
                dtype=self._dtype,
                device=self._device,
            )
            if text_embeds.ndim == 2:
                text_embeds = text_embeds.unsqueeze(0)
            supplied_mask = first_present(observation, "text_attention_mask", "language_attention_mask")
            if supplied_mask is None:
                text_mask = torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=self._device)
            else:
                text_mask = torch.as_tensor(supplied_mask, dtype=torch.bool, device=self._device)
                if text_mask.ndim == 1:
                    text_mask = text_mask.unsqueeze(0)

        set_seed_everywhere(self.config.seed)
        control_frequency = torch.tensor([self.config.control_frequency], device=self._device)
        with torch.inference_mode():
            universal_actions = policy.predict_action(
                lang_tokens=text_embeds.to(device=self._device, dtype=self._dtype),
                lang_attn_mask=text_mask.to(self._device),
                img_tokens=image_embeds,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=control_frequency,
            ).to(torch.float32)
        actions = universal_actions
        if indices is not None:
            actions = universal_actions[..., list(indices)]
            if self.config.action_scale is not None:
                if len(self.config.action_scale) != len(indices):
                    raise ValueError("RDT-1B action_scale must match state_indices")
                scale = torch.tensor(
                    self.config.action_scale,
                    dtype=torch.float32,
                    device=actions.device,
                )
                actions = actions * scale
        actions = actions.cpu()
        return completed_action_result(
            model_id="rdt-1b",
            instruction=instruction,
            actions=actions,
            checkpoint_path=str(self._checkpoint_location),
            device=str(self._device),
            runtime="worldfoundry.rdt_1b.in_tree",
            metadata={
                "vision_encoder_path": self._vision_location,
                "text_encoder_path": self._text_location,
                "image_views": len(image_values),
                "valid_image_views": valid_image_views,
                "control_frequency": self.config.control_frequency,
                "denoising_steps": self.config.denoising_steps,
                "embodiment": self.config.embodiment,
                "universal_action_dim": self._action_dim,
                "seed": self.config.seed,
            },
        )


def _indices(value: Any) -> tuple[int, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _float_values(value: Any) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def _camera_views(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("RDT-1B camera_views must be a sequence of alias sequences")
    views = tuple(
        tuple(str(alias) for alias in aliases)
        for aliases in value
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes, bytearray))
    )
    if not views or any(not aliases for aliases in views):
        raise ValueError("RDT-1B camera_views cannot be empty")
    return views


def _required_option(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"RDT-1B runtime config requires {key!r}")
    return value


_RUNTIME_CACHE: dict[tuple[str, str, str], RDT1BRuntime] = {}


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
    """Run RDT-1B through the shared action-runtime contract."""

    del action_context
    options = dict(runtime_options or {})
    config = RDT1BRuntimeConfig(
        checkpoint_location=checkpoint_path,
        vision_encoder_location=str(_required_option(options, "vision_encoder_path")),
        text_encoder_location=str(_required_option(options, "text_encoder_path")),
        device=device,
        torch_dtype=str(_required_option(options, "torch_dtype")),
        control_frequency=option_int(_required_option(options, "control_frequency"), 0),
        denoising_steps=option_int(_required_option(options, "denoising_steps"), 0),
        seed=option_int(_required_option(options, "seed"), 0),
        release_text_encoder=option_bool(_required_option(options, "release_text_encoder")),
        embodiment=str(_required_option(options, "embodiment")),
        state_indices=_indices(_required_option(options, "state_indices")),
        state_scale=_float_values(_required_option(options, "state_scale")),
        action_scale=_float_values(_required_option(options, "action_scale")),
        camera_views=_camera_views(_required_option(options, "camera_views")),
        history_frames=option_int(_required_option(options, "history_frames"), 0),
        image_width=option_int(options.get("image_width"), 640),
        image_height=option_int(options.get("image_height"), 480),
        jpeg_roundtrip=option_bool(options.get("jpeg_roundtrip"), True),
    )
    # Per-call task text is carried in runtime_options by the public pipeline.
    # Cache by the parsed model configuration so changing the instruction does
    # not reload the 1B policy and SigLIP on every action step. T5 release is
    # governed separately by release_text_encoder.
    cache_key = (config.checkpoint_location, config.device, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.setdefault(cache_key, RDT1BRuntime(config))
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["RDT1BRuntime", "RDT1BRuntimeConfig", "predict_action"]

# SPDX-License-Identifier: MPL-2.0
"""Checkpoint-backed in-tree H-RDT action inference."""

from __future__ import annotations

import json
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

@dataclass(frozen=True)
class HRDTRuntimeConfig:
    checkpoint_location: str
    model_relative_dir: str
    model_files: Sequence[str]
    vision_relative_dir: str
    vision_required_files: Sequence[str]
    vision_image_size: int
    vision_tokens_per_view: int
    device: str = "cuda"
    torch_dtype: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = True
    revision: str | None = None
    attention_backend: str = "auto"
    seed: int = 0
    num_inference_timesteps: int = 5
    vision_encoder_path: str | None = None
    text_encoder_path: str | None = None
    text_encoder_ref: str | None = None
    text_encoder_device: str | None = None
    max_language_length: int = 120
    image_resize_strategy: str = "letterbox"
    statistics_path: str | None = None
    statistics_key: str | None = None
    normalize_state: bool = False
    denormalize_action: bool = False


class HRDTRuntime:
    """Persistent H-RDT policy, vision encoder, and optional T5 encoder."""

    def __init__(self, config: HRDTRuntimeConfig) -> None:
        if not config.local_files_only:
            raise ValueError("H-RDT requires local_files_only=true; runtime downloads are disabled")
        if not config.model_relative_dir or not config.model_files:
            raise ValueError("H-RDT model_relative_dir and model_files are required")
        if not config.vision_relative_dir or not config.vision_required_files:
            raise ValueError("H-RDT vision checkpoint layout is required")
        if config.vision_image_size <= 0 or config.vision_tokens_per_view <= 0:
            raise ValueError("H-RDT vision image/token dimensions must be positive")
        self.config = config
        self._model: Any = None
        self._vision_encoder: Any = None
        self._text_encoder: Any = None
        self._tokenizer: Any = None
        self._text_cache: dict[str, tuple[Any, Any]] = {}
        self._device: str | None = None
        self._dtype: Any = None
        self._snapshot: Path | None = None
        self._model_dir: Path | None = None
        self._model_config: dict[str, Any] | None = None

    def _direct_model_dir(self, path: Path) -> Path | None:
        if all((path / name).is_file() for name in self.config.model_files):
            return path
        nested = path / self.config.model_relative_dir
        if all((nested / name).is_file() for name in self.config.model_files):
            return nested
        return None

    def _resolve_snapshot(self) -> tuple[Path, Path]:
        if self._snapshot is not None and self._model_dir is not None:
            return self._snapshot, self._model_dir

        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot
        from worldfoundry.core.io.paths import (
            resolve_local_hf_model_path,
            resolve_worldfoundry_path,
        )

        location = self.config.checkpoint_location
        root_files = tuple(
            str(Path(self.config.model_relative_dir) / name)
            for name in self.config.model_files
        )
        direct = resolve_worldfoundry_path(location)
        if direct.exists():
            direct = direct.resolve()
            model_dir = self._direct_model_dir(direct)
            if model_dir is None:
                raise FileNotFoundError(
                    f"H-RDT checkpoint directory {direct} does not contain "
                    f"{tuple(self.config.model_files)} or {self.config.model_relative_dir}"
                )
            snapshot = direct if model_dir != direct else direct
        else:
            try:
                snapshot = resolve_local_hf_model_path(location, required_files=root_files)
            except FileNotFoundError:
                snapshot = materialize_hf_snapshot(
                    location,
                    revision=self.config.revision,
                    cache_dir=self.config.cache_dir,
                    required_files=root_files,
                    local_files_only=self.config.local_files_only,
                )
            model_dir = self._direct_model_dir(snapshot)
            if model_dir is None:
                raise FileNotFoundError(f"resolved H-RDT snapshot {snapshot} has no policy checkpoint")
        self._snapshot = snapshot
        self._model_dir = model_dir
        return snapshot, model_dir

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        import torch

        from worldfoundry.core.model_loading.checkpoints.assignment import assign_state_dict_strict
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.core.model_loading.checkpoints.file import load_torch_state_dict

        from .modeling import HRDTRunner

        _, model_dir = self._resolve_snapshot()
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        with torch.device("meta"):
            model = HRDTRunner(
                model_config,
                dtype=dtype,
                attention_backend=self.config.attention_backend,
            )
        state_dict = load_torch_state_dict(model_dir / "pytorch_model.bin", map_location="cpu")
        for outer_key in ("state_dict", "model", "module"):
            if (
                isinstance(state_dict, Mapping)
                and len(state_dict) == 1
                and isinstance(state_dict.get(outer_key), Mapping)
            ):
                state_dict = state_dict[outer_key]
                break
        if not isinstance(state_dict, Mapping):
            raise TypeError("H-RDT checkpoint does not contain a state dictionary")
        assign_state_dict_strict(model, state_dict, label="H-RDT official checkpoint")
        model = model.to(device=device, dtype=dtype).eval()
        self._model = model
        self._device = device
        self._dtype = dtype
        self._model_config = model_config
        return model

    def _vision_root(self) -> Path:
        snapshot, model_dir = self._resolve_snapshot()
        candidates: list[Path] = []
        if self.config.vision_encoder_path:
            from worldfoundry.core.io.paths import resolve_worldfoundry_path

            candidates.append(resolve_worldfoundry_path(self.config.vision_encoder_path))
        candidates.extend(
            [
                snapshot / self.config.vision_relative_dir,
                model_dir / self.config.vision_relative_dir,
                model_dir.parent / self.config.vision_relative_dir,
                model_dir.parent.parent / self.config.vision_relative_dir,
                model_dir.parent.parent.parent / self.config.vision_relative_dir,
            ]
        )
        for candidate in candidates:
            if all((candidate / relative).is_file() for relative in self.config.vision_required_files):
                return candidate.resolve()
        raise FileNotFoundError(
            "H-RDT raw-image inference requires the official DINOv2 and SigLIP files; "
            "provide vision_encoder_path or a complete official snapshot"
        )

    def _load_vision(self) -> Any:
        if self._vision_encoder is None:
            from .vision import DinoSigLIPEncoder

            self._load_model()
            self._vision_encoder = DinoSigLIPEncoder(
                self._vision_root(),
                device=str(self._device),
                dtype=self._dtype,
                image_size=self.config.vision_image_size,
                resize_strategy=self.config.image_resize_strategy,
            )
        return self._vision_encoder

    def _text_location(self) -> str:
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        if self.config.text_encoder_path:
            path = resolve_worldfoundry_path(self.config.text_encoder_path)
            if path.exists():
                return str(path.resolve())
            if self.config.local_files_only and not self.config.text_encoder_ref:
                raise FileNotFoundError(f"H-RDT text encoder path does not exist: {path}")
        if self.config.text_encoder_ref:
            return self.config.text_encoder_ref
        raise ValueError(
            "H-RDT needs language_tokens/language_embeddings in the observation or "
            "a text_encoder_path/text_encoder_ref"
        )

    def _load_text_encoder(self) -> tuple[Any, Any]:
        if self._text_encoder is not None and self._tokenizer is not None:
            return self._tokenizer, self._text_encoder

        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        self._load_model()
        location = materialize_hf_snapshot(
            self._text_location(),
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            location,
            model_max_length=int(self.config.max_language_length),
            local_files_only=True,
            trust_remote_code=False,
        )
        has_safetensors = any(location.glob("model*.safetensors")) or any(
            location.glob("model*.safetensors.index.json")
        )
        has_pytorch = any(location.glob("pytorch_model*.bin")) or any(
            location.glob("pytorch_model*.bin.index.json")
        )
        if not has_safetensors and not has_pytorch:
            raise FileNotFoundError(f"H-RDT text encoder has no local weight files: {location}")
        text_device = self.config.text_encoder_device or str(self._device)
        encoder_dtype = torch.float32 if torch.device(text_device).type == "cpu" else self._dtype
        encoder = T5EncoderModel.from_pretrained(
            location,
            torch_dtype=encoder_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=has_safetensors,
        ).to(text_device).eval()
        self._tokenizer = tokenizer
        self._text_encoder = encoder
        return tokenizer, encoder

    def _encode_instruction(self, instruction: str) -> tuple[Any, Any]:
        import torch

        cached = self._text_cache.get(instruction)
        if cached is not None:
            return cached
        tokenizer, encoder = self._load_text_encoder()
        text_device = next(encoder.parameters()).device
        encoded = tokenizer(
            [instruction],
            max_length=int(self.config.max_language_length),
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(text_device)
        attention_mask = encoded["attention_mask"].to(text_device)
        with torch.inference_mode():
            tokens = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        result = (
            tokens.to(device=self._device, dtype=self._dtype),
            attention_mask.to(device=self._device, dtype=torch.bool),
        )
        self._text_cache[instruction] = result
        return result

    @staticmethod
    def _state(observation: Mapping[str, Any]) -> Any:
        state = first_present(
            observation,
            "state",
            "agent_pos",
            "proprio",
            "robot_state",
            "joint_state",
        )
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(
                nested,
                "state",
                "agent_pos",
                "proprio",
                "robot_state",
                "joint_state",
            )
        return state

    def _statistics(self) -> Mapping[str, Any] | None:
        if not self.config.statistics_path:
            return None
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        path = resolve_worldfoundry_path(self.config.statistics_path)
        if not path.is_file():
            raise FileNotFoundError(f"H-RDT statistics file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if self.config.statistics_key:
            payload = payload[self.config.statistics_key]
        return payload if isinstance(payload, Mapping) else None

    @staticmethod
    def _range_from_stats(stats: Mapping[str, Any] | None, kind: str) -> tuple[Any, Any] | None:
        if not stats:
            return None
        selected = stats.get(kind)
        if isinstance(selected, Mapping) and "min" in selected and "max" in selected:
            return selected["min"], selected["max"]
        if "min" in stats and "max" in stats:
            return stats["min"], stats["max"]
        return None

    def _language_inputs(
        self,
        instruction: str,
        observation: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        import torch

        tokens = first_present(
            observation,
            "language_tokens",
            "lang_tokens",
            "language_embeddings",
            "lang_embeddings",
        )
        mask = first_present(
            observation,
            "language_attention_mask",
            "lang_attention_mask",
            "lang_attn_mask",
        )
        if tokens is None:
            return self._encode_instruction(instruction)
        tokens = torch.as_tensor(tokens, device=self._device, dtype=self._dtype)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError("H-RDT language tokens must have shape [tokens, 4096] or [batch, tokens, 4096]")
        model = self._model
        if model is None:
            model = self._load_model()
        if tokens.shape[0] != 1:
            raise ValueError(f"H-RDT runtime expects one language batch, got {tokens.shape[0]}")
        if tokens.shape[-1] != model.language_feature_dim:
            raise ValueError(
                f"H-RDT language tokens must end in {model.language_feature_dim}, "
                f"got {tokens.shape[-1]}"
            )
        if tokens.shape[1] > model.max_lang_len:
            raise ValueError(
                f"H-RDT language token length {tokens.shape[1]} exceeds {model.max_lang_len}"
            )
        if mask is None:
            mask = torch.ones(tokens.shape[:2], device=self._device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=self._device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                "H-RDT language_attention_mask must match the language token dimensions; "
                f"got {tuple(mask.shape)} for tokens {tuple(tokens.shape)}"
            )
        return tokens, mask

    def _image_inputs(
        self,
        observation: Mapping[str, Any],
        image: Any,
        expected_tokens: int,
    ) -> tuple[Any, int]:
        import torch

        precomputed = first_present(
            observation,
            "image_tokens",
            "vision_tokens",
            "image_embeddings",
            "vision_embeddings",
        )
        if precomputed is not None:
            tokens = torch.as_tensor(precomputed, device=self._device, dtype=self._dtype)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim == 4:
                tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
            if tokens.ndim != 3:
                raise ValueError("H-RDT image tokens must have shape [tokens, 2176] or [batch, tokens, 2176]")
            return tokens, max(1, int(tokens.shape[1]) // self.config.vision_tokens_per_view)

        from worldfoundry.core.media.processing.image_utils import load_pil_image

        values = collect_images(
            observation,
            image,
            (
                "head_cam",
                "right_cam",
                "left_cam",
                "head_camera",
                "right_camera",
                "left_camera",
                "image0",
                "image1",
                "image2",
                "full_image",
                "wrist_image",
            ),
        )
        expected_views = max(1, expected_tokens // self.config.vision_tokens_per_view)
        if len(values) != expected_views:
            raise ValueError(
                f"H-RDT checkpoint expects {expected_views} image view(s), got {len(values)}; "
                "pass the exact camera set or precomputed image_tokens"
            )
        images = [load_pil_image(value, first_sequence_item=False) for value in values]
        return self._load_vision().encode(images), len(images)

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
        state = self._state(observation)
        if state is None:
            raise ValueError(f"H-RDT requires a {model.state_dim}D state/agent_pos vector")
        state_tensor = torch.as_tensor(np.asarray(state), device=self._device, dtype=self._dtype)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
        elif state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(1)
        if tuple(state_tensor.shape[:2]) != (1, 1):
            raise ValueError(f"H-RDT runtime expects one current state, got {tuple(state_tensor.shape)}")

        stats = self._statistics()
        if self.config.normalize_state:
            state_range = self._range_from_stats(stats, "state")
            if state_range is None:
                raise ValueError("normalize_state=true but the statistics file has no state min/max")
            minimum = torch.as_tensor(state_range[0], device=self._device, dtype=self._dtype)
            maximum = torch.as_tensor(state_range[1], device=self._device, dtype=self._dtype)
            state_tensor = (state_tensor - minimum) / torch.clamp(maximum - minimum, min=1e-6)

        image_tokens, image_views = self._image_inputs(observation, image, model.max_img_len)
        language_tokens, language_mask = self._language_inputs(instruction, observation)
        generator = torch.Generator(device=torch.device(self._device))
        generator.manual_seed(int(self.config.seed))
        actions = model.predict_action(
            state_tokens=state_tensor,
            image_tokens=image_tokens,
            lang_tokens=language_tokens,
            lang_attn_mask=language_mask,
            generator=generator,
            num_inference_timesteps=int(self.config.num_inference_timesteps),
        )[0].float().cpu()

        if self.config.denormalize_action:
            action_range = self._range_from_stats(stats, "action")
            if action_range is None:
                raise ValueError("denormalize_action=true but the statistics file has no action min/max")
            minimum = torch.as_tensor(action_range[0], dtype=actions.dtype)
            maximum = torch.as_tensor(action_range[1], dtype=actions.dtype)
            actions = actions * (maximum - minimum) + minimum

        return completed_action_result(
            model_id="h-rdt",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=str(self._model_dir or self.config.checkpoint_location),
            device=str(self._device),
            runtime="worldfoundry.h_rdt.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "state_dim": model.state_dim,
                "action_dim": model.action_dim,
                "image_views": image_views,
                "image_tokens": int(image_tokens.shape[1]),
                "language_tokens": int(language_tokens.shape[1]),
                "flow_steps": int(self.config.num_inference_timesteps),
                "dtype": str(self._dtype),
                "seed": int(self.config.seed),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], HRDTRuntime] = {}


def _required_option(options: Mapping[str, Any], name: str) -> Any:
    value = options.get(name)
    if value in (None, ""):
        raise ValueError(
            f"H-RDT runtime option {name!r} is required; load the model's data runtime config"
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
    """Callable WorldFoundry entrypoint for H-RDT inference."""

    del action_context
    options = dict(runtime_options or {})
    checkpoint = (
        checkpoint_path
        or str(options.get("checkpoint_ref") or "")
        or str(options.get("repo_id") or "")
    )
    if not checkpoint:
        raise ValueError("H-RDT checkpoint_path or checkpoint_ref is required")
    config = HRDTRuntimeConfig(
        checkpoint_location=checkpoint,
        model_relative_dir=str(_required_option(options, "model_relative_dir")),
        model_files=tuple(_required_option(options, "model_files")),
        vision_relative_dir=str(_required_option(options, "vision_relative_dir")),
        vision_required_files=tuple(_required_option(options, "vision_required_files")),
        vision_image_size=int(_required_option(options, "vision_image_size")),
        vision_tokens_per_view=int(_required_option(options, "vision_tokens_per_view")),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
        local_files_only=option_bool(options.get("local_files_only"), True),
        revision=str(options["revision"]) if options.get("revision") else None,
        attention_backend=str(options.get("attention_backend") or "auto"),
        seed=option_int(options.get("seed"), 0),
        num_inference_timesteps=option_int(
            options.get("num_inference_timesteps") or options.get("denoising_steps"),
            5,
        ),
        vision_encoder_path=(
            str(options["vision_encoder_path"]) if options.get("vision_encoder_path") else None
        ),
        text_encoder_path=(
            str(options["text_encoder_path"]) if options.get("text_encoder_path") else None
        ),
        text_encoder_ref=(
            str(options["text_encoder_ref"])
            if options.get("text_encoder_ref") is not None
            else None
        ),
        text_encoder_device=(
            str(options["text_encoder_device"]) if options.get("text_encoder_device") else None
        ),
        max_language_length=option_int(options.get("max_language_length"), 120),
        image_resize_strategy=str(options.get("image_resize_strategy") or "letterbox"),
        statistics_path=(
            str(options["statistics_path"]) if options.get("statistics_path") else None
        ),
        statistics_key=(
            str(options["statistics_key"]) if options.get("statistics_key") else None
        ),
        normalize_state=option_bool(options.get("normalize_state"), False),
        denormalize_action=option_bool(options.get("denormalize_action"), False),
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = HRDTRuntime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["HRDTRuntime", "HRDTRuntimeConfig", "predict_action"]

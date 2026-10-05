"""Local-only, component-parallel LDA-1B inference runtime."""

from __future__ import annotations

import gc
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
class LDA1BRuntimeConfig:
    checkpoint_location: str
    checkpoint_filename: str
    qwen_location: str
    vision_location: str
    action_config: Mapping[str, Any]
    variant: Mapping[str, Any]
    prompt_template: str
    device: str = "cuda"
    vlm_device: str | None = None
    action_device: str | None = None
    torch_dtype: str = "auto"
    attention_backend: str = "auto"
    local_files_only: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    qwen_revision: str | None = None
    vision_revision: str | None = None
    statistics_path: str | None = None
    statistics_file: str = "dataset_statistics.json"
    statistics_key: str | None = None
    action_horizon: int | None = None
    seed: int = 42
    compile_action_model: bool = False
    compile_vision_encoder: bool = False
    compile_mode: str = "reduce-overhead"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"LDA-1B {label} must be a mapping")
    return value


class LDA1BRuntime:
    """One persistent released-checkpoint runtime with optional two-GPU placement."""

    def __init__(self, config: LDA1BRuntimeConfig) -> None:
        if not config.local_files_only:
            raise ValueError("LDA-1B in-tree inference requires local_files_only=true")
        if not config.checkpoint_filename.endswith(".pt"):
            raise ValueError("LDA-1B checkpoint_filename must name a .pt state dict")
        self.config = config
        self.variant = _mapping(config.variant, "variant")
        self._checkpoint_root: Path | None = None
        self._checkpoint_file: Path | None = None
        self._qwen_root: Path | None = None
        self._vision_root: Path | None = None
        self._statistics: Mapping[str, Any] | None = None
        self._model: Any = None
        self._qwen_processor: Any = None
        self._vision_processor: Any = None
        self._vlm_device: str | None = None
        self._action_device: str | None = None
        self._vlm_dtype: Any = None
        self._action_dtype: Any = None
        self._attention_implementation: str | None = None

    @staticmethod
    def _local_path(value: str | None) -> Path | None:
        if not value:
            return None
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        path = resolve_worldfoundry_path(value)
        return path.resolve() if path.exists() else None

    def _materialize(
        self,
        location: str,
        *,
        revision: str | None,
        required_files: Sequence[str] = (),
    ) -> Path:
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        direct = self._local_path(location)
        source = str(direct) if direct is not None else location
        return materialize_hf_snapshot(
            source,
            revision=revision,
            cache_dir=self.config.cache_dir,
            required_files=tuple(required_files),
            local_files_only=True,
        )

    def _resolve_checkpoint(self) -> tuple[Path, Path]:
        if self._checkpoint_root is not None and self._checkpoint_file is not None:
            return self._checkpoint_root, self._checkpoint_file
        direct = self._local_path(self.config.checkpoint_location)
        if direct is not None and direct.is_file():
            root, checkpoint = direct.parent, direct
        else:
            root = self._materialize(
                self.config.checkpoint_location,
                revision=self.config.revision,
            )
            candidates = (
                root / self.config.checkpoint_filename,
                root / "checkpoints" / self.config.checkpoint_filename,
            )
            checkpoint = next((path for path in candidates if path.is_file()), candidates[0])
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"LDA-1B checkpoint is not staged: expected {self.config.checkpoint_filename} "
                f"at the snapshot root or checkpoints/ under {root}"
            )
        self._checkpoint_root = root.resolve()
        self._checkpoint_file = checkpoint.resolve()
        return self._checkpoint_root, self._checkpoint_file

    def _resolve_qwen(self) -> Path:
        if self._qwen_root is None:
            self._qwen_root = self._materialize(
                self.config.qwen_location,
                revision=self.config.qwen_revision,
                required_files=("config.json", "tokenizer_config.json"),
            )
        return self._qwen_root

    def _resolve_vision(self) -> Path:
        if self._vision_root is None:
            self._vision_root = self._materialize(
                self.config.vision_location,
                revision=self.config.vision_revision,
                required_files=("config.json", "preprocessor_config.json"),
            )
        return self._vision_root

    @staticmethod
    def _load_state_dict(path: Path) -> Mapping[str, Any]:
        import torch

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("LDA-1B checkpoint does not contain a non-empty state dict")
        state = dict(payload)
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
            raise ValueError("LDA-1B checkpoint state dict must contain tensor values only")
        return state

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .dinov3_configuration import DINOv3ViTConfig
        from .modeling import LDAInferenceModel

        _, checkpoint = self._resolve_checkpoint()
        qwen_root = self._resolve_qwen()
        vision_root = self._resolve_vision()
        base_device = resolve_inference_device(self.config.device)
        vlm_device = resolve_inference_device(self.config.vlm_device or base_device)
        action_device = resolve_inference_device(self.config.action_device or base_device)
        vlm_dtype = resolve_inference_dtype(vlm_device, self.config.torch_dtype)
        action_dtype = resolve_inference_dtype(action_device, self.config.torch_dtype)
        attention_implementation = resolve_transformers_attention_implementation(
            self.config.attention_backend,
            vlm_device,
        )
        qwen_config = AutoConfig.from_pretrained(
            qwen_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        qwen_config._attn_implementation = attention_implementation
        if hasattr(qwen_config, "text_config"):
            qwen_config.text_config._attn_implementation = attention_implementation
        vision_config = DINOv3ViTConfig.from_pretrained(
            vision_root,
            local_files_only=True,
        )
        with init_empty_weights(include_buffers=False):
            model = LDAInferenceModel(
                qwen_config=qwen_config,
                action_config=self.config.action_config,
                vision_config=vision_config,
            )
        state = self._load_state_dict(checkpoint)
        try:
            missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
        except TypeError as exc:
            raise RuntimeError("LDA-1B meta loading requires load_state_dict(assign=True)") from exc
        if missing or unexpected:
            raise RuntimeError(
                f"LDA-1B checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
            )
        model.qwen_vl_interface.model.tie_weights()
        model.qwen_vl_interface.to(device=vlm_device, dtype=vlm_dtype).eval()
        model.action_model.to(device=action_device, dtype=action_dtype).eval()
        del state
        gc.collect()
        if self.config.compile_vision_encoder:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_vision_encoder requires torch.compile")
            model.action_model.vision_encoder = torch.compile(
                model.action_model.vision_encoder,
                mode=self.config.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        if self.config.compile_action_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_action_model requires torch.compile")
            model.action_model.model = torch.compile(
                model.action_model.model,
                mode=self.config.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        self._qwen_processor = AutoProcessor.from_pretrained(
            qwen_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._qwen_processor.tokenizer.padding_side = "left"
        self._vision_processor = AutoImageProcessor.from_pretrained(
            vision_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._model = model
        self._vlm_device = str(vlm_device)
        self._action_device = str(action_device)
        self._vlm_dtype = vlm_dtype
        self._action_dtype = action_dtype
        self._attention_implementation = attention_implementation
        return model

    def _statistics_payload(self) -> Mapping[str, Any]:
        if self._statistics is not None:
            return self._statistics
        explicit = self._local_path(self.config.statistics_path)
        if explicit is not None:
            path = explicit
        else:
            root, _ = self._resolve_checkpoint()
            path = root / self.config.statistics_file
        if not path.is_file():
            raise FileNotFoundError(
                "LDA-1B requires dataset_statistics.json paired with the staged checkpoint: "
                f"{path}"
            )
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if self.config.statistics_key:
            for key in self.config.statistics_key.split("."):
                payload = payload[key]
        payload = _mapping(payload, "statistics")
        if "action" not in payload:
            raise ValueError("LDA-1B selected statistics block has no action section")
        self._statistics = payload
        return payload

    def _state_value(self, observation: Mapping[str, Any]) -> Any:
        value = first_present(
            observation,
            "state",
            "proprio",
            "joint_state",
            "robot_state",
        )
        nested = observation.get("observation")
        if value is None and isinstance(nested, Mapping):
            value = first_present(nested, "state", "proprio", "joint_state", "robot_state")
        if value is None:
            parts = self.variant["state_parts"]
            if all(str(part["key"]) in observation for part in parts):
                value = observation
            elif all(f"state.{part['key']}" in observation for part in parts):
                value = {
                    str(part["key"]): observation[f"state.{part['key']}"]
                    for part in parts
                }
        if value is None:
            raise ValueError("LDA-1B requires robot state/proprio input")
        return value

    def _image_value(self, observation: Mapping[str, Any], image: Any) -> Any:
        camera_key = str(self.variant["camera_key"])
        aliases = tuple(str(item) for item in self.variant.get("camera_aliases", ()))
        names = (camera_key, *aliases)
        containers: list[Mapping[str, Any]] = []
        for key in ("images", "vision", "observation"):
            nested = observation.get(key)
            if isinstance(nested, Mapping):
                containers.append(nested)
        containers.append(observation)
        if isinstance(image, Mapping):
            containers.insert(0, image)
        for container in containers:
            selected = first_present(container, *names)
            if selected is not None:
                return selected
        if image is not None and not isinstance(image, Mapping):
            return image
        raise ValueError(f"LDA-1B requires camera {camera_key!r}; accepted aliases are {names}")

    @staticmethod
    def _move_batch(batch: Mapping[str, Any], *, device: str, dtype: Any) -> dict[str, Any]:
        import torch

        output: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device=device)
                if value.is_floating_point():
                    value = value.to(dtype=dtype)
            output[key] = value
        return output

    def _qwen_inputs(self, frames: Sequence[Any], instruction: str) -> Mapping[str, Any]:
        prompt = self.config.prompt_template.format(instruction=instruction)
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": prompt})
        messages = [[{"role": "user", "content": content}]]
        return self._qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
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

        from .preprocessing import (
            denormalize_minmax,
            pack_parts,
            select_history,
            state_sincos,
            temporal_images,
        )

        model = self._load_model()
        steps = int(self.variant["observation_steps"])
        frames = select_history(temporal_images(self._image_value(observation, image)), steps)
        image_size = tuple(int(item) for item in self.variant["image_size"])
        frames = [frame.resize(image_size) for frame in frames]
        raw_state = pack_parts(
            self._state_value(observation),
            self.variant["state_parts"],
            label="state",
        )
        encoded_state = state_sincos(
            raw_state,
            steps=steps,
            expected_dim=int(self.variant["model_state_dim"]),
        )
        qwen_inputs = self._move_batch(
            self._qwen_inputs(frames, instruction),
            device=self._vlm_device,
            dtype=self._vlm_dtype,
        )
        vision_pixels = self._vision_processor(images=list(frames), return_tensors="pt")[
            "pixel_values"
        ]
        vision_pixels = vision_pixels.reshape(1, 1, steps, *vision_pixels.shape[1:])
        with torch.inference_mode():
            outputs = model.qwen_vl_interface.model(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1].to(
                device=self._action_device,
                dtype=self._action_dtype,
            )
            attention_mask = qwen_inputs["attention_mask"].to(
                device=self._action_device,
                dtype=self._action_dtype,
            )
            state = torch.as_tensor(
                encoded_state,
                device=self._action_device,
                dtype=self._action_dtype,
            ).unsqueeze(0)
            pixels = vision_pixels.to(device=self._action_device, dtype=self._action_dtype)
            embodiment_id = torch.tensor(
                [int(self.variant["embodiment_id"])],
                device=self._action_device,
                dtype=torch.long,
            )
            generator = torch.Generator(device=self._action_device)
            generator.manual_seed(int(self.config.seed))
            normalized_padded = model.action_model.predict_action(
                hidden,
                state=state,
                curr_pixels=pixels,
                embodiment_id=embodiment_id,
                attention_mask=attention_mask,
                generator=generator,
            )[0]
        raw_dim = int(self.variant["raw_action_dim"])
        normalized = normalized_padded[:, :raw_dim].float().cpu().numpy()
        action_stats = _mapping(self._statistics_payload()["action"], "action statistics")
        actions = denormalize_minmax(normalized, action_stats, expected_dim=raw_dim)
        horizon = min(int(self.config.action_horizon or actions.shape[0]), actions.shape[0])
        actions = np.asarray(actions[:horizon], dtype=np.float32)
        return completed_action_result(
            model_id="lda-1b",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=str(self._checkpoint_file),
            device=str(self._action_device),
            runtime="worldfoundry.lda_1b.in_tree_runtime",
            metadata={
                "variant": str(self.variant["name"]),
                "action_shape": list(actions.shape),
                "model_action_dim": int(self.config.action_config["action_dim"]),
                "raw_action_dim": raw_dim,
                "vlm_device": str(self._vlm_device),
                "action_device": str(self._action_device),
                "vlm_dtype": str(self._vlm_dtype),
                "action_dtype": str(self._action_dtype),
                "attention_implementation": str(self._attention_implementation),
                "flow_steps": int(self.config.action_config["num_inference_timesteps"]),
                "compile_action_model": bool(self.config.compile_action_model),
                "compile_vision_encoder": bool(self.config.compile_vision_encoder),
                "seed": int(self.config.seed),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], LDA1BRuntime] = {}


def _required(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"LDA-1B runtime option {key!r} is required; load its data config")
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
    """Callable entrypoint used by the shared action-policy runtime."""

    del action_context
    options = dict(runtime_options or {})
    checkpoint = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not checkpoint:
        raise ValueError("LDA-1B checkpoint_path or checkpoint_ref is required")
    config = LDA1BRuntimeConfig(
        checkpoint_location=checkpoint,
        checkpoint_filename=str(_required(options, "checkpoint_filename")),
        qwen_location=str(_required(options, "qwen_checkpoint_path")),
        vision_location=str(_required(options, "vision_checkpoint_path")),
        action_config=dict(_required(options, "action_config")),
        variant=dict(_required(options, "variant")),
        prompt_template=str(_required(options, "prompt_template")),
        device=device,
        vlm_device=str(options["vlm_device"]) if options.get("vlm_device") else None,
        action_device=str(options["action_device"]) if options.get("action_device") else None,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        attention_backend=str(options.get("attention_backend") or "auto"),
        local_files_only=option_bool(options.get("local_files_only"), True),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
        revision=str(options["revision"]) if options.get("revision") else None,
        qwen_revision=str(options["qwen_revision"]) if options.get("qwen_revision") else None,
        vision_revision=(
            str(options["vision_revision"]) if options.get("vision_revision") else None
        ),
        statistics_path=str(options["statistics_path"]) if options.get("statistics_path") else None,
        statistics_file=str(options.get("statistics_file") or "dataset_statistics.json"),
        statistics_key=str(options["statistics_key"]) if options.get("statistics_key") else None,
        action_horizon=(
            option_int(options.get("action_horizon"), 0)
            if options.get("action_horizon") is not None
            else None
        ),
        seed=option_int(options.get("seed"), 42),
        compile_action_model=option_bool(options.get("compile_action_model"), False),
        compile_vision_encoder=option_bool(options.get("compile_vision_encoder"), False),
        compile_mode=str(options.get("compile_mode") or "reduce-overhead"),
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = LDA1BRuntime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["LDA1BRuntime", "LDA1BRuntimeConfig", "predict_action"]

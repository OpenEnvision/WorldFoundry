"""Local-only G0Plus checkpoint runtime."""

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
class GalaxeaVLARuntimeConfig:
    checkpoint_location: str
    tokenizer_location: str
    checkpoint_subdir: str
    variant: str
    architecture: Mapping[str, Any]
    variants: Mapping[str, Mapping[str, Any]]
    prompt_template: str
    instruction_template: str
    normalization_epsilon: float
    image_mean: Sequence[float]
    image_std: Sequence[float]
    device: str = "cuda"
    torch_dtype: str = "auto"
    local_files_only: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    statistics_path: str | None = None
    statistics_file: str = "dataset_stats.json"
    statistics_key: str | None = None
    action_horizon: int | None = None
    seed: int = 0
    compile_vision: bool = False
    compile_mode: str = "reduce-overhead"


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"G0Plus {label} must be a mapping")
    return value


class GalaxeaVLARuntime:
    """Persistent, in-process inference runtime for one G0Plus variant."""

    def __init__(self, config: GalaxeaVLARuntimeConfig) -> None:
        if not config.local_files_only:
            raise ValueError("G0Plus is an in-tree runtime and requires local_files_only=true")
        if config.variant not in config.variants:
            raise ValueError(
                f"Unknown G0Plus variant {config.variant!r}; choose one of {sorted(config.variants)}"
            )
        if len(config.image_mean) != 3 or len(config.image_std) != 3:
            raise ValueError("G0Plus image mean/std must each contain three values")
        self.config = config
        self.variant = _required_mapping(config.variants[config.variant], "variant")
        self._checkpoint_root: Path | None = None
        self._checkpoint_file: Path | None = None
        self._tokenizer_root: Path | None = None
        self._statistics: Mapping[str, Any] | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dtype: Any = None

    @staticmethod
    def _local_path(value: str | None) -> Path | None:
        if not value:
            return None
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        path = resolve_worldfoundry_path(value)
        return path.resolve() if path.exists() else None

    def _resolve_checkpoint(self) -> tuple[Path, Path]:
        if self._checkpoint_root is not None and self._checkpoint_file is not None:
            return self._checkpoint_root, self._checkpoint_file
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        direct = self._local_path(self.config.checkpoint_location)
        if direct is not None and direct.is_file():
            root, checkpoint = direct.parent, direct
        else:
            source = str(direct) if direct is not None else self.config.checkpoint_location
            snapshot = materialize_hf_snapshot(
                source,
                revision=self.config.revision,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
            )
            roots = [snapshot]
            if self.config.checkpoint_subdir:
                roots.insert(0, snapshot / self.config.checkpoint_subdir)
            # The pinned G0-VLA snapshot has used both released layouts below.
            # Keep this list explicit: checkpoint discovery must never recurse into
            # arbitrary pickle files in the repository (which also contains data).
            candidates = [
                candidate
                for root in roots
                for candidate in (
                    (root, root / "model.pt"),
                    (root / "checkpoints", root / "checkpoints" / "model_state_dict.pt"),
                )
            ]
            match = next(
                (
                    (candidate_root, candidate_file)
                    for candidate_root, candidate_file in candidates
                    if candidate_file.is_file()
                ),
                None,
            )
            if match is None:
                expected = ", ".join(
                    str(path.relative_to(snapshot))
                    for _, path in candidates
                )
                raise FileNotFoundError(
                    "G0Plus checkpoint is not staged: expected "
                    f"one of [{expected}] under {snapshot}"
                )
            root, checkpoint = match
        if checkpoint.suffix != ".pt" or not checkpoint.is_file():
            raise FileNotFoundError(f"G0Plus requires a staged .pt checkpoint, got {checkpoint}")
        self._checkpoint_root = root.resolve()
        self._checkpoint_file = checkpoint.resolve()
        return self._checkpoint_root, self._checkpoint_file

    def _resolve_tokenizer(self) -> Path:
        if self._tokenizer_root is not None:
            return self._tokenizer_root
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        direct = self._local_path(self.config.tokenizer_location)
        source = str(direct) if direct is not None else self.config.tokenizer_location
        root = materialize_hf_snapshot(
            source,
            revision=self.config.tokenizer_revision,
            cache_dir=self.config.cache_dir,
            required_files=("tokenizer_config.json",),
            local_files_only=self.config.local_files_only,
        )
        self._tokenizer_root = root
        return root

    @staticmethod
    def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
        if isinstance(payload, Mapping) and isinstance(payload.get("model_state_dict"), Mapping):
            payload = payload["model_state_dict"]
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("G0Plus checkpoint does not contain a non-empty state dict")
        state = dict(payload)
        for prefix in ("module.", "model."):
            if state and all(str(key).startswith(prefix) for key in state):
                state = {str(key)[len(prefix) :]: value for key, value in state.items()}
        return state

    def _model_config(self) -> Any:
        from .configuration import as_config

        architecture = json.loads(json.dumps(dict(self.config.architecture)))
        architecture["action_dim"] = int(self.variant["action_dim"])
        architecture["proprio_dim"] = int(self.variant["state_dim"])
        architecture["num_input_images"] = int(self.variant["num_input_images"])
        architecture["cond_steps"] = int(self.variant["observation_steps"])
        architecture["horizon_steps"] = int(self.variant["action_horizon"])
        architecture["max_image_text_tokens"] = (
            int(architecture["max_text_tokens"])
            + int(self.variant["num_input_images"]) * int(architecture["vision"]["num_image_tokens"])
        )
        return as_config(architecture)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import GalaxeaZero
        from .preprocessing import PaliGemmaTokenizer

        root, checkpoint_file = self._resolve_checkpoint()
        tokenizer_root = self._resolve_tokenizer()
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        state = self._extract_state_dict(payload)
        if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
            raise ValueError("G0Plus checkpoint state dict must contain tensor values only")
        model_config = self._model_config()
        with torch.device("meta"):
            model = GalaxeaZero(model_config)
        try:
            model.load_state_dict(state, strict=True, assign=True)
        except TypeError as exc:
            raise RuntimeError("G0Plus meta loading requires PyTorch with load_state_dict(assign=True)") from exc
        model.tie_action_proprio_weights()
        del state, payload
        gc.collect()
        model = model.to(device=device, dtype=dtype).eval()
        if self.config.compile_vision:
            if not hasattr(torch, "compile"):
                raise RuntimeError("G0Plus compile_vision requires torch.compile")
            model.vision_tower = torch.compile(
                model.vision_tower,
                mode=self.config.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        tokenizer = PaliGemmaTokenizer(
            tokenizer_root,
            pad_token_id=int(model_config.pad_token_id),
            image_token_index=int(model_config.image_token_index),
            max_text_tokens=int(model_config.max_text_tokens),
            num_tokens_per_image=int(model_config.vision.num_image_tokens),
            num_input_images=int(model_config.num_input_images),
            prompt_template=self.config.prompt_template,
        )
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._dtype = dtype
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
                "G0Plus requires the exact dataset_stats.json paired with its staged checkpoint: "
                f"{path}"
            )
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if self.config.statistics_key:
            for key in self.config.statistics_key.split("."):
                payload = payload[key]
        payload = _required_mapping(payload, "statistics")
        if "state" not in payload or "action" not in payload:
            raise ValueError("G0Plus statistics must contain 'state' and 'action' mappings")
        self._statistics = payload
        return payload

    @staticmethod
    def _parts(config: Mapping[str, Any], label: str) -> list[tuple[str, int]]:
        values = config.get(f"{label}_parts")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"G0Plus variant has no {label}_parts sequence")
        return [(str(item["key"]), int(item["dim"])) for item in values]

    def _state_fields(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        value = first_present(observation, "state", "proprio", "agent_pos", "joint_state", "robot_state")
        nested = observation.get("observation")
        if value is None and isinstance(nested, Mapping):
            value = first_present(nested, "state", "proprio", "agent_pos", "joint_state", "robot_state")
        if value is None:
            raise ValueError("G0Plus requires a state/proprio observation")
        parts = self._parts(self.variant, "state")
        fields: dict[str, Any] = {}
        if isinstance(value, Mapping):
            for key, dim in parts:
                if key not in value:
                    raise ValueError(f"G0Plus state mapping is missing field {key!r}")
                array = np.asarray(value[key], dtype=np.float32)
                if array.shape[-1] != dim:
                    raise ValueError(f"G0Plus state field {key!r} expects {dim} values, got {array.shape}")
                fields[key] = array.reshape(-1, dim)
        else:
            array = np.asarray(value, dtype=np.float32)
            width = sum(dim for _, dim in parts)
            if array.shape[-1] != width:
                raise ValueError(f"G0Plus variant {self.config.variant!r} expects {width} state values, got {array.shape}")
            array = array.reshape(-1, width)
            offset = 0
            for key, dim in parts:
                fields[key] = array[:, offset : offset + dim]
                offset += dim
        steps = int(self.variant["observation_steps"])
        for key in fields:
            if fields[key].shape[0] < steps:
                raise ValueError(f"G0Plus state field {key!r} has fewer than {steps} observation steps")
            fields[key] = fields[key][-steps:]
        for key in self.variant.get("wrap_state_angle_keys", ()):
            fields[str(key)] = np.arctan2(np.sin(fields[str(key)]), np.cos(fields[str(key)]))
        return fields

    @staticmethod
    def _stats_values(stats: Mapping[str, Any], prefix: str, name: str) -> Any:
        key = f"{prefix}_{name}"
        if key not in stats:
            raise ValueError(f"G0Plus statistics field is missing {key!r}")
        return stats[key]

    def _linear_params(
        self,
        stats: Mapping[str, Any],
        *,
        prefix: str,
        mode: str,
    ) -> tuple[Any, Any]:
        import numpy as np

        epsilon = float(self.config.normalization_epsilon)
        if mode == "z-score":
            mean = np.asarray(self._stats_values(stats, prefix, "mean"), dtype=np.float32)
            std = np.asarray(self._stats_values(stats, prefix, "std"), dtype=np.float32)
            return 1.0 / (std + epsilon), -mean / (std + epsilon)
        if mode in {"min/max", "q01/q99"}:
            low_name, high_name = ("min", "max") if mode == "min/max" else ("q01", "q99")
            low = np.asarray(self._stats_values(stats, prefix, low_name), dtype=np.float32)
            high = np.asarray(self._stats_values(stats, prefix, high_name), dtype=np.float32)
        else:
            try:
                low_value, high_value = (float(value) for value in mode.split("/", 1))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unsupported G0Plus normalization mode {mode!r}") from exc
            reference = np.asarray(self._stats_values(stats, prefix, "min"), dtype=np.float32)
            low = np.full_like(reference, low_value)
            high = np.full_like(reference, high_value)
        input_range = high - low
        ignored = input_range < 1.0e-4
        safe_range = input_range.copy()
        safe_range[ignored] = 2.0
        scale = 2.0 / safe_range
        offset = -1.0 - scale * low
        offset[ignored] = -low[ignored]
        return scale, offset

    def _normalize_state(self, fields: Mapping[str, Any]) -> Any:
        import numpy as np

        state_stats = _required_mapping(self._statistics_payload()["state"], "state statistics")
        exceptions = _required_mapping(self.variant.get("normalization_exceptions", {}), "exceptions")
        state_exceptions = _required_mapping(exceptions.get("state", {}), "state exceptions")
        output = []
        for key, _ in self._parts(self.variant, "state"):
            stats = _required_mapping(state_stats.get(key), f"state statistics for {key}")
            mode = str(state_exceptions.get(key, self.variant["normalization_mode"]))
            scale, offset = self._linear_params(stats, prefix="global", mode=mode)
            output.append(np.asarray(fields[key], dtype=np.float32) * scale + offset)
        return np.concatenate(output, axis=-1).astype(np.float32)

    def _denormalize_actions(self, values: Any, raw_state: Mapping[str, Any]) -> Any:
        import numpy as np

        action_stats = _required_mapping(self._statistics_payload()["action"], "action statistics")
        exceptions = _required_mapping(self.variant.get("normalization_exceptions", {}), "exceptions")
        action_exceptions = _required_mapping(exceptions.get("action", {}), "action exceptions")
        prefix = "stepwise" if bool(self.variant["use_stepwise_action_norm"]) else "global"
        parts = self._parts(self.variant, "action")
        values = np.asarray(values, dtype=np.float32)
        fields: dict[str, Any] = {}
        offset_index = 0
        for key, dim in parts:
            normalized = values[:, offset_index : offset_index + dim]
            stats = _required_mapping(action_stats.get(key), f"action statistics for {key}")
            mode = str(action_exceptions.get(key, self.variant["normalization_mode"]))
            scale, offset = self._linear_params(stats, prefix=prefix, mode=mode)
            if scale.ndim == 2:
                scale = scale[: normalized.shape[0]]
                offset = offset[: normalized.shape[0]]
            fields[key] = (normalized - offset) / scale
            offset_index += dim
        for key in self.variant.get("relative_action_keys", ()):
            name = str(key)
            fields[name] = fields[name] + np.asarray(raw_state[name], dtype=np.float32)[-1:]
        return np.concatenate([fields[key] for key, _ in parts], axis=-1).astype(np.float32)

    def _camera_values(self, observation: Mapping[str, Any], image: Any) -> list[Any]:
        aliases = _required_mapping(self.variant["camera_aliases"], "camera aliases")
        containers = []
        for key in ("images", "vision"):
            nested = observation.get(key)
            if isinstance(nested, Mapping):
                containers.append(nested)
        containers.append(observation)
        if isinstance(image, Mapping):
            containers.append(image)
        output = []
        for index, key in enumerate(self.variant["camera_keys"]):
            names = (str(key), *tuple(str(value) for value in aliases.get(key, ())))
            selected = None
            for container in containers:
                selected = first_present(container, *names)
                if isinstance(selected, Mapping):
                    selected = first_present(selected, "color", "rgb", "image")
                if selected is not None:
                    break
            if selected is None and index == 0 and image is not None and not isinstance(image, Mapping):
                selected = image
            if selected is None:
                raise ValueError(f"G0Plus requires camera {key!r}; accepted aliases are {names}")
            output.append(selected)
        return output

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch

        from .preprocessing import prepare_images

        model = self._load_model()
        raw_state = self._state_fields(observation)
        state = self._normalize_state(raw_state)
        prompt = self.config.instruction_template.format(instruction=instruction)
        tokens = self._tokenizer.tokenize(prompt)
        pixels = prepare_images(
            self._camera_values(observation, image),
            image_size=int(self.config.architecture["vision"]["image_size"]),
            expected_count=int(self.variant["num_input_images"]),
            mean=self.config.image_mean,
            std=self.config.image_std,
        )
        devices = []
        if torch.device(self._device).type == "cuda":
            devices = [torch.device(self._device).index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config.seed))
            with torch.inference_mode():
                actions = model.infer_action(
                    input_ids=tokens["input_ids"].to(self._device),
                    attention_mask=tokens["attention_mask"].to(self._device),
                    pixel_values=pixels.unsqueeze(0).to(device=self._device, dtype=self._dtype),
                    proprios=torch.as_tensor(state, device=self._device, dtype=self._dtype).unsqueeze(0),
                )
        normalized = actions[0].float().cpu().numpy()
        output = self._denormalize_actions(normalized, raw_state)
        horizon = min(int(self.config.action_horizon or output.shape[0]), output.shape[0])
        output = np.asarray(output[:horizon], dtype=np.float32)
        return completed_action_result(
            model_id="galaxea-vla",
            instruction=instruction,
            actions=output.tolist(),
            checkpoint_path=str(self._checkpoint_file),
            device=str(self._device),
            runtime="worldfoundry.galaxea_vla.in_tree_runtime",
            metadata={
                "variant": self.config.variant,
                "action_shape": list(output.shape),
                "camera_keys": list(self.variant["camera_keys"]),
                "dtype": str(self._dtype),
                "flow_steps": int(self.config.architecture["num_inference_steps"]),
                "compile_vision": bool(self.config.compile_vision),
                "seed": int(self.config.seed),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], GalaxeaVLARuntime] = {}


def _required(options: Mapping[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, ""):
        raise ValueError(f"G0Plus runtime option {key!r} is required; load its data config")
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
        raise ValueError("G0Plus checkpoint_path or checkpoint_ref is required")
    config = GalaxeaVLARuntimeConfig(
        checkpoint_location=checkpoint,
        tokenizer_location=str(_required(options, "tokenizer_checkpoint_path")),
        checkpoint_subdir=str(options.get("checkpoint_subdir") or ""),
        variant=str(_required(options, "variant")),
        architecture=dict(_required(options, "architecture")),
        variants=dict(_required(options, "variants")),
        prompt_template=str(_required(options, "prompt_template")),
        instruction_template=str(_required(options, "instruction_template")),
        normalization_epsilon=float(_required(options, "normalization_epsilon")),
        image_mean=tuple(_required(options, "image_mean")),
        image_std=tuple(_required(options, "image_std")),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        local_files_only=option_bool(options.get("local_files_only"), True),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
        revision=str(options["revision"]) if options.get("revision") else None,
        tokenizer_revision=(
            str(options["tokenizer_revision"]) if options.get("tokenizer_revision") else None
        ),
        statistics_path=str(options["statistics_path"]) if options.get("statistics_path") else None,
        statistics_file=str(options.get("statistics_file") or "dataset_stats.json"),
        statistics_key=str(options["statistics_key"]) if options.get("statistics_key") else None,
        action_horizon=(
            option_int(options.get("action_horizon"), 0)
            if options.get("action_horizon") is not None
            else None
        ),
        seed=option_int(options.get("seed"), 0),
        compile_vision=option_bool(options.get("compile_vision"), False),
        compile_mode=str(options.get("compile_mode") or "reduce-overhead"),
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = GalaxeaVLARuntime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["GalaxeaVLARuntime", "GalaxeaVLARuntimeConfig", "predict_action"]

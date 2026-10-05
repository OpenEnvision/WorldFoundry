"""Local-only checkpoint runtime for Xiaomi-Robotics-1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import (
    resolve_data_path,
    resolve_local_hf_model_path,
    resolve_worldfoundry_path,
)
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)

MODEL_ID = "xiaomi-robotics-1"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected a YAML mapping in {path}")
    return dict(payload)


def _runtime_defaults() -> dict[str, Any]:
    return _load_yaml(resolve_data_path("models", "runtime", "configs", "vla_va_wam", f"{MODEL_ID}.yaml"))


def _resolve_local(value: str, *, required_files: Sequence[str] = ()) -> Path:
    # Keep direct checkpoint paths on the same completeness path as hfd-style
    # model IDs.  In particular, a partially downloaded aria2 target already
    # has its final filename, so checking only ``is_file()`` can expose a
    # truncated checkpoint to ``torch.load`` while the transfer is active.
    return resolve_local_hf_model_path(value, required_files=required_files)


@dataclass(frozen=True)
class XiaomiRobotics1RuntimeConfig:
    checkpoint_path: str
    model_state_filename: str
    base_vlm_path: str
    processor_path: str
    architecture_path: str
    normalization_path: str | None
    checkpoint_metadata_filename: str
    task_id: str
    action_type: str
    device: str
    torch_dtype: str
    attn_implementation: str
    seed: int
    compile_model: bool
    low_cpu_mem_usage: bool

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        *,
        checkpoint_path: str,
        device: str,
    ) -> "XiaomiRobotics1RuntimeConfig":
        action_type = str(options.get("action_type") or "joint").lower()
        if action_type not in {"joint", "ee"}:
            raise ValueError("action_type must be 'joint' or 'ee'")
        if not option_bool(options.get("local_files_only"), True):
            raise ValueError("Xiaomi-Robotics-1 runtime is local-only")
        checkpoint = str(
            checkpoint_path or first_present(options, "checkpoint_path", "checkpoint_dir", "model_path") or ""
        )
        if not checkpoint:
            raise ValueError("Xiaomi-Robotics-1 requires a local checkpoint_path")
        normalization_value = options.get("normalization_path")
        return cls(
            checkpoint_path=checkpoint,
            model_state_filename=str(options["model_state_filename"]),
            base_vlm_path=str(options["base_vlm_path"]),
            processor_path=str(options.get("processor_path") or options["base_vlm_path"]),
            architecture_path=str(options["architecture_path"]),
            normalization_path=str(normalization_value) if normalization_value else None,
            checkpoint_metadata_filename=str(options.get("checkpoint_metadata_filename") or "config.py"),
            task_id=str(options.get("task_id") or "default"),
            action_type=action_type,
            device=str(device or options.get("device") or "cuda"),
            torch_dtype=str(options.get("torch_dtype") or options.get("dtype") or "auto"),
            attn_implementation=str(options.get("attn_implementation") or "auto"),
            seed=option_int(options.get("seed"), 42),
            compile_model=option_bool(options.get("compile_model"), False),
            low_cpu_mem_usage=option_bool(options.get("low_cpu_mem_usage"), True),
        )


class XiaomiRobotics1Runtime:
    """Persistent in-process policy runtime."""

    def __init__(self, config: XiaomiRobotics1RuntimeConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._torch: Any = None
        self._device = ""
        self._dtype: Any = None
        self._checkpoint = ""
        self._architecture: dict[str, Any] = {}
        self._statistics: dict[str, Any] = {}
        self._action_mask: Any = None

    @staticmethod
    def _checkpoint_state_dict(payload: Any) -> dict[str, Any]:
        import torch

        if not isinstance(payload, Mapping):
            raise TypeError("XR1 model_states.pt must contain a state-dict mapping")
        values: Mapping[str, Any] = payload
        # Released evaluation weights use a DeepSpeed checkpoint envelope;
        # its tensor-only policy weights live under ``module``.  Keep the
        # narrower conventional wrappers for converted/exported checkpoints.
        for key in ("state_dict", "model_state_dict", "module"):
            nested = values.get(key)
            if isinstance(nested, Mapping):
                values = nested
                break
        if not values or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in values.items()
        ):
            raise TypeError(
                "XR1 model_states.pt must contain a non-empty string-to-tensor state dict"
            )
        prefixes = ("module.model.", "model.")
        for prefix in prefixes:
            selected = {key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)}
            if selected:
                return selected
        return dict(values)

    @staticmethod
    def _configure_attention(config: Any, implementation: str) -> None:
        for target in (config, config.text_config, config.vision_config):
            target._attn_implementation = implementation
            target._attn_implementation_internal = implementation

    @staticmethod
    def _load_normalization(path: Path, task_id: str) -> dict[str, Any]:
        if path.suffix.lower() == ".json":
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = _load_yaml(path)
        if not isinstance(payload, Mapping):
            raise TypeError(f"normalization data must be a mapping: {path}")
        tasks = payload.get("tasks", payload)
        if not isinstance(tasks, Mapping) or not tasks:
            raise ValueError(
                f"normalization file {path} has no task statistics; "
                "export checkpoint statistics into worldfoundry/data"
            )
        selected = tasks.get(task_id)
        if selected is None and len(tasks) == 1:
            selected = next(iter(tasks.values()))
        if not isinstance(selected, Mapping):
            raise KeyError(f"normalization task {task_id!r} is unavailable; choices={tuple(tasks)}")
        if not isinstance(selected.get("state"), Mapping) or not isinstance(selected.get("action"), Mapping):
            raise ValueError("normalization task must define state and action mappings")
        return dict(selected)

    @staticmethod
    def _validate_special_tokens(processor: Any, architecture: Mapping[str, Any]) -> None:
        tokenizer = processor.tokenizer
        tokens = architecture["special_tokens"]
        expected = {
            str(tokens["score"]): int(tokens["score_token_id"]),
            str(tokens["state"]): int(tokens["state_token_id"]),
        }
        action_count = int(tokens["action_token_count"])
        state_id = int(tokens["state_token_id"])
        expected.update({f"<a_{index}>": state_id + index + 1 for index in range(action_count)})
        mismatches = {
            token: (expected_id, int(tokenizer.convert_tokens_to_ids(token)))
            for token, expected_id in expected.items()
            if int(tokenizer.convert_tokens_to_ids(token)) != expected_id
        }
        if mismatches:
            raise ValueError(f"processor special-token IDs do not match the XR1 checkpoint: {mismatches}")

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.synthesis.action_generation.xiaomi_robotics_0.configuration_mibot import (
            Qwen3VLConfig,
        )

        from .modeling import XR1

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        attention = resolve_transformers_attention_implementation(
            self.config.attn_implementation,
            device,
        )
        checkpoint = _resolve_local(
            self.config.checkpoint_path,
            required_files=(
                self.config.model_state_filename,
                self.config.checkpoint_metadata_filename,
            ),
        )
        base_vlm = _resolve_local(self.config.base_vlm_path, required_files=("config.json",))
        processor_path = _resolve_local(
            self.config.processor_path,
            required_files=("preprocessor_config.json", "tokenizer_config.json"),
        )
        architecture_path = resolve_worldfoundry_path(self.config.architecture_path)
        if not architecture_path.is_file():
            raise FileNotFoundError(f"XR1 central architecture config is missing: {architecture_path}")
        architecture = _load_yaml(architecture_path)
        if self.config.normalization_path:
            normalization_path = resolve_worldfoundry_path(self.config.normalization_path)
            if not normalization_path.is_file():
                raise FileNotFoundError(f"XR1 central normalization export is missing: {normalization_path}")
            statistics = self._load_normalization(normalization_path, self.config.task_id)
        else:
            from .checkpoint_metadata import extract_normalization

            payload = extract_normalization(
                checkpoint / self.config.checkpoint_metadata_filename,
                expected_state_shape=tuple(int(value) for value in architecture["state_shape"]),
                expected_action_shape=tuple(int(value) for value in architecture["action_shape"]),
            )
            tasks = payload["tasks"]
            selected = tasks.get(self.config.task_id)
            if selected is None and len(tasks) == 1:
                selected = next(iter(tasks.values()))
            if not isinstance(selected, Mapping):
                raise KeyError(
                    f"checkpoint normalization task {self.config.task_id!r} is unavailable; choices={tuple(tasks)}"
                )
            statistics = dict(selected)
        vlm_config = Qwen3VLConfig.from_pretrained(
            str(base_vlm),
            local_files_only=True,
            trust_remote_code=False,
        )
        self._configure_attention(vlm_config, attention)
        tokens = architecture["special_tokens"]
        model_kwargs = {
            "vlm_config": vlm_config,
            "state_shape": architecture["state_shape"],
            "action_shape": architecture["action_shape"],
            "n_choices": architecture["n_choices"],
            "dit_config": architecture["dit"],
            "num_steps": architecture["num_steps"],
            "knowledge_insulation": architecture["knowledge_insulation"],
            "score_token_id": tokens["score_token_id"],
            "state_token_id": tokens["state_token_id"],
            "action_token_count": tokens["action_token_count"],
            "timestep_frequency_size": architecture["timestep_frequency_size"],
        }
        if self.config.low_cpu_mem_usage:
            with torch.device("meta"):
                model = XR1(**model_kwargs)
        else:
            model = XR1(**model_kwargs)
            model = model.to(dtype=dtype)

        checkpoint_payload = torch.load(
            checkpoint / self.config.model_state_filename,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        state_dict = self._checkpoint_state_dict(checkpoint_payload)
        model.load_state_dict(
            state_dict,
            strict=True,
            assign=self.config.low_cpu_mem_usage,
        )
        # ``assign=True`` replaces the two tied Qwen parameters with distinct
        # Parameter objects.  Although the released checkpoint tensors share
        # CPU storage, an ensuing ``to(device)`` would otherwise materialize
        # the 151936 x 2560 embedding twice on the accelerator.
        model.vlm.tie_weights()
        del checkpoint_payload, state_dict
        if self.config.low_cpu_mem_usage:
            model.refresh_nonpersistent_rotary_buffers(device="cpu")
            meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
            meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
            if meta_parameters or meta_buffers:
                raise RuntimeError(
                    "checkpoint did not materialize the full model: "
                    f"parameters={meta_parameters}, buffers={meta_buffers}"
                )
        model = model.to(device=device, dtype=dtype).eval()
        model.requires_grad_(False)

        special_tokens = {
            "score": str(tokens["score"]),
            "state": str(tokens["state"]),
        }
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(int(tokens["action_token_count"]))})
        processor = AutoProcessor.from_pretrained(
            str(processor_path),
            use_fast=True,
            extra_special_tokens=special_tokens,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._validate_special_tokens(processor, architecture)
        action_dim = int(architecture["action_shape"][-1])
        action_mask = torch.zeros(action_dim, dtype=dtype, device=device)
        for start, end in architecture["active_action_ranges"]:
            action_mask[int(start) : int(end)] = 1

        if self.config.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model, dynamic=False)
        self.model = model
        self.processor = processor
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._checkpoint = str(checkpoint)
        self._architecture = architecture
        self._statistics = statistics
        self._action_mask = action_mask

    @staticmethod
    def _instruction(observation: Mapping[str, Any], fallback: str) -> str:
        value = first_present(observation, "instruction", "instructions", "prompt", "task")
        if isinstance(value, Sequence) and not isinstance(value, str):
            value = value[0] if value else None
        text = str(value or fallback).strip()
        if not text:
            raise ValueError("Xiaomi-Robotics-1 requires a non-empty instruction")
        return text.rstrip(".") + "."

    def _prepare_batch(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from .processing import (
            add_end_effector_state,
            center_crop_image,
            normalize_tensor,
            pack_robot_state,
            resolve_camera_images,
        )

        architecture = self._architecture
        camera_aliases = tuple(tuple(group) for group in architecture["camera_aliases"])
        images = resolve_camera_images(observation, image, camera_aliases)
        images = [
            center_crop_image(
                value,
                crop_ratio=float(architecture["crop_ratio"]),
                output_size=architecture["image_size"],
            )
            for value in images
        ]
        state_dim = int(architecture["state_shape"][-1])
        state, current_state = pack_robot_state(observation, state_dim=state_dim)
        if self.config.action_type == "ee":
            add_end_effector_state(current_state)
        labels = tuple(str(value) for value in architecture["camera_labels"])
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "The following observations are captured from multiple views.\n",
            }
        ]
        for label, value in zip(labels, images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": f"# {label}\n"},
                    {"type": "image", "image": value},
                    {"type": "text", "text": "\n"},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": f"Generate robot actions for the task:\n{instruction} /no_cot",
            }
        )
        base_messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        base = self.processor.apply_chat_template(
            base_messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
        )
        condition_length = int(base["input_ids"].shape[1])
        tokens = architecture["special_tokens"]
        state_tokens = str(tokens["state"]) * int(architecture["state_shape"][0])
        action_tokens = "".join(f"<a_{index}>" for index in range(int(architecture["action_shape"][0])))
        messages = base_messages + [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Robot state: {state_tokens}"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"{action_tokens}{tokens['score']}"}],
            },
        ]
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )
        state_tensor = self._torch.as_tensor(
            state,
            device=self._device,
            dtype=self._dtype,
        ).view(1, *architecture["state_shape"])
        batch["state"] = normalize_tensor(state_tensor, self._statistics["state"])
        batch["action_vlm_condition_segments"] = self._torch.tensor(
            [[0, condition_length]],
            dtype=self._torch.long,
        )
        action_shape = tuple(int(value) for value in architecture["action_shape"])
        batch["action"] = self._torch.zeros(
            (1, *action_shape),
            dtype=self._dtype,
            device=self._device,
        )
        batch["action_mask"] = self._action_mask.view(1, 1, -1)
        batch = {
            key: value.to(self._device) if isinstance(value, self._torch.Tensor) else value
            for key, value in batch.items()
        }
        return batch, current_state

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        from .processing import decode_action_chunk, denormalize_tensor

        self._load()
        instruction = self._instruction(observation, instruction)
        batch, current_state = self._prepare_batch(
            instruction=instruction,
            image=image,
            observation=observation,
        )
        actions = self.model.generate(batch, seed=self.config.seed)
        actions = denormalize_tensor(actions, self._statistics["action"])
        if not bool(self._torch.isfinite(actions).all()):
            raise FloatingPointError("Xiaomi-Robotics-1 produced non-finite actions")
        raw_actions = actions[0].float().cpu().numpy()
        decoded = decode_action_chunk(
            raw_actions,
            current_state=current_state,
            action_type=self.config.action_type,
        )
        return completed_action_result(
            model_id=MODEL_ID,
            instruction=instruction,
            actions=decoded,
            raw_output=raw_actions.tolist(),
            checkpoint_path=self._checkpoint,
            device=self._device,
            runtime="worldfoundry.xiaomi_robotics_1.in_tree_runtime",
            metadata={
                "action_shape": list(raw_actions.shape),
                "action_type": self.config.action_type,
                "dtype": str(self._dtype),
                "attention": self.config.attn_implementation,
                "camera_views": len(self._architecture["camera_labels"]),
                "task_id": self.config.task_id,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], XiaomiRobotics1Runtime] = {}


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
    config = XiaomiRobotics1RuntimeConfig.from_options(
        options,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    key = (config.checkpoint_path, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = XiaomiRobotics1Runtime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["XiaomiRobotics1Runtime", "XiaomiRobotics1RuntimeConfig", "predict_action"]

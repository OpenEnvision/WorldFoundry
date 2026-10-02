"""In-tree inference runtime for Xiaomi-Robotics-0.

The official GitHub deployment imports model code dynamically from each
Hugging Face checkpoint.  This module keeps that code in-tree and loads the
local classes explicitly, so inference never imports an external checkout or
the Hugging Face remote-code cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.core.io.formats.serialization import write_json
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    load_image,
    option_bool,
    option_int,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

MODEL_ID = "xiaomi-robotics-0"
_MODEL_CONFIG = load_vla_va_wam_runtime_config(MODEL_ID)
UPSTREAM_GITHUB_REVISION = str(_MODEL_CONFIG["upstream_github_revision"])
ACTION_HORIZONS_BY_ROBOT_TYPE: dict[str, int] = {
    str(name): int(value)
    for name, value in dict(_MODEL_CONFIG["action_horizons_by_robot_type"]).items()
}
_PRETRAIN_ROBOT_TYPES = tuple(str(value) for value in _MODEL_CONFIG["pretrain_robot_types"])
_REQUIRED_CHECKPOINT_FILES = tuple(
    str(value) for value in _MODEL_CONFIG["required_checkpoint_files"]
)


@dataclass(frozen=True)
class CheckpointSpec:
    """Pinned official checkpoint and its demonstrated inference contract."""

    variant: str
    repo_id: str
    revision: str
    robot_type: str | None
    camera_keys: tuple[str, ...]
    view_labels: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    image_size: tuple[int, int] | None = None
    center_crop_ratio: float | None = None


OFFICIAL_CHECKPOINTS: dict[str, CheckpointSpec] = {
    str(variant): CheckpointSpec(
        variant=str(variant),
        repo_id=str(values["repo_id"]),
        revision=str(values["revision"]),
        robot_type=str(values["robot_type"]) if values.get("robot_type") else None,
        camera_keys=tuple(str(value) for value in values.get("camera_keys", ())),
        view_labels=tuple(str(value) for value in values.get("view_labels", ())),
        aliases=tuple(str(value) for value in values.get("aliases", ())),
        image_size=(
            tuple(int(value) for value in values["image_size"])
            if values.get("image_size") is not None
            else None
        ),
        center_crop_ratio=(
            float(values["center_crop_ratio"])
            if values.get("center_crop_ratio") is not None
            else None
        ),
    )
    for variant, values in dict(_MODEL_CONFIG["variants"]).items()
}


_VARIANT_ALIASES = {
    alias.lower().replace("_", "-"): name
    for name, spec in OFFICIAL_CHECKPOINTS.items()
    for alias in (name, f"{MODEL_ID}-{name}", spec.repo_id, spec.repo_id.rsplit("/", 1)[-1], *spec.aliases)
}

_CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    str(name): tuple(str(value) for value in aliases)
    for name, aliases in dict(_MODEL_CONFIG["camera_aliases"]).items()
}


def checkpoint_spec(variant: Any = None) -> CheckpointSpec:
    """Resolve a checkpoint variant or official repository ID."""

    key = str(variant or _MODEL_CONFIG["default_variant"]).strip().lower().replace("_", "-")
    try:
        return OFFICIAL_CHECKPOINTS[_VARIANT_ALIASES[key]]
    except KeyError as exc:
        available = ", ".join(sorted(OFFICIAL_CHECKPOINTS))
        raise ValueError(f"unknown Xiaomi-Robotics-0 variant {variant!r}; choose one of: {available}") from exc


def build_instruction_prompt(instruction: str, view_labels: Sequence[str]) -> str:
    """Build the exact no-CoT prompt used by the official deployment."""

    view_block = "".join(
        f"# {label}\n<|vision_start|><|image_pad|><|vision_end|>\n" for label in view_labels
    )
    return (
        "<|im_start|>user\n"
        "The following observations are captured from multiple views.\n"
        f"{view_block}"
        "Generate robot actions for the task:\n"
        f"{instruction} /no_cot<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<cot></cot><|im_end|>\n"
    )


def _pil_rgb_image(value: Any) -> Any:
    """Convert one array/tensor/PIL observation to a CPU RGB PIL image."""

    import numpy as np
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Xiaomi-Robotics-0 camera must be unbatched, got {array.shape}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Xiaomi-Robotics-0 camera must be HWC or CHW, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Xiaomi-Robotics-0 camera contains non-finite values")
        low = float(array.min(initial=0.0))
        high = float(array.max(initial=0.0))
        if low < 0.0:
            if low < -1.0 or high > 1.0:
                raise ValueError(f"unsupported Xiaomi-Robotics-0 camera range [{low}, {high}]")
            array = (array + 1.0) * 127.5
        elif high <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array).convert("RGB")


@dataclass(frozen=True)
class XiaomiRobotics0RuntimeConfig:
    """Resolved lazy-load settings for one checkpoint-backed runtime."""

    variant: str = str(_MODEL_CONFIG["default_variant"])
    checkpoint: str = OFFICIAL_CHECKPOINTS["libero"].repo_id
    revision: str | None = OFFICIAL_CHECKPOINTS["libero"].revision
    robot_type: str = "libero_all"
    device: str = str(_MODEL_CONFIG["device"])
    torch_dtype: str = str(_MODEL_CONFIG["torch_dtype"])
    attn_implementation: str = str(_MODEL_CONFIG["attn_implementation"])
    num_steps: int = int(_MODEL_CONFIG["num_steps"])
    seed: int = int(_MODEL_CONFIG["seed"])
    expected_action_horizon: int | None = ACTION_HORIZONS_BY_ROBOT_TYPE["libero_all"]
    view_labels: tuple[str, ...] = OFFICIAL_CHECKPOINTS["libero"].view_labels
    camera_keys: tuple[str, ...] = OFFICIAL_CHECKPOINTS["libero"].camera_keys
    image_size: tuple[int, int] | None = OFFICIAL_CHECKPOINTS["libero"].image_size
    center_crop_ratio: float | None = OFFICIAL_CHECKPOINTS["libero"].center_crop_ratio

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any] | None = None,
        *,
        device: str | None = None,
    ) -> "XiaomiRobotics0RuntimeConfig":
        values = dict(options or {})
        variant = first_present(
            values,
            "variant",
            "variant_id",
            "model_variant",
            "checkpoint_variant",
        )
        checkpoint_value = first_present(
            values,
            "checkpoint",
            "checkpoint_path",
            "checkpoint_dir",
            "model_path",
            "pretrained_model_path",
            "ckpt_path",
            "repo_id",
        )
        if variant is None and checkpoint_value is not None:
            checkpoint_key = str(checkpoint_value).strip().lower().replace("_", "-")
            if checkpoint_key in _VARIANT_ALIASES:
                variant = checkpoint_value
        spec = checkpoint_spec(variant)
        checkpoint = str(
            checkpoint_value or spec.repo_id
        )
        revision_value = first_present(values, "revision", "checkpoint_revision")
        if Path(checkpoint).expanduser().exists():
            revision = None
        elif revision_value is not None:
            revision = str(revision_value)
        elif checkpoint == spec.repo_id:
            revision = spec.revision
        else:
            revision = None
        robot_type = str(values.get("robot_type") or spec.robot_type or "").strip()
        if not robot_type:
            raise ValueError(
                "the pretrain checkpoint contains multiple embodiment statistics; pass robot_type explicitly "
                "(one of agibot_pt_v3, droid_pt, midata_mix_ann_pt_v1, molmoact_pt)"
            )
        if (
            spec.variant == "pretrain"
            and checkpoint == spec.repo_id
            and robot_type not in _PRETRAIN_ROBOT_TYPES
        ):
            available = ", ".join(sorted(_PRETRAIN_ROBOT_TYPES))
            raise ValueError(
                f"unknown Xiaomi-Robotics-0 pretrain robot_type {robot_type!r}; choose one of: {available}"
            )
        view_labels_value = first_present(values, "view_labels")
        camera_keys_value = first_present(values, "camera_keys")
        if spec.variant == "pretrain" and (view_labels_value is None or camera_keys_value is None):
            raise ValueError(
                "the pretrain checkpoint has no universal view contract; pass matching view_labels and camera_keys "
                "explicitly for the selected robot_type"
            )
        view_labels_value = view_labels_value or spec.view_labels
        camera_keys_value = camera_keys_value or spec.camera_keys
        if isinstance(view_labels_value, str):
            view_labels_value = tuple(part.strip() for part in view_labels_value.split(",") if part.strip())
        if isinstance(camera_keys_value, str):
            camera_keys_value = tuple(part.strip() for part in camera_keys_value.split(",") if part.strip())
        view_labels = tuple(str(item).strip() for item in view_labels_value)
        camera_keys = tuple(str(item).strip() for item in camera_keys_value)
        if not view_labels or not camera_keys or not all(view_labels) or not all(camera_keys):
            raise ValueError("view_labels and camera_keys must each contain at least one entry")
        if len(view_labels) != len(camera_keys):
            raise ValueError("view_labels and camera_keys must have the same length")
        image_size_value = values.get("image_size", spec.image_size)
        if image_size_value is None:
            image_size = None
        else:
            if isinstance(image_size_value, str):
                image_size_value = tuple(
                    part.strip() for part in image_size_value.split(",") if part.strip()
                )
            image_size = tuple(int(value) for value in image_size_value)
            if len(image_size) != 2 or min(image_size) <= 0:
                raise ValueError("image_size must contain positive width and height")
        center_crop_value = values.get("center_crop_ratio", spec.center_crop_ratio)
        center_crop_ratio = float(center_crop_value) if center_crop_value is not None else None
        if center_crop_ratio is not None and not 0.0 < center_crop_ratio <= 1.0:
            raise ValueError("center_crop_ratio must be in (0, 1]")
        num_steps = option_int(values.get("num_steps"), int(_MODEL_CONFIG["num_steps"]))
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        expected_action_horizon = None
        if checkpoint == spec.repo_id and revision == spec.revision:
            expected_action_horizon = ACTION_HORIZONS_BY_ROBOT_TYPE.get(robot_type)
        return cls(
            variant=spec.variant,
            checkpoint=checkpoint,
            revision=revision,
            robot_type=robot_type,
            device=str(device or values.get("device") or _MODEL_CONFIG["device"]),
            torch_dtype=str(values.get("torch_dtype") or values.get("dtype") or _MODEL_CONFIG["torch_dtype"]),
            attn_implementation=str(values.get("attn_implementation") or _MODEL_CONFIG["attn_implementation"]),
            num_steps=num_steps,
            seed=option_int(values.get("seed"), int(_MODEL_CONFIG["seed"])),
            expected_action_horizon=expected_action_horizon,
            view_labels=view_labels,
            camera_keys=camera_keys,
            image_size=image_size,
            center_crop_ratio=center_crop_ratio,
        )


class XiaomiRobotics0Runtime:
    """Lazy, in-process Xiaomi-Robotics-0 policy runtime."""

    def __init__(self, config: XiaomiRobotics0RuntimeConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._attn_implementation: str | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            import transformers  # noqa: F401 - validates the required runtime dependency.
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Xiaomi-Robotics-0 inference requires torch and transformers==4.57.1; "
                "install the xiaomi-robotics-0 runtime environment"
            ) from exc

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling_mibot import MiBoTForActionGeneration
        from .processing_mibot import MiBotProcessor

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        attn_implementation = resolve_transformers_attention_implementation(
            self.config.attn_implementation,
            device,
        )
        checkpoint = str(
            resolve_local_hf_model_path(
                self.config.checkpoint,
                required_files=_REQUIRED_CHECKPOINT_FILES,
            )
        )
        load_kwargs: dict[str, Any] = {
            "attn_implementation": attn_implementation,
            "dtype": dtype,
            "trust_remote_code": False,
            "use_safetensors": True,
            "local_files_only": True,
        }
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "use_fast": False,
            "local_files_only": True,
        }

        # These are explicit in-tree classes; remote-code model dispatch is forbidden.
        model = MiBoTForActionGeneration.from_pretrained(checkpoint, **load_kwargs)
        processor = MiBotProcessor.from_pretrained(checkpoint, **processor_kwargs)
        model = model.to(device).eval()

        available_robot_types = tuple(processor.list_robot_types())
        if self.config.robot_type not in available_robot_types:
            raise ValueError(
                f"robot_type {self.config.robot_type!r} is not present in {self.config.checkpoint}; "
                f"available={available_robot_types}"
            )
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self.model = model
        self.processor = processor

    @staticmethod
    def _normalized_observation(observation: Mapping[str, Any], image: Any) -> dict[str, Any]:
        normalized = dict(image) if isinstance(image, Mapping) else {}
        normalized.update(observation)
        nested = normalized.get("images")
        if isinstance(nested, Mapping):
            nested = dict(nested)
            for canonical, aliases in _CAMERA_ALIASES.items():
                if canonical not in nested:
                    value = first_present(nested, *aliases)
                    if value is None:
                        value = first_present(normalized, canonical, *aliases)
                    if value is not None:
                        nested[canonical] = value
            normalized["images"] = nested
        for canonical, aliases in _CAMERA_ALIASES.items():
            if canonical not in normalized:
                value = first_present(normalized, *aliases)
                if value is not None:
                    normalized[canonical] = value
        return normalized

    def _images(self, image: Any, observation: Mapping[str, Any]) -> list[Any]:
        normalized = self._normalized_observation(observation, image)
        # Workspace carries the primary uploaded image through the pipeline's
        # ``images`` argument while named auxiliary views (for example
        # ``wrist_image``) remain in the observation.  ``collect_images``
        # intentionally prefers named observation values, so passing those
        # sources separately used to drop the primary/base view as soon as an
        # auxiliary view was present.  Merge them into the checkpoint's
        # canonical camera order before collection.
        if not isinstance(image, Mapping) and self.config.camera_keys:
            if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)):
                primary_values = [value for value in image if value is not None]
            else:
                primary_values = [] if image is None else [image]
            missing_keys = [key for key in self.config.camera_keys if normalized.get(key) is None]
            for key, value in zip(missing_keys, primary_values):
                normalized[key] = value
        values = collect_images(normalized, None, self.config.camera_keys)
        if len(values) != len(self.config.camera_keys):
            raise ValueError(
                f"variant {self.config.variant!r} requires {len(self.config.camera_keys)} image(s) in order "
                f"{self.config.camera_keys}; received {len(values)}"
            )
        images = [load_image(value) for value in values]
        if self.config.image_size is None:
            return images

        from PIL import Image

        width, height = self.config.image_size
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        processed = []
        for value in images:
            value = _pil_rgb_image(value).resize((width, height), resampling)
            if self.config.center_crop_ratio is not None:
                crop_width = max(1, int(width * self.config.center_crop_ratio))
                crop_height = max(1, int(height * self.config.center_crop_ratio))
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                value = value.crop(
                    (left, top, left + crop_width, top + crop_height)
                ).resize((width, height), resampling)
            processed.append(value)
        return processed

    def _state(self, observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, "state", "proprio_state", "proprio", "robot_state")
        if state is None:
            raise ValueError("Xiaomi-Robotics-0 requires state/proprio_state in the observation")
        if isinstance(state, Mapping):
            raise TypeError("robot_state mappings must be converted to the checkpoint's ordered state vector")
        tensor = self._torch.as_tensor(state)
        tensor = tensor.reshape(1, 1, -1)
        if self.config.robot_type == "bridge_delta" and tensor.shape[-1] == 7:
            tensor = self._torch.cat(
                (
                    tensor[..., :-1],
                    self._torch.zeros_like(tensor[..., -1:]),
                    tensor[..., -1:],
                ),
                dim=-1,
            )
        state_dim = int(self.model.config.state_dim)
        if tensor.shape[-1] > state_dim:
            raise ValueError(f"state dimension {tensor.shape[-1]} exceeds checkpoint state_dim={state_dim}")
        if tensor.shape[-1] < state_dim:
            tensor = self._torch.nn.functional.pad(tensor, (0, state_dim - tensor.shape[-1]))
        return tensor.to(device=self.model.device, dtype=self.model.dtype)

    def predict(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
        output_path: str | Path | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        prompt_is_formatted: bool = False,
    ) -> dict[str, Any]:
        """Generate one denormalized action chunk and optionally write its trace."""

        self._load()
        images = self._images(image, observation)
        prompt = instruction if prompt_is_formatted else build_instruction_prompt(instruction, self.config.view_labels)
        inputs = self.processor(
            text=[prompt],
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs["state"] = self._state(observation)
        inputs["action_mask"] = self.processor.get_action_mask(self.config.robot_type).to(
            self.model.device,
            self.model.dtype,
        )
        inputs["seed"] = int(self.config.seed if seed is None else seed)
        resolved_steps = int(self.config.num_steps if num_steps is None else num_steps)
        if resolved_steps <= 0:
            raise ValueError("num_steps must be positive")
        with self._torch.inference_mode():
            outputs = self.model(**inputs, num_steps=resolved_steps)
        actions = self.processor.decode_action(outputs.actions, robot_type=self.config.robot_type).detach().cpu()
        action_shape = tuple(int(dimension) for dimension in actions.shape)
        if len(action_shape) < 2:
            raise RuntimeError(f"checkpoint returned an invalid action shape: {action_shape}")
        action_horizon = action_shape[-2]
        if self.config.expected_action_horizon is not None and action_horizon != self.config.expected_action_horizon:
            raise RuntimeError(
                f"checkpoint returned action horizon {action_horizon}, expected {self.config.expected_action_horizon} "
                f"for robot_type={self.config.robot_type!r}"
            )
        result = completed_action_result(
            model_id=MODEL_ID,
            instruction=instruction,
            actions=actions,
            raw_output=outputs.actions.detach().cpu(),
            checkpoint_path=self.config.checkpoint,
            device=str(self._device),
            runtime="worldfoundry.xiaomi_robotics_0.in_tree_hf_runtime",
            metadata={
                "variant": self.config.variant,
                "checkpoint_revision": self.config.revision,
                "robot_type": self.config.robot_type,
                "camera_keys": self.config.camera_keys,
                "view_labels": self.config.view_labels,
                "image_size": self.config.image_size,
                "center_crop_ratio": self.config.center_crop_ratio,
                "seed": inputs["seed"],
                "num_steps": resolved_steps,
                "action_shape": action_shape,
                "action_horizon": action_horizon,
                "action_dim": action_shape[-1],
                "model_config_action_length": int(self.model.config.action_length),
                "torch_dtype": str(self._dtype).removeprefix("torch."),
                "attn_implementation": self._attn_implementation,
                "official_entrypoint": "MiBoTForActionGeneration.forward",
                "trust_remote_code": False,
                "upstream_github_revision": UPSTREAM_GITHUB_REVISION,
            },
        )
        if output_path is not None:
            artifact_path = write_json(Path(output_path), result)
            result["artifact_kind"] = "action_trace"
            result["artifact_path"] = str(artifact_path)
        return result


_RUNTIME_CACHE: dict[XiaomiRobotics0RuntimeConfig, XiaomiRobotics0Runtime] = {}


def clear_runtime_cache() -> None:
    """Drop process-local runtime references without touching checkpoint files."""

    _RUNTIME_CACHE.clear()


def runtime_for(config: XiaomiRobotics0RuntimeConfig) -> XiaomiRobotics0Runtime:
    """Return a cached runtime for an immutable configuration."""

    if config not in _RUNTIME_CACHE:
        _RUNTIME_CACHE[config] = XiaomiRobotics0Runtime(config)
    return _RUNTIME_CACHE[config]


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any] = (),
    checkpoint_path: str = "",
    device: str = "cuda",
    output_path: str | Path | None = None,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared callable-entrypoint form used by WorldFoundry action runtimes."""

    del action_context
    options = dict(runtime_options or {})
    if checkpoint_path:
        options["checkpoint_path"] = checkpoint_path
    config = XiaomiRobotics0RuntimeConfig.from_options(options, device=device)
    return runtime_for(config).predict(
        instruction=instruction,
        image=image,
        observation=observation,
        output_path=output_path,
        seed=options.get("seed"),
        num_steps=options.get("num_steps"),
        prompt_is_formatted=option_bool(options.get("prompt_is_formatted"), False),
    )


__all__ = [
    "ACTION_HORIZONS_BY_ROBOT_TYPE",
    "CheckpointSpec",
    "MODEL_ID",
    "OFFICIAL_CHECKPOINTS",
    "UPSTREAM_GITHUB_REVISION",
    "XiaomiRobotics0Runtime",
    "XiaomiRobotics0RuntimeConfig",
    "build_instruction_prompt",
    "checkpoint_spec",
    "clear_runtime_cache",
    "predict_action",
    "runtime_for",
]

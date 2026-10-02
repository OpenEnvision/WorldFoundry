"""Checkpoint-backed, in-tree inference runtime for Hy-Embodied-0.5-VLA."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from worldfoundry.core.io.paths import resolve_local_hf_model_path, resolve_worldfoundry_path
from worldfoundry.core.io.formats.serialization import jsonable, write_json
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

from .preprocessing import CAMERA_KEYS, build_model_batch
from .stats import HyVLANormalizationStats

_DATA_CONFIG = load_vla_va_wam_runtime_config("hy-embodied-vla")
_VARIANTS = _DATA_CONFIG.get("variants")
if not isinstance(_VARIANTS, Mapping):
    raise TypeError("hy-embodied-vla data config requires a variants mapping")
OFFICIAL_REPOSITORIES = {
    str(name): str(payload["repo_id"])
    for name, payload in _VARIANTS.items()
    if isinstance(payload, Mapping)
}
OFFICIAL_REVISIONS = {
    str(name): str(payload["revision"])
    for name, payload in _VARIANTS.items()
    if isinstance(payload, Mapping)
}
REQUIRED_CHECKPOINT_FILES = tuple(str(item) for item in _DATA_CONFIG["required_checkpoint_files"])
DEFAULT_VARIANT = str(_DATA_CONFIG.get("default_variant") or "")
_DEFAULTS = _DATA_CONFIG.get("defaults")
if not isinstance(_DEFAULTS, Mapping):
    raise TypeError("hy-embodied-vla data config requires a defaults mapping")
DEFAULT_TORCH_DTYPE = str(_DEFAULTS["torch_dtype"])
DEFAULT_LOCAL_FILES_ONLY = bool(_DEFAULTS["local_files_only"])
DEFAULT_ALLOW_CHECKPOINT_DOWNLOAD = bool(_DEFAULTS["allow_checkpoint_download"])
DEFAULT_REPLICATE_SINGLE_IMAGE = bool(_DEFAULTS["replicate_single_image"])
DEFAULT_STATE_FORMAT = str(_DEFAULTS["state_format"])
DEFAULT_BLEND_MODE = str(_DEFAULTS["blend_mode"])
DEFAULT_STRICT_CHECKPOINT = bool(_DEFAULTS["strict_checkpoint"])
_VARIANT_ALIASES = {
    str(alias).strip().lower().replace("_", "-"): str(name)
    for name, payload in _VARIANTS.items()
    if isinstance(payload, Mapping)
    for alias in payload.get("aliases", ())
}


def _normalize_variant(value: str | None) -> str:
    normalized = str(value or DEFAULT_VARIANT).strip().lower().replace("_", "-")
    normalized = _VARIANT_ALIASES.get(normalized, normalized)
    if normalized not in OFFICIAL_REPOSITORIES:
        raise ValueError(f"Unsupported Hy-VLA variant {value!r}; choose 'umi' or 'robotwin'")
    return normalized


def select_hy_embodied_vla_checkpoint(
    *,
    checkpoint: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]] = (),
    variant: str = DEFAULT_VARIANT,
) -> str:
    """Select an explicit path/repo or one matching profile checkpoint record."""

    if checkpoint:
        return str(checkpoint)
    selected_variant = _normalize_variant(variant)
    records = [dict(item) for item in checkpoints]
    for record in records:
        haystack = " ".join(
            str(record.get(key) or "") for key in ("id", "role", "repo_id", "local_dir")
        ).lower()
        if selected_variant not in haystack:
            continue
        local = record.get("local_dir")
        if local:
            expanded = resolve_worldfoundry_path(str(local))
            if expanded.is_dir() and all(
                (expanded / filename).is_file() for filename in REQUIRED_CHECKPOINT_FILES
            ):
                return str(expanded.resolve())
        if record.get("repo_id"):
            return str(record["repo_id"])
        if local:
            return str(local)
    return OFFICIAL_REPOSITORIES[selected_variant]


@dataclass(frozen=True)
class HyEmbodiedVLARuntimeConfig:
    """All options that affect model loading or one-call inference semantics."""

    checkpoint: str
    variant: str = DEFAULT_VARIANT
    revision: str | None = None
    device: str = "cuda"
    torch_dtype: str = DEFAULT_TORCH_DTYPE
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY
    allow_checkpoint_download: bool = DEFAULT_ALLOW_CHECKPOINT_DOWNLOAD
    history_size: int | None = None
    replicate_single_image: bool = DEFAULT_REPLICATE_SINGLE_IMAGE
    state_format: str = DEFAULT_STATE_FORMAT
    blend_mode: str = DEFAULT_BLEND_MODE
    strict_checkpoint: bool = DEFAULT_STRICT_CHECKPOINT

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", _normalize_variant(self.variant))
        if not self.local_files_only or self.allow_checkpoint_download:
            raise ValueError(
                "Hy-VLA inference is local-only. Stage checkpoints with hfd under "
                "WORLDFOUNDRY_HFD_ROOT instead of enabling runtime downloads."
            )
        if self.revision is None and str(self.checkpoint).rstrip("/") == OFFICIAL_REPOSITORIES[self.variant]:
            object.__setattr__(self, "revision", OFFICIAL_REVISIONS[self.variant])
        state_format = self.state_format.strip().lower().replace("-", "_")
        if state_format not in {"normalized", "posrot20", "robotwin_wxyz"}:
            raise ValueError(
                "Hy-VLA state_format must be normalized, posrot20, or robotwin_wxyz"
            )
        object.__setattr__(self, "state_format", state_format)
        blend_mode = self.blend_mode.strip().lower().replace("-", "_")
        if blend_mode not in {"auto", "rel_only", "abs_only", "rel_abs"}:
            raise ValueError("Hy-VLA blend_mode must be auto, rel_only, abs_only, or rel_abs")
        object.__setattr__(self, "blend_mode", blend_mode)
        if self.history_size is not None and self.history_size < 1:
            raise ValueError("Hy-VLA history_size must be positive")


def build_plan_payload(
    *,
    config: HyEmbodiedVLARuntimeConfig,
    context: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a serializable plan without importing torch or resolving weights."""

    return {
        "schema_version": "worldfoundry-hy-embodied-vla-plan-v1",
        "model_id": "hy-embodied-vla",
        "profile": jsonable(profile),
        "context": jsonable(context),
        "runtime": {
            **jsonable(asdict(config)),
            "backend": (
                "worldfoundry.synthesis.action_generation.hy_embodied_vla.runtime:"
                "HyEmbodiedVLARuntime.predict_action"
            ),
            "official_entrypoint": "HyVLA.forward_evaluate(batch)['pred']",
            "required_checkpoint_files": list(REQUIRED_CHECKPOINT_FILES),
            "attention_backend_policy": (
                "external FlashAttention 2 on compatible SM80-SM90 CUDA devices; "
                "in-tree PyTorch SDPA on CPU, SM75, SM100+, or FA2 import/runtime failure"
            ),
        },
    }


class HyEmbodiedVLARuntime:
    """Lazy model runtime; importing or planning never loads torch/weights."""

    def __init__(self, config: HyEmbodiedVLARuntimeConfig):
        self.config = config
        self.checkpoint_dir: Path | None = None
        self.device = ""
        self.dtype: Any | None = None
        self.model_config: Any | None = None
        self.policy: Any | None = None
        self.stats: HyVLANormalizationStats | None = None

    def _resolve_checkpoint(self) -> Path:
        try:
            return resolve_local_hf_model_path(
                self.config.checkpoint,
                required_files=REQUIRED_CHECKPOINT_FILES,
            )
        except FileNotFoundError as local_error:
            raise FileNotFoundError(
                f"Hy-VLA checkpoint {self.config.checkpoint!r} is not staged locally. "
                "Stage the official snapshot under WORLDFOUNDRY_HFD_ROOT with hfd, "
                "or pass a complete local directory.\n"
                f"{local_error}"
            ) from local_error

    def _ensure_loaded(self) -> None:
        if self.policy is not None:
            return

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .configuration_hy_vla import HyVLAConfig
        from .modeling_hy_vla import HyVLA

        checkpoint = self._resolve_checkpoint()
        self.device = resolve_inference_device(self.config.device)
        self.dtype = resolve_inference_dtype(self.device, self.config.torch_dtype)
        model_config = HyVLAConfig.from_pretrained(checkpoint, local_files_only=True)
        if not getattr(model_config, "vlm_config_dict", None):
            raise ValueError(
                "WorldFoundry Hy-VLA inference requires a released self-contained VLA "
                "checkpoint with embedded vlm_config_dict; refusing an implicit base-VLM fetch"
            )
        policy = HyVLA.from_pretrained(
            str(checkpoint),
            config=model_config,
            local_files_only=True,
            map_location="cpu",
            torch_dtype=self.dtype,
            strict=self.config.strict_checkpoint,
        )
        policy.enable_video_encoder_if_needed()
        policy = policy.to(device=self.device, dtype=self.dtype).eval()
        policy.requires_grad_(False)

        stats_path = checkpoint / "norm_stats.pkl"
        self.stats = HyVLANormalizationStats.load(stats_path) if stats_path.is_file() else None
        self.checkpoint_dir = checkpoint
        self.model_config = model_config
        self.policy = policy

    @property
    def history_size(self) -> int:
        if self.config.history_size is not None:
            return self.config.history_size
        variant = _VARIANTS[self.config.variant]
        if not isinstance(variant, Mapping):
            raise TypeError(f"Hy-VLA variant {self.config.variant!r} must be a mapping")
        return int(variant["history_size"])

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()

    def _prepare_state(self, state: Any) -> tuple[np.ndarray, np.ndarray | None]:
        value = np.asarray(state, dtype=np.float32)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        start_pose_xyzw = None
        if self.config.state_format == "robotwin_wxyz":
            from .transforms import pose16_wxyz_to_posrot20, pose16_wxyz_to_xyzw

            start_pose_xyzw = pose16_wxyz_to_xyzw(value)
            value = pose16_wxyz_to_posrot20(value)
        if self.config.state_format in {"robotwin_wxyz", "posrot20"}:
            if self.stats is None:
                raise FileNotFoundError(
                    "Raw Hy-VLA state normalization requires norm_stats.pkl next to the checkpoint"
                )
            value = self.stats.normalize_state(value)
        if value.ndim != 1:
            raise ValueError(f"Hy-VLA state must resolve to one vector, got {value.shape}")
        return value.astype(np.float32, copy=False), start_pose_xyzw

    def _decode_actions(
        self,
        normalized: np.ndarray,
        *,
        start_pose_xyzw: np.ndarray | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"normalized_actions": normalized}
        if self.stats is None:
            result.update(
                actions=normalized,
                action_representation="normalized_model_action_tokens",
                blend_mode="none",
            )
            return result

        horizon = self.stats.action_horizon
        if self.stats.has_absolute_actions:
            if normalized.shape[0] != horizon * 2:
                raise ValueError(
                    f"Hy-VLA rel+abs checkpoint returned {normalized.shape[0]} steps; expected {horizon * 2}"
                )
            relative_normalized = normalized[:horizon]
            absolute_normalized = normalized[horizon:]
            absolute = self.stats.unnormalize_absolute(absolute_normalized)
            result["normalized_relative_actions"] = relative_normalized
            result["normalized_absolute_actions"] = absolute_normalized
            result["absolute_actions"] = absolute
        else:
            if normalized.shape[0] != horizon:
                raise ValueError(
                    f"Hy-VLA checkpoint returned {normalized.shape[0]} steps; expected {horizon}"
                )
            relative_normalized = normalized
            absolute = None

        relative = self.stats.unnormalize_relative(relative_normalized)
        result["relative_actions"] = relative
        requested_blend_mode = self.config.blend_mode
        blend_mode = requested_blend_mode
        if blend_mode == "auto":
            blend_mode = "rel_abs" if absolute is not None else "rel_only"
        if blend_mode in {"abs_only", "rel_abs"} and absolute is None:
            raise ValueError(f"blend_mode={blend_mode!r} requires absolute action statistics")

        if start_pose_xyzw is None:
            if blend_mode == "rel_abs":
                if requested_blend_mode != "auto":
                    raise ValueError(
                        "blend_mode='rel_abs' requires state_format='robotwin_wxyz' "
                        "so relative actions can be composed with the current pose"
                    )
                blend_mode = "rel_only"
                result["blend_fallback_reason"] = (
                    "state_format did not provide the robotwin_wxyz start pose required "
                    "for executable relative/absolute pose blending"
                )
            if blend_mode == "abs_only":
                result["actions"] = absolute
                result["action_representation"] = "absolute_pos_rot6d_gripper"
            else:
                result["actions"] = relative
                result["action_representation"] = "rt_relative_pos_rot6d_gripper"
            result["blend_mode"] = blend_mode
            return result

        from .transforms import (
            absolute20_to_pose16_xyzw,
            blend_pose16_xyzw,
            pose16_xyzw_to_wxyz,
            relative20_to_pose16_xyzw,
        )

        relative_pose = relative20_to_pose16_xyzw(relative, start_pose_xyzw)
        if blend_mode == "rel_only":
            pose = relative_pose
        else:
            absolute_pose = absolute20_to_pose16_xyzw(absolute)
            pose = absolute_pose if blend_mode == "abs_only" else blend_pose16_xyzw(relative_pose, absolute_pose)
        result["actions"] = pose16_xyzw_to_wxyz(pose)
        result["action_representation"] = "robotwin_dual_arm_xyz_quat_wxyz_gripper"
        result["blend_mode"] = blend_mode
        return result

    def predict_action(
        self,
        *,
        instruction: str,
        images: Any = None,
        state: Any = None,
        observation: Mapping[str, Any] | None = None,
        output_path: str | Path | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the official ``forward_evaluate`` path and emit an action trace."""

        self._ensure_loaded()
        import torch

        assert self.policy is not None
        assert self.model_config is not None
        assert self.checkpoint_dir is not None
        assert self.dtype is not None

        source = dict(observation or {})
        if images is None:
            camera_mapping = {key: source[key] for key in CAMERA_KEYS if source.get(key) is not None}
            images = camera_mapping or source.get("images") or source.get("image")
        if state is None:
            state = source.get("observation.state", source.get("state"))
        if state is None:
            raise ValueError("Hy-VLA inference requires observation.state or state")

        normalized_state, start_pose_xyzw = self._prepare_state(state)
        batch = build_model_batch(
            images=images,
            state=normalized_state,
            instruction=instruction,
            use_video_encoder=bool(self.model_config.use_video_encoder),
            history_size=self.history_size,
            replicate_single_image=self.config.replicate_single_image,
        )
        for key, value in tuple(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device=self.device, dtype=self.dtype)

        started = time.perf_counter()
        self.policy.reset()
        with torch.inference_mode():
            prediction = self.policy.forward_evaluate(batch)["pred"]
        from .modeling_hunyuan_vl_mot import (
            get_hunyuan_attention_backend,
            get_hunyuan_vision_attention_backend,
        )

        mot_backend = get_hunyuan_attention_backend()
        if mot_backend == "uninitialized":
            # HyVLA inference normally enters the shared dual-tower attention
            # loop directly, so the standalone VLM MoT dispatcher is not run.
            mot_backend = "not_invoked_by_vla_dual_tower"
        normalized = prediction[0].float().cpu().numpy().astype(np.float32, copy=False)
        decoded = self._decode_actions(normalized, start_pose_xyzw=start_pose_xyzw)
        config_bytes = (self.checkpoint_dir / "config.json").read_bytes()
        payload = {
            "schema_version": "worldfoundry-hy-embodied-vla-action-trace-v1",
            "status": "completed",
            "model_id": "hy-embodied-vla",
            "variant": self.config.variant,
            "artifact_kind": "action_trace",
            "instruction": instruction,
            "action_horizon": int(np.asarray(decoded["actions"]).shape[0]),
            "runtime": {
                "backend": "HyVLA.forward_evaluate(batch)['pred']",
                "checkpoint": self.config.checkpoint,
                "checkpoint_revision": self.config.revision,
                "checkpoint_dir": str(self.checkpoint_dir),
                "checkpoint_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "device": self.device,
                "torch_dtype": str(self.dtype).removeprefix("torch."),
                "use_video_encoder": bool(self.model_config.use_video_encoder),
                "history_size": self.history_size,
                "attention_backends": {
                    "vla_dual_tower": self.model_config.attention_implementation,
                    "mot": mot_backend,
                    "vision_spatial": get_hunyuan_vision_attention_backend(),
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
            **decoded,
            "metadata": dict(extra_metadata or {}),
        }
        serializable = jsonable(payload)
        if output_path is not None:
            destination = Path(output_path).expanduser().resolve()
            serializable["artifact_path"] = str(destination)
            write_json(destination, serializable)
        return serializable


__all__ = [
    "HyEmbodiedVLARuntime",
    "HyEmbodiedVLARuntimeConfig",
    "OFFICIAL_REPOSITORIES",
    "OFFICIAL_REVISIONS",
    "REQUIRED_CHECKPOINT_FILES",
    "DEFAULT_VARIANT",
    "DEFAULT_TORCH_DTYPE",
    "DEFAULT_LOCAL_FILES_ONLY",
    "DEFAULT_ALLOW_CHECKPOINT_DOWNLOAD",
    "DEFAULT_REPLICATE_SINGLE_IMAGE",
    "DEFAULT_STATE_FORMAT",
    "DEFAULT_BLEND_MODE",
    "DEFAULT_STRICT_CHECKPOINT",
    "build_plan_payload",
    "select_hy_embodied_vla_checkpoint",
]

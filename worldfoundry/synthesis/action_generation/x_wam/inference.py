"""Checkpoint-backed X-WAM world-action inference."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.media.types import MediaKind, infer_media_kind
from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config


@dataclass(frozen=True)
class OfficialCheckpoint:
    """Immutable metadata for the official X-WAM Hub repository."""

    repo_id: str
    revision: str
    license: str


@dataclass(frozen=True)
class OfficialVariant:
    """One named checkpoint subtree in the official repository."""

    checkpoint_dir: str
    benchmark: str
    state_dimension: int
    action_dimension: int
    camera_keys: tuple[str, str, str]
    deployable: bool = True

    @property
    def config_file(self) -> str:
        return f"{self.checkpoint_dir}/config.yaml"

    @property
    def weight_file(self) -> str:
        return f"{self.checkpoint_dir}/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt"


_DATA_CONFIG = load_vla_va_wam_runtime_config("x-wam")
_CHECKPOINT_SPEC = _DATA_CONFIG["official_checkpoint"]
_BASE_MODEL_SPEC = _DATA_CONFIG["official_base_model"]
OFFICIAL_CHECKPOINT = OfficialCheckpoint(
    repo_id=str(_CHECKPOINT_SPEC["repo_id"]),
    revision=str(_CHECKPOINT_SPEC["revision"]),
    license=str(_CHECKPOINT_SPEC["license"]),
)
OFFICIAL_BASE_MODEL = OfficialCheckpoint(
    repo_id=str(_BASE_MODEL_SPEC["repo_id"]),
    revision=str(_BASE_MODEL_SPEC["revision"]),
    license=str(_BASE_MODEL_SPEC["license"]),
)
OFFICIAL_VARIANTS: dict[str, OfficialVariant] = {
    str(name): OfficialVariant(
        checkpoint_dir=str(payload["checkpoint_dir"]),
        benchmark=str(payload["benchmark"]),
        state_dimension=int(payload["state_dimension"]),
        action_dimension=int(payload["action_dimension"]),
        camera_keys=tuple(str(item) for item in payload["camera_keys"]),
        deployable=bool(payload["deployable"]),
    )
    for name, payload in _DATA_CONFIG["variants"].items()
}
DEFAULT_VARIANT = str(_DATA_CONFIG["variant"])

_BASE_REQUIRED_FILES = tuple(str(item) for item in _DATA_CONFIG["base_required_files"])
_BASE_ALLOW_PATTERNS = _BASE_REQUIRED_FILES


@dataclass(frozen=True)
class XWAMRuntimeConfig:
    """Runtime settings for one official X-WAM checkpoint variant."""

    checkpoint_location: str = OFFICIAL_CHECKPOINT.repo_id
    revision: str | None = None
    variant: str = DEFAULT_VARIANT
    base_model_location: str = OFFICIAL_BASE_MODEL.repo_id
    base_revision: str | None = None
    device: str = "cuda"
    torch_dtype: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = True
    denoise_steps: int = 50
    action_denoise_steps: int = 10
    cfg_scale: float = 0.0
    compile_model: bool = False
    generate_world: bool = False
    run_depth: bool = False
    world_video_path: str | None = None

    def __post_init__(self) -> None:
        if not self.local_files_only:
            raise ValueError("X-WAM runtime is local-only")


def _resolve_local_snapshot(
    location: str,
    *,
    revision: str,
    required_files: Sequence[str],
    allow_patterns: Sequence[str],
    cache_dir: str | None,
    local_files_only: bool,
) -> Path:
    del revision, allow_patterns, cache_dir
    if not local_files_only:
        raise ValueError("X-WAM runtime is local-only")
    try:
        return resolve_local_hf_model_path(location, required_files=required_files)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "X-WAM requires a complete checkpoint staged in local WorldFoundry/HF storage; "
            f"no runtime download is permitted for {location!r}"
        ) from error


def _resolve_policy_assets(config: XWAMRuntimeConfig, variant: OfficialVariant) -> tuple[Path, Path, Path]:
    location = Path(config.checkpoint_location).expanduser()
    relative_weight = Path("checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt")

    def complete_weight(path: Path) -> bool:
        return path.is_file() and not Path(f"{path}.aria2").exists()

    if location.is_file():
        if location.name != relative_weight.name:
            raise ValueError(f"X-WAM expected a DeepSpeed model-state .pt file, got: {location}")
        if not complete_weight(location):
            raise FileNotFoundError(f"X-WAM checkpoint transfer is incomplete: {location}")
        variant_root = location.parents[3]
        config_file = variant_root / "config.yaml"
        if not config_file.is_file():
            raise FileNotFoundError(f"X-WAM config.yaml was not found beside checkpoint: {config_file}")
        return variant_root, config_file, location.resolve()
    if location.is_dir():
        direct_config = location / "config.yaml"
        direct_weight = location / relative_weight
        if direct_config.is_file() and complete_weight(direct_weight):
            return location.resolve(), direct_config.resolve(), direct_weight.resolve()
        nested_config = location / variant.config_file
        nested_weight = location / variant.weight_file
        if nested_config.is_file() and complete_weight(nested_weight):
            return nested_config.parent.resolve(), nested_config.resolve(), nested_weight.resolve()

    revision = str(config.revision or OFFICIAL_CHECKPOINT.revision)
    required = (variant.config_file, variant.weight_file)
    root = _resolve_local_snapshot(
        config.checkpoint_location,
        revision=revision,
        required_files=required,
        allow_patterns=required,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    return (
        (root / variant.checkpoint_dir).resolve(),
        (root / variant.config_file).resolve(),
        (root / variant.weight_file).resolve(),
    )


def _resolve_base_assets(config: XWAMRuntimeConfig) -> tuple[Path, Path, Path]:
    revision = str(config.base_revision or OFFICIAL_BASE_MODEL.revision)
    root = _resolve_local_snapshot(
        config.base_model_location,
        revision=revision,
        required_files=_BASE_REQUIRED_FILES,
        allow_patterns=_BASE_ALLOW_PATTERNS,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    return root, root / "config.json", root / "google/umt5-xxl"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"X-WAM config must be a mapping: {path}")
    return {str(key): item for key, item in value.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        try:
            return _jsonable(value.detach().cpu())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value)


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if _present(value):
            return value
    return None


def _select_views(image: Any, observation: Mapping[str, Any], variant: OfficialVariant) -> list[Any]:
    candidates: list[Any] = []
    mappings: list[Mapping[str, Any]] = [observation]
    if isinstance(image, Mapping):
        mappings.insert(0, image)
    for mapping in tuple(mappings):
        for key in ("camera_views", "rgb_views", "multi_view_rgb", "images"):
            nested = mapping.get(key)
            if isinstance(nested, Mapping):
                mappings.append(nested)
            elif isinstance(nested, (list, tuple)) and len(nested) >= 3:
                return list(nested[:3])

    for mapping in mappings:
        values = [mapping.get(key) for key in variant.camera_keys]
        if all(_present(value) for value in values):
            return values
    for mapping in mappings:
        for key in (
            "head_camera",
            "left_camera",
            "right_camera",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
            "full_image",
            "wrist_image",
            "image",
            "rgb",
        ):
            value = mapping.get(key)
            if _present(value) and all(value is not existing for existing in candidates):
                candidates.append(value)
            if len(candidates) == 3:
                return candidates
    if isinstance(image, (list, tuple)) and len(image) >= 3:
        return list(image[:3])
    if not isinstance(image, Mapping) and _present(image):
        candidates.insert(0, image)
    if len(candidates) < 3:
        raise ValueError(
            "X-WAM requires exactly three RGB views; provide a three-item image list or official camera keys "
            f"{variant.camera_keys}"
        )
    return candidates[:3]


def _to_normalized_rgb(value: Any) -> Any:
    import numpy as np
    from PIL import Image

    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"))
    elif isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"X-WAM image path does not exist: {path}")
        if infer_media_kind(path) is not MediaKind.IMAGE:
            raise ValueError(f"X-WAM expected an image file, got: {path}")
        array = np.asarray(Image.open(path).convert("RGB"))
    else:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.numpy()
            else:
                array = np.asarray(value)
        except ImportError:
            array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(f"X-WAM image must be HxWxC or CxHxW, got shape {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"X-WAM image must have three color channels, got shape {array.shape}")

    original_kind = array.dtype.kind
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("X-WAM image contains NaN or infinite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if original_kind in {"u", "i"} or (minimum >= 0.0 and maximum > 1.0 and maximum <= 255.0):
        array = array / 127.5 - 1.0
    elif minimum >= 0.0 and maximum <= 1.0:
        array = array * 2.0 - 1.0
    elif minimum < -1.0001 or maximum > 1.0001:
        raise ValueError(f"X-WAM float image range must be [-1,1], [0,1], or [0,255], got [{minimum}, {maximum}]")
    return np.ascontiguousarray(array, dtype=np.float32)


def _resize_and_center_crop(tensor: Any, size: Sequence[int], crop_ratio: float) -> Any:
    import torch.nn.functional as functional

    batch, views, _channels, _height, _width = tensor.shape
    target_h, target_w = (int(size[0]), int(size[1]))
    flattened = tensor.flatten(0, 1)
    resized = functional.interpolate(
        flattened,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).unflatten(0, (batch, views))
    crop_h = max(1, int(target_h * crop_ratio))
    crop_w = max(1, int(target_w * crop_ratio))
    top = (target_h - crop_h) // 2
    left = (target_w - crop_w) // 2
    cropped = resized[..., top : top + crop_h, left : left + crop_w]
    return functional.interpolate(
        cropped.flatten(0, 1),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).unflatten(0, (batch, views))


def _build_statistics(policy_config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, bool]:
    import numpy as np

    dataset = policy_config.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("statistics"), Mapping):
        raise ValueError(
            "The X-WAM pretrained checkpoint has no deployment normalization statistics; "
            "select robocasa_sft or robotwin_sft for policy inference."
        )
    statistics = dataset["statistics"]
    q01 = statistics.get("q01")
    q99 = statistics.get("q99")
    if not isinstance(q01, Mapping) or not isinstance(q99, Mapping):
        raise ValueError("X-WAM dataset statistics must contain q01 and q99 mappings")
    has_right_arm = "proprio_right_ee_xyz" in q01

    state_q01 = list(q01["proprio_left_ee_xyz"]) + [-1.0] * 4 + list(q01["gripper_pos"])
    state_q99 = list(q99["proprio_left_ee_xyz"]) + [1.0] * 4 + list(q99["gripper_pos"])
    if has_right_arm:
        state_q01 += list(q01["proprio_right_ee_xyz"]) + [-1.0] * 4 + list(q01["gripper_pos"])
        state_q99 += list(q99["proprio_right_ee_xyz"]) + [1.0] * 4 + list(q99["gripper_pos"])
    else:
        state_q01 += [-1.0] * 8
        state_q99 += [1.0] * 8

    action_q01 = list(q01["action_left_ee_xyz"]) + list(q01["action_left_ee_axisangle"]) + list(q01["gripper_action"])
    action_q99 = list(q99["action_left_ee_xyz"]) + list(q99["action_left_ee_axisangle"]) + list(q99["gripper_action"])
    if has_right_arm:
        action_q01 += (
            list(q01["action_right_ee_xyz"]) + list(q01["action_right_ee_axisangle"]) + list(q01["gripper_action"])
        )
        action_q99 += (
            list(q99["action_right_ee_xyz"]) + list(q99["action_right_ee_axisangle"]) + list(q99["gripper_action"])
        )
    return (
        np.asarray(state_q01, dtype=np.float32),
        np.asarray(state_q99, dtype=np.float32),
        np.asarray(action_q01, dtype=np.float32),
        np.asarray(action_q99, dtype=np.float32),
        has_right_arm,
    )


def _compute_seed(env_rank: int, rollout_id: int, step_id: int) -> int:
    key = f"{env_rank}_{rollout_id}_{step_id}"
    return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16) % (2**32)


def _reset_rope_tables(model: Any) -> None:
    import torch

    from .modeling import rope_params

    dimension = model.dim // model.num_heads
    model.video_freqs = [
        rope_params(1024, dimension - 4 * (dimension // 6)),
        rope_params(1024, 2 * (dimension // 6)),
        rope_params(1024, 2 * (dimension // 6)),
    ]
    model.action_freqs = torch.cat(
        [
            rope_params(1024, dimension - 4 * (dimension // 6), scale=model.action_num * 4),
            model.video_freqs[1][0:1].repeat(1024 * model.action_num * 4, 1),
            model.video_freqs[2][0:1].repeat(1024 * model.action_num * 4, 1),
        ],
        dim=-1,
    )
    model.proprio_freqs = torch.cat(
        [
            rope_params(1024, dimension - 4 * (dimension // 6), scale=4),
            model.video_freqs[1][0:1].repeat(1024 * 4, 1),
            model.video_freqs[2][0:1].repeat(1024 * 4, 1),
        ],
        dim=-1,
    )


def _runner_class() -> type[Any]:
    import torch
    from einops import rearrange

    from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl
    from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
    from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import (
        FlowUniPCMultistepScheduler,
    )
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import WanVAE_

    from .modeling import XWAMModel

    class TextEncoder(torch.nn.Module):
        def __init__(self, *, text_len: int, dtype: Any, tokenizer_path: Path) -> None:
            super().__init__()
            self.text_len = text_len
            self.dtype = dtype
            self.checkpoint_path = None
            self.tokenizer_path = str(tokenizer_path)
            with torch.device("meta"):
                self.model = umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=dtype,
                    device="meta",
                ).eval()
            self.tokenizer = HuggingfaceTokenizer(
                name=str(tokenizer_path),
                seq_len=text_len,
                clean="whitespace",
                local_files_only=True,
            )

        def forward(self, texts: Sequence[str]) -> Any:
            ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
            device = self.model.token_embedding.weight.device
            return self.model(ids.to(device), mask.to(device))

    class VideoVAE(torch.nn.Module):
        def __init__(self, *, dtype: Any) -> None:
            super().__init__()
            self.dtype = dtype
            mean = torch.tensor(
                [
                    -0.2289,
                    -0.0052,
                    -0.1323,
                    -0.2339,
                    -0.2799,
                    0.0174,
                    0.1838,
                    0.1557,
                    -0.1382,
                    0.0542,
                    0.2813,
                    0.0891,
                    0.1570,
                    -0.0098,
                    0.0375,
                    -0.1825,
                    -0.2246,
                    -0.1207,
                    -0.0698,
                    0.5109,
                    0.2665,
                    -0.2108,
                    -0.2158,
                    0.2502,
                    -0.2055,
                    -0.0322,
                    0.1109,
                    0.1567,
                    -0.0729,
                    0.0899,
                    -0.2799,
                    -0.1230,
                    -0.0313,
                    -0.1649,
                    0.0117,
                    0.0723,
                    -0.2839,
                    -0.2083,
                    -0.0520,
                    0.3748,
                    0.0152,
                    0.1957,
                    0.1433,
                    -0.2944,
                    0.3573,
                    -0.0548,
                    -0.1681,
                    -0.0667,
                ],
                dtype=dtype,
            )
            std = torch.tensor(
                [
                    0.4765,
                    1.0364,
                    0.4514,
                    1.1677,
                    0.5313,
                    0.4990,
                    0.4818,
                    0.5013,
                    0.8158,
                    1.0344,
                    0.5894,
                    1.0901,
                    0.6885,
                    0.6165,
                    0.8454,
                    0.4978,
                    0.5759,
                    0.3523,
                    0.7135,
                    0.6804,
                    0.5833,
                    1.4146,
                    0.8986,
                    0.5659,
                    0.7069,
                    0.5338,
                    0.4889,
                    0.4917,
                    0.4069,
                    0.4999,
                    0.6866,
                    0.4093,
                    0.5709,
                    0.6065,
                    0.6415,
                    0.4944,
                    0.5726,
                    1.2042,
                    0.5458,
                    1.6887,
                    0.3971,
                    1.0600,
                    0.3943,
                    0.5537,
                    0.5444,
                    0.4089,
                    0.7468,
                    0.7744,
                ],
                dtype=dtype,
            )
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            with torch.device("meta"):
                self.model = WanVAE_(
                    dim=160,
                    z_dim=48,
                    dim_mult=[1, 2, 4, 4],
                    num_res_blocks=2,
                    attn_scales=[],
                    temperal_downsample=[False, True, True],
                    dropout=0.0,
                )

        def encode(self, videos: Any) -> Any:
            enabled = videos.device.type == "cuda"
            with torch.amp.autocast(videos.device.type, dtype=self.dtype, enabled=enabled):
                return self.model.encode(videos, [self.mean, 1.0 / self.std]).float()

        def decode(self, latents: Any) -> Any:
            enabled = latents.device.type == "cuda"
            with torch.amp.autocast(latents.device.type, dtype=self.dtype, enabled=enabled):
                return self.model.decode(latents, [self.mean, 1.0 / self.std]).float().clamp_(-1, 1)

    class InferenceRunner(torch.nn.Module):
        def __init__(
            self,
            *,
            policy_config: Mapping[str, Any],
            model_config: Mapping[str, Any],
            tokenizer_path: Path,
            dtype: Any,
        ) -> None:
            super().__init__()
            dataset = policy_config["dataset"]
            self.num_views = 3
            self.num_modalities = 2 if bool(policy_config.get("use_depth", True)) else 1
            self.frame_num = int(policy_config.get("frame_num", 9))
            self.sample_steps = int(policy_config.get("sample_steps", 50))
            self.action_denoise_steps = int(policy_config.get("action_denoise_steps", 10))
            self.use_decoupled_inference = bool(policy_config.get("use_decoupled_inference", True))
            self.flow_matching_num_train_timesteps = int(policy_config.get("flow_matching_num_train_timesteps", 1000))
            self.time_shifting = float(policy_config.get("time_shifting", 5.0))
            self.use_depth = bool(policy_config.get("use_depth", True))
            self.sample_fps = int(policy_config.get("sample_fps", 5))
            self.action_dim = int(policy_config.get("action_dim", 14))
            self.proprio_dim = int(policy_config.get("proprio_dim", 16))
            self.action_num = int(dataset["frame_skip"]) // int(dataset["action_skip"])
            self.text_encoder = TextEncoder(
                text_len=int(policy_config.get("text_len", 512)),
                dtype=dtype,
                tokenizer_path=tokenizer_path,
            )
            self.vae = VideoVAE(dtype=dtype)
            constructor = dict(model_config)
            constructor.pop("_class_name", None)
            constructor.pop("_diffusers_version", None)
            constructor.update(
                num_modalities=self.num_modalities,
                num_views=self.num_views,
                action_dim=self.action_dim,
                action_num=self.action_num,
                proprio_dim=self.proprio_dim,
                num_extra_layers=int(policy_config.get("num_extra_layers", 10)),
            )
            with torch.device("meta"):
                self.model = XWAMModel(**constructor)

        def _prepare_condition(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
            context = self.text_encoder(batch["prompt"])
            batch_size = batch["video"].shape[0]
            rgb = rearrange(batch["video"], "b v t c h w -> (b v) c t h w")
            latents = self.vae.encode(rgb)
            latents = rearrange(latents, "(b v) c t h w -> b c t v h w", b=batch_size, v=self.num_views)
            return context, latents

        def forward(
            self,
            batch: Mapping[str, Any],
            *,
            seeds: Sequence[int],
            early_stop: bool,
            cfg: float,
            run_depth: bool,
        ) -> tuple[Any, Any, Any, Any]:
            context, clean_latents = self._prepare_condition(batch)
            batch_size, _channels, latent_frames, _views, _height, _width = clean_latents.shape
            clean_actions = batch["actions"].float()
            clean_proprios = batch["proprios"].float()
            if cfg > 0.0:
                context = torch.cat([context, self.text_encoder([""] * batch_size)], dim=0)

            latent_noise: list[Any] = []
            action_noise: list[Any] = []
            proprio_noise: list[Any] = []
            for index, seed in enumerate(seeds):
                generator = torch.Generator(device=clean_latents.device).manual_seed(int(seed))
                latent_noise.append(
                    torch.randn(
                        clean_latents[index : index + 1].shape,
                        generator=generator,
                        device=clean_latents.device,
                        dtype=clean_latents.dtype,
                    )
                )
                action_noise.append(
                    torch.randn(
                        clean_actions[index : index + 1].shape,
                        generator=generator,
                        device=clean_actions.device,
                        dtype=clean_actions.dtype,
                    )
                )
                proprio_noise.append(
                    torch.randn(
                        clean_proprios[index : index + 1].shape,
                        generator=generator,
                        device=clean_proprios.device,
                        dtype=clean_proprios.dtype,
                    )
                )
            noisy_latents = torch.cat(latent_noise)
            noisy_actions = torch.cat(action_noise)
            noisy_proprios = torch.cat(proprio_noise)

            latent_mask = torch.zeros(
                (batch_size, 1, latent_frames, 1, 1, 1),
                dtype=torch.long,
                device=clean_latents.device,
            )
            latent_mask[:, :, 0] = 1
            latents = clean_latents * latent_mask + noisy_latents * (1 - latent_mask)
            action_mask = torch.zeros(
                (batch_size, clean_actions.shape[1], 1),
                dtype=torch.long,
                device=clean_actions.device,
            )
            actions = clean_actions * action_mask + noisy_actions * (1 - action_mask)
            proprio_mask = torch.zeros(
                (batch_size, clean_proprios.shape[1], 1),
                dtype=torch.long,
                device=clean_proprios.device,
            )
            proprio_mask[:, 0] = 1
            proprios = clean_proprios * proprio_mask + noisy_proprios * (1 - proprio_mask)

            action_steps = self.action_denoise_steps if self.use_decoupled_inference else self.sample_steps
            scheduler_kwargs = {
                "num_train_timesteps": self.flow_matching_num_train_timesteps,
                "shift": 1,
                "use_dynamic_shifting": False,
            }
            video_scheduler = FlowUniPCMultistepScheduler(**scheduler_kwargs)
            action_scheduler = FlowUniPCMultistepScheduler(**scheduler_kwargs)
            proprio_scheduler = FlowUniPCMultistepScheduler(**scheduler_kwargs)
            video_scheduler.set_timesteps(
                self.sample_steps,
                device=clean_latents.device,
                shift=self.time_shifting,
            )
            action_scheduler.set_timesteps(
                action_steps,
                device=clean_latents.device,
                shift=self.time_shifting,
            )
            proprio_scheduler.set_timesteps(
                action_steps,
                device=clean_latents.device,
                shift=self.time_shifting,
            )

            extra_predictions: Any = [None]
            for index in range(self.sample_steps):
                video_timestep = video_scheduler.timesteps[index]
                if self.use_decoupled_inference and index >= action_steps:
                    if early_stop:
                        break
                    action_timestep: Any = 0
                    proprio_timestep: Any = 0
                else:
                    action_timestep = action_scheduler.timesteps[index]
                    proprio_timestep = proprio_scheduler.timesteps[index]

                latent_timesteps = video_timestep * (1 - latent_mask).view(batch_size, latent_frames)
                action_timesteps = action_timestep * (1 - action_mask).view(batch_size, clean_actions.shape[1])
                proprio_timesteps = proprio_timestep * (1 - proprio_mask).view(batch_size, clean_proprios.shape[1])
                latent_velocity, action_velocity, proprio_velocity, extra_predictions = self.model(
                    x=latents,
                    t=latent_timesteps,
                    context=context,
                    actions=actions,
                    t_actions=action_timesteps,
                    proprios=proprios,
                    t_proprios=proprio_timesteps,
                    cfg=cfg,
                    run_depth=run_depth,
                )
                latents = video_scheduler.step(
                    latent_velocity,
                    video_timestep,
                    latents,
                    return_dict=False,
                )[0]
                latents = clean_latents * latent_mask + latents * (1 - latent_mask)
                if not self.use_decoupled_inference or index < action_steps:
                    actions = action_scheduler.step(
                        action_velocity,
                        action_timestep,
                        actions,
                        return_dict=False,
                    )[0]
                    actions = clean_actions * action_mask + actions * (1 - action_mask)
                    proprios = proprio_scheduler.step(
                        proprio_velocity,
                        proprio_timestep,
                        proprios,
                        return_dict=False,
                    )[0]
                    proprios = clean_proprios * proprio_mask + proprios * (1 - proprio_mask)

            depth_latents = extra_predictions[0] if self.use_depth and run_depth else None
            return latents, actions, proprios, depth_latents

        def generate(
            self,
            rgb: Any,
            proprio: Any,
            prompt: Sequence[str],
            *,
            seeds: Sequence[int],
            early_stop: bool,
            cfg: float,
            run_depth: bool,
        ) -> tuple[Any, Any, Any, Any]:
            batch_size = rgb.shape[0]
            action_count = (self.frame_num - 1) * self.action_num
            batch = {
                "video": rgb.unsqueeze(2).repeat(1, 1, self.frame_num, 1, 1, 1),
                "proprios": proprio.unsqueeze(1).repeat(1, self.frame_num, 1),
                "actions": torch.zeros(
                    (batch_size, action_count, self.action_dim),
                    device=proprio.device,
                    dtype=proprio.dtype,
                ),
                "prompt": list(prompt),
            }
            latents, actions, proprios, depth_latents = self.forward(
                batch,
                seeds=seeds,
                early_stop=early_stop,
                cfg=cfg,
                run_depth=run_depth,
            )
            if early_stop:
                return None, actions, proprios, None
            if self.use_depth and run_depth:
                latents = torch.cat([latents, depth_latents], dim=3)
            latents = rearrange(latents, "b c t (m v) h w -> (b m v) c t h w", v=self.num_views)
            videos = self.vae.decode(latents)
            videos = rearrange(
                videos,
                "(b m v) c t h w -> t (m h) (b v w) c",
                b=batch_size,
                v=self.num_views,
            )
            videos = torch.clamp((videos + 1) * 127.5, 0, 255).byte().cpu().numpy()
            return videos, actions, proprios, depth_latents

    return InferenceRunner


class XWAMRuntime:
    """Lazy, in-process runtime for the released X-WAM policy checkpoints."""

    def __init__(self, config: XWAMRuntimeConfig) -> None:
        if config.variant not in OFFICIAL_VARIANTS:
            raise ValueError(f"Unknown X-WAM variant {config.variant!r}; choose from {sorted(OFFICIAL_VARIANTS)}")
        if config.denoise_steps <= 0:
            raise ValueError("X-WAM denoise_steps must be positive")
        if config.action_denoise_steps <= 0 or config.action_denoise_steps > config.denoise_steps:
            raise ValueError("X-WAM action_denoise_steps must be in [1, denoise_steps]")
        self.config = config
        self.variant = OFFICIAL_VARIANTS[config.variant]
        if not self.variant.deployable:
            raise ValueError(
                "The official X-WAM pretrained subtree is a post-training base and contains no deployment "
                "normalization statistics; choose robocasa_sft or robotwin_sft."
            )
        self._runner: Any = None
        self._policy_config: dict[str, Any] | None = None
        self._statistics: tuple[Any, Any, Any, Any, bool] | None = None
        self._checkpoint_paths: tuple[Path, Path] | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._runner is not None:
            return
        import torch

        from worldfoundry.core.model_loading.checkpoints import assign_state_dict_strict
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        _variant_root, config_path, weight_path = _resolve_policy_assets(self.config, self.variant)
        base_root, model_config_path, tokenizer_path = _resolve_base_assets(self.config)
        del base_root
        policy_config = _load_yaml(config_path)
        dataset = policy_config.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError(f"X-WAM deployable config is missing dataset settings: {config_path}")
        if int(policy_config.get("action_dim", -1)) != 14 or int(policy_config.get("proprio_dim", -1)) != 16:
            raise ValueError("X-WAM official checkpoints require model action_dim=14 and proprio_dim=16")
        if int(dataset.get("frame_skip", 0)) // int(dataset.get("action_skip", 1)) != 4:
            raise ValueError("X-WAM official checkpoints require four actions per predicted video frame")
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if model_config.get("model_type") != "ti2v" or int(model_config.get("dim", 0)) != 3072:
            raise ValueError(f"X-WAM requires the official Wan2.2-TI2V-5B base config: {model_config_path}")

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        runner_cls = _runner_class()
        runner = runner_cls(
            policy_config=policy_config,
            model_config=model_config,
            tokenizer_path=tokenizer_path,
            dtype=dtype,
        )
        # The official DeepSpeed bundle stores a built-in set alongside the
        # tensor state. Keep restricted loading enabled and allow only that
        # audited built-in type instead of unpickling arbitrary objects.
        with torch.serialization.safe_globals([set]):
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        state_dict = checkpoint.get("module") if isinstance(checkpoint, Mapping) else None
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"X-WAM DeepSpeed checkpoint has no mapping-valued 'module' state: {weight_path}")
        if not state_dict or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise TypeError(
                f"X-WAM DeepSpeed 'module' state must be a non-empty string-to-tensor mapping: {weight_path}"
            )
        assign_state_dict_strict(runner, state_dict, label=f"X-WAM checkpoint {weight_path}")
        _reset_rope_tables(runner.model)
        runner.sample_steps = self.config.denoise_steps
        runner.action_denoise_steps = self.config.action_denoise_steps
        runner.use_decoupled_inference = self.config.action_denoise_steps < self.config.denoise_steps
        runner = runner.to(device=device, dtype=dtype).eval().requires_grad_(False)
        if self.config.compile_model:
            runner.model = torch.compile(runner.model)
        self._runner = runner
        self._policy_config = policy_config
        self._statistics = _build_statistics(policy_config)
        self._checkpoint_paths = (config_path, weight_path)
        self._device = device

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Predict the official 32-step action chunk and optional future world video."""

        import numpy as np
        import torch

        self._load()
        assert self._runner is not None
        assert self._policy_config is not None
        assert self._statistics is not None
        assert self._checkpoint_paths is not None
        assert self._device is not None

        views = _select_views(image, observation, self.variant)
        dataset = self._policy_config["dataset"]
        # Real multi-camera deployments do not necessarily expose identical
        # native resolutions.  Normalize and resize each view independently;
        # stacking before this step rejects otherwise valid camera triples.
        resized_views = []
        for view in views:
            view_tensor = torch.from_numpy(_to_normalized_rgb(view)).permute(2, 0, 1)
            view_tensor = _resize_and_center_crop(
                view_tensor.unsqueeze(0).unsqueeze(0),
                size=dataset.get("video_size", (256, 320)),
                crop_ratio=float(dataset.get("crop_ratio", 0.95)),
            )
            resized_views.append(view_tensor)
        rgb = torch.cat(resized_views, dim=1)
        rgb = rgb.to(device=self._device, dtype=next(self._runner.parameters()).dtype)

        state_value = _first_present(observation, "proprios", "proprio", "robot_state", "state")
        if state_value is None:
            raise ValueError("X-WAM requires current robot state under proprios/proprio/robot_state/state")
        state = np.asarray(state_value, dtype=np.float32).reshape(-1)
        if state.shape[0] == 8 and self.variant.state_dimension == 8:
            state = np.concatenate([state, np.zeros(8, dtype=np.float32)])
        if state.shape[0] != 16:
            raise ValueError(
                f"X-WAM {self.config.variant} expects {self.variant.state_dimension} state values "
                f"(internally padded to 16), got {state.shape[0]}"
            )
        state_q01, state_q99, action_q01, action_q99, has_right_arm = self._statistics
        normalized_state = 2.0 * (state - state_q01) / (state_q99 - state_q01) - 1.0
        if not has_right_arm:
            normalized_state[8:] = 0.0
        proprio = torch.from_numpy(normalized_state).unsqueeze(0)
        proprio = proprio.to(device=self._device, dtype=rgb.dtype)

        explicit_seed = observation.get("seed")
        seed = (
            int(explicit_seed)
            if explicit_seed is not None
            else _compute_seed(
                int(observation.get("env_rank", 0)),
                int(observation.get("rollout_id", 0)),
                int(observation.get("step_id", 0)),
            )
        )
        started = time.time()
        with torch.inference_mode():
            videos, actions, proprios, _depth = self._runner.generate(
                rgb,
                proprio,
                [instruction],
                seeds=[seed],
                early_stop=not self.config.generate_world,
                cfg=self.config.cfg_scale,
                run_depth=self.config.run_depth,
            )
        action_array = actions[0].float().cpu().numpy()
        proprio_array = proprios[0].float().cpu().numpy()
        proprio_array = (proprio_array + 1.0) / 2.0 * (state_q99 - state_q01) + state_q01
        action_array = (action_array[:, : self.variant.action_dimension] + 1.0) / 2.0 * (
            action_q99 - action_q01
        ) + action_q01
        if not has_right_arm:
            action_array[:, 6] *= -1.0

        result: dict[str, Any] = {
            "actions": action_array,
            "proprios": proprio_array,
            "seed": seed,
            "variant": self.config.variant,
            "benchmark": self.variant.benchmark,
            "action_shape": list(action_array.shape),
            "proprio_shape": list(proprio_array.shape),
            "elapsed_seconds": time.time() - started,
            "checkpoint": {
                "repo_id": OFFICIAL_CHECKPOINT.repo_id,
                "revision": str(self.config.revision or OFFICIAL_CHECKPOINT.revision),
                "config_path": str(self._checkpoint_paths[0]),
                "weight_path": str(self._checkpoint_paths[1]),
            },
            "base_model": {
                "repo_id": OFFICIAL_BASE_MODEL.repo_id,
                "revision": str(self.config.base_revision or OFFICIAL_BASE_MODEL.revision),
            },
        }
        if videos is not None:
            if not self.config.world_video_path:
                raise ValueError(
                    "X-WAM generate_world=True requires world_video_path to avoid embedding video in JSON"
                )
            from worldfoundry.core.media.codecs.video import write_video

            video_path = Path(self.config.world_video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            write_video(videos, video_path, fps=self._runner.sample_fps)
            result["predicted_world_video"] = str(video_path)
            result["predicted_world_video_shape"] = list(videos.shape)
            result["includes_depth_mosaic"] = bool(self.config.run_depth)
        return _jsonable(result)


__all__ = [
    "DEFAULT_VARIANT",
    "OFFICIAL_BASE_MODEL",
    "OFFICIAL_CHECKPOINT",
    "OFFICIAL_VARIANTS",
    "OfficialCheckpoint",
    "OfficialVariant",
    "XWAMRuntime",
    "XWAMRuntimeConfig",
]

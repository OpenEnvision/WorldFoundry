# SPDX-License-Identifier: MIT
"""Direct in-tree runtime for the released FastWAM action checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_bool,
    option_float,
    option_int,
    runtime_options_cache_key,
)

@dataclass(frozen=True)
class FastWAMRuntimeConfig:
    checkpoint_location: str
    variant: str
    checkpoint_file: str
    statistics_file: str
    action_dim: int
    proprio_dim: int
    image_height: int
    image_width: int
    normalization: str
    prompt_template: str
    wan_assets_ref: str
    tokenizer_ref: str
    video_config: Mapping[str, Any]
    action_config: Mapping[str, Any]
    vae_mean: Sequence[float]
    vae_std: Sequence[float]
    action_infer_shift: float
    action_num_train_timesteps: int
    vae_tile_size: Sequence[int]
    vae_tile_stride: Sequence[int]
    device: str = "cuda"
    torch_dtype: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = True
    revision: str | None = None
    statistics_path: str | None = None
    wan_assets_path: str | None = None
    tokenizer_path: str | None = None
    text_encoder_device: str | None = "cpu"
    action_horizon: int = 32
    num_inference_steps: int = 10
    sigma_shift: float | None = None
    seed: int = 42
    rand_device: str = "cpu"
    tiled: bool = False
    binarize_libero_gripper: bool = False
    action_infer_mode: str = "first_frame"
    num_video_frames: int = 5
    video_infer_shift: float = 5.0

class FastWAMRuntime:
    """Persistent released-checkpoint FastWAM runtime."""

    def __init__(self, config: FastWAMRuntimeConfig) -> None:
        if not config.local_files_only:
            raise ValueError("FastWAM requires local_files_only=true; runtime downloads are disabled")
        if config.variant not in {"libero", "libero_optional_idm", "robotwin"}:
            raise ValueError(f"Unsupported FastWAM variant: {config.variant!r}")
        if config.action_infer_mode not in {"first_frame", "idm"}:
            raise ValueError(f"Unsupported FastWAM action_infer_mode: {config.action_infer_mode!r}")
        if config.action_infer_mode == "idm" and config.variant != "libero_optional_idm":
            raise ValueError("FastWAM IDM mode requires the LIBERO Optional IDM checkpoint")
        if config.action_dim <= 0 or config.proprio_dim <= 0:
            raise ValueError("FastWAM action_dim and proprio_dim must be positive")
        if config.image_height <= 0 or config.image_width <= 0:
            raise ValueError("FastWAM image dimensions must be positive")
        if config.normalization not in {"min/max", "z-score"}:
            raise ValueError(f"Unsupported FastWAM normalization: {config.normalization!r}")
        if not config.checkpoint_file or not config.statistics_file:
            raise ValueError("FastWAM checkpoint_file and statistics_file are required")
        if config.action_num_train_timesteps <= 0:
            raise ValueError("FastWAM action_num_train_timesteps must be positive")
        if len(config.vae_tile_size) != 2 or len(config.vae_tile_stride) != 2:
            raise ValueError("FastWAM VAE tile size and stride must each contain two integers")
        self.config = config
        self._policy: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._release_root: Path | None = None
        self._checkpoint_file: Path | None = None
        self._statistics: dict[str, Any] | None = None
        self._text_encoder: Any = None
        self._tokenizer: Any = None
        self._context_cache: dict[str, tuple[Any, Any]] = {}

    @staticmethod
    def _existing_path(value: str | None) -> Path | None:
        if not value:
            return None
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        path = resolve_worldfoundry_path(value)
        return path.resolve() if path.exists() else None

    def _resolve_release(self) -> tuple[Path, Path]:
        if self._release_root is not None and self._checkpoint_file is not None:
            return self._release_root, self._checkpoint_file
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        direct = self._existing_path(self.config.checkpoint_location)
        if direct is not None and direct.is_file():
            checkpoint = direct
            root = direct.parent
        elif direct is not None:
            root = direct
            checkpoint = root / self.config.checkpoint_file
        else:
            root = materialize_hf_snapshot(
                self.config.checkpoint_location,
                revision=self.config.revision,
                cache_dir=self.config.cache_dir,
                allow_patterns=[
                    self.config.checkpoint_file,
                    self.config.statistics_file,
                ],
                required_files=[
                    self.config.checkpoint_file,
                    self.config.statistics_file,
                ],
                local_files_only=self.config.local_files_only,
            )
            checkpoint = root / self.config.checkpoint_file
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"FastWAM {self.config.variant} checkpoint is missing: {checkpoint}"
            )
        self._release_root = root
        self._checkpoint_file = checkpoint
        return root, checkpoint

    def _component_root(
        self,
        *,
        explicit: str | None,
        reference: str,
        required_files: Sequence[str],
        allow_patterns: Sequence[str] | None = None,
    ) -> Path:
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot

        direct = self._existing_path(explicit)
        location = str(direct) if direct is not None else reference
        return materialize_hf_snapshot(
            location,
            cache_dir=self.config.cache_dir,
            allow_patterns=allow_patterns,
            required_files=required_files,
            local_files_only=self.config.local_files_only,
        )

    @staticmethod
    def _reset_rope(video_expert: Any, action_expert: Any) -> None:
        from .wan_video_dit import precompute_freqs_cis, precompute_freqs_cis_3d

        video_expert.freqs = precompute_freqs_cis_3d(int(video_expert.attn_head_dim))
        action_expert.freqs = precompute_freqs_cis(int(action_expert.attn_head_dim), end=1024)

    def _reset_vae_scale(self, vae: Any) -> None:
        import torch

        if len(self.config.vae_mean) != 48 or len(self.config.vae_std) != 48:
            raise ValueError("FastWAM Wan VAE mean/std must each contain 48 values")
        vae.mean = torch.tensor(tuple(float(value) for value in self.config.vae_mean))
        vae.std = torch.tensor(tuple(float(value) for value in self.config.vae_std))
        vae.scale = [vae.mean, 1.0 / vae.std]

    def _load_policy(self) -> Any:
        if self._policy is not None:
            return self._policy
        import torch
        from torch import nn

        from worldfoundry.core.model_loading.checkpoints.assignment import assign_state_dict_strict
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.core.model_loading.checkpoints.file import load_state_dict, load_torch_checkpoint

        from .action_dit import ActionDiT
        from .mot import MoT
        from .policy import FastWAMPolicy
        from .state_dict_converters import wan_video_vae_state_dict_converter
        from .wan_video_dit import WanVideoDiT
        from .wan_video_vae import WanVideoVAE38

        _, checkpoint_file = self._resolve_release()
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        payload = load_torch_checkpoint(checkpoint_file, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("mot"), Mapping):
            raise ValueError("FastWAM released checkpoint has no mot state dictionary")

        video_config = dict(self.config.video_config)
        video_config["action_dim"] = self.config.action_dim
        action_config = dict(self.config.action_config)
        action_config["action_dim"] = self.config.action_dim
        with torch.device("meta"):
            video_expert = WanVideoDiT(**video_config)
            action_expert = ActionDiT(**action_config)
            mot = MoT(
                mixtures={"video": video_expert, "action": action_expert},
                mot_checkpoint_mixed_attn=False,
            )
        assign_state_dict_strict(mot, payload["mot"], label="FastWAM released MoT checkpoint")
        self._reset_rope(video_expert, action_expert)
        mot = mot.to(device=device, dtype=dtype).eval()

        with torch.device("meta"):
            proprio_encoder = nn.Linear(self.config.proprio_dim, 4096)
        proprio_state = payload.get("proprio_encoder")
        if not isinstance(proprio_state, Mapping):
            raise ValueError("FastWAM released checkpoint has no proprio_encoder state dictionary")
        assign_state_dict_strict(
            proprio_encoder,
            proprio_state,
            label="FastWAM released proprio encoder",
        )
        proprio_encoder = proprio_encoder.to(device=device, dtype=dtype).eval()

        asset_root = self._component_root(
            explicit=self.config.wan_assets_path,
            reference=self.config.wan_assets_ref,
            required_files=("Wan2.2_VAE.safetensors",),
            allow_patterns=("Wan2.2_VAE.safetensors",),
        )
        vae_state = load_state_dict(
            asset_root / "Wan2.2_VAE.safetensors",
            torch_dtype=None,
            device="cpu",
        )
        vae_state = wan_video_vae_state_dict_converter(vae_state)
        with torch.device("meta"):
            vae = WanVideoVAE38()
        assign_state_dict_strict(vae, vae_state, label="Wan2.2 VAE checkpoint")
        self._reset_vae_scale(vae)
        vae = vae.to(device=device, dtype=dtype).eval()

        policy = FastWAMPolicy(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            proprio_encoder=proprio_encoder,
            device=device,
            dtype=dtype,
            action_infer_shift=self.config.action_infer_shift,
            action_num_train_timesteps=self.config.action_num_train_timesteps,
            video_infer_shift=self.config.video_infer_shift,
            vae_tile_size=tuple(int(value) for value in self.config.vae_tile_size),
            vae_tile_stride=tuple(int(value) for value in self.config.vae_tile_stride),
        ).eval()
        self._policy = policy
        self._device = device
        self._dtype = dtype
        return policy

    def _load_text(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._text_encoder is not None:
            return self._tokenizer, self._text_encoder
        import torch

        from worldfoundry.core.model_loading.checkpoints.assignment import assign_state_dict_strict
        from worldfoundry.core.model_loading.checkpoints.file import load_state_dict

        from .wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

        self._load_policy()
        assets = self._component_root(
            explicit=self.config.wan_assets_path,
            reference=self.config.wan_assets_ref,
            required_files=("models_t5_umt5-xxl-enc-bf16.safetensors",),
            allow_patterns=("models_t5_umt5-xxl-enc-bf16.safetensors",),
        )
        text_state = load_state_dict(
            assets / "models_t5_umt5-xxl-enc-bf16.safetensors",
            device="cpu",
        )
        with torch.device("meta"):
            encoder = WanTextEncoder()
        assign_state_dict_strict(encoder, text_state, label="Wan UMT5 text encoder checkpoint")
        text_device = self.config.text_encoder_device or str(self._device)
        encoder_dtype = torch.float32 if torch.device(text_device).type == "cpu" else self._dtype
        encoder = encoder.to(device=text_device, dtype=encoder_dtype).eval()

        tokenizer_root = self._component_root(
            explicit=self.config.tokenizer_path,
            reference=self.config.tokenizer_ref,
            required_files=("google/umt5-xxl/tokenizer_config.json",),
            allow_patterns=("google/umt5-xxl/*",),
        )
        tokenizer = HuggingfaceTokenizer(
            str(tokenizer_root / "google/umt5-xxl"),
            seq_len=128,
            clean="whitespace",
            local_files_only=True,
            trust_remote_code=False,
        )
        self._tokenizer = tokenizer
        self._text_encoder = encoder
        return tokenizer, encoder

    def _encode_instruction(self, instruction: str) -> tuple[Any, Any]:
        import torch

        prompt = self.config.prompt_template.format(task=instruction)
        cached = self._context_cache.get(prompt)
        if cached is not None:
            return cached
        tokenizer, encoder = self._load_text()
        ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        text_device = next(encoder.parameters()).device
        ids = ids.to(text_device)
        mask = mask.to(text_device, dtype=torch.bool)
        with torch.inference_mode():
            context = encoder(ids, mask)
        # PyTorch 2.11 forbids mutating tensors created in inference mode once
        # that context has exited.  Clone preserves the encoder values while
        # yielding a normal tensor for deterministic padding zeroing/caching.
        context = context.clone()
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        for index, length in enumerate(sequence_lengths):
            context[index, length:] = 0
        # This is the released implementation's intentional all-valid mask.
        context_mask = torch.ones_like(mask)
        result = (
            context.to(device=self._device, dtype=self._dtype),
            context_mask.to(device=self._device, dtype=torch.bool),
        )
        self._context_cache[prompt] = result
        return result

    def _statistics_path(self) -> Path:
        if self.config.statistics_path:
            path = self._existing_path(self.config.statistics_path)
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"FastWAM statistics path does not exist: {self.config.statistics_path}"
                )
            return path
        root, _ = self._resolve_release()
        path = root / self.config.statistics_file
        if not path.is_file():
            raise FileNotFoundError(f"FastWAM statistics file is missing: {path}")
        return path

    def _load_statistics(self) -> Mapping[str, Any]:
        if self._statistics is None:
            self._statistics = json.loads(self._statistics_path().read_text(encoding="utf-8"))
        return self._statistics

    def _normalize_state(self, state: Any) -> Any:
        import numpy as np

        values = np.asarray(state, dtype=np.float32)
        if values.shape != (self.config.proprio_dim,):
            raise ValueError(
                f"FastWAM {self.config.variant} expects {self.config.proprio_dim} state values, "
                f"got {values.shape}"
            )
        stats = self._load_statistics()["state"]["default"]
        if self.config.normalization == "z-score":
            values = (values - np.asarray(stats["global_mean"], dtype=np.float32)) / (
                np.asarray(stats["global_std"], dtype=np.float32) + 1e-8
            )
        else:
            minimum = np.asarray(stats["global_min"], dtype=np.float32)
            maximum = np.asarray(stats["global_max"], dtype=np.float32)
            value_range = maximum - minimum
            ignore = value_range < 1e-4
            value_range[ignore] = 2.0
            scale = 2.0 / value_range
            offset = -1.0 - scale * minimum
            offset[ignore] = -minimum[ignore]
            values = values * scale + offset
        return np.clip(values, -5.0, 5.0)

    def _denormalize_action(self, actions: Any) -> Any:
        import numpy as np

        values = np.asarray(actions, dtype=np.float32)
        stats = self._load_statistics()["action"]["default"]
        if self.config.normalization == "z-score":
            values = values * (
                np.asarray(stats["global_std"], dtype=np.float32) + 1e-8
            ) + np.asarray(stats["global_mean"], dtype=np.float32)
        else:
            minimum = np.asarray(stats["global_min"], dtype=np.float32)
            maximum = np.asarray(stats["global_max"], dtype=np.float32)
            value_range = maximum - minimum
            ignore = value_range < 1e-4
            value_range[ignore] = 2.0
            scale = 2.0 / value_range
            offset = -1.0 - scale * minimum
            offset[ignore] = -minimum[ignore]
            values = (values - offset) / scale
        if self.config.variant.startswith("libero"):
            values[..., -1] = -(values[..., -1] * 2.0 - 1.0)
            if self.config.binarize_libero_gripper:
                values[..., -1] = np.sign(values[..., -1])
        return values

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
        if value is None and isinstance(observation.get("joint_action"), Mapping):
            value = observation["joint_action"].get("vector")
        nested = observation.get("observation")
        if value is None and isinstance(nested, Mapping):
            value = first_present(nested, "state", "proprio", "agent_pos", "joint_state")
        return value

    @staticmethod
    def _nested_camera(observation: Mapping[str, Any], name: str) -> Any:
        direct = observation.get(name)
        if direct is not None:
            return direct
        nested = observation.get("observation")
        if isinstance(nested, Mapping):
            camera = nested.get(name)
            if isinstance(camera, Mapping):
                return camera.get("rgb")
            return camera
        return None

    @staticmethod
    def _pil_image(image: Any) -> Any:
        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)):
            from worldfoundry.core.media.processing.image_utils import load_pil_image

            return load_pil_image(image, first_sequence_item=False)
        return Image.fromarray(np.asarray(image).astype(np.uint8))

    @staticmethod
    def _center_crop_resize(image: Any, width: int, height: int) -> Any:
        import numpy as np
        from PIL import Image

        source = FastWAMRuntime._pil_image(image).convert("RGB")
        source_width, source_height = source.size
        scale = max(width / source_width, height / source_height)
        resized = source.resize(
            (round(source_width * scale), round(source_height * scale)),
            resample=Image.Resampling.BILINEAR,
        )
        left = max((resized.width - width) // 2, 0)
        top = max((resized.height - height) // 2, 0)
        return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)

    @staticmethod
    def _resize(image: Any, width: int, height: int) -> Any:
        import numpy as np
        from PIL import Image

        source = FastWAMRuntime._pil_image(image)
        return np.asarray(
            source.convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    def _image_tensor(self, observation: Mapping[str, Any], image: Any) -> Any:
        import numpy as np
        import torch

        from worldfoundry.core.media.processing.image_utils import load_pil_image

        image_mapping = image if isinstance(image, Mapping) else {}
        image_sequence = (
            list(image)
            if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray))
            else []
        )
        combined = first_present(observation, "combined_image", "model_image")
        if combined is None:
            combined = first_present(image_mapping, "combined_image", "model_image")
        if self.config.variant.startswith("libero"):
            explicit_views = (
                first_present(
                    observation,
                    "wrist_image",
                    "robot0_eye_in_hand_image",
                    "image1",
                ),
                first_present(image_mapping, "wrist_image", "image1"),
            )
        else:
            explicit_views = (
                self._nested_camera(observation, "head_camera"),
                self._nested_camera(observation, "left_camera"),
                self._nested_camera(observation, "right_camera"),
                first_present(
                    observation,
                    "cam_high",
                    "head_cam",
                    "cam_left_wrist",
                    "left_cam",
                    "cam_right_wrist",
                    "right_cam",
                ),
                first_present(
                    image_mapping,
                    "head_camera",
                    "left_camera",
                    "right_camera",
                    "cam_high",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ),
            )
        # Workspace represents input_path as a one-item `images` sequence.  It
        # is a pre-composed model image only when no explicit auxiliary camera
        # views were supplied; otherwise it is the primary view and must be
        # combined with the wrist/head views below.
        if combined is None and len(image_sequence) == 1 and not any(
            value is not None for value in explicit_views
        ):
            combined = image_sequence[0]
        if (
            combined is None
            and image is not None
            and not isinstance(image, Mapping)
            and not image_sequence
        ):
            combined = image
        if combined is not None:
            source = load_pil_image(combined, first_sequence_item=False)
            array = self._resize(source, self.config.image_width, self.config.image_height)
        elif self.config.variant.startswith("libero"):
            primary = first_present(
                observation,
                "image",
                "full_image",
                "agentview_image",
                "image0",
            )
            wrist = first_present(
                observation,
                "wrist_image",
                "robot0_eye_in_hand_image",
                "image1",
            )
            if primary is None or wrist is None:
                images = observation.get("images")
                if isinstance(images, Mapping):
                    primary = primary if primary is not None else first_present(images, "image", "full_image", "image0")
                    wrist = wrist if wrist is not None else first_present(images, "wrist_image", "image1")
                elif isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
                    primary = primary if primary is not None else (images[0] if len(images) > 0 else None)
                    wrist = wrist if wrist is not None else (images[1] if len(images) > 1 else None)
            primary = primary if primary is not None else first_present(image_mapping, "image", "full_image", "image0")
            wrist = wrist if wrist is not None else first_present(image_mapping, "wrist_image", "image1")
            primary = primary if primary is not None else (image_sequence[0] if len(image_sequence) > 0 else None)
            wrist = wrist if wrist is not None else (image_sequence[1] if len(image_sequence) > 1 else None)
            if primary is None or wrist is None:
                raise ValueError("FastWAM LIBERO requires primary and wrist RGB views")
            primary = self._center_crop_resize(primary, 224, 224)
            wrist = self._center_crop_resize(wrist, 224, 224)
            array = np.concatenate([primary, wrist], axis=1)
        else:
            head = self._nested_camera(observation, "head_camera")
            left = self._nested_camera(observation, "left_camera")
            right = self._nested_camera(observation, "right_camera")
            head = head if head is not None else first_present(observation, "cam_high", "head_cam", "image0")
            left = left if left is not None else first_present(observation, "cam_left_wrist", "left_cam", "image1")
            right = right if right is not None else first_present(observation, "cam_right_wrist", "right_cam", "image2")
            images = observation.get("images")
            if isinstance(images, Mapping):
                head = head if head is not None else first_present(images, "head_camera", "cam_high", "head_cam", "image0")
                left = left if left is not None else first_present(images, "left_camera", "cam_left_wrist", "left_cam", "image1")
                right = right if right is not None else first_present(images, "right_camera", "cam_right_wrist", "right_cam", "image2")
            elif isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
                head = head if head is not None else (images[0] if len(images) > 0 else None)
                left = left if left is not None else (images[1] if len(images) > 1 else None)
                right = right if right is not None else (images[2] if len(images) > 2 else None)
            head = head if head is not None else first_present(image_mapping, "head_camera", "cam_high", "head_cam", "image0")
            left = left if left is not None else first_present(image_mapping, "left_camera", "cam_left_wrist", "left_cam", "image1")
            right = right if right is not None else first_present(image_mapping, "right_camera", "cam_right_wrist", "right_cam", "image2")
            head = head if head is not None else (image_sequence[0] if len(image_sequence) > 0 else None)
            left = left if left is not None else (image_sequence[1] if len(image_sequence) > 1 else None)
            right = right if right is not None else (image_sequence[2] if len(image_sequence) > 2 else None)
            if head is None or left is None or right is None:
                raise ValueError("FastWAM RoboTwin requires head, left-wrist, and right-wrist RGB views")
            head = self._resize(head, 320, 256)
            left = self._resize(left, 160, 128)
            right = self._resize(right, 160, 128)
            array = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        tensor = torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _context(
        self,
        instruction: str,
        observation: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        import torch

        context = first_present(
            observation,
            "context",
            "text_embeddings",
            "language_embeddings",
            "language_tokens",
        )
        mask = first_present(observation, "context_mask", "text_attention_mask", "language_attention_mask")
        if context is None:
            return self._encode_instruction(instruction)
        context = torch.as_tensor(context, device=self._device, dtype=self._dtype)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if mask is None:
            mask = torch.ones(context.shape[:2], device=self._device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=self._device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        return context, mask

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import torch

        policy = self._load_policy()
        state = self._state(observation)
        if state is None:
            raise ValueError(
                f"FastWAM {self.config.variant} requires a {self.config.proprio_dim}D state vector"
            )
        proprio = torch.as_tensor(
            self._normalize_state(state),
            dtype=torch.float32,
        ).unsqueeze(0)
        image_tensor = self._image_tensor(observation, image)
        context, context_mask = self._context(instruction, observation)
        infer = (
            policy.infer_action_idm
            if self.config.action_infer_mode == "idm"
            else policy.infer_action
        )
        action = infer(
            input_image=image_tensor,
            action_horizon=int(self.config.action_horizon),
            **({"num_video_frames": int(self.config.num_video_frames)} if self.config.action_infer_mode == "idm" else {}),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=int(self.config.num_inference_steps),
            sigma_shift=self.config.sigma_shift,
            seed=int(self.config.seed),
            rand_device=self.config.rand_device,
            tiled=self.config.tiled,
        )
        actions = self._denormalize_action(action.numpy())
        return completed_action_result(
            model_id="fastwam",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=str(self._checkpoint_file or self.config.checkpoint_location),
            device=str(self._device),
            runtime="worldfoundry.fastwam.in_tree_runtime",
            metadata={
                "variant": self.config.variant,
                "action_infer_mode": self.config.action_infer_mode,
                "num_video_frames": self.config.num_video_frames if self.config.action_infer_mode == "idm" else None,
                "action_shape": list(actions.shape),
                "action_horizon": int(self.config.action_horizon),
                "denoising_steps": int(self.config.num_inference_steps),
                "image_size": [self.config.image_height, self.config.image_width],
                "seed": int(self.config.seed),
                "dtype": str(self._dtype),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], FastWAMRuntime] = {}


def _required_option(options: Mapping[str, Any], name: str) -> Any:
    value = options.get(name)
    if value in (None, ""):
        raise ValueError(
            f"FastWAM runtime option {name!r} is required; load the model's data runtime config"
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
    variant_name = str(_required_option(options, "variant")).lower()
    variants = options.get("variants")
    variant_options: Mapping[str, Any] = {}
    if isinstance(variants, Mapping):
        selected = variants.get(variant_name)
        if not isinstance(selected, Mapping):
            raise ValueError(
                f"FastWAM variant {variant_name!r} has no entry in the data runtime config"
            )
        variant_options = selected
    effective = {**dict(variant_options), **options}
    checkpoint = (
        checkpoint_path
        or str(effective.get("checkpoint_ref") or "")
        or str(effective.get("repo_id") or "")
    )
    if not checkpoint:
        raise ValueError("FastWAM checkpoint_path or checkpoint_ref is required")
    # The Optional IDM release was trained/evaluated with shift 1.0, while the
    # base variants use 5.0. Keep checkpoint-specific values authoritative.
    sigma_value = variant_options.get("sigma_shift", effective.get("sigma_shift"))
    config = FastWAMRuntimeConfig(
        checkpoint_location=checkpoint,
        variant=variant_name,
        checkpoint_file=str(_required_option(effective, "checkpoint_file")),
        statistics_file=str(_required_option(effective, "statistics_file")),
        action_dim=int(_required_option(effective, "action_dim")),
        proprio_dim=int(_required_option(effective, "proprio_dim")),
        image_height=int(_required_option(effective, "image_height")),
        image_width=int(_required_option(effective, "image_width")),
        normalization=str(_required_option(effective, "normalization")),
        prompt_template=str(_required_option(effective, "prompt_template")),
        wan_assets_ref=str(_required_option(effective, "wan_assets_ref")),
        tokenizer_ref=str(_required_option(effective, "tokenizer_ref")),
        video_config=dict(_required_option(effective, "video_config")),
        action_config=dict(_required_option(effective, "action_config")),
        vae_mean=tuple(_required_option(effective, "vae_mean")),
        vae_std=tuple(_required_option(effective, "vae_std")),
        action_infer_shift=float(
            variant_options["action_infer_shift"]
            if "action_infer_shift" in variant_options
            else _required_option(effective, "action_infer_shift")
        ),
        action_num_train_timesteps=int(
            _required_option(effective, "action_num_train_timesteps")
        ),
        vae_tile_size=tuple(_required_option(effective, "vae_tile_size")),
        vae_tile_stride=tuple(_required_option(effective, "vae_tile_stride")),
        device=device,
        torch_dtype=str(effective.get("torch_dtype") or "auto"),
        cache_dir=str(effective["cache_dir"]) if effective.get("cache_dir") else None,
        local_files_only=option_bool(effective.get("local_files_only"), True),
        revision=str(effective["revision"]) if effective.get("revision") else None,
        statistics_path=str(effective["statistics_path"]) if effective.get("statistics_path") else None,
        wan_assets_path=str(effective["wan_assets_path"]) if effective.get("wan_assets_path") else None,
        tokenizer_path=str(effective["tokenizer_path"]) if effective.get("tokenizer_path") else None,
        text_encoder_device=(
            str(effective["text_encoder_device"])
            if effective.get("text_encoder_device") not in (None, "")
            else "cpu"
        ),
        action_horizon=option_int(effective.get("action_horizon"), 32),
        num_inference_steps=option_int(
            effective.get("num_inference_steps") or effective.get("denoising_steps"),
            10,
        ),
        sigma_shift=(
            None if sigma_value in (None, "", "none", "null") else option_float(sigma_value, 5.0)
        ),
        seed=option_int(effective.get("seed"), 42),
        rand_device=str(effective.get("rand_device") or "cpu"),
        tiled=option_bool(effective.get("tiled"), False),
        binarize_libero_gripper=option_bool(effective.get("binarize_libero_gripper"), False),
        action_infer_mode=str(effective.get("action_infer_mode") or "first_frame"),
        num_video_frames=option_int(effective.get("num_video_frames"), 5),
        video_infer_shift=option_float(effective.get("video_infer_shift"), 5.0),
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = FastWAMRuntime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["FastWAMRuntime", "FastWAMRuntimeConfig", "predict_action"]

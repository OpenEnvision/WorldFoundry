"""Resident, in-memory interaction for Matrix-Game 3.

The batch runner remains available for artifact generation.  This adapter keeps
the model, text embeddings, camera trajectory, latent memory, and streaming VAE
state alive between controls and advances exactly one native rollout window at
a time.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from bisect import bisect_right
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec

_KEY_TO_ACTION = {
    "w": "forward",
    "s": "back",
    "a": "left",
    "d": "right",
    "i": "camera_up",
    "k": "camera_down",
    "j": "camera_l",
    "l": "camera_r",
}
_ACTION_ALIASES = {
    "backward": "back",
    "camera_left": "camera_l",
    "camera_right": "camera_r",
}


class MatrixGame3RealtimeSession:
    """One native Matrix-Game 3 rollout with 57/40-frame cadence."""

    fps = 17
    first_output_frames = 57
    steady_output_frames = 40
    clip_frames = 56
    overlap_frames = 16
    latent_overlap_frames = 4
    memory_frames = 5

    def __init__(self, runtime: Any, operator: Any) -> None:
        self.runtime = runtime
        self.operator = operator
        self.core = runtime.ensure_resident_pipeline()
        self.distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if self.distributed else 0
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        if int(getattr(self.core, "sp_size", 1)) != self.world_size:
            raise RuntimeError(
                "Matrix-Game 3 resident core parallelism does not match torchrun: "
                f"core.sp_size={getattr(self.core, 'sp_size', None)}, "
                f"world_size={self.world_size}."
            )
        self.device = torch.device(self.core.device)
        self.dtype = torch.bfloat16
        self._helpers = self._load_helpers()
        self._clear_state()

    @staticmethod
    def _load_helpers() -> dict[str, Any]:
        cam_utils = importlib.import_module("utils.cam_utils")
        runtime_utils = importlib.import_module("utils.utils")
        return {
            "build_plucker_from_c2ws": runtime_utils.build_plucker_from_c2ws,
            "build_plucker_from_pose": runtime_utils.build_plucker_from_pose,
            "compute_all_poses_from_actions": runtime_utils.compute_all_poses_from_actions,
            "compute_relative_poses": cam_utils.compute_relative_poses,
            "get_extrinsics": runtime_utils.get_extrinsics,
            "get_intrinsics": cam_utils.get_intrinsics,
            "interpolate_poses": cam_utils._interpolate_camera_poses_handedness,
            "select_memory_idx_fov": cam_utils.select_memory_idx_fov,
        }

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
            controls=(
                "forward",
                "backward",
                "left",
                "right",
                "camera_up",
                "camera_down",
                "camera_l",
                "camera_r",
            ),
        )

    def next_output_frames(self) -> int:
        return self.first_output_frames if self._clip_index == 0 else self.steady_output_frames

    @staticmethod
    def _normalize_size(size: str | Sequence[int]) -> tuple[int, int]:
        if isinstance(size, str):
            parts = size.lower().replace("x", "*").split("*")
            if len(parts) != 2:
                raise ValueError(f"Matrix-Game 3 size must be HEIGHT*WIDTH, got {size!r}.")
            height, width = (int(part) for part in parts)
        else:
            if len(size) != 2:
                raise ValueError(f"Matrix-Game 3 size requires two values, got {size!r}.")
            height, width = (int(part) for part in size)
        if height <= 0 or width <= 0:
            raise ValueError("Matrix-Game 3 realtime dimensions must be positive.")
        return height, width

    @staticmethod
    def _normalize_actions(actions: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for action in actions:
            value = str(action).strip().lower()
            if value:
                normalized.append(_ACTION_ALIASES.get(value, value))
        return normalized

    def _actions_for_frames(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        *,
        num_frames: int,
    ) -> list[list[str]]:
        fallback = self._normalize_actions(interactions)
        segments: list[tuple[float, list[str]]] = []
        for segment in control_segments or ():
            duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
            actions = [
                _KEY_TO_ACTION[key]
                for key in (str(item).lower() for item in segment.get("keys", ()))
                if key in _KEY_TO_ACTION
            ]
            if duration > 0.0:
                segments.append((duration, actions))
        if not segments:
            return [fallback] * num_frames

        total_duration = sum(duration for duration, _ in segments)
        if total_duration <= 0.0:
            return [fallback] * num_frames
        result: list[list[str]] = []
        segment_index = 0
        segment_end = segments[0][0]
        for frame_index in range(num_frames):
            sample_time = (frame_index + 0.5) * total_duration / num_frames
            while segment_index + 1 < len(segments) and sample_time > segment_end:
                segment_index += 1
                segment_end += segments[segment_index][0]
            result.append(segments[segment_index][1])
        return result

    def _encode_frame_actions(
        self, frame_actions: Sequence[Sequence[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keyboard_rows: list[torch.Tensor] = []
        mouse_rows: list[torch.Tensor] = []
        for actions in frame_actions:
            keyboard = torch.zeros(self.operator.KEYBOARD_DIM, dtype=torch.float32)
            mouse = torch.zeros(2, dtype=torch.float32)
            for action in actions:
                action_keyboard, action_mouse = self.operator._encode_action(action)
                keyboard.copy_(torch.maximum(keyboard, action_keyboard))
                if torch.count_nonzero(action_mouse):
                    mouse.copy_(action_mouse)
            keyboard_rows.append(keyboard)
            mouse_rows.append(mouse)
        return torch.stack(keyboard_rows), torch.stack(mouse_rows)

    def _append_actions(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        *,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keyboard_cpu, mouse_cpu = self._encode_frame_actions(
            self._actions_for_frames(
                interactions,
                control_segments,
                num_frames=num_frames,
            )
        )
        poses, self._last_pose = self._helpers["compute_all_poses_from_actions"](
            keyboard_cpu,
            mouse_cpu,
            first_pose=self._last_pose,
            return_last_pose=True,
        )
        positions = poses[:, :3].tolist()
        rotations = np.concatenate(
            (np.zeros((poses.shape[0], 1), dtype=np.float32), poses[:, 3:5]),
            axis=1,
        ).tolist()
        extrinsics = self._helpers["get_extrinsics"](rotations, positions)

        keyboard = keyboard_cpu.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        mouse = mouse_cpu.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        self._keyboard_history = (
            keyboard
            if self._keyboard_history is None
            else torch.cat((self._keyboard_history, keyboard), dim=1)
        )
        self._mouse_history = (
            mouse if self._mouse_history is None else torch.cat((self._mouse_history, mouse), dim=1)
        )
        self._extrinsics = (
            extrinsics
            if self._extrinsics is None
            else torch.cat((self._extrinsics, extrinsics), dim=0)
        )
        return keyboard, mouse

    @torch.inference_mode()
    def configure(
        self,
        image: Image.Image,
        *,
        prompt: str = "",
        seed: int = 42,
        fps: int = 17,
        size: str | Sequence[int] = "704*1280",
        num_inference_steps: int | None = None,
        sample_shift: float | None = None,
        sample_guide_scale: float | None = None,
        use_base_model: bool | None = None,
    ) -> dict[str, Any]:
        """Encode immutable session conditions without generating a video."""

        if not isinstance(image, Image.Image):
            raise TypeError("Matrix-Game 3 realtime configuration requires a PIL image.")
        self.reset()
        started = time.perf_counter()
        self._height, self._width = self._normalize_size(size)
        self.fps = max(int(fps), 1)
        defaults = self.runtime.defaults
        self._num_inference_steps = max(
            int(
                num_inference_steps
                if num_inference_steps is not None
                else defaults.get("num_inference_steps", 3)
            ),
            1,
        )
        self._shift = float(
            sample_shift
            if sample_shift is not None
            else defaults.get("sample_shift", self.core.config.sample_shift)
        )
        self._guide_scale = float(
            sample_guide_scale
            if sample_guide_scale is not None
            else defaults.get("sample_guide_scale", self.core.config.sample_guide_scale)
        )
        self._use_base_model = bool(
            defaults.get("use_base_model", False)
            if use_base_model is None
            else use_base_model
        )
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))

        image_started = time.perf_counter()
        input_image = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).unsqueeze(0)
        input_image = input_image.float().permute(0, 3, 1, 2) / 127.5 - 1.0
        input_image = torch_F.interpolate(
            input_image,
            size=(self._height, self._width),
            mode="bicubic",
            align_corners=False,
        )
        current_image = input_image.transpose(0, 1).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        image_ms = (time.perf_counter() - image_started) * 1000.0

        prompt_started = time.perf_counter()
        normalized_prompt = str(prompt or "")
        self._prompt_context = self.core.text_encoder([normalized_prompt], device=self.device)
        self._prompt = normalized_prompt
        self._negative_context = (
            self.core.text_encoder([self.core.config.sample_neg_prompt], device=self.device)
            if self._use_base_model
            else None
        )
        prompt_ms = (time.perf_counter() - prompt_started) * 1000.0

        aspect_ratio = self._height / self._width
        max_area = self._height * self._width
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.core.vae_stride[1]
            // self.core.patch_size[1]
            * self.core.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.core.vae_stride[2]
            // self.core.patch_size[2]
            * self.core.patch_size[2]
        )
        self._target_h = lat_h * self.core.vae_stride[1]
        self._target_w = lat_w * self.core.vae_stride[2]
        self._base_intrinsics = self._helpers["get_intrinsics"](self._target_h, self._target_w)

        vae_started = time.perf_counter()
        if self.rank == 0:
            self._image_condition = self.core.vae.encode([current_image[0]])[0].unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ).contiguous()
        else:
            self._image_condition = torch.zeros(
                (1, 48, 1, lat_h, lat_w),
                device=self.device,
                dtype=self.dtype,
            )
        if self.distributed:
            torch.distributed.broadcast(self._image_condition, src=0)
        vae_encode_ms = (time.perf_counter() - vae_started) * 1000.0

        max_latent_frames = (self.first_output_frames - 1) // self.core.vae_stride[0] + 1
        max_total_frames = max_latent_frames + self.memory_frames
        self._max_seq_len = (
            max_total_frames
            * lat_h
            * lat_w
            // (self.core.patch_size[1] * self.core.patch_size[2])
        )
        if self.world_size > 1:
            self._max_seq_len = int(
                math.ceil(self._max_seq_len / self.world_size) * self.world_size
            )
        self._configured = True
        self._configure_metrics = {
            "image_ms": image_ms,
            "prompt_ms": prompt_ms,
            "vae_encode_ms": vae_encode_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self._configure_metrics),
        }

    @torch.inference_mode()
    def update_prompt(self, prompt: str) -> bool:
        """Replace future text conditioning without resetting rollout state."""

        if not self._configured:
            raise RuntimeError("Matrix-Game 3 realtime session is not configured.")
        normalized_prompt = str(prompt or "")
        if normalized_prompt == self._prompt:
            return False

        # Encode before committing so an encoder failure leaves the current
        # prompt and every causal world-state buffer usable.
        prompt_context = self.core.text_encoder([normalized_prompt], device=self.device)
        self._prompt_context = prompt_context
        self._prompt = normalized_prompt
        return True

    @staticmethod
    def _align_frame_to_block(frame_index: int) -> int:
        return (frame_index - 1) // 4 * 4 + 1 if frame_index > 0 else 1

    @staticmethod
    def _latent_index(frame_index: int) -> int:
        return (frame_index - 1) // 4 + 1

    def _build_camera_conditions(
        self,
        *,
        first_clip: bool,
        current_start: int,
        current_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Any, Any, Any, Any]:
        assert self._extrinsics is not None
        c2ws_chunk = self._extrinsics[current_start:current_end]
        source_indices = np.linspace(
            current_start,
            current_end - 1,
            self.first_output_frames if first_clip else self.clip_frames,
        )
        target_length = (
            (self.first_output_frames - 1) // 4 + 1 if first_clip else self.clip_frames // 4
        )
        target_indices = np.linspace(
            0 if first_clip else current_start + 3,
            current_end - 1,
            target_length,
        )
        plucker_without_memory = self._helpers["build_plucker_from_c2ws"](
            c2ws_chunk.to(device=self.device),
            source_indices,
            target_indices,
            framewise=True,
            base_K=self._base_intrinsics,
            target_h=self._target_h,
            target_w=self._target_w,
            lat_h=self._image_condition.shape[-2],
            lat_w=self._image_condition.shape[-1],
        )
        if first_clip:
            return plucker_without_memory, plucker_without_memory, None, None, None, None, None

        selected_base = [current_end - offset for offset in range(1, 34, 8)]
        if self.rank == 0:
            selected = self._helpers["select_memory_idx_fov"](
                self._extrinsics,
                current_start,
                selected_base,
                use_gpu=True,
            )
        else:
            selected = [0] * len(selected_base)
        if self.distributed:
            payload = [selected]
            torch.distributed.broadcast_object_list(payload, src=0)
            selected = payload[0]
        # The released rollout anchors one memory slot to the seed view.
        selected[-1] = 4
        memory_pluckers: list[torch.Tensor] = []
        latent_indices: list[int] = []
        for memory_index, reference_index in zip(selected, selected_base):
            latent_indices.append(self._latent_index(memory_index))
            aligned_index = self._align_frame_to_block(memory_index)
            memory_block = self._extrinsics[aligned_index : aligned_index + 4]
            memory_source = np.linspace(
                aligned_index,
                aligned_index + 3,
                memory_block.shape[0],
            )
            memory_target = np.asarray([aligned_index + 3], dtype=np.float32)
            memory_pose = self._helpers["interpolate_poses"](
                src_indices=memory_source,
                src_rot_mat=memory_block[:, :3, :3].cpu().numpy(),
                src_trans_vec=memory_block[:, :3, 3].cpu().numpy(),
                tgt_indices=memory_target,
            )
            reference_pose = self._extrinsics[reference_index : reference_index + 1]
            relative_pair = torch.cat((reference_pose, memory_pose), dim=0)
            relative_pose = self._helpers["compute_relative_poses"](
                relative_pair,
                framewise=False,
            )[1:2]
            memory_pluckers.append(
                self._helpers["build_plucker_from_pose"](
                    relative_pose.to(device=self.device),
                    base_K=self._base_intrinsics,
                    target_h=self._target_h,
                    target_w=self._target_w,
                    lat_h=self._image_condition.shape[-2],
                    lat_w=self._image_condition.shape[-1],
                )
            )
        latent_memory = self._gather_latent_memory(latent_indices)
        mouse_memory = torch.ones(
            (1, len(selected), 2),
            device=self.device,
            dtype=self.dtype,
        )
        keyboard_memory = -torch.ones(
            (1, len(selected), 6),
            device=self.device,
            dtype=self.dtype,
        )
        timestep_memory = latent_memory.new_zeros(
            (1, latent_memory.shape[2] * latent_memory.shape[3] * latent_memory.shape[4] // 4)
        )
        return (
            torch.cat((*memory_pluckers, plucker_without_memory), dim=2),
            plucker_without_memory,
            latent_memory,
            latent_indices,
            timestep_memory,
            keyboard_memory,
            mouse_memory,
        )

    def _gather_latent_memory(self, indices: Sequence[int]) -> torch.Tensor:
        """Gather sparse native memory frames without copying the full history."""

        frames: list[torch.Tensor] = []
        for index in indices:
            value = int(index)
            if value < 0 or value >= self._latent_frame_count:
                raise IndexError(
                    "Matrix-Game 3 latent memory index is outside the resident history: "
                    f"index={value}, frames={self._latent_frame_count}."
                )
            block_index = bisect_right(self._latent_offsets, value) - 1
            local_index = value - self._latent_offsets[block_index]
            frames.append(
                self._latent_history[block_index][:, :, local_index : local_index + 1]
            )
        return torch.cat(frames, dim=2)

    def _append_latents(self, latents: torch.Tensor) -> None:
        self._latent_offsets.append(self._latent_frame_count)
        self._latent_history.append(latents)
        self._latent_frame_count += int(latents.shape[2])

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Advance one native camera-aware rollout window and return RGB frames."""

        if not self._configured or self._generator is None or self._image_condition is None:
            raise RuntimeError("Matrix-Game 3 realtime session is not configured.")
        first_clip = self._clip_index == 0
        output_frames = self.next_output_frames()
        condition_started = time.perf_counter()
        self._append_actions(
            interactions,
            control_segments,
            num_frames=output_frames,
        )
        assert self._keyboard_history is not None
        assert self._mouse_history is not None
        current_end = self.first_output_frames + self._clip_index * (
            self.clip_frames - self.overlap_frames
        )
        current_start = 0 if first_clip else current_end - self.clip_frames
        (
            plucker,
            plucker_without_memory,
            latent_memory,
            latent_indices,
            timestep_memory,
            keyboard_memory,
            mouse_memory,
        ) = self._build_camera_conditions(
            first_clip=first_clip,
            current_start=current_start,
            current_end=current_end,
        )
        keyboard_condition = self._keyboard_history[:, current_start:current_end]
        mouse_condition = self._mouse_history[:, current_start:current_end]
        plucker = plucker.to(device=self.device, dtype=self.dtype)
        plucker_without_memory = plucker_without_memory.to(device=self.device, dtype=self.dtype)
        latent_start = self._latent_index(current_start)
        latent_end = self._latent_index(current_end)
        condition_ms = (time.perf_counter() - condition_started) * 1000.0

        scheduler_class = importlib.import_module(
            "worldfoundry.base_models.diffusion_model.schedulers"
        ).FlowUniPCMultistepScheduler
        scheduler = scheduler_class()
        scheduler.set_timesteps(
            self._num_inference_steps,
            device=self.device,
            shift=self._shift,
        )
        timesteps = scheduler.timesteps
        latents = torch.randn(
            (
                1,
                48,
                latent_end - latent_start,
                self._image_condition.shape[-2],
                self._image_condition.shape[-1],
            ),
            generator=self._generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = torch.cat(
            (self._image_condition, latents[:, :, self._image_condition.shape[2] :]),
            dim=2,
        )
        conditions_full = {
            "mouse_cond": mouse_condition,
            "keyboard_cond": keyboard_condition,
            "context": self._prompt_context,
            "plucker_emb": plucker,
            "x_memory": latent_memory,
            "timestep_memory": timestep_memory,
            "keyboard_cond_memory": keyboard_memory,
            "mouse_cond_memory": mouse_memory,
            "memory_latent_idx": latent_indices,
            "predict_latent_idx": (latent_start, latent_end),
            "fa_version": self.core.fa_version,
        }
        conditions_null = {
            "mouse_cond": torch.ones_like(mouse_condition),
            "keyboard_cond": -torch.ones_like(keyboard_condition),
            "context": self._negative_context,
            "plucker_emb": plucker_without_memory,
            "x_memory": None,
            "timestep_memory": None,
            "keyboard_cond_memory": None,
            "mouse_cond_memory": None,
            "memory_latent_idx": None,
            "predict_latent_idx": (latent_start, latent_end),
        }

        model_started = time.perf_counter()
        for timestep_value in timesteps:
            timestep = latents.new_full(
                (latents.shape[2], latents.shape[3] * latents.shape[4] // 4),
                timestep_value,
            )
            timestep[: self._image_condition.shape[2]].zero_()
            timestep = timestep.flatten().unsqueeze(0)
            model_kwargs = {
                "x": latents,
                "t": timestep,
                "seq_len": self._max_seq_len,
                **conditions_full,
            }
            if self._use_base_model:
                model_kwargs_null = {
                    "x": latents,
                    "t": timestep,
                    "seq_len": self._max_seq_len,
                    **conditions_null,
                }
                prediction_full = self.core.model(**model_kwargs)
                prediction_null = self.core.model(**model_kwargs_null)
                prediction = prediction_null + self._guide_scale * (
                    prediction_full - prediction_null
                )
            else:
                prediction = self.core.model(**model_kwargs)
            latents = scheduler.step(
                prediction,
                timestep_value,
                latents,
                return_dict=False,
            )[0]
            latents = torch.cat(
                (self._image_condition, latents[:, :, self._image_condition.shape[2] :]),
                dim=2,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter() - model_started) * 1000.0

        self._image_condition = latents[:, :, -self.latent_overlap_frames :].contiguous()
        denoised = (latents if first_clip else latents[:, :, -10:]).contiguous()
        self._append_latents(denoised)
        self._clip_index += 1
        frames: np.ndarray | None = None
        decode_ms = 0.0
        if self.rank == 0:
            decode_started = time.perf_counter()
            video, self._vae_cache = self.core.vae.stream_decode(
                denoised.to(dtype=self.core.vae.dtype),
                self._vae_cache,
                first_chunk=first_clip,
                segment_size=int(os.getenv("WAN_VAE_SEGMENT_SIZE", "4")),
                compile_decoder=(
                    bool(self.runtime.defaults.get("compile_vae", True)) and not first_clip
                ),
            )
            if video is None:
                raise RuntimeError(
                    "Matrix-Game 3 streaming VAE failed to decode the rollout block."
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            frames = (
                ((video.float() + 1.0) * 127.5)
                .clamp_(0, 255)
                .to(device="cpu", dtype=torch.uint8)
                .permute(0, 2, 3, 4, 1)
                .numpy()[0]
            )
            if frames.shape[0] != output_frames:
                raise RuntimeError(
                    "Matrix-Game 3 VAE cadence mismatch: "
                    f"expected {output_frames} RGB frames, decoded {frames.shape[0]}."
                )
        return {
            "video": np.ascontiguousarray(frames) if frames is not None else None,
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                "condition_ms": condition_ms,
                "model_ms": model_ms,
                "decode_ms": decode_ms,
            },
        }

    def _clear_state(self) -> None:
        self._configured = False
        self._clip_index = 0
        self._generator: torch.Generator | None = None
        self._height = 704
        self._width = 1280
        self._num_inference_steps = 3
        self._shift = 5.0
        self._guide_scale = 5.0
        self._use_base_model = False
        self._prompt = ""
        self._prompt_context: Any = None
        self._negative_context: Any = None
        self._image_condition: torch.Tensor | None = None
        self._base_intrinsics: torch.Tensor | None = None
        self._target_h = 0
        self._target_w = 0
        self._max_seq_len = 0
        self._last_pose = np.zeros(5, dtype=np.float32)
        self._keyboard_history: torch.Tensor | None = None
        self._mouse_history: torch.Tensor | None = None
        self._extrinsics: torch.Tensor | None = None
        self._latent_history: list[torch.Tensor] = []
        self._latent_offsets: list[int] = []
        self._latent_frame_count = 0
        self._vae_cache: list[Any] = [None] * 32
        self._configure_metrics: dict[str, float] = {}

    def reset(self) -> None:
        """Drop rollout state while retaining all model weights."""

        self._clear_state()


__all__ = ["MatrixGame3RealtimeSession"]

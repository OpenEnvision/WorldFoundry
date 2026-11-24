"""Resident incremental runtime for Matrix-Game 2.

The released batch path already contains the right causal algorithm, but used
to discard transformer, action-attention, cross-attention, and VAE decoder
caches at every Studio interaction.  This adapter owns one rollout and feeds
the core exactly one three-latent-frame block per action.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import (
    DEFAULT_REALTIME_KEYS,
    RealtimeAxis,
    RealtimeKey,
    RealtimeSpec,
)
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.core.utils.tensors.torch import set_seed_everywhere as set_seed
from worldfoundry.operators.matrix_game_2_operator import encode_actions

_ACTION_ALIASES = {
    "backward": "back",
    "camera_left": "camera_l",
    "camera_right": "camera_r",
}
# Official per-step camera magnitude; mouse conditions stay within it per frame.
_CAMERA_LIMIT = 0.1
# Initial calibration: about 40 px of mouse motion in one frame saturates the
# official camera step; full stick deflection equals one camera key.
_MOUSE_CAMERA_SCALE = 0.0025
_STICK_DEADZONE = 0.15
# Mouse condition column of each camera axis: 0 is pitch (+ looks up), 1 is yaw (+ turns right).
_AXIS_MOUSE_COLUMN = {
    "camera_pitch": 0,
    "camera_pitch_stick": 0,
    "camera_yaw": 1,
    "camera_yaw_stick": 1,
}
_CAMERA_YAW_STICK = RealtimeAxis(
    "camera_yaw_stick", "gamepad_rx", mode="level", scale=_CAMERA_LIMIT, deadzone=_STICK_DEADZONE
)


@dataclass(frozen=True, slots=True)
class _ModeControls:
    """Official realtime keymap and camera axes of one Matrix-Game 2 mode.

    ``idle_action`` encodes frames without an active key.
    """

    keys: tuple[RealtimeKey, ...]
    axes: tuple[RealtimeAxis, ...] = ()
    idle_action: str | None = None


# Keymaps follow ``pipeline/causal_inference.py::get_current_action``.
_MODE_CONTROLS = {
    "universal": _ModeControls(
        keys=DEFAULT_REALTIME_KEYS,
        axes=(
            # Moving the mouse down looks down.
            RealtimeAxis("camera_pitch", "mouse_y", scale=-_MOUSE_CAMERA_SCALE, limit=_CAMERA_LIMIT),
            RealtimeAxis("camera_yaw", "mouse_x", scale=_MOUSE_CAMERA_SCALE, limit=_CAMERA_LIMIT),
            RealtimeAxis(
                "camera_pitch_stick", "gamepad_ry", mode="level", scale=-_CAMERA_LIMIT, deadzone=_STICK_DEADZONE
            ),
            _CAMERA_YAW_STICK,
        ),
    ),
    "gta_drive": _ModeControls(
        keys=(
            RealtimeKey("w", "forward", "forward_back"),
            RealtimeKey("s", "back", "forward_back"),
            RealtimeKey("a", "camera_l", "steer"),
            RealtimeKey("d", "camera_r", "steer"),
        ),
        axes=(_CAMERA_YAW_STICK,),
    ),
    "templerun": _ModeControls(
        keys=(
            RealtimeKey("w", "jump", "action"),
            RealtimeKey("s", "slide", "action"),
            RealtimeKey("a", "leftside", "action"),
            RealtimeKey("d", "rightside", "action"),
            RealtimeKey("z", "turnleft", "action"),
            RealtimeKey("c", "turnright", "action"),
        ),
        idle_action="nomove",
    ),
}


class MatrixGame2RealtimeSession:
    """One resident Matrix-Game 2 rollout with native 9/12-frame cadence."""

    fps = 12
    latent_frames_per_block = 3
    first_output_frames = 9
    steady_output_frames = 12
    height = 352
    width = 640

    def __init__(self, runtime: Any, operator: Any) -> None:
        self.runtime = runtime
        self.core = runtime.pipeline
        self.operator = operator
        self.device = torch.device(runtime.device)
        self.dtype = runtime.weight_dtype
        block = int(getattr(self.core, "num_frame_per_block", 0) or 0)
        if block != self.latent_frames_per_block:
            raise ValueError(
                "Matrix-Game 2 realtime requires the released three-latent-frame "
                f"checkpoint configuration; got num_frame_per_block={block}."
            )
        self.condition_prefetch_blocks = max(
            int(os.getenv("WORLDFOUNDRY_MATRIX_REALTIME_CONDITION_BLOCKS", "5") or "5"),
            2,
        )
        # The real encoder has a long causal receptive field: the old five
        # blocks still contained the seed-image transient in the reused tail.
        # Lightweight runtime fixtures without an encoder retain their own
        # prefetch contract; only the real graph establishes stationarity.
        vae = getattr(runtime.vae, "vae", None)
        encoder = getattr(getattr(vae, "model", None), "encoder", None)
        self.condition_first_stable_latent: int | None = None
        if encoder is not None:
            from .conditioning import first_stable_encoder_latent, minimum_condition_prefetch_blocks

            self.condition_first_stable_latent = first_stable_encoder_latent(encoder)
            self.condition_prefetch_blocks = max(
                self.condition_prefetch_blocks,
                minimum_condition_prefetch_blocks(encoder, self.latent_frames_per_block),
            )
        self._visual_context: torch.Tensor | None = None
        self._condition_concat: torch.Tensor | None = None
        self._keyboard_history: torch.Tensor | None = None
        self._mouse_history: torch.Tensor | None = None
        self._noise_generator: torch.Generator | None = None
        self.configured = False
        self.configure_metrics: dict[str, float] = {}

    def realtime_spec(self) -> RealtimeSpec:
        """Native cadence plus the official keymap and camera axes of the loaded mode."""

        controls = _MODE_CONTROLS[self.runtime.mode]
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
            controls=tuple(binding.action for binding in controls.keys),
            keys=controls.keys,
            axes=controls.axes,
        )

    def next_output_frames(self) -> int:
        state = getattr(self.core, "_session", None)
        if state is None or int(state.current_start_frame) == 0:
            return self.first_output_frames
        return self.steady_output_frames

    @torch.inference_mode()
    def configure(self, image: Image.Image, *, seed: int = 42) -> dict[str, Any]:
        """Encode the seed image and allocate causal caches without generating."""

        if not isinstance(image, Image.Image):
            raise TypeError("Matrix-Game 2 realtime configuration requires a PIL image.")
        self.reset()
        set_seed(int(seed))
        self._noise_generator = torch.Generator(device=self.device).manual_seed(int(seed))

        started = time.perf_counter()
        perception_started = started
        perception = self.operator.process_perception(
            image.convert("RGB"),
            self.latent_frames_per_block * self.condition_prefetch_blocks,
            self.height,
            self.width,
            device=str(self.device),
            weight_dtype=self.dtype,
        )
        perception_ms = (time.perf_counter() - perception_started) * 1000.0

        vae_started = time.perf_counter()
        image_latent = self.runtime.vae.encode(
            perception["img_cond"],
            device=str(self.device),
            **perception["tiler_kwargs"],
        ).to(device=self.device, dtype=self.dtype)
        mask = torch.ones_like(image_latent)
        mask[:, :, 1:] = 0
        self._condition_concat = torch.cat((mask[:, :4], image_latent), dim=1).contiguous()
        vae_ms = (time.perf_counter() - vae_started) * 1000.0

        clip_started = time.perf_counter()
        self._visual_context = self.runtime.vae.clip.encode_video(perception["image"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        clip_ms = (time.perf_counter() - clip_started) * 1000.0

        base_condition = {
            "cond_concat": self._condition_concat[:, :, : self.latent_frames_per_block],
            "visual_context": self._visual_context,
            "_cond_concat_start_frame": 0,
        }
        cache_started = time.perf_counter()
        self.core.start_session(base_condition, mode=self.runtime.mode)
        cache_ms = (time.perf_counter() - cache_started) * 1000.0

        decoder_warmup_started = time.perf_counter()
        decoder = getattr(self.core, "vae_decoder", None)
        warmup_decoder = os.getenv("WORLDFOUNDRY_MATRIX_REALTIME_WARMUP_DECODER", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if warmup_decoder and callable(decoder):
            # First and steady blocks take different causal VAE paths (9 then
            # 12 RGB frames). Exercise both with an isolated cache so no
            # synthetic content can leak into the actual rollout.
            warmup_latent = torch.zeros(
                (
                    1,
                    self.latent_frames_per_block,
                    16,
                    image_latent.shape[-2],
                    image_latent.shape[-1],
                ),
                device=self.device,
                dtype=torch.float16,
            )
            warmup_cache: list[torch.Tensor | None] = [None] * 32
            warmup_video, warmup_cache = decoder(warmup_latent, *warmup_cache)
            warmup_video, warmup_cache = decoder(warmup_latent, *warmup_cache)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            del warmup_latent, warmup_video, warmup_cache
        decoder_warmup_ms = (time.perf_counter() - decoder_warmup_started) * 1000.0
        self.configured = True
        self.configure_metrics = {
            "perception_ms": perception_ms,
            "vae_encode_ms": vae_ms,
            "clip_encode_ms": clip_ms,
            "cache_init_ms": cache_ms,
            "decoder_warmup_ms": decoder_warmup_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.configure_metrics),
        }

    def _condition_block(self, start_frame: int) -> torch.Tensor:
        """Return a prefetched block or reuse the encoder's proven stationary tail."""

        assert self._condition_concat is not None
        block_end = start_frame + self.latent_frames_per_block
        if block_end <= self._condition_concat.shape[2]:
            return self._condition_concat[:, :, start_frame:block_end]
        return self._condition_concat[:, :, -self.latent_frames_per_block :]

    @staticmethod
    def _normalize_actions(actions: Sequence[str]) -> list[str]:
        return [
            _ACTION_ALIASES.get(str(action).strip().lower(), str(action).strip().lower())
            for action in actions
            if str(action).strip()
        ]

    def _actions_for_frames(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        *,
        num_frames: int,
    ) -> list[list[str]]:
        """Sample browser control segments onto the model's RGB action timeline."""

        fallback = self._normalize_actions(interactions)
        key_to_action = {
            binding.key: self._normalize_actions([binding.action])[0]
            for binding in _MODE_CONTROLS[self.runtime.mode].keys
        }
        segments: list[tuple[float, list[str]]] = []
        for segment in control_segments or ():
            duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
            actions = [
                key_to_action[key]
                for key in (str(item).lower() for item in segment.get("keys", ()))
                if key in key_to_action
            ]
            if duration > 0.0:
                segments.append((duration, actions))
        if not segments:
            return [fallback] * num_frames

        total = sum(duration for duration, _ in segments)
        if total <= 0.0:
            return [fallback] * num_frames
        result: list[list[str]] = []
        segment_index = 0
        segment_end = segments[0][0]
        for frame_index in range(num_frames):
            sample_time = (frame_index + 0.5) * total / num_frames
            while segment_index + 1 < len(segments) and sample_time > segment_end:
                segment_index += 1
                segment_end += segments[segment_index][0]
            result.append(segments[segment_index][1])
        return result

    def _camera_offsets(
        self,
        realtime_axes: Mapping[str, Sequence[float]] | None,
        *,
        num_frames: int,
    ) -> torch.Tensor | None:
        """Sum the mode's declared camera axes into per-frame ``(pitch, yaw)`` offsets."""

        if not realtime_axes:
            return None
        for name, values in realtime_axes.items():
            if len(values) != num_frames:
                raise ValueError(
                    f"Matrix-Game 2 realtime axis {name!r} has {len(values)} values; "
                    f"the chunk has {num_frames} output frames."
                )
        offsets = torch.zeros((num_frames, 2), dtype=torch.float32)
        for axis in _MODE_CONTROLS[self.runtime.mode].axes:
            values = realtime_axes.get(axis.name)
            if values is not None:
                offsets[:, _AXIS_MOUSE_COLUMN[axis.name]] += torch.tensor(values, dtype=torch.float32)
        return offsets

    def _append_actions(
        self,
        frame_actions: Sequence[Sequence[str]],
        camera_offsets: torch.Tensor | None = None,
    ) -> None:
        """Encode one chunk of per-frame actions and append them to the condition history.

        ``camera_offsets`` adds axis values to the key-derived mouse condition;
        the sum is clamped to the official camera step per component.
        """

        idle_action = _MODE_CONTROLS[self.runtime.mode].idle_action
        keyboard_rows: list[torch.Tensor] = []
        mouse_rows: list[torch.Tensor] = []
        for actions in frame_actions:
            if not actions and idle_action is not None:
                actions = [idle_action]
            keyboard, mouse = encode_actions(list(actions), self.runtime.mode)
            keyboard_rows.append(keyboard)
            if mouse is not None:
                mouse_rows.append(mouse)
        keyboard_chunk = torch.stack(keyboard_rows).to(device=self.device, dtype=self.dtype)
        self._keyboard_history = (
            keyboard_chunk
            if self._keyboard_history is None
            else torch.cat((self._keyboard_history, keyboard_chunk), dim=0)
        )
        if mouse_rows:
            mouse_chunk = torch.stack(mouse_rows)
            if camera_offsets is not None:
                mouse_chunk = (mouse_chunk + camera_offsets).clamp_(-_CAMERA_LIMIT, _CAMERA_LIMIT)
            mouse_chunk = mouse_chunk.to(device=self.device, dtype=self.dtype)
            self._mouse_history = (
                mouse_chunk if self._mouse_history is None else torch.cat((self._mouse_history, mouse_chunk), dim=0)
            )

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        realtime_axes: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, Any]:
        """Advance the rollout by one native latent block and return RGB frames.

        ``realtime_axes`` carries one value per output frame for each declared axis.
        """

        if not self.configured or self._noise_generator is None:
            raise RuntimeError("Matrix-Game 2 realtime session is not configured.")
        state = getattr(self.core, "_session", None)
        if state is None:
            raise RuntimeError("Matrix-Game 2 causal session was unexpectedly released.")
        start_frame = int(state.current_start_frame)
        output_frames = self.next_output_frames()

        condition_started = time.perf_counter()
        self._append_actions(
            self._actions_for_frames(
                interactions,
                control_segments,
                num_frames=output_frames,
            ),
            self._camera_offsets(realtime_axes, num_frames=output_frames),
        )
        assert self._visual_context is not None
        assert self._condition_concat is not None
        assert self._keyboard_history is not None
        conditional_dict: dict[str, Any] = {
            "cond_concat": self._condition_block(start_frame),
            "_cond_concat_start_frame": start_frame,
            "visual_context": self._visual_context,
            "keyboard_cond": self._keyboard_history.unsqueeze(0),
        }
        if self.runtime.mode != "templerun":
            assert self._mouse_history is not None
            conditional_dict["mouse_cond"] = self._mouse_history.unsqueeze(0)
        condition_ms = (time.perf_counter() - condition_started) * 1000.0

        noise = torch.randn(
            (
                1,
                16,
                self.latent_frames_per_block,
                self._condition_concat.shape[-2],
                self._condition_concat.shape[-1],
            ),
            generator=self._noise_generator,
            device=self.device,
            dtype=self.dtype,
        )
        video = self.core.step_session(noise, conditional_dict, mode=self.runtime.mode)
        block = getattr(self.core, "_last_block", None)
        model_ms = float(getattr(block, "model_ms", 0.0) or 0.0)
        decode_ms = float(getattr(block, "decode_ms", 0.0) or 0.0)
        frames = (
            ((video.float() + 1.0) * 127.5)
            .clamp_(0, 255)
            .to(device="cpu", dtype=torch.uint8)
            .permute(0, 1, 3, 4, 2)
            .numpy()[0]
        )
        frames = np.ascontiguousarray(frames)
        return {
            "video": frames,
            "video_chunk": VideoChunk(frames=frames, layout="thwc", value_range="0,255"),
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                "condition_ms": condition_ms,
                "model_ms": model_ms,
                "decode_ms": decode_ms,
            },
        }

    @torch.inference_mode()
    def stream_overlapped(
        self,
        actions_sequence: Sequence[Sequence[str]],
    ):
        """Yield uint8 video chunks while overlapping chunk ``k`` DiT with chunk ``k-1`` VAE decode + D2H.

        Uses two dedicated non-blocking CUDA streams (compute vs decode) to avoid
        implicit CUDA Stream-0 synchronization barriers, matching the pipelined
        streaming pattern of WorldFoundry's native forcing pipelines.
        """
        if not self.configured or self._noise_generator is None:
            raise RuntimeError("Matrix-Game 2 realtime session is not configured.")
        state = getattr(self.core, "_session", None)
        if state is None:
            raise RuntimeError("Matrix-Game 2 causal session was unexpectedly released.")
        spec_payload = self.realtime_spec().to_payload()
        is_cuda = self.device.type == "cuda"
        entry_stream = torch.cuda.current_stream(self.device) if is_cuda else None
        compute_stream = torch.cuda.Stream(device=self.device) if is_cuda else None
        decode_stream = torch.cuda.Stream(device=self.device) if is_cuda else None
        if is_cuda and entry_stream is not None and compute_stream is not None and decode_stream is not None:
            init_ev = torch.cuda.Event()
            init_ev.record(entry_stream)
            compute_stream.wait_event(init_ev)
            decode_stream.wait_event(init_ev)

        pending_cpu_uint8: torch.Tensor | None = None
        pending_event: torch.cuda.Event | None = None
        pending_meta: dict[str, Any] = {}

        def _flush_pending():
            nonlocal pending_cpu_uint8, pending_event
            if pending_cpu_uint8 is None or pending_event is None:
                return None
            pending_event.synchronize()
            frames_np = np.ascontiguousarray(pending_cpu_uint8.numpy())
            pending_meta["wall_ms"] = (time.perf_counter() - pending_meta["chunk_t0"]) * 1000.0
            payload = {
                "video": frames_np,
                "video_chunk": VideoChunk(frames=frames_np, layout="thwc", value_range="0,255"),
                "realtime_spec": spec_payload,
                "realtime_metrics": dict(pending_meta),
            }
            pending_cpu_uint8 = None
            pending_event = None
            return payload

        try:
            for chunk_idx, interactions in enumerate(actions_sequence):
                if not (is_cuda and compute_stream is not None and decode_stream is not None):
                    yield self.generate(interactions=interactions)
                    continue

                t0 = time.perf_counter()
                start_frame = int(state.current_start_frame)
                output_frames = self.next_output_frames()
                with torch.cuda.stream(compute_stream):
                    self._append_actions(
                        self._actions_for_frames(interactions, None, num_frames=output_frames),
                        None,
                    )
                    assert self._visual_context is not None
                    assert self._condition_concat is not None
                    assert self._keyboard_history is not None
                    conditional_dict: dict[str, Any] = {
                        "cond_concat": self._condition_block(start_frame),
                        "_cond_concat_start_frame": start_frame,
                        "visual_context": self._visual_context,
                        "keyboard_cond": self._keyboard_history.unsqueeze(0),
                    }
                    if self.runtime.mode != "templerun":
                        assert self._mouse_history is not None
                        conditional_dict["mouse_cond"] = self._mouse_history.unsqueeze(0)
                    state.conditional_dict = conditional_dict
                    state.mode = self.runtime.mode

                    noise = torch.randn(
                        (
                            1,
                            16,
                            self.latent_frames_per_block,
                            self._condition_concat.shape[-2],
                            self._condition_concat.shape[-1],
                        ),
                        generator=self._noise_generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    denoised_pred, _, _ = self.core.denoise_next_block(state, noise)
                    latent_for_decode = denoised_pred.transpose(1, 2).half().contiguous()
                    latent_ready = torch.cuda.Event()
                    latent_ready.record(compute_stream)
                    latent_for_decode.record_stream(decode_stream)

                prev_payload = _flush_pending()
                if prev_payload is not None:
                    yield prev_payload

                with torch.cuda.stream(decode_stream):
                    decode_stream.wait_event(latent_ready)
                    video, next_vae_cache = self.core.vae_decoder(latent_for_decode, *state.vae_cache)
                    state.vae_cache = next_vae_cache
                    uint8_gpu = (
                        ((video.float() + 1.0) * 127.5)
                        .clamp_(0, 255)
                        .to(dtype=torch.uint8)
                        .permute(0, 1, 3, 4, 2)[0]
                        .contiguous()
                    )
                    cpu_buf = torch.empty(uint8_gpu.shape, dtype=torch.uint8, pin_memory=True)
                    cpu_buf.copy_(uint8_gpu, non_blocking=True)
                    done_ev = torch.cuda.Event()
                    done_ev.record(decode_stream)
                pending_cpu_uint8 = cpu_buf
                pending_event = done_ev
                pending_meta = {"chunk_t0": t0, "chunk_index": chunk_idx + 1, "overlapped": True}

            final_payload = _flush_pending()
            if final_payload is not None:
                yield final_payload
        finally:
            if is_cuda and entry_stream is not None and compute_stream is not None and decode_stream is not None:
                entry_stream.wait_stream(compute_stream)
                entry_stream.wait_stream(decode_stream)

    def reset(self) -> None:
        """Release rollout state while keeping all model weights resident."""

        self.core.reset_session()
        self._visual_context = None
        self._condition_concat = None
        self._keyboard_history = None
        self._mouse_history = None
        self._noise_generator = None
        self.configured = False
        self.configure_metrics = {}


__all__ = ["MatrixGame2RealtimeSession"]

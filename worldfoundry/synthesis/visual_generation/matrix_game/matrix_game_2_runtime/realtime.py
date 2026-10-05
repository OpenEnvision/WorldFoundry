"""Resident incremental runtime for Matrix-Game 2.

The released batch path already contains the right causal algorithm, but used
to discard transformer, action-attention, cross-attention, and VAE decoder
caches at every Studio interaction.  This adapter owns one rollout and feeds
the core exactly one three-latent-frame block per action.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.core.utils.tensors.torch import set_seed_everywhere as set_seed
from worldfoundry.operators.matrix_game_2_operator import encode_actions

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
        warmup_decoder = os.getenv(
            "WORLDFOUNDRY_MATRIX_REALTIME_WARMUP_DECODER", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
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
        return self._condition_concat[:, :, -self.latent_frames_per_block:]

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

    def _append_actions(self, frame_actions: Sequence[Sequence[str]]) -> None:
        keyboard_rows: list[torch.Tensor] = []
        mouse_rows: list[torch.Tensor] = []
        for actions in frame_actions:
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
            mouse_chunk = torch.stack(mouse_rows).to(device=self.device, dtype=self.dtype)
            self._mouse_history = (
                mouse_chunk
                if self._mouse_history is None
                else torch.cat((self._mouse_history, mouse_chunk), dim=0)
            )

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Advance the rollout by one native latent block and return RGB frames."""

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
            )
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
        return {
            "video": np.ascontiguousarray(frames),
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                "condition_ms": condition_ms,
                "model_ms": model_ms,
                "decode_ms": decode_ms,
            },
        }

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

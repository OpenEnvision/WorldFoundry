"""Framework-owned segment autoregression with caller-owned text/video KV history."""

from __future__ import annotations

import math
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from worldfoundry.core.attention.cache.kv_quantization import KVQuantConfig, KVQuantDType, QuantizedKVStore

from ..contracts import DenoiserInput, DiffusionOutput, DiffusionRequest, SchedulerStep
from .base import NativeDiffusionRunner


@dataclass
class SegmentGenerationState:
    """One rollout; shared weights never retain its caches, RNG, or decoder history."""

    owner: object
    geometry: tuple[int, int]
    generator: torch.Generator
    positive: list[dict[str, Any]]
    negative: list[dict[str, Any]]
    frame_offset: int = 0
    decoder: Any = None
    failed: bool = False
    segment_tokens: list[int] = field(default_factory=list)

    def invalidate(self) -> None:
        self.positive.clear()
        self.negative.clear()
        self.decoder = None
        self.failed = True


class SegmentedAutoregressiveRunner(NativeDiffusionRunner):
    """Sample complete segments, then commit noisy-context KV once per segment."""

    def new_session(self, request: DiffusionRequest) -> SegmentGenerationState:
        component = cast(Any, self.components.denoiser)
        config = KVQuantConfig(
            dtype=cast(KVQuantDType, str(request.inputs.get("kv_cache_quantization", "none"))),
            group_size=component.cache_head_dim,
            symmetric=True,
        )

        def cache() -> list[dict[str, Any]]:
            return [{"store": QuantizedKVStore(config, seq_dim=1)} for _ in range(component.num_cache_layers)]

        return SegmentGenerationState(
            owner=self,
            geometry=(request.height, request.width),
            generator=self._generator(request.sampling.seed),
            positive=cache(),
            negative=cache(),
        )

    @contextmanager
    def _phase(self, component: Any) -> Generator[None]:
        """Use the canonical quantized modules; only placement changes per phase."""
        offload = bool(getattr(component, "low_memory", False))
        if offload:
            component.model.to(self.device)
        try:
            yield
        finally:
            if offload:
                component.model.to("cpu")

    @staticmethod
    def _lengths(request: DiffusionRequest, count: int) -> list[int]:
        frames = math.ceil(request.num_frames / 4)
        lengths = request.inputs.get("segment_latent_lengths")
        if lengths is None:
            lengths = [frames // count + (index < frames % count) for index in range(count)]
        if (
            not isinstance(lengths, (list, tuple))
            or len(lengths) != count
            or any(type(n) is not int or n <= 0 for n in lengths)
            or sum(lengths) != frames
        ):
            raise ValueError(
                "segment_latent_lengths must contain one positive length per prompt and sum to ceil(num_frames/4)"
            )
        return list(lengths)

    @torch.inference_mode()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        return self.continue_session(request, self.new_session(request), final=True)

    @torch.inference_mode()
    def continue_session(
        self,
        request: DiffusionRequest,
        session: SegmentGenerationState,
        *,
        final: bool = False,
    ) -> DiffusionOutput:
        if session.owner is not self or session.geometry != (request.height, request.width):
            raise ValueError("segment session belongs to a different runner or output geometry")
        if session.failed:
            raise RuntimeError("failed segment history must be reset before a new rollout")
        if request.batch_size != 1:
            raise ValueError("segment autoregression generates one world at a time")
        if self.extensions:
            raise ValueError("segment autoregression does not support standard-loop extensions")
        request_options: dict[str, Any] = dict(request.inputs)
        sigma = float(request_options.get("cache_noise_sigma", 0.206897))
        if not 0 <= sigma <= 1:
            raise ValueError("cache_noise_sigma must lie in [0, 1]")
        try:
            with self._phase(self.components.conditioner):
                conditioning = self.components.conditioner.encode(request, device=self.device, dtype=self.dtype)
            segments = cast(list[dict[str, Any]], conditioning.positive["segments"])
            lengths = self._lengths(request, len(segments))
            noise = self.components.latent_initializer.initialize(
                request,
                generator=session.generator,
                device=self.device,
                dtype=self.dtype,
            )
            chunks = list(cast(torch.Tensor, noise).split(lengths, dim=2))
            camera = request.inputs.get("camera_extrinsics")
            if camera is None:
                cameras = [None] * len(lengths)
            else:
                camera = torch.as_tensor(camera, device=self.device, dtype=torch.float32)
                if camera.ndim == 3:
                    camera = camera.unsqueeze(0)
                if camera.shape != (1, request.num_frames, 4, 4):
                    raise ValueError("camera_extrinsics must have shape (1, num_frames, 4, 4)")
                padding = sum(lengths) * 4 - request.num_frames
                camera = torch.cat((camera, camera[:, -1:].expand(-1, padding, -1, -1)), dim=1)
                cameras = list(camera.split([length * 4 for length in lengths], dim=1))
            schedule = self.components.scheduler.schedule(request.sampling, device=self.device, dtype=self.dtype)
            from ..models.encoders.hy_nomas import action_windows

            generated = []
            for index, (latents, text, camera) in enumerate(zip(chunks, segments, cameras, strict=False)):
                inputs = dict(text)
                inputs.update(frame_offset=session.frame_offset, camera_extrinsics=camera)
                prompts = cast(list[str], request_options.get("segments") or list(request.prompts))
                windows = action_windows(prompts[index], lengths[index], float(request_options.get("fps", 24)))
                inputs["action_frame_windows"] = windows
                if windows is None:
                    inputs["action_text_offsets"] = None
                branches = [(inputs, session.positive)]
                if request.sampling.guidance_scale != 1:
                    negative = dict(conditioning.negative)
                    negative.update(frame_offset=session.frame_offset, camera_extrinsics=camera)
                    branches.append((negative, session.negative))
                with self._phase(self.components.denoiser):
                    for step in schedule:
                        outputs = [
                            self._predict_segment(latents, step, values, cache, total_steps=len(schedule))
                            for values, cache in branches
                        ]
                        velocity = outputs[0]
                        if len(outputs) == 2:
                            velocity = outputs[1] + request.sampling.guidance_scale * (velocity - outputs[1])
                        latents = self.components.scheduler.step(
                            velocity.float(),
                            step,
                            latents.float(),
                            generator=session.generator,
                        ).to(dtype=self.dtype)
                    cache_latents = latents
                    if sigma:
                        cache_noise = torch.randn(
                            latents.shape, device=latents.device, dtype=latents.dtype, generator=session.generator
                        )
                        cache_latents = ((1 - sigma) * latents.float() + sigma * cache_noise.float()).to(latents.dtype)
                    cache_timestep = torch.tensor(sigma * 1000, device=self.device)
                    cache_step = SchedulerStep(
                        index=len(schedule), timestep=cache_timestep, next_timestep=cache_timestep
                    )
                    for values, cache in branches:
                        self._predict_segment(
                            cache_latents,
                            cache_step,
                            values,
                            cache,
                            total_steps=len(schedule) + 1,
                            commit=True,
                        )
                session.frame_offset += latents.shape[2]
                generated.append(latents)
                session.segment_tokens.append(
                    latents.shape[2] * latents.shape[3] * latents.shape[4] + text["encoder_hidden_states"].shape[1]
                )
                limit = request_options.get("max_cache_segments")
                if limit is not None:
                    if type(limit) is not int or limit <= 0:
                        raise ValueError("max_cache_segments must be a positive integer")
                    self._trim_history(session, limit)
            latents = torch.cat(generated, dim=2)
            with self._phase(self.components.decoder):
                if final and session.decoder is None:
                    pixels = self.components.decoder.decode(latents, request)
                else:
                    if session.decoder is None:
                        session.decoder = cast(Any, self.components.decoder).model.create_streaming_decode_session(
                            to_cpu=True, retain_history=False
                        )
                    pixels = session.decoder.push(latents, pixel_frames=request.num_frames)
                    if final:
                        tail = session.decoder.finish()
                        pixels = tail if not pixels.numel() else torch.cat((pixels, tail), dim=2)
                    if pixels.numel():
                        pixels = pixels[:, :, :, : request.height, : request.width]
            return DiffusionOutput(
                sample=pixels,
                latents=latents,
                metadata={
                    "model_id": self.model_id,
                    "seed": request.sampling.seed,
                    "segments": len(segments),
                    "frame_offset": session.frame_offset,
                    "kv_cache_bytes": sum(
                        item["store"].nbytes() for cache in (session.positive, session.negative) for item in cache
                    ),
                },
            )
        except BaseException:
            session.invalidate()
            raise

    def _predict_segment(
        self,
        latents: torch.Tensor,
        step: SchedulerStep,
        values: Mapping[str, Any],
        cache: list[dict[str, Any]],
        *,
        total_steps: int,
        commit: bool = False,
    ) -> torch.Tensor:
        return self.components.denoiser(
            DenoiserInput(
                latents=latents,
                timestep=step.timestep,
                next_timestep=step.next_timestep,
                step_index=step.index,
                total_steps=total_steps,
                conditioning={**values, "kv_cache": cache, "update_cache": commit},
            )
        ).sample

    @staticmethod
    def _trim_history(session: SegmentGenerationState, limit: int) -> None:
        if len(session.segment_tokens) <= limit:
            return
        remove = sum(session.segment_tokens[:-limit])
        session.segment_tokens[:] = session.segment_tokens[-limit:]
        for cache in (session.positive, session.negative):
            for layer in cache:
                store = layer["store"]
                if not store.length:
                    continue
                device = layer["mask"].device
                store.compact(torch.arange(remove, store.length, device=device))
                layer["mask"] = layer["mask"][:, remove:]


def build_segmented_strategy(context: Any) -> SegmentedAutoregressiveRunner:
    from .strategies import build_standard_strategy

    return cast(
        SegmentedAutoregressiveRunner, build_standard_strategy(context, runner_type=SegmentedAutoregressiveRunner)
    )

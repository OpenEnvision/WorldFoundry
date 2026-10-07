"""Framework-owned joint generation with rebuilt bounded clean context."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldfoundry.core.attention.cache.context import ContextAttentionCache

from ..contracts import Conditioning, DiffusionOutput, ModalityState
from .multistage import JointMultiStageDiffusionRunner


@dataclass(frozen=True)
class JointChunk:
    ranges: dict[str, tuple[int, int]]


class JointChunkedDiffusionRunner(JointMultiStageDiffusionRunner):
    """Denoise aligned chunks after replaying a prefix plus recent clean chunks.

    Each target gets a fresh cache and compact temporal coordinates. Components
    own tokenization, conditions, numerical updates and decoding.
    """

    def __init__(
        self, *, frames_per_chunk=4, temporal_compression=8, history_chunks=3,
        prefix_chunks=1, sampling_seed_offset=0, token_conditions=None, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if len(self.stage_steps) != 1:
            raise ValueError("joint chunked execution requires one generation stage")
        if frames_per_chunk <= 0 or history_chunks <= 0 or not 0 <= prefix_chunks < history_chunks:
            raise ValueError("invalid joint chunk or history geometry")
        self.frames_per_chunk = int(frames_per_chunk)
        self.temporal_compression = int(temporal_compression)
        self.history_chunks = int(history_chunks)
        self.prefix_chunks = int(prefix_chunks)
        self.sampling_seed_offset = int(sampling_seed_offset)
        self.token_conditions = dict(token_conditions or {})

    def _chunks(self, states, request):
        video = states["video"]
        latent_frames = (request.num_frames - 1) // self.temporal_compression + 1
        tokens_per_frame = video.latent.shape[1] // latent_frames
        if latent_frames % self.frames_per_chunk or tokens_per_frame * latent_frames != video.latent.shape[1]:
            raise ValueError("video tokens must form complete latent-frame chunks")
        temporal = video.positions[0, 0]
        stride = self.frames_per_chunk * tokens_per_frame
        ends = temporal[stride - 1::stride, 1]
        assignments = {
            name: torch.bucketize(state.positions[0, 0].mean(-1).contiguous(), ends.contiguous(), right=False)
            for name, state in states.items() if name != "video"
        }
        chunks = []
        for index in range(latent_frames // self.frames_per_chunk):
            ranges = {"video": (index * stride, (index + 1) * stride)}
            for name, assigned in assignments.items():
                indices = torch.nonzero(assigned == index).flatten()
                if not indices.numel():
                    raise ValueError(f"joint chunk {index} has no aligned {name} tokens")
                ranges[name] = (int(indices[0]), int(indices[-1]) + 1)
            chunks.append(JointChunk(ranges))
        for name, state in states.items():
            if chunks[-1].ranges[name][1] != state.latent.shape[1]:
                raise ValueError(f"{name} timeline extends past the reference video")
        return tuple(chunks)

    def _history_indices(self, count):
        if count <= self.history_chunks:
            return tuple(range(count))
        return (*range(self.prefix_chunks), *range(count - self.history_chunks + self.prefix_chunks, count))

    @staticmethod
    def _slice(states, chunk):
        result = {}
        for name, state in states.items():
            start, end = chunk.ranges[name]
            mask = state.attention_mask
            if mask is not None:
                mask = mask[..., start:end, start:end]
            result[name] = ModalityState(
                latent=state.latent[:, start:end], denoise_mask=state.denoise_mask[:, start:end],
                positions=state.positions[:, :, start:end], clean_latent=state.clean_latent[:, start:end],
                attention_mask=mask,
            )
        return result

    @staticmethod
    def _reframe(states, cursor):
        shift = cursor - states["video"].positions[:, 0, :, 0].amin(dim=1)
        result = {}
        for name, state in states.items():
            positions = state.positions.clone()
            positions[:, 0] += shift[:, None, None]
            result[name] = state.with_updates(positions=positions)
        return result, result["video"].positions[:, 0, :, 1].amax(dim=1)

    def _chunk_conditioning(self, conditioning, chunk, cache):
        shared = {**conditioning.shared, "attention_cache": cache}
        for key, modality in self.token_conditions.items():
            shared[key] = shared[key].token_slice(*chunk.ranges[modality])
        return Conditioning(positive=conditioning.positive, negative=conditioning.negative, shared=shared)

    @torch.no_grad()
    def run(self, request):
        if request.sampling.guidance_scale != 1 or request.negative_prompt is not None:
            raise ValueError("joint clean-context chunking requires unguided predictions")
        schedule = tuple(self.components.schedulers[0].schedule(request.sampling, device=self.device, dtype=self.dtype))
        conditioning = self.components.conditioner.encode(request, device=self.device, dtype=self.dtype)
        states = self._validate_states(self.components.latent_initializer.initialize(
            request, generator=self._generator(request.sampling.seed), device=self.device, dtype=self.dtype,
        ))
        generator = self._generator((request.sampling.seed + self.sampling_seed_offset) % (2**63 - 1))
        chunks = self._chunks(states, request)
        history = []
        for chunk in chunks:
            cache = ContextAttentionCache()
            cursor = torch.zeros(request.batch_size, device=self.device, dtype=states["video"].positions.dtype)
            for index in self._history_indices(len(history)):
                previous, cursor = self._reframe(history[index], cursor)
                self._predict(
                    states=previous,
                    conditioning=self._chunk_conditioning(conditioning, chunks[index], cache.writing()),
                    timestep=torch.zeros((), device=self.device), step_index=0,
                    total_steps=len(schedule), guidance_scale=1,
                )
            current, _ = self._reframe(self._slice(states, chunk), cursor)
            chunk_conditioning = self._chunk_conditioning(conditioning, chunk, cache)
            for step in schedule:
                prediction = self._predict(
                    states=current, conditioning=chunk_conditioning, timestep=step.timestep,
                    step_index=step.index, total_steps=len(schedule), guidance_scale=1,
                )
                if set(prediction.samples) != set(current):
                    raise ValueError("joint denoiser must return every initialized modality")
                for name, state in tuple(current.items()):
                    clean = prediction.samples[name]
                    if clean.shape != state.latent.shape:
                        raise ValueError(f"{name} prediction does not match its latent shape")
                    clean = (state.clean_latent.float() + state.denoise_mask.float() * (clean.float() - state.clean_latent.float())).to(state.latent.dtype)
                    latent = self.components.schedulers[0].step(clean, step, state.latent, generator=generator)
                    latent = (state.clean_latent.float() + state.denoise_mask.float() * (latent.float() - state.clean_latent.float())).to(state.latent.dtype)
                    current[name] = state.with_updates(latent=latent)
            del chunk_conditioning
            source = self._slice(states, chunk)
            history.append({name: state.with_updates(latent=current[name].latent) for name, state in source.items()})
        final = {
            name: state.with_updates(latent=torch.cat([chunk[name].latent for chunk in history], dim=1))
            for name, state in states.items()
        }
        artifacts = dict(self.components.decoder.decode_modalities(final, request))
        return DiffusionOutput(
            sample=artifacts["video"], latents=final["video"].latent, artifacts=artifacts,
            metadata={"model_id": self.model_id, "seed": request.sampling.seed,
                      "execution_strategy": "joint-chunked", "chunks": len(chunks), "output_layout": "FHWC"},
        )

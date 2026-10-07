"""Framework-owned chunk sampling with compressed HR/LR memory."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput
from worldfoundry.base_models.diffusion_model.extensions import DiffusionRunContext
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner


class CompressedMemoryRunner(NativeDiffusionRunner):
    """Keep rollout, causal VAE state and per-expert KV outside model components."""

    execution_strategy = "compressed-memory"

    @torch.no_grad()
    def run(self, request):
        denoiser = self.components.denoiser
        codec = self.components.decoder
        mode = denoiser.mode
        generator = self._generator(request.sampling.seed)
        conditioning = self.components.conditioner.encode(request, device=self.device, dtype=self.dtype)
        context = DiffusionRunContext(request=request, components=self.components,
                                      conditioning=conditioning, generator=generator)
        error = None
        try:
            for extension in self.extensions:
                extension.on_run_start(context)
            for extension in self.extensions:
                context.conditioning = extension.prepare_conditioning(context, context.conditioning)
            initialized = self.components.latent_initializer.initialize_with_encoder(
                request, latent_encoder=self.components.latent_encoder, generator=generator,
                device=self.device, dtype=self.dtype,
            )
            latents = initialized.latents
            high_condition = initialized.conditioning["condition_latents"].to(dtype=self.dtype)
            low_condition = initialized.conditioning["low_resolution_condition"].to(dtype=self.dtype)
            shared = context.conditioning.shared
            actions = shared["actions"]
            chunk_length = int(shared["chunk_length"])
            prompts = shared["chunk_prompts"]
            contexts = shared["chunk_contexts"]
            negative = context.conditioning.negative.get("context")
            chunks = latents.shape[2] // chunk_length
            sink = int(request.inputs.get("sink_size", 1))
            recent = int(request.inputs.get("temporal_size", 1))
            if not 0 <= sink <= chunk_length or not 1 <= recent <= chunk_length:
                raise ValueError("WorldPlay2 sink_size and temporal_size must fit one latent chunk")
            caches = denoiser.new_cache() if mode != "bi" else None
            stream = codec.new_stream_state()
            memories = {"high": None, "low": None}
            low_latents = []
            output_chunks = []
            return_latent = bool(request.inputs.get("return_latent", False))
            calls = 0
            context.total_steps = chunks * request.sampling.num_inference_steps
            for chunk_index in range(chunks):
                start, end = chunk_index * chunk_length, (chunk_index + 1) * chunk_length
                text = contexts[prompts[chunk_index]]
                chunk = latents[:, :, start:end].contiguous()
                schedule = self.components.scheduler.schedule(request.sampling, device=self.device, dtype=self.dtype)
                if len(schedule) != request.sampling.num_inference_steps:
                    raise ValueError("WorldPlay2 scheduler returned an unexpected number of steps")
                experts = (tuple(block.expert for block in self.components.scheduler.ordered_blocks)
                           if mode == "few_step" else tuple(
                               "high" if value >= 900 else "low"
                               for value in torch.stack([step.timestep for step in schedule]).cpu().tolist()
                           ))
                for step, expert in zip(schedule, experts):
                    context.step = replace(step, index=chunk_index * len(schedule) + step.index)
                    common = {
                        "expert": expert, "context": text,
                        "condition_latents": high_condition[:, :, start:end],
                        "actions": actions[start:end], "chunk_start": start,
                        "memory": memories[expert],
                        "cache": caches[expert]["positive"] if caches is not None else None,
                    }
                    if mode == "few_step":
                        common["pdd_block_index"] = self.components.scheduler.ordered_blocks[step.index].local_index
                    prediction = self._call_denoiser_with_conditioning(
                        context, latents=chunk, branch="positive", conditioning=common,
                    )
                    calls += 1
                    if request.sampling.guidance_scale != 1.0:
                        negative_condition = {**common, "context": negative,
                                              "cache": caches[expert]["negative"] if caches is not None else None}
                        unconditional = self._call_denoiser_with_conditioning(
                            context, latents=chunk, branch="negative", conditioning=negative_condition,
                        )
                        calls += 1
                        sample = unconditional.sample + request.sampling.guidance_scale * (prediction.sample - unconditional.sample)
                    elif mode == "few_step":
                        sample = torch.cat((prediction.sample, prediction.extras["endpoint_velocity"]), dim=1)
                    else:
                        sample = prediction.sample
                    chunk = self.components.scheduler.step(sample, step, chunk, generator=generator)
                    for extension in self.extensions:
                        chunk = extension.after_step(context, chunk)
                latents[:, :, start:end] = chunk.to(latents.dtype)
                if chunk_index + 1 < chunks or not return_latent:
                    rgb = codec.decode_chunk(chunk, state=stream, is_first_chunk=chunk_index == 0)
                    if not return_latent:
                        output_chunks.append(rgb.cpu())
                if chunk_index + 1 < chunks:
                    pixels = rgb[:, :, ::2]
                    batch, channels, frames, height, width = pixels.shape
                    pixels = F.interpolate(
                        pixels.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width),
                        size=(height // 4, width // 4), mode="bilinear", align_corners=False,
                    ).reshape(batch, frames, channels, height // 4, width // 4).permute(0, 2, 1, 3, 4).contiguous()
                    low_latents.append(codec.encode_chunk(pixels, state=stream, is_first_chunk=chunk_index == 0))
                    inputs = {
                        "hr_input": torch.cat((latents[:, :, :end], high_condition[:, :, :end]), dim=1).to(self.dtype),
                        "lr_input": torch.cat((torch.cat(low_latents, dim=2), low_condition[:, :, :end // 2]), dim=1).to(self.dtype),
                        "lr_actions": actions[::2][:end // 2],
                        "temporal_input": torch.cat((latents[:, :, end - recent:end],
                                                     high_condition[:, :, end - recent:end]), dim=1).to(self.dtype),
                        "temporal_actions": actions[end - recent:end],
                        "sink_actions": actions[:sink] if sink else None,
                    }
                    memories = denoiser.prepare_memory(memory_inputs=inputs, contexts=text,
                                                       negative_context=negative, caches=caches)
            self._notify_diffusion_complete(context, latents)
            sample = (latents if return_latent else
                      torch.cat(output_chunks, dim=2)[0].permute(1, 2, 3, 0).add(1).mul(0.5))
            for extension in self.extensions:
                sample = extension.after_decode(context, sample)
            output = DiffusionOutput(sample=sample, latents=latents, metadata={
                "model_id": self.model_id, "seed": request.sampling.seed, "mode": mode,
                "num_inference_steps": request.sampling.num_inference_steps,
                "guidance_scale": request.sampling.guidance_scale, "chunks": chunks,
                "model_evaluations": chunks * request.sampling.num_inference_steps,
                "denoiser_calls": calls, "execution_strategy": self.execution_strategy,
                "chunk_length": chunk_length, "sink_size": sink, "temporal_size": recent,
                "extensions": [extension.extension_id for extension in self.extensions],
            })
            for extension in reversed(self.extensions):
                extension.on_run_end(context)
            return output
        except BaseException as run_error:
            error = run_error
            for extension in reversed(self.extensions):
                extension.on_run_error(context, run_error)
            raise
        finally:
            self._end_denoiser_request(context, error=error)

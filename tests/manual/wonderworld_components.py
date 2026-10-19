# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Check real WonderWorld component weights and native math against primitives."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any


def verify_state(module: Any, path: Path) -> int:
    """Require every loaded tensor to equal the checkpoint cast to its dtype."""
    import torch
    from safetensors import safe_open

    loaded = module.state_dict()
    with safe_open(path, framework="pt", device="cpu") as source:
        keys = set(source.keys()) - {"text_model.embeddings.position_ids"}
        assert set(loaded) == keys, (path, set(loaded) - keys, keys - set(loaded))
        for name in sorted(keys):
            value = loaded[name].detach().cpu()
            assert torch.equal(value, source.get_tensor(name).to(value.dtype)), (path, name)
    return len(keys)


def main() -> None:
    """Validate released weights at 512px through canonical native recipes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput, DiffusionRequest, SamplingConfig
    from worldfoundry.base_models.diffusion_model.extensions.base import DiffusionRunContext
    from worldfoundry.base_models.diffusion_model.extensions.depth_guidance import DecodedDepthGuidance

    device, dtype = torch.device("cuda"), torch.bfloat16
    reports = []
    for model_id, dirname, mode in (
        ("sd2-inpaint", "stable-diffusion-2-inpainting", "rgb"),
        ("wonderworld-marigold-depth", "prs-eth--marigold-depth-v1-0", "depth"),
        ("marigold-normals-v0-1", "prs-eth--marigold-normals-v0-1", "normals"),
    ):
        root = args.checkpoint_root / dirname
        pipeline = NativeDiffusionPipeline.from_pretrained(
            model_id,
            policy=RuntimePolicy(device="cuda", dtype=dtype),
            checkpoint_overrides={key: str(root) for key in ("unet", "vae", "text_encoder", "tokenizer", "scheduler")},
        )
        parts = pipeline.components
        counts = {
            "unet": verify_state(parts.denoiser.unet, root / "unet/diffusion_pytorch_model.fp16.safetensors"),
            "vae": verify_state(parts.decoder.vae, root / "vae/diffusion_pytorch_model.fp16.safetensors"),
            "clip": verify_state(parts.conditioner.model.model, root / "text_encoder/model.fp16.safetensors"),
        }
        pixels = torch.linspace(-1, 1, 3 * 512 * 512, device=device, dtype=dtype).reshape(1, 3, 512, 512)
        mask = torch.zeros_like(pixels[:, :1])
        mask[:, :, 128:256] = 1
        inputs = {"image": pixels, "mask": mask}
        if mode == "normals":
            inputs["processing_resolution"] = 768
        request = DiffusionRequest(
            prompt="A furnished room",
            height=512,
            width=512,
            sampling=SamplingConfig(num_inference_steps=4, seed=42),
            inputs=inputs,
        )
        generator = torch.Generator(device=device).manual_seed(42)
        reference_generator = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            initialized = parts.latent_initializer.initialize_with_encoder(
                request, latent_encoder=parts.decoder, generator=generator, device=device, dtype=dtype
            )
            conditioning = parts.conditioner.encode(request, device=device, dtype=dtype)
            tokens = parts.conditioner.tokenizer(
                [request.prompt if mode == "rgb" else ""],
                padding="max_length" if mode == "rgb" else "do_not_pad",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text = parts.conditioner.model.model(tokens.input_ids.to(device))[0]
            torch.testing.assert_close(conditioning.positive["context"], text, rtol=0, atol=0)
            vae = parts.decoder.vae
            scale = float(vae.config.scaling_factor)
            if mode == "rgb":
                noise = torch.randn((1, 4, 64, 64), device=device, dtype=dtype, generator=reference_generator)
                masked = (pixels * (mask < 0.5)).contiguous(memory_format=torch.channels_last)
                observed = vae.encode(masked).latent_dist.sample(generator=reference_generator) * scale
                torch.testing.assert_close(initialized.conditioning["masked_image_latents"], observed, rtol=0, atol=0)
                joined = torch.cat((noise, functional.interpolate(mask, (64, 64), mode="nearest"), observed), dim=1)
            else:
                observation = (
                    functional.interpolate(pixels, (768, 768), mode="bilinear", align_corners=False, antialias=True)
                    if mode == "normals"
                    else pixels
                )
                observed = vae.encode(observation).latent_dist.mode() * scale
                noise = torch.randn(observed.shape, device=device, dtype=dtype, generator=reference_generator)
                torch.testing.assert_close(initialized.conditioning["rgb_latents"], observed, rtol=0, atol=0)
                joined = torch.cat((observed, noise), dim=1)
            torch.testing.assert_close(initialized.latents, noise, rtol=0, atol=0)
            schedule = parts.scheduler.schedule(request.sampling, device=device, dtype=dtype)
            step = schedule[0]
            values = {**conditioning.positive, **initialized.conditioning}
            model_input = DenoiserInput(noise, step.timestep, step.next_timestep, values, 0, 4)
            actual = parts.denoiser(model_input).sample
            expected = parts.denoiser.unet(joined, step.timestep, encoder_hidden_states=text, return_dict=False)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            config = json.loads((root / "scheduler/scheduler_config.json").read_text())
            reference: Any = (DDIMScheduler if mode != "depth" else EulerDiscreteScheduler).from_config(config)
            reference.set_timesteps(4, device=device)
            if mode == "depth":
                reference.is_scale_input_called = True
            expected_latent = reference.step(expected, step.timestep, noise, return_dict=False)[0]
            actual_latent = parts.scheduler.step(actual, step, noise, generator=generator)
            torch.testing.assert_close(actual_latent, expected_latent, rtol=0, atol=0)
            decoded = vae.decode(expected_latent / scale, return_dict=False)[0]
            if mode == "depth":
                normalized = (decoded.mean(1, keepdim=True).clamp(-1, 1) + 1) / 2
                expected_pixels = normalized.float() / 200 + 0.001
            elif mode == "normals":
                decoded = decoded.clamp(-1, 1)
                decoded = functional.normalize(
                    torch.cat((decoded[:, :2], decoded[:, 2:3].clamp_min(0)), dim=1), dim=1, eps=1e-6
                )
                expected_pixels = functional.normalize(
                    functional.interpolate(decoded, (512, 512), mode="bilinear", align_corners=False), dim=1, eps=1e-6
                )
            else:
                expected_pixels = decoded.clamp(-1, 1) / 2 + 0.5
            torch.testing.assert_close(parts.decoder.decode(actual_latent, request), expected_pixels, rtol=0, atol=0)
            if mode == "depth":
                guided_request = replace(
                    request,
                    inputs={
                        **inputs,
                        "target_depth": torch.full((1, 1, 512, 512), 0.002, device=device),
                        "align_mask": torch.ones((1, 1, 512, 512), dtype=torch.bool, device=device),
                        "guidance_steps": 4,
                    },
                )
                context = DiffusionRunContext(
                    request=guided_request, components=parts, conditioning=conditioning, generator=generator, step=step
                )
                guidance = DecodedDepthGuidance()
                guidance.before_denoiser(context, model_input)
                guided = guidance.after_denoiser(context, parts.denoiser(model_input))
                assert torch.isfinite(guided.sample).all() and not torch.equal(guided.sample, actual)
                assert not guided.sample.requires_grad and context.state == {}
                assert all(parameter.grad is None for parameter in parts.denoiser.unet.parameters())
        reports.append(
            {"model_id": model_id, "keys": counts, "latent_shape": list(noise.shape), "primitive_max_abs_diff": 0.0}
        )
        print(json.dumps(reports[-1]), flush=True)
        del pipeline, parts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "precision": "bfloat16",
                "scope": "checkpoint loading, component primitive parity and guided-depth gradients; not official full-pipeline parity",
                "models": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

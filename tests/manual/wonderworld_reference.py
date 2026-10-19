# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Compare native diffusion stages with pinned WonderWorld reference inference.

The reference checkout is used only by this validation tool. Production
inference uses the in-tree recipes, components and framework-owned runner.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

REFERENCE_REVISION = "5cf11467ad21a9d7bca269f961855a235c7d352e"


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Record exact comparison and absolute error without changing tolerances."""
    import torch

    error = (actual.float() - expected.float()).abs()
    return {"equal": torch.equal(actual, expected), "max_abs": float(error.max()), "mean_abs": float(error.mean())}


def main() -> None:
    """Execute one released inference stage with identical primitive weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("depth", "normals", "inpaint"), required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--farther", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    arguments = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, EulerDiscreteScheduler
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.version import cuda as cuda_version

    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline, RuntimePolicy
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
    from worldfoundry.base_models.diffusion_model.extensions.depth_guidance import DecodedDepthGuidance
    from worldfoundry.runtime.wonderworld_inference_spec import WONDERWORLD_NEGATIVE_PROMPT
    from worldfoundry.synthesis.visual_generation.wonderworld.runtime import image_tensor

    if (arguments.guided or arguments.farther) and arguments.mode != "depth":
        parser.error("decoded-depth guidance requires --mode depth")
    sys.path.insert(0, str(arguments.reference_root.resolve()))
    marigold = importlib.import_module("marigold_lcm.marigold_pipeline")
    inpainting = importlib.import_module("util.stable_diffusion_inpaint")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_id, directory, default_steps = {
        "depth": ("wonderworld-marigold-depth", "prs-eth--marigold-depth-v1-0", 30),
        "normals": ("marigold-normals-v0-1", "prs-eth--marigold-normals-v0-1", 10),
        "inpaint": ("sd2-inpaint", "stable-diffusion-2-inpainting", 50),
    }[arguments.mode]
    steps = default_steps if arguments.steps is None else arguments.steps
    guided = arguments.guided or arguments.farther
    guidance_steps = min(8, steps)
    root = arguments.checkpoint_root / directory
    native = NativeDiffusionPipeline.from_pretrained(
        model_id,
        policy=RuntimePolicy(device="cuda", dtype=getattr(torch, arguments.dtype)),
        checkpoint_overrides={name: str(root) for name in ("unet", "vae", "text_encoder", "tokenizer", "scheduler")},
        extensions=(DecodedDepthGuidance(),) if guided else (),
    )
    parts: Any = native.components
    config = json.loads((root / "scheduler/scheduler_config.json").read_text())
    scheduler: Any = (EulerDiscreteScheduler if arguments.mode == "depth" else DDIMScheduler).from_config(config)
    scheduler.is_scale_input_called = True
    components = {
        "unet": parts.denoiser.unet,
        "vae": parts.decoder.vae,
        "text_encoder": parts.conditioner.model.model,
        "tokenizer": parts.conditioner.tokenizer,
        "scheduler": scheduler,
    }
    if arguments.mode == "depth":
        reference = marigold.MarigoldPipeline(**components)
    elif arguments.mode == "normals":
        reference = marigold.MarigoldNormalsPipeline(
            **components, use_full_z_range=True, default_processing_resolution=768, default_denoising_steps=10
        )
    else:
        reference = inpainting.StableDiffusionInpaintPipeline(
            **components, safety_checker=None, feature_extractor=None, requires_safety_checker=False
        )
    reference.set_progress_bar_config(disable=True)
    with Image.open(arguments.image) as source:
        image = source.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    pixels = image_tensor(image, "cuda")
    mask = torch.zeros(1, 1, 512, 512, device="cuda")
    mask[..., 300:] = 1
    align = ~mask.bool() if arguments.farther else mask.bool()
    farther = mask.bool() if arguments.farther else None
    target = torch.full_like(mask, 0.003)
    inputs: dict[str, Any] = {"image": pixels * 2 - 1}
    if arguments.mode == "normals":
        inputs["processing_resolution"] = 768
    elif arguments.mode == "inpaint":
        inputs["mask"] = mask
    if guided:
        inputs.update(target_depth=target, align_mask=align, farther_mask=farther, guidance_steps=guidance_steps)
    request = DiffusionRequest(
        prompt="An abandoned wooden room, shelves and wooden floor, detailed photography",
        negative_prompt=WONDERWORLD_NEGATIVE_PROMPT,
        height=512,
        width=512,
        sampling=SamplingConfig(
            num_inference_steps=steps, guidance_scale=7.5 if arguments.mode == "inpaint" else 1, seed=42
        ),
        inputs=inputs,
    )
    traces: dict[str, list[dict[str, torch.Tensor]]] = {"native": [], "reference": []}
    encoder_inputs: dict[str, dict[str, Any]] = {}
    active = "native"

    def capture_encoder(module: Any, positional: Any) -> None:
        """Capture normalized pixels before the first checkpoint convolution."""
        del module
        encoder_inputs[active] = {
            "pixels": positional[0].detach().cpu().clone(),
            "stride": list(positional[0].stride()),
        }

    def capture_unet(module: Any, positional: Any, keyword: Any, output: Any) -> None:
        """Capture the complete released model-call boundary."""
        del module
        prediction = output[0] if isinstance(output, tuple) else output.sample
        traces[active].append(
            {
                "input": positional[0].detach().cpu().clone(),
                "prediction": prediction.detach().cpu().clone(),
                "context": keyword["encoder_hidden_states"].detach().cpu().clone(),
                "timestep": positional[1].detach().cpu().clone(),
            }
        )

    encoder_hook = parts.decoder.vae.encoder.register_forward_pre_hook(capture_encoder)
    unet_hook = parts.denoiser.unet.register_forward_hook(capture_unet, with_kwargs=True)
    try:
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            actual = native(request).sample.detach().float()
            active = "reference"
            if arguments.mode == "depth":
                normalized = reference(
                    image,
                    denoising_steps=steps,
                    ensemble_size=1,
                    processing_res=0,
                    batch_size=1,
                    seed=42,
                    show_progress_bar=False,
                    color_map=None,
                    depth_conditioning=guided,
                    target_depth=target if guided else None,
                    mask_align=align if guided else None,
                    mask_farther=farther,
                    guidance_steps=guidance_steps,
                )
                expected = normalized[None, None].float() / 200 + 0.001
            elif arguments.mode == "normals":
                expected = reference(
                    pixels * 2 - 1,
                    num_inference_steps=steps,
                    processing_resolution=768,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                ).float()
            else:
                decoded = reference(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    image=image,
                    mask_image=Image.fromarray((mask[0, 0].cpu().numpy() * 255).astype(np.uint8)),
                    height=512,
                    width=512,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                ).images
                expected = (decoded / 2 + 0.5).clamp(0, 1).float()
    finally:
        unet_hook.remove()
        encoder_hook.remove()
    comparisons = []
    first_guided = steps - guidance_steps
    assert len(traces["reference"]) == steps
    for index, row in enumerate(traces["reference"]):
        native_index = first_guided + 2 * (index - first_guided) + 1 if guided and index >= first_guided else index
        other = traces["native"][native_index]
        comparisons.append(
            {name: difference(other[name], row[name]) for name in ("input", "context", "prediction", "timestep")}
        )
    result: dict[str, Any] = {
        "source_revision": REFERENCE_REVISION,
        "scope": "released diffusion stages; not end-to-end world acceptance",
        "mode": arguments.mode,
        "guided": guided,
        "farther": arguments.farther,
        "steps": steps,
        "dtype": arguments.dtype,
        "torch": torch.__version__,
        "cuda": cuda_version,
        "device": torch.cuda.get_device_name(),
        "attention": "MATH",
        "output": difference(actual, expected),
        "calls": comparisons,
        "encoder_input": difference(encoder_inputs["native"]["pixels"], encoder_inputs["reference"]["pixels"]),
        "encoder_strides": {name: values["stride"] for name, values in encoder_inputs.items()},
    }
    passed = (
        result["output"]["equal"]
        and result["encoder_input"]["equal"]
        and all(values["equal"] for row in comparisons for values in row.values())
    )
    result["status"] = "passed" if passed else "failed"
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.save({"native": actual.cpu(), "reference": expected.cpu(), "traces": traces}, arguments.output / "tensors.pt")
    (arguments.output / "contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert passed, "native stage differs from the released reference"


if __name__ == "__main__":
    main()

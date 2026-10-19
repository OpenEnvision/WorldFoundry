# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free WonderWorld CLI contract for native offline scene generation."""

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

WONDERWORLD_NEGATIVE_PROMPT = (
    "collage, text, writings, signs, text, white border, photograph border, artifacts, blur, blurry, "
    "foggy, fog, person, bad quality, distortions, distorted image, watermark, signature, fisheye look"
)

WONDERWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="wonderworld",
    display_name="WonderWorld",
    default_variant_id="native-offline",
    default_task_id="image-to-3d-world",
    variants=(
        InferenceVariantSpec(
            variant_id="native-offline", label="Native layered Gaussian world", status="native_offline_inference"
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="image-to-3d-world",
            label="Image to layered Gaussian world",
            inputs=(
                InferenceFieldSpec("input_path", "Initial RGB image", kind="path", target="input_path", required=True),
                InferenceFieldSpec(
                    "prompt", "Scene completion prompt", default="A detailed photographic view of the scene."
                ),
                InferenceFieldSpec(
                    "camera_path",
                    "OpenCV world-to-camera path JSON",
                    kind="path",
                    description="Optional [N,4,4] matrices in exported scene units.",
                ),
                InferenceFieldSpec(
                    "checkpoint_root", "Shared component checkpoint root", kind="path", target="load_kwargs"
                ),
                InferenceFieldSpec("inpainting_path", "SD2 Inpainting directory", kind="path", target="load_kwargs"),
                InferenceFieldSpec("depth_path", "Marigold depth v1-0 directory", kind="path", target="load_kwargs"),
                InferenceFieldSpec(
                    "normals_path", "Marigold normals v0-1 directory", kind="path", target="load_kwargs"
                ),
                InferenceFieldSpec("oneformer_path", "OneFormer ADE20K directory", kind="path", target="load_kwargs"),
                InferenceFieldSpec("repvit_checkpoint", "RepViT-SAM checkpoint", kind="path", target="load_kwargs"),
                InferenceFieldSpec("negative_prompt", "Negative prompt", default=WONDERWORLD_NEGATIVE_PROMPT),
                InferenceFieldSpec("seed", "Seed", kind="integer", default=42),
                InferenceFieldSpec("depth_steps", "Depth diffusion steps", kind="integer", default=30),
                InferenceFieldSpec("inpaint_steps", "Inpainting steps", kind="integer", default=50),
                InferenceFieldSpec("normal_steps", "Normal diffusion steps", kind="integer", default=10),
                InferenceFieldSpec("refinement_steps", "Surfel reconstruction iterations", kind="integer", default=100),
                InferenceFieldSpec("focal_length", "Focal length in pixels", kind="number", default=500),
                InferenceFieldSpec("gen_layer", "Complete hidden background layer", kind="boolean", default=True),
            ),
            outputs=(InferenceArtifactSpec("wonderworld.splat", "generated_world", required=True, preview=True),),
            default_call_kwargs={
                "seed": 42,
                "depth_steps": 30,
                "inpaint_steps": 50,
                "normal_steps": 10,
                "refinement_steps": 100,
                "focal_length": 500,
                "gen_layer": True,
                "negative_prompt": WONDERWORLD_NEGATIVE_PROMPT,
                "return_dict": True,
            },
            description="Offline 512x512 seed scene plus optional finite camera-path extension; each request starts an independent scene.",
        ),
    ),
    notes=(
        "Uses public component weights, not a separate WonderWorld generator checkpoint. Full official rasterizer/quality parity and interactive browser acceptance are separate scopes.",
    ),
)

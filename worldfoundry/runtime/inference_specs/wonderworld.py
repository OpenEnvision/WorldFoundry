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
                InferenceFieldSpec("seed", "Seed", kind="integer"),
                InferenceFieldSpec("depth_steps", "Depth diffusion steps", kind="integer"),
                InferenceFieldSpec("inpaint_steps", "Inpainting steps", kind="integer"),
                InferenceFieldSpec("normal_steps", "Normal diffusion steps", kind="integer"),
                InferenceFieldSpec("refinement_steps", "Surfel reconstruction iterations", kind="integer"),
                InferenceFieldSpec("focal_length", "Focal length in pixels", kind="number"),
                InferenceFieldSpec("gen_layer", "Complete hidden background layer", kind="boolean"),
            ),
            outputs=(InferenceArtifactSpec("wonderworld.splat", "generated_world", required=True, preview=True),),
            default_call_kwargs={
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

# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``cosmos3`` model family."""

from __future__ import annotations

from typing import Any

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

COSMOS3_NANO_REPO_ID = "nvidia/Cosmos3-Nano"


COSMOS3_SUPER_REPO_ID = "nvidia/Cosmos3-Super"


COSMOS3_NANO_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"


COSMOS3_SUPER_REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"


COSMOS3_DEFAULT_PROMPT = "A robot arm is cleaning a plate in the kitchen"


COSMOS3_T2V_CALL_KWARGS = {
    "task_type": "text-to-video",
    "fps": 24,
    "guidance_scale": 6.0,
    "height": 720,
    "num_inference_steps": 35,
    "num_frames": 189,
    "output_type": "video",
    "seed": 0,
    "width": 1280,
    "flow_shift": 10.0,
    "use_karras_sigmas": False,
    "enable_safety_check": True,
}


def _cosmos3_task_profile(
    task_id: str,
    label: str,
    *,
    input_kind: str | None = None,
    image_output: bool = False,
) -> InferenceTaskProfile:
    """Build one Cosmos3 generator-inference task contract."""

    num_frames = 1 if image_output else 189
    shifted_scheduler = task_id in {"t2v", "v2v"}
    task_types = {
        "t2i": "text-to-image",
        "t2v": "text-to-video",
        "i2v": "image-to-video",
        "v2v": "video-to-video",
    }
    output_type = "pil" if image_output else "video"
    output_type_choices = ("pil", "pt", "np") if image_output else ("video",)
    defaults = {
        "task_type": task_types[task_id],
        "fps": 24,
        "guidance_scale": 6.0,
        "height": 720,
        "num_inference_steps": 35,
        "num_frames": num_frames,
        "output_type": output_type,
        "seed": 0,
        "width": 1280,
        "flow_shift": 10.0 if shifted_scheduler else 1.0,
        "use_karras_sigmas": not shifted_scheduler,
        "enable_safety_check": True,
    }
    supports_sound = not image_output
    if supports_sound:
        defaults["enable_sound"] = False
    fields: list[InferenceFieldSpec] = [
        _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS3_DEFAULT_PROMPT),
        _field(
            "load_sound_tokenizer",
            "Load Sound Tokenizer",
            kind="boolean",
            target="load_kwargs",
            default=True,
            description="Disable for visual-only runs to save memory; keep enabled when enable_sound may be used.",
        ),
    ]
    if input_kind is not None:
        fields.append(
            _field(
                "input_path",
                "Input Image" if input_kind == "image" else "Input Video",
                kind="path",
                target="input_path",
                required=True,
                default="",
            )
        )
    fields.extend(
        (
            _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
            _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=num_frames),
            _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
            _field("height", "Height", kind="integer", target="call_kwargs", default=720),
            _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
            _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=35),
            _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=6.0),
            _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
            _field(
                "flow_shift",
                "Flow Shift",
                kind="number",
                target="call_kwargs",
                default=defaults["flow_shift"],
            ),
            _field(
                "use_karras_sigmas",
                "Use Karras Sigmas",
                kind="boolean",
                target="call_kwargs",
                default=defaults["use_karras_sigmas"],
            ),
            _field(
                "enable_safety_check",
                "Run Safety Check",
                kind="boolean",
                target="call_kwargs",
                default=True,
            ),
            _field(
                "output_type",
                "Output Type",
                target="call_kwargs",
                default=output_type,
                choices=output_type_choices,
                description=(
                    "Workspace persists single-frame PIL/NumPy/PyTorch results as an image artifact."
                    if image_output
                    else "Workspace persists multi-frame Cosmos3 results as a video artifact."
                ),
            ),
            _field("output_path", "Output Path", kind="path", target="call_kwargs"),
        )
    )
    if supports_sound:
        fields.append(
            _field(
                "enable_sound",
                "Generate Synchronized Sound",
                kind="boolean",
                target="call_kwargs",
                default=False,
                description="Decode the checkpoint's synchronized sound stream and preserve its audio artifact.",
            )
        )
    artifact = (
        _artifact("image", "generated_image", required=True, preview=True)
        if image_output
        else _artifact("video", "video", required=True, preview=True)
    )
    outputs = [artifact]
    if supports_sound:
        outputs.append(
            _artifact(
                "audio",
                "audio",
                description="Optional synchronized audio waveform and muxed video track when enable_sound is true.",
            )
        )
    outputs.append(_artifact("manifest", "manifest", required=True))
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Cosmos3 {label.lower()} through the in-tree Diffusers generator runtime.",
        aliases=(task_types[task_id],),
        inputs=tuple(fields),
        outputs=tuple(outputs),
        default_call_kwargs=defaults,
    )


def _cosmos3_action_task_profile(mode: str, label: str) -> InferenceTaskProfile:
    """Build an official-default Cosmos3 action inference task."""

    task_id = f"action-{mode.replace('_', '-')}"
    defaults: dict[str, Any] = {
        "task_type": task_id,
        "action_mode": mode,
        "action_chunk_size": 16,
        "domain_name": "bridge_orig_lerobot",
        "resolution_tier": 480,
        "view_point": "ego_view",
        "num_frames": 17,
        "fps": 5,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "use_karras_sigmas": False,
        "use_system_prompt": False,
        "enable_safety_check": True,
        "enable_sound": False,
        "output_type": "video",
        "seed": 0,
    }
    fields: list[InferenceFieldSpec] = [
        _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS3_DEFAULT_PROMPT),
        _field(
            "load_sound_tokenizer",
            "Load Sound Tokenizer",
            kind="boolean",
            target="load_kwargs",
            default=False,
            description="Action inference is visual/action-only and does not require the AVAE sound tokenizer.",
        ),
        _field(
            "input_path",
            "Input Video",
            kind="path",
            target="input_path",
            required=True,
            default="",
            description=("Policy and forward dynamics use the first frame; inverse dynamics conditions on the clip."),
        ),
        _field(
            "action_mode",
            "Action Mode",
            target="call_kwargs",
            required=True,
            default=mode,
            choices=(mode,),
        ),
        _field(
            "action_chunk_size",
            "Action Chunk Size",
            kind="integer",
            target="call_kwargs",
            required=True,
            default=16,
            description="The generated clip contains chunk_size + 1 frames.",
        ),
        _field(
            "domain_name",
            "Action Domain",
            target="call_kwargs",
            required=True,
            default="bridge_orig_lerobot",
        ),
        _field(
            "resolution_tier",
            "Action Resolution Tier",
            kind="integer",
            target="call_kwargs",
            default=480,
            choices=(256, 480, 704, 720),
        ),
        _field(
            "view_point",
            "Action Viewpoint",
            target="call_kwargs",
            default="ego_view",
            choices=("ego_view", "third_person_view", "wrist_view", "concat_view"),
        ),
    ]
    if mode == "forward_dynamics":
        fields.append(
            _field(
                "raw_actions",
                "Raw Actions",
                kind="json",
                target="call_kwargs",
                required=True,
                description="A [T, D] action array driving the forward-dynamics rollout.",
            )
        )
    fields.extend(
        (
            _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=17),
            _field("fps", "FPS", kind="integer", target="call_kwargs", default=5),
            _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=30),
            _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
            _field("flow_shift", "Flow Shift", kind="number", target="call_kwargs", default=10.0),
            _field(
                "use_karras_sigmas",
                "Use Karras Sigmas",
                kind="boolean",
                target="call_kwargs",
                default=False,
            ),
            _field(
                "use_system_prompt",
                "Use System Prompt",
                kind="boolean",
                target="call_kwargs",
                default=False,
                description="Official action prompts are plain task descriptions and skip LLM system upsampling.",
            ),
            _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
            _field(
                "enable_safety_check",
                "Run Safety Check",
                kind="boolean",
                target="call_kwargs",
                default=True,
            ),
            _field(
                "output_type",
                "Output Type",
                target="call_kwargs",
                default="video",
                choices=("video",),
            ),
            _field("output_path", "Output Path", kind="path", target="call_kwargs"),
        )
    )
    outputs = [_artifact("video", "video", required=True, preview=True)]
    if mode in {"policy", "inverse_dynamics"}:
        outputs.append(
            _artifact(
                "action_trace",
                "action_trace",
                description="Predicted normalized actions emitted by the policy or inverse-dynamics head.",
            )
        )
    outputs.append(_artifact("manifest", "manifest", required=True))
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Cosmos3 {label.lower()} with the official action-inference defaults.",
        aliases=tuple(dict.fromkeys((mode, mode.replace("_", "-")))),
        inputs=tuple(fields),
        outputs=tuple(outputs),
        default_call_kwargs=defaults,
    )


COSMOS3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cosmos3",
    display_name="Cosmos3",
    default_variant_id="cosmos3-nano",
    default_task_id="t2v",
    aliases=("cosmos-3", "cosmos3-nano", "cosmos-3-nano", "cosmos3-super", "cosmos-3-super"),
    variants=(
        InferenceVariantSpec(
            variant_id="cosmos3-nano",
            label="Cosmos3 Nano",
            status="integrated",
            checkpoints=(InferenceCheckpointRef(role="primary", uri=COSMOS3_NANO_REPO_ID, status="official"),),
            load_kwargs={
                "variant_id": "cosmos3-nano",
                "profile_id": "cosmos3-nano",
                "runtime_profile": "cosmos3-nano",
                "revision": COSMOS3_NANO_REVISION,
                "load_sound_tokenizer": True,
            },
            call_kwargs=COSMOS3_T2V_CALL_KWARGS,
            aliases=("default", "nano", "cosmos-3-nano"),
        ),
        InferenceVariantSpec(
            variant_id="cosmos3-super",
            label="Cosmos3 Super",
            status="checkpoint-required",
            checkpoints=(InferenceCheckpointRef(role="primary", uri=COSMOS3_SUPER_REPO_ID, status="official"),),
            load_kwargs={
                "variant_id": "cosmos3-super",
                "profile_id": "cosmos3-super",
                "runtime_profile": "cosmos3-super",
                "revision": COSMOS3_SUPER_REVISION,
                "device_map": "balanced",
                "load_sound_tokenizer": True,
            },
            call_kwargs=COSMOS3_T2V_CALL_KWARGS,
            aliases=("super", "cosmos-3-super"),
            notes=("Requires a high-memory configuration; use device_map when distributing across GPUs.",),
        ),
    ),
    tasks=(
        _cosmos3_task_profile("t2i", "Text to Image", image_output=True),
        _cosmos3_task_profile("t2v", "Text to Video"),
        _cosmos3_task_profile("i2v", "Image to Video", input_kind="image"),
        _cosmos3_task_profile("v2v", "Video to Video", input_kind="video"),
        _cosmos3_action_task_profile("policy", "Action Policy"),
        _cosmos3_action_task_profile("forward_dynamics", "Action Forward Dynamics"),
        _cosmos3_action_task_profile("inverse_dynamics", "Action Inverse Dynamics"),
    ),
    notes=(
        "The Workspace contract exposes generator inference only: T2I/T2V/I2V/V2V, optional synchronized sound, "
        "and the checkpoint's structured policy/forward-dynamics/inverse-dynamics action modes.",
        "Cosmos3 Reasoner, training, and vLLM/vLLM-Omni serving workflows are intentionally not integrated.",
    ),
)

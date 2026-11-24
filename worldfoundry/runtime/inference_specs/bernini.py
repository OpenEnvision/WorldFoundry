# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``bernini`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import GENERIC_IMAGE_FIXTURE, _artifact, _field, _local_hfd_checkpoint_path

BERNINI_CHECKPOINT = _local_hfd_checkpoint_path("ByteDance--Bernini-Diffusers")


BERNINI_R_CHECKPOINT = _local_hfd_checkpoint_path("ByteDance--Bernini-R-Diffusers")


BERNINI_R_1P3B_CHECKPOINT = _local_hfd_checkpoint_path("ByteDance--Bernini-R-1.3B-Diffusers")


BERNINI_WORKSPACE_NPROC = 1


def _bernini_task_profile(
    task_id: str,
    label: str,
    *,
    input_kind: str | None = None,
    image_output: bool = False,
    reference_image: bool = False,
) -> InferenceTaskProfile:
    """Build one of Bernini's six official generation/editing contracts."""

    fields = [_field("prompt", "Prompt", target="prompt", required=True)]
    if input_kind is not None:
        fields.append(
            _field(
                "input_path",
                "Source Image" if input_kind == "image" else "Source Video",
                kind="path",
                target="input_path",
                required=True,
                default=GENERIC_IMAGE_FIXTURE if input_kind == "image" else None,
            )
        )
    if reference_image:
        fields.append(
            _field(
                "images",
                "Reference Image",
                kind="path",
                target="call_kwargs",
                required=True,
                default=GENERIC_IMAGE_FIXTURE,
            )
        )
    frames = 1 if image_output else 81
    height, width = (512, 512) if image_output else (480, 848)
    steps = 50 if task_id in {"t2i", "t2v"} else 40
    fields.extend(
        (
            _field("task_type", "Task Type", target="call_kwargs", default=task_id, choices=(task_id,)),
            _field("neg_prompt", "Negative Prompt", target="call_kwargs"),
            _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=frames),
            _field("height", "Height", kind="integer", target="call_kwargs", default=height),
            _field("width", "Width", kind="integer", target="call_kwargs", default=width),
            _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=steps),
            _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
            _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
            _field(
                "nproc_per_node",
                "GPU Processes",
                kind="integer",
                target="call_kwargs",
                default=BERNINI_WORKSPACE_NPROC,
            ),
            _field(
                "ulysses_size",
                "Ulysses Size",
                kind="integer",
                target="call_kwargs",
                default=BERNINI_WORKSPACE_NPROC,
            ),
            _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
        )
    )
    artifact = (
        _artifact("image", "generated_image", required=True, preview=True)
        if image_output
        else _artifact("video", "video", required=True, preview=True)
    )
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Bernini {label.lower()} through the official in-tree planner/renderer runtime.",
        aliases=(task_id, label.lower().replace(" ", "-")),
        inputs=tuple(fields),
        outputs=(artifact, _artifact("manifest", "manifest", required=True)),
        default_call_kwargs={
            "task_type": task_id,
            "num_frames": frames,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "seed": 42,
            "fps": 16,
            "nproc_per_node": BERNINI_WORKSPACE_NPROC,
            "ulysses_size": BERNINI_WORKSPACE_NPROC,
        },
    )


BERNINI_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="bernini",
    display_name="Bernini",
    default_variant_id="planner-renderer",
    default_task_id="t2v",
    aliases=("bernini-diffusers", "bernini-7b-14b"),
    variants=(
        InferenceVariantSpec(
            variant_id="planner-renderer",
            label="7B Planner + 14B Renderer",
            status="local_required",
            checkpoints=(InferenceCheckpointRef(role="primary", uri=BERNINI_CHECKPOINT, status="local_required"),),
            load_kwargs={"model_id": "bernini"},
            notes=(
                "Uses the full 81-frame/50-step recipe on one H100 because each additional Ulysses rank "
                "replicates about 179 GiB of host weights; the current cgroup does not leave a safe transient "
                "margin even for two ranks. "
                "The upstream eight-rank topology remains available as an explicit override on larger hosts.",
            ),
        ),
        InferenceVariantSpec(
            variant_id="renderer-14b",
            label="14B Renderer",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=BERNINI_R_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={"model_id": "bernini-r-14b"},
            call_kwargs={"nproc_per_node": 1, "ulysses_size": 1},
            aliases=("bernini-r-14b", "14b-renderer"),
            notes=("Streams the high- and low-noise experts sequentially on one GPU.",),
        ),
        InferenceVariantSpec(
            variant_id="renderer-1.3b",
            label="1.3B Renderer",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=BERNINI_R_1P3B_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={"model_id": "bernini-r-1.3b"},
            call_kwargs={"nproc_per_node": 1, "ulysses_size": 1},
            aliases=("bernini-r-1.3b", "1.3b-renderer"),
        ),
    ),
    tasks=(
        _bernini_task_profile("t2i", "Text to Image", image_output=True),
        _bernini_task_profile("i2i", "Image to Image", input_kind="image", image_output=True),
        _bernini_task_profile("t2v", "Text to Video"),
        _bernini_task_profile("v2v", "Video to Video", input_kind="video"),
        _bernini_task_profile("r2v", "Reference to Video", input_kind="image"),
        _bernini_task_profile(
            "rv2v",
            "Reference Video to Video",
            input_kind="video",
            reference_image=True,
        ),
    ),
    notes=("Checkpoint integrity and host/GPU memory are preflighted before any heavyweight model load.",),
)

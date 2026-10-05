"""RT-1 SavedModel inference contract shared by CLI, Studio, and model pages."""

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)


def _variant(name: str, label: str) -> InferenceVariantSpec:
    checkpoint = f"${{WORLDFOUNDRY_CKPT_DIR}}/embodied_action/rt-1/{name}"
    return InferenceVariantSpec(
        variant_id=name,
        label=label,
        status="checkpoint_backed_runtime",
        checkpoints=(InferenceCheckpointRef("primary", checkpoint),),
        load_kwargs={"checkpoint_dir": checkpoint},
    )


RT1_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="rt-1",
    display_name="RT-1",
    default_variant_id="rt1main",
    default_task_id="vla",
    aliases=("rt1", "robotics-transformer"),
    variants=(
        _variant("rt1main", "RT-1 main"),
        _variant("rt1multirobot", "RT-1 multirobot"),
        _variant("rt1simreal", "RT-1 sim-to-real"),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="vla",
            label="Robot action prediction",
            description="Run one RT-1 SavedModel action step with an image and a checkpoint-compatible language embedding.",
            inputs=(
                InferenceFieldSpec("input_path", "RGB observation", kind="path", target="input_path", required=True),
                InferenceFieldSpec("prompt", "Instruction", target="prompt", required=True),
                InferenceFieldSpec(
                    "language_embedding",
                    "Language embedding file",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    description="Path to a JSON or NPY file containing a checkpoint-compatible 512-D vector. Plain text alone is not encoded.",
                ),
                InferenceFieldSpec("checkpoint_dir", "SavedModel directory", kind="path", target="load_kwargs"),
                InferenceFieldSpec("plan_only", "Plan only", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(InferenceArtifactSpec("rt1_action_trace.json", "action_trace", required=True),),
            default_call_kwargs={"plan_only": False},
        ),
    ),
    notes=(
        "The released SavedModel does not encode prompt strings; a matching 512-D embedding must be supplied.",
        "Synthetic embedding GPU probes verify routing and output structure, not task semantics.",
    ),
)


__all__ = ["RT1_INFERENCE_SPEC"]

"""Shared HyperFlow input contracts for CLI, TUI and Studio."""

from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.runtime.interactive_inference_catalog import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

_BASE = str(hfd_root_path("MiniMaxAI--MiniMax-H3"))
_WEIGHTS = str(hfd_root_path("videorebirth--hyperflow"))
_COMMON = (
    InferenceFieldSpec("prompt", "Prompt", target="prompt", required=True),
    InferenceFieldSpec(
        "num_frames",
        "Frames",
        kind="integer",
        default=124,
        description="Aligned to 17n+5 frames, between 124 and 345, at 24 fps.",
    ),
    InferenceFieldSpec("height", "Height", kind="integer", description="Optional multiple of 32."),
    InferenceFieldSpec("width", "Width", kind="integer", description="Optional multiple of 32."),
    InferenceFieldSpec("seed", "Seed", kind="integer", default=42),
    InferenceFieldSpec("weights_path", "HyperFlow weights", kind="path", target="load_kwargs", default=_WEIGHTS),
    InferenceFieldSpec("python_executable", "Inference Python", kind="path", target="load_kwargs"),
    InferenceFieldSpec("gpus", "GPUs", kind="integer", target="load_kwargs", default=1),
    InferenceFieldSpec("memory_reserve_margin", "Reserved GPU memory", default="24GB"),
)
_OUTPUTS = (
    InferenceArtifactSpec("video", "video", required=True, preview=True, description="Video with generated audio."),
)


def _task(workflow, label, extra=()):
    return InferenceTaskProfile(
        workflow,
        label,
        (*_COMMON, *extra),
        _OUTPUTS,
        default_call_kwargs={"workflow": workflow, "num_frames": 124, "seed": 42, "return_dict": True},
    )


HYPERFLOW_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hyperflow",
    display_name="HyperFlow for MiniMax-H3",
    variants=(
        InferenceVariantSpec(
            "hyperflow",
            "HyperFlow 8-step v1.0",
            checkpoints=(InferenceCheckpointRef("base_model", _BASE), InferenceCheckpointRef("lora", _WEIGHTS)),
            load_kwargs={"model_path": _BASE, "weights_path": _WEIGHTS},
            aliases=("hyperflow-h3", "minimax-h3-hyperflow"),
        ),
    ),
    tasks=(
        _task("t2va", "Text to video + audio"),
        _task(
            "fl2va",
            "First / last frame to video + audio",
            (
                InferenceFieldSpec("input_path", "First frame", kind="path", target="input_path"),
                InferenceFieldSpec("last_image", "Last frame", kind="path"),
            ),
        ),
        _task(
            "ref2va",
            "Multimodal references to video + audio",
            (
                InferenceFieldSpec(
                    "references",
                    "Ordered references",
                    kind="json",
                    required=True,
                    description="List of local paths or kind/path objects. Ref2VA requires 124–345 aligned frames at 24 fps.",
                ),
            ),
        ),
    ),
    default_variant_id="hyperflow",
    default_task_id="t2va",
    aliases=("hyperflow-h3", "minimax-h3-hyperflow"),
)

"""Shared Pika H3 Flex input contracts for CLI, TUI and Studio."""

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
_FLEX = str(hfd_root_path("Pika-Labs--Pika-H3-Flex"))
_COMMON = (
    InferenceFieldSpec("prompt", "Prompt", target="prompt", required=True),
    InferenceFieldSpec("num_frames", "Frames", kind="integer"),
    InferenceFieldSpec("short_edge", "Short edge", kind="integer", default=768),
    InferenceFieldSpec("aspect_ratio", "Aspect ratio", default="16:9"),
    InferenceFieldSpec("seed", "Seed", kind="integer"),
    InferenceFieldSpec("adapter_task", "Adapter", target="load_kwargs", default="fl2va"),
    InferenceFieldSpec("base_model_path", "MiniMax-H3 base", kind="path", target="load_kwargs", default=_BASE),
)
_OUTPUTS = (
    InferenceArtifactSpec(
        "video", "video", required=True, preview=True, description="Video with generated stereo audio."
    ),
)


def _task(task_id: str, label: str, extra=()) -> InferenceTaskProfile:
    return InferenceTaskProfile(
        task_id,
        label,
        (*_COMMON, *extra),
        _OUTPUTS,
        default_call_kwargs={"task": task_id, "return_dict": True},
    )


PIKA_H3_FLEX_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="pika-h3-flex",
    display_name="Pika H3 Flex",
    variants=(
        InferenceVariantSpec(
            "pika-h3-flex",
            "Pika H3 Flex six-step adapters",
            checkpoints=(
                InferenceCheckpointRef("base_model", _BASE),
                InferenceCheckpointRef("fl2va_adapter", _FLEX + "/fl2va/adapter.safetensors"),
                InferenceCheckpointRef("ref2va_adapter", _FLEX + "/ref2va/adapter.safetensors"),
            ),
            load_kwargs={
                "base_model_path": _BASE,
                "adapter_task": "fl2va",
            },
            aliases=("pika-h3", "pika-h3-flex-fl2va", "pika-h3-flex-ref2va"),
        ),
    ),
    tasks=(
        _task("t2va", "Text to video + audio"),
        _task("i2va", "First frame to video + audio", (InferenceFieldSpec("images", "First image", kind="path"),)),
        _task("l2va", "Last frame to video + audio", (InferenceFieldSpec("images", "Last image", kind="path"),)),
        _task(
            "fl2va",
            "First and last frame to video + audio",
            (InferenceFieldSpec("images", "Endpoint images", kind="json"),),
        ),
        _task(
            "ref2va",
            "Image references to video + audio",
            (InferenceFieldSpec("references", "Ordered image references", kind="json", required=True),),
        ),
    ),
    default_variant_id="pika-h3-flex",
    default_task_id="t2va",
    aliases=("pika-h3", "pika-h3-flex-fl2va", "pika-h3-flex-ref2va"),
)

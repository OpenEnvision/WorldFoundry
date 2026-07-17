"""Pipeline invocation construction and execution helpers.

Builds normalized :class:`PipelineInvocation` objects from raw
:class:`GenerationRequest` inputs, resolves prompts / images / video /
interaction controls, and dispatches calls to pipeline objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.api import GenerationRequest

from .loading import first_text

# ── Input key constants ──────────────────────────────────────────────

# Keys that carry a text prompt from the request inputs.
TEXT_INPUT_KEYS = ("prompt", "instruction", "caption", "text")
# Keys that carry an image input from the request inputs.
IMAGE_INPUT_KEYS = ("image", "input_image", "ref_image")
# Keys that carry a video input from the request inputs.
VIDEO_INPUT_KEYS = ("video", "video_context")
# Keys that carry action / interaction control signals (robotics, latent tokens, etc.).
ACTION_CONTROL_KEYS = (
    "interactions",
    "interaction",
    "actions",
    "action",
    "action_chunk",
    "action_chunks",
    "action_sequence",
    "continuous_action",
    "giga_brain_actions",
    "interaction_signal",
    "latent_action_tokens",
    "world_action",
    "world_actions",
    "robot_action",
    "robot_actions",
    "joint_position",
    "gripper_position",
    "dreamzero_actions",
)
# All input keys that are consumed by the invocation builder and excluded
# from operator-specific kwargs.
CONSUMED_INPUT_KEYS = frozenset((*TEXT_INPUT_KEYS, *IMAGE_INPUT_KEYS, *VIDEO_INPUT_KEYS, "ref_image_path"))


# ── Invocation dataclass ──────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineInvocation:
    """Normalized inputs for one pipeline-backed generation request.

    Attributes:
        request: The original :class:`GenerationRequest` from the evaluation API.
        prompt: Resolved text prompt (first truthy value across text input keys).
        image: Resolved image input, or ``None`` if not provided.
        video: Resolved video input, or ``None`` if not provided.
        interactions: Resolved action / interaction control signals.
        ref_image_path: Optional path to a reference image on disk.
        output_path: Absolute file path for the generated artifact.
        operator_kwargs: Extra keyword arguments forwarded to the pipeline operator.
        pipeline_kwargs: Remaining generation kwargs not consumed by the builder.
    """

    request: GenerationRequest
    prompt: str
    image: Any
    video: Any
    interactions: Any
    ref_image_path: Any
    output_path: Path
    operator_kwargs: Mapping[str, Any]
    pipeline_kwargs: Mapping[str, Any]


# ── Output path helpers ──────────────────────────────────────────────

def request_output_dir(runner_output_dir: Path | None, kwargs: dict[str, Any]) -> Path:
    """Resolve and create the output directory for a request execution."""
    output_dir = runner_output_dir or Path(kwargs.pop("output_dir", "tmp/pipeline_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sample_output_path(output_dir: Path, request: GenerationRequest, artifact_filename: str) -> Path:
    """Generate the absolute file path for a request's generated artifact."""
    return output_dir / f"{request.sample_id}_{artifact_filename}"


# ── Private helpers ──────────────────────────────────────────────────

def _first_truthy(source: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first value in ``source`` mapping associated with ``keys`` that evaluates to True."""
    for key in keys:
        value = source.get(key)
        if value:
            return value
    return None


def _is_empty_value(value: Any) -> bool:
    """Determine if a given value is considered empty (``None``, empty string, or empty container)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _sample_controls(controls: Mapping[str, Any]) -> Mapping[str, Any]:
    """Safely extract nested ``sample_controls`` mapping from parent controls."""
    sample_controls = controls.get("sample_controls")
    return sample_controls if isinstance(sample_controls, Mapping) else {}


def _controls_actions(controls: Mapping[str, Any], kwargs: dict[str, Any]) -> Any:
    """Resolve and retrieve interaction or action control signals from arguments or controls.

    Priority order: kwargs > controls > nested ``sample_controls``.
    """
    nested_controls = _sample_controls(controls)
    # NOTE: kwargs-level keys are popped so they don't leak into pipeline_kwargs.
    for key in ACTION_CONTROL_KEYS:
        if key in kwargs:
            return kwargs.pop(key)
    for source in (controls, nested_controls):
        for key in ACTION_CONTROL_KEYS:
            if key in source:
                return source[key]
    return ()


def _operator_specific_inputs(inputs: Mapping[str, Any], controls: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble non-consumed inputs and controls as operator-specific execution arguments.

    Filters out keys already consumed by the invocation builder (text, image,
    video, ``ref_image_path``) and action control keys, then merges remaining
    control entries with ``setdefault`` to avoid overwriting explicit inputs.
    """
    payload = {
        str(key): value
        for key, value in inputs.items()
        if key not in CONSUMED_INPUT_KEYS and not _is_empty_value(value)
    }
    for source in (controls, _sample_controls(controls)):
        for key, value in source.items():
            if key == "sample_controls" or key in ACTION_CONTROL_KEYS or _is_empty_value(value):
                continue
            payload.setdefault(str(key), value)
    return payload


# ── Public invocation builders ───────────────────────────────────────

def build_pipeline_invocation(
    *,
    request: GenerationRequest,
    output_dir: Path,
    artifact_filename: str,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> PipelineInvocation:
    """Construct a :class:`PipelineInvocation` from a request, resolving inputs, prompts, and options.

    Args:
        request: The original :class:`GenerationRequest` carrying inputs and controls.
        output_dir: Base directory for generated artifacts.
        artifact_filename: Filename suffix for the output artifact.
        generation_kwargs: Optional override kwargs; falls back to ``request.generation_kwargs``.
    """
    inputs: Mapping[str, Any] = request.inputs
    controls = dict(request.controls or {})
    kwargs = dict(generation_kwargs if generation_kwargs is not None else request.generation_kwargs or {})
    prompt = first_text(kwargs.pop("prompt", None), *(inputs.get(key) for key in TEXT_INPUT_KEYS))
    explicit_operator_kwargs = dict(kwargs.pop("operator_kwargs", {}) or {})
    operator_kwargs = _operator_specific_inputs(inputs, controls)
    operator_kwargs.setdefault("sample_id", request.sample_id)
    operator_kwargs.setdefault("task_name", request.task_name)
    operator_kwargs.update(explicit_operator_kwargs)
    return PipelineInvocation(
        request=request,
        prompt=prompt,
        image=_first_truthy(inputs, IMAGE_INPUT_KEYS),
        video=_first_truthy(inputs, VIDEO_INPUT_KEYS),
        interactions=_controls_actions(controls, kwargs),
        ref_image_path=inputs.get("ref_image_path"),
        output_path=sample_output_path(output_dir, request, artifact_filename),
        operator_kwargs=operator_kwargs,
        pipeline_kwargs=kwargs,
    )


# ── Pipeline dispatch ───────────────────────────────────────────────

def invoke_pipeline(pipeline: Any, invocation: PipelineInvocation) -> Mapping[str, Any]:
    """Invoke the pipeline using either native ``run_pipeline_invocation`` or standard ``__call__``.

    Args:
        pipeline: The loaded pipeline object.
        invocation: The fully-resolved :class:`PipelineInvocation` with all inputs.

    Returns:
        A mapping of result keys from the pipeline execution.

    Raises:
        TypeError: If the pipeline does not return a mapping when ``return_dict=True``.
    """
    # Prefer the native invocation method if the pipeline defines one.
    native_invoker = getattr(pipeline, "run_pipeline_invocation", None)
    if callable(native_invoker):
        result = native_invoker(invocation)
    else:
        result = pipeline(
            prompt=invocation.prompt,
            images=invocation.image,
            video=invocation.video,
            interactions=invocation.interactions,
            ref_image_path=invocation.ref_image_path,
            output_path=invocation.output_path,
            return_dict=True,
            operator_kwargs=invocation.operator_kwargs,
            **invocation.pipeline_kwargs,
        )
    if not isinstance(result, Mapping):
        raise TypeError(
            f"pipeline must return a mapping when called with return_dict=True; got {type(result).__name__}."
        )
    return result


__all__ = [
    "ACTION_CONTROL_KEYS",
    "CONSUMED_INPUT_KEYS",
    "IMAGE_INPUT_KEYS",
    "PipelineInvocation",
    "TEXT_INPUT_KEYS",
    "VIDEO_INPUT_KEYS",
    "build_pipeline_invocation",
    "invoke_pipeline",
    "request_output_dir",
    "sample_output_path",
]

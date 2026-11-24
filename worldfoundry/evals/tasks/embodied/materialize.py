"""Embodied Evaluation Request Materialization Engine.

This module provides the logic to convert abstract datasets or lists of unstructured samples
into concrete `GenerationRequest` instances required by VLA, Video Action, and WAM runners.

It handles:
1. ID Resolution: Resolves deduplicated sample IDs using prioritized key fallback routes.
2. Context and Observation Gathering: Merges unstructured key-value sensory inputs (camera views,
   images, prompt instructions, proprioceptions) into standard observation formats.
3. Control Parametrization: Resolves environmental settings, trajectories, and step boundaries.
4. Schema Resolution: Dynamically matches ground truth outputs with the output schema expected by
   downstream metrics evaluators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from worldfoundry.evals.api import GenerationRequest

from .contracts import EmbodiedGenerationSpec, EvaluationTrack

# Standard schema version for materialized request outputs.
VLA_VA_WAM_REQUESTS_SCHEMA_VERSION = "worldfoundry-vla-va-wam-requests"

# Default observation keys extracted from loose datasets if not explicitly specified.
DEFAULT_OBSERVATION_FALLBACK_KEYS = (
    "instruction",
    "prompt",
    "text",
    "image",
    "video",
    "observations",
    "observation",
    "state",
    "proprio",
    "camera",
    "trajectory",
)

# Standard control parameters that shape episode reset behavior and target lengths.
DEFAULT_SAMPLE_CONTROL_KEYS = (
    "action",
    "actions",
    "control",
    "controls",
    "episode_id",
    "env",
    "horizon_steps",
    "initial_state",
    "reset_seed",
    "scenario",
    "task_id",
)


@dataclass(frozen=True)
class VlaVaWamMaterializedRequests:
    """Encapsulates a fully materialized, immutable batch of GenerationRequests.

    Represents the complete task batch targeting a specific split and evaluation track,
    ready for execution by VLA, Video Action, or WAM runners.

    Attributes:
        schema_version: The version string identifying the schema of the materialized requests.
        track: The name of the evaluation track (e.g., 'vla', 'video_action', 'wam').
        task_name: The specific task identifier within the track.
        split: The data split (e.g., 'train', 'validation', 'test') these requests belong to.
        requests: An immutable tuple of `GenerationRequest` instances.
    """

    schema_version: str
    track: str
    task_name: str
    split: str
    requests: tuple[GenerationRequest, ...]

    @property
    def sample_count(self) -> int:
        """Returns the total number of individual `GenerationRequest` samples in this batch."""
        return len(self.requests)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the materialized batch into a JSON-compatible dictionary.

        Returns:
            A dictionary representation of the `VlaVaWamMaterializedRequests` object,
            suitable for storage or transmission.
        """
        return {
            "schema_version": self.schema_version,
            "track": self.track,
            "task_name": self.task_name,
            "split": self.split,
            "sample_count": len(self.requests),
            "requests": [request.to_dict() for request in self.requests],
        }


def _sample_id(sample: Mapping[str, Any], index: int) -> str:
    """Extracts a unique, human-readable identifier from unstructured sample dictionary fields.

    It attempts to find an ID using a prioritized list of common keys. If no suitable key is found,
    a fallback ID is generated using the provided index.

    Args:
        sample: A mapping representing the dataset sample.
        index: Fallback index to construct a sample ID if no pre-defined keys match.

    Returns:
        A unique string sample ID.
    """
    for key in ("sample_id", "episode_id", "id", "task_id", "name"):
        value = sample.get(key)
        if value not in (None, ""):
            return str(value)
    # If no specific ID key is found, generate a unique ID using the provided index.
    return f"sample-{index:06d}"


def _merge_mapping(target: dict[str, Any], value: Any) -> None:
    """Safely merges stringified key-value fields from a source mapping into a target dictionary.

    This utility function ensures that keys are always stored as strings in the target dictionary.

    Args:
        target: The destination dictionary to update.
        value: Source value, expected to be a mapping. If not a mapping, no action is taken.
    """
    if isinstance(value, Mapping):
        # Convert all keys in the source mapping to strings before updating the target.
        target.update({str(key): item for key, item in value.items()})


def _inputs_for_sample(sample: Mapping[str, Any], spec: EmbodiedGenerationSpec) -> dict[str, Any]:
    """Assembles all multimodal inputs and observations representing the current step's sensory state.

    This function aggregates inputs from various sources within the raw sample, prioritizing
    explicitly defined 'inputs' or 'initial_context', then track-specific observation keys,
    and finally common fallback observation keys.

    Args:
        sample: A raw dataset sample mapping, potentially containing sensory data.
        spec: The embodied generation specification, defining track-specific observation keys.

    Returns:
        A dictionary containing assembled sensory inputs (e.g., images, text prompts, states).
    """
    inputs: dict[str, Any] = {}
    # Merge explicitly defined 'inputs' or 'initial_context' blocks first.
    _merge_mapping(inputs, sample.get("inputs"))
    _merge_mapping(inputs, sample.get("initial_context"))

    # Iterate through specified and default observation keys, adding them if present and not already added.
    for key in (*spec.observation_keys, *DEFAULT_OBSERVATION_FALLBACK_KEYS):
        if key in sample and key not in inputs:
            inputs[str(key)] = sample[key]

    # Handle optional 'references' block.
    references = sample.get("references")
    if isinstance(references, Mapping):
        inputs["references"] = dict(references)  # Store a copy of references.
    return inputs


def _sample_controls(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Assembles granular control configurations to parameterize individual simulation episodes or steps.

    This function gathers control parameters from specific 'sample_controls' or 'controls' keys,
    as well as from a list of common default sample control keys present directly in the sample.

    Args:
        sample: Raw dataset sample mapping, potentially containing control parameters.

    Returns:
        A dictionary of control parameters (e.g., episode ID, reset seed, horizon steps).
    """
    controls: dict[str, Any] = {}
    # Merge explicitly defined control blocks from 'sample_controls' or 'controls'.
    _merge_mapping(controls, sample.get("sample_controls"))
    _merge_mapping(controls, sample.get("controls"))

    # Merge individual control parameters specified by DEFAULT_SAMPLE_CONTROL_KEYS.
    for key in DEFAULT_SAMPLE_CONTROL_KEYS:
        if key == "controls":  # Avoid re-processing the 'controls' block already handled above.
            continue
        if key in sample and key not in controls:
            controls[str(key)] = sample[key]
    return controls


def _output_schema(spec: EmbodiedGenerationSpec, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Compiles the expected target schema for evaluation outputs and matches ground truths.

    This function constructs the output schema based on the `EmbodiedGenerationSpec`'s
    `output_keys` and then overlays any specific `expected_outputs` defined in the sample.

    Args:
        spec: EmbodiedGenerationSpec specifying the expected output keys for the task.
        sample: Raw dataset sample mapping, potentially containing `expected_outputs`.

    Returns:
        A dictionary representing the full output schema, including expected ground truth values
        where provided.
    """
    # Initialize schema from the spec's output_keys, setting a default 'kind' for each.
    schema = {key: {"kind": key} for key in spec.output_keys}

    # Overlay specific expected outputs from the sample.
    expected_outputs = sample.get("expected_outputs")
    if isinstance(expected_outputs, Mapping):
        for key, value in expected_outputs.items():
            # If the expected output value is a mapping, copy it directly.
            # Otherwise, wrap it in a dictionary with 'expected' key.
            schema[str(key)] = dict(value) if isinstance(value, Mapping) else {"expected": value}
    return schema


def request_from_vla_va_wam_sample(
    sample: Mapping[str, Any],
    *,
    spec: EmbodiedGenerationSpec | Mapping[str, Any],
    index: int = 0,
    split: str = "default",
    generation_defaults: Mapping[str, Any] | None = None,
    cache_policy: Mapping[str, Any] | None = None,
) -> GenerationRequest:
    """Builds a single detailed `GenerationRequest` from a raw dataset sample dictionary.

    This function orchestrates the resolution of sample IDs, inputs, controls, and output
    schemas, combining information from the raw sample, the `EmbodiedGenerationSpec`,
    and various default/fallback mechanisms.

    Args:
        sample: A single unstructured or partially-structured dictionary item from a dataset.
        spec: The schema specification (or a dictionary that can be converted to one)
              governing the target evaluation run.
        index: The positional index of the sample, used as a fallback for sample ID generation.
        split: The target data split identifier, defaults to "default".
        generation_defaults: Global generation parameters acting as base defaults for the request.
        cache_policy: Caching constraints mapping specific to this request.

    Returns:
        A fully formed `GenerationRequest` instance, ready for evaluation.
    """
    # Resolve the spec into an EmbodiedGenerationSpec object if it's not already one.
    resolved_spec = spec if isinstance(spec, EmbodiedGenerationSpec) else EmbodiedGenerationSpec.from_dict(spec)

    sample_id = _sample_id(sample, index)

    # Initialize generation kwargs with global defaults and then merge sample-specific overrides.
    generation_kwargs = dict(generation_defaults or {})
    _merge_mapping(generation_kwargs, sample.get("generation_kwargs"))

    # Initialize controls with resolved spec values.
    controls = resolved_spec.to_dict()
    # Merge sample-specific controls into the main controls dictionary.
    sample_controls = _sample_controls(sample)
    if sample_controls:
        controls["sample_controls"] = sample_controls

    return GenerationRequest(
        sample_id=sample_id,
        task_name=resolved_spec.task_name,
        split=str(sample.get("split", split)),  # Use sample's split if available, otherwise default.
        request_id=str(sample.get("request_id", f"{resolved_spec.task_name}:{sample_id}")),
        inputs=_inputs_for_sample(sample, resolved_spec),
        controls=controls,
        generation_kwargs=generation_kwargs,
        output_schema=_output_schema(resolved_spec, sample),
        cache_policy=cache_policy or {},
    )


def materialize_vla_va_wam_requests(
    samples: Sequence[Mapping[str, Any]],
    *,
    spec: EmbodiedGenerationSpec | Mapping[str, Any],
    split: str = "default",
    generation_defaults: Mapping[str, Any] | None = None,
    cache_policy: Mapping[str, Any] | None = None,
) -> VlaVaWamMaterializedRequests:
    """Materializes a list of loose dataset sample dictionaries into a complete, structured request batch.

    This is the primary entry point for converting a sequence of raw dataset samples into a
    batch of `GenerationRequest` objects, encapsulating them within a `VlaVaWamMaterializedRequests`
    container.

    Args:
        samples: A sequence of unstructured or partially-structured dictionary items, each
                 representing a single evaluation sample.
        spec: The schema specification (or a dictionary that can be converted to one)
              governing the target evaluation run.
        split: The target data split identifier (e.g., 'train', 'validation', 'test').
               Defaults to "default".
        generation_defaults: Global generation parameters acting as base defaults for all requests
                             in this batch.
        cache_policy: Global caching constraints mapping to apply to all requests in this batch.

    Returns:
        VlaVaWamMaterializedRequests: An object capturing all compiled `GenerationRequest`s
                                      and track-specific details.
    """
    # Resolve the spec into an EmbodiedGenerationSpec object if it's not already one.
    resolved_spec = spec if isinstance(spec, EmbodiedGenerationSpec) else EmbodiedGenerationSpec.from_dict(spec)

    # Convert each raw sample into a GenerationRequest using the helper function.
    requests = tuple(
        request_from_vla_va_wam_sample(
            sample,
            spec=resolved_spec,
            index=index,
            split=split,
            generation_defaults=generation_defaults,
            cache_policy=cache_policy,
        )
        for index, sample in enumerate(samples)
    )
    return VlaVaWamMaterializedRequests(
        schema_version=VLA_VA_WAM_REQUESTS_SCHEMA_VERSION,
        track=cast(EvaluationTrack, resolved_spec.track).value,
        task_name=resolved_spec.task_name,
        split=split,
        requests=requests,
    )


__all__ = [
    "VLA_VA_WAM_REQUESTS_SCHEMA_VERSION",
    "VlaVaWamMaterializedRequests",
    "materialize_vla_va_wam_requests",
    "request_from_vla_va_wam_sample",
]

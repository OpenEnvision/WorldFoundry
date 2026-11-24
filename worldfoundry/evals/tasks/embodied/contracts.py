"""Embodied AI, VLA, Video Action, and World Action Model Evaluation Contracts.

This module defines the unified interfaces, capability tags, validation contracts,
and execution protocols for evaluating Embodied AI models.

The evaluation covers four distinct tracks:
1. **VLA (Vision-Language-Action)**: Evaluates models predicting discrete or continuous actions
   directly from multimodal (visual & text) observations under a policy rollout.
2. **VA (Video Action)**: Evaluates video-based action prediction where spatial-temporal video cues
   are converted into physical control vectors.
3. **VAM (Video Action Modeling)**: Evaluates the generation and reconstruction of future visual
   states coupled with action trajectories (world model and policy integration).
4. **WAM (World Action Modeling)**: Evaluates world models capable of simulating environmental
   transitions, accepting reset/branch operations, and executing simulated rollouts.

All serialization and validation rely on JSON-compatible schemas and schemas-versioned
contracts (`JsonContract`) to ensure interoperability between runners and harnesses.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from worldfoundry.core.json_contract import JsonContract, copy_mapping, to_plain
from worldfoundry.evals.api import GenerationRequest, GenerationResult, WorldModelConfig

# Canonical schema version for all contracts in this module.
VLA_VA_WAM_CONTRACT_SCHEMA_VERSION = "worldfoundry-vla-va-wam-contract"

# --- Capability Constants ---
# These represent discrete functional requirements that a model runner must expose
# to be deemed capable of handling a given evaluation task or track.
CAPABILITY_VLA_ACTION_PREDICTION = "vla.action_prediction"
CAPABILITY_VLA_POLICY_ROLLOUT = "vla.policy_rollout"
CAPABILITY_VA_VIDEO_ACTION = "va.video_action"
CAPABILITY_VAM_VIDEO_ACTION_MODELING = "vam.video_action_modeling"
CAPABILITY_WAM_WORLD_ACTION_MODELING = "wam.world_action_modeling"
CAPABILITY_WAM_RESET = "wam.reset"
CAPABILITY_WAM_BRANCH = "wam.branch"
CAPABILITY_SESSION_CONTROL = "session.control"
CAPABILITY_MULTIMODAL_OBSERVATION = "observation.multimodal"

CAPABILITY_TAGS = (
    CAPABILITY_VLA_ACTION_PREDICTION,
    CAPABILITY_VLA_POLICY_ROLLOUT,
    CAPABILITY_VA_VIDEO_ACTION,
    CAPABILITY_VAM_VIDEO_ACTION_MODELING,
    CAPABILITY_WAM_WORLD_ACTION_MODELING,
    CAPABILITY_WAM_RESET,
    CAPABILITY_WAM_BRANCH,
    CAPABILITY_SESSION_CONTROL,
    CAPABILITY_MULTIMODAL_OBSERVATION,
)


class EvaluationTrack(str, Enum):
    """Categorization of evaluation tasks within WorldFoundry Embodied tasks."""

    VLA = "vla"  # Vision-Language-Action
    VA = "va"  # Video Action
    VAM = "vam"  # Video Action Modeling
    WAM = "wam"  # World Action Modeling (Environment Simulators)


class RequestKind(str, Enum):
    """Types of runtime requests dispatched to the model runner."""

    ACTION = "action"  # Predict immediate actions from step observation
    GENERATION = "generation"  # Generate sequence of future actions/frames
    ROLLOUT = "rollout"  # Execute closed-loop rollout inside a simulator
    RESET = "reset"  # Hard reset of environmental state
    BRANCH = "branch"  # Branch/Fork environmental state to start an alternative trajectory
    SESSION = "session"  # Session handshake and lifecycle management


class ActionSpaceKind(str, Enum):
    """Supported mathematical formats of the action prediction space."""

    DISCRETE = "discrete"  # Select from finite discrete actions (e.g., [left, right, grab])
    CONTINUOUS = "continuous"  # N-dimensional floating-point vectors (e.g., Delta-pose control)
    HYBRID = "hybrid"  # Mix of discrete choices and continuous parameters
    TEXT = "text"  # Natural language commands or symbolic expressions
    POSE = "pose"  # Position & Orientation (7D/6D vectors)
    JOINT = "joint"  # Direct joint angle or torque control vectors


JsonValue = Any
JsonDataclass = JsonContract
_to_plain = to_plain
_copy_mapping = copy_mapping
_EnumT = TypeVar("_EnumT", bound=Enum)


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Safely coerce iterable elements into a clean, immutable tuple of strings.

    Args:
        value: Any iterable or None.

    Returns:
        A tuple of stringified elements.
    """
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _coerce_enum(enum_cls: type[_EnumT], value: Any, field_name: str) -> _EnumT:
    """Coerces arbitrary string/value inputs into their proper Enum representations with strict checking.

    Args:
        enum_cls: The Target Enum class.
        value: Value to coerce.
        field_name: The name of the field being coerced for descriptive errors.

    Returns:
        The coerced Enum instance.

    Raises:
        ValueError: If value cannot be coerced into enum_cls.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"Unsupported {field_name}: {value!r}. Expected one of: {allowed}.") from exc


@dataclass(frozen=True)
class ActionSpaceSpec(JsonDataclass):
    """Specification of the action space of the robot/embodied agent.

    This contract dictates how the model runner should structure, clip, scale,
    and output predicted action tokens/vectors.
    """

    kind: ActionSpaceKind | str
    actions: tuple[str, ...] = ()
    dimensions: int | None = None
    bounds: Mapping[str, Any] = field(default_factory=dict)
    schema: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VLA_VA_WAM_CONTRACT_SCHEMA_VERSION
    __hash__ = JsonDataclass.__hash__

    def __post_init__(self) -> None:
        """Performs validation and type coercion after object initialization."""
        if self.schema_version != VLA_VA_WAM_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported ActionSpaceSpec schema_version: {self.schema_version}")
        # Coerce 'kind' to its ActionSpaceKind enum.
        kind = _coerce_enum(ActionSpaceKind, self.kind, "action_space.kind")
        object.__setattr__(self, "kind", kind)
        # Ensure 'actions' is an immutable tuple of strings.
        object.__setattr__(self, "actions", _tuple_of_str(self.actions))
        # Create immutable copies of nested mappings.
        object.__setattr__(self, "bounds", _copy_mapping(self.bounds))
        object.__setattr__(self, "schema", _copy_mapping(self.schema))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        # Validate specific constraints based on 'kind'.
        if kind == ActionSpaceKind.DISCRETE and not self.actions:
            raise ValueError("Discrete ActionSpaceSpec requires at least one action.")
        # Validate 'dimensions' if provided.
        if self.dimensions is not None and int(self.dimensions) < 1:
            raise ValueError("ActionSpaceSpec dimensions must be positive when provided.")
        # Ensure 'dimensions' is an integer if provided.
        if self.dimensions is not None:
            object.__setattr__(self, "dimensions", int(self.dimensions))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionSpaceSpec:
        """Instantiate ActionSpaceSpec from a raw dictionary mapping.

        Args:
            data: A dictionary containing initialization fields.

        Returns:
            An ActionSpaceSpec instance.
        """
        return cls(
            kind=data["kind"],
            actions=data.get("actions", ()),
            dimensions=data.get("dimensions"),
            bounds=cast(Mapping[str, Any], data.get("bounds")),
            schema=cast(Mapping[str, Any], data.get("schema")),
            metadata=cast(Mapping[str, Any], data.get("metadata")),
            schema_version=data.get("schema_version", VLA_VA_WAM_CONTRACT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class SessionControl(JsonDataclass):
    """Configures the execution session for stateful interactions (primarily WAM).

    Enables simulator reset, deterministic branching, state checkpointing, and
    horizon constraints.
    """

    reset_supported: bool = False
    branch_supported: bool = False
    session_id: str | None = None
    reset_seed: int | None = None
    branch_from_session_id: str | None = None
    branch_from_step: int | None = None
    max_session_steps: int | None = None
    deterministic_reset: bool = False
    state_checkpoint_format: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VLA_VA_WAM_CONTRACT_SCHEMA_VERSION
    __hash__ = JsonDataclass.__hash__

    def __post_init__(self) -> None:
        """Performs validation and type coercion after object initialization."""
        if self.schema_version != VLA_VA_WAM_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported SessionControl schema_version: {self.schema_version}")
        # Coerce boolean fields to ensure they are bool types.
        object.__setattr__(self, "reset_supported", bool(self.reset_supported))
        object.__setattr__(self, "branch_supported", bool(self.branch_supported))
        object.__setattr__(self, "deterministic_reset", bool(self.deterministic_reset))
        # Create immutable copy of metadata.
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        # Validate and coerce integer fields if provided.
        if self.branch_from_step is not None and int(self.branch_from_step) < 0:
            raise ValueError("SessionControl branch_from_step must be non-negative when provided.")
        if self.max_session_steps is not None and int(self.max_session_steps) < 1:
            raise ValueError("SessionControl max_session_steps must be positive when provided.")
        if self.reset_seed is not None:
            object.__setattr__(self, "reset_seed", int(self.reset_seed))
        if self.branch_from_step is not None:
            object.__setattr__(self, "branch_from_step", int(self.branch_from_step))
        if self.max_session_steps is not None:
            object.__setattr__(self, "max_session_steps", int(self.max_session_steps))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionControl:
        """Instantiate SessionControl from a raw dictionary mapping.

        Args:
            data: A dictionary containing initialization fields.

        Returns:
            A SessionControl instance.
        """
        return cls(
            reset_supported=data.get("reset_supported", False),
            branch_supported=data.get("branch_supported", False),
            session_id=data.get("session_id"),
            reset_seed=data.get("reset_seed"),
            branch_from_session_id=data.get("branch_from_session_id"),
            branch_from_step=data.get("branch_from_step"),
            max_session_steps=data.get("max_session_steps"),
            deterministic_reset=data.get("deterministic_reset", False),
            state_checkpoint_format=data.get("state_checkpoint_format", ""),
            metadata=cast(Mapping[str, Any], data.get("metadata")),
            schema_version=data.get("schema_version", VLA_VA_WAM_CONTRACT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class EmbodiedGenerationSpec(JsonDataclass):
    """Specification schema governing a sequence of physical generation requests.

    Defines the track, task context, capabilities, and spatial-temporal constraints (horizon steps)
    which must be satisfied during runner execution.
    """

    track: EvaluationTrack | str
    kind: RequestKind | str
    task_name: str
    action_space: ActionSpaceSpec | Mapping[str, Any]
    observation_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ("actions",)
    required_capabilities: tuple[str, ...] = ()
    session_control: SessionControl | Mapping[str, Any] | None = None
    horizon_steps: int | None = None
    controls: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VLA_VA_WAM_CONTRACT_SCHEMA_VERSION
    __hash__ = JsonDataclass.__hash__

    def __post_init__(self) -> None:
        """Performs validation and type coercion after object initialization."""
        if self.schema_version != VLA_VA_WAM_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported EmbodiedGenerationSpec schema_version: {self.schema_version}")

        # Coerce track and kind to their respective Enum types.
        track = _coerce_enum(EvaluationTrack, self.track, "track")
        kind = _coerce_enum(RequestKind, self.kind, "kind")

        # Instantiate ActionSpaceSpec from dict if necessary.
        action_space = (
            self.action_space
            if isinstance(self.action_space, ActionSpaceSpec)
            else ActionSpaceSpec.from_dict(self.action_space)
        )
        # Instantiate SessionControl from dict if necessary.
        session_control = self.session_control
        if isinstance(session_control, Mapping):
            session_control = SessionControl.from_dict(session_control)

        # Collect and normalize required capabilities, including defaults for the track.
        required_capabilities = set(_tuple_of_str(self.required_capabilities))
        required_capabilities.update(default_capabilities_for_track(track))

        # Add session control capabilities if supported and applicable for WAM track.
        if track == EvaluationTrack.WAM and session_control is not None:
            if session_control.reset_supported:
                required_capabilities.add(CAPABILITY_WAM_RESET)
            if session_control.branch_supported:
                required_capabilities.add(CAPABILITY_WAM_BRANCH)
            if session_control.reset_supported or session_control.branch_supported:
                required_capabilities.add(CAPABILITY_SESSION_CONTROL)

        # Set attributes immutably after validation and coercion.
        object.__setattr__(self, "track", track)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "task_name", str(self.task_name))
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "observation_keys", _tuple_of_str(self.observation_keys))
        object.__setattr__(self, "output_keys", _tuple_of_str(self.output_keys))
        object.__setattr__(self, "required_capabilities", tuple(sorted(required_capabilities)))
        object.__setattr__(self, "session_control", session_control)
        object.__setattr__(self, "controls", _copy_mapping(self.controls))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

        # Perform final data validation checks.
        if not self.task_name:
            raise ValueError("EmbodiedGenerationSpec requires task_name.")
        if self.horizon_steps is not None and int(self.horizon_steps) < 1:
            raise ValueError("EmbodiedGenerationSpec horizon_steps must be positive when provided.")
        # Ensure 'horizon_steps' is an integer if provided.
        if self.horizon_steps is not None:
            object.__setattr__(self, "horizon_steps", int(self.horizon_steps))
        if track == EvaluationTrack.VLA and not self.observation_keys:
            raise ValueError("VLA EmbodiedGenerationSpec requires observation_keys.")
        wam_kinds = {
            RequestKind.GENERATION,
            RequestKind.ROLLOUT,
            RequestKind.RESET,
            RequestKind.BRANCH,
            RequestKind.SESSION,
        }
        if track == EvaluationTrack.WAM and kind not in wam_kinds:
            raise ValueError("WAM EmbodiedGenerationSpec requires a WAM-compatible request kind.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbodiedGenerationSpec:
        """Instantiate EmbodiedGenerationSpec from a raw dictionary mapping.

        Args:
            data: A dictionary containing initialization fields.

        Returns:
            An EmbodiedGenerationSpec instance.
        """
        return cls(
            track=data["track"],
            kind=data["kind"],
            task_name=data["task_name"],
            action_space=data["action_space"],
            observation_keys=data.get("observation_keys", ()),
            output_keys=data.get("output_keys", ("actions",)),
            required_capabilities=data.get("required_capabilities", ()),
            session_control=data.get("session_control"),
            horizon_steps=data.get("horizon_steps"),
            controls=cast(Mapping[str, Any], data.get("controls")),
            metadata=cast(Mapping[str, Any], data.get("metadata")),
            schema_version=data.get("schema_version", VLA_VA_WAM_CONTRACT_SCHEMA_VERSION),
        )

    def to_generation_request(
        self,
        *,
        sample_id: str,
        split: str = "default",
        request_id: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> GenerationRequest:
        """Converts the high-level Embodied AI contract into an abstract evaluation GenerationRequest.

        Ensures that downstream model runners receive all required controls and validation schemata.

        Args:
            sample_id: A unique identifier for the sample being processed.
            split: The dataset split (e.g., "train", "validation", "test"). Defaults to "default".
            request_id: An optional unique identifier for the specific generation request.
            inputs: A mapping of input data required for generation (e.g., observations).
            generation_kwargs: Additional keyword arguments for the generation process.

        Returns:
            A GenerationRequest instance encapsulating the task details.
        """
        return GenerationRequest(
            sample_id=sample_id,
            task_name=self.task_name,
            split=split,
            request_id=request_id,
            inputs=inputs,
            controls=self.to_dict(),  # Serialize the full spec into controls for the runner.
            generation_kwargs=generation_kwargs,
            output_schema={key: {} for key in self.output_keys},
        )


@dataclass(frozen=True)
class RunnerCapabilities(JsonDataclass):
    """Manifest of the hardware and architectural features supported by a specific Model Runner.

    Before executing a benchmark, the evaluation harness requests this manifest from the runner
    to verify that the runner possesses the required capabilities and action spaces.
    """

    model_id: str
    tracks: tuple[EvaluationTrack | str, ...] = ()
    capabilities: tuple[str, ...] = ()
    action_spaces: tuple[ActionSpaceSpec | Mapping[str, Any], ...] = ()
    max_horizon_steps: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VLA_VA_WAM_CONTRACT_SCHEMA_VERSION
    __hash__ = JsonDataclass.__hash__

    def __post_init__(self) -> None:
        """Performs validation and type coercion after object initialization."""
        if self.schema_version != VLA_VA_WAM_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported RunnerCapabilities schema_version: {self.schema_version}")
        # Coerce track names to EvaluationTrack enums.
        tracks = tuple(_coerce_enum(EvaluationTrack, track, "tracks") for track in self.tracks)
        # Instantiate ActionSpaceSpec for each entry if needed.
        action_spaces = tuple(
            spec if isinstance(spec, ActionSpaceSpec) else ActionSpaceSpec.from_dict(spec)
            for spec in (self.action_spaces or ())
        )
        # Set attributes immutably after validation and coercion.
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "tracks", tracks)
        # Normalize capabilities to a sorted, unique tuple of strings.
        object.__setattr__(self, "capabilities", tuple(sorted(set(_tuple_of_str(self.capabilities)))))
        object.__setattr__(self, "action_spaces", action_spaces)
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        # Validate and coerce 'max_horizon_steps' if provided.
        if self.max_horizon_steps is not None and int(self.max_horizon_steps) < 1:
            raise ValueError("RunnerCapabilities max_horizon_steps must be positive when provided.")
        if self.max_horizon_steps is not None:
            object.__setattr__(self, "max_horizon_steps", int(self.max_horizon_steps))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunnerCapabilities:
        """Instantiate RunnerCapabilities from a raw dictionary mapping.

        Args:
            data: A dictionary containing initialization fields.

        Returns:
            A RunnerCapabilities instance.
        """
        return cls(
            model_id=data["model_id"],
            tracks=data.get("tracks", ()),
            capabilities=data.get("capabilities", ()),
            action_spaces=data.get("action_spaces", ()),
            max_horizon_steps=data.get("max_horizon_steps"),
            metadata=cast(Mapping[str, Any], data.get("metadata")),
            schema_version=data.get("schema_version", VLA_VA_WAM_CONTRACT_SCHEMA_VERSION),
        )


def default_capabilities_for_track(track: EvaluationTrack | str) -> tuple[str, ...]:
    """Retrieves standard capability tags associated with a specific evaluation track.

    Args:
        track: The evaluation track (e.g., EvaluationTrack.VLA).

    Returns:
        A tuple of capability strings.
    """
    resolved = _coerce_enum(EvaluationTrack, track, "track")
    if resolved == EvaluationTrack.VLA:
        return (CAPABILITY_VLA_ACTION_PREDICTION,)
    if resolved == EvaluationTrack.VA:
        return (CAPABILITY_VA_VIDEO_ACTION,)
    if resolved == EvaluationTrack.VAM:
        return (CAPABILITY_VAM_VIDEO_ACTION_MODELING,)
    if resolved == EvaluationTrack.WAM:
        return (CAPABILITY_WAM_WORLD_ACTION_MODELING,)
    return ()


@runtime_checkable
class VlaVaWamRunnerContract(Protocol):
    """Protocol defining the structural interface that a Model Runner must implement.

    Any model runner participating in VLA, VA, VAM, or WAM evaluations must satisfy this protocol
    to interface with the `worldfoundry` task framework.
    """

    model_id: str
    capabilities: Collection[str]

    @classmethod
    def from_config(cls, config: WorldModelConfig) -> VlaVaWamRunnerContract:
        """Instantiate the runner asynchronously or synchronously using a WorldModelConfig.

        Args:
            config: Configuration object for the world model.

        Returns:
            An instance of the VlaVaWamRunnerContract implementing class.
        """
        ...

    def describe_capabilities(self) -> RunnerCapabilities:
        """Query the runner's supported evaluation tracks, actions spaces, and limitations.

        Returns:
            A RunnerCapabilities object detailing what the runner can do.
        """
        ...

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationResult]:
        """Core batch-generation execution point.

        Accepts unified GenerationRequests containing spatial observations and environment controls,
        returning physical actions or simulation transitions formatted as GenerationResults.

        Args:
            requests: A sequence of GenerationRequest objects to be processed.

        Returns:
            A sequence of GenerationResult objects corresponding to the inputs.
        """
        ...

    def cleanup(self) -> None:
        """Release underlying system resources (e.g., PyTorch CUDA cache, virtual displays, simulators)."""
        ...


__all__ = [
    "CAPABILITY_MULTIMODAL_OBSERVATION",
    "CAPABILITY_SESSION_CONTROL",
    "CAPABILITY_TAGS",
    "CAPABILITY_VAM_VIDEO_ACTION_MODELING",
    "CAPABILITY_VA_VIDEO_ACTION",
    "CAPABILITY_VLA_ACTION_PREDICTION",
    "CAPABILITY_VLA_POLICY_ROLLOUT",
    "CAPABILITY_WAM_BRANCH",
    "CAPABILITY_WAM_RESET",
    "CAPABILITY_WAM_WORLD_ACTION_MODELING",
    "VLA_VA_WAM_CONTRACT_SCHEMA_VERSION",
    "ActionSpaceKind",
    "ActionSpaceSpec",
    "EmbodiedGenerationSpec",
    "EvaluationTrack",
    "RequestKind",
    "RunnerCapabilities",
    "SessionControl",
    "VlaVaWamRunnerContract",
    "default_capabilities_for_track",
]

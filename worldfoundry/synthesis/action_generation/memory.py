"""
Provides a bounded memory store for action-generation traces,
along with factory functions to create specialized memory classes
for various robot and agent policies.

This module defines `ActionTraceMemory` as a base class for storing
and retrieving action-related data, such as trajectories, action sequences,
or policy rollouts. It also includes a pattern for dynamically
creating subclasses with predefined model-specific attributes,
ensuring consistent metadata tagging for different policy models.
"""

from __future__ import annotations

from typing import Any

from worldfoundry.core.memory import BaseMemory


class ActionTraceMemory(BaseMemory):
    """
    Bounded memory store for action-generation traces.

    This class extends `BaseMemory` to provide a specific implementation
    for storing and retrieving action traces, such as those generated
    by policies or models. It allows for capacity management and
    metadata tagging specific to the generating model.
    """

    MODEL_ID = ""
    POLICY_FAMILY = ""
    DEFAULT_TYPE = "action_trace"
    STORAGE_ATTR = "records"

    def __init__(self, capacity: int | None = None, **kwargs: Any):
        """
        Initialize the model-scoped trace buffer.

        Args:
            capacity: Optional maximum number of trace records to keep. If None, capacity is unbounded.
            **kwargs: Reserved for pipeline-specific construction metadata, passed to BaseMemory.
        """
        super().__init__(capacity=capacity, **kwargs)
        self.storage_attr = self.STORAGE_ATTR

    @property
    def records(self) -> list[dict[str, Any]]:
        """
        Return the concrete trace list used by the public memory class.

        This property exposes the underlying storage of records.
        """
        return self.storage

    @property
    def storage_attr_records(self) -> list[dict[str, Any]]:
        """
        Provides direct access to the records, primarily for consistency with `STORAGE_ATTR`.
        """
        return self.records

    def __getattr__(self, name: str) -> Any:
        """
        Custom attribute access to route `STORAGE_ATTR` to the `records` property.

        This allows instances to access their stored records via the `STORAGE_ATTR` name
        (e.g., `memory.records` or `memory.act_chunks`) dynamically.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the requested attribute.

        Raises:
            AttributeError: If the attribute does not exist and is not `STORAGE_ATTR`.
        """
        # If the requested attribute name matches the configured STORAGE_ATTR,
        # return the records property. This provides a flexible way to access
        # the stored data using a dynamic name.
        if name == self.STORAGE_ATTR:
            return self.records
        raise AttributeError(name)

    def record(self, data: Any, metadata: dict[str, Any] | None = None, **kwargs: Any):
        """
        Append one action trace entry with model metadata.

        The entry will include `model_id` and `policy_family` from the class
        attributes, merged with any provided `metadata`.

        Args:
            data: Action payload or rollout data to store.
            metadata: Optional dictionary of additional metadata to merge into the stored entry.
            **kwargs: Reserved call-site metadata accepted for compatibility, but ignored.
        """
        del kwargs  # kwargs are explicitly ignored for this method
        # Construct the metadata for the record, prioritizing class-level defaults
        # and then merging any provided metadata.
        entry_metadata = {
            "model_id": self.MODEL_ID,
            "policy_family": self.POLICY_FAMILY,
            **dict(metadata or {}),
        }
        self.append_record(
            data,
            kind=str(entry_metadata.get("type", self.DEFAULT_TYPE)),
            metadata=entry_metadata,
        )

    def select(self, context_query: Any = None, prefer_type: str | None = None, **kwargs: Any):
        """
        Select the latest trace, optionally constrained by stored type.

        This method retrieves the content of the most recent record. If `prefer_type`
        is specified, it searches for the latest record of that specific type.

        Args:
            context_query: Accepted for a uniform memory interface, but ignored for this implementation.
            prefer_type: Optional record type to search from newest to oldest.
            **kwargs: Reserved selector metadata accepted for compatibility, but ignored.

        Returns:
            The content of the latest record, or `None` if no record is found.
        """
        del context_query, kwargs  # context_query and kwargs are explicitly ignored
        # Retrieve the latest record, optionally filtering by a preferred type.
        record = self.latest_record() if prefer_type is None else self.latest_record(prefer_type=prefer_type)
        return record["content"] if record is not None else None

    def manage(self, action: str = "reset", **kwargs: Any):
        """
        Reset or evict records according to the configured capacity.

        Allows for managing the memory content by either clearing all records
        or applying eviction rules if a capacity is set.

        Args:
            action: Lifecycle command; supports ``reset`` (clear all records)
                    and ``evict`` (apply capacity limits).
            **kwargs: Reserved lifecycle metadata accepted for compatibility, but ignored.
        """
        del kwargs  # kwargs are explicitly ignored for this method
        # Perform the specified memory management action.
        if action == "reset":
            self.reset_records()
            return
        # Evict records only if an eviction action is specified and a capacity is set.
        if action == "evict" and self.capacity is not None:
            self._store.evict()


def _action_memory_class(
    name: str,
    *,
    model_id: str,
    policy_family: str,
    default_type: str,
    storage_attr: str,
    doc: str,
) -> type[ActionTraceMemory]:
    """
    Factory function to create a new `ActionTraceMemory` subclass with predefined attributes.

    This function dynamically generates a new class, inheriting from `ActionTraceMemory`,
    and sets its `MODEL_ID`, `POLICY_FAMILY`, `DEFAULT_TYPE`, `STORAGE_ATTR`, and `__doc__`
    attributes based on the provided arguments. This allows for creating specialized
    memory types for different models without repetitive class definitions.

    Args:
        name: The name of the new class.
        model_id: The identifier for the model associated with this memory.
        policy_family: The family or category of the policy.
        default_type: The default type string for records stored in this memory.
        storage_attr: The attribute name used to access records (e.g., 'act_chunks').
        doc: The docstring for the new class.

    Returns:
        A new subclass of `ActionTraceMemory`.
    """
    return type(
        name,
        (ActionTraceMemory,),
        {
            "__doc__": doc,
            "__module__": __name__,
            "MODEL_ID": model_id,
            "POLICY_FAMILY": policy_family,
            "DEFAULT_TYPE": default_type,
            "STORAGE_ATTR": storage_attr,
        },
    )


# --- Specialized ActionTraceMemory Subclasses for various policies ---

ACTMemory = _action_memory_class(
    "ACTMemory",
    model_id="act",
    policy_family="action_chunking_transformer",
    default_type="act_chunked_action_sequence",
    storage_attr="act_chunks",
    doc="ACT chunked-action memory.",
)
BeingH05Memory = _action_memory_class(
    "BeingH05Memory",
    model_id="being-h05",
    policy_family="cross_embodiment_vla",
    default_type="being_h05_action_trace",
    storage_attr="being_h05_traces",
    doc="Being-H0.5 cross-embodiment action memory.",
)
DiffusionPolicyMemory = _action_memory_class(
    "DiffusionPolicyMemory",
    model_id="diffusion-policy",
    policy_family="visuomotor_diffusion_policy",
    default_type="diffusion_policy_action_trajectory",
    storage_attr="diffusion_policy_trajectories",
    doc="Diffusion Policy trajectory memory.",
)
DreamZeroMemory = _action_memory_class(
    "DreamZeroMemory",
    model_id="dreamzero",
    policy_family="world_action_model",
    default_type="dreamzero_world_action_trace",
    storage_attr="dreamzero_rollouts",
    doc="DreamZero world-action rollout memory.",
)
GigaBrain0Memory = _action_memory_class(
    "GigaBrain0Memory",
    model_id="giga-brain-0",
    policy_family="world_model_powered_vla",
    default_type="giga_brain_0_action_trace",
    storage_attr="giga_brain_0_traces",
    doc="GigaBrain-0 action prediction memory scoped to one pipeline instance.",
)
GR00TMemory = _action_memory_class(
    "GR00TMemory",
    model_id="gr00t",
    policy_family="humanoid_foundation_policy",
    default_type="gr00t_action_trace",
    storage_attr="gr00t_action_history",
    doc="GR00T humanoid action memory.",
)
LAPAMemory = _action_memory_class(
    "LAPAMemory",
    model_id="lapa",
    policy_family="visual_action_model",
    default_type="lapa_action_tokens",
    storage_attr="lapa_token_history",
    doc="LAPA latent action token memory.",
)
LingBotVAMemory = _action_memory_class(
    "LingBotVAMemory",
    model_id="lingbot-va",
    policy_family="video_action_world_model",
    default_type="lingbot_va_video_action_trace",
    storage_attr="lingbot_va_chunks",
    doc="LingBot-VA video-action memory.",
)
MolmoAct2Memory = _action_memory_class(
    "MolmoAct2Memory",
    model_id="molmoact2",
    policy_family="flow_matching_action_reasoning_vla",
    default_type="molmoact2_action_trace",
    storage_attr="molmoact2_traces",
    doc="MolmoAct2 action prediction memory scoped to one pipeline instance.",
)
OctoMemory = _action_memory_class(
    "OctoMemory",
    model_id="octo",
    policy_family="generalist_robot_transformer",
    default_type="octo_action_chunk",
    storage_attr="octo_action_windows",
    doc="Octo action-window memory.",
)
OpenPIMemory = _action_memory_class(
    "OpenPIMemory",
    model_id="openpi",
    policy_family="flow_matching_vla",
    default_type="openpi_action_chunk",
    storage_attr="openpi_rollouts",
    doc="OpenPI/pi0 action-chunk memory scoped to one pipeline instance.",
)
OpenVLAMemory = _action_memory_class(
    "OpenVLAMemory",
    model_id="openvla",
    policy_family="autoregressive_vla",
    default_type="openvla_action_trace",
    storage_attr="openvla_records",
    doc="OpenVLA rollout memory scoped to one pipeline instance.",
)
RoboFlamingoMemory = _action_memory_class(
    "RoboFlamingoMemory",
    model_id="roboflamingo",
    policy_family="flamingo_vlm_robot_policy",
    default_type="roboflamingo_eef_action",
    storage_attr="roboflamingo_actions",
    doc="RoboFlamingo VLM policy memory.",
)
RT1Memory = _action_memory_class(
    "RT1Memory",
    model_id="rt-1",
    policy_family="robotics_transformer",
    default_type="rt1_discrete_action_tokens",
    storage_attr="rt1_token_steps",
    doc="RT-1 discrete action token memory.",
)
StarVLAMemory = _action_memory_class(
    "StarVLAMemory",
    model_id="starvla",
    policy_family="vla_wam",
    default_type="starvla_embodied_action_trace",
    storage_attr="starvla_segments",
    doc="StarVLA embodied action memory.",
)


__all__ = [
    "ACTMemory",
    "ActionTraceMemory",
    "BeingH05Memory",
    "DiffusionPolicyMemory",
    "DreamZeroMemory",
    "GR00TMemory",
    "GigaBrain0Memory",
    "LAPAMemory",
    "LingBotVAMemory",
    "MolmoAct2Memory",
    "OctoMemory",
    "OpenPIMemory",
    "OpenVLAMemory",
    "RT1Memory",
    "RoboFlamingoMemory",
    "StarVLAMemory",
]

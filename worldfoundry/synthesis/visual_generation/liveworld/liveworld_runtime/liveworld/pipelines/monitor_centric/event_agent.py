"""Event agent implementation.

EventAgent manages the lifecycle of a single event: prompt evolution and state.
Detection is handled separately by DetectorAgent.
Video generation is handled by the pipeline via _evolve_event_with_observer().

The evolution logic is simple "continuation":
- First iteration: use the detected entity's initial state/action
- Subsequent iterations: continue from the last frame of the previous video
"""

from __future__ import annotations

from dataclasses import dataclass

from .event_types import EventID, EventScript, EventState


@dataclass
class EventPrompts:
    """Prompt configuration for event evolution.

    These prompts are shared across all EventAgents via EventPool.
    """

    # System prompt template for generating I2V prompts.
    # Available variables: {entity}, {entities}, {iteration}
    system_prompt: str


class EventAgent:
    """Event agent that manages prompt evolution and state for a single event.

    Evolution logic (simple continuation):
    - Uses current_anchor_frame as the starting point
    - Generates a continuation prompt based on the entity and iteration

    Video generation is handled by the pipeline via _evolve_event_with_observer().
    Detection is NOT handled here - use DetectorAgent for that.
    """

    def __init__(
        self,
        event_id: EventID,
        state: EventState,
        device: str,
        prompts: EventPrompts,
        default_horizon: int,
        default_fps: int,
    ) -> None:
        if state is None:
            raise ValueError("EventAgent requires a non-null state")
        if prompts is None:
            raise ValueError("prompts must be provided")
        if state.current_anchor_frame is None:
            raise ValueError("state.current_anchor_frame must be set before creating EventAgent")

        self.event_id = event_id
        self.state = state
        self.device = device
        self.prompts = prompts
        self.default_horizon = default_horizon
        self.default_fps = default_fps

    def evolve_prompt(self) -> EventScript:
        """Generate a continuation prompt for the next evolution step.

        The prompt is generated from the system_prompt template using entity name
        and iteration count. For now, this is simple continuation.
        """
        entity = self.state.entities[0] if self.state.entities else "object"
        entities = ", ".join(self.state.entities) if self.state.entities else "object"
        iteration = self.state.iteration_count

        prompt_text = self.prompts.system_prompt.format(
            entity=entity,
            entities=entities,
            iteration=iteration,
        )

        return EventScript(
            event_id=self.event_id,
            text=prompt_text,
            horizon=self.default_horizon,
            fps=self.default_fps,
        )

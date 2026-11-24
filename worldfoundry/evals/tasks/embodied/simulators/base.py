"""Module defining the base interfaces for synchronous simulators in the WorldFoundry framework.

This module provides the core `BaseSimulator` abstract base class, which all
simulated benchmarks must inherit from to be integrated into the evaluation system.
It also includes utility classes like `StepResult` for standardized step outcomes
and `DummyRecorder` for basic tracking functionality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    """Dataclass capturing the outcome of a single step execution in the simulator.

    Attributes:
        obs: The raw sensory observation returned by the environment.
        reward: The immediate scalar reward obtained.
        done: A boolean flag indicating whether the episode has terminated.
        info: Additional diagnostic information dictionary.
    """

    obs: Any
    reward: float
    done: bool
    info: dict[str, Any]


class DummyRecorder:
    """A placeholder recorder that performs no operations when tracking is disabled.

    This class serves as a no-op implementation for recording functionality,
    allowing simulators to call recording methods without needing to check
    if a recorder is active.
    """

    _next_step: int = 0

    def record_step(self, *args: Any, **kwargs: Any) -> None:
        """Mock method for recording a step. Does nothing."""

    def record_video(self, *args: Any, **kwargs: Any) -> None:
        """Mock method for recording a video. Does nothing."""


class BaseSimulator(ABC):
    """Native WorldFoundry Sync Simulator Interface.

    All simulated benchmarks must inherit from this class to participate in
    closed-loop evaluation within the WorldFoundry framework. This ABC defines
    the standard API for interacting with a simulated environment.
    """

    def __init__(self) -> None:
        """Initialize the base simulator state.

        Sets up default values for the last step result, current task,
        and an inactive recorder.
        """
        self._last_result: StepResult = StepResult(obs=None, reward=0.0, done=False, info={})
        self._task: dict[str, Any] = {}
        self._recorder = DummyRecorder()

    @abstractmethod
    def reset(self, task: dict[str, Any]) -> Any:
        """Reset the simulator for a new task episode.

        This method should initialize the environment to a starting state
        based on the provided task parameters and return the initial raw observation.

        Args:
            task: A dictionary specifying task parameters, seed, indices, etc.

        Returns:
            The initial raw observation from the environment.
        """

    @abstractmethod
    def step(self, action: dict[str, Any]) -> StepResult:
        """Execute a physical action in the simulator.

        This method applies the given action to the environment, advances the simulation
        by one step, and returns the outcome of that step.

        Args:
            action: A dictionary representing the predicted control action.

        Returns:
            A StepResult containing the new observation, reward, done flag, and info.
        """

    @abstractmethod
    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw simulator observation into a structured, track-standard observation dict.

        This method is responsible for transforming the environment's native observation
        format into a standardized dictionary suitable for model input.

        Args:
            raw_obs: The raw, environment-specific observation.
            task: Task description mapping, potentially useful for observation processing.

        Returns:
            A standard observation dictionary containing images, proprioception, etc.
        """

    @abstractmethod
    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Construct the final evaluation scorecard/metrics from a step result.

        This method should compute and return a dictionary of relevant metrics
        for the given step or the entire episode if called at termination.

        Args:
            step_result: The last StepResult of the episode.

        Returns:
            A dictionary of evaluation metrics (e.g. success, reward, path length).
        """

    def check_done(self, step_result: StepResult) -> bool:
        """Check whether the episode should terminate based on the latest step result.

        This default implementation simply returns the `done` flag from the `StepResult`.
        Subclasses can override this to implement custom termination logic.

        Args:
            step_result: The current StepResult.

        Returns:
            True if the episode is done, False otherwise.
        """
        return step_result.done

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve additional simulator metadata (e.g., max_steps, frame_rate).

        This method can provide general information about the simulator or task
        that might be useful for evaluation or analysis.

        Returns:
            A dictionary containing metadata. Defaults to an empty dictionary.
        """
        return {}

    def cleanup(self) -> None:
        """Release any allocated simulator resources, close windows, or clear GPU caches.

        This method should be called at the end of an evaluation run to ensure
        proper resource management and avoid memory leaks. Defaults to a no-op.
        """
        return None

    def get_action_spec(self) -> dict[str, Any]:
        """Get the expected action specification for the simulator.

        This method should return a description of the valid actions, including
        their dimensions, types, and ranges.

        Returns:
            A dictionary mapping control fields to DimSpec constraints. Defaults to an empty dictionary.
        """
        return {}

    def get_observation_spec(self) -> dict[str, Any]:
        """Get the observation specification provided by the simulator.

        This method should return a description of the observations produced by
        `make_obs`, including their dimensions, types, and expected ranges.

        Returns:
            A dictionary mapping sensory keys to DimSpec constraints. Defaults to an empty dictionary.
        """
        return {}

    def get_metric_keys(self) -> dict[str, str]:
        """Get the metric aggregation mapping (metric_id -> aggregation_fn).

        This method defines how different metrics returned by `get_step_result`
        should be aggregated across multiple episodes (e.g., 'mean', 'sum', 'last').

        Returns:
            A dictionary mapping metric keys to aggregation methods.
            Defaults to {"success": "mean"}.
        """
        return {"success": "mean"}

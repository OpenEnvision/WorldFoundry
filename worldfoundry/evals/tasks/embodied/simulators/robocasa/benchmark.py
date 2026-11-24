"""RoboCasa benchmark implementation.

RoboCasa is a large-scale simulation framework for kitchen manipulation
tasks built on top of robosuite v2 and MuJoCo.  It provides 365 tasks
(atomic + composite) across 2500+ procedurally-generated kitchen scenes.

Actions are 7-D by default when using the ``PandaOmron`` robot with a
standard ``OSC_POSE`` composite controller: ``[dx, dy, dz, drx, dry, drz,
gripper]``.

Observations expose RGB images from configurable cameras (default:
``robot0_agentview_left`` and ``robot0_eye_in_hand``) plus a natural
language task description obtained via ``env.get_ep_meta()["lang"]``.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import (
    GRIPPER_RAW,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_EULER,
    DimSpec,
)

# Set the MuJoCo rendering backend to EGL, which is suitable for headless servers
# and often performs better than GLFW.
os.environ.setdefault("MUJOCO_GL", "egl")

# Subset of atomic tasks suitable for quick evaluation.
DEFAULT_TASKS = [
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToSink",
    "OpenSingleDoor",
    "CloseDoubleDoor",
    "TurnOnSinkFaucet",
    "PreheatOven",
]


class RoboCasaBenchmark(BaseSimulator):
    """A wrapper for the RoboCasa kitchen manipulation benchmark environments.

    This class interfaces with RoboCasa environments, managing their creation,
    resetting, stepping, and observation processing for evaluation purposes.

    Attributes:
        _task_names (list[str]): List of RoboCasa environment names to evaluate.
        _robot (str): Robot model name used in the simulation.
        _camera_names (list[str]): Names of cameras to retrieve observations from.
        _camera_size (int): Resolution (width and height) for camera images.
        _max_steps (int): Maximum number of simulation steps per episode.
        _split (str): Dataset split used for environment generation ("pretrain" or "target").
        _seed (int | None): Random seed for reproducible environment creation.
        _env (Any): The active RoboCasa environment instance.
        _current_task (str | None): The name of the task currently loaded in `_env`.
        _lang (str): The natural language description of the current task.
    """

    # Defines the set of fields that should be recorded by the recorder.
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "PandaOmron",
        camera_names: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int = 500,
        split: str = "pretrain",
        seed: int | None = None,
    ) -> None:
        """Initializes the RoboCasaBenchmark simulator.

        Args:
            tasks: List of RoboCasa environment names to evaluate.
                Defaults to a small atomic-task subset.
            robot: Robot model name (default ``"PandaOmron"``).
            camera_names: Camera names for observations. Defaults to
                ``["robot0_agentview_left", "robot0_eye_in_hand"]``.
            camera_size: Camera resolution (square, default 256).
            max_steps: Maximum steps per episode (default 500).
            split: Dataset split — ``"pretrain"`` or ``"target"``.
            seed: Random seed for environment creation.
        """
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._robot = robot
        self._camera_names = camera_names or [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ]
        self._camera_size = camera_size
        self._max_steps = max_steps
        self._split = split
        self._seed = seed
        self._env: Any = None
        self._current_task: str | None = None
        self._lang: str = ""

    def cleanup(self) -> None:
        """Safely close the RoboCasa simulation environment.

        If an environment is active, it attempts to close it to release resources.
        """
        if self._env is not None:
            # Catch any errors during environment closing to prevent crashes
            # but ensure the environment reference is cleared.
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None

    def get_tasks(self) -> list[dict[str, Any]]:
        """Returns a list of available tasks for this simulator.

        Returns:
            A list of dictionaries, where each dictionary represents a task
            and contains a 'name' key.
        """
        return [{"name": t} for t in self._task_names]

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the RoboCasa environment to start a new episode for a given task.

        If the environment is not yet created or the task has changed, a new
        environment instance is created.

        Args:
            task: A dictionary specifying the task to reset to, typically with a 'name' key.

        Returns:
            The initial observation from the environment after reset.
        """
        create_env = __import__("robocasa.utils.env_utils", fromlist=["create_env"]).create_env

        task_name = task["name"]

        # If no environment is active, or the requested task is different from the current one,
        # close the existing environment (if any) and create a new one for the specified task.
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                # Close the old environment if a different task is being loaded
                self._env.close()
            # Create a new RoboCasa environment instance
            self._env = create_env(
                env_name=task_name,
                robots=self._robot,
                camera_names=self._camera_names,
                camera_widths=self._camera_size,
                camera_heights=self._camera_size,
                render_onscreen=False,
                split=self._split,
                seed=self._seed,
            )
            self._current_task = task_name

        obs = self._env.reset()
        # Retrieve the natural language task description from environment metadata
        self._lang = self._env.get_ep_meta().get("lang", task_name)
        # Record the initial frame of the episode for video logging.
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Executes a single step in the simulation using the provided action.

        Processes the action, adapts its dimensions to the environment's requirements,
        and performs a simulation step. Records video frames and step metrics.

        Args:
            action: A dictionary containing the action to perform. Expected to have
                an "actions" or "action" key with a 7-dimensional numpy array or list.

        Returns:
            A StepResult named tuple containing the observation, reward, done status,
            and additional information after the step.
        """
        # Extract the raw action from the input dictionary, preferring "actions" then "action".
        # If neither is found, default to a 7-dimensional zero array.
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(7)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        assert raw_action.shape[-1] == 7, f"dict[str, Any] dimension mismatch: got {raw_action.shape[-1]}, expected 7"

        # Get the expected action dimension from the RoboCasa environment
        act_dim = self._env.action_spec[0].shape[0]
        # Pad or truncate the raw action to match the environment's action dimension.
        # This ensures the action sent to the environment has the correct size.
        if raw_action.shape[0] < act_dim:
            raw_action = np.concatenate([raw_action, np.zeros(act_dim - raw_action.shape[0])])
        elif raw_action.shape[0] > act_dim:
            raw_action = raw_action[:act_dim]

        obs, reward, done, info = self._env.step(raw_action)
        # Determine if the task was successfully completed in this step.
        success = bool(self._env._check_success())
        info["success"] = success
        # Record the current frame for video logging.
        self._recorder.record_video(self._extract_frame(obs))
        # Record step metrics, using 1.0 for success and 0.0 otherwise as the reward
        # for evaluation consistency, while also logging the environment's internal success.
        self._recorder.record_step(reward=float(success), done=bool(done), success=success)
        return StepResult(obs=obs, reward=float(success), done=done, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extract and format camera views for recording logs.

        Args:
            raw_obs: The raw observation mapping.

        Returns:
            The image numpy array, or None if no suitable image is found or raw_obs is not a dict.
        """
        if not isinstance(raw_obs, dict):
            return None
        for cam in self._camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                # Match make_obs's vertical flip so recorded video matches what the model sees.
                return np.ascontiguousarray(raw_obs[key][::-1])
        return None

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Converts raw environment observations into a standardized format.

        This method extracts specified camera images and the task description
        from the raw observations.

        Args:
            raw_obs: The raw observation dictionary from the environment.
            task: The current task dictionary (unused in this implementation, but required by API).

        Returns:
            A dictionary containing processed observations, including 'images'
            (a dict of camera_name: numpy_array) and 'task_description'.
        """
        images: dict[str, Any] = {}
        for cam in self._camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                # RoboCasa images are typically upside-down; flip vertically to present correctly
                # and ensure memory contiguity.
                images[cam] = np.ascontiguousarray(raw_obs[key][::-1])
        return {
            "images": images,
            "task_description": self._lang,
        }

    def check_done(self, step_result: StepResult) -> bool:
        """Determines if an episode is complete based on the step result.

        An episode is considered done if the environment signals done or if
        the task has been successfully completed.

        Args:
            step_result: The result of the latest simulation step.

        Returns:
            True if the episode is done, False otherwise.
        """
        return step_result.done or step_result.info.get("success", False)

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Extracts key information from a StepResult for logging or further processing.

        Args:
            step_result: The result of the latest simulation step.

        Returns:
            A dictionary containing relevant metrics, such as 'success'.
        """
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata about the simulator configuration.

        Returns:
            A dictionary containing metadata, such as the maximum steps per episode.
        """
        return {"max_steps": self._max_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Returns the specification for the actions expected by this simulator.

        Describes the structure and types of the action space.

        Returns:
            A dictionary mapping action component names (e.g., "position", "rotation")
            to their respective DimSpec objects.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Returns the specification for the observations provided by this simulator.

        Describes the structure and types of the observation space.

        Returns:
            A dictionary mapping observation component names (e.g., "robot0_agentview_left",
            "language") to their respective DimSpec objects.
        """
        return {
            "robot0_agentview_left": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def render(self) -> np.ndarray | None:
        """Render the current environment state.

        Returns:
            The visual image array, or None if rendering fails.
        """
        try:
            return self._env.render()
        except Exception:
            # If rendering fails (e.g., no display context), return None.
            return None

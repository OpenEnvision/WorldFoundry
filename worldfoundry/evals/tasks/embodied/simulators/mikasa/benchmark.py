"""MIKASA-Robo benchmark implementation.

Memory-intensive robotic manipulation tasks built on ManiSkill3/SAPIEN.
This benchmark consists of 32 tasks across various categories such as 'remember',
'shell-game', 'rotate', and 'intercept', designed to evaluate an agent's
memory and manipulation capabilities.

Key details:
- Utilizes the ManiSkill3 gymnasium API with `obs_mode="rgb"` and a Panda robot.
- Requires the `StateOnlyTensorToDictWrapper` for processing observations.
- Actions are represented as `dict[str, Any]`, corresponding to an 8D vector
  (7 joint deltas + 1 gripper command) for `pd_joint_delta_pos` control.
- Success is indicated by `info["success"]` (a batched tensor, index `[0]`
  for a single environment).
- `max_episode_steps` varies per task, ranging from 60 to 180 steps.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, DimSpec

os.environ.setdefault("DISPLAY", "")

# Representative subset — one task per category
DEFAULT_TASKS = [
    "RememberColor3-v0",
    "ShellGameTouch-v0",
    "RotateLenientPos-v0",
    "InterceptSlow-v0",
    "TakeItBack-v0",
]

# Human-readable descriptions for all MIKASA-Robo tasks.
TASK_DESCRIPTIONS: dict[str, str] = {
    "RememberColor3-v0": "Remember the color of the target cube, then touch it after it reappears",
    "RememberColor5-v0": "Remember the target color among 5 cubes",
    "RememberColor9-v0": "Remember the target color among 9 cubes",
    "RememberShape3-v0": "Remember the shape of the target object",
    "RememberShape6-v0": "Remember the target shape among 6 objects",
    "RememberShape9-v0": "Remember the target shape among 9 objects",
    "ShellGameTouch-v0": "Track a ball hidden under shuffling cups, then touch the correct cup",
    "ShellGamePush-v0": "Track and push the cup hiding the ball",
    "ShellGamePick-v0": "Track and pick up the cup hiding the ball",
    "RotateLenientPos-v0": "Rotate object to the target angle (lenient, positive only)",
    "RotateLenientPosNeg-v0": "Rotate object to the target angle (lenient, both directions)",
    "RotateStrictPos-v0": "Rotate object to exact target angle (strict, positive)",
    "RotateStrictPosNeg-v0": "Rotate object to exact target angle (strict, both)",
    "InterceptSlow-v0": "Intercept a slow-moving object",
    "InterceptMedium-v0": "Intercept a medium-speed object",
    "InterceptFast-v0": "Intercept a fast-moving object",
    "InterceptGrabSlow-v0": "Grab a slow-moving object",
    "InterceptGrabMedium-v0": "Grab a medium-speed object",
    "InterceptGrabFast-v0": "Grab a fast-moving object",
    "TakeItBack-v0": "Pick up object and return it to its original position",
    "BunchOfColors3-v0": "Touch all cubes of the target color in a bunch of 3",
    "BunchOfColors5-v0": "Touch all cubes of the target color in a bunch of 5",
    "BunchOfColors7-v0": "Touch all cubes of the target color in a bunch of 7",
    "SeqOfColors3-v0": "Touch 3 colored cubes in the memorised sequence",
    "SeqOfColors5-v0": "Touch 5 colored cubes in sequence",
    "SeqOfColors7-v0": "Touch 7 colored cubes in sequence",
    "ChainOfColors3-v0": "Follow a chain of 3 colors",
    "ChainOfColors5-v0": "Follow a chain of 5 colors",
    "ChainOfColors7-v0": "Follow a chain of 7 colors",
    "RememberShapeAndColor3x2-v0": "Remember shape and color (3 shapes x 2 colors)",
    "RememberShapeAndColor3x3-v0": "Remember shape and color (3x3)",
    "RememberShapeAndColor5x3-v0": "Remember shape and color (5x3)",
}


class MIKASABenchmark(BaseSimulator):
    """MIKASA-Robo memory-intensive manipulation benchmark.

    This class provides an interface to the MIKASA-Robo suite of tasks,
    handling environment initialization, step execution, observation
    processing, and result reporting. It conforms to the `BaseSimulator`
    API for embodied task evaluations, specifically designed for
    memory and manipulation challenges.
    """

    # Fields that are recorded for video logging during evaluation.
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        episodes_per_task: int = 10,
        max_episode_steps: int | None = None,
        render_resolution: list[int] | tuple[int, int] = (256, 256),
    ) -> None:
        """Initializes the MIKASABenchmark simulator.

        Args:
            tasks: A list of specific task names to run. If None, uses DEFAULT_TASKS.
            episodes_per_task: The number of episodes to run for each task. (Currently unused internally)
            max_episode_steps: Overrides the default maximum steps for an episode
                if provided.
            render_resolution: The [width, height] for rendered observations.
        """
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._max_steps_override = max_episode_steps
        self._render_resolution = tuple(render_resolution)
        self._env: Any = None  # Holds the active ManiSkill3 environment instance
        self._current_task: str | None = None  # Tracks the name of the currently loaded task
        self._task_desc: str = ""  # Stores the human-readable description for the current task

    def cleanup(self) -> None:
        """Safely close and clean up active SAPIEN environment allocations.

        This method attempts to close the underlying ManiSkill3 environment to
        release resources, handling potential exceptions during closure.
        """
        if self._env is not None:
            # Catch any exceptions during env closure to prevent crashing
            # in case the environment is already in an invalid state.
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None

    def get_tasks(self) -> list[dict[str, Any]]:
        """List registered tasks for the MIKASA-Robo evaluation suite.

        Returns:
            A list of task dictionary configurations, where each dict
            contains at least a "name" key specifying the task identifier.
        """
        return [{"name": t} for t in self._task_names]

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the environment to start a new episode for the given task.

        If the environment is not initialized or the task has changed, a new
        ManiSkill3 environment instance is created and wrapped.

        Args:
            task: A dictionary containing task details, must include a "name" key.

        Returns:
            The initial observation from the environment after reset.
        """
        # Imports are placed here to allow for lazy loading of ManiSkill3 and
        # Mikasa-Robo, which can be computationally expensive to import.
        gym = __import__("gymnasium")
        __import__("mikasa_robo_suite")  # registers envs with gymnasium

        env_name = task["name"]

        # Re-create the environment only if it's not initialized or the task has changed.
        if self._env is None or self._current_task != env_name:
            if self._env is not None:
                self._env.close()
            self._env = gym.make(
                env_name,
                num_envs=1,
                obs_mode="rgb",
                render_mode="cameras",
            )
            # Wrap the environment to convert state observations (which MIKASA uses)
            # into a dictionary format compatible with the expected API.
            StateOnlyTensorToDictWrapper = __import__(
                "mikasa_robo_suite.utils.wrappers", fromlist=["StateOnlyTensorToDictWrapper"]
            ).StateOnlyTensorToDictWrapper

            self._env = StateOnlyTensorToDictWrapper(self._env)
            self._current_task = env_name

        obs, info = self._env.reset()
        # Retrieve the human-readable description for the current task.
        self._task_desc = TASK_DESCRIPTIONS.get(env_name, f"Complete {env_name}")
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Takes a step in the environment using the provided action.

        Processes the action, converts it to the required format, performs
        the environment step, and records relevant metrics.

        Args:
            action: A dictionary containing the agent's action, typically under
                an "actions" or "action" key. Expected to be an 8D vector
                (7 joint deltas + 1 gripper command).

        Returns:
            A StepResult named tuple containing the observation, reward,
            done status, and additional info.
        """
        import torch

        # Extract raw action data, supporting both "actions" and "action" keys.
        # Defaults to an 8-D zero vector if no action is provided.
        raw = action.get("actions", action.get("action"))
        if raw is None:
            raw = np.zeros(8, dtype=np.float32)
        raw = np.asarray(raw, dtype=np.float32).flatten()
        assert raw.shape[-1] == 8, f"dict[str, Any] dimension mismatch: got {raw.shape[-1]}, expected 8"

        # Pad or truncate the action to match the environment's expected action space dimension.
        act_dim = self._env.action_space.shape[-1]
        if raw.shape[0] < act_dim:
            raw = np.pad(raw, (0, act_dim - raw.shape[0]))
        raw = raw[:act_dim]

        # Convert the NumPy array action to a PyTorch tensor and add a batch dimension.
        act_tensor = torch.from_numpy(raw).unsqueeze(0)
        obs, reward, terminated, truncated, info = self._env.step(act_tensor)

        # Determine the episode's done status and extract reward and success.
        done = bool(terminated.any()) or bool(truncated.any())
        rew = float(reward.sum())
        # Check for success, handling cases where 'success' might not be present or is a tensor.
        success = bool(info.get("success", torch.tensor(False)).any())

        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=rew, done=done, success=success)

        return StepResult(obs=obs, reward=rew, done=done, info={"success": success})

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extract and format camera views for recording logs.

        This internal helper method looks for RGB image data within the raw
        observation and converts it to a NumPy array suitable for video recording.

        Args:
            raw_obs: Raw observation from the environment, potentially a dictionary
                     containing 'sensor_data'.

        Returns:
            The RGB image numpy array (H, W, 3) in uint8 format, or None if no RGB data is found.
        """
        if not isinstance(raw_obs, dict) or "sensor_data" not in raw_obs:
            return None
        # Iterate through camera data to find an RGB image.
        for _cam_name, cam_data in raw_obs["sensor_data"].items():
            if "rgb" in cam_data:
                img = cam_data["rgb"]
                # Convert from PyTorch tensor to NumPy array if applicable.
                if hasattr(img, "cpu"):
                    img = img.cpu().numpy()
                # Remove batch dimension if present (e.g., from `num_envs=1`).
                if img.ndim == 4:
                    img = img[0]
                return np.asarray(img)
        return None

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Constructs a standardized observation dictionary.

        Processes raw observations from the ManiSkill3 environment into a format
        expected by the evaluation framework, including image data and the
        human-readable task description.

        Args:
            raw_obs: Raw observation from the environment, potentially a dictionary
                     containing 'sensor_data'.
            task: The current task dictionary (unused in this specific implementation,
                  but part of the `BaseSimulator` interface).

        Returns:
            A dictionary containing processed observations, including images
            under the "images" key and the "task_description".
        """
        images: dict[str, np.ndarray] = {}
        if isinstance(raw_obs, dict) and "sensor_data" in raw_obs:
            # Extract RGB images from all available cameras.
            for cam_name, cam_data in raw_obs["sensor_data"].items():
                if "rgb" in cam_data:
                    img = cam_data["rgb"]
                    # Convert from PyTorch tensor to NumPy array if applicable.
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                    # Remove batch dimension if present (e.g., from `num_envs=1`).
                    if img.ndim == 4:
                        img = img[0]
                    images[cam_name] = img

        # If no images are extracted, provide a blank image as a fallback
        # to ensure the "images" key is always present.
        if not images:
            images["base_camera"] = np.zeros((*self._render_resolution, 3), dtype=np.uint8)
        return {"images": images, "task_description": self._task_desc}

    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the episode is finished based on the step result.

        Args:
            step_result: The result of a single environment step, containing
                         `done` status.

        Returns:
            True if the episode is done (terminated or truncated), False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Retrieves specific metrics from the step result for logging.

        This method extracts key performance indicators, such as 'success',
        from the `StepResult` for comprehensive evaluation logging.

        Args:
            step_result: The result of a single environment step, containing
                         `info` with success status.

        Returns:
            A dictionary containing key performance indicators, such as 'success'.
        """
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata about the simulator and tasks.

        This includes configuration details like the maximum number of steps
        per episode, which might be overridden by the user.

        Returns:
            A dictionary containing metadata, including the maximum number of steps
            per episode.
        """
        # Provide either the user-overridden max steps or the MIKASA default of 90.
        return {"max_steps": self._max_steps_override or 90}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Defines the structure and dimensions of the action space.

        The MIKASA-Robo tasks use a 7-dimensional joint delta for robot arm
        control and a 1-dimensional gripper control, totaling an 8D action space.

        Returns:
            A dictionary mapping action component names to their `DimSpec`
            objects, describing the type and dimensions of each action.
        """
        return {
            "joints": DimSpec("joints", 7, "joint_delta_pos"),
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Defines the structure and dimensions of the observation space.

        The MIKASA-Robo tasks typically provide RGB images from a camera
        and a language description of the current task.

        Returns:
            A dictionary mapping observation component names to their `DimSpec`
            objects, describing the type and dimensions of each observation.
        """
        return {
            "base_camera": IMAGE_RGB,
            "language": LANGUAGE,
        }

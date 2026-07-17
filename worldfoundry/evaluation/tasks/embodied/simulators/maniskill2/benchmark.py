"""ManiSkill2 benchmark implementation.

Evaluates 5 ManiSkill2 tasks:
PickCube-v0, StackCube-v0, PickSingleYCB-v0, PickSingleEGAD-v0, PickClutterYCB-v0.

Key details:
- gymnasium API with obs_mode="rgbd", control_mode="pd_ee_delta_pose"
- Gripper state tracking: self.gripper_state = -action[-1]
- Gripper binarization: < 0.5 → 1 (open), >= 0.5 → -1 (close)
- Done condition: terminated or truncated
- Success: info.get("success", False)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evaluation.tasks.embodied.simulators.specs import (
    GRIPPER_CLOSE_NEG,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_EULER,
    DimSpec,
)

# Prevent display issues in headless environments
os.environ.setdefault("DISPLAY", "")

# dict[str, Any] name → goal description
TASK_GOALS: dict[str, str] = {
    "PickCube-v0": "pick up a cube and move it to the green point",
    "StackCube-v0": "pick up a red cube and place it on a green cube",
    "PickSingleYCB-v0": "pick up the {} and move it to the green point",
    "PickSingleEGAD-v0": "pick up the {} and move it to the green point",
    "PickClutterYCB-v0": "pick up the {} and move it to the green point",
}

DEFAULT_TASKS = list(TASK_GOALS.keys())


class ManiSkill2Benchmark(BaseSimulator):
    """ManiSkill2 manipulation benchmark (SAPIEN physics).

    This class provides a standardized interface for interacting with selected
    ManiSkill2 environments for benchmarking.

    Non-obvious behaviors:
        - **Goal site visibility**: ManiSkill2 hides the goal sphere by
          default.  This benchmark explicitly makes it visible before each
          step, matching the training setup where models see the green target.
        - **Gripper binarization**: Threshold is 0.5 (not 0.0).
          ``raw[-1] < 0.5 → 1.0 (close), ≥ 0.5 → -1.0 (open)``.
        - **Camera names**: ``enabled_cameras`` values must exactly match
          camera names defined in the ManiSkill2 environment.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "terminated", "truncated", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        episodes_per_task: int = 50,
        max_episode_steps: int = 400,
        render_resolution: list[int] | tuple[int, int] = (256, 256),
        enabled_cameras: list[str] | None = None,
    ) -> None:
        """Initializes the ManiSkill2Benchmark simulator.

        Args:
            tasks: List of ManiSkill2 task IDs (e.g. ``["PickCube-v0"]``).
                Defaults to all tasks defined in `TASK_GOALS`.
            episodes_per_task: Number of episodes to run for each task.
            max_episode_steps: Maximum number of steps allowed per episode.
            render_resolution: Desired camera resolution as a ``[width, height]`` tuple.
            enabled_cameras: List of camera names to enable for observations.
                Defaults to ``["base_camera"]``.
        """
        super().__init__()
        self.tasks = tasks or DEFAULT_TASKS
        self.episodes_per_task = episodes_per_task
        self.max_episode_steps = max_episode_steps
        self.render_resolution = tuple(render_resolution)
        self.enabled_cameras = enabled_cameras or ["base_camera"]

        self._env = None
        self._current_task: str | None = None
        self._goal: str = ""
        self.gripper_state: float = -1.0

    def cleanup(self) -> None:
        """Safely close and clean up the active SAPIEN ManiSkill2 environment.

        This method ensures the underlying ManiSkill2 gymnasium environment
        is properly closed to release resources, even if an error occurs during closing.
        """
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                # Catch any exception during close to ensure graceful cleanup.
                pass
            self._env = None

    def get_tasks(self) -> list[dict[str, Any]]:
        """Returns a list of tasks configured for the benchmark.

        Each task is represented as a dictionary with "name" and "env_name" keys.

        Returns:
            A list of task dictionaries.
        """
        return [{"name": t, "env_name": t} for t in self.tasks]

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the ManiSkill2 environment for a new episode of the specified task.

        If the task changes or the environment is not initialized, a new ManiSkill2
        environment instance is created.

        Args:
            task: A dictionary containing task information, including "env_name".

        Returns:
            The initial observation from the environment.
        """
        import gymnasium as gym
        import mani_skill2.envs  # noqa: F401 — register envs

        env_name = task["env_name"]

        # Recreate the environment if it's not initialized or if the task has changed.
        if self._env is None or self._current_task != env_name:
            if self._env is not None:
                self._env.close()  # Close the previous environment before creating a new one.
            self._env = gym.make(
                env_name,
                obs_mode="rgbd",
                control_mode="pd_ee_delta_pose",
                render_mode="cameras",
                camera_cfgs=dict(width=self.render_resolution[0], height=self.render_resolution[1]),
                max_episode_steps=self.max_episode_steps,
            )
            self._current_task = env_name

        # Reset the environment and initialize gripper state.
        obs, info = self._env.reset()
        self.gripper_state = -1.0  # Gripper starts open.

        # Resolve the goal description, filling in object names for templated tasks.
        goal_template = TASK_GOALS.get(env_name, "complete the task")
        if "{}" in goal_template:
            obj_name = self._get_obj_name()
            self._goal = goal_template.format(obj_name)
        else:
            self._goal = goal_template

        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Performs a step in the ManiSkill2 environment using the given action.

        Processes the input action, binarizes the gripper command, renders the
        goal site, and then steps the underlying ManiSkill2 environment.

        Args:
            action: A dictionary containing the "actions" or "action" key,
                with a list or numpy array of 7 floats:
                ``[delta_x, delta_y, delta_z, roll, pitch, yaw, gripper_command]``.

        Returns:
            A StepResult named tuple containing the observation, reward,
            done status, and environment info.

        Raises:
            AssertionError: If the action dimension is not 7.
        """
        raw = action.get("actions", action.get("action"))
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        assert len(raw) == 7, f"dict[str, Any] dimension mismatch: got {len(raw)}, expected 7"

        # Binarize gripper command: raw value < 0.5 means close (1.0), >= 0.5 means open (-1.0).
        gripper = 1.0 if raw[-1] < 0.5 else -1.0
        env_action = np.array(raw[:6] + [gripper], dtype=np.float32)

        # Render goal site to make the green target sphere visible in camera observations.
        # ManiSkill2 hides it by default (_hidden_objects), but the model was trained
        # with it visible. This must be done before env.step().
        assert self._env is not None
        self._render_goal_site(self._env)

        obs, reward, terminated, truncated, info = self._env.step(env_action)

        # Update and track the current gripper state, which is the negative of the action applied.
        self.gripper_state = -env_action[-1]

        # Determine if the episode is done (terminated or truncated).
        done = bool(terminated) or bool(truncated)
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(
            reward=float(reward),
            done=done,
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=bool(info.get("success", False)),
        )
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extract and format the RGB frame from a raw ManiSkill2 observation.

        This method attempts to locate and return the RGB image data from the
        observation dictionary, prioritizing cameras that might be configured.

        Args:
            raw_obs: The raw observation dictionary provided by the ManiSkill2 environment.

        Returns:
            A numpy array representing the RGB image (H, W, 3), or None if no
            valid RGB image data is found or `raw_obs` is not a dictionary.
        """
        if not isinstance(raw_obs, dict):
            return None
        # Iterate through configured cameras and return the first available RGB image.
        for cam in self.enabled_cameras:
            cam_data = raw_obs.get("image", {}).get(cam)
            if cam_data is not None and "rgb" in cam_data:
                return np.asarray(cam_data["rgb"])
        return None

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Formats the raw ManiSkill2 observation into a standardized observation dictionary.

        Extracts camera images and combines them with the task description.

        Args:
            raw_obs: The raw observation dictionary from the ManiSkill2 environment.
            task: A dictionary containing task information (though not directly used here,
                  `_goal` derived from `task` is used).

        Returns:
            A dictionary containing:
            - "images": A dictionary mapping camera names to their RGB numpy arrays.
            - "task_description": The natural language goal description for the current task.
        """
        # Extract RGB images from all enabled cameras.
        images: dict[str, np.ndarray] = {}
        for cam in self.enabled_cameras:
            images[cam] = raw_obs["image"][cam]["rgb"]

        return {
            "images": images,
            "task_description": self._goal,
        }

    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the episode is finished based on the step result.

        Args:
            step_result: The result of a simulation step.

        Returns:
            True if the episode is done (terminated or truncated), False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Extracts key results from a simulation step for evaluation.

        Args:
            step_result: The result of a simulation step.

        Returns:
            A dictionary containing a "success" boolean indicating task completion.
        """
        return {"success": bool(step_result.info.get("success", False))}

    def get_metadata(self) -> dict[str, Any]:
        """Provides metadata about the simulator configuration.

        Returns:
            A dictionary containing "max_steps" for the current environment.
        """
        return {"max_steps": self.max_episode_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Returns the action space specification for the simulator.

        Describes the expected dimensions and types for position, rotation, and gripper actions.

        Returns:
            A dictionary mapping action component names to their `DimSpec` definitions.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Returns the observation space specification for the simulator.

        Describes the expected dimensions and types for camera images and language goals.

        Returns:
            A dictionary mapping observation component names to their `DimSpec` definitions.
        """
        return {
            "base_camera": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def _get_obj_name(self) -> str:
        """Extract the target object's name from the environment to construct goal descriptions.

        Attempts to retrieve the object name from the unwrapped ManiSkill2 environment.

        Returns:
            The object name string, or "object" if extraction fails.
        """
        try:
            assert self._env is not None
            obj = self._env.unwrapped.obj
            # Extract the meaningful part of the object name, e.g., "ycb_chips_can" -> "chips can".
            return " ".join(obj.name.split("_")[1:])
        except (AttributeError, IndexError):
            # Fallback if the object or its name is not available or malformed.
            return "object"

    @staticmethod
    def _render_goal_site(env: Any) -> None:
        """Makes the goal_site sphere visible in camera observations.

        ManiSkill2 environments often hide the goal visual by default. This method
        explicitly sets its visibility to 1.0, as models are typically trained
        with the green goal sphere visible.

        Args:
            env: The ManiSkill2 environment instance.
        """
        # Access the unwrapped environment to reliably find the goal_site attribute.
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(unwrapped, "goal_site"):
            # Iterate through all visual bodies of the goal_site and make them visible.
            for v in unwrapped.goal_site.get_visual_bodies():
                v.set_visibility(1.0)

"""VLABench benchmark implementation.

VLABench is a large-scale benchmark for language-conditioned robotics
manipulation with long-horizon reasoning tasks, built on dm_control (MuJoCo).

Actions from the model server are 7-D: ``[dx, dy, dz, droll, dpitch, dyaw,
gripper]``. These deltas are added to the current end-effector pose, then
converted to joint-space via inverse kinematics before being sent to the
dm_control environment as ``[7D qpos, 2D gripper]``.

Success is detected via dm_control's ``timestep.last()`` termination signal.
"""

from __future__ import annotations

import contextlib
import logging
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

logger = logging.getLogger(__name__)

# Configure MuJoCo to use EGL for headless rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

# Default VLABench task names for evaluation.
DEFAULT_TASKS = [
    "select_fruit",
    "select_toy",
    "select_drink",
    "select_book",
    "select_painting",
]

# Default maximum number of simulation steps allowed per episode.
DEFAULT_MAX_STEPS = 200


class VLABenchBenchmark(BaseSimulator):
    """VLABench manipulation benchmark (dm_control / MuJoCo).

    This simulator handles interaction with the VLABench environment, including
    action translation, observation extraction, and episode management.

    Args:
        tasks: List of VLABench task names to evaluate. If `None`, uses `DEFAULT_TASKS`.
        robot: Robot name (default ``"franka"``).
        max_steps: Maximum steps per episode (default 200).
    """

    # Fields that should be recorded by the episode recorder.
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "franka",
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        """Initializes the VLABench benchmark simulator.

        Args:
            tasks: A list of VLABench task names to be used for evaluation.
                   If None, `DEFAULT_TASKS` will be used.
            robot: The name of the robot to use in the VLABench environment.
            max_steps: The maximum number of simulation steps allowed per episode.
        """
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._robot = robot
        self._max_steps = max_steps
        self._env: Any = None  # The VLABench environment instance
        self._current_task: str | None = None
        self._instruction: str = ""  # Current task instruction
        self._last_ee_state: np.ndarray | None = None  # Cached end-effector state

    def cleanup(self) -> None:
        """Safely close the VLABench environment.

        Attempts to close the underlying VLABench environment if it exists.
        Errors during closing are ignored to prevent crashes in already
        degraded states.
        """
        if self._env is not None:
            # Ignore errors during closing if the environment is already in a bad state.
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None

    def _ensure_vlabench(self) -> None:
        """Lazy-import VLABench and register robots/tasks with stubbed point-cloud generator.

        This method ensures VLABench modules are loaded and performs a monkey-patch
        to prevent issues with Open3D in headless environments.
        """
        # Importing VLABench modules dynamically to avoid potential issues if not installed
        # and to allow for custom environment setup before imports.
        __import__("VLABench")  # triggers VLABENCH_ROOT setup
        __import__("VLABench.robots")  # registers robot classes with VLABench
        __import__("VLABench.tasks")  # registers task classes with VLABench

        # Monkey-patch to skip PCD generator (Open3D segfaults in headless
        # containers and we never request point clouds). This prevents crashes
        # when running in environments without GUI/Open3D support or when Open3D
        # causes conflicts.
        LM4ManipDMEnv = __import__("VLABench.envs.dm_env", fromlist=["LM4ManipDMEnv"]).LM4ManipDMEnv

        if not hasattr(LM4ManipDMEnv, "_pcd_patched"):
            # Define a stub class that mimics the necessary attribute for the PCD generator.
            class _PcdStub:
                physics = None

            # Overwrite the `register_pcd_generator` method to use the stub instead of actual Open3D.
            LM4ManipDMEnv.register_pcd_generator = lambda self: setattr(self, "pcd_generator", _PcdStub())
            # Mark the class as patched to prevent re-patching.
            LM4ManipDMEnv._pcd_patched = True

    def get_tasks(self) -> list[dict[str, Any]]:
        """Returns a list of tasks configured for this benchmark.

        Returns:
            A list of dictionaries, each describing a task with a "name" key.
        """
        return [{"name": t} for t in self._task_names]

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the VLABench environment to start a new episode.

        A new environment instance is loaded for each reset to ensure proper
        scene setup for the specified task.

        Args:
            task: A dictionary containing task-specific information, including its "name".

        Returns:
            The initial observation from the environment after reset.
        """
        self._ensure_vlabench()
        load_env = __import__("VLABench.envs", fromlist=["load_env"]).load_env

        task_name = task["name"]

        # Close previous environment instance and create a new one.
        # This is important because VLABench tasks may change scene layouts
        # which requires a fresh environment instance to ensure proper setup.
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning("Failed to close VLABench environment: %s", e)
        self._env = load_env(task_name, robot=self._robot)
        self._current_task = task_name

        obs = self._env.get_observation(require_pcd=False)
        self._instruction = self._env.task.get_instruction()
        self._last_ee_state = obs.get("ee_state", None)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Applies an action to the VLABench environment and steps it forward.

        The input action is a 7-dimensional delta in end-effector space
        ([dx, dy, dz, droll, dpitch, dyaw, gripper]). This is converted
        to an absolute end-effector pose, then to joint positions via
        inverse kinematics, and finally combined with the gripper state
        before being sent to the VLABench environment.

        Args:
            action: A dictionary containing the action to be performed, typically
                    under the key "actions" or "action", formatted as 7D:
                    [dx, dy, dz, droll, dpitch, dyaw, gripper].

        Returns:
            A StepResult containing the new observation, reward, done status, and info.
        """
        rotation_utils = __import__("VLABench.utils.utils", fromlist=["euler_to_quaternion", "quaternion_to_euler"])
        euler_to_quaternion, quaternion_to_euler = (
            rotation_utils.euler_to_quaternion,
            rotation_utils.quaternion_to_euler,
        )

        # Extract the raw 7D action from the input dictionary.
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(7, dtype=np.float32)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        assert raw_action.shape[-1] == 7, f"dict[str, Any] dimension mismatch: got {raw_action.shape[-1]}, expected 7"

        # Interpret the 7D action components as deltas for position and Euler angles,
        # and a command for the gripper.
        delta_pos = raw_action[:3]  # Change in x, y, z position
        delta_euler = raw_action[3:6]  # Change in roll, pitch, yaw
        gripper_cmd = raw_action[6] if len(raw_action) > 6 else 0.0  # Gripper command: >0 for open, <=0 for close

        # Get current end-effector (EE) state to calculate absolute target pose.
        # The last EE state is cached for efficiency, but queried from env if missing.
        ee_state = self._last_ee_state
        if ee_state is None:
            # If cached state is missing, query the environment for current EE pose.
            ee_state = np.concatenate([self._env.get_ee_pos(), self._env.get_ee_quat(), [0.0]])
        current_pos = ee_state[:3]
        current_quat = ee_state[3:7]
        current_euler = np.array(quaternion_to_euler(current_quat))

        # Calculate the target end-effector position and orientation by adding deltas.
        target_pos = current_pos + delta_pos
        target_euler = current_euler + delta_euler
        target_quat = euler_to_quaternion(*target_euler)

        # Inverse kinematics: Convert desired EE pose (position + quaternion)
        # to joint positions (qpos) for the robot.
        _, qpos = self._env.robot.get_qpos_from_ee_pos(
            physics=self._env.physics,
            pos=target_pos,
            quat=target_quat,
        )

        # Gripper command interpretation: >0 means open (0.04), <=0 means closed (0.0).
        # VLABench gripper uses a 2D array [state, state] for its state.
        grip_val = 0.04 if gripper_cmd > 0 else 0.0
        gripper_state = np.array([grip_val, grip_val])

        # Combine joint positions (from IK) and gripper state into the full environment action.
        env_action = np.concatenate([qpos, gripper_state])

        timestep = self._env.step(env_action)
        # Success is determined by the environment's termination signal (timestep.last()).
        success = bool(timestep.last())

        # Update cached EE state and instruction from the new observation.
        obs = self._env.get_observation(require_pcd=False)
        self._last_ee_state = obs.get("ee_state", None)
        self._instruction = self._env.task.get_instruction()

        # Record video frame and step metrics.
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=1.0 if success else 0.0, done=success, success=success)

        return StepResult(
            obs=obs,
            reward=1.0 if success else 0.0,
            done=success,
            info={"success": success, "timestep": timestep},
        )

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        """Extract and format the main RGB camera image.

        Args:
            raw_obs: The raw observation mapping from the VLABench environment.

        Returns:
            The image numpy array (H, W, 3) suitable for video recording,
            or None if the image data is missing or malformed.
        """
        if not isinstance(raw_obs, dict):
            return None
        rgb = raw_obs.get("rgb")
        if rgb is None or len(rgb) == 0:
            return None
        # Assuming 'rgb' is a list of camera images, take the first one.
        return np.asarray(rgb[0])

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Converts raw environment observations into a standardized format.

        Extracts the primary RGB image and the current task instruction
        from the raw VLABench observation.

        Args:
            raw_obs: The raw observation object from the VLABench environment.
            task: The current task dictionary.

        Returns:
            A dictionary containing standardized observations, including images
            and the task description.
        """
        images: dict[str, np.ndarray] = {}
        rgb = raw_obs.get("rgb", None)
        if rgb is not None and len(rgb) > 0:
            # VLABench returns (N_cams, H, W, 3) for 'rgb' - we take the image from the first camera.
            images["primary"] = rgb[0]

        return {
            "images": images,
            "task_description": self._instruction,
        }

    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the episode has finished based on the step result.

        Args:
            step_result: The result of the last simulation step.

        Returns:
            True if the episode is done, False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Extracts key metrics from the step result.

        Args:
            step_result: The result of the last simulation step.

        Returns:
            A dictionary containing relevant metrics for the step, e.g., success status.
        """
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        """Provides metadata about the simulator and its configuration.

        Returns:
            A dictionary containing metadata, such as the maximum number of steps per episode.
        """
        return {"max_steps": self._max_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Returns the specification of the action space.

        The action space consists of 7 dimensions: 3 for delta position,
        3 for delta Euler angles, and 1 for gripper state.

        Returns:
            A dictionary mapping action component names to their dimension specifications.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Returns the specification of the observation space.

        The observation space includes a primary RGB image and a language
        instruction.

        Returns:
            A dictionary mapping observation component names to their dimension specifications.
        """
        return {
            "primary": IMAGE_RGB,
            "language": LANGUAGE,
        }

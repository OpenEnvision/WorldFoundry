"""RLBench benchmark implementation.

Uses CoppeliaSim 4.1.0 + PyRep for simulation, shipped in the
``rlbench`` Docker image.  Xvfb is started by the entrypoint for
headless rendering.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, DimSpec

# The Docker entrypoint starts Xvfb and sets DISPLAY=:99.
# Do NOT set QT_QPA_PLATFORM=offscreen — CoppeliaSim needs xcb+Xvfb.

DEFAULT_TASKS = [
    "reach_target",
    "pick_up_cup",
    "push_button",
    "close_drawer",
    "open_door",
]


class RLBenchBenchmark(BaseSimulator):
    """RLBench manipulation benchmark (CoppeliaSim / PyRep).

    Args:
        tasks: List of RLBench task file names (snake_case).
        render_resolution: Camera resolution (square).
        max_steps: Maximum steps per episode.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        render_resolution: int = 256,
        max_steps: int = 200,
    ) -> None:
        """Initializes the RLBench benchmark simulator.

        Args:
            tasks: A list of RLBench task names (e.g., "reach_target"). If None, uses DEFAULT_TASKS.
            render_resolution: The resolution (width and height) for the camera observations.
            max_steps: The maximum number of simulation steps allowed per episode.
        """
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._render_resolution = render_resolution
        self._max_steps = max_steps
        self._env = None  # rlbench.environment.Environment
        self._task_env = None  # rlbench.task_environment.TaskEnvironment
        self._descriptions: list[str] = []

    def cleanup(self) -> None:
        """Safely close the CoppeliaSim engine and PyRep allocations.

        This method ensures that the simulation environment is properly shut down
        to release resources, handling potential exceptions during shutdown.
        """
        if self._env is not None:
            # Suppress exceptions during shutdown to ensure cleanup completes
            with contextlib.suppress(Exception):
                self._env.shutdown()
            self._env = None
            self._task_env = None

    # ------------------------------------------------------------------ #
    # lazy init
    # ------------------------------------------------------------------ #
    def _ensure_env(self):
        """Lazy-initialize the CoppeliaSim environment and configure camera properties.

        This method creates and launches the RLBench environment if it hasn't been
        initialized yet, setting up camera configurations and action modes.
        """
        if self._env is not None:
            return
        MoveArmThenGripper = __import__(
            "rlbench.action_modes.action_mode", fromlist=["MoveArmThenGripper"]
        ).MoveArmThenGripper
        JointVelocity = __import__("rlbench.action_modes.arm_action_modes", fromlist=["JointVelocity"]).JointVelocity
        Discrete = __import__("rlbench.action_modes.gripper_action_modes", fromlist=["Discrete"]).Discrete
        Environment = __import__("rlbench.environment", fromlist=["Environment"]).Environment
        observation_config = __import__("rlbench.observation_config", fromlist=["CameraConfig", "ObservationConfig"])
        CameraConfig, ObservationConfig = observation_config.CameraConfig, observation_config.ObservationConfig

        # Configure the primary camera for RGB observations at the specified resolution
        cam = CameraConfig(
            rgb=True,
            depth=False,
            point_cloud=False,
            mask=False,
            render_resolution=(self._render_resolution, self._render_resolution),
        )
        # Configure other cameras to be off to optimize performance
        cam_off = CameraConfig()
        cam_off.set_all(False)

        # Define the overall observation configuration, enabling only front and wrist RGB cameras
        obs_cfg = ObservationConfig(
            front_camera=cam,
            wrist_camera=cam,
            left_shoulder_camera=cam_off,
            right_shoulder_camera=cam_off,
            overhead_camera=cam_off,
            joint_positions=True,
            gripper_open=True,
        )

        # Define the action mode: move arm via joint velocities, then gripper discretely
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        )
        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_cfg,
            headless=True,
        )
        self._env.launch()

    # ------------------------------------------------------------------ #
    # BaseSimulator interface
    # ------------------------------------------------------------------ #
    def get_tasks(self) -> list[dict[str, Any]]:
        """Returns a list of available tasks for this benchmark.

        Each task is represented as a dictionary containing its name and task file identifier.

        Returns:
            A list of dictionaries, where each dictionary represents a task.
        """
        return [{"name": t, "task_file": t} for t in self._task_names]

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the simulation environment for a new episode with the specified task.

        This method initializes the RLBench environment if not already done,
        loads the specific task, samples a variation, and performs the first reset,
        recording the initial frame.

        Args:
            task: A dictionary containing task details, including "task_file".

        Returns:
            The initial observation from the environment after reset.
        """
        rlbench_utils = __import__("rlbench", fromlist=["utils"]).utils

        self._ensure_env()
        assert self._env is not None

        # Convert the task file name to its corresponding RLBench task class
        task_class = rlbench_utils.name_to_task_class(task["task_file"])
        self._task_env = self._env.get_task(task_class)
        self._task_env.sample_variation()
        # Reset the task environment and get initial descriptions and observation
        self._descriptions, obs = self._task_env.reset()
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Applies an action to the simulation and advances the environment by one step.

        Args:
            action: A dictionary containing the raw action, typically under "actions" or "action" key.
                    Expected to be an 8D action (7 joint velocities + 1 gripper discrete action).

        Returns:
            A StepResult object containing the observation, reward, done status, and info.
        """
        # Extract the raw action from the input dictionary
        raw_action = action.get("actions", action.get("action"))
        act = np.asarray(raw_action, dtype=np.float64)
        assert act.shape[-1] == 8, f"dict[str, Any] dimension mismatch: got {act.shape[-1]}, expected 8"

        # Ensure the action has 8 dimensions (7 joint velocities + 1 gripper discrete)
        if act.shape[0] < 8:
            # Pad with zeros if the action is shorter than expected
            act = np.pad(act, (0, 8 - act.shape[0]))
        # Truncate if the action is longer than expected, taking the first 8 dimensions
        act = act[:8]

        assert self._task_env is not None
        obs, reward, terminate = self._task_env.step(act)
        success = reward > 0.99
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=bool(terminate), success=bool(success))
        return StepResult(obs=obs, reward=reward, done=terminate, info={"success": bool(success)})

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        """Extract and format the front camera view from RLBench observations.

        Args:
            raw_obs: The raw observation object.

        Returns:
            The image numpy array, or None if missing.
        """
        front = getattr(raw_obs, "front_rgb", None)
        if front is None:
            return None
        return np.asarray(front, dtype=np.uint8)

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Converts raw RLBench observations into a standardized observation dictionary.

        Extracts RGB images from specified cameras and includes a task description.

        Args:
            raw_obs: The raw observation object from the RLBench environment.
            task: A dictionary containing task details.

        Returns:
            A dictionary containing processed observations, including images and task description.
        """
        images = {}
        if raw_obs.front_rgb is not None:
            images["front"] = np.asarray(raw_obs.front_rgb, dtype=np.uint8)
        if raw_obs.wrist_rgb is not None:
            images["wrist"] = np.asarray(raw_obs.wrist_rgb, dtype=np.uint8)

        # Use the first available description or fall back to the task name
        description = self._descriptions[0] if self._descriptions else task["name"]
        return {
            "images": images,
            "task_description": description,
        }

    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the episode is finished based on the step result.

        Args:
            step_result: The result of the current simulation step.

        Returns:
            True if the episode is done, False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Extracts the final outcome or success status from a step result.

        Args:
            step_result: The result of the current simulation step.

        Returns:
            A dictionary indicating the success status of the step/episode.
        """
        return {"success": step_result.reward > 0.99}

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata about the simulator and its configuration.

        Returns:
            A dictionary containing metadata, such as maximum steps per episode.
        """
        return {"max_steps": self._max_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Returns the specification of the action space.

        Describes the dimensions and types of actions the simulator accepts.

        Returns:
            A dictionary mapping action component names to their dimension specifications.
        """
        # Actions are 8-dimensional: 7 joint velocities + 1 discrete gripper action
        return {
            "joints": DimSpec("joints", 7, "joint_velocity"),
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Returns the specification of the observation space.

        Describes the dimensions and types of observations the simulator provides.

        Returns:
            A dictionary mapping observation component names to their dimension specifications.
        """
        return {
            "front": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "language": LANGUAGE,
        }

from __future__ import annotations

import contextlib
import math
import os
from typing import Any, cast

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.libero.utils import preprocess_libero_image
from worldfoundry.evals.tasks.embodied.simulators.rotation import matrix_to_quat, quat_to_axisangle
from worldfoundry.evals.tasks.embodied.simulators.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)

# EGL for headless rendering using GPU acceleration.
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _quat_to_axisangle_robosuite(quat: np.ndarray) -> np.ndarray:
    """Convert Robosuite-style quaternion [x, y, z, w] to axis-angle representation.

    Does not apply antipodal normalization, matching robosuite conventions.

    Args:
        quat: A 4D numpy array representing a quaternion.

    Returns:
        A 3D numpy array representing the corresponding axis-angle rotation.
    """
    q = quat.copy()
    # Clamp 'w' component to prevent numerical issues with acos if it slightly exceeds 1.
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    # Handle the case where the angle is 0 or pi (quaternion is [0,0,0,1] or [0,0,0,-1]),
    # where sin(angle/2) is zero, leading to division by zero.
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    # Convert quaternion to axis-angle: axis * 2 * acos(w) / sin(angle/2)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


# Default resolution for LIBERO environment images.
LIBERO_ENV_RESOLUTION = 256
# A dummy action used for initial wait steps in the environment, with an open gripper.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Mapping of LIBERO suite names to their respective maximum step counts per episode.
MAX_STEP_MAPPING = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
    "libero_90": 400,
}


class LIBEROBenchmark(BaseSimulator):
    """LIBERO tabletop manipulation benchmark simulator (MuJoCo/robosuite).

    This class provides an interface to interact with the LIBERO benchmark environments,
    handling task loading, environment resets, action execution, and observation
    processing. It integrates with the worldfoundry framework.

    Non-obvious behaviors:
        - **PyTorch compat**: Patches ``torch.load`` to use
          ``weights_only=False`` for PyTorch ≥2.6 compatibility with LIBERO's
          initial-state files (numpy arrays stored via ``torch.save``).
        - **Headless rendering**: Sets ``EGL_PLATFORM=device`` and
          ``PYOPENGL_PLATFORM=egl`` on import for GPU-accelerated headless
          rendering.
        - **Dummy wait steps**: At episode start, ``num_steps_wait`` steps
          (default 10) are executed with a fixed open-gripper action to let
          objects settle in the physics simulation.
        - **Suite-specific max_steps**: libero_spatial=220, libero_object=280,
          libero_goal=300, libero_10=520, libero_90=400.
        - **Image preprocessing**: robosuite renders images with inverted axes.
          Both agentview and wrist images are flipped ``[::-1, ::-1]`` to
          correct orientation, then resized to 256×256 with padding.

    Args:
        suite: LIBERO suite name (e.g. "libero_spatial", "libero_10").
        seed: Random seed for environment initialization.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive 8-D state
            ``[pos3, axisangle3, gripper2]`` in observations.
        absolute_action: Use absolute (world-frame) actions instead of delta.
            When True, sets ``robot.controller.use_delta = False`` after the
            initial dummy-wait steps.
        max_steps: Override the default suite-specific max step count.
            When None, uses ``MAX_STEP_MAPPING[suite]``.
        env_seed: Seed for ``env.seed()``. When None, defaults to ``seed``.
            OpenVLA reference uses ``env_seed=0`` separately from ``seed=7``.
        quat_no_antipodal: If True, uses Robosuite's quaternion to axis-angle
            conversion which does not handle antipodal normalization.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        suite: str = "libero_spatial",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
        quat_no_antipodal: bool = False,
    ) -> None:
        """Initialize the LIBEROBenchmark simulator."""
        super().__init__()
        self.suite = suite
        self.seed = seed
        # Select the appropriate quaternion to axis-angle conversion function based on `quat_no_antipodal`.
        self._quat_to_aa = _quat_to_axisangle_robosuite if quat_no_antipodal else quat_to_axisangle
        self.env_seed = env_seed if env_seed is not None else seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.absolute_action = absolute_action
        self._max_steps = max_steps
        self._env = None
        self._task_suite = None
        self._current_task_id: int | None = None

    def cleanup(self) -> None:
        """Safely close and clean up the active LIBERO environment.

        This method attempts to close the robosuite environment and resets internal
        references to prevent resource leaks.
        """
        if self._env is not None:
            # Ignore errors during close, as env might already be partially destroyed
            # or in an inconsistent state, especially during abnormal termination.
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None

    def _init_libero(self) -> None:
        """Lazily initialize LIBERO (heavy imports).

        This method ensures LIBERO benchmark utilities are loaded only when needed,
        and includes a patch for PyTorch's `torch.load` to ensure compatibility
        with LIBERO's initial state files which may contain numpy arrays saved via `torch.save`.
        """
        if self._task_suite is not None:
            return
        # LIBERO init states use torch.save with numpy arrays.
        # PyTorch ≥2.6 defaults weights_only=True which blocks numpy globals.
        # Patch torch.load to default weights_only=False for LIBERO compatibility.
        import functools

        import torch

        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        cast(Any, torch).load = _patched_load

        benchmark = __import__("libero.libero", fromlist=["benchmark"]).benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[self.suite]()

    def get_tasks(self) -> list[dict[str, Any]]:
        """Retrieve the list of registered tasks and language instructions for this suite.

        Initializes the LIBERO benchmark if not already done, then iterates
        through available tasks in the configured suite. Each task is represented
        as a dictionary containing its name (language instruction), suite name,
        task ID, and the underlying LIBERO task object.

        Returns:
            A list of task dictionaries containing name (language instruction),
            suite name, task_id, and the underlying LIBERO task object.
        """
        self._init_libero()
        assert self._task_suite is not None
        tasks = []
        for task_id in range(self._task_suite.n_tasks):
            task = self._task_suite.get_task(task_id)
            tasks.append(
                {
                    "name": task.language,
                    "suite": self.suite,
                    "task_id": task_id,
                    "task_obj": task,
                }
            )
        return tasks

    def reset(self, task: dict[str, Any]) -> Any:
        """Reset the environment for a new task episode.

        This method configures the robosuite environment based on the provided task,
        sets the initial state, and performs dummy wait steps to stabilize the
        physics simulation.

        Args:
            task: Task dictionary containing 'task_obj', 'task_id', and 'episode_idx'.
                'episode_idx' is used to select an initial state from the task suite.

        Returns:
            The initial raw observation dictionary from MuJoCo/robosuite after reset
            and dummy steps.
        """
        from pathlib import Path

        get_libero_path = __import__("libero.libero", fromlist=["get_libero_path"]).get_libero_path
        OffScreenRenderEnv = __import__("libero.libero.envs", fromlist=["OffScreenRenderEnv"]).OffScreenRenderEnv

        task_obj = task["task_obj"]
        task_id = task["task_id"]
        episode_idx = task.get("episode_idx", 0)

        # Only create a new environment when the task changes to avoid re-initializing
        # heavy resources like Mujoco for every episode of the same task.
        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._env.close()

            bddl_file = Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
            env_args = {
                "bddl_file_name": str(bddl_file),
                "camera_heights": LIBERO_ENV_RESOLUTION,
                "camera_widths": LIBERO_ENV_RESOLUTION,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(self.env_seed)
            self._env = env
            self._current_task_id = task_id

        # Reset env before setting init state (matches reference for LIBERO to clear previous episode state).
        self._env.reset()

        # Set initial state for the specific task and episode.
        assert self._task_suite is not None
        initial_states = self._task_suite.get_task_init_states(task_id)
        obs = self._env.set_init_state(initial_states[episode_idx])

        # Run dummy action wait steps to allow objects to settle in the physics simulation.
        # This is always done in delta mode to prevent abrupt movements upon reset.
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        # Switch to absolute action mode after settling if configured for it.
        # This ensures the dummy actions are always relative, then switches if needed.
        if self.absolute_action:
            for robot in self._env.robots:
                robot.controller.use_delta = False

        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Apply a step action in the environment and return the outcome.

        Processes the input action, discretizes the gripper command, executes
        the action in the robosuite environment, and records the results.

        Args:
            action: Action targets dictionary containing "actions" (7-dim: pos[3], rot_aa[3], gripper[1]).
                    Can also accept 'action' as a key.

        Returns:
            A StepResult containing observation, reward, done, and info.
        """
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"dict[str, Any] dimension mismatch: got {len(raw_action)}, expected 7"

        # Discretize gripper action: -1 for open, 1 for close based on sign.
        # This maps the continuous gripper command to a discrete Robosuite gripper state.
        gripper = -1.0 if raw_action[-1] < 0 else 1.0
        processed_action = [*raw_action[:-1], gripper]

        assert self._env is not None
        obs, reward, done, info = self._env.step(processed_action)
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=bool(done), success=bool(done))
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        """Extract and correctly orient the agentview image from raw observation.

        Robosuite renders images inverted; this method flips them along both axes to the standard
        upright orientation before returning.

        Args:
            raw_obs: The raw observation dictionary from robosuite.

        Returns:
            A correctly-oriented agentview image array (HxWx3), or None if not available.
        """
        if not isinstance(raw_obs, dict):
            return None
        frame = raw_obs.get("agentview_image")
        if frame is None:
            return None
        # Robosuite renders agentview/wrist images inverted; flip them to be upright.
        return np.ascontiguousarray(frame[::-1, ::-1])

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Convert raw MuJoCo/robosuite observation to a standard track observation dictionary.

        This method processes raw observations, applies image preprocessing,
        and aggregates relevant information like state and language descriptions
        into a standardized format.

        Args:
            raw_obs: Raw observation dictionary from the robosuite environment.
            task: Task dictionary containing target task metadata (e.g., 'name' for language instruction).

        Returns:
            A standard observation dictionary containing visual and language keys,
            and optionally proprioceptive state information.
        """
        # Preprocess the primary agentview image by flipping and resizing.
        img = preprocess_libero_image(raw_obs["agentview_image"], LIBERO_ENV_RESOLUTION)

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
        }

        if self.send_wrist_image:
            # Preprocess and include the wrist camera image if configured.
            wrist = preprocess_libero_image(raw_obs["robot0_eye_in_hand_image"], LIBERO_ENV_RESOLUTION)
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            # Proprioceptive state from observation, typically used by models like Pi0/OFT/GR00T.
            # This aggregates end-effector position, orientation, and gripper state.
            obs_dict["states"] = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    self._quat_to_aa(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            # Proprioceptive state from controller, typically used by models like X-VLA.
            # This provides a different source for end-effector state, sometimes preferred for consistency.
            assert self._env is not None
            robot = self._env.robots[0]
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            ee_ori_mat = np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
            ee_aa = quat_to_axisangle(matrix_to_quat(ee_ori_mat))
            obs_dict["controller_states"] = np.concatenate(
                [ee_pos, ee_aa, np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)]
            )

        return obs_dict

    def check_done(self, step_result: StepResult) -> bool:
        """Check if environment episode has terminated.

        In LIBERO, the `done` flag from the environment directly indicates episode termination.

        Args:
            step_result: Current StepResult payload containing the `done` flag from the environment.

        Returns:
            True if the environment indicates done, False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Compile a dictionary of step results and outcome metrics.

        For LIBERO, success is typically synonymous with the episode being done,
        as defined by the task's BDDL logic.

        Args:
            step_result: Final StepResult of the episode.

        Returns:
            A results dictionary containing a boolean 'success' indicator.
        """
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve simulation metadata including max_steps and suite.

        This method provides configuration details relevant to the simulation's
        operational parameters.

        Returns:
            A metadata dictionary mapping, including default max_episodes_per_task.
        """
        return {
            "max_steps": self._max_steps or MAX_STEP_MAPPING.get(self.suite, 300),
            "max_episodes_per_task": 50,  # Bounded by the number of initial_states provided per task in LIBERO.
            "suite": self.suite,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Get expected action constraints of the simulation model.

        This defines the format and bounds for actions that can be sent to the simulator.

        Returns:
            A dictionary mapping action keys ('position', 'rotation', 'gripper')
            to their respective DimSpec bounds and dimensions.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Get the observation contract provided by the simulator.

        This defines the format and expected content of observations received from the simulator.

        Returns:
            A dictionary mapping observation keys to DimSpec constraints,
            including image resolution, language type, and optionally state specs.
        """
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_EEF_POS_AA_GRIP
        return spec

    def render(self) -> np.ndarray | None:
        """Render the current environment visual frame.

        Attempts to render the current view from the robosuite environment's
        main camera.

        Returns:
            An image array (HxWx3) representing the current view if rendering succeeds,
            otherwise None (e.g., if rendering is not supported or fails).
        """
        try:
            assert self._env is not None
            return self._env.render()
        except Exception:
            # Handle cases where rendering might not be available (e.g., no active display)
            # or fails due to other environment-specific issues.
            return None

"""RoboCerebra benchmark — long-horizon manipulation on LIBERO/robosuite."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evaluation.tasks.embodied.simulators.rotation import quat_to_axisangle
from worldfoundry.evaluation.tasks.embodied.simulators.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)

logger = logging.getLogger(__name__)
ROBOCEREBRA_ROOT_ENV = "WORLDFOUNDRY_ROBOCEREBRA_ROOT"

# Set EGL and PyOpenGL platforms for offscreen rendering, ensuring compatibility
# with environments like Docker or remote servers without a display.
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Default dummy action used during initial environment steps to let physics settle.
# It consists of 6 position/rotation delta values and a gripper control value (-1.0 for open).
_DUMMY_ACTION = [0.0] * 6 + [-1.0]


class RoboCerebraBenchmark(BaseSimulator):
    """Simulator for RoboCerebra long-horizon manipulation tasks using the LIBERO framework.

    This class interfaces with RoboCerebra benchmark tasks, each defined by a specific directory
    structure under `robocerebra_root/<task_type>/<case>/`. Each task directory typically
    contains:
    - A `.bddl` file specifying the environment configuration.
    - A `demo.hdf5` file for initial environment states.
    - A `task_description.txt` file providing a human-readable task description.
    - A `goal.json` file defining success criteria.

    The simulator handles environment initialization, action execution, observation
    processing, and success evaluation, integrating with `robosuite` and `libero`.

    Args:
        robocerebra_root: Path to the downloaded RoboCerebra_Bench data.
        task_types: A list of task-type folders to include (e.g., "Ideal", "Composite").
                    Defaults to `["Ideal"]`.
        seed: Random seed for environment initialization and reproducibility.
        num_steps_wait: Number of dummy action steps to perform at the beginning of each
                        episode to allow physics to settle.
        send_wrist_image: If True, include observations from the robot's wrist camera.
        send_state: If True, include proprioceptive state (e.g., end-effector pose, gripper position)
                    in observations.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        robocerebra_root: str | os.PathLike[str] | None = None,
        task_types: list[str] | None = None,
        seed: int = 7,
        num_steps_wait: int = 15,
        send_wrist_image: bool = False,
        send_state: bool = False,
    ) -> None:
        """Initializes the RoboCerebraBenchmark simulator.

        Configures paths, task types, rendering options, and internal states.
        """
        super().__init__()
        resolved_root = robocerebra_root if robocerebra_root is not None else os.getenv(ROBOCEREBRA_ROOT_ENV)
        if not resolved_root:
            raise ValueError(
                "RoboCerebra root is required; pass robocerebra_root or set "
                f"{ROBOCEREBRA_ROOT_ENV} (the docker profile passes it explicitly)."
            )
        self.robocerebra_root = Path(resolved_root).expanduser()
        self.task_types = task_types or ["Ideal"]
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self._env: Any = None  # Holds the active robosuite/libero environment instance
        self._current_goal: dict | None = None  # Stores the parsed goal for the current task
        self._libero_inited = False  # Flag to ensure LIBERO is initialized only once

    def cleanup(self) -> None:
        """Safely close and clean up the active RoboCerebra environment.

        This method attempts to close the underlying `robosuite` environment if one is active,
        suppressing any errors during the cleanup process to ensure robust termination.
        """
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._env = None

    # ------------------------------------------------------------------
    def _ensure_libero(self) -> None:
        """Ensures that LIBERO environments and their corresponding task registers are loaded.

        This is a lazy initialization to avoid importing heavy LIBERO modules unless needed.
        """
        if self._libero_inited:
            return
        # Importing this module triggers the registration of all RoboCerebra tasks
        # in `libero.libero.envs.TASK_MAPPING`.
        import libero.libero.envs  # noqa: F401

        self._libero_inited = True

    # ------------------------------------------------------------------
    def get_tasks(self) -> list[dict[str, Any]]:
        """Retrieves RoboCerebra evaluation tasks and case specifications from the root folder.

        It scans the `robocerebra_root` for task directories based on the configured `task_types`,
        identifies BDDL files, and extracts task descriptions to compile a list of task configurations.

        Returns:
            A list of task dictionary configurations, each specifying the task name,
            type, case name, directory path, and BDDL file path.
        """
        self._ensure_libero()
        root = Path(self.robocerebra_root)
        tasks: list[dict[str, Any]] = []
        for task_type in self.task_types:
            type_dir = root / task_type
            if not type_dir.is_dir():
                # Skip if the task type directory does not exist
                continue
            for case_dir in sorted(type_dir.iterdir()):
                if not case_dir.is_dir():
                    # Skip if not a directory (e.g., a file)
                    continue
                bddl_files = list(case_dir.glob("*.bddl"))
                if not bddl_files:
                    # Skip if no BDDL file is found, as it's required for task definition
                    continue
                description = f"{task_type}/{case_dir.name}"
                desc_file = case_dir / "task_description.txt"
                if desc_file.exists():
                    # Parse task description from a specific line in the file
                    # The description is expected to follow a "dict[str, Any]: <description_text>" format
                    for line in desc_file.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("dict[str, Any]:"):
                            description = line.split(":", 1)[1].strip()
                            break
                tasks.append(
                    {
                        "name": description,
                        "task_type": task_type,
                        "case_name": case_dir.name,
                        "task_dir": str(case_dir),
                        "bddl_file": str(bddl_files[0]),
                    }
                )
        return tasks

    # ------------------------------------------------------------------
    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the environment for a new RoboCerebra task case.

        This involves:
        1. Closing any previously active environment.
        2. Loading the BDDL file to configure the specific task.
        3. Initializing a `libero` environment with specified camera and control settings.
        4. Applying an initial state from `demo.hdf5` for consistency if available.
        5. Loading and parsing the `goal.json` file to set success criteria.
        6. Performing initial dummy steps to stabilize the physics simulation.

        Args:
            task: Task dictionary containing `bddl_file` (path to BDDL task definition)
                  and `task_dir` (path to the task's data directory).

        Returns:
            The initial raw observation dictionary from MuJoCo/robosuite after reset and warm-up.
        """
        import h5py
        import libero.libero.envs.bddl_utils as BDDLUtils
        from libero.libero.envs import TASK_MAPPING
        from robosuite import load_controller_config

        if self._env is not None:
            # Clean up any existing environment before creating a new one
            try:
                self._env.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._env = None

        bddl_file = task["bddl_file"]
        task_dir = Path(task["task_dir"])

        problem_info = BDDLUtils.get_problem_info(bddl_file)
        problem_name = problem_info["problem_name"]
        controller_config = load_controller_config(default_controller="OSC_POSE")

        # Initialize the LIBERO environment with specified configurations.
        # This includes camera setup, offscreen rendering, and reward shaping.
        self._env = TASK_MAPPING[problem_name](
            bddl_file_name=bddl_file,
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=False,  # No on-screen renderer
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            ignore_done=True,  # Success is checked externally via _check_success
            use_camera_obs=True,
            reward_shaping=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        self._env.seed(self.seed)
        obs = self._env.reset()

        # Apply initial state from demo.hdf5 if available.
        # This ensures the environment starts from a consistent configuration
        # derived from a demonstration trajectory, crucial for reproducibility.
        h5_path = task_dir / "demo.hdf5"
        if h5_path.exists():
            with h5py.File(str(h5_path), "r") as h5f:
                init_state = h5f["data"]["demo_1"]["states"][0]
            # Set the simulation state from the loaded HDF5 data
            self._env.sim.set_state_from_flattened(init_state)
            self._env.sim.forward()  # Propagate changes in simulation
            self._env._post_process()  # Update internal environment state after state setting
            self._env._update_observables(force=True)  # Ensure observations are updated
            obs = self._env._get_observations()  # Retrieve new observations after state update

        # Load goal for success checking from goal.json if available.
        # The goal file needs conversion from its raw format to the monitor_dict
        # format expected by the internal `_check_success` method.
        goal_path = task_dir / "goal.json"
        if goal_path.exists():
            raw = json.loads(goal_path.read_text())
            # Convert goal.json structure (object_id: [relation_triples])
            # into a format compatible with `_check_success`, which expects
            # a monitor_dict where relations might be nested or have specific keys.
            goal: dict[str, list] = {}
            for obj_id, relations in raw.items():
                processed = []
                for item in relations:
                    # Extract the core relation triple, handling different potential formats
                    if isinstance(item, dict) and "state_pair" in item:
                        triple = item["state_pair"]
                    elif isinstance(item, list):
                        triple = item
                    else:
                        continue  # Skip malformed items
                    # Convert the first element of the triple to lowercase for consistency
                    # This often represents predicates or object properties, which are case-insensitive
                    processed.append([t.lower() if i == 0 else t for i, t in enumerate(triple)])
                goal[obj_id] = processed
            self._current_goal = goal
        else:
            self._current_goal = None

        # Apply dummy wait steps to let the physics engine settle before any real actions.
        # This prevents instability from initial environment setup.
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(_DUMMY_ACTION)

        self._recorder.record_video(self._extract_frame(obs))
        return obs

    # ------------------------------------------------------------------
    def step(self, action: dict[str, Any]) -> StepResult:
        """Applies a step action in the RoboCerebra environment.

        Processes the input action, translates it into the `robosuite` compatible format,
        executes it, and then evaluates the new state for success and records information.

        Args:
            action: Action targets dictionary. Expected to contain "actions" or "action" key
                    with a 7-dimensional array: [x, y, z, roll, pitch, yaw, gripper_control].

        Returns:
            The outcome StepResult of the step action, including observations, reward,
            done status, and additional information (like success).
        """
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"dict[str, Any] dimension mismatch: got {len(raw_action)}, expected 7"

        # Convert gripper action to expected robosuite format:
        # -1.0 for open (when raw_action[-1] < 0), 1.0 for close (when raw_action[-1] >= 0).
        gripper = -1.0 if raw_action[-1] < 0 else 1.0
        # Combine position/rotation deltas with the processed gripper command
        processed = raw_action[:6] + [gripper]

        obs, reward, done, info = self._env.step(processed)

        # Check success via _check_success if a goal is available.
        # This method uses the internal BDDLUtils to verify task completion based on the
        # parsed `goal.json` criteria.
        success = False
        if self._current_goal is not None:
            try:
                # _check_success returns (raw_success_status, reward_info, all_goals_achieved_bool)
                _, _, all_done = self._env._check_success(self._current_goal)
                success = bool(all_done)
                if success:
                    # If all success criteria are met, explicitly mark the episode as done
                    done = True
            except Exception as e:
                logger.warning("Failed to check success criteria: %s", e)
        info["success"] = success  # Add success status to the info dictionary

        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=bool(done), success=success)

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        """Extracts and formats the visual frame from raw observations for video recording.

        Args:
            raw_obs: The raw observation dictionary from the environment, expected to contain
                     an 'agentview_image'.

        Returns:
            The agentview image as a NumPy array with a specific flip for consistency,
            or None if the image is missing or `raw_obs` is not a dictionary.
        """
        if not isinstance(raw_obs, dict):
            return None
        img = raw_obs.get("agentview_image")
        if img is None:
            return None
        # Flip image vertically and horizontally ([::-1, ::-1]) to match the orientation
        # expected by the model and consistent with `make_obs`. Ensures recorded
        # frames align with what the agent perceives.
        return np.ascontiguousarray(img[::-1, ::-1])

    # ------------------------------------------------------------------
    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Converts a raw MuJoCo/robosuite observation to the standard track observation format.

        This method processes raw observations into a standardized dictionary format,
        including images (agentview, and optionally wrist camera) and task descriptions.
        If `send_state` is enabled, it also adds proprioceptive state information.

        Args:
            raw_obs: The raw observation dictionary from robosuite, containing camera images
                     and proprioceptive data.
            task: Task dictionary containing target metadata, specifically "name" for description.

        Returns:
            A standard observation dictionary containing processed images, task description,
            and optionally, proprioceptive state.
        """
        img = raw_obs["agentview_image"]
        # Flip image vertically and horizontally ([::-1, ::-1]) to match conventional
        # image orientations and ensure consistency with how models might be trained or displayed.
        img = img[::-1, ::-1].copy()

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
        }

        if self.send_wrist_image:
            wrist = raw_obs["robot0_eye_in_hand_image"]
            # Flip wrist camera image similarly for consistency
            wrist = wrist[::-1, ::-1].copy()
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            # Concatenate end-effector position (3D), axis-angle rotation (3D),
            # and gripper state (1D) into a single proprioceptive state vector.
            state = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    quat_to_axisangle(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            obs_dict["states"] = state

        return obs_dict

    # ------------------------------------------------------------------
    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the environment episode has terminated.

        Args:
            step_result: The current `StepResult` payload, which includes the `done` status.

        Returns:
            True if the episode is done (either naturally or marked as success), False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Compiles a dictionary of step results and outcome metrics.

        This method extracts the 'success' flag from the `step_result.info` dictionary.

        Args:
            step_result: The final `StepResult` of the episode, containing observations,
                         reward, done status, and an info dictionary.

        Returns:
            A results dictionary containing a boolean indicator for task success.
        """
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        """Retrieves simulation metadata including maximum number of steps allowed per episode.

        Returns:
            A metadata dictionary mapping, currently including 'max_steps'.
        """
        return {"max_steps": 400}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Gets the expected action constraints (specifications) of the simulation model.

        Defines the dimensions and bounds for position, rotation, and gripper control actions.

        Returns:
            A dictionary mapping action keys ("position", "rotation", "gripper") to `DimSpec`
            objects that describe their expected ranges and shapes.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Gets the observation contract provided by the simulator.

        Describes the expected dimensions and types of observations (e.g., images, language,
        and optionally proprioceptive state or wrist images).

        Returns:
            A dictionary mapping observation keys ("agentview", "language", "wrist", "state")
            to `DimSpec` objects that describe their expected shapes and types.
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
        """Renders the current visual frame of the active environment.

        Attempts to retrieve a rendered image from the underlying `robosuite` environment.

        Returns:
            The visual image as a NumPy array (RGB) if rendering is successful,
            or None if rendering fails (e.g., no active environment or renderer issues).
        """
        try:
            return self._env.render()
        except Exception:
            return None

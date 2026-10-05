"""RoboTwin 2.0 benchmark — dual-arm manipulation on SAPIEN/CuRobo.

Ported from the upstream embodied evaluation implementation shipped in the
``robotwin`` Docker image.

Non-obvious behaviors:
    - **Expert check**: ``get_tasks()`` optionally runs the oracle planner
      per seed to verify solvability (``skip_expert_check=False``).
    - **Lazy init**: Heavy imports happen on first use, not at construction.
    - **14D action**: dual-arm qpos; 16D inputs are trimmed to 14D.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evaluation.tasks.embodied.simulators.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec

logger = logging.getLogger(__name__)

ROBOTWIN_ROOT = "/app/RoboTwin"
ROBOTWIN_ROOT_ENV = "WORLDFOUNDRY_ROBOTWIN_ROOT"


def _configured_robotwin_root() -> str:
    value = os.environ.get(ROBOTWIN_ROOT_ENV, ROBOTWIN_ROOT)
    return os.path.abspath(os.path.expanduser(value))


@contextmanager
def _working_directory(path: str | os.PathLike[str]):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _EvalGripperPlanner:
    """Minimal planner shim for eval-only RoboTwin startup.

    RoboTwin's qpos evaluation path still calls ``plan_grippers()`` during
    env setup, but it never uses CuRobo path planning afterwards.  This shim
    keeps gripper interpolation working while avoiding the expensive CuRobo
    warmup in ``Robot.set_planner()``.
    """

    def plan_grippers(self, now_val: float, target_val: float) -> dict[str, Any]:
        """Interpolate gripper values over steps.

        This method generates a sequence of gripper values from the current
        to the target value over a fixed number of steps.

        Args:
            now_val: Current gripper pose value.
            target_val: Target gripper pose value.

        Returns:
            A dictionary containing planned steps count, increment, and trajectory values.
        """
        num_step = 200
        per_step = (target_val - now_val) / num_step
        vals = np.linspace(now_val, target_val, num_step)
        return {"num_step": num_step, "per_step": per_step, "result": vals}

    def update_point_cloud(self, _world_pcd: Any, _resolution: float = 0.02) -> None:
        """Update point cloud in the mock planner.

        This method is a no-op in the evaluation context as point cloud
        updates are not relevant for the mocked gripper planner.

        Args:
            world_pcd: Input point cloud.
            resolution: Resolution.
        """
        return None

    def plan_path(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Placeholder for plan_path, disabled in eval.

        Raises:
            RuntimeError: Always raised since CuRobo path planning is explicitly disabled
                          during episode execution in the evaluation fast-path.
        """
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo path planning during episode execution.")

    def plan_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Placeholder for plan_batch, disabled in eval.

        Raises:
            RuntimeError: Always raised since CuRobo batch planning is explicitly disabled
                          during episode execution in the evaluation fast-path.
        """
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo batch planning during episode execution.")


class _LazyOpen3D(types.ModuleType):
    """Imports open3d only when one of its attributes is first accessed.

    This avoids an expensive import during initial module loading if open3d is not needed.
    It acts as a proxy that loads the real module on demand.
    """

    def __init__(self) -> None:
        """Initialize lazy module."""
        super().__init__("open3d")
        self._real_module: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        """Load and cache the actual open3d module.

        Returns:
            The loaded open3d module.
        """
        if self._real_module is not None:
            return self._real_module

        # Remove the proxy from sys.modules before importing the real module
        # to prevent recursive imports if 'open3d' tries to import itself.
        if sys.modules.get("open3d") is self:
            sys.modules.pop("open3d", None)
        try:
            module = importlib.import_module("open3d")
        except Exception:
            # If import fails, restore the proxy to allow retrying or consistent state.
            sys.modules["open3d"] = self
            raise
        # Update proxy's dictionary with real module's attributes and cache it.
        # This makes subsequent attribute accesses directly hit the cached attributes.
        self.__dict__.update(module.__dict__)
        self._real_module = module
        # Replace the proxy with the real module in sys.modules.
        # From this point on, direct imports of 'open3d' will get the real module.
        sys.modules["open3d"] = module
        return module

    def __getattr__(self, name: str) -> Any:
        """Retrieve attribute from lazy module on first access.

        This method is called when an attribute is accessed on the proxy
        and is not found in the proxy's own dictionary. It triggers the
        actual loading of the `open3d` module.

        Args:
            name: Attribute name.

        Returns:
            The corresponding attribute value from the loaded `open3d` module.
        """
        return getattr(self._load(), name)


@contextmanager
def _defer_open3d_import(enabled: bool):
    """Context manager to defer open3d import during RoboTwin module import when pointclouds are unused.

    This temporarily replaces the `open3d` entry in `sys.modules` with a lazy proxy.
    The actual `open3d` module is only loaded if one of its attributes is accessed.

    Args:
        enabled: Whether deferring is enabled. If False, the context manager does nothing.
    """
    if not enabled:
        yield
        return

    # Store existing 'open3d' entry in sys.modules to restore it later.
    previous = sys.modules.get("open3d")
    # Replace 'open3d' in sys.modules with a lazy proxy.
    proxy = _LazyOpen3D()
    sys.modules["open3d"] = proxy
    try:
        yield
    finally:
        # If the proxy is still in sys.modules (meaning open3d wasn't actually loaded or
        # was loaded via the proxy), clean it up.
        if sys.modules.get("open3d") is proxy:
            if previous is None:
                # If there was no previous 'open3d' entry, remove it completely.
                sys.modules.pop("open3d", None)
            else:
                # Otherwise, restore the original 'open3d' entry.
                sys.modules["open3d"] = previous


def _make_fast_set_planner(robot_mod: Any):
    """Create a fast mock set_planner method for robots to skip CuRobo warmup.

    This function returns a new `set_planner` method that replaces the original
    `Robot.set_planner` to bypass the heavy CuRobo planner initialization
    during evaluation when only gripper planning is needed.

    Args:
        robot_mod: The robot module reference, typically `envs.robot.robot`.

    Returns:
        The fast `_set_planner_fast` function, designed to replace `Robot.set_planner`.
    """
    def _set_planner_fast(self: Any, scene: Any = None) -> None:
        """A mocked `set_planner` that uses `_EvalGripperPlanner` instead of CuRobo.

        This method initializes gripper planners with `_EvalGripperPlanner` to handle
        gripper interpolation, but avoids the full CuRobo planner setup for path planning.
        It preserves the `communication_flag` and `need_topp` logic of the original
        method to maintain compatible behavior where necessary.

        Args:
            self: The robot instance.
            scene: The simulation scene (optional, but passed by original method).
        """
        self.communication_flag = False
        # Initialize light-weight gripper planners to handle gripper control.
        self.left_planner = _EvalGripperPlanner()
        self.right_planner = _EvalGripperPlanner()

        # If topp (Time-Optimal Path Planning) is needed, still initialize MplilbPlanner,
        # but its internal plan_path will not be called for qpos evaluation as it's
        # bypassed by the _EvalGripperPlanner. This ensures that the robot object
        # has the expected attributes, even if they aren't fully utilized for path planning.
        if self.need_topp:
            self.left_mplib_planner = robot_mod.MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = robot_mod.MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    return _set_planner_fast


@contextmanager
def _patched_robot_set_planner(enabled: bool):
    """Context manager to temporarily skip CuRobo planner warmup during env setup.

    This is achieved by patching `envs.robot.robot.Robot.set_planner` with a
    lighter version that uses `_EvalGripperPlanner`. This significantly
    reduces initialization time for evaluation episodes that do not rely
    on CuRobo for full path planning.

    Args:
        enabled: Whether patching is enabled. If False, the context manager does nothing.
    """
    if not enabled:
        yield
        return

    # Import the robot module dynamically as it might not be loaded yet.
    import envs.robot.robot as robot_mod

    # Store the original method to restore it later, ensuring proper cleanup.
    original = robot_mod.Robot.set_planner
    # Replace the method with the fast mock created by `_make_fast_set_planner`.
    robot_mod.Robot.set_planner = _make_fast_set_planner(robot_mod)
    try:
        yield
    finally:
        # Restore the original method after exiting the context, cleaning up the patch.
        robot_mod.Robot.set_planner = original


@contextmanager
def _patched_render_setup(enabled: bool):
    """Context manager to temporarily use SAPIEN's default shader during env setup.

    This bypasses RoboTwin's custom ray-traced renderer setup, potentially speeding
    up environment initialization and rendering, but might alter observation fidelity.
    It specifically patches SAPIEN's rendering configuration functions.

    Args:
        enabled: Whether patching is enabled. If False, the context manager does nothing.
    """
    if not enabled:
        yield
        return

    # Import sapien.render dynamically.
    import sapien.render as sapien_render

    # Store original functions to restore them later, ensuring proper cleanup.
    originals = {
        "set_camera_shader_dir": sapien_render.set_camera_shader_dir,
        "set_ray_tracing_samples_per_pixel": sapien_render.set_ray_tracing_samples_per_pixel,
        "set_ray_tracing_path_depth": sapien_render.set_ray_tracing_path_depth,
        "set_ray_tracing_denoiser": sapien_render.set_ray_tracing_denoiser,
    }

    def _set_camera_shader_dir_fast(_shader_dir: str) -> None:
        """Mock `set_camera_shader_dir` to always use the 'default' shader.

        This overrides any attempts by RoboTwin to set custom shaders,
        forcing the use of SAPIEN's default for faster rendering.
        """
        originals["set_camera_shader_dir"]("default")

    # Apply patches: replace set_camera_shader_dir to enforce 'default' and
    # make ray tracing settings no-ops to prevent expensive ray tracing initialization.
    sapien_render.set_camera_shader_dir = _set_camera_shader_dir_fast
    sapien_render.set_ray_tracing_samples_per_pixel = lambda _spp: None
    sapien_render.set_ray_tracing_path_depth = lambda depth: None
    sapien_render.set_ray_tracing_denoiser = lambda name: None
    try:
        yield
    finally:
        # Restore all original functions after exiting the context, cleaning up patches.
        for name, func in originals.items():
            setattr(sapien_render, name, func)


class RoboTwinBenchmark(BaseSimulator):
    """RoboTwin dual-arm manipulation benchmark (SAPIEN/CuRobo).

    This class provides an interface to the RoboTwin simulation environment,
    allowing tasks to be loaded, reset, and stepped through for evaluation
    within the `worldfoundry` framework. It supports optimizations like
    skipping expert checks, fast initialization by bypassing CuRobo planner
    warmup, and faster rendering using SAPIEN's default shader.

    Args:
        task_name: RoboTwin task identifier (e.g., ``"grab_roller"``).
        task_config: Configuration name for the task, located under ``task_config/``
                     (default ``"demo_clean"``).
        seed: Base seed index for episode generation. The starting seed for
              each episode is derived as ``100000 * (1 + seed)``.
        instruction_type: Variant of task instructions to use (``"seen"`` or ``"unseen"``).
        test_num: Number of distinct, valid episodes to evaluate.
        skip_expert_check: If ``True``, bypasses the oracle planner verification in
                           ``get_tasks()``. This is useful for quick task discovery checks
                           but might include unsolvable tasks.
        fast_init: If ``True``, skips the CuRobo planner warmup for evaluation
                   episodes after task discovery. This substantially reduces
                   cold-start time by using a mock gripper planner.
        fast_render: If ``True``, uses SAPIEN's default camera shader instead of
                     RoboTwin's ray-traced renderer. This is faster but may
                     result in observations that differ from the reference benchmark.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean",
        seed: int = 0,
        instruction_type: str = "seen",
        test_num: int = 100,
        skip_expert_check: bool = False,
        fast_init: bool = True,
        fast_render: bool = False,
        robotwin_root: str | os.PathLike[str] | None = None,
    ) -> None:
        import re

        super().__init__()
        # Validate task_name and task_config against expected patterns for security and consistency.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task_name):
            raise ValueError(f"Invalid task_name: {task_name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_config):
            raise ValueError(f"Invalid task_config: {task_config!r}")
        self.task_name = task_name
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.test_num = test_num
        self.skip_expert_check = skip_expert_check
        self.fast_init = fast_init
        self.fast_render = fast_render
        self.robotwin_root = Path(
            robotwin_root or os.getenv(ROBOTWIN_ROOT_ENV) or ROBOTWIN_ROOT
        ).expanduser()
        self._env: Any = None
        self._env_class: Any = None
        self._args: dict[str, Any] | None = None

    # -----------------------------------------------------------------
    # Lazy init
    # -----------------------------------------------------------------

    def _init_robotwin(self) -> None:
        """Add RoboTwin paths, load YAML configs, and resolve embodiment configurations lazily.

        This method is called on first use to ensure all necessary RoboTwin
        dependencies and configurations are set up. It modifies `sys.path` and
        `os.chdir` temporarily for RoboTwin's module loading logic.
        """
        if self._args is not None:  # Already initialized
            return

        robotwin_root = self.robotwin_root
        # Add RoboTwin root and policy directories to sys.path to enable module imports.
        # This is necessary because RoboTwin expects its modules to be discoverable relative to its root.
        for path in (robotwin_root, robotwin_root / "policy", robotwin_root / "description" / "utils"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)

        # RoboTwin still resolves some imports/configs relative to its checkout.  Keep that
        # compatibility local to initialisation instead of leaking a process-wide cwd change.
        with _working_directory(robotwin_root):
            self._load_robotwin_config(robotwin_root)

    def _load_robotwin_config(self, robotwin_root: str) -> None:
        import yaml

        # Load the task-specific configuration YAML.
        config_path = os.path.join(
            robotwin_root,
            "task_config",
            f"{self.task_config}.yml",
        )
        with open(config_path) as f:
            args: dict[str, Any] = yaml.safe_load(f)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config

        # Resolve embodiment configurations based on the loaded YAML.
        from envs import CONFIGS_PATH

        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
            _embodiment_types = yaml.safe_load(f)

        def _get_file(etype: str) -> str:
            return _embodiment_types[etype]["file_path"]

        # Handle single-arm vs. dual-arm embodiment configurations by populating robot file paths.
        # The logic depends on the number of entries in `embodiment_type`.
        if len(embodiment_type) == 1:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        # Load specific robot embodiment configurations from their respective YAML files.
        def _get_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml")) as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_config(args["left_robot_file"])
        args["right_embodiment_config"] = _get_config(args["right_robot_file"])

        # Load camera configurations and set head camera dimensions based on the config.
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
            _camera_config = yaml.safe_load(f)

        hcam = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[hcam]["h"]
        args["head_camera_w"] = _camera_config[hcam]["w"]
        args["eval_mode"] = True

        # Defer open3d import if pointcloud data is not used, to speed up initialization.
        # This is a critical optimization as open3d can be a heavy dependency and
        # may not be needed for all evaluation setups.
        with _defer_open3d_import(enabled=not args.get("data_type", {}).get("pointcloud", False)):
            envs_module = importlib.import_module(f"envs.{self.task_name}")
        env_class = getattr(envs_module, self.task_name)
        self._args = args
        self._env_class = env_class
        logger.info("RoboTwin initialised: task=%s", self.task_name)

    def _create_env(self) -> Any:
        """Instantiate the designated RoboTwin task environment class.

        This method should only be called after `_init_robotwin` has completed,
        which ensures that `self._env_class` is correctly set.

        Returns:
            The instantiated environment.
        """
        assert self._env_class is not None
        with _working_directory(self.robotwin_root):
            return self._env_class()

    def cleanup(self) -> None:
        """Safely close and clean up active RoboTwin SAPIEN environment allocations.

        This ensures proper resource release, preventing potential memory leaks
        or simulation conflicts if multiple environments are created or if the
        simulator is reinitialized. Errors during cleanup are suppressed to
        allow robust operation even if the environment is in an invalid state.
        """
        if self._env is not None:
            try:
                with _working_directory(self.robotwin_root):
                    self._env.close_env(clear_cache=True)
            except Exception:
                # Suppress errors during cleanup if the environment is already in a bad state.
                pass
            self._env = None

    # -----------------------------------------------------------------
    # BaseSimulator interface
    # -----------------------------------------------------------------

    def get_tasks(self) -> list[dict[str, Any]]:
        """Generate a list of task specifications for evaluation.

        This method optionally performs an expert check to ensure task solvability
        by the oracle planner before including them in the evaluation set. If
        `skip_expert_check` is True, it generates tasks directly without verification,
        which is faster but may include unsolvable tasks.

        Returns:
            A list of dictionaries, each describing a unique task episode,
            including name, suite, seed, episode index, and instruction.
        """
        self._init_robotwin()
        assert self._args is not None
        st_seed = 100000 * (1 + self.seed)

        # If expert check is skipped, generate tasks directly based on test_num and seed.
        # This path is faster as it avoids running the simulation for each episode.
        if self.skip_expert_check:
            return [
                {
                    "name": self.task_name,
                    "suite": "robotwin",
                    "seed": st_seed + i,
                    "episode_idx": i,
                    "instruction": f"Perform the {self.task_name} task.",
                }
                for i in range(self.test_num)
            ]

        # Full expert check — run oracle planner per seed to verify solvability.
        # This is an expensive process but ensures all generated tasks are solvable by the oracle.
        with _working_directory(self.robotwin_root):
            from generate_episode_instructions import generate_episode_descriptions

        env = self._create_env()
        tasks: list[dict[str, Any]] = []
        now_seed = st_seed
        episode_idx = 0
        logger.info("Running expert checks from seed %d ...", st_seed)

        while len(tasks) < self.test_num:
            try:
                with _working_directory(self.robotwin_root):
                    # Setup environment for a demo run with the current seed.
                    env.setup_demo(
                        now_ep_num=episode_idx,
                        seed=now_seed,
                        is_test=True,
                        **self._args,
                    )
                    # Attempt to play out the episode using the oracle planner.
                    episode_info = env.play_once()
                    env.close_env()
                    plan_succeeded = bool(env.plan_success and env.check_success())
                # Check if the oracle planner succeeded and the task was completed.
                if plan_succeeded:
                    # Generate human-readable instructions for the successful episode.
                    with _working_directory(self.robotwin_root):
                        results = generate_episode_descriptions(
                            self.task_name,
                            [episode_info["info"]],
                            self.test_num,
                        )
                    # Select an instruction based on the specified type ("seen" or "unseen").
                    instruction = np.random.choice(
                        results[0][self.instruction_type],
                    )
                    # Add the valid task to the list of tasks for evaluation.
                    tasks.append(
                        {
                            "name": self.task_name,
                            "suite": "robotwin",
                            "seed": now_seed,
                            "episode_idx": episode_idx,
                            "instruction": instruction,
                        }
                    )
                    episode_idx += 1
            except Exception as e:
                # Log any failures during the expert check but continue with the next seed.
                # This ensures that the process doesn't halt on a single difficult seed.
                logger.warning("Expert check failed for seed %d: %s", now_seed, e)
                try:
                    with _working_directory(self.robotwin_root):
                        env.close_env()
                except Exception:
                    # Further suppress errors if closing the environment also fails.
                    pass
            now_seed += 1
        return tasks

    def reset(self, task: dict[str, Any]) -> Any:
        """Reset the environment to the beginning of a new task episode.

        This method initializes the RoboTwin environment with the given task
        parameters and applies optional performance patches for initialization
        and rendering. It also handles cleanup of any previous environment.

        Args:
            task: A dictionary containing task details, including `seed` and `instruction`.

        Returns:
            The initial observation from the reset environment.
        """
        self._init_robotwin()
        assert self._args is not None

        # Clean up any previously active environment before creating a new one.
        if self._env is not None:
            try:
                with _working_directory(self.robotwin_root):
                    self._env.close_env(clear_cache=True)
            except Exception as e:
                logger.warning("Failed to close previous RoboTwin env: %s", e)
            self._env = None

        self._env = self._create_env()
        with _working_directory(self.robotwin_root):
            # Apply patches for faster initialization and rendering if enabled.
            # These context managers temporarily modify RoboTwin's behavior to optimize performance.
            with _patched_robot_set_planner(self.fast_init), _patched_render_setup(self.fast_render):
                self._env.setup_demo(
                    now_ep_num=task.get("episode_idx", 0),
                    seed=task["seed"],
                    is_test=True,
                    **self._args,
                )
            self._env.set_instruction(instruction=task["instruction"])
            raw_obs = self._env.get_obs()
        self._recorder.record_video(self._extract_frame(raw_obs))
        return raw_obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Execute a single action in the environment and return the result.

        This method processes the input action, converts it to the expected 14D
        qpos format, executes it in the RoboTwin environment, and captures
        the resulting observation, reward, and done status.

        Args:
            action: A dictionary containing the action to be taken, typically under "actions" or "action".

        Returns:
            A StepResult object containing the observation, reward, done status, and info.
        """
        raw = action.get("actions", action.get("action"))
        act = np.asarray(raw, dtype=np.float64).flatten()
        # RoboTwin expects 14D actions (dual-arm joint positions: 7 for left arm, 7 for right arm).
        # Trim or pad the input action to match the expected dimension, ensuring compatibility.
        if len(act) > 14:
            act = act[:14]
        elif len(act) < 14:
            act = np.pad(act, (0, 14 - len(act)))
        assert act.shape[-1] == 14, f"dict[str, Any] dimension mismatch: got {act.shape[-1]}, expected 14"

        with _working_directory(self.robotwin_root):
            self._env.take_action(act, action_type="qpos")
            raw_obs = self._env.get_obs()
            success = bool(self._env.eval_success)
            done = success or (self._env.take_action_cnt >= self._env.step_lim)
        self._recorder.record_video(self._extract_frame(raw_obs))
        self._recorder.record_step(reward=1.0 if success else 0.0, done=done, success=success)
        return StepResult(obs=raw_obs, reward=1.0 if success else 0.0, done=done, info={"success": success})

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        """Extract the RGB camera image from the head camera.

        This static method safely retrieves the head camera RGB image from
        the raw observation dictionary, handling cases where the expected keys
        might be missing or the observation is not a dictionary. It's used for video recording.

        Args:
            raw_obs: Raw observation dictionary from the RoboTwin environment.

        Returns:
            The head camera image array as a NumPy array (H, W, 3) in RGB format,
            or None if the image data cannot be found or accessed.
        """
        if not isinstance(raw_obs, dict):
            return None
        try:
            return np.asarray(raw_obs["observation"]["head_camera"]["rgb"])
        except (KeyError, TypeError):
            # If the expected keys are missing or types are incorrect, return None,
            # indicating that a frame could not be extracted.
            return None

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Convert raw environment observations into a standardized format.

        This method transforms the RoboTwin-specific raw observation dictionary
        into a common format expected by the evaluation harness, including
        images from multiple cameras, the task description, and robot joint state.

        Args:
            raw_obs: The raw observation dictionary returned by the environment.
            task: The task dictionary, used for retrieving instruction if not present in raw_obs.

        Returns:
            A standardized observation dictionary suitable for agents,
            containing "images", "task_description", and "joint_state".
        """
        return {
            "images": {
                "head_camera": raw_obs["observation"]["head_camera"]["rgb"],
                "left_camera": raw_obs["observation"]["left_camera"]["rgb"],
                "right_camera": raw_obs["observation"]["right_camera"]["rgb"],
            },
            # Prioritize language from raw_obs, fall back to task instruction if missing.
            "task_description": raw_obs.get(
                "language",
                task.get("instruction", ""),
            ),
            "joint_state": np.array(raw_obs["joint_action"]["vector"]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        """Check if the episode is done based on the step result.

        Args:
            step_result: The result of a single environment step.

        Returns:
            True if the episode is done, False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Extract the relevant outcome information from a step result.

        Args:
            step_result: The result of a single environment step.

        Returns:
            A dictionary containing key outcome metrics, e.g., 'success'.
        """
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        """Return static metadata about the simulation environment.

        Returns:
            A dictionary containing metadata such as maximum steps, task name, and action dimensions.
        """
        return {
            "max_steps": 400,
            "task_name": self.task_name,
            "action_dim": 14,  # Corresponds to 7 joints per arm for dual-arm control.
            "max_episodes_per_task": self.test_num,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Return the specification for the actions accepted by the simulator.

        RoboTwin uses 14D dual-arm joint positions as actions, where each value
        represents a target joint position for one of the 14 controlled joints.

        Returns:
            A dictionary mapping action names to their dimension specifications.
        """
        return {
            "joints": DimSpec("joints", 14, "joint_positions"),
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Return the specification for the observations provided by the simulator.

        This outlines the structure and types of data an agent can expect
        from the RoboTwin environment.

        Returns:
            A dictionary mapping observation names to their dimension specifications.
        """
        return {
            "head_camera": IMAGE_RGB,
            "left_camera": IMAGE_RGB,
            "right_camera": IMAGE_RGB,
            "state": STATE_JOINT,
            "language": LANGUAGE,
        }

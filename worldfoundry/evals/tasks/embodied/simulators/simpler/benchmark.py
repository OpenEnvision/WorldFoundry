"""SimplerEnv benchmark implementation.

Two evaluation protocols:

**Visual Matching (VM)** -- the default.  Uses
``simpler_env.make(task_name)`` which internally calls
``gym.make(env_name, obs_mode="rgbd", prepackaged_config=True)``.
The prepackaged config sets the correct control mode, scene, robot,
camera, RGB overlay, and robot init position for each task.

**Variant Aggregation (VA)** -- activated when *env_name* is set.
Calls ``gym.make(env_name, ...)`` directly with explicit scene,
lighting, distractor, and camera overrides.  Robot/object init
positions are controlled via *init_config* to reproduce the exact
grids from the official SimplerEnv evaluation scripts.

Success modes (set via model server ``get_observation_params()``):
    - ``truncation``: Run until ``max_episode_steps``.  Success =
      ``terminated`` on the final step.
    - ``early_stop``: Stop on the first ``terminated=True``.  Matches
      X-VLA official eval (``if done: break``).
    - ``accumulate``: Run until ``max_episode_steps``.  Success =
      ``terminated`` at any point during the episode.  Matches GR00T
      official eval (OR-accumulation).
"""

from __future__ import annotations

import contextlib
from itertools import product
from typing import Any, cast

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_EULER,
    DimSpec,
)


def _linspace_from_range(spec: list[float]) -> np.ndarray:
    """Convert ``[start, end, n]`` to ``np.linspace(start, end, int(n))``.

    Args:
        spec: A list specifying start, end, and steps.

    Returns:
        The generated linspace numpy array.
    """
    return np.linspace(spec[0], spec[1], int(spec[2]))


class SimplerEnvBenchmark(BaseSimulator):
    """SimplerEnv (ManiSkill2 real2sim) benchmark.

    Args:
        task_name: SimplerEnv task identifier (e.g. ``"widowx_stack_cube"``).
            Must be a key in ``simpler_env.ENVIRONMENT_MAP``.
        max_episode_steps: Override environment's default episode length.
            ``None`` keeps the prepackaged default.  X-VLA uses 1200,
            GR00T uses 10000, starVLA/DB-CogACT use 120.
        success_mode: How to determine episode success:
            ``"truncation"`` -- success = terminated on the final step.
            ``"early_stop"`` -- stop on first terminated, count as success.
            ``"accumulate"`` -- run to end, success if ever terminated.
        send_state: Include proprioceptive state (base_pose, tcp_pose,
            EE pose) in observations for models that need it.
        seed: Random seed for ``env.reset()``.
        env_name: Gymnasium env ID for VA evaluation.  When set, calls
            ``gym.make()`` directly instead of ``simpler_env.make()``.
        scene_name: Scene name passed to ``gym.make()`` (VA only).
        env_build_kwargs: Extra kwargs forwarded to ``gym.make()``
            (VA only).  Lighting, distractors, shader, etc.
        init_config: Position grid config for VA evaluation.  Keys:
            ``robot_x/robot_y`` ([start, end, n] linspace),
            ``robot_rot_quat`` ([x, y, z, w] quaternion),
            ``obj_x/obj_y`` ([start, end, n] linspace, xy mode),
            ``obj_episode_range`` ([start, end) end-exclusive, episode mode).
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "terminated", "truncated", "success"})

    def __init__(
        self,
        task_name: str = "widowx_stack_cube",
        max_episode_steps: int | None = None,
        success_mode: str = "truncation",
        send_state: bool = False,
        seed: int | None = None,
        deterministic_episodes: bool = True,
        control_mode: str | None = None,
        gripper_mode: str = "binary",
        env_name: str | None = None,
        scene_name: str | None = None,
        env_build_kwargs: dict[str, Any] | None = None,
        init_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert success_mode in ("truncation", "early_stop", "accumulate"), (
            f"Invalid success_mode={success_mode!r}. Expected: truncation, early_stop, accumulate"
        )
        assert gripper_mode in ("binary", "sticky"), f"Invalid gripper_mode={gripper_mode!r}. Expected: binary, sticky"
        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.success_mode = success_mode
        self.send_state = send_state
        self.seed = seed
        self.deterministic_episodes = deterministic_episodes
        self.control_mode = control_mode
        self.gripper_mode = gripper_mode
        self.env_name = env_name
        self.scene_name = scene_name
        self.env_build_kwargs = env_build_kwargs or {}

        self._env: Any = None
        self._task_description: str = ""
        self._success_seen: bool = False
        # State for Google Robot's sticky gripper mechanism
        self._sticky_action_is_on: bool = False
        self._sticky_gripper_action: float = 0.0
        self._gripper_action_repeat: int = 0
        self._sticky_gripper_num_repeat: int = 15

        # Precompute VA position grid if init_config is provided
        self._va_grid: list[tuple[np.ndarray, dict[str, Any]]] | None = None
        if init_config is not None:
            self._va_grid = self._build_position_grid(init_config)

    @staticmethod
    def _build_position_grid(cfg: dict[str, Any]) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Build a flat list of ``(robot_xy, obj_options)`` from *init_config*.

        Each entry maps to one episode.  The grid order matches the
        nested-for-loop order in SimplerEnv's evaluation scripts:
        outer = robot positions, inner = object positions / episode ids.

        Args:
            cfg: The initialization grid config dictionary.

        Returns:
            A list of tuples containing robot position arrays and target options.
        """
        # Parse robot X and Y ranges, defaulting to a single point if not specified
        robot_xs = _linspace_from_range(cfg["robot_x"]) if "robot_x" in cfg else np.array([0.0])
        robot_ys = _linspace_from_range(cfg["robot_y"]) if "robot_y" in cfg else np.array([0.0])
        # Parse robot rotation quaternion, defaulting to identity if not specified
        rot_quat = np.asarray(cfg.get("robot_rot_quat", [0, 0, 0, 1]), dtype=np.float64)

        grid: list[tuple[np.ndarray, dict[str, Any]]] = []

        # Handle object initialization based on episode ID or specific XY coordinates
        if "obj_episode_range" in cfg:
            # If object initialization uses an episode ID range
            ep_start, ep_end = int(cfg["obj_episode_range"][0]), int(cfg["obj_episode_range"][1])
            for rx, ry in product(robot_xs, robot_ys):
                robot_xy = np.array([rx, ry])
                for ep_id in range(ep_start, ep_end):
                    grid.append(
                        (
                            robot_xy,
                            {"robot_rot_quat": rot_quat, "obj_init_options": {"episode_id": ep_id}},
                        )
                    )
        else:
            # If object initialization uses explicit XY coordinates
            obj_xs = _linspace_from_range(cfg["obj_x"]) if "obj_x" in cfg else np.array([0.0])
            obj_ys = _linspace_from_range(cfg["obj_y"]) if "obj_y" in cfg else np.array([0.0])
            for rx, ry in product(robot_xs, robot_ys):
                robot_xy = np.array([rx, ry])
                for ox, oy in product(obj_xs, obj_ys):
                    grid.append(
                        (
                            robot_xy,
                            {"robot_rot_quat": rot_quat, "obj_init_options": {"init_xy": np.array([ox, oy])}},
                        )
                    )
        return grid

    def _build_va_reset_kwargs(self, task: dict[str, Any]) -> dict[str, Any]:
        """Map ``task["episode_idx"]`` to a position in the precomputed grid.

        Args:
            task: Task dictionary settings.

        Returns:
            A reset configuration dictionary.
        """
        assert self._va_grid is not None
        # Use modulo to cycle through the precomputed VA grid for episode-specific initializations
        idx = task.get("episode_idx", 0) % len(self._va_grid)
        # Extract robot and object initialization parameters from the grid
        robot_xy, extra = self._va_grid[idx]
        return {
            "options": {
                "robot_init_options": {
                    "init_xy": robot_xy,
                    "init_rot_quat": extra["robot_rot_quat"],
                },
                "obj_init_options": extra["obj_init_options"],
            },
        }

    def cleanup(self) -> None:
        """Safely close and clean up the active Simpler environment."""
        if self._env is not None:
            # Ignore errors during cleanup
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[dict[str, Any]]:
        """Returns a list of tasks for the benchmark.

        For SimplerEnv, this typically returns a single task dictionary
        representing the configured environment.
        """
        return [{"name": self.task_name, "task_name": self.task_name}]

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the environment for a new episode.

        Initializes a new environment instance, applies task-specific
        reset parameters, and returns the initial observation.

        Args:
            task: A dictionary containing task-specific information,
                  such as episode index.

        Returns:
            The initial observation from the environment.
        """
        # Close previous env -- new env per episode (matches reference) to ensure isolation
        self._success_seen = False
        # Reset sticky gripper state for the new episode
        self._sticky_action_is_on = False
        self._sticky_gripper_action = 0.0
        self._gripper_action_repeat = 0
        if self._env is not None:
            self._env.close()

        # Create the environment based on whether VA or VM protocol is used
        if self.env_name is not None:
            self._env, reset_kwargs = self._make_va_env(task)
        else:
            self._env, reset_kwargs = self._make_vm_env(task)
        # Apply random seed if specified
        if self.seed is not None:
            reset_kwargs["seed"] = self.seed

        obs, _ = self._env.reset(**reset_kwargs)

        # Retrieve and store the task description provided by the environment
        try:
            self._task_description = self._env.unwrapped.get_language_instruction()
        except AttributeError:
            self._task_description = self._env.get_wrapper_attr("get_language_instruction")()

        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def _common_make_kwargs(self) -> dict[str, Any]:
        """Build kwargs shared by both VA and VM env creation.

        Returns:
            A dictionary containing shared arguments.
        """
        kw: dict[str, Any] = {}
        if self.control_mode is not None:
            kw["control_mode"] = self.control_mode
        if self.max_episode_steps is not None:
            kw["max_episode_steps"] = self.max_episode_steps
        return kw

    def _make_va_env(self, task: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """VA path: ``gym.make()`` with explicit scene/lighting/kwargs.

        Args:
            task: Task dictionary settings.

        Returns:
            A tuple of (gym_env, reset_kwargs).
        """
        gym = __import__("gymnasium")
        __import__("simpler_env")  # registers ManiSkill2 envs

        # Construct make_kwargs, merging common args and VA-specific build kwargs
        make_kwargs: dict[str, Any] = {"obs_mode": "rgbd", **self.env_build_kwargs, **self._common_make_kwargs()}
        if self.scene_name is not None:
            make_kwargs["scene_name"] = self.scene_name
        env = gym.make(cast(str, self.env_name), **make_kwargs)

        # Build reset_kwargs using the precomputed VA grid if available
        reset_kwargs = self._build_va_reset_kwargs(task) if self._va_grid is not None else {}
        return env, reset_kwargs

    def _make_vm_env(self, task: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """VM path: ``simpler_env.make()`` with prepackaged config.

        Args:
            task: Task dictionary settings.

        Returns:
            A tuple of (simpler_env, reset_kwargs).
        """
        simpler_env = __import__("simpler_env")

        env = simpler_env.make(self.task_name, **self._common_make_kwargs())

        # deterministic_episodes=True: pass episode_id for reproducible
        # object placement (matches X-VLA, starVLA reference evals).
        # deterministic_episodes=False: random placement each reset
        # (matches GR00T with vectorized envs + auto-reset).
        reset_kwargs: dict[str, Any] = {}
        if self.deterministic_episodes:
            episode_idx = task.get("episode_idx", 0)
            reset_kwargs["options"] = {"obj_init_options": {"episode_id": episode_idx}}
        return env, reset_kwargs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Applies an action to the environment and processes the result.

        Args:
            action: A dictionary containing the action to be taken, expected
                    to have a key 'actions' or 'action' with a 7-dimensional
                    array ([pos3, rot3, gripper]).

        Returns:
            A StepResult object containing the observation, reward, done status, and info.
        """
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"dict[str, Any] dimension mismatch: got {len(raw_action)}, expected 7"

        # Separate position, rotation, and gripper components from the raw action.
        # [pos3, rot3, gripper] — pass directly to env.step().
        # No rotation conversion: the controller (arm_pd_ee_target_delta_pose_align2)
        # uses Rotation.from_rotvec() which interprets action[3:6] as a rotation vector.
        # All reference implementations feed their rotation values straight through.
        pos = np.array(raw_action[:3])
        rot = np.array(raw_action[3:6])

        # Handle different gripper control modes (binary vs. sticky)
        if self.gripper_mode == "sticky":
            # Google Robot sticky gripper logic: relative gripper action with a repeat mechanism
            g = float(raw_action[6]) * 2 - 1  # Convert [0,1] gripper action to [-1,1] relative action
            relative = -g  # Gripper open/close direction is inverted for `relative`
            # Activate sticky action if a significant relative gripper action is detected and not already active
            if abs(relative) > 0.5 and not self._sticky_action_is_on:
                self._sticky_action_is_on = True
                self._sticky_gripper_action = relative
            # If sticky action is on, repeat the previously set sticky gripper action
            if self._sticky_action_is_on:
                self._gripper_action_repeat += 1
                relative = self._sticky_gripper_action
            # Deactivate sticky action after the specified number of repeats
            if self._gripper_action_repeat == self._sticky_gripper_num_repeat:
                self._sticky_action_is_on = False
                self._gripper_action_repeat = 0
                self._sticky_gripper_action = 0.0
            gripper = relative
        else:
            # Binary gripper mode: gripper is either fully open or fully closed based on a threshold
            gripper = 1.0 if raw_action[6] > 0.5 else -1.0

        # Concatenate position, rotation, and gripper to form the environment action
        env_action = np.concatenate([pos, rot, [gripper]])

        assert self._env is not None
        obs, reward, done, truncated, info = self._env.step(env_action)

        info["truncated"] = truncated
        # If the episode is marked as 'done', record that success has been seen at least once
        if done:
            self._success_seen = True

        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(
            reward=float(reward),
            done=bool(done),
            terminated=bool(done),
            truncated=bool(truncated),
            success=bool(done),
        )

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extract and format the primary visual camera frame from the environment.

        Args:
            raw_obs: Raw observation.

        Returns:
            The image numpy array, or None if extraction fails.
        """
        if self._env is None:
            return None
        try:
            get_image_from_maniskill2_obs_dict = __import__(
                "simpler_env.utils.env.observation_utils", fromlist=["get_image_from_maniskill2_obs_dict"]
            ).get_image_from_maniskill2_obs_dict

            # Attempt to extract the primary image from the ManiSkill2 observation dictionary
            return np.asarray(get_image_from_maniskill2_obs_dict(self._env, raw_obs))
        except Exception:
            # Return None if image extraction fails (e.g., no visual observation mode)
            return None

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Convert raw ManiSkill2/SimplerEnv observation to standard track observation format.

        Args:
            raw_obs: Raw observation dictionary.
            task: Task dictionary containing target metadata.

        Returns:
            A standard observation dictionary containing image, language description, and optional state.
        """
        get_image_from_maniskill2_obs_dict = __import__(
            "simpler_env.utils.env.observation_utils", fromlist=["get_image_from_maniskill2_obs_dict"]
        ).get_image_from_maniskill2_obs_dict

        # Extract the primary image from the raw observation
        image = get_image_from_maniskill2_obs_dict(self._env, raw_obs)

        # Initialize the observation dictionary with image, task description, and task name
        obs: dict[str, Any] = {
            "images": {"primary": image},
            "task_description": self._task_description,
            "task_name": self.task_name,
        }

        # Conditionally include proprioceptive state if send_state is enabled
        if self.send_state:
            agent = raw_obs.get("agent", {})
            extra = raw_obs.get("extra", {})

            # Send base_pose + tcp_pose for model servers that compute
            # base-relative EE pose (X-VLA, GR00T, etc.)
            base_pose = agent.get("base_pose")
            tcp_pose = extra.get("tcp_pose")
            if base_pose is not None and tcp_pose is not None:
                obs["base_pose"] = np.asarray(base_pose, dtype=np.float32)
                obs["tcp_pose"] = np.asarray(tcp_pose, dtype=np.float32)

            # Send pre-computed EE state if available (8D: pos3 + quat4_wxyz + gripper)
            eef = agent.get("eef_pos")
            if eef is not None:
                obs["states"] = np.asarray(eef, dtype=np.float32)
            elif base_pose is not None and tcp_pose is not None:
                # If EE state is not directly available, compute it from base_pose and tcp_pose
                from worldfoundry.evals.tasks.embodied.simulators.rotation import (
                    matrix_to_quat,
                    pose7_wxyz_to_mat4,
                    quat_xyzw_to_wxyz,
                )

                bp = np.asarray(base_pose, dtype=np.float64).flatten()
                tp = np.asarray(tcp_pose, dtype=np.float64).flatten()

                # Convert poses to transformation matrices
                base_mat = pose7_wxyz_to_mat4(bp)
                tcp_mat = pose7_wxyz_to_mat4(tp)
                # Compute end-effector pose relative to the base
                ee_in_base = np.linalg.inv(base_mat) @ tcp_mat
                pos = ee_in_base[:3, 3]  # Extract position
                q_wxyz = quat_xyzw_to_wxyz(matrix_to_quat(ee_in_base[:3, :3]))  # Extract orientation as wxyz quaternion

                assert self._env is not None
                # Get gripper closedness from environment or agent qpos
                try:
                    closedness = self._env.unwrapped.agent.get_gripper_closedness()
                    gripper_open = 1.0 - float(closedness)
                except Exception:
                    qpos = agent.get("qpos")
                    gripper_open = float(qpos[-1]) if qpos is not None else 0.0
                # Combine position, quaternion, and gripper state into a single state vector
                obs["states"] = np.concatenate([pos, q_wxyz, [gripper_open]]).astype(np.float32)

        return obs

    def check_done(self, step_result: StepResult) -> bool:
        """Check if SimplerEnv episode has terminated or truncated.

        Args:
            step_result: Current StepResult payload.

        Returns:
            True if done, False otherwise.
        """
        # For 'early_stop' mode, the episode ends as soon as 'done' or 'truncated' is true
        if self.success_mode == "early_stop":
            return step_result.done or step_result.info.get("truncated", False)
        # For 'truncation' and 'accumulate' modes, the episode only ends when truncated
        # (i.e., reaches max_episode_steps), allowing 'done' to be checked later for success.
        return step_result.info.get("truncated", False)

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Compile a dictionary of step results and outcome metrics.

        Args:
            step_result: Final StepResult of the episode.

        Returns:
            A results dictionary containing success boolean indicator.
        """
        # For 'accumulate' mode, success is true if terminated at any point during the episode
        if self.success_mode == "accumulate":
            return {"success": self._success_seen}
        # For 'truncation' mode, success is true only if terminated on the final step.
        # For 'early_stop' mode, success is true if termination occurred (which triggered the stop).
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve simulation metadata including task_name, success_mode, max_steps.

        Returns:
            A metadata dictionary mapping.
        """
        meta: dict[str, Any] = {
            "task_name": self.task_name,
            "success_mode": self.success_mode,
            "max_steps": self.max_episode_steps,
        }
        # Include VA-specific environment details if applicable
        if self.env_name is not None:
            meta["env_name"] = self.env_name
            if self.scene_name is not None:
                meta["scene_name"] = self.scene_name
            if self.env_build_kwargs:
                meta["env_build_kwargs"] = self.env_build_kwargs
        return meta

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Get expected action constraints of the simulation model.

        Returns:
            A dictionary mapping action keys to DimSpec bounds.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Get the observation contract provided by the simulator.

        Returns:
            A dictionary mapping observation keys to DimSpec constraints.
        """
        spec: dict[str, DimSpec] = {
            "primary": IMAGE_RGB,
            "language": LANGUAGE,
        }
        # Add state specification if proprioceptive state is being sent
        if self.send_state:
            spec["state"] = RAW
        return spec

"""MolmoSpaces-Bench benchmark implementation.

Wraps MolmoSpaces's JSON-based evaluation pipeline (JsonEvalTaskSampler →
BaseMujocoTask) so VLA model servers can evaluate on MolmoSpaces-Bench via
the WorldFoundry WebSocket/msgpack protocol.

This implementation matches the paper's evaluation path
(``olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig``) using:
- ``action_type = joint_pos`` (absolute joint positions, NOT relative)
- ``policy_dt_ms = 66.0``, command_mode = ``joint_position``
- ``task_horizon = 600`` (for pick-and-place tasks)

Camera name mapping:
- Primary camera: exo_camera_1 (maps to MolmoSpaces's
  ``droid_shoulder_light_randomization`` in the sensor suite output)
- Wrist camera: wrist_camera (maps to ``wrist_camera_zed_mini``)

dict[str, Any] format (over the wire from the model server):
- ``obs["actions"]`` is an 8-dim vector: 7 absolute arm joint positions +
  1 gripper command (0 or 255 after clamping, done by the model server).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.core.io.paths import local_data_root_path
from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import (
    GRIPPER_01,
    IMAGE_RGB,
    LANGUAGE,
    STATE_JOINT,
    DimSpec,
)

logger = logging.getLogger(__name__)

# Set environment variables for headless MuJoCo rendering.
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MUJOCO_GL", "egl")

# Fallback max steps used if ``task_horizon`` is not set in the config.
# The MolmoBot paper's README specifies ``task_horizon=600`` for pick-and-place
# (https://github.com/allenai/MolmoBot/blob/main/MolmoBot/README.md). Other task
# types have not been end-to-end verified through this runner yet; users should
# set ``task_horizon`` explicitly in the benchmark YAML when running them.
DEFAULT_TASK_HORIZON = 600

# Camera name aliases: primary = exo_camera_1, wrist = wrist_camera.
# MolmoSpaces env emits them under different names depending on the camera system.
PRIMARY_CAM_ALIASES = ("droid_shoulder_light_randomization", "exo_camera_1", "exo_camera")
WRIST_CAM_ALIASES = ("wrist_camera_zed_mini", "wrist_camera")

# Canonical wire names (what the model server expects).
PRIMARY_CAM = "exo_camera_1"
WRIST_CAM = "wrist_camera"


class MolmoSpacesBenchmark(BaseSimulator):
    """MolmoSpaces-Bench manipulation benchmark.

    This class serves as a wrapper around the MolmoSpaces evaluation pipeline,
    allowing VLA model servers to interact with MolmoSpaces-Bench tasks
    via the WorldFoundry WebSocket/msgpack protocol. It handles environment
    setup, episode management, observation processing, and action execution.

    Args:
        benchmark_dir: Path to a benchmark directory containing
            ``benchmark.json`` (or ``house_*/episode_*.json`` layout).
            If None, attempts to resolve from environment variables or a default path.
        eval_config_cls: Import string ``module:Class`` for the
            MlSpacesExpConfig subclass. Defaults to the paper's Franka
            state-8 clamped absolute-joint-position config.
        task_horizon: Override max steps per episode. If ``None``, uses
            the per-task defaults above.
        send_wrist_image: Include wrist camera in wire observations.
        send_state: Include proprioceptive state (qpos) in wire observations.
    """

    # Set of fields that are recorded by the video recorder for each step.
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        benchmark_dir: str | None = None,
        eval_config_cls: str = "olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig",
        task_horizon: int | None = None,
        send_wrist_image: bool = True,
        send_state: bool = True,
    ) -> None:
        """Initializes the MolmoSpacesBenchmark simulator."""
        super().__init__()
        # Resolve benchmark directory from provided path, environment variables, or default.
        resolved_benchmark_dir = (
            benchmark_dir
            or os.environ.get("WORLDFOUNDRY_MOLMOSPACES_ASSET_ROOT")
            or os.environ.get("WORLDFOUNDRY_MOLMOSPACES_DATASET_ROOT")
            or os.environ.get("WORLDFOUNDRY_MOLMOSPACES_ROOT")
        )
        if not resolved_benchmark_dir:
            resolved_benchmark_dir = str(local_data_root_path() / "datasets" / "molmospaces")
        self.benchmark_dir = Path(resolved_benchmark_dir)
        self.eval_config_cls = eval_config_cls
        self.task_horizon = task_horizon
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state

        # Internal state for the MolmoSpaces environment and tasks.
        self._episodes: list[Any] = []
        self._exp_config: Any = None
        self._sampler: Any = None
        self._task: Any = None
        self._task_description: str = ""
        self._step_count: int = 0

    # -- data -------------------------------------------------------------

    def cleanup(self) -> None:
        """Safely dispose of MolmoSpaces sampler and Mujoco environment allocations.

        This method attempts to close the underlying MolmoSpaces sampler and
        its associated MuJoCo environment to release resources. Exceptions
        during cleanup are suppressed.
        """
        if self._sampler is not None:
            try:
                # Attempt to close the sampler if it has a close method.
                if hasattr(self._sampler, "close"):
                    self._sampler.close()
                # Otherwise, attempt to close the environment if the sampler holds it.
                elif (
                    hasattr(self._sampler, "_env")
                    and self._sampler._env is not None
                    and hasattr(self._sampler._env, "close")
                ):
                    self._sampler._env.close()
            except Exception:
                # Suppress exceptions during cleanup.
                pass
            self._sampler = None
        self._task = None

    def get_tasks(self) -> list[dict[str, Any]]:
        """List registered tasks for the MolmoSpaces evaluation suite.

        Loads all episode specifications from the benchmark directory and
        converts them into a list of task dictionaries.

        Returns:
            A list of task dictionary configurations, each containing
            "name", "episode_index", and "task_cls".

        Raises:
            RuntimeError: If no episodes are found in the specified benchmark directory.
        """
        load_all_episodes = __import__(
            "molmo_spaces.evaluation.benchmark_schema", fromlist=["load_all_episodes"]
        ).load_all_episodes

        self._episodes = load_all_episodes(self.benchmark_dir)
        if not self._episodes:
            raise RuntimeError(f"No episodes found in {self.benchmark_dir}")
        logger.info("Loaded %d episodes from %s", len(self._episodes), self.benchmark_dir)

        tasks: list[dict[str, Any]] = []
        for i, ep in enumerate(self._episodes):
            # Extract task class name and truncate task description for a concise task name.
            task_cls = ep.get_task_cls().rsplit(".", 1)[-1]
            desc = ep.language.task_description[:60].replace("/", "_")
            tasks.append(
                {
                    "name": f"{task_cls}_{i:04d}_{desc}",
                    "episode_index": i,
                    "task_cls": task_cls,
                }
            )
        return tasks

    # -- episode lifecycle ------------------------------------------------

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the MolmoSpaces environment for a new episode.

        Initializes the MolmoSpaces task sampler and the MuJoCo task
        based on the provided task configuration. Sets up the task horizon
        and records the initial frame.

        Args:
            task: A dictionary containing task-specific information,
                including at least "episode_index".

        Returns:
            The initial observation from the environment.

        Raises:
            RuntimeError: If the MuJoCo task fails to be created.
        """
        JsonEvalTaskSampler = __import__(
            "molmo_spaces.tasks.json_eval_task_sampler", fromlist=["JsonEvalTaskSampler"]
        ).JsonEvalTaskSampler

        # Build the eval exp_config lazily to avoid importing MuJoCo/JAX at class load time.
        if self._exp_config is None:
            self._exp_config = self._build_exp_config()

        ep_idx = task["episode_index"]
        episode_spec = self._episodes[ep_idx]
        self._task_description = episode_spec.language.task_description

        # Apply task_horizon from instance configuration, falling back to default.
        horizon = self.task_horizon or DEFAULT_TASK_HORIZON
        self._exp_config.task_horizon = int(horizon)

        # Add EvalRuntimeParams to the config, necessary for JsonEvalRunner.patch_config.
        EvalRuntimeParams = __import__(
            "molmo_spaces.evaluation.eval_main", fromlist=["EvalRuntimeParams"]
        ).EvalRuntimeParams

        self._exp_config.eval_runtime_params = EvalRuntimeParams()

        # Clean up any previous sampler's environment before creating a new one to prevent resource leaks.
        if self._sampler is not None:
            try:
                if (
                    hasattr(self._sampler, "_env")
                    and self._sampler._env is not None
                    and hasattr(self._sampler._env, "close")
                ):
                    self._sampler._env.close()
            except Exception:
                pass

        # Create the per-episode task sampler and materialize the MuJoCo task.
        self._sampler = JsonEvalTaskSampler(self._exp_config, episode_spec)
        mujoco_task = self._sampler.sample_task(house_index=episode_spec.house_index)
        if mujoco_task is None:
            raise RuntimeError(f"Failed to create task for episode {ep_idx}")
        self._task = mujoco_task
        self._step_count = 0

        # BaseMujocoTask.reset() returns (obs_list, info) per gym convention.
        reset_output = self._task.reset()
        # Extract raw observation from the reset output, handling tuple or direct obs.
        raw_obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        unwrapped = self._unwrap_batch(raw_obs)
        self._recorder.record_video(self._extract_frame(unwrapped))
        return unwrapped

    def step(self, action: dict[str, Any]) -> StepResult:
        """Executes a single step in the MolmoSpaces environment.

        Processes the incoming action, converts it to the environment's format,
        steps the environment, and processes the raw observation, reward,
        and done signals into a StepResult.

        Args:
            action: A dictionary containing the "actions" (or "action") key,
                which is an 8-dimensional vector representing joint positions
                and gripper command.

        Returns:
            A StepResult containing the processed observation, reward,
            done status, and additional info.

        Raises:
            ValueError: If the action vector does not have 8 dimensions.
        """
        # Extract and flatten the raw action, ensuring it's a float32 numpy array.
        raw = action.get("actions", action.get("action"))
        raw = np.asarray(raw, dtype=np.float32).flatten()
        if raw.size < 8:
            raise ValueError(f"Expected 8D action, got {raw.size}D: {raw}")

        # Split the 8D action into MolmoSpaces's per-move-group action dict (arm and gripper).
        env_action = {
            "arm": raw[:7].astype(np.float32).copy(),
            "gripper": raw[7:8].astype(np.float32).copy(),
        }

        assert self._task is not None
        step_output = self._task.step(env_action)
        self._step_count += 1

        # task.step() returns (obs, reward, terminated, truncated, info) in Gymnasium-like format.
        obs, reward, terminated, truncated, info = step_output
        obs = self._unwrap_batch(obs)
        reward = self._scalar(reward, default=0.0)
        terminated = self._boolean(terminated)
        truncated = self._boolean(truncated)
        # Extract info, handling cases where it might be a list/tuple.
        if isinstance(info, (list, tuple)):
            info = info[0] if info else {}

        done = bool(terminated or truncated)
        # Extract success status from info, handling dict or non-dict cases.
        success = bool(self._scalar(info.get("success", False), default=False)) if isinstance(info, dict) else False

        # Create output info dictionary, ensuring success is set.
        out_info = dict(info) if isinstance(info, dict) else {}
        out_info.setdefault("success", success)

        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=done, success=success)

        return StepResult(obs=obs, reward=float(reward), done=done, info=out_info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extract and format camera views for recording logs.

        Args:
            raw_obs: Raw observation.

        Returns:
            The image numpy array, or None if missing.
        """
        if not isinstance(raw_obs, dict):
            return None
        # Reuse the primary-camera resolution logic that make_obs uses for the model server.
        return self._find_camera(raw_obs, PRIMARY_CAM_ALIASES)

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Converts raw environment observations into the VLA model server's expected format.

        This method extracts primary and wrist camera images (if enabled)
        and proprioceptive states (if enabled), along with the task description,
        into a structured dictionary.

        Args:
            raw_obs: The raw observation dictionary from the MolmoSpaces environment.
            task: The current task dictionary (unused in this implementation but part of signature).

        Returns:
            A dictionary formatted for the VLA model server, including
            "images", "task_description", and optionally "states".
        """
        if not isinstance(raw_obs, dict):
            # Ensure raw_obs is a dictionary by unwrapping if it's a batch.
            raw_obs = self._unwrap_batch(raw_obs)
        if not isinstance(raw_obs, dict):
            # If still not a dict, return a minimal observation.
            return {"images": {}, "task_description": self._task_description}

        images: dict[str, np.ndarray] = {}

        # Find and add the primary camera image if available.
        primary = self._find_camera(raw_obs, PRIMARY_CAM_ALIASES)
        if primary is not None:
            images[PRIMARY_CAM] = primary

        # Find and add the wrist camera image if sending wrist images is enabled.
        if self.send_wrist_image:
            wrist = self._find_camera(raw_obs, WRIST_CAM_ALIASES)
            if wrist is not None:
                images[WRIST_CAM] = wrist

        result: dict[str, Any] = {
            "images": images,
            "task_description": self._task_description,
        }

        # Extract and add proprioceptive state (qpos) if sending state is enabled.
        if self.send_state:
            qpos = self._extract_qpos(raw_obs)
            if qpos is not None:
                result["states"] = qpos

        return result

    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the episode is finished.

        Determines if the episode is done based on the step result's done flag
        or by querying the underlying MolmoSpaces task's `is_done()` method.

        Args:
            step_result: The result of the current simulation step.

        Returns:
            True if the episode is done, False otherwise.
        """
        if step_result.done:
            return True
        if self._task is not None:
            try:
                # Query the task for its internal done status.
                return bool(self._scalar(self._task.is_done(), default=False))
            except Exception:
                # Suppress exceptions if is_done() fails.
                pass
        return False

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Provides a summary of the episode's outcome and steps.

        Queries the MolmoSpaces task for success status and returns
        the total steps taken.

        Args:
            step_result: The final StepResult of the episode.

        Returns:
            A dictionary containing "success" (boolean) and "steps" (integer).
        """
        success = False
        if self._task is not None:
            try:
                # Attempt to get success status from the task's judge_success method.
                judged = self._task.judge_success()
                success = bool(self._scalar(judged, default=False))
            except Exception:
                # Fallback to success from step_result info if judge_success fails.
                success = bool(step_result.info.get("success", False))
        else:
            # If no task, rely solely on success from step_result info.
            success = bool(step_result.info.get("success", False))
        return {"success": success, "steps": self._step_count}

    # -- specs / metadata -------------------------------------------------

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata about the MolmoSpaces benchmark.

        Returns:
            A dictionary containing "max_steps" and "eval_config_cls".
        """
        return {
            "max_steps": self.task_horizon or 600,
            "eval_config_cls": self.eval_config_cls,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Defines the action space specification for the MolmoSpaces benchmark.

        The action space consists of 7 absolute joint positions and 1 gripper command.

        Returns:
            A dictionary mapping action component names to their DimSpec.
        """
        return {
            "joints": DimSpec("joints", 7, "joint_pos_abs", (-3.15, 3.15)),
            "gripper": GRIPPER_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Defines the observation space specification for the MolmoSpaces benchmark.

        Includes primary camera image and language description by default,
        with optional wrist camera image and proprioceptive state (joint states).

        Returns:
            A dictionary mapping observation component names to their DimSpec.
        """
        spec: dict[str, DimSpec] = {
            PRIMARY_CAM: IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec[WRIST_CAM] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_JOINT
        return spec

    def get_metric_keys(self) -> dict[str, str]:
        """Returns the keys for metrics to be reported and how they should be aggregated.

        Returns:
            A dictionary mapping metric names to their aggregation method (e.g., "mean").
        """
        return {"success": "mean"}

    # -- private helpers --------------------------------------------------

    def _build_exp_config(self) -> Any:
        """Instantiate the eval config class specified by eval_config_cls.

        Dynamically imports the specified MolmoSpaces experiment configuration
        class and initializes it. It also applies standard evaluation settings
        like disabling action noise and setting a fixed seed.

        Returns:
            An instance of the MolmoSpaces experiment configuration class.
        """
        import importlib

        # Dynamically import the experiment configuration class.
        module_path, class_name = self.eval_config_cls.split(":")
        module = importlib.import_module(module_path)
        eval_config_cls = getattr(module, class_name)
        exp_config = eval_config_cls()

        # Ensure no action noise is applied during evaluation, enforcing consistency.
        if hasattr(exp_config.robot_config, "action_noise_config"):
            exp_config.robot_config.action_noise_config.enabled = False

        # Match the production evaluation pipeline defaults for reproducible results.
        exp_config.filter_for_successful_trajectories = False
        exp_config.num_workers = 1
        exp_config.seed = 42
        return exp_config

    @staticmethod
    def _unwrap_batch(obs: Any) -> Any:
        """Sensor suite returns list[dict] per batch; take the first element.

        Args:
            obs: Batched observation data.

        Returns:
            The unwrapped observation dict.
        """
        if isinstance(obs, (list, tuple)) and obs:
            head = obs[0]
            if isinstance(head, dict):
                return head
        return obs

    @staticmethod
    def _scalar(value: Any, default: Any = None) -> Any:
        """Collapse per-batch scalars/arrays down to a single Python scalar value.

        Args:
            value: Scalar or array value.
            default: Default value if value is None.

        Returns:
            The collapsed scalar value.
        """
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        if isinstance(value, np.ndarray):
            return value.flatten()[0] if value.size else default
        return value

    @staticmethod
    def _boolean(value: Any) -> bool:
        """Coerce batched arrays or scalar values to a boolean flag.

        Args:
            value: Input value.

        Returns:
            Coerced boolean result.
        """
        val = MolmoSpacesBenchmark._scalar(value, default=False)
        try:
            return bool(val)
        except Exception:
            return False

    @staticmethod
    def _find_camera(obs: dict[str, Any], aliases: tuple[str, ...]) -> np.ndarray | None:
        """Find an RGB camera image under any of the given camera aliases.

        Iterates through provided aliases to find a matching key in the
        observation dictionary. If no alias matches, it performs a fallback scan
        for any 3-channel, uint8 image array containing "camera" in its key.

        Args:
            obs: Observation dict.
            aliases: Tuple of allowed camera alias names.

        Returns:
            The found camera image array, or None.
        """
        for key in aliases:
            if key in obs:
                img = obs[key]
                # Validate that the found object is a 3D RGB image array.
                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                    return np.asarray(img, dtype=np.uint8)
        # Fallback: scan for any (H,W,3) image in the obs if no specific alias matched.
        for key, val in obs.items():
            if (
                isinstance(val, np.ndarray)
                and val.ndim == 3
                and val.shape[-1] == 3
                and val.dtype == np.uint8
                and "camera" in key
            ):
                return val
        return None

    @staticmethod
    def _extract_qpos(obs: dict[str, Any]) -> np.ndarray | None:
        """Build the 8-dim proprioceptive state: 7 arm joints + 1 gripper.

        Extracts arm and gripper joint positions from the "robot_state"
        or flat "qpos" in the observation dictionary and concatenates them
        into a single 8-dimensional numpy array.

        Args:
            obs: Observation dict.

        Returns:
            The constructed qpos state array, or None if components are missing.
        """
        robot_state = obs.get("robot_state")
        if isinstance(robot_state, dict) and "qpos" in robot_state:
            qpos = robot_state["qpos"]
            if isinstance(qpos, dict):
                arm = qpos.get("arm")
                gripper = qpos.get("gripper")
                parts: list[np.ndarray] = []
                if arm is not None:
                    parts.append(np.asarray(arm, dtype=np.float32).flatten())
                if gripper is not None:
                    # Gripper is represented as 1 dimension in SynthVLAPolicyConfig.
                    parts.append(np.asarray(gripper, dtype=np.float32).flatten()[:1])
                if parts:
                    return np.concatenate(parts)
        # Fallback: if 'robot_state' dict format is not found, try to find a flat 'qpos' array.
        qpos = obs.get("qpos")
        if qpos is not None:
            return np.asarray(qpos, dtype=np.float32).flatten()
        return None

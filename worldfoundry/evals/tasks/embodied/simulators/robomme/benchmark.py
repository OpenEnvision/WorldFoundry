"""RoboMME benchmark implementation using ManiSkill3 fork + SAPIEN.

This module provides the RoboMMEBenchmark simulator, which creates a fresh
environment per episode via BenchmarkEnvBuilder. Each episode produces a
conditioning video (via motion planning) that is sent to the model server
as ``video_history`` on the first observation.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
from typing import Any, Literal, cast

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec

logger = logging.getLogger(__name__)

# A grounded subgoal that hasn't had its `<obj_center>`-style placeholders
# substituted with image coordinates still has bracketed identifiers (alpha/_).
# Filled coordinates look like `<70, 84>` — the first char inside `<` is a digit.
_UNFILLED_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_]")

# Probe script for ROBOMME_USE_LAVAPIPE=auto. Runs in a child process so a hang
# in SAPIEN's Vulkan instance creation can be timed out without poisoning the
# parent process (SAPIEN's VkInstance is created at import time and cannot be
# reset in-process). subprocess.run's timeout is the watchdog — the child has
# no internal timer.
_NATIVE_PROBE = """
import sapien
import sapien.render
import sapien.physx as physx
rs = sapien.render.RenderSystem('cuda:0')
scene = sapien.Scene([physx.PhysxCpuSystem(), rs])
cam = scene.add_camera(name='t', width=64, height=64, fovy=1.0, near=0.01, far=10.0)
cam.set_pose(sapien.Pose([0, 0, 1]))
scene.step()
scene.update_render()
cam.take_picture()
cam.get_picture('Color')
"""


def native_render_path_works(timeout_s: int = 15) -> bool:
    """Probe whether SAPIEN's native NVIDIA Vulkan path works on this host.

    Spawns a child Python process that attempts to initialize SAPIEN's render
    system and take a picture. This is used to detect hosts where SAPIEN's
    `RenderSystem("cuda:0")` path hangs.

    Args:
        timeout_s: The maximum time in seconds to wait for the child process to complete.

    Returns:
        True if the child completes successfully within the timeout, False
        if it hangs, crashes, or any unexpected error occurs. Any non-zero
        exit code from the child process is considered a failure.
    """
    try:
        # Execute the probe script in a child process, capturing no output.
        result = subprocess.run(
            [sys.executable, "-c", _NATIVE_PROBE],
            timeout=timeout_s,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Check if the child process exited successfully.
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Process timed out, indicating a hang.
        return False
    except Exception as e:
        logger.warning("Native render-path probe error: %s", e)
        return False


def _resolve_lavapipe_icd() -> str | None:
    """Find the lavapipe ICD (Installable Client Driver) path.

    This function attempts to locate the lavapipe ICD JSON file by first checking
    the `ROBOMME_LAVAPIPE_ICD` environment variable. If not set, or if the path
    is invalid, it checks predefined default paths.

    Returns:
        The full path to the lavapipe ICD JSON file if found and valid, otherwise None.
    """
    user_icd = os.environ.get("ROBOMME_LAVAPIPE_ICD")
    if user_icd:
        # If user explicitly set an ICD, ensure it exists.
        if os.path.isfile(user_icd):
            return user_icd
        logger.error(
            "ROBOMME_LAVAPIPE_ICD=%s does not exist; refusing to silently fall back to a different ICD path",
            user_icd,
        )
        return None
    # Check common lavapipe ICD paths.
    for candidate in ("/opt/lavapipe/lvp_icd.json", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"):
        if os.path.isfile(candidate):
            return candidate
    return None


# Default list of RoboMME tasks to be run if not specified.
_DEFAULT_TASK_LIST = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


class RoboMMEBenchmark(BaseSimulator):
    """RoboMME (Memory-Augmented Manipulation Evaluation) benchmark simulator.

    This simulator provides access to 16 tasks across 4 cognitive suites
    (Counting, Permanence, Reference, Imitation), built on a ManiSkill3 fork
    with SAPIEN rendering.

    Non-obvious behaviors:
        - **Conditioning video**: On ``reset``, the environment runs motion
          planning to produce a demonstration trajectory. These frames are
          sent as ``video_history`` in the first observation only.
        - **Fresh env per episode**: ``BenchmarkEnvBuilder.make_env_for_episode``
          creates a full wrapper chain for each episode, ensuring a clean state.
        - **Error obs**: ``FailAwareWrapper`` returns ``obs=None`` on exception;
          ``EndeffectorDemonstrationWrapper`` returns ``obs={}`` on IK failure.
          Both are handled gracefully in ``make_obs``.
        - **Torch scalars**: ``reward``, ``terminated``, ``truncated`` may be
          torch tensors — always cast with ``float()`` / ``bool()`` to ensure
          standard Python types.

    Args:
        tasks: Subset of task names to evaluate. If `None`, all 16 default tasks are run.
        action_space: Type of action space to use: ``"joint_angle"`` (8D) or ``"ee_pose"`` (7D).
        dataset: Dataset split to use for the tasks — ``"test"``, ``"val"``, or ``"train"``.
        max_steps: Maximum number of steps allowed per episode (paper default: 1300).
        send_wrist_image: If True, include wrist camera images in observations.
        send_state: If True, include proprioceptive state (joint and gripper) in observations.
        send_video_history: If True, send the conditioning video history on the first observation.
        send_subgoal: If True, attach the per-step subgoal text to ``obs["subgoal"]``.
        subgoal_mode: Specifies how subgoals are presented:
            - ``"grounded"``: Uses ``info['grounded_subgoal_online']`` (subgoal with
              image-coordinate placeholders filled, e.g., ``"pick up the green cube at <77, 170>"``).
              Falls back to simple if grounded is empty or contains unfilled placeholders.
            - ``"simple"``: Uses ``info['simple_subgoal_online']`` (no coordinates).
            Both come from ``DemonstrationWrapper`` in the upstream RoboMME environment.
    """

    # Set of all fields that are recorded per step for logging and analysis.
    _ALL_RECORD_FIELDS = frozenset(
        {"simple_subgoal_online", "grounded_subgoal_online", "reward", "state_fq", "terminated"}
    )

    # Tracks if SAPIEN rendering has been configured (e.g., for lavapipe fallback)
    # This prevents re-attempting configuration or patching during subsequent resets.
    _rendering_configured: bool = False

    def __init__(
        self,
        tasks: list[str] | None = None,
        action_space: str = "joint_angle",
        dataset: str = "test",
        max_steps: int = 1300,
        send_wrist_image: bool = True,
        send_state: bool = True,
        send_video_history: bool = True,
        send_subgoal: bool = False,
        subgoal_mode: Literal["grounded", "simple"] = "grounded",
    ) -> None:
        """Initializes the RoboMMEBenchmark simulator.

        Args:
            tasks: Subset of task names to evaluate. If `None`, all 16 default tasks are run.
            action_space: Type of action space to use: ``"joint_angle"`` (8D) or ``"ee_pose"`` (7D).
            dataset: Dataset split to use for the tasks — ``"test"``, ``"val"``, or ``"train"``.
            max_steps: Maximum number of steps allowed per episode (paper default: 1300).
            send_wrist_image: If True, include wrist camera images in observations.
            send_state: If True, include proprioceptive state (joint and gripper) in observations.
            send_video_history: If True, send the conditioning video history on the first observation.
            send_subgoal: If True, attach the per-step subgoal text to ``obs["subgoal"]``.
            subgoal_mode: Specifies how subgoals are presented:
                - ``"grounded"``: Uses ``info['grounded_subgoal_online']`` (subgoal with
                  image-coordinate placeholders filled, e.g., ``"pick up the green cube at <77, 170>"``).
                  Falls back to simple if grounded is empty or contains unfilled placeholders.
                - ``"simple"``: Uses ``info['simple_subgoal_online']`` (no coordinates).
        """
        super().__init__()
        if subgoal_mode not in ("grounded", "simple"):
            raise ValueError(f"subgoal_mode must be 'grounded' or 'simple', got {subgoal_mode!r}")
        self.tasks = tasks or list(_DEFAULT_TASK_LIST)
        self.action_space = action_space
        self.dataset = dataset
        self.max_steps = max_steps
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.send_video_history = send_video_history
        self.send_subgoal = send_subgoal
        self.subgoal_mode = subgoal_mode

        # Internal state for the current environment and episode, initialized to None/empty.
        self._env: Any = None
        self._task: dict[str, Any] = {}
        self._task_description: str = ""
        self._video_frames: list[np.ndarray] = []
        self._wrist_video_frames: list[np.ndarray] = []
        self._current_subgoal: str = ""

    def get_tasks(self) -> list[dict[str, Any]]:
        """List registered tasks for the RoboMME evaluation suite.

        Returns:
            A list of dictionaries, where each dictionary represents a task
            with "name" and "env_id" keys.
        """
        return [{"name": t, "env_id": t} for t in self.tasks]

    @staticmethod
    def _setup_rendering() -> None:
        """Optionally switch SAPIEN to lavapipe software Vulkan.

        This static method configures SAPIEN's rendering backend based on the
        `ROBOMME_USE_LAVAPIPE` environment variable. On certain hosts,
        SAPIEN's native NVIDIA Vulkan path can hang. This method provides
        a workaround by falling back to the lavapipe software renderer.

        The `ROBOMME_USE_LAVAPIPE` environment variable controls this behavior:
            - unset / ``0`` / ``false`` (default): Use the native NVIDIA path.
              This will hang on affected hosts.
            - ``1`` / ``true`` / ``yes``: Always engage lavapipe.
            - ``auto``: Probe the native path in a child process. If it hangs,
              engage lavapipe in the current process. This adds startup time
              but allows a single launcher config across heterogeneous hosts.

        The decision is cached per-process via `_rendering_configured`.
        Changing `ROBOMME_USE_LAVAPIPE` after the first `reset()` call will have
        no effect, as the Vulkan ICD is loaded at the first `import sapien.render`.

        Lavapipe rendering is significantly slower (~5–10x) than the native path.
        When engaged, `LP_NUM_THREADS=4` and `OMP_NUM_THREADS=1` are set for
        empirical performance tuning with Mesa lavapipe.
        """
        if RoboMMEBenchmark._rendering_configured:
            return

        mode = os.environ.get("ROBOMME_USE_LAVAPIPE", "").strip().lower()

        # Handle 'auto' mode: probe native path to decide whether to use lavapipe.
        if mode == "auto":
            if native_render_path_works():
                logger.info("SAPIEN auto-detect: native NVIDIA Vulkan path works, skipping lavapipe")
                RoboMMEBenchmark._rendering_configured = True
                return
            logger.warning(
                "SAPIEN auto-detect: native render path hung within watchdog timeout; engaging lavapipe fallback"
            )
        # If mode is not one of the explicit engagement flags (including 'auto' where probe failed), then skip lavapipe.
        elif mode not in ("1", "true", "yes", "on"):
            RoboMMEBenchmark._rendering_configured = True
            return

        # Lavapipe engagement requested (explicitly or via 'auto' fallback).
        if not RoboMMEBenchmark._engage_lavapipe():
            raise RuntimeError(
                f"Lavapipe rendering requested (ROBOMME_USE_LAVAPIPE={mode!r}) but could not "
                "be engaged. Check earlier log lines for the specific reason "
                "(sapien.render already imported, ICD missing, etc.); continuing on the "
                "native NVIDIA path would hang on affected hosts."
            )
        RoboMMEBenchmark._rendering_configured = True

    @staticmethod
    def _engage_lavapipe() -> bool:
        """Apply the three-piece lavapipe patch + perf-tuning environment variables.

        This method must be called BEFORE `import sapien.render` in the current
        process, as `VK_ICD_FILENAMES` and `LP_NUM_THREADS` only take effect
        during the initial Vulkan initialization.

        The patch involves:
        1. Setting `VK_ICD_FILENAMES` to point to the lavapipe ICD.
        2. Monkey-patching `sapien.render.RenderSystem` to remove the `device`
           positional argument, as lavapipe does not use CUDA.
        3. Patching `mani_skill.envs.utils.system.backend.parse_sim_and_render_backend`
           to ensure the render backend resolves to `sapien_cpu`, which matches
           the lavapipe device.

        Returns:
            True on successful application of the patch, False if the patch
            could not be applied (e.g., `sapien.render` already imported,
            lavapipe ICD missing). The caller is responsible for treating
            a False return as a fatal error.
        """
        # Check if sapien.render has already been imported, as the patch must occur before its first use.
        if "sapien.render" in sys.modules:
            logger.error(
                "Cannot engage lavapipe: sapien.render is already imported. "
                "VK_ICD_FILENAMES / LP_NUM_THREADS only take effect on first "
                "Vulkan init. Set ROBOMME_USE_LAVAPIPE before any sapien import."
            )
            return False

        # Attempt to find the lavapipe ICD path.
        lavapipe_icd = _resolve_lavapipe_icd()
        if lavapipe_icd is None:
            logger.error("Lavapipe ICD not found; cannot engage lavapipe rendering")
            return False

        # Set environment variables for Mesa lavapipe performance tuning if not already set.
        os.environ.setdefault("LP_NUM_THREADS", "4")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Configure Vulkan to use the lavapipe Installable Client Driver.
        os.environ["VK_ICD_FILENAMES"] = lavapipe_icd
        logger.info("SAPIEN rendering: using lavapipe software Vulkan (%s)", lavapipe_icd)

        sr = __import__("sapien.render", fromlist=["__name__"])

        # Store the original RenderSystem class before patching.
        _OrigRenderSystem = sr.RenderSystem

        # Define a patched RenderSystem that ignores the 'device' argument, as lavapipe doesn't use CUDA devices.
        def _lavapipe_render_system(*args, **kwargs):
            return _OrigRenderSystem()

        # Apply the monkey patch to sapien.render.RenderSystem.
        cast(Any, sr).RenderSystem = _lavapipe_render_system

        # Patch `parse_sim_and_render_backend` in ManiSkill3 to ensure render backend resolves to `sapien_cpu`.
        try:
            _backend_mod = __import__("mani_skill.envs.utils.system", fromlist=["backend"]).backend

            _orig_parse = _backend_mod.parse_sim_and_render_backend

            def _patched_parse(sim_backend, render_backend):
                result = _orig_parse(sim_backend, render_backend)
                # If the original backend was 'sapien_cuda', switch it to 'sapien_cpu' for lavapipe compatibility.
                if result.render_backend == "sapien_cuda":
                    result.render_backend = "sapien_cpu"
                return result

            # Apply the patch to the module's function.
            _backend_mod.parse_sim_and_render_backend = _patched_parse

            # Also patch the imported reference in `mani_skill.envs.sapien_env`
            # in case it was imported before this patch was applied.
            mani_skill = __import__("mani_skill.envs.sapien_env")

            mani_skill.envs.sapien_env.parse_sim_and_render_backend = _patched_parse
        except Exception as e:
            logger.warning("Could not patch mani_skill render backend to sapien_cpu: %s", e)

        return True

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the environment for a new episode.

        This method sets up the SAPIEN rendering configuration, creates a fresh
        ManiSkill3 environment, extracts initial observations and conditioning
        video frames, and sets the task description.

        Args:
            task: A dictionary containing task-specific information, including
                  "env_id" and optional "episode_idx".

        Returns:
            The raw observation dictionary from the environment's initial reset.
        """
        self._setup_rendering()
        # Import RoboMME environments to ensure they are registered with gym.
        __import__("robomme.robomme_env")
        BenchmarkEnvBuilder = __import__(
            "robomme.env_record_wrapper", fromlist=["BenchmarkEnvBuilder"]
        ).BenchmarkEnvBuilder

        # Close the previous environment if it exists, to ensure a fresh start for each episode.
        if self._env is not None:
            with contextlib.suppress(Exception):
                self._env.close()

        episode_idx = task.get("episode_idx", 0)
        self._task = task
        # Create a new environment builder for the current episode with specified parameters.
        builder = BenchmarkEnvBuilder(
            env_id=task["env_id"],
            dataset=self.dataset,
            action_space=self.action_space,
            gui_render=False,
            max_steps=self.max_steps,
        )
        # Make a new environment instance for the episode and perform the initial reset.
        self._env = builder.make_env_for_episode(episode_idx)
        obs_batch, info_flat = self._env.reset()

        # Store conditioning video frames (demo trajectory, excluding the final init frame).
        self._video_frames = list(obs_batch["front_rgb_list"][:-1])
        if self.send_wrist_image:
            self._wrist_video_frames = list(obs_batch.get("wrist_rgb_list", [])[:-1])

        # Extract the task description from the info dictionary.
        task_goal = info_flat["task_goal"]
        self._task_description = task_goal[0] if isinstance(task_goal, list) else str(task_goal)

        if self.send_subgoal:
            self._current_subgoal = self._extract_subgoal(info_flat)

        # Record the initial frame for video logging.
        self._recorder.record_video(self._extract_frame(obs_batch))
        return obs_batch

    def step(self, action: dict[str, Any]) -> StepResult:
        """Executes a single step in the environment using the provided action.

        Args:
            action: A dictionary containing the action to be taken, expected
                    under the key "actions" or "action".

        Returns:
            A StepResult object containing the raw observation, reward,
            done status, and info dictionary from the environment.
        """
        # Extract the raw action from the input dictionary, supporting multiple keys.
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raise ValueError("dict[str, Any] dict must contain 'actions' or 'action' key")
        # Ensure the action is a flat list for the environment's step method.
        if hasattr(raw_action, "flatten"):
            raw_action = raw_action.flatten().tolist()
        elif not isinstance(raw_action, list):
            raw_action = list(raw_action)

        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(raw_action)

        if self.send_subgoal:
            self._current_subgoal = self._extract_subgoal(info)

        # Cast boolean and float values to standard Python types, as they might be torch tensors.
        terminated = bool(terminated)
        truncated = bool(truncated)
        reward = float(reward)
        # Determine overall done status, including error conditions reported in info.
        done = terminated or truncated or info.get("status") == "error"

        # Prepare data for recording the step, including subgoals, reward, and state.
        row: dict[str, Any] = {
            "simple_subgoal_online": info.get("simple_subgoal_online", ""),
            "grounded_subgoal_online": info.get("grounded_subgoal_online", ""),
            "reward": reward,
            "terminated": terminated,
        }
        if isinstance(obs, dict):
            state = obs.get("state_fq")
            if state is not None:
                row["state_fq"] = state.tolist() if hasattr(state, "tolist") else list(state)
        # Record the current frame and step data.
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(**row)

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        """Extract and format the main agent-view camera frame for recording logs.

        Args:
            raw_obs: The raw observation dictionary from the environment.

        Returns:
            The last (current) image from the "front_rgb_list" as a NumPy array,
            or None if the raw observation is not a dictionary or if the list is empty.
        """
        # Ensure the observation is a dictionary and contains the front camera images.
        if not isinstance(raw_obs, dict):
            return None
        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            return None
        return np.asarray(front_list[-1])

    def _extract_subgoal(self, info: dict[str, Any]) -> str:
        """Picks the configured subgoal text from the environment's info dictionary.

        `DemonstrationWrapper` always populates `simple_subgoal_online` and
        `grounded_subgoal_online`. The `grounded` version may be empty or
        still contain raw placeholder templates (e.g., `<obj_center>`) if
        segmentation has not been computed for the current frame.
        This method handles fallback to the simple subgoal in such cases
        to prevent the model from seeing unfilled templates.

        Args:
            info: The info dictionary from the environment step.

        Returns:
            The selected subgoal text as a string.
        """
        if self.subgoal_mode == "grounded":
            grounded = str(info.get("grounded_subgoal_online") or "")
            # Return grounded if it's not empty and contains no unfilled placeholders.
            # Otherwise, fall back to the simple subgoal.
            if grounded and not _UNFILLED_PLACEHOLDER_RE.search(grounded):
                return grounded
        # Fallback to simple subgoal if grounded is not preferred or invalid.
        return str(info.get("simple_subgoal_online") or "")

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Transforms raw observations from the environment into the desired format.

        This method handles various error cases (e.g., `None` or empty dict for obs)
        and constructs the observation dictionary including images, task description,
        state, video history, and subgoals based on simulator configuration.

        Args:
            raw_obs: The raw observation received from the environment.
            task: The current task dictionary.

        Returns:
            A dictionary containing the formatted observations for the agent.
        """
        # Handle error cases where raw_obs might be None or an empty dictionary from wrappers.
        if not raw_obs:
            return {"images": {}, "task_description": self._task_description}

        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            # If no front camera images are available, return minimal observation.
            return {"images": {}, "task_description": self._task_description}

        front = front_list[-1]

        obs: dict[str, Any] = {
            "images": {"agentview": front},
            "task_description": self._task_description,
        }

        # Include wrist image if configured and available in the raw observations.
        if self.send_wrist_image:
            wrist_list = raw_obs.get("wrist_rgb_list")
            if wrist_list:
                obs["images"]["wrist"] = wrist_list[-1]

        # Concatenate joint and gripper states into a single 'states' array if configured.
        if self.send_state:
            joint = np.asarray(raw_obs["joint_state_list"][-1], dtype=np.float64)
            gripper = np.asarray(raw_obs["gripper_state_list"][-1], dtype=np.float64)[:1]
            obs["states"] = np.concatenate([joint, gripper]).astype(np.float32)

        # Send video history only once at the beginning of the episode.
        if self.send_video_history and self._video_frames:
            obs["video_history"] = list(self._video_frames)
            if self.send_wrist_image and self._wrist_video_frames:
                obs["wrist_video_history"] = list(self._wrist_video_frames)
            obs["episode_restart"] = True
            # Clear the frames after sending to ensure they are sent only once per episode.
            self._video_frames = []
            self._wrist_video_frames = []

        # Include current subgoal text if configured.
        if self.send_subgoal:
            obs["subgoal"] = self._current_subgoal

        return obs

    def check_done(self, step_result: StepResult) -> bool:
        """Determines if the episode is finished based on the step result.

        Args:
            step_result: The result of the latest environment step.

        Returns:
            True if the episode is done (terminated or truncated), False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Extracts the success status from the step result.

        Args:
            step_result: The result of the latest environment step.

        Returns:
            A dictionary containing the success status for the step.
        """
        success = step_result.info.get("status") == "success"
        return {"success": success}

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata about the simulator configuration.

        Returns:
            A dictionary containing "max_steps" and "action_space".
        """
        return {"max_steps": self.max_steps, "action_space": self.action_space}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Returns the specification for the action space.

        Returns:
            A dictionary mapping action names to their dimension specifications.
        """
        return {"action": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Returns the specification for the observation space.

        The returned specification includes agentview image, language, and
        optionally wrist image, state, and subgoal based on the simulator's
        initialization parameters.

        Returns:
            A dictionary mapping observation names to their dimension specifications.
        """
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "language": LANGUAGE,
        }
        # Conditionally add wrist camera image to spec.
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        # Conditionally add proprioceptive state to spec.
        if self.send_state:
            spec["state"] = RAW
        # Conditionally add subgoal text to spec.
        if self.send_subgoal:
            spec["subgoal"] = LANGUAGE
        return spec

    def cleanup(self) -> None:
        """Safely close and clean up active SAPIEN environment allocations.

        This method ensures that the underlying ManiSkill3 environment is
        properly closed and internal state variables are reset.
        """
        if self._env is not None:
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None
        self._video_frames = []
        self._wrist_video_frames = []
        self._task = {}

"""BEHAVIOR-1K benchmark implementation.

BEHAVIOR-1K is a long-horizon household-activity benchmark built on OmniGibson (NVIDIA Isaac Sim).
The 2025 BEHAVIOR Challenge defines a 50-task evaluation suite (B10/B20/B30/B40/B50) using the
R1Pro mobile-manipulation robot.

References:
    - https://behavior.stanford.edu
    - https://github.com/StanfordVL/BEHAVIOR-1K
    - OmniGibson/omnigibson/learning/eval.py (official Evaluator)

Key facts:
    - Robot: R1Pro (23-D absolute joint-position action space).
    - dict[str, Any] layout (matching ``ACTION_QPOS_INDICES["R1Pro"]``):
        base[0:3], torso[3:7], left_arm[7:14], left_gripper[14:15],
        right_arm[15:22], right_gripper[22:23].
    - Cameras: head 720x720, left_wrist 480x480, right_wrist 480x480.
    - Success: ``info["done"]["success"]`` (binary); the challenge separately reports a partial
      Q-score, but we only surface the binary flag here — partial scoring lives in the official
      ``score_utils.compute_final_q_score``.
    - Max steps default: 5000 (or 2× human demo length when known).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.dirs import ensure_license
from worldfoundry.evals.tasks.embodied.simulators.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec

logger = logging.getLogger(__name__)

# 50-task BEHAVIOR Challenge 2025 evaluation suite.
# Mirrors omnigibson.learning.utils.eval_utils.TASK_NAMES_TO_INDICES.
B50_TASKS: list[str] = [
    # B10
    "turning_on_radio",
    "picking_up_trash",
    "putting_away_Halloween_decorations",
    "cleaning_up_plates_and_food",
    "can_meat",
    "setting_mousetraps",
    "hiding_Easter_eggs",
    "picking_up_toys",
    "rearranging_kitchen_furniture",
    "putting_up_Christmas_decorations_inside",
    # B20
    "set_up_a_coffee_station_in_your_kitchen",
    "putting_dishes_away_after_cleaning",
    "preparing_lunch_box",
    "loading_the_car",
    "carrying_in_groceries",
    "bringing_in_wood",
    "moving_boxes_to_storage",
    "bringing_water",
    "tidying_bedroom",
    "outfit_a_basic_toolbox",
    # B30
    "sorting_vegetables",
    "collecting_childrens_toys",
    "putting_shoes_on_rack",
    "boxing_books_up_for_storage",
    "storing_food",
    "clearing_food_from_table_into_fridge",
    "assembling_gift_baskets",
    "sorting_household_items",
    "getting_organized_for_work",
    "clean_up_your_desk",
    # B40
    "setting_the_fire",
    "clean_boxing_gloves",
    "wash_a_baseball_cap",
    "wash_dog_toys",
    "hanging_pictures",
    "attach_a_camera_to_a_tripod",
    "clean_a_patio",
    "clean_a_trumpet",
    "spraying_for_bugs",
    "spraying_fruit_trees",
    # B50
    "make_microwave_popcorn",
    "cook_cabbage",
    "chop_an_onion",
    "slicing_vegetables",
    "chopping_wood",
    "cook_hot_dogs",
    "cook_bacon",
    "freeze_pies",
    "canning_food",
    "make_pizza",
]

# The R1Pro robot's action space dimension (23-D), matching ACTION_QPOS_INDICES["R1Pro"].
R1PRO_ACTION_DIM = 23

# Suffix used in OmniGibson's flattened observation dictionary for RGB camera data.
RGB_SUFFIX = "::rgb"
# Key used in OmniGibson's flattened observation dictionary for R1Pro robot proprioceptive data.
PROPRIO_KEY = "robot_r1::proprio"

# Mapping from canonical camera names to their full sensor keys in OmniGibson for the R1Pro robot.
R1PRO_CAMERAS: dict[str, str] = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
}


def _humanize(task_name: str) -> str:
    """Convert snake_case task names to human-readable strings.

    Args:
        task_name: Snake case task identifier.

    Returns:
        Space-separated human-readable task description.
    """
    return task_name.replace("_", " ")


class Behavior1KBenchmark(BaseSimulator):
    """BEHAVIOR-1K (OmniGibson) household-activity benchmark.

    This class provides an interface to interact with the BEHAVIOR-1K simulation environment
    for evaluating mobile manipulation policies, specifically with the R1Pro robot.

    Non-obvious behaviors:
        - **Heavy lazy imports**: ``omnigibson`` and Isaac Sim are imported inside ``_init_og()``
          rather than at module top. Importing OmniGibson boots the Isaac Sim runtime and consumes
          several gigabytes of VRAM, so we delay until ``get_tasks()`` / ``reset()`` actually need
          it. This also keeps pure import-string validation fast.
        - **dict[str, Any] format**: ``env.step()`` expects a ``torch.Tensor``, not numpy. Converted in
          ``step()``.
        - **dict[str, Any] flattening**: OmniGibson's nested observation
          (``obs["robot_r1"]["sensors"]["zed"]["rgb"]``) is flattened with a ``::`` delimiter via
          the official ``flatten_obs_dict`` helper. We then look up cameras by their canonical
          sensor key.
        - **dict[str, Any] description**: BehaviorTask does not expose a natural language instruction; we use
          the snake-case task name with underscores replaced by spaces, matching common VLA practice.
        - **Single robot supported**: R1Pro only (the BEHAVIOR Challenge 2025 standard track). A1
          is reachable through OmniGibson but not exercised here.
    """

    # Set of fields expected in the `info` dictionary for recording evaluation results.
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "terminated", "truncated", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        partial_scene_load: bool = True,
        max_steps: int | None = None,
        send_proprio: bool = False,
        camera_names: list[str] | None = None,
        env_wrapper_target: str = "omnigibson.envs.env_wrapper.EnvironmentWrapper",
        task_instance_id: int | list[int] | None = None,
    ) -> None:
        """Initialize the BEHAVIOR-1K simulator.

        Args:
            tasks: Subset of B50 task names to evaluate. ``None`` runs all 50.
            partial_scene_load: Pass through to OmniGibson — load only rooms relevant to the task to
                speed up scene construction.
            max_steps: Per-episode step cap. ``None`` keeps OmniGibson's default (5000 in
                ``generate_basic_environment_config``).
            send_proprio: Include the R1Pro proprio vector (``robot_r1::proprio``, 256-D) in observations.
            camera_names: Which cameras to forward to the model server. Defaults to all three
                (``head``, ``left_wrist``, ``right_wrist``).
            env_wrapper_target: Hydra ``_target_`` for the env wrapper. By default we use OmniGibson's
                ``EnvironmentWrapper`` no-op wrapper; override to plug in challenge-specific behaviour.
            task_instance_id: Per-instance TRO state(s) to load after ``env.reset()``, mirroring the
                official ``Evaluator.load_task_instance``. Without this the env starts from
                BehaviorTask's default instance (idx 0); with it set, the cached
                ``<scene>_task_<activity>_instances/<...>-tro_state.json`` is applied so the initial
                object placement matches the recorded demos. Required for demo-replay reproductions.

                Accepts:
                    - ``None`` — use BehaviorTask's default instance every episode (no TRO state load).
                    - ``int`` — fix the same instance for every episode.
                    - ``list[int]`` — sweep instances; episode ``i`` uses ``ids[i % len(ids)]``. Use
                      this to reproduce the challenge protocol (50 tasks × 10 instances).
        """
        super().__init__()
        if tasks is not None:
            unknown = [t for t in tasks if t not in B50_TASKS]
            if unknown:
                raise ValueError(f"Unknown BEHAVIOR-1K tasks: {unknown}")
        self._task_names: list[str] = list(tasks) if tasks else list(B50_TASKS)
        self._partial_scene_load = partial_scene_load
        self._max_steps = max_steps
        self._send_proprio = send_proprio
        self._camera_names = camera_names or list(R1PRO_CAMERAS.keys())
        unknown_cams = [c for c in self._camera_names if c not in R1PRO_CAMERAS]
        if unknown_cams:
            raise ValueError(f"Unknown R1Pro cameras: {unknown_cams}. Valid: {list(R1PRO_CAMERAS)}")
        self._env_wrapper_target = env_wrapper_target

        # Normalize int|list|None for task_instance_id to list[int]|None
        # so reset() can index by ``episode_idx`` uniformly.
        if task_instance_id is None:
            self._task_instance_ids: list[int] | None = None
        elif isinstance(task_instance_id, int):
            self._task_instance_ids = [task_instance_id]
        else:
            if not task_instance_id:
                raise ValueError("task_instance_id list must not be empty")
            self._task_instance_ids = [int(i) for i in task_instance_id]

        self._env: Any = None
        self._current_task_name: str | None = None
        self._available_tasks: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _init_og(self) -> None:
        """First-time import and side-effect setups for OmniGibson and Isaac Sim.

        This method performs lazy initialization of OmniGibson and Isaac Sim. It ensures that
        OmniGibson is only imported and configured when actually needed, preventing premature
        resource consumption.
        """
        if self._available_tasks is not None:
            return
        load_available_tasks = __import__(
            "gello.robots.sim_robot.og_teleop_utils", fromlist=["load_available_tasks"]
        ).load_available_tasks
        macro_module = __import__("omnigibson.macros", fromlist=["gm", "macros"])
        gm, macros = macro_module.gm, macro_module.macros

        # Match the official challenge eval defaults from learning/eval.py.
        # HEADLESS=True is critical: without it Isaac Sim tries to start
        # the XR viewport extension and segfaults on a headless GPU node.
        gm.HEADLESS = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = True
        with macros.unlocked():
            macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

        self._ensure_assets(Path(gm.DATA_PATH))
        self._available_tasks = load_available_tasks()
        missing = [t for t in self._task_names if t not in self._available_tasks]
        if missing:
            raise RuntimeError(
                f"BEHAVIOR-1K tasks not available in installed dataset: {missing}. "
                "Check that the 2025-challenge-task-instances data is mounted at gm.DATA_PATH."
            )

    def _ensure_assets(self, data_path: Path) -> None:
        """Make sure BEHAVIOR-1K scene + task data is available at ``data_path``.

        First call on a fresh host prompts for licence acceptance and runs OmniGibson's three
        ``download_*`` helpers. Idempotent: a populated directory short-circuits via the marker check.

        Args:
            data_path: The path where OmniGibson data assets are expected to be stored.
        """
        marker = data_path / "2025-challenge-task-instances"
        if marker.exists():
            return
        ensure_license(
            "behavior-dataset-tos",
            url="https://behavior.stanford.edu/dataset",
            description="BEHAVIOR Dataset ToS (one-time, ~35 GiB download).",
        )
        data_path.mkdir(parents=True, exist_ok=True)
        asset_utils = __import__(
            "omnigibson.utils.asset_utils",
            fromlist=[
                "download_2025_challenge_task_instances",
                "download_behavior_1k_assets",
                "download_omnigibson_robot_assets",
            ],
        )
        download_2025_challenge_task_instances = asset_utils.download_2025_challenge_task_instances
        download_behavior_1k_assets = asset_utils.download_behavior_1k_assets
        download_omnigibson_robot_assets = asset_utils.download_omnigibson_robot_assets

        logger.info("Fetching BEHAVIOR-1K assets into %s", data_path)
        download_omnigibson_robot_assets()
        download_behavior_1k_assets(accept_license=True)
        download_2025_challenge_task_instances()

    def _make_env(self, task_name: str) -> Any:
        """Build a fresh OmniGibson env for *task_name* synchronously or asynchronously.

        Args:
            task_name: Snake case task identifier.

        Returns:
            An instantiated environment object.
        """
        # Isaac Sim's SimulationApp.__init__ calls signal.signal(SIGINT, ...) which raises ValueError
        # when invoked from a non-main thread. This monkey-patch ensures that the signal handler
        # registration does not occur if not in the main thread, preventing crashes
        # while allowing env construction to be offloaded.
        import signal as _signal
        import threading

        _orig_signal = None
        if threading.current_thread() is not threading.main_thread():
            _orig_signal = _signal.signal
            cast(Any, _signal).signal = lambda *a, **kw: None

        try:
            return self._make_env_inner(task_name)
        finally:
            if _orig_signal is not None:
                _signal.signal = _orig_signal

    def _make_env_inner(self, task_name: str) -> Any:
        """Inner method constructing the environment with Hydra configuration.

        This method is responsible for generating the OmniGibson environment configuration
        and instantiating the environment, potentially applying partial scene loading and
        robot-specific observations.

        Args:
            task_name: Snake case task identifier.

        Returns:
            An instantiated environment object.
        """
        og = __import__("omnigibson")
        DISABLED_TRANSITION_RULES = __import__(
            "gello.robots.sim_robot.og_teleop_cfg", fromlist=["DISABLED_TRANSITION_RULES"]
        ).DISABLED_TRANSITION_RULES
        teleop_utils = __import__(
            "gello.robots.sim_robot.og_teleop_utils",
            fromlist=["augment_rooms", "generate_robot_config", "get_task_relevant_room_types"],
        )
        augment_rooms = teleop_utils.augment_rooms
        generate_robot_config = teleop_utils.generate_robot_config
        get_task_relevant_room_types = teleop_utils.get_task_relevant_room_types
        instantiate = __import__("hydra.utils", fromlist=["instantiate"]).instantiate
        from omegaconf import OmegaConf

        eval_utils = __import__(
            "omnigibson.learning.utils.eval_utils",
            fromlist=["PROPRIOCEPTION_INDICES", "generate_basic_environment_config"],
        )
        PROPRIOCEPTION_INDICES = eval_utils.PROPRIOCEPTION_INDICES
        generate_basic_environment_config = eval_utils.generate_basic_environment_config

        # Disable a curated set of transition rules to match the data-collection setup for official evaluation.
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        assert self._available_tasks is not None
        task_cfg = self._available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)

        # Apply partial scene loading if enabled, loading only rooms relevant to the task.
        if self._partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        # Configure robot observations to include proprioception and RGB images.
        cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

        # Set maximum steps for the task termination if a specific limit is provided.
        if self._max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = self._max_steps
        cfg["task"]["include_obs"] = False

        # Instantiate the OmniGibson environment and wrap it if an `env_wrapper_target` is specified.
        env = og.Environment(configs=cfg)
        wrapper_cfg = OmegaConf.create({"_target_": self._env_wrapper_target})
        env = instantiate(wrapper_cfg, env=env)
        return env

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[dict[str, Any]]:
        """List registered tasks for the BEHAVIOR-1K evaluation suite.

        Returns:
            A list of dictionary configurations representing each task, including a human-readable
            name, the snake_case task identifier, and the suite.
        """
        # Avoid booting Isaac Sim during config validation: defer the
        # import-side-effect until we actually have a chance to run.
        return [{"name": _humanize(t), "task_name": t, "suite": "behavior_1k"} for t in self._task_names]

    def reset(self, task: dict[str, Any]) -> Any:
        """Reset the environment for a new BEHAVIOR-1K household-task episode.

        This method initializes or reconfigures the OmniGibson environment for a specific task.
        If the environment needs to change tasks or is not yet created, it will be closed and
        rebuilt. It also handles the optional loading of a specific task instance state.

        Args:
            task: Task dictionary containing task_name and optionally episode_idx.

        Returns:
            The initial raw environment observation.
        """
        self._init_og()
        task_name = task["task_name"]
        # If the environment is not yet created or the task has changed, close the old one and build a new one.
        if self._env is None or self._current_task_name != task_name:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    logger.exception("Failed to close previous OmniGibson env")
            self._env = self._make_env(task_name)
            self._current_task_name = task_name
        obs, _ = self._env.reset()

        # Optional per-instance TRO state load (matches official ``Evaluator.load_task_instance``).
        # When unset, BehaviorTask uses its default instance (idx 0) — the env still runs, but object
        # placements may diverge from a particular demo. When a list is provided, sweep instances by
        # ``episode_idx`` so consecutive episodes hit different recorded states (the 50-task ×
        # 10-instance challenge protocol).
        if self._task_instance_ids is not None:
            episode_idx = int(task.get("episode_idx", 0))
            instance_id = self._task_instance_ids[episode_idx % len(self._task_instance_ids)]
            obs = self._load_task_instance(instance_id)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def _load_task_instance(self, instance_id: int) -> Any:
        """Apply per-instance object/robot state JSON, then re-fetch obs.

        Ports the v3.7.2 ``Evaluator.load_task_instance`` (public-test branch). Reads
        ``<get_task_instance_path(scene)>/json/<scene>_task_<activity>_instances/<...>-tro_state.json``
        and pushes the recorded object/robot state into the running env. This allows reproducing
        specific initial configurations as recorded in demonstration data.

        Compatible only with the v3.7.2 OmniGibson API: uses ``robot.model_name``,
        ``entity.is_system`` / ``entity.exists``.

        Args:
            instance_id: The ID of the specific task instance state to load.

        Returns:
            The environment observation after the task instance state has been loaded.
        """
        import json
        import os

        og = __import__("omnigibson")
        get_task_instance_path = __import__(
            "omnigibson.utils.asset_utils", fromlist=["get_task_instance_path"]
        ).get_task_instance_path
        recursively_convert_to_torch = __import__(
            "omnigibson.utils.python_utils", fromlist=["recursively_convert_to_torch"]
        ).recursively_convert_to_torch

        env = self._env
        task = env.task
        scene_model = task.scene_name
        tro_filename = task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=task.activity_name,
            activity_definition_id=task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path) as f:
            # Load the state from JSON and convert to PyTorch tensors.
            tro_state = recursively_convert_to_torch(json.load(f))

        robot = env.scene.object_registry("name", "robot_r1")
        for tro_key, tro_substate in tro_state.items():
            if tro_key == "robot_poses":
                # Apply robot pose from the loaded state.
                if robot is None:
                    raise RuntimeError("BEHAVIOR-1K _load_task_instance: robot 'robot_r1' not found in scene")
                model_name = getattr(robot, "model_name", None) or getattr(robot, "model", None)
                if model_name not in tro_substate:
                    raise KeyError(
                        f"BEHAVIOR-1K instance {instance_id}: no presampled robot pose "
                        f"for robot.model_name={model_name!r}; keys={list(tro_substate)}"
                    )
                pose0 = tro_substate[model_name][0]
                robot.set_position_orientation(pose0["position"], pose0["orientation"])
                env.scene.write_task_metadata(key=tro_key, data=tro_substate)
            else:
                # Apply object states for other entities in the scene.
                task.object_scope[tro_key].load_state(tro_substate, serialized=False)

        # Settle objects so loaded poses are stable before evaluation by stepping physics.
        for _ in range(25):
            og.sim.step_physics()
            for entity in task.object_scope.values():
                if entity is not None and not getattr(entity, "is_system", False) and getattr(entity, "exists", True):
                    entity.keep_still()

        env.scene.update_initial_file()
        env.scene.reset()

        # Re-fetch the observation after the state load so the model server sees the post-load
        # images / proprio.
        obs, _ = env.get_obs()
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Step the simulator using joint targets predicted by the policy runner.

        This method converts the input action into a PyTorch tensor, validates its dimensions,
        and then applies it to the OmniGibson environment. It also extracts success
        information and records relevant metrics.

        Args:
            action: Action targets dictionary, expected to contain "actions" or "action".

        Returns:
            The outcome StepResult of the action execution.
        """
        import torch as th

        # Extract action and convert it to a flattened PyTorch tensor.
        raw = action.get("actions", action.get("action"))
        tensor = th.as_tensor(raw, dtype=th.float32).flatten()
        if tensor.shape[0] != R1PRO_ACTION_DIM:
            raise ValueError(f"BEHAVIOR-1K expects a {R1PRO_ACTION_DIM}-D R1Pro joint action, got {tensor.shape[0]}D.")

        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(tensor, n_render_iterations=1)

        # Process and normalize 'info' dictionary and done status.
        info = dict(info)
        info["truncated"] = bool(truncated)
        done = bool(terminated) or bool(truncated)
        done_info = info.get("done") or {}
        success = bool(done_info.get("success", False))

        # Record video frame and step metrics for evaluation.
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(
            reward=float(reward),
            done=done,
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
        )

        return StepResult(obs=obs, reward=float(reward), done=done, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extract and format the robot camera view for video logging.

        This method flattens the raw OmniGibson observation dictionary and attempts to extract
        an RGB image from one of the configured cameras. It handles PyTorch tensor conversion
        to NumPy and RGBA to RGB format if necessary.

        Args:
            raw_obs: Raw observation dictionary from the simulator.

        Returns:
            The image numpy array (H, W, 3) in uint8 format, or None if no suitable camera
            image is found.
        """
        flatten_obs_dict = __import__(
            "omnigibson.learning.utils.eval_utils", fromlist=["flatten_obs_dict"]
        ).flatten_obs_dict

        flat = flatten_obs_dict(raw_obs)
        for cam in self._camera_names:
            key = R1PRO_CAMERAS[cam] + RGB_SUFFIX
            value = flat.get(key)
            if value is None:
                continue
            # Convert PyTorch tensor to NumPy array if applicable.
            if hasattr(value, "cpu"):
                value = value.cpu().numpy()
            arr = np.asarray(value, dtype=np.uint8)
            # OmniGibson VisionSensor returns (H, W, 4) RGBA — drop the alpha channel.
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            return np.ascontiguousarray(arr)
        return None

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw OmniGibson observation dictionary to standard track observation format.

        This method processes the raw, nested observation from the simulator, flattening it
        and extracting images from specified cameras, as well as an optional proprioceptive
        state and the task description.

        Args:
            raw_obs: Raw nested observation from simulator.
            task: Task dictionary containing target metadata, specifically "name" for task description.

        Returns:
            A standard observation dictionary containing images (per camera), task description,
            and optionally proprioceptive states.
        """
        flatten_obs_dict = __import__(
            "omnigibson.learning.utils.eval_utils", fromlist=["flatten_obs_dict"]
        ).flatten_obs_dict

        flat = flatten_obs_dict(raw_obs)

        images: dict[str, np.ndarray] = {}
        for cam in self._camera_names:
            key = R1PRO_CAMERAS[cam] + RGB_SUFFIX
            if key not in flat:
                continue
            value = flat[key]
            # Convert PyTorch tensor to NumPy array if applicable.
            if hasattr(value, "cpu"):
                value = value.cpu().numpy()
            arr = np.asarray(value, dtype=np.uint8)
            # OmniGibson VisionSensor returns (H, W, 4) RGBA — drop alpha.
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            images[cam] = np.ascontiguousarray(arr)

        out: dict[str, Any] = {
            "images": images,
            "task_description": task["name"],
        }

        # Include proprioceptive data if enabled and available.
        if self._send_proprio:
            proprio = flat.get(PROPRIO_KEY)
            if proprio is not None:
                if hasattr(proprio, "cpu"):
                    proprio = proprio.cpu().numpy()
                out["states"] = np.asarray(proprio, dtype=np.float32)

        return out

    def check_done(self, step_result: StepResult) -> bool:
        """Check whether the episode should terminate based on the step result.

        Args:
            step_result: The current StepResult.

        Returns:
            True if the episode is done (terminated or truncated), False otherwise.
        """
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Construct evaluation result metrics from a step result.

        Extracts the binary success flag from the step result's info dictionary.

        Args:
            step_result: The final StepResult.

        Returns:
            A dictionary containing the binary success metric outcome.
        """
        done_info = step_result.info.get("done", {}) or {}
        success = bool(done_info.get("success", False))
        return {"success": success}

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve additional simulator metadata.

        Returns:
            A dictionary containing robot model name, action dimension, task count,
            max steps, and other relevant information about the simulator.
        """
        return {
            "action_dim": R1PRO_ACTION_DIM,
            "max_steps": self._max_steps if self._max_steps is not None else 5000,
            "robot": "R1Pro",
            "n_tasks": len(self._task_names),
        }

    def cleanup(self) -> None:
        """Safely dispose of OmniGibson environment resources.

        This method attempts to close the OmniGibson environment if it exists.
        Note: It intentionally does NOT call ``omnigibson.shutdown()`` because Isaac Sim's shutdown
        path can sometimes hang, preventing the orchestrator from completing its tasks.
        Process exit reclaims all resources, making this the more robust approach.
        """
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.exception("BEHAVIOR-1K env close failed")
            self._env = None

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Get the expected action specification for the simulator.

        Returns:
            A dictionary mapping control fields to DimSpec constraints, specifically
            for the R1Pro robot's 23-dimensional joint position control.
        """
        return {
            "joints": DimSpec("joints", R1PRO_ACTION_DIM, "joint_positions_r1pro"),
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Get the observation specification provided by the simulator.

        Returns:
            A dictionary mapping sensory keys (e.g., camera names, language, state)
            to DimSpec constraints describing their dimensions and types.
        """
        spec: dict[str, DimSpec] = {"language": LANGUAGE}
        for cam in self._camera_names:
            spec[cam] = IMAGE_RGB
        if self._send_proprio:
            spec["state"] = RAW
        return spec

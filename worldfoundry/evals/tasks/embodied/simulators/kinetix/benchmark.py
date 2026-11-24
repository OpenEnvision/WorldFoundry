"""Kinetix benchmark implementation for RTC evaluation.

Kinetix is a JAX-based 2D physics engine with dynamic manipulation tasks
(throwing, catching, balancing, locomotion). This benchmark wraps the 12
tasks used in the RTC paper (arXiv:2506.07339) for evaluation under both
sync and sim2live conditions.

The environment uses a gymnax-style functional API where state is passed
explicitly on every call. This runtime stores JAX state as instance variables
and exposes the BaseSimulator step interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec

logger = logging.getLogger(__name__)

# Defines the dimensionality of the action space for Kinetix tasks:
# 4 motor bindings + 2 thruster bindings = 6 dimensions.
ACTION_DIM = 6

# Default maximum timesteps allowed per episode, as specified in the RTC paper.
MAX_TIMESTEPS = 256

# The 12 tasks from the RTC paper (arXiv:2506.07339, Table 1).
# Each dictionary defines a task with a display "name" and its corresponding
# Kinetix "level" file identifier, which typically lives under the "l/" (large)
# size directory within Kinetix's level structure.
RTC_12_TASKS = [
    {"name": "Grasp Easy", "level": "grasp_easy"},
    {"name": "Catapult", "level": "catapult"},
    {"name": "Cartpole Thrust", "level": "cartpole_thrust"},
    {"name": "Hard Lunar Lander", "level": "hard_lunar_lander"},
    {"name": "Half Cheetah", "level": "mjc_half_cheetah"},
    {"name": "Swimmer", "level": "mjc_swimmer"},
    {"name": "Walker", "level": "mjc_walker"},
    {"name": "Unicycle", "level": "h17_unicycle"},
    {"name": "Chain Lander", "level": "chain_lander"},
    {"name": "Catcher", "level": "catcher_v3"},
    {"name": "Trampoline", "level": "trampoline"},
    {"name": "Car Launch", "level": "car_launch"},
]


def _resolve_level_path(level: str, rtc_worlds_dir: str | None) -> str:
    """Resolves a Kinetix level name to its full loadable file path.

    This function prioritizes custom level paths provided by `rtc_worlds_dir`
    before falling back to Kinetix's built-in level path format.

    Args:
        level: The base name of the Kinetix level (e.g., "grasp_easy").
        rtc_worlds_dir: Optional path to the `worlds/` directory from the RTC
            repository, containing custom level definitions. If `None`, only
            built-in Kinetix levels are considered.

    Returns:
        The resolved file path string for the Kinetix level.
    """
    if rtc_worlds_dir is not None:
        # Construct candidate path for custom RTC levels.
        candidate = Path(rtc_worlds_dir) / "l" / f"{level}.json"
        # Check if the custom level file exists.
        if candidate.is_file():
            return str(candidate)

    # Fall back to kinetix's built-in level path format if no custom path is found or provided.
    return f"l/{level}.json"


class KinetixBenchmark(BaseSimulator):
    """Kinetix 2D physics benchmark for RTC evaluation.

    This class provides an interface to Kinetix environments, implementing the
    `BaseSimulator` API for standardized evaluation. It supports 12 specific
    tasks used in the RTC paper (arXiv:2506.07339).

    Non-obvious behaviors:
        - **JAX functional API**: Kinetix uses gymnax-style functional state.
          ``env.step(rng, state, action, params)`` returns new state — no
          mutation. JAX RNG keys are split on every step.
        - **Pixel observations**: Rendered from the 2D physics state. Default
          resolution is 125×125 (screen_dim=500, downscale=4).
        - **Symbolic state**: Also included in observations under ``"state"``
          for models (like RTC) that use symbolic input.
        - **Env recreation per task**: Each task loads a different level file,
          so the env (with its reset_fn) is recreated when the task changes.

    Attributes:
        _ALL_RECORD_FIELDS (frozenset): Set of fields that are recorded per step.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        max_episode_steps: int = MAX_TIMESTEPS,
        seed: int = 0,
        rtc_worlds_dir: str | None = None,
        observation_type: str = "pixels",
        action_noise_std: float = 0.0,
    ) -> None:
        """Initializes the Kinetix benchmark simulator.

        Args:
            tasks: A list of task names to evaluate. If `None`, all 12 RTC tasks are used.
            max_episode_steps: The maximum number of steps allowed in a single episode.
            seed: The base random seed. This seed is used to generate episode-specific
                JAX random number generator (RNG) keys.
            rtc_worlds_dir: Path to the RTC repo `worlds/` directory for custom
                levels. If `None`, Kinetix's built-in levels are used.
            observation_type: Specifies the type of observations to return:
                `"pixels"` for rendered images (default), or `"symbolic"` for
                the flat symbolic state vector.
            action_noise_std: Standard deviation of Gaussian noise to add to actions.
                A value of 0.0 means no noise.
        """
        super().__init__()
        self._task_names = tasks
        self._max_episode_steps = max_episode_steps
        self._seed = seed
        self._rtc_worlds_dir = rtc_worlds_dir
        if observation_type not in ("pixels", "symbolic"):
            raise ValueError(f"observation_type must be 'pixels' or 'symbolic', got {observation_type!r}")
        self._observation_type = observation_type
        self._action_noise_std = action_noise_std

        # Lazy-initialized JAX state and environment components
        self._env = None  # Kinetix environment instance.
        self._env_state = None  # Current JAX environment state.
        self._env_params = None  # Parameters specific to the current environment configuration.
        self._static_env_params = None  # Static environment parameters loaded from the level.
        self._rng = None  # JAX random number generator key.
        self._current_level: str | None = None  # Name of the currently loaded Kinetix level.
        self._level_state = None  # Initial JAX state loaded from the level file.
        self._step_count = 0  # Counter for steps within the current episode.
        self._episode_success = False  # Flag indicating if the current episode has achieved success.
        self._jax = None  # JAX module, lazily imported to avoid early heavy dependency.
        self._jnp = None  # JAX NumPy module, lazily imported.

    def _init_jax(self) -> None:
        """Lazily imports JAX and JAX NumPy modules.

        This approach avoids a potentially slow import process if JAX is not
        immediately needed, ensuring it's only loaded once.
        """
        if self._jax is not None:
            return
        # Perform JAX and JAX NumPy imports.
        jax = __import__("jax")
        jnp = __import__("jax.numpy", fromlist=["__name__"])

        self._jax = jax
        self._jnp = jnp
        logger.info("JAX initialized, devices: %s", jax.devices())

    def _make_env(self, level: str) -> None:
        """Creates and configures a Kinetix environment for a specific level.

        This method handles loading the level file, setting up environment
        parameters, and JIT compiling the core Kinetix step and reset functions
        for performance.

        Args:
            level: The name of the Kinetix level to load.
        """
        self._init_jax()  # Ensure JAX is initialized.
        assert self._jax is not None
        jax = self._jax

        # Kinetix environment related imports are done here to avoid circular dependencies
        # or early imports of heavy modules if KinetixBenchmark is only partially used.
        kenv = __import__("kinetix.environment", fromlist=["env"]).env
        load_from_json_file = __import__("kinetix.util.saving", fromlist=["load_from_json_file"]).load_from_json_file

        # Resolve the level file path and load its state and parameters.
        level_path = _resolve_level_path(level, self._rtc_worlds_dir)
        level_state, level_static_params, level_env_params = load_from_json_file(level_path)

        # Use the level's own static parameters, as they encode the correct physics configuration.
        static_env_params = level_static_params
        # Override the max_timesteps in environment parameters to match the simulator's configuration.
        env_params = level_env_params.replace(max_timesteps=self._max_episode_steps)

        # Determine the Kinetix environment name based on the desired observation type.
        env_name = (
            "Kinetix-Symbolic-Continuous-v1" if self._observation_type == "symbolic" else "Kinetix-Pixels-Continuous-v1"
        )
        # Create the Kinetix environment instance using the resolved name and static parameters.
        env = kenv.make_kinetix_env_from_name(env_name, static_env_params=static_env_params)

        # Store the environment components as instance variables.
        self._env = env
        self._env_params = env_params
        self._static_env_params = static_env_params
        self._level_state = level_state
        self._current_level = level

        # JIT compile the core environment step and reset functions for performance.
        self._jit_step = jax.jit(env.step_env)
        self._jit_reset = jax.jit(env.reset_env_to_level)

        logger.info("Kinetix env created for level: %s (path: %s)", level, level_path)

    def cleanup(self) -> None:
        """Safely disposes of Kinetix and JAX environment allocations.

        This method helps release resources by nullifying references to JAX
        states and environment objects, preventing potential memory leaks or
        stale references.
        """
        self._env = None
        self._env_state = None
        self._level_state = None
        self._rng = None
        self._current_level = None

    def get_tasks(self) -> list[dict[str, Any]]:
        """Lists the registered tasks for the Kinetix evaluation suite.

        If a subset of task names was specified during initialization, only
        those tasks are returned. Otherwise, all 12 RTC tasks are included.

        Returns:
            A list of task dictionary configurations, each containing at least
            "name" and "level" keys.
        """
        if self._task_names is not None:
            name_set = set(self._task_names)
            # Filter RTC_12_TASKS to only include specified task names.
            return [t for t in RTC_12_TASKS if t["name"] in name_set]
        # Return all 12 predefined RTC tasks if no specific subset was requested.
        return list(RTC_12_TASKS)

    def reset(self, task: dict[str, Any]) -> Any:
        """Resets the Kinetix environment for a new episode.

        This method ensures the correct environment for the given task level
        is loaded and initializes the JAX state and RNG for a new episode.

        Args:
            task: A dictionary containing task-specific information, including
                the "level" name and an optional "episode_idx".

        Returns:
            The initial observation from the reset environment.
        """
        self._init_jax()  # Ensure JAX is initialized.
        assert self._jax is not None
        jax = self._jax

        level = task["level"]
        episode_idx = task.get("episode_idx", 0)

        # Recreate the environment if the level changes or if it hasn't been created yet.
        if self._env is None or self._current_level != level:
            self._make_env(level)

        # Initialize a deterministic JAX RNG key for the episode, based on the base seed and episode index.
        rng = jax.random.PRNGKey(self._seed + episode_idx)
        # Split the RNG key into two: one for the reset operation and one for subsequent steps.
        rng, reset_rng = jax.random.split(rng)

        # Reset the environment to the stored level state (matching RTC's evaluation pattern).
        # This uses the JIT-compiled reset function.
        obs, env_state = self._jit_reset(reset_rng, self._level_state, self._env_params)
        self._env_state = env_state
        self._rng = rng
        self._step_count = 0
        self._episode_success = False  # Reset success flag for the new episode.

        # Record the initial frame for video generation.
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Applies an action to the Kinetix environment and advances the simulation.

        This method handles action formatting, applies optional noise, and
        updates the internal JAX state and episode tracking.

        Args:
            action: A dictionary containing the "actions" or "action" to be
                applied to the environment.

        Returns:
            A `StepResult` object containing the new observation, reward,
            done flag, and additional info.
        """
        assert self._jax is not None and self._jnp is not None
        jax = self._jax
        jnp = self._jnp

        # Extract the raw action array from the input dictionary, supporting both "actions" and "action" keys.
        raw_action = action.get("actions", action.get("action"))
        # Ensure action is a 1D NumPy array of float32, even if it's a scalar.
        raw_action = np.atleast_1d(np.asarray(raw_action, dtype=np.float32))
        # Validate that the action dimension matches the expected `ACTION_DIM`.
        assert raw_action.shape[-1] == ACTION_DIM, (
            f"dict[str, Any] dimension mismatch: got {raw_action.shape[-1]}, expected {ACTION_DIM}"
        )

        # Pad or truncate the action to precisely match the expected `ACTION_DIM`.
        if raw_action.shape[-1] < ACTION_DIM:
            raw_action = np.pad(raw_action, (0, ACTION_DIM - raw_action.shape[-1]))
        elif raw_action.shape[-1] > ACTION_DIM:
            raw_action = raw_action[..., :ACTION_DIM]

        # Convert the NumPy action array to a JAX array for JAX computation.
        jax_action = jnp.array(raw_action)

        assert self._rng is not None
        # Split the current RNG key for the step operation, ensuring functional purity.
        self._rng, step_rng = jax.random.split(self._rng)

        # Apply Gaussian noise to actions if `action_noise_std` is set.
        # This accurately mirrors the behavior of RTC's `NoisyActionWrapper`.
        if self._action_noise_std > 0:
            # Further split `step_rng` to get a separate key for noise generation.
            noise_rng, step_rng = jax.random.split(step_rng)
            jax_action = jax_action + jax.random.normal(noise_rng, jax_action.shape) * self._action_noise_std

        # Perform the environment step using the JIT-compiled function.
        obs, env_state, reward, done, info = self._jit_step(step_rng, self._env_state, jax_action, self._env_params)
        self._env_state = env_state  # Update the internal environment state with the new state.
        self._step_count += 1  # Increment episode step counter.

        # Convert JAX scalars to standard Python types for external use and compatibility.
        reward_val = float(reward)
        done_val = bool(done)

        # Track episode success: in Kinetix RTC tasks, a positive reward indicates contact and thus success.
        if reward_val > 0:
            self._episode_success = True

        # Record the current frame and step metrics for video generation and evaluation.
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=reward_val, done=done_val, success=bool(self._episode_success))

        return StepResult(obs=obs, reward=reward_val, done=done_val, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        """Extracts and formats the rendered viewport frame from raw JAX observations.

        Args:
            raw_obs: The raw observation object from the Kinetix environment.
                This can be a JAX array for symbolic observations or a
                `PixelsObservation` object with an `image` attribute.

        Returns:
            A NumPy array representing the image (HWC, uint8) if observation_type
            is "pixels", otherwise `None`.
        """
        # Symbolic observation runs do not produce a renderable frame.
        if self._observation_type != "pixels":
            return None
        # Access the image attribute or assume raw_obs itself is the image array if directly passed.
        img = np.asarray(getattr(raw_obs, "image", raw_obs))
        # Convert float32 image data in [0,1] range to uint8 in [0,255] if necessary.
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return img

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        """Converts raw JAX observations into a standardized observation dictionary.

        The format of the returned dictionary depends on the `observation_type`
        configured during initialization.

        Args:
            raw_obs: The raw observation object from the Kinetix environment.
            task: A dictionary containing task-specific information, primarily
                used to extract the task name.

        Returns:
            A dictionary containing processed observations, typically including
            image data, symbolic state, and/or task description.
        """
        if self._observation_type == "symbolic":
            # For symbolic observations, raw_obs is expected to be a flat JAX array.
            state = np.asarray(raw_obs, dtype=np.float32)
            # Return a dictionary with the symbolic state vector and task description.
            return {"state": state, "task_description": task["name"]}

        # For pixel observations, raw_obs is a PixelsObservation object with an `image` attribute.
        img = np.asarray(raw_obs.image)

        # Convert float32 image data in [0,1] range to uint8 in [0,255] if necessary.
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

        # Construct the observation dictionary for pixel-based observations.
        obs_dict: dict[str, Any] = {
            "images": {"viewport": img},
            "task_description": task["name"],
        }
        return obs_dict

    def check_done(self, step_result: StepResult) -> bool:
        """Checks if the current episode is finished.

        An episode is considered done if the environment's `done` flag is true,
        or if the maximum number of steps for the episode has been reached.

        Args:
            step_result: The result from the last `step` call.

        Returns:
            `True` if the episode is done, `False` otherwise.
        """
        # The episode is done if the environment signals it, or if the maximum allowed steps have been exceeded.
        return step_result.done or self._step_count >= self._max_episode_steps

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        """Returns additional information about the step, including success status.

        For Kinetix RTC tasks, success is determined by whether any positive
        reward was accumulated during the episode, which is tracked by `_episode_success`.

        Args:
            step_result: The result from the last `step` call.

        Returns:
            A dictionary containing step-specific metrics, such as "success".
        """
        # The success status is derived from the accumulated `_episode_success` flag for the current episode.
        return {"success": self._episode_success}

    def get_metadata(self) -> dict[str, Any]:
        """Provides metadata about the Kinetix environment.

        Returns:
            A dictionary containing metadata such as maximum steps per episode
            and the action space dimension.
        """
        return {
            "max_steps": self._max_episode_steps,
            "action_dim": ACTION_DIM,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Returns the specification for the action space.

        Returns:
            A dictionary mapping action names to their dimension specifications.
            Kinetix actions are raw continuous values.
        """
        return {"action": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Returns the specification for the observation space.

        Returns:
            A dictionary mapping observation names to their dimension specifications.
            The primary observations include 'viewport' for rendered images and
            'language' for task descriptions. Note: 'state' (for symbolic observations)
            is handled by `make_obs` but its spec can vary, so it's not explicitly
            listed here, assuming a typical pixel + language setup.
        """
        return {
            "viewport": IMAGE_RGB,
            "language": LANGUAGE,
        }

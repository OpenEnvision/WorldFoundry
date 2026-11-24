"""LIBERO-Mem benchmark — memory-dependent, non-Markovian manipulation tasks."""

from __future__ import annotations

from typing import Any

from worldfoundry.evals.tasks.embodied.simulators.base import StepResult
from worldfoundry.evals.tasks.embodied.simulators.libero.benchmark import LIBEROBenchmark

_MAX_STEPS = 1000


class LIBEROMemBenchmark(LIBEROBenchmark):
    """Extends LIBEROBenchmark with sequential subgoal tracking for libero-mem."""

    def __init__(
        self,
        suite: str = "libero_mem",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
    ) -> None:
        """Initialize the LIBERO-Mem benchmark.

        Args:
            suite: The suite name (defaults to "libero_mem").
            seed: Random seed for initialization.
            num_steps_wait: Number of open-gripper steps to wait at startup.
            send_wrist_image: Flag to include wrist camera image.
            send_state: Flag to include proprioceptive state.
        """
        super().__init__(
            suite=suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
        )

    def reset(self, task: dict[str, Any]) -> Any:
        """Reset the environment for a memory-dependent LIBERO-mem task episode.

        Extends base reset to raise Robosuite's internal horizon and reset the
        subgoal progress state machine.

        Args:
            task: Task parameters dictionary.

        Returns:
            The initial raw observation after environment reset.
        """
        obs = super().reset(task)
        # Raise robosuite's internal horizon so the harness's max_steps
        # controls episode length (default horizon ~500 < our 1000).
        assert self._env is not None
        self._env.env.horizon = _MAX_STEPS + self.num_steps_wait + 10
        # Reset subgoal state machine — without this, completed subgoals
        # from a previous episode leak into the next one.
        if hasattr(self._env.env, "reset_subgoal_progress"):
            self._env.env.reset_subgoal_progress()
        return obs

    def step(self, action: dict[str, Any]) -> StepResult:
        """Execute action, advancing the sequential subgoal state machine for libero-mem.

        Args:
            action: Predicted control actions.

        Returns:
            The outcome StepResult of the action.
        """
        result = super().step(action)
        # libero-mem's Sequence/Or goals require inc=True to advance the
        # subgoal state machine. Preserve the env's own done flag too.
        assert self._env is not None
        done = result.done or self._env.env._check_success(inc=True)
        # super().step already recorded one row; overwrite its done/success
        # fields with the mutated value (json_patch merges by step_id).
        last_step_id = self._recorder._next_step - 1
        if last_step_id >= 0:
            self._recorder.record_step(step=last_step_id, done=bool(done), success=bool(done))
        return StepResult(obs=result.obs, reward=result.reward, done=done, info=result.info)

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve metadata tailored to LIBERO-mem sequential horizon specs.

        Returns:
            A metadata dictionary containing max_steps and suite identifier.
        """
        return {
            "max_steps": _MAX_STEPS,
            "max_episodes_per_task": 50,
            "suite": self.suite,
        }

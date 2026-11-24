"""AI2-THOR benchmark implementation for native embodied closed-loop evaluation."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from worldfoundry.evals.tasks.embodied.simulators.ai2thor.actions import (
    DEFAULT_MOVE_MAGNITUDE,
    DEFAULT_ROTATE_DEGREES,
    resolve_action_token,
    token_to_thor_action,
)
from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.specs import IMAGE_RGB, LANGUAGE, RAW

logger = logging.getLogger(__name__)

DEFAULT_SCENES: tuple[str, ...] = (
    "FloorPlan1",
    "FloorPlan2",
    "FloorPlan3",
    "FloorPlan4",
    "FloorPlan5",
)

DEFAULT_MAX_STEPS = 200


def _resolve_executable_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for env_key in ("WORLDFOUNDRY_AI2THOR_EXECUTABLE", "AI2THOR_EXECUTABLE_PATH"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value
    return None


class AI2ThorBenchmark(BaseSimulator):
    """AI2-THOR indoor navigation / interaction benchmark.

    This wrapper exposes AI2-THOR through the ``BaseSimulator`` contract used by
    ``EmbodiedClosedLoopRunner``. Policies should emit discrete action tokens such
    as ``forward``, ``left``, or ``pickup``; zero-policy wiring tests default to
    ``forward``.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success", "last_action_success"})

    def __init__(
        self,
        scene: str = "FloorPlan1",
        seed: int | None = 7,
        headless: bool = True,
        executable_path: str | None = None,
        width: int = 300,
        height: int = 300,
        grid_size: float = 0.25,
        max_steps: int = DEFAULT_MAX_STEPS,
        rotate_degrees: float = DEFAULT_ROTATE_DEGREES,
        move_magnitude: float = DEFAULT_MOVE_MAGNITUDE,
        scenes: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.scene = scene
        self.seed = seed
        self.headless = headless
        self.executable_path = _resolve_executable_path(executable_path)
        self.width = int(width)
        self.height = int(height)
        self.grid_size = float(grid_size)
        self.max_steps = int(max_steps)
        self.rotate_degrees = float(rotate_degrees)
        self.move_magnitude = float(move_magnitude)
        self.scenes = tuple(scenes or DEFAULT_SCENES)

        self._controller: Any = None
        self._step_count = 0
        self._episode_success = False

    @staticmethod
    def get_tasks() -> list[dict[str, Any]]:
        """Return default scene-based tasks for smoke and wiring tests."""
        return [
            {
                "task_id": index,
                "name": f"Explore {scene}",
                "scene": scene,
                "suite": "ai2thor_default",
            }
            for index, scene in enumerate(DEFAULT_SCENES)
        ]

    def _import_controller(self) -> Any:
        try:
            Controller = __import__("ai2thor.controller", fromlist=["Controller"]).Controller
        except ImportError as exc:
            raise ImportError(
                "AI2-THOR is not installed. Install `ai2thor` in the active evaluation environment."
            ) from exc
        return Controller

    def _build_controller(self, scene: str) -> Any:
        Controller = self._import_controller()
        kwargs: dict[str, Any] = {
            "scene": scene,
            "gridSize": self.grid_size,
            "width": self.width,
            "height": self.height,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
        }
        if self.executable_path:
            kwargs["local_executable_path"] = self.executable_path
        if self.headless:
            try:
                CloudRendering = __import__("ai2thor.platform", fromlist=["CloudRendering"]).CloudRendering

                kwargs["platform"] = CloudRendering
            except ImportError:
                logger.warning("CloudRendering unavailable; falling back to default AI2-THOR platform.")
        return Controller(**kwargs)

    def reset(self, task: dict[str, Any]) -> Any:
        self.cleanup()
        self._task = dict(task)
        scene = str(task.get("scene") or self.scene)
        self._controller = self._build_controller(scene)
        self._step_count = 0
        self._episode_success = False

        reset_kwargs: dict[str, Any] = {}
        if self.seed is not None:
            reset_kwargs["seed"] = int(task.get("seed", self.seed))
        event = self._controller.reset(**reset_kwargs)
        return event

    def step(self, action: dict[str, Any]) -> StepResult:
        if self._controller is None:
            raise RuntimeError("AI2-THOR controller is not initialized; call reset() first.")

        token = resolve_action_token(action)
        thor_action = token_to_thor_action(
            token,
            rotate_degrees=self.rotate_degrees,
            move_magnitude=self.move_magnitude,
        )
        event = self._controller.step(**thor_action)
        self._step_count += 1

        metadata = dict(getattr(event, "metadata", {}) or {})
        last_action_success = bool(metadata.get("lastActionSuccess", True))
        reward = 1.0 if last_action_success else 0.0
        success = self._evaluate_success(metadata)
        if success:
            self._episode_success = True

        done = success or self._step_count >= self.max_steps
        info = {
            "token": token,
            "thor_action": thor_action,
            "last_action_success": last_action_success,
            "success": self._episode_success,
            "step_count": self._step_count,
            "metadata": metadata,
        }
        result = StepResult(obs=event, reward=reward, done=done, info=info)
        self._last_result = result
        return result

    def _evaluate_success(self, metadata: dict[str, Any]) -> bool:
        goal = str(self._task.get("goal") or "").strip().lower()
        if not goal:
            return False
        if goal == "pickup":
            target_type = self._task.get("object_type")
            inventory = metadata.get("inventoryObjects") or []
            if target_type:
                return any(obj.get("objectType") == target_type for obj in inventory)
            return bool(inventory)
        if goal == "navigate":
            return bool(metadata.get("agentReachedGoal"))
        return False

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        frame = getattr(raw_obs, "frame", None)
        if frame is None and isinstance(raw_obs, dict):
            frame = raw_obs.get("frame")
        if frame is None:
            raise ValueError("AI2-THOR event did not include an RGB frame.")
        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        instruction = str(task.get("instruction") or task.get("name") or "Complete the AI2-THOR task.")
        return {
            "images": {"agentview": rgb},
            "task_description": instruction,
        }

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        info = step_result.info or {}
        return {
            "success": bool(info.get("success", False)),
            "task_success": bool(info.get("success", False)),
            "episode_success": bool(info.get("success", False)),
            "reward": float(step_result.reward),
            "last_action_success": bool(info.get("last_action_success", False)),
            "step_count": int(info.get("step_count", 0)),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "scene": self.scene,
            "headless": self.headless,
            "action_space": "discrete_token",
        }

    def get_action_spec(self) -> dict[str, Any]:
        return {"token": RAW}

    def get_observation_spec(self) -> dict[str, Any]:
        return {
            "agentview": IMAGE_RGB,
            "task_description": LANGUAGE,
        }

    def cleanup(self) -> None:
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception as exc:
                logger.debug("AI2-THOR controller cleanup failed: %s", exc)
        self._controller = None


__all__ = ["DEFAULT_MAX_STEPS", "DEFAULT_SCENES", "AI2ThorBenchmark"]

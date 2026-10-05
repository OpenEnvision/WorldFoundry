"""Runtime registry for the eleven HarnessEval skill entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..protocols import SKILLS
from ..skills import skill_appearance_consistency
from ..skills import skill_drift_degradation
from ..skills import skill_intentional_change_vlm
from ..skills import skill_motion_quality
from ..skills import skill_offscreen_evolution
from ..skills import skill_physical_law
from ..skills import skill_physical_plausibility
from ..skills import skill_physical_response_vlm
from ..skills import skill_render_quality
from ..skills import skill_return_consistency
from ..skills import skill_viewpoint_trajectory


SkillEvaluator = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class SkillEntry:
    skill_id: str
    evaluate: SkillEvaluator
    context_kwargs: tuple[tuple[str, str], ...] = ()


def _entry(module: Any, *context_kwargs: tuple[str, str]) -> SkillEntry:
    return SkillEntry(module.SKILL_ID, module.evaluate, context_kwargs)


SKILL_REGISTRY = {
    entry.skill_id: entry
    for entry in (
        _entry(skill_render_quality),
        _entry(skill_motion_quality),
        _entry(skill_appearance_consistency),
        _entry(skill_physical_plausibility),
        _entry(skill_viewpoint_trajectory, ("case", "case")),
        _entry(
            skill_intentional_change_vlm,
            ("case", "case"),
            ("initial_observation_path", "initial_observation_path"),
        ),
        _entry(
            skill_physical_response_vlm,
            ("case", "case"),
            ("initial_observation_path", "initial_observation_path"),
            ("runtime_metadata", "generation_metadata"),
        ),
        _entry(skill_physical_law, ("case", "case")),
        _entry(
            skill_drift_degradation,
            ("case", "case"),
            ("metadata", "generation_metadata"),
        ),
        _entry(skill_return_consistency, ("case", "case")),
        _entry(
            skill_offscreen_evolution,
            ("case", "case"),
            ("initial_observation_path", "initial_observation_path"),
        ),
    )
}

if tuple(SKILL_REGISTRY) != SKILLS:
    raise RuntimeError("HarnessEval runtime registry does not match the protocol skill order")


def get_skill(skill_id: str) -> SkillEntry:
    try:
        return SKILL_REGISTRY[skill_id]
    except KeyError as exc:
        raise ValueError(f"unknown HarnessEval skill: {skill_id}") from exc


def evaluate_skill(
    skill_id: str,
    *,
    video_path: Path,
    case: Mapping[str, Any],
    initial_observation_path: Path | None,
    generation_metadata: Mapping[str, Any],
    backend: Any,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Call one skill with only the context fields declared by its registry entry."""

    entry = get_skill(skill_id)
    context = {
        "case": case,
        "initial_observation_path": initial_observation_path,
        "generation_metadata": generation_metadata,
    }
    kwargs = {}
    for target, source in entry.context_kwargs:
        value = context[source]
        if value is None:
            raise ValueError(f"{skill_id} requires {source}")
        kwargs[target] = value
    return entry.evaluate(
        video_path=video_path,
        backend=backend,
        cache_root=cache_root,
        **kwargs,
    )

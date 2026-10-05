"""Coverage and exact-common-case leaderboard reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .aggregate import mean_valid, numeric
from .model_order import MODEL_REGISTRY
from .protocols import FAMILIES, get_protocol


OBSERVATION_SKILLS = get_protocol("harnesseval").observation_skills
QUALITY_OBSERVATION_SKILLS = (
    "render_quality_inspector",
    "motion_quality_inspector",
    "appearance_consistency_inspector",
)
PHYSICAL_PLAUSIBILITY_SKILL = "physical_plausibility_inspector"
CORE_WEIGHT = 0.5
OBSERVATION_WEIGHT = 0.5
SCORING_POLICY = {
    "name": "all_families_core_observation_50_50",
    "core_weight": CORE_WEIGHT,
    "observation_weight": OBSERVATION_WEIGHT,
    "observation_skills": list(OBSERVATION_SKILLS),
    "observation_denominator": "numeric_available_dimensions",
}


def leaderboard_case_components(score: dict[str, Any]) -> dict[str, Any]:
    """Return the published score and its auditable per-case components."""

    core = numeric((score.get("case_score") or {}).get("score"))
    dimensions = (score.get("observation_quality") or {}).get("dimensions") or {}
    observation_values = [
        value
        for skill_id in OBSERVATION_SKILLS
        if numeric(value := dimensions.get(skill_id)) is not None
    ]
    observation = (
        sum(float(value) for value in observation_values) / len(observation_values)
        if observation_values
        else None
    )
    published = (
        round(CORE_WEIGHT * core + OBSERVATION_WEIGHT * observation, 6)
        if core is not None and observation is not None
        else None
    )
    return {
        "score": published,
        "core_score": core,
        "observation_score": round(observation, 6) if observation is not None else None,
        "observation_count": len(observation_values),
    }


def build_report(scores: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = Counter()
    by_model_family: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    by_model_family_core: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    by_model_family_observation: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    by_model_family_quality: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    by_model_family_physical: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    observation_count_distribution = Counter()
    schemas = Counter()
    for score in scores:
        model = str(score["model_id"])
        family = str(score["taxonomy"]["probe_family"])
        case_id = str(score["case_id"])
        components = leaderboard_case_components(score)
        value = numeric(components["score"])
        coverage[(model, family)] += 1
        schemas[str(score.get("schema_version"))] += 1
        observation_count_distribution[str(components["observation_count"])] += 1
        if value is not None:
            by_model_family[(model, family)][case_id] = value
            by_model_family_core[(model, family)][case_id] = components["core_score"]
            if components["observation_score"] is not None:
                by_model_family_observation[(model, family)][case_id] = components[
                    "observation_score"
                ]
            dimensions = (score.get("observation_quality") or {}).get("dimensions") or {}
            quality_values = [
                float(value)
                for skill_id in QUALITY_OBSERVATION_SKILLS
                if numeric(value := dimensions.get(skill_id)) is not None
            ]
            if quality_values:
                by_model_family_quality[(model, family)][case_id] = round(
                    sum(quality_values) / len(quality_values), 6
                )
            physical = numeric(dimensions.get(PHYSICAL_PLAUSIBILITY_SKILL))
            if physical is not None:
                by_model_family_physical[(model, family)][case_id] = physical

    observed_models = {model for model, _ in coverage}
    models = [id for id in MODEL_REGISTRY if id in observed_models]
    models.extend(sorted(observed_models - set(MODEL_REGISTRY)))
    common_cases: dict[str, list[str]] = {}
    for family in FAMILIES:
        sets = [set(by_model_family.get((model, family), {})) for model in models]
        common_cases[family] = sorted(set.intersection(*sets)) if sets and all(sets) else []

    rows = []
    for model in models:
        family_scores = {}
        family_core_scores = {}
        family_observation_scores = {}
        family_quality_scores = {}
        family_physical_plausibility_scores = {}
        for family in FAMILIES:
            case_ids = common_cases[family]
            values = [by_model_family[(model, family)][case_id] for case_id in case_ids]
            family_scores[family] = mean_valid(values)
            family_core_scores[family] = mean_valid(
                by_model_family_core[(model, family)].get(case_id) for case_id in case_ids
            )
            family_observation_scores[family] = mean_valid(
                by_model_family_observation[(model, family)].get(case_id)
                for case_id in case_ids
            )
            family_quality_scores[family] = mean_valid(
                by_model_family_quality[(model, family)].get(case_id)
                for case_id in case_ids
            )
            family_physical_plausibility_scores[family] = mean_valid(
                by_model_family_physical[(model, family)].get(case_id)
                for case_id in case_ids
            )
        overall = (
            mean_valid(family_scores.values())
            if all(numeric(value) is not None for value in family_scores.values())
            else None
        )
        overall_core = (
            mean_valid(family_core_scores.values())
            if all(numeric(value) is not None for value in family_core_scores.values())
            else None
        )
        overall_observation = (
            mean_valid(family_observation_scores.values())
            if all(
                numeric(value) is not None
                for value in family_observation_scores.values()
            )
            else None
        )
        observation_quality = (
            mean_valid(family_quality_scores.values())
            if all(numeric(value) is not None for value in family_quality_scores.values())
            else None
        )
        observation_physical_plausibility = (
            mean_valid(family_physical_plausibility_scores.values())
            if all(
                numeric(value) is not None
                for value in family_physical_plausibility_scores.values()
            )
            else None
        )
        rows.append(
            {
                "model_id": model,
                "overall_macro": overall,
                "overall_core_macro": overall_core,
                "overall_observation_macro": overall_observation,
                "observation_quality_macro": observation_quality,
                "observation_physical_plausibility_macro": (
                    observation_physical_plausibility
                ),
                "family_scores": family_scores,
                "family_core_scores": family_core_scores,
                "family_observation_scores": family_observation_scores,
                "family_quality_scores": family_quality_scores,
                "family_physical_plausibility_scores": (
                    family_physical_plausibility_scores
                ),
                "coverage": {
                    family: coverage[(model, family)] for family in FAMILIES
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["overall_macro"] is not None,
            row["overall_macro"] if row["overall_macro"] is not None else -1,
        ),
        reverse=True,
    )
    return {
        "schema_version": "harnesseval.report",
        "score_count": len(scores),
        "model_count": len(models),
        "schemas": dict(schemas),
        "common_case_counts": {key: len(value) for key, value in common_cases.items()},
        "scoring_policy": SCORING_POLICY,
        "observation_groups": {
            "quality": list(QUALITY_OBSERVATION_SKILLS),
            "physical_plausibility": [PHYSICAL_PLAUSIBILITY_SKILL],
        },
        "observation_count_distribution": dict(
            sorted(observation_count_distribution.items())
        ),
        "leaderboard": rows,
    }

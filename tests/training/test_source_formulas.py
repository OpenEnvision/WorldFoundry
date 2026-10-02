"""Direct numerical checks derived from fixed upstream source formulas."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.objectives import (  # noqa: E402
    flow_interpolate,
    flow_matching_mse,
    flow_velocity_target,
)
from worldfoundry.training.post_training import (  # noqa: E402
    FlowSDEIndexSchedule,
    FlowSDEWindowSchedule,
    PostTrainingParallelContext,
    WeightedRewardScalarizer,
    clipped_policy_loss,
    constant_diffusion_flow_transition,
    dmd_distribution_gradient,
    dmd_proxy_loss,
    dmd_teacher_guidance,
    flow_match_sigma_schedule,
    flow_sde_transition,
    normalize_grouped_advantages,
    normalize_weighted_component_advantages,
)
from worldfoundry.training.post_training.rl.algorithms.flow_dppo.objective import (  # noqa: E402
    flow_dppo_policy_loss,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402


def _load_fixture(name: str) -> dict[str, object]:
    path = Path(__file__).parent / "fixtures/source_formulas" / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"source-formula fixture must be a mapping: {path}")
    return value


def _plain(value: object) -> object:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _assert_source_formula(fixture: Mapping[str, object], actual: Mapping[str, object]) -> int:
    expected = fixture["expected"]
    atol = float(fixture["atol"])
    rtol = float(fixture["rtol"])
    compared = 0

    def compare(left: object, right: object, path: str) -> None:
        nonlocal compared
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping):
                raise AssertionError(f"{path}: expected mapping, got {type(right).__name__}")
            if set(left) != set(right):
                raise AssertionError(f"{path}: keys differ")
            for key in sorted(left):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                raise AssertionError(f"{path}: sequence length differs")
            for index, (expected_item, actual_item) in enumerate(zip(left, right, strict=True)):
                compare(expected_item, actual_item, f"{path}[{index}]")
            return
        compared += 1
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
                raise AssertionError(f"{path}: expected {left!r}, got {right!r}")
            return
        if left != right:
            raise AssertionError(f"{path}: expected {left!r}, got {right!r}")

    compare(expected, _plain(actual), "expected")
    return compared


def test_pinned_unirl_source_formulas_match_flow_grpo_math() -> None:
    fixture = _load_fixture("unirl-flow-grpo.json")
    inputs = fixture["inputs"]
    schedule_input = inputs["schedule"]
    transition_input = inputs["transition"]
    objective_input = inputs["objective"]
    reward_input = inputs["reward"]

    index_schedule = FlowSDEIndexSchedule(
        transition_count=int(schedule_input["num_inference_steps"]),
        timestep_fraction=tuple(schedule_input["timestep_fraction"]),
        num_sde_steps=int(schedule_input["num_sde_steps"]),
    )
    transition = flow_sde_transition(
        torch.tensor(transition_input["velocity"], dtype=torch.float32),
        torch.tensor(transition_input["sample"], dtype=torch.float32),
        torch.tensor(transition_input["sigma"], dtype=torch.float32),
        torch.tensor(transition_input["sigma_next"], dtype=torch.float32),
        eta=float(transition_input["eta"]),
        sigma_max=float(transition_input["sigma_max"]),
        next_sample=torch.tensor(
            transition_input["observed_next"],
            dtype=torch.float32,
        ),
        trajectory_dtype=torch.float32,
    )
    advantages = normalize_grouped_advantages(
        torch.tensor(objective_input["rewards"], dtype=torch.float32),
        tuple(objective_input["group_ids"]),
        epsilon=float(objective_input["advantage_epsilon"]),
        normalization=str(objective_input["advantage_normalization"]),
    ).advantages
    objective = clipped_policy_loss(
        torch.tensor(objective_input["new_log_probs"], dtype=torch.float32),
        torch.tensor(objective_input["old_log_probs"], dtype=torch.float32),
        advantages,
        clip_range=float(objective_input["clip_range"]),
    )
    scalarizer = WeightedRewardScalarizer(
        reward_input["weights"],
        calibration_mean=reward_input["calibration_mean"],
        calibration_std=reward_input["calibration_std"],
        normalization_epsilon=float(reward_input["normalization_epsilon"]),
    )
    scalarized = scalarizer.scalarize(
        {key: torch.tensor(value, dtype=torch.float32) for key, value in reward_input["raw_components"].items()}
    )
    actual = {
        "sigma_schedule": flow_match_sigma_schedule(
            int(schedule_input["num_inference_steps"]),
            shift=float(schedule_input["shift"]),
        ),
        "sde_indices": {
            f"rollout-{rollout_id}": index_schedule.resolve(int(rollout_id))
            for rollout_id in schedule_input["rollout_ids"]
        },
        "transition": {
            "mean": transition.mean,
            "scale": transition.scale,
            "log_prob": transition.log_prob,
        },
        "objective": {
            "advantages": advantages,
            "ratio": objective.ratio,
            "loss": objective.loss,
            "approx_kl": objective.approx_kl,
            "clip_fraction": objective.clip_fraction,
            "lower_clip_fraction": objective.lower_clip_fraction,
            "upper_clip_fraction": objective.upper_clip_fraction,
        },
        "reward": {
            "normalized_components": scalarized.normalized_components,
            "scalar_rewards": scalarized.scalar_rewards,
        },
    }

    assert _assert_source_formula(fixture, actual) == 82


def test_source_formula_assertion_reports_exact_tensor_path() -> None:
    fixture = _load_fixture("unirl-flow-grpo.json")
    expected = fixture["expected"]
    assert isinstance(expected, dict)

    with pytest.raises(AssertionError, match=r"expected\.sigma_schedule"):
        _assert_source_formula(fixture, {**expected, "sigma_schedule": [0.0]})


def test_pinned_dance_mix_source_formulas_match_masked_and_component_first_math() -> None:
    fixture = _load_fixture("dance-mix-grpo.json")
    inputs = fixture["inputs"]
    component_input = inputs["component_advantages"]
    objective_input = inputs["masked_objective"]
    components = normalize_weighted_component_advantages(
        {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in component_input["rewards"].items()
        },
        component_input["weights"],
        tuple(component_input["group_ids"]),
        parallel_context=PostTrainingParallelContext.current(),
        epsilon=float(component_input["epsilon"]),
        normalization="group-sample-std",
    )
    objective = clipped_policy_loss(
        torch.tensor(objective_input["new_log_probs"], dtype=torch.float32),
        torch.tensor(objective_input["old_log_probs"], dtype=torch.float32),
        components.advantages,
        clip_range=float(objective_input["clip_range"]),
        step_mask=torch.tensor(objective_input["step_mask"], dtype=torch.bool),
    )
    actual = {
        "component_advantages": {
            name: result.advantages for name, result in components.components.items()
        },
        "normalized_weights": components.normalized_weights,
        "merged_advantages": components.advantages,
        "masked_objective": {
            "loss": objective.loss,
            "approx_kl": objective.approx_kl,
            "clip_fraction": objective.clip_fraction,
            "lower_clip_fraction": objective.lower_clip_fraction,
            "upper_clip_fraction": objective.upper_clip_fraction,
        },
    }

    assert _assert_source_formula(fixture, actual) == 19


def test_pinned_verl_omni_source_formulas_match_flow_dppo_math() -> None:
    fixture = _load_fixture("verl-omni-flow-dppo.json")
    inputs = fixture["inputs"]
    objective = flow_dppo_policy_loss(
        torch.tensor(inputs["new_log_probs"], dtype=torch.float32),
        torch.tensor(inputs["old_log_probs"], dtype=torch.float32),
        torch.tensor(inputs["new_transition_means"], dtype=torch.float32),
        torch.tensor(inputs["old_transition_means"], dtype=torch.float32),
        torch.tensor(inputs["transition_scales"], dtype=torch.float32),
        torch.tensor(inputs["advantages"], dtype=torch.float32),
        kl_mask_threshold=float(inputs["kl_mask_threshold"]),
        add_kl_coefficient=bool(inputs["add_kl_coefficient"]),
    )

    actual = {
        "ratio": objective.ratio,
        "old_policy_kl": objective.old_policy_kl,
        "keep_mask": objective.keep_mask,
        "per_transition": objective.per_transition,
        "loss": objective.loss,
        "masked_fraction": objective.masked_fraction,
        "positive_masked_fraction": objective.positive_masked_fraction,
        "negative_masked_fraction": objective.negative_masked_fraction,
        "approx_kl": objective.approx_kl,
    }

    assert _assert_source_formula(fixture, actual) == 21


def test_fixed_transition_and_window_source_formulas_match_primitives() -> None:
    fixture = _load_fixture("flow-transition-window-primitives.json")
    inputs = fixture["inputs"]
    transition_input = inputs["constant_diffusion"]
    window_input = inputs["window"]
    transition = constant_diffusion_flow_transition(
        torch.tensor(transition_input["velocity"], dtype=torch.float32),
        torch.tensor(transition_input["sample"], dtype=torch.float32),
        torch.tensor(transition_input["sigma"], dtype=torch.float32),
        torch.tensor(transition_input["sigma_next"], dtype=torch.float32),
        eta=float(transition_input["eta"]),
        next_sample=torch.tensor(
            transition_input["observed_next"],
            dtype=torch.float32,
        ),
        trajectory_dtype=torch.float32,
    )
    window = FlowSDEWindowSchedule(
        transition_count=int(window_input["transition_count"]),
        window_size=int(window_input["window_size"]),
        iterations_per_window=int(window_input["iterations_per_window"]),
        stride=int(window_input["stride"]),
        initial_index=int(window_input["initial_index"]),
        rollback=bool(window_input["rollback"]),
    )
    actual = {
        "constant_diffusion": {
            "mean": transition.mean,
            "scale": transition.scale,
            "log_prob": transition.log_prob,
        },
        "window_indices": {
            f"rollout-{rollout_id}": window.resolve(int(rollout_id)) for rollout_id in window_input["rollout_ids"]
        },
    }

    assert _assert_source_formula(fixture, actual) == 28


def test_pinned_fastvideo_source_formulas_match_dmd_math_and_profile() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = _load_fixture("fastvideo-wan-dmd.json")
    inputs = fixture["inputs"]
    recipe = PostTrainingRecipe.from_file(root / str(inputs["profile_recipe"]))
    guidance_input = inputs["guidance"]
    distribution_input = inputs["distribution"]
    renoise_input = inputs["renoise"]
    fake_input = inputs["fake_score"]
    generated = torch.tensor(
        distribution_input["generated"],
        dtype=torch.float32,
        requires_grad=True,
    )
    guided = dmd_teacher_guidance(
        torch.tensor(guidance_input["conditional"], dtype=torch.float32),
        torch.tensor(guidance_input["unconditional"], dtype=torch.float32),
        float(guidance_input["scale"]),
    )
    gradient, denominator = dmd_distribution_gradient(
        generated,
        torch.tensor(distribution_input["fake_score"], dtype=torch.float32),
        guided,
        normalization_epsilon=float(distribution_input["normalization_epsilon"]),
    )
    proxy_loss = dmd_proxy_loss(generated, gradient)
    proxy_loss.backward()
    clean = torch.tensor(renoise_input["clean"], dtype=torch.float32)
    noise = torch.tensor(renoise_input["noise"], dtype=torch.float32)
    target_velocity = flow_velocity_target(clean, noise)
    fake_loss = flow_matching_mse(
        torch.tensor(fake_input["prediction"], dtype=torch.float32),
        target_velocity,
    ).loss
    algorithm = recipe.algorithm
    assert recipe.fake_score_optimizer is not None
    actual = {
        "profile": {
            "student_timesteps": algorithm.student_timesteps,
            "student_sigmas": algorithm.student_sigmas,
            "score_min_sigma": algorithm.score_min_sigma,
            "score_max_sigma": algorithm.score_max_sigma,
            "score_flow_shift": algorithm.score_flow_shift,
            "teacher_guidance_scale": algorithm.teacher_guidance_scale,
            "generator_update_interval": algorithm.generator_update_interval,
            "student_learning_rate": recipe.optimizer.learning_rate,
            "fake_score_learning_rate": recipe.fake_score_optimizer.learning_rate,
            "student_weight_decay": recipe.optimizer.weight_decay,
            "fake_score_weight_decay": recipe.fake_score_optimizer.weight_decay,
            "student_gradient_accumulation_steps": (
                recipe.optimizer.gradient_accumulation_steps
            ),
            "fake_score_gradient_accumulation_steps": (
                recipe.fake_score_optimizer.gradient_accumulation_steps
            ),
        },
        "generator_update_cadence": [
            (step + 1) % algorithm.generator_update_interval == 0
            for step in range(10)
        ],
        "guided_real_score": guided,
        "distribution": {
            "denominator": denominator,
            "gradient": gradient,
            "proxy_loss": proxy_loss,
            "proxy_gradient": generated.grad,
        },
        "renoised": flow_interpolate(
            clean,
            noise,
            torch.tensor(renoise_input["sigma"], dtype=torch.float32),
        ),
        "fake_score": {
            "target_velocity": target_velocity,
            "loss": fake_loss,
        },
    }

    assert _assert_source_formula(fixture, actual) == 50

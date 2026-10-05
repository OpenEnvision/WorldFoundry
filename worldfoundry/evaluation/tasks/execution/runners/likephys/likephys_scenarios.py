"""LikePhys scenario, variation, and probe-configuration tables.

LikePhys is a white-box benchmark: instead of scoring generated videos, it feeds a
curated set of paired physically-valid / physically-impossible videos through a video
diffusion model and uses the denoising objective as an ELBO-based likelihood surrogate.
A model "understands" a physical rule when it assigns a lower denoising loss to the
valid video than to its impossible counterpart.

Every constant below mirrors the official release:

* Scenario ids, dataset directory names, and text prompts follow ``evaluator.py``
  (``data_config`` and ``get_prompt``).
* Variation names follow the ``JianhaoDYDY/LikePhys-Benchmark`` dataset tree, where each
  scenario holds ``subgroup_XXX/<variation>_<index>.mp4`` clips and ``valid`` marks the
  physically plausible clip of the subgroup.
* Probe geometry (resolution, frame count, fps) and guidance strengths follow
  ``get_model_params`` and ``initialize_model``.
* ``REPORTED_VARIATION_FILTERS`` mirrors ``read_exp_final.py``'s ``filter_config``: those
  variations are excluded from the aggregates reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

BENCHMARK_ID = "likephys"
DISPLAY_NAME = "LikePhys"

# Shared negative prompt used for every scenario in ``evaluator.py:get_prompt``.
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Variation name that marks the physically plausible clip inside each subgroup.
VALID_VARIATION = "valid"


@dataclass(frozen=True)
class LikePhysScenario:
    """One LikePhys physics scenario and its impossible-variation family."""

    scenario_id: str
    dataset_dir: str
    prompt: str
    domain: str
    variations: tuple[str, ...]

    @property
    def invalid_variations(self) -> tuple[str, ...]:
        """Impossible variations paired against ``valid`` inside every subgroup."""
        return tuple(name for name in self.variations if name != VALID_VARIATION)


SCENARIOS: tuple[LikePhysScenario, ...] = (
    LikePhysScenario(
        "ball_drop",
        "ball_drop_videos",
        "ball dropping and colliding with the ground, in empty background",
        "rigid_body",
        (
            "color_change",
            "dynamic_scaling",
            "over_bounce",
            "penetration",
            "teleportation",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "ball_collision",
        "ball_collision_videos",
        "two balls colliding with each other",
        "rigid_body",
        (
            "invalid_momentum_amplification",
            "invalid_penetration",
            "invalid_phantom_force",
            "invalid_size_change",
            "invalid_teleportation",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "pendulum",
        "pendulum_videos",
        "a pendulum swinging",
        "rigid_body",
        (
            "break_conservation_energy",
            "break_continuity",
            "disappearing_bob",
            "path_deviation",
            "reverse_gravity",
            "temporal_disorder",
            "valid",
            "varying_frequency",
        ),
    ),
    LikePhysScenario(
        "block_slide",
        "block_slide_videos",
        "a block sliding on a slope",
        "rigid_body",
        (
            "invalid_hovering",
            "invalid_irregular_motion",
            "invalid_jittering",
            "invalid_size_changing",
            "invalid_teleportation",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "pyramid",
        "pyramid_videos",
        "a cube crash into a pile of spheres",
        "rigid_body",
        (
            "invalid_anti_gravity",
            "invalid_momentum_multiplication",
            "invalid_phase_shifting",
            "invalid_sphere_fusion",
            "invalid_teleporting_spheres",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "fluid",
        "fluid_videos",
        "a droplet falling",
        "fluid",
        (
            "antigravity_fluid",
            "discontinuous_fluid",
            "matter_creation",
            "non_conservation_momentum",
            "phase_transition",
            "self_attracting",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "faucet",
        "faucet_videos",
        "fluid flowing from a faucet",
        "fluid",
        (
            "color_change",
            "fracturing_fluid",
            "negative_viscosity",
            "non_conservation_fluid",
            "oscillating_viscosity",
            "phase_shifting_fluid",
            "self_attracting",
            "teleporting_fluid",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "river",
        "river_videos",
        "fluid flowing in a tank with obstacles",
        "fluid",
        (
            "color_change",
            "fracturing_fluid",
            "invisible_wall",
            "negative_viscosity",
            "non_conservation_fluid",
            "phase_shifting_fluid",
            "teleporting_fluid",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "cloth",
        "cloth_drape_videos",
        "a piece of cloth dropping to the obstacle on the ground",
        "deformable",
        (
            "color_change",
            "ground_penetration",
            "impossible_folding",
            "penetration",
            "rubber_cloth",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "flag",
        "flag_videos",
        "a piece of cloth waving in the wind",
        "deformable",
        (
            "color_change",
            "elastic_explosion",
            "flag_shatter",
            "flag_teleport",
            "impossible_twist",
            "sudden_freeze",
            "temporal_disorder",
            "valid",
        ),
    ),
    LikePhysScenario(
        "shadow",
        "shadow_videos",
        "light source moving around an object showing its shadow",
        "optics",
        (
            "inverted_shadow",
            "no_object",
            "no_shadow",
            "temporal_disorder",
            "valid",
            "varying_object",
            "wrong_shadow_shape",
        ),
    ),
    LikePhysScenario(
        "shadowm",
        "shadow_camera_videos",
        "camera moving around an object",
        "optics",
        (
            "impossible_reflection",
            "no_shadow",
            "shadow_disconnection",
            "sudden_disappearance",
            "temporal_disorder",
            "valid",
            "varying_size",
        ),
    ),
)

SCENARIOS_BY_ID: Mapping[str, LikePhysScenario] = {scenario.scenario_id: scenario for scenario in SCENARIOS}
SCENARIO_ORDER: tuple[str, ...] = tuple(scenario.scenario_id for scenario in SCENARIOS)

# Order used by ``run_eval.sh``; kept so batch runs reproduce the official sweep order.
OFFICIAL_SCENARIO_SWEEP: tuple[str, ...] = (
    "ball_drop",
    "ball_collision",
    "pendulum",
    "block_slide",
    "pyramid",
    "fluid",
    "faucet",
    "river",
    "flag",
    "cloth",
    "shadow",
    "shadowm",
)

DOMAIN_SCENARIOS: Mapping[str, tuple[str, ...]] = {
    "rigid_body": ("ball_drop", "ball_collision", "pendulum", "block_slide", "pyramid"),
    "fluid": ("fluid", "faucet", "river"),
    "deformable": ("cloth", "flag"),
    "optics": ("shadow", "shadowm"),
}

# ``read_exp_final.py:filter_config`` — variations dropped from the reported aggregates.
REPORTED_VARIATION_FILTERS: Mapping[str, tuple[str, ...]] = {
    "shadowm": ("shadow_disconnection",),
    "pendulum": ("reverse_gravity", "break_conservation_energy"),
    "river": ("color_change", "teleporting_fluid", "negative_viscosity"),
}

# ``read_exp_final.py:ignore_models`` — probe backends excluded from the published table.
REPORTED_IGNORED_MODELS: frozenset[str] = frozenset({"svd"})

CANONICAL_SCENARIO_COUNT = len(SCENARIOS)
CANONICAL_SUBGROUP_COUNT = 10
CANONICAL_VARIATION_COUNT = sum(len(scenario.invalid_variations) for scenario in SCENARIOS)
CANONICAL_VIDEO_COUNT = CANONICAL_SUBGROUP_COUNT * sum(len(scenario.variations) for scenario in SCENARIOS)


@dataclass(frozen=True)
class LikePhysProbeModel:
    """Probe geometry and guidance strength for one supported diffusion backend."""

    model_key: str
    display_name: str
    pretrained_id: str
    height: int
    width: int
    fps: int
    num_frames: int
    guidance_scale: float


# ``get_model_params`` + ``initialize_model``; ``guidance_scale`` is the CFG strength used
# when ``--guidance_scale`` is passed (the official sweep always passes it).
PROBE_MODELS: tuple[LikePhysProbeModel, ...] = (
    LikePhysProbeModel(
        "animatediff",
        "AnimateDiff",
        "SG161222/Realistic_Vision_V5.1_noVAE+guoyww/animatediff-motion-adapter-v1-5-2",
        512,
        512,
        16,
        16,
        7.5,
    ),
    LikePhysProbeModel(
        "animatediff_sdxl",
        "AnimateDiff SDXL",
        "stabilityai/stable-diffusion-xl-base-1.0+guoyww/animatediff-motion-adapter-sdxl-beta",
        1024,
        1024,
        16,
        16,
        7.5,
    ),
    LikePhysProbeModel("modelscope", "ModelScope", "damo-vilab/text-to-video-ms-1.7b", 320, 576, 16, 24, 5.0),
    LikePhysProbeModel("zeroscope", "ZeroScope", "cerspense/zeroscope_v2_576w", 320, 576, 16, 24, 5.0),
    LikePhysProbeModel("cogvideox", "CogVideoX-2B", "THUDM/CogVideoX-2b", 480, 720, 16, 49, 6.0),
    LikePhysProbeModel("cogvideox-5b", "CogVideoX-5B", "THUDM/CogVideoX-5B", 480, 720, 16, 49, 6.0),
    LikePhysProbeModel("cogvideox1.5-5b", "CogVideoX1.5-5B", "THUDM/CogVideoX1.5-5B", 768, 1360, 16, 85, 6.0),
    LikePhysProbeModel("hunyuan_t2v", "HunyuanVideo T2V", "hunyuanvideo-community/HunyuanVideo", 320, 512, 16, 61, 6.0),
    LikePhysProbeModel(
        "wan2.1-T2V-1.3b",
        "Wan2.1-T2V-1.3B",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        480,
        832,
        16,
        33,
        5.0,
    ),
    LikePhysProbeModel(
        "wan2.1-T2V-14b",
        "Wan2.1-T2V-14B",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        480,
        832,
        16,
        33,
        5.0,
    ),
    LikePhysProbeModel("ltx", "LTX-Video", "Lightricks/LTX-Video", 480, 704, 25, 161, 3.0),
    LikePhysProbeModel("ltx-0.9.1", "LTX-Video 0.9.1", "Lightricks/LTX-Video-0.9.1", 480, 704, 25, 161, 3.0),
    LikePhysProbeModel("ltx-0.9.5", "LTX-Video 0.9.5", "Lightricks/LTX-Video-0.9.5", 480, 704, 25, 161, 3.0),
    LikePhysProbeModel("mochi", "Mochi 1", "genmo/mochi-1-preview", 480, 848, 16, 85, 4.5),
    LikePhysProbeModel(
        "svd",
        "Stable Video Diffusion",
        "stabilityai/stable-video-diffusion-img2vid",
        224,
        224,
        7,
        14,
        3.0,
    ),
)

PROBE_MODELS_BY_KEY: Mapping[str, LikePhysProbeModel] = {model.model_key: model for model in PROBE_MODELS}

# ``run_eval.sh:MODELS`` — the backends swept for the published table.
OFFICIAL_PROBE_SWEEP: tuple[str, ...] = (
    "animatediff",
    "zeroscope",
    "modelscope",
    "wan2.1-T2V-1.3b",
    "hunyuan_t2v",
    "ltx-0.9.5",
    "animatediff_sdxl",
    "cogvideox",
    "mochi",
    "cogvideox-5b",
    "wan2.1-T2V-14b",
)

# ``evaluator.py`` defaults: ``--exp_name`` / ``--timestep_num`` / ``--timestep_strategy``.
DEFAULT_EXPERIMENT_NAME = "evaluation_t10_uniform"
DEFAULT_TIMESTEP_NUM = 10
DEFAULT_TIMESTEP_STRATEGY = "uniform"
DEFAULT_SEED = 42


def scenario_for_id(scenario_id: str) -> LikePhysScenario:
    """Return the scenario table entry for ``scenario_id``.

    Raises:
        KeyError: If ``scenario_id`` is not an official LikePhys scenario.
    """
    try:
        return SCENARIOS_BY_ID[scenario_id]
    except KeyError as exc:
        known = ", ".join(SCENARIO_ORDER)
        raise KeyError(f"unknown LikePhys scenario {scenario_id!r}; known: {known}") from exc


_VARIATION_SIGNATURES: Mapping[frozenset[str], str] = {
    frozenset(scenario.invalid_variations): scenario.scenario_id for scenario in SCENARIOS
}


def scenario_id_for_variations(variations: Iterable[str]) -> str | None:
    """Infer a scenario id from its impossible-variation set.

    Result artifacts do not name their scenario, but each of the 12 scenarios owns a
    unique set of impossible variations, so an exact match identifies it. Used when
    results reach the scorer without their ``<scenario>/`` directory context.
    """
    signature = frozenset(str(name) for name in variations if str(name) != VALID_VARIATION)
    return _VARIATION_SIGNATURES.get(signature)


def reported_variation_filter(scenario_id: str) -> frozenset[str]:
    """Return variations excluded from the reported aggregate for ``scenario_id``."""
    return frozenset(REPORTED_VARIATION_FILTERS.get(scenario_id, ()))


def domain_for_scenario(scenario_id: str) -> str | None:
    """Return the physics domain group owning ``scenario_id``."""
    for domain, scenario_ids in DOMAIN_SCENARIOS.items():
        if scenario_id in scenario_ids:
            return domain
    return None


def official_experiment_dirname(
    *,
    seed: int = DEFAULT_SEED,
    guidance_scale: bool = True,
    tag_name: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    """Return the ``results/<exp_name>`` directory name produced by ``evaluator.py``."""
    cfg_tag = "cfg" if guidance_scale else "no_cfg"
    if tag_name and tag_name.strip():
        return f"{experiment_name}_{seed}_{cfg_tag}_{tag_name.strip()}"
    return f"{experiment_name}_{seed}_{cfg_tag}"


__all__ = [
    "BENCHMARK_ID",
    "CANONICAL_SCENARIO_COUNT",
    "CANONICAL_SUBGROUP_COUNT",
    "CANONICAL_VARIATION_COUNT",
    "CANONICAL_VIDEO_COUNT",
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_SEED",
    "DEFAULT_TIMESTEP_NUM",
    "DEFAULT_TIMESTEP_STRATEGY",
    "DISPLAY_NAME",
    "DOMAIN_SCENARIOS",
    "LikePhysProbeModel",
    "LikePhysScenario",
    "NEGATIVE_PROMPT",
    "OFFICIAL_PROBE_SWEEP",
    "OFFICIAL_SCENARIO_SWEEP",
    "PROBE_MODELS",
    "PROBE_MODELS_BY_KEY",
    "REPORTED_IGNORED_MODELS",
    "REPORTED_VARIATION_FILTERS",
    "SCENARIOS",
    "SCENARIOS_BY_ID",
    "SCENARIO_ORDER",
    "VALID_VARIATION",
    "domain_for_scenario",
    "official_experiment_dirname",
    "reported_variation_filter",
    "scenario_for_id",
    "scenario_id_for_variations",
]

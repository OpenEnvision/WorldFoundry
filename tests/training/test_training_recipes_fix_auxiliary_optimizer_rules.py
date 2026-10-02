"""TR-10: auxiliary-optimizer validation is declared per algorithm spec.

The former ~120-line isinstance chain in ``PostTrainingRecipe.__post_init__``
now dispatches through ``auxiliary_optimizer_rules()`` on each spec.  These
tests pin the exact legacy error message for every branch of the old chain and
exercise the default reject-all fallback for undeclared specs.
"""

from __future__ import annotations

import pytest

from worldfoundry.training.recipes.post_training.algorithms.adaptive_video import (
    AdaptiveVideoAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.adversarial_diffusion import (
    AdversarialDiffusionAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.anyflow import (
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.auxiliary_optimizers import (
    DEFAULT_AUXILIARY_OPTIMIZER_RULES,
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
    resolve_auxiliary_optimizer_rules,
    validate_auxiliary_optimizers,
)
from worldfoundry.training.recipes.post_training.algorithms.ddrl import DDRLAlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.dfd import DFDAlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.diagonal import DiagonalAlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.diffusion_dpo import (
    DiffusionDPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.dmd import DMDAlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.dmd2 import DMD2AlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.rcm import (
    CausalRCMAlgorithmSpec,
    RCMAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.reward_forcing import (
    RewardForcingAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.scale_wise import (
    ScaleWiseAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.scm_ladd import SCMLADDAlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.self_forcing import (
    SelfForcingAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.self_gradient_forcing import (
    SelfGradientForcingAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.senseflow import (
    SenseFlowAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.sgmd import SGMDAlgorithmSpec
from worldfoundry.training.recipes.post_training.algorithms.sid import SIDAlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.post_training.rewards import (
    VIDEOALIGN_REWARD_IDS,
    VideoAlignRewardSpec,
)

_PRESENT = object()


def _validate(algorithm: object, *present: str) -> None:
    values = {name: _PRESENT for name in present}
    validate_auxiliary_optimizers(
        algorithm,
        fake_score_optimizer=values.get("fake_score_optimizer"),
        guidance_optimizer=values.get("guidance_optimizer"),
        discriminator_optimizer=values.get("discriminator_optimizer"),
    )


def _dmd() -> DMDAlgorithmSpec:
    return DMDAlgorithmSpec(
        student_timesteps=(1000.0, 500.0),
        student_sigmas=(1.0, 0.5),
        real_score_checkpoint="teacher",
        fake_score_checkpoint="critic",
    )


def _adaptive_video() -> AdaptiveVideoAlgorithmSpec:
    return AdaptiveVideoAlgorithmSpec(
        student_timesteps=(1000.0, 500.0),
        student_sigmas=(1.0, 0.5),
        real_score_checkpoint="teacher",
        fake_score_checkpoint="critic",
    )


def _anyflow_on_policy() -> AnyFlowFAROnPolicyAlgorithmSpec:
    return AnyFlowFAROnPolicyAlgorithmSpec(
        real_score_checkpoint="real",
        fake_score_checkpoint="fake",
    )


def _adversarial_diffusion() -> AdversarialDiffusionAlgorithmSpec:
    return AdversarialDiffusionAlgorithmSpec(
        teacher_checkpoint="teacher",
        decoder_checkpoint="decoder",
        feature_checkpoint="features",
        student_alpha_cumprods=(0.9, 0.6, 0.4, 0.2, 0.0),
        teacher_alpha_cumprods=(0.9, 0.5, 0.1),
        student_timesteps=(0, 1, 2, 4),
        teacher_timestep_min=0,
        teacher_timestep_max=2,
        feature_resolutions=(8,),
        feature_layers=("blocks.0",),
    )


def _senseflow() -> SenseFlowAlgorithmSpec:
    return SenseFlowAlgorithmSpec.sd35_large_released(
        teacher_checkpoint="teacher",
        fake_score_checkpoint="critic",
        discriminator_checkpoint="disc",
    )


def _dmd2() -> DMD2AlgorithmSpec:
    return DMD2AlgorithmSpec(
        student_timesteps=(1000.0, 500.0),
        student_sigmas=(1.0, 0.5),
        real_score_checkpoint="teacher",
        guidance_checkpoint="guidance",
        normalization_axes=(1, 2, 3),
    )


def _dfd_gan() -> DFDAlgorithmSpec:
    return DFDAlgorithmSpec(
        teacher_checkpoint="teacher",
        fake_score_checkpoint="critic",
        discriminator_checkpoint="disc",
    )


def _dfd_plain() -> DFDAlgorithmSpec:
    return DFDAlgorithmSpec(
        teacher_checkpoint="teacher",
        fake_score_checkpoint="critic",
        generator_adversarial_weight=0.0,
        discriminator_weight=0.0,
    )


def _diagonal() -> DiagonalAlgorithmSpec:
    return DiagonalAlgorithmSpec(
        real_score_checkpoint="real",
        fake_score_checkpoint="fake",
        fixed_teacher_checkpoint="fixed",
    )


def _scale_wise_mmd_only() -> ScaleWiseAlgorithmSpec:
    return ScaleWiseAlgorithmSpec(
        teacher_checkpoint="teacher",
        fake_score_checkpoint="critic",
        dmd_enabled=False,
        gan_enabled=False,
        mmd_enabled=True,
        fake_updates_per_iteration=0,
    )


def _ddrl() -> DDRLAlgorithmSpec:
    return DDRLAlgorithmSpec(
        train_on=(0,),
        reward_weights={reward_id: 1.0 for reward_id in VIDEOALIGN_REWARD_IDS},
        reward_model=VideoAlignRewardSpec(),
    )


CASES = [
    # (case id, spec factory, provided optimizers, expected message or None)
    ("anyflow-on-policy-missing", _anyflow_on_policy, (), "AnyFlow on-policy training requires fake_score_optimizer"),
    (
        "anyflow-on-policy-extra",
        _anyflow_on_policy,
        ("fake_score_optimizer", "guidance_optimizer"),
        "AnyFlow on-policy training only accepts fake_score_optimizer",
    ),
    ("anyflow-on-policy-ok", _anyflow_on_policy, ("fake_score_optimizer",), None),
    (
        "anyflow-pretrain-extra",
        AnyFlowBidirectionalPretrainAlgorithmSpec,
        ("fake_score_optimizer",),
        "AnyFlow pretraining accepts only the primary optimizer",
    ),
    ("anyflow-pretrain-ok", AnyFlowBidirectionalPretrainAlgorithmSpec, (), None),
    (
        "add-missing",
        _adversarial_diffusion,
        (),
        "adversarial diffusion distillation requires discriminator_optimizer",
    ),
    (
        "add-extra",
        _adversarial_diffusion,
        ("discriminator_optimizer", "fake_score_optimizer"),
        "adversarial diffusion distillation only accepts discriminator_optimizer",
    ),
    ("add-ok", _adversarial_diffusion, ("discriminator_optimizer",), None),
    ("reward-forcing-missing", RewardForcingAlgorithmSpec, (), "Reward-Forcing requires fake_score_optimizer"),
    (
        "reward-forcing-extra",
        RewardForcingAlgorithmSpec,
        ("fake_score_optimizer", "discriminator_optimizer"),
        "Reward-Forcing only accepts fake_score_optimizer",
    ),
    ("reward-forcing-ok", RewardForcingAlgorithmSpec, ("fake_score_optimizer",), None),
    ("senseflow-missing-fake", _senseflow, (), "SenseFlow requires fake_score_optimizer"),
    (
        "senseflow-missing-disc",
        _senseflow,
        ("fake_score_optimizer",),
        "SenseFlow requires discriminator_optimizer",
    ),
    (
        "senseflow-guidance",
        _senseflow,
        ("fake_score_optimizer", "discriminator_optimizer", "guidance_optimizer"),
        "SenseFlow does not accept guidance_optimizer",
    ),
    ("senseflow-ok", _senseflow, ("fake_score_optimizer", "discriminator_optimizer"), None),
    ("dmd-missing", _dmd, (), "DMD requires fake_score_optimizer"),
    (
        "dmd-extra",
        _dmd,
        ("fake_score_optimizer", "guidance_optimizer"),
        "dmd only accepts fake_score_optimizer",
    ),
    ("dmd-ok", _dmd, ("fake_score_optimizer",), None),
    ("adaptive-missing", _adaptive_video, (), "adaptive video distillation requires fake_score_optimizer"),
    (
        "adaptive-extra",
        _adaptive_video,
        ("fake_score_optimizer", "discriminator_optimizer"),
        "adaptive-video-distillation only accepts fake_score_optimizer",
    ),
    ("rcm-dmd-missing", RCMAlgorithmSpec, (), "rCM DMD requires fake_score_optimizer"),
    (
        "rcm-no-dmd-extra",
        lambda: RCMAlgorithmSpec(dmd_loss_scale=0.0, fake_score_checkpoint=None),
        ("fake_score_optimizer",),
        "rCM without DMD cannot configure fake_score_optimizer",
    ),
    (
        "rcm-guidance",
        RCMAlgorithmSpec,
        ("fake_score_optimizer", "guidance_optimizer"),
        "rCM only accepts fake_score_optimizer when DMD is enabled",
    ),
    ("rcm-ok", RCMAlgorithmSpec, ("fake_score_optimizer",), None),
    ("causal-rcm-missing", CausalRCMAlgorithmSpec, (), "rCM DMD requires fake_score_optimizer"),
    ("dmd2-missing", _dmd2, (), "DMD2 requires guidance_optimizer"),
    (
        "dmd2-extra",
        _dmd2,
        ("guidance_optimizer", "fake_score_optimizer"),
        "DMD2 only accepts guidance_optimizer",
    ),
    ("dmd2-ok", _dmd2, ("guidance_optimizer",), None),
    ("dfd-missing", _dfd_plain, (), "DFD requires fake_score_optimizer"),
    (
        "dfd-gan-missing-disc",
        _dfd_gan,
        ("fake_score_optimizer",),
        "DFD GAN loss requires discriminator_optimizer",
    ),
    (
        "dfd-plain-disc",
        _dfd_plain,
        ("fake_score_optimizer", "discriminator_optimizer"),
        "DFD without GAN loss cannot configure discriminator_optimizer",
    ),
    (
        "dfd-guidance",
        _dfd_plain,
        ("fake_score_optimizer", "guidance_optimizer"),
        "DFD does not accept guidance_optimizer",
    ),
    ("dfd-gan-ok", _dfd_gan, ("fake_score_optimizer", "discriminator_optimizer"), None),
    ("dfd-plain-ok", _dfd_plain, ("fake_score_optimizer",), None),
    ("diagonal-missing", _diagonal, (), "diagonal distillation requires fake_score_optimizer"),
    (
        "diagonal-extra",
        _diagonal,
        ("fake_score_optimizer", "guidance_optimizer"),
        "diagonal distillation only accepts fake_score_optimizer",
    ),
    ("scm-ladd-missing", SCMLADDAlgorithmSpec, (), "SCM-LADD requires discriminator_optimizer"),
    (
        "scm-ladd-extra",
        SCMLADDAlgorithmSpec,
        ("discriminator_optimizer", "guidance_optimizer"),
        "SCM-LADD only accepts discriminator_optimizer",
    ),
    ("scm-ladd-ok", SCMLADDAlgorithmSpec, ("discriminator_optimizer",), None),
    (
        "scale-wise-dmd-missing",
        lambda: ScaleWiseAlgorithmSpec(teacher_checkpoint="t", fake_score_checkpoint="f"),
        (),
        "scale-wise DMD requires fake_score_optimizer",
    ),
    (
        "scale-wise-mmd-only-extra",
        _scale_wise_mmd_only,
        ("fake_score_optimizer",),
        "MMD-only scale-wise training cannot configure fake_score_optimizer",
    ),
    (
        "scale-wise-guidance",
        lambda: ScaleWiseAlgorithmSpec(teacher_checkpoint="t", fake_score_checkpoint="f"),
        ("fake_score_optimizer", "guidance_optimizer"),
        "scale-wise distillation only accepts fake_score_optimizer",
    ),
    ("scale-wise-mmd-only-ok", _scale_wise_mmd_only, (), None),
    (
        "self-forcing-missing",
        lambda: SelfForcingAlgorithmSpec(denoising_timesteps=(1000.0, 500.0)),
        (),
        "self-forcing DMD requires fake_score_optimizer",
    ),
    (
        "self-forcing-extra",
        lambda: SelfForcingAlgorithmSpec(denoising_timesteps=(1000.0, 500.0)),
        ("fake_score_optimizer", "discriminator_optimizer"),
        "self-forcing only accepts fake_score_optimizer",
    ),
    (
        "self-gradient-forcing-missing",
        lambda: SelfGradientForcingAlgorithmSpec(real_score_checkpoint="r", fake_score_checkpoint="f"),
        (),
        "self-gradient-forcing DMD requires fake_score_optimizer",
    ),
    (
        "sid-missing",
        lambda: SIDAlgorithmSpec(
            student_timesteps=(1000.0, 500.0),
            student_sigmas=(1.0, 0.5),
            teacher_checkpoint="t",
            fake_score_checkpoint="f",
            alpha=1.2,
        ),
        (),
        "SiD requires fake_score_optimizer",
    ),
    (
        "sid-extra",
        lambda: SIDAlgorithmSpec(
            student_timesteps=(1000.0, 500.0),
            student_sigmas=(1.0, 0.5),
            teacher_checkpoint="t",
            fake_score_checkpoint="f",
            alpha=1.2,
        ),
        ("fake_score_optimizer", "guidance_optimizer"),
        "SiD only accepts fake_score_optimizer",
    ),
    (
        "sgmd-missing",
        lambda: SGMDAlgorithmSpec(
            student_timesteps=(1000.0, 500.0),
            teacher_checkpoint="t",
            fake_score_checkpoint="f",
        ),
        (),
        "SGMD requires fake_score_optimizer",
    ),
    ("ddrl-fake", _ddrl, ("fake_score_optimizer",), "DDRL cannot configure fake_score_optimizer"),
    ("ddrl-guidance", _ddrl, ("guidance_optimizer",), "DDRL cannot configure guidance_optimizer"),
    (
        "ddrl-disc",
        _ddrl,
        ("discriminator_optimizer",),
        "DDRL cannot configure discriminator_optimizer",
    ),
    ("ddrl-ok", _ddrl, (), None),
    (
        "default-fake",
        lambda: DiffusionDPOAlgorithmSpec(beta=1000.0),
        ("fake_score_optimizer", "guidance_optimizer"),
        "this algorithm cannot configure fake_score_optimizer",
    ),
    (
        "default-guidance",
        lambda: DiffusionDPOAlgorithmSpec(beta=1000.0),
        ("guidance_optimizer",),
        "this algorithm cannot configure guidance_optimizer",
    ),
    (
        "default-disc",
        lambda: DiffusionDPOAlgorithmSpec(beta=1000.0),
        ("discriminator_optimizer",),
        "this algorithm cannot configure discriminator_optimizer",
    ),
    ("default-ok", lambda: DiffusionDPOAlgorithmSpec(beta=1000.0), (), None),
]


@pytest.mark.parametrize(
    ("factory", "present", "message"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_legacy_chain_messages_are_preserved(factory, present, message) -> None:
    algorithm = factory()
    if message is None:
        _validate(algorithm, *present)
    else:
        with pytest.raises(ValueError) as excinfo:
            _validate(algorithm, *present)
        assert str(excinfo.value) == message


def test_specs_without_declaration_fall_back_to_reject_all() -> None:
    assert (
        resolve_auxiliary_optimizer_rules(object())
        is DEFAULT_AUXILIARY_OPTIMIZER_RULES
    )
    assert (
        resolve_auxiliary_optimizer_rules(DiffusionDPOAlgorithmSpec(beta=1000.0))
        is DEFAULT_AUXILIARY_OPTIMIZER_RULES
    )


def test_rule_constructor_rejects_bad_declarations() -> None:
    with pytest.raises(ValueError, match="unknown auxiliary optimizers"):
        requires_auxiliary("main_optimizer", "message")
    with pytest.raises(ValueError, match="must name at least one optimizer"):
        AuxiliaryOptimizerRule(optimizers=(), required=False, message="message")
    with pytest.raises(ValueError, match="must be non-empty"):
        forbids_auxiliary("guidance_optimizer", message="   ")


def _recipe_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run": {"id": "tr10", "output_dir": "runs/tr10"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
        "tuning": {"mode": "lora", "preset": "wan-attention", "rank": 8, "alpha": 16},
        "data": {"manifest": "data/latents.jsonl"},
        "algorithm": {
            "type": "dmd",
            "student_timesteps": [1000, 500],
            "student_sigmas": [1.0, 0.5],
            "real_score_checkpoint": "teacher",
            "fake_score_checkpoint": "critic",
        },
        "optimizer": {"type": "adamw", "learning_rate": 2.0e-6},
        "fake_score_optimizer": {"type": "adamw", "learning_rate": 2.0e-6},
    }
    payload.update(overrides)
    return payload


def test_recipe_end_to_end_accepts_and_rejects_auxiliary_optimizers() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    assert recipe.fake_score_optimizer is not None

    without_fake = _recipe_mapping()
    del without_fake["fake_score_optimizer"]
    with pytest.raises(ValueError, match="DMD requires fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(without_fake)

    with_guidance = _recipe_mapping(
        guidance_optimizer={"type": "adamw", "learning_rate": 2.0e-6},
    )
    with pytest.raises(ValueError, match="dmd only accepts fake_score_optimizer"):
        PostTrainingRecipe.from_mapping(with_guidance)

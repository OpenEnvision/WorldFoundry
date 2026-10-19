# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""DreamDojo 2B GR-1 action-conditioned composition of canonical Cosmos roles."""

from __future__ import annotations

from dataclasses import replace

from ..components import ComponentKind, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.denoisers.dreamdojo import build_dreamdojo_denoiser
from ..models.encoders.dreamdojo import build_dreamdojo_conditioner
from .cosmos2p5 import cosmos25_2b_recipe
from .spec import NativeDiffusionRecipe


def dreamdojo_recipe() -> NativeDiffusionRecipe:
    """Bind action AdaLN, Reason1, Wan video codec, and positive-CFG UniPC."""
    base = cosmos25_2b_recipe()
    checkpoints = dict(base.checkpoints)
    checkpoints.pop("transformer-pretrained", None)
    checkpoints["transformer"] = CheckpointSpec(
        repo_id="nvidia/DreamDojo",
        revision="89d029e10816d2995d700cb8ba06f171e0504203",
        files=("2B_GR1_post-train/iter_000050000/model_ema_bf16.pt",),
    )
    components = []
    for component in base.components:
        if component.key.kind is ComponentKind.DENOISER:
            component = replace(component, factory=build_dreamdojo_denoiser)
        elif component.key.kind is ComponentKind.CONDITIONER:
            component = replace(component, factory=build_dreamdojo_conditioner)
        elif component.key.kind is ComponentKind.LATENT_INITIALIZER:
            component = replace(component, options={**component.options, "minimum_frames": 13})
        elif component.key.kind is ComponentKind.SCHEDULER:
            component = replace(
                component, options={**component.options, "use_karras_sigma": False, "karras_steps_are_intervals": False}
            )
        components.append(component)
    return NativeDiffusionRecipe(
        model_id="dreamdojo",
        aliases=("nvidia/DreamDojo",),
        components=tuple(components),
        execution=base.execution,
        checkpoints=checkpoints,
        capabilities=frozenset({"action-conditioned-video2world", "robot-world-model"}),
        options=base.options,
        metadata={
            "native_inference": True,
            "output_layout": "BCTHW",
            "upstream_revision": "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff",
            "action_dimension": 384,
        },
    )


def dreamdojo_student_recipe() -> NativeDiffusionRecipe:
    """Bind the causal student, fixed TrigFlow schedule, and shared Reason1/Wan.

    Student weights are operator-supplied. The pinned public Hub snapshot
    contains post-trained teachers, not the self-forcing student checkpoint.
    """
    from worldfoundry.core.io.paths import hfd_root_path

    from ..models.autoencoders.cosmos2p5.student import build_dreamdojo_student_codec
    from ..models.denoisers.dreamdojo_student import build_dreamdojo_student_denoiser
    from ..schedulers.trigflow import build_trigflow_scheduler

    base = dreamdojo_recipe()
    factories = {
        ComponentKind.DENOISER: build_dreamdojo_student_denoiser,
        ComponentKind.LATENT_INITIALIZER: build_dreamdojo_student_codec,
        ComponentKind.SCHEDULER: build_trigflow_scheduler,
    }
    checkpoints = dict(base.checkpoints)
    checkpoints["transformer"] = CheckpointSpec(
        source=str(hfd_root_path() / "nvidia--DreamDojo" / "student" / "model_ema_bf16.pt"),
        metadata={"operator_supplied_student": True},
    )
    return replace(
        base,
        model_id="dreamdojo-student",
        aliases=(),
        checkpoints=checkpoints,
        components=tuple(
            replace(component, factory=factories[component.key.kind], options={})
            if component.key.kind in factories
            else component
            for component in base.components
        ),
        execution=ExecutionSpec(strategy="causal-trigflow", bindings=base.execution.bindings),
        capabilities=frozenset({"action-conditioned-video2world", "robot-world-model", "realtime-student"}),
        metadata={**base.metadata, "variant": "self-forcing-student", "student_checkpoint_required": True},
    )

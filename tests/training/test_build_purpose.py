from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler  # noqa: E402
from worldfoundry.base_models.diffusion_model.components import (  # noqa: E402
    BuildPurpose,
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
    ComponentSpec,
    ExecutionSpec,
)
from worldfoundry.base_models.diffusion_model.optimizations import (  # noqa: E402
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
)
from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe  # noqa: E402


class _Denoiser:
    def __call__(self, model_input):
        return model_input


class _Codec:
    def encode(self, images):
        return images


def test_component_build_context_defaults_to_inference() -> None:
    context = ComponentBuildContext(
        model_id="tiny",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(options={"teacache": True}),
    )

    assert context.purpose is BuildPurpose.INFERENCE


def test_training_build_rejects_inference_only_policy() -> None:
    with pytest.raises(ValueError, match="offload=block"):
        ComponentBuildContext(
            model_id="tiny",
            key=ComponentKey(ComponentKind.DENOISER),
            policy=RuntimePolicy(offload=OffloadPolicy(mode=OffloadMode.BLOCK)),
            purpose=BuildPurpose.TRAINING,
        )

    with pytest.raises(ValueError, match="teacache"):
        ComponentBuildContext(
            model_id="tiny",
            key=ComponentKey(ComponentKind.DENOISER),
            policy=RuntimePolicy(options={"teacache": True}),
            purpose=BuildPurpose.TRAINING,
        )


def test_training_assembler_can_materialize_only_requested_components() -> None:
    built: list[ComponentKey] = []
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")

    def denoiser_factory(context):
        built.append(context.key)
        return _Denoiser()

    def codec_factory(context):
        built.append(context.key)
        return _Codec()

    recipe = NativeDiffusionRecipe(
        model_id="tiny-training",
        components=(
            ComponentSpec(denoiser_key, denoiser_factory),
            ComponentSpec(codec_key, codec_factory),
        ),
        execution=ExecutionSpec(bindings={"denoiser": denoiser_key}),
    )

    components = NativeDiffusionAssembler().build_components(
        recipe,
        purpose=BuildPurpose.TRAINING,
        component_keys=(denoiser_key,),
    )

    assert tuple(components) == (denoiser_key,)
    assert built == [denoiser_key]

    with pytest.raises(KeyError, match="unknown component keys"):
        NativeDiffusionAssembler().build_components(
            recipe,
            purpose=BuildPurpose.TRAINING,
            component_keys=(ComponentKey(ComponentKind.SCHEDULER),),
        )

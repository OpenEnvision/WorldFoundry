# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_load_allegro_components_returns_the_owned_classes() -> None:
    from worldfoundry.synthesis.visual_generation.allegro.allegro_runtime import load_allegro_components
    from worldfoundry.synthesis.visual_generation.allegro.allegro_runtime.allegro.models.transformers.transformer_3d_allegro_ti2v import (
        AllegroTransformerTI2V3DModel,
    )
    from worldfoundry.synthesis.visual_generation.allegro.allegro_runtime.allegro.models.vae.vae_allegro import (
        AllegroAutoencoderKL3D,
    )
    from worldfoundry.synthesis.visual_generation.allegro.allegro_runtime.allegro.pipelines.data_process import (
        CenterCropResizeVideo,
        ToTensorVideo,
    )
    from worldfoundry.synthesis.visual_generation.allegro.allegro_runtime.allegro.pipelines.pipeline_allegro_ti2v import (
        AllegroTI2VPipeline,
    )

    components = load_allegro_components()
    assert components.transformer_cls is AllegroTransformerTI2V3DModel
    assert components.autoencoder_cls is AllegroAutoencoderKL3D
    assert components.pipeline_cls is AllegroTI2VPipeline
    assert components.to_tensor_video_cls is ToTensorVideo
    assert components.center_crop_resize_video_cls is CenterCropResizeVideo


def test_load_allegro_components_preserves_import_paths_and_unrelated_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    from worldfoundry.synthesis.visual_generation.allegro.allegro_runtime import load_allegro_components

    unrelated = ModuleType("allegro")
    monkeypatch.setitem(sys.modules, "allegro", unrelated)
    paths = list(sys.path)
    components = load_allegro_components()
    assert components.pipeline_cls.__module__.startswith("worldfoundry.synthesis.visual_generation.allegro.")
    assert sys.path == paths
    assert sys.modules["allegro"] is unrelated

from __future__ import annotations

from dataclasses import dataclass

ALLEGRO_RUNTIME_UNAVAILABLE = (
    "Allegro source code is not available from the in-tree runtime package. "
    "WorldFoundry expects the upstream Allegro inference modules vendored under "
    "worldfoundry.synthesis.visual_generation.allegro.allegro_runtime/allegro."
)


@dataclass(frozen=True)
class AllegroComponents:
    transformer_cls: object
    autoencoder_cls: object
    to_tensor_video_cls: object
    center_crop_resize_video_cls: object
    pipeline_cls: object


def load_allegro_components() -> AllegroComponents:
    """Load the installed TI2V components; checkpoints remain external assets."""
    try:
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
    except ImportError as exc:
        raise RuntimeError(ALLEGRO_RUNTIME_UNAVAILABLE) from exc

    return AllegroComponents(
        transformer_cls=AllegroTransformerTI2V3DModel,
        autoencoder_cls=AllegroAutoencoderKL3D,
        to_tensor_video_cls=ToTensorVideo,
        center_crop_resize_video_cls=CenterCropResizeVideo,
        pipeline_cls=AllegroTI2VPipeline,
    )

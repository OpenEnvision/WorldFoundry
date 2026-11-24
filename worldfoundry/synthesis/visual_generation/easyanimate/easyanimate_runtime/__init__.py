from __future__ import annotations

from dataclasses import dataclass

EASYANIMATE_RUNTIME_UNAVAILABLE = (
    "EasyAnimate source code is not available from the in-tree runtime package. "
    "WorldFoundry expects the upstream inference modules vendored under "
    "worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime/easyanimate."
)


@dataclass(frozen=True)
class EasyAnimateComponents:
    name_to_autoencoder_magvit: object
    name_to_transformer3d: object
    inpaint_pipeline_cls: object
    multi_text_encoder_inpaint_pipeline_cls: object
    convert_weight_dtype_wrapper: object
    get_image_to_video_latent: object


def load_easyanimate_components() -> EasyAnimateComponents:
    """Load the installed EasyAnimate components; checkpoints remain external assets."""
    try:
        from worldfoundry.base_models.diffusion_model.models.initializers.easyanimate_conditioning import (
            get_image_to_video_latent,
        )
        from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.models import (
            name_to_autoencoder_magvit,
            name_to_transformer3d,
        )
        from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.pipeline.pipeline_easyanimate_inpaint import (
            EasyAnimateInpaintPipeline,
        )
        from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.utils.fp8_optimization import (
            convert_weight_dtype_wrapper,
        )
    except ImportError as exc:
        raise RuntimeError(EASYANIMATE_RUNTIME_UNAVAILABLE) from exc

    return EasyAnimateComponents(
        name_to_autoencoder_magvit=name_to_autoencoder_magvit,
        name_to_transformer3d=name_to_transformer3d,
        inpaint_pipeline_cls=EasyAnimateInpaintPipeline,
        multi_text_encoder_inpaint_pipeline_cls=EasyAnimateInpaintPipeline,
        convert_weight_dtype_wrapper=convert_weight_dtype_wrapper,
        get_image_to_video_latent=get_image_to_video_latent,
    )

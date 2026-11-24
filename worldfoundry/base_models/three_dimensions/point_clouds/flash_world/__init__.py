__all__ = [
    "AutoencoderKLWan",
    "WANDecoderPixelAligned3DGSReconstructionModel",
    "WanTransformer3DModel",
]


def __getattr__(name):
    if name == "AutoencoderKLWan":
        from .autoencoder_kl_wan import AutoencoderKLWan

        return AutoencoderKLWan
    if name == "WanTransformer3DModel":
        from .transformer_wan import WanTransformer3DModel

        return WanTransformer3DModel
    if name == "WANDecoderPixelAligned3DGSReconstructionModel":
        from .reconstruction_model import WANDecoderPixelAligned3DGSReconstructionModel

        return WANDecoderPixelAligned3DGSReconstructionModel
    raise AttributeError(name)

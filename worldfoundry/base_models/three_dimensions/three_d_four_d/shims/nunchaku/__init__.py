class NunchakuFluxTransformer2dModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError(
            "nunchaku is not installed in the WorldFoundry unified environment. "
            "Run WorldGen with low_vram=False on A100 80GB GPUs, or install a "
            "nunchaku wheel compatible with this PyTorch/CUDA stack."
        )

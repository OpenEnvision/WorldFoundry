"""Inference-only Hy-Nomas component graph; no external runtime or training package."""

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.pixelumm_component import build_pixelumm_decoder
from ..models.denoisers.hy_nomas import build_hy_nomas_denoiser
from ..models.encoders.hy_nomas import build_hy_nomas_conditioner
from ..models.initializers.hy_nomas import build_hy_nomas_initializer
from ..schedulers.wan import build_wan_flow_match_euler_scheduler
from .spec import NativeDiffusionRecipe

REVISION = "bac719a80c8d02f91c485fca804bb8221e149b70"


def _recipe(*, low_memory: bool) -> NativeDiffusionRecipe:
    roles = {
        "denoiser": (ComponentKind.DENOISER, build_hy_nomas_denoiser, "transformer"),
        "conditioner": (ComponentKind.CONDITIONER, build_hy_nomas_conditioner, "text-encoder"),
        "latent_initializer": (ComponentKind.LATENT_INITIALIZER, build_hy_nomas_initializer, None),
        "scheduler": (ComponentKind.SCHEDULER, build_wan_flow_match_euler_scheduler, None),
        "decoder": (ComponentKind.DECODER, build_pixelumm_decoder, "vae"),
    }
    bindings = {name: ComponentKey(kind) for name, (kind, _, _) in roles.items()}
    return NativeDiffusionRecipe(
        model_id="hy-nomas-low-memory" if low_memory else "hy-nomas",
        aliases=() if low_memory else ("hynomas", "tencent/Hy-Nomas"),
        components=tuple(
            ComponentSpec(
                key=bindings[name],
                factory=factory,
                checkpoints={} if checkpoint is None else {"weights": checkpoint},
                options={"low_memory": low_memory, **({"shift": 5.0} if name == "scheduler" else {})},
            )
            for name, (_, factory, checkpoint) in roles.items()
        ),
        execution=ExecutionSpec(strategy="segment-autoregressive", bindings=bindings),
        checkpoints={
            "transformer": CheckpointSpec(
                repo_id="tencent/Hy-Nomas",
                revision=REVISION,
                files=("DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",),
            ),
            "vae": CheckpointSpec(
                repo_id="tencent/Hy-Nomas",
                revision=REVISION,
                files=("VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors",),
            ),
            "text-encoder": CheckpointSpec(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
                files=tuple(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
                allow_patterns=("*.json", "*.safetensors", "*.jinja", "*.txt"),
            ),
        },
        capabilities=frozenset({"text-to-video", "camera-control", "segment-autoregressive"}),
        metadata={"source_revision": "911cea6458a89c2663513ff7099561ec190dd491"},
    )


def hy_nomas_recipe() -> NativeDiffusionRecipe:
    return _recipe(low_memory=False)


def hy_nomas_low_memory_recipe() -> NativeDiffusionRecipe:
    return _recipe(low_memory=True)

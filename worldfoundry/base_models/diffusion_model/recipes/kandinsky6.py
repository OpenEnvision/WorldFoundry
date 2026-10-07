"""Kandinsky-6 Lite/Pro native audio-video recipes."""

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.denoisers.kandinsky6 import build_kandinsky6_denoiser
from ..models.encoders.kandinsky6.component import build_kandinsky6_conditioner
from ..models.autoencoders.kandinsky6.component import build_kandinsky6_decoder
from .spec import NativeDiffusionRecipe

CHECKPOINT_NAMES = ("pro-distill", "pro", "pro-pretrain", "lite", "lite-distill", "lite-pretrain")


CHECKPOINT_REVISIONS = {
    "pro-distill": "4a852c02855395e2757f852e9b1d2bf537cec517",
    "pro": "e8e4a00259946892631a33a899237edf50c7232f",
    "pro-pretrain": "fd9a3fcd1967acc20a1911444b733405f51b6530",
    "lite": "50029a58a512c694d5995bdde9f5cae955e4655b",
    "lite-distill": "e056a84f34d9617e86b3ae47c792fe626cbe53f8",
    "lite-pretrain": "ae9378bb1d3cd382aa96ce36cc2df59fe7d28556",
}
TEXT_ENCODER_FILES = tuple(f"text_encoder/model-{index:05d}-of-00005.safetensors" for index in range(1, 6))


def kandinsky6_recipe(name="pro-distill"):
    if name not in CHECKPOINT_NAMES:
        raise ValueError(f"Unknown Kandinsky-6 checkpoint: {name}")
    repo = "kandinskylab/Kandinsky-6.0-" + name.replace("pro", "Pro", 1).replace("lite", "Lite", 1) + "-5s-Diffusers"
    factories = {
        "denoiser": build_kandinsky6_denoiser,
        "conditioner": build_kandinsky6_conditioner,
        "decoder": build_kandinsky6_decoder,
    }
    files = {
        "denoiser": ("transformer/diffusion_pytorch_model.safetensors",),
        "conditioner": (
            "text_encoder/config.json",
            "text_encoder_2/model.safetensors",
            *TEXT_ENCODER_FILES,
            "text_encoder/model.safetensors.index.json",
            "text_encoder_2/config.json",
        ),
        "decoder": (
            "vae/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
            "audio_vae/config.json",
            "vocoder/diffusion_pytorch_model.safetensors",
            "vocoder/config.json",
        ),
    }
    patterns = {
        "denoiser": ("transformer/**",),
        "conditioner": ("text_encoder/**", "text_encoder_2/**", "tokenizer/**", "tokenizer_2/**", "processor/**"),
        "decoder": ("vae/**", "audio_vae/**", "vocoder/**"),
    }
    bindings = {role: ComponentKey(ComponentKind(role)) for role in factories}
    return NativeDiffusionRecipe(
        model_id="kandinsky6-" + name,
        aliases=("kandinsky6", "kandinsky-6", repo) if name == "pro-distill" else (repo,),
        components=tuple(
            ComponentSpec(bindings[role], factory, {"weights": role}) for role, factory in factories.items()
        ),
        execution=ExecutionSpec(strategy="kandinsky6-joint", bindings=bindings),
        checkpoints={
            role: CheckpointSpec(
                repo_id=repo, revision=CHECKPOINT_REVISIONS[name], files=paths, allow_patterns=patterns[role]
            )
            for role, paths in files.items()
        },
        capabilities=frozenset({"text-to-video", "image-to-video", "audio-video-generation"}),
        options={"checkpoint_name": name},
        metadata={"native_inference": True, "output_layout": "BCTHW"},
    )

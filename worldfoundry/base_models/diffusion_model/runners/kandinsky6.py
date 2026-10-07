"""Framework-owned Kandinsky-6 joint audio/video execution strategy."""

import torch

from ..contracts import DiffusionOutput


class Kandinsky6Runner:
    def __init__(self, *, model_id, sampler, components):
        self.model_id = model_id
        self.sampler = sampler
        self.components = components
        self.device = sampler.device
        self.dtype = torch.bfloat16

    def run(self, request):
        options = dict(
            text=list(request.prompts),
            negative_text=list(request.negative_prompts) if request.negative_prompts else "",
            height=request.height,
            width=request.width,
            latent_frames=(request.num_frames - 1) // 4 + 1,
            num_steps=request.sampling.num_inference_steps,
            guidance_weight=request.sampling.guidance_scale,
            seed=request.sampling.seed,
            image=request.inputs.get("image"),
            sample_audio=bool(request.inputs.get("sample_audio", True)),
            save_path=request.inputs.get("save_path"),
        )
        result = self.sampler(**options)
        return DiffusionOutput(
            sample=result.frames,
            latents=result.latents.video,
            artifacts={"audio": result.audio, "path": result.path, "prompts": result.prompts},
            metadata={"model_id": self.model_id, "seed": request.sampling.seed, "fps": 24, "audio_sample_rate": 44100},
        )


def build_kandinsky6_strategy(context):
    from ..optimizations.kandinsky6.offload import build_offload
    from ..recipes.kandinsky6_config import _CONFIGS, load_config
    from .kandinsky6_sampling import Kandinsky6Sampler

    if context.extensions:
        raise ValueError("Kandinsky-6 joint sampling does not support diffusion extensions")
    cfg = load_config(_CONFIGS / "checkpoints" / (context.recipe.options["checkpoint_name"] + ".yaml"))
    roles = {role: context.components[key] for role, key in context.recipe.execution.bindings.items()}
    codecs = roles["decoder"]
    modules = {
        "dit": roles["denoiser"],
        "text_embedder": roles["conditioner"].model,
        "vae": codecs.video,
        "audio_vae": codecs.audio,
        "vocoder": codecs.vocoder,
    }
    mode = context.policy.offload.mode.value
    mode = {"component": "module", "cpu": "module"}.get(mode, mode)
    offload = build_offload(mode, context.policy.device)
    for name, module in modules.items():
        offload.register(name, module)
    gen = cfg.generation
    sampler = Kandinsky6Sampler(
        **modules,
        device=context.policy.device,
        offload=offload,
        num_steps=gen.num_steps,
        guidance_weight=gen.guidance_weight,
        scheduler_scale=gen.scheduler_scale,
        scale_factor=gen.scale_factor,
        height=gen.height,
        width=gen.width,
        latent_frames=gen.latent_frames,
        sample_frames=gen.sample_frames,
        visual_cond_scheme=gen.visual_cond_scheme,
        max_area=gen.max_area,
        image_divisibility=gen.image_divisibility,
        piflow_conf=cfg.piflow,
    )
    return Kandinsky6Runner(model_id=context.recipe.model_id, sampler=sampler, components=context.components)

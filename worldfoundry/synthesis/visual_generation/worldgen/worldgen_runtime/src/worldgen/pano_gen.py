import os
import torch
import tempfile
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from huggingface_hub import hf_hub_download
from .models.flux_pano_gen_pipeline import FluxPipeline
from .models.flux_pano_fill_pipeline import FluxFillPipeline
from .utils.lora_utils import load_and_fix_lora


CKPT_ROOT = Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/worldfoundry/checkpoints")).expanduser()


@contextmanager
def _compatible_peft_lora_dispatch():
    """Skip PEFT's torchao dispatcher when an older, unused torchao is installed."""
    try:
        from packaging.version import Version

        torchao_version = Version(version("torchao"))
    except (PackageNotFoundError, ValueError):
        yield
        return

    if torchao_version >= Version("0.16.0"):
        yield
        return

    # PEFT >= 0.20 raises for torchao < 0.16 even on ordinary nn.Linear
    # layers. FLUX.1-dev uses those layers, so its LoRA needs no torchao path.
    peft_torchao = import_module("peft.tuners.lora.torchao")
    original = peft_torchao.is_torchao_available
    peft_torchao.is_torchao_available = lambda: False
    try:
        yield
    finally:
        peft_torchao.is_torchao_available = original


def _load_standard_lora(pipe, lora_path: str) -> None:
    with _compatible_peft_lora_dispatch():
        # Diffusers requires an explicit weight name for local files in offline
        # mode, even though its loader later recognizes the full file path.
        pipe.load_lora_weights(lora_path, weight_name=Path(lora_path).name)


def _first_existing(*candidates):
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _worldgen_lora(filename: str) -> str:
    local_path = _first_existing(
        CKPT_ROOT / "WorldGen" / "models--WorldGen-Flux-Lora" / filename,
        CKPT_ROOT / "LeoXie--WorldGen" / "models--WorldGen-Flux-Lora" / filename,
        CKPT_ROOT / "hfd" / "custom--WorldGen" / "models--WorldGen-Flux-Lora" / filename,
    )
    if local_path:
        return local_path
    return hf_hub_download(repo_id="LeoXie/WorldGen", filename=f"models--WorldGen-Flux-Lora/{filename}")


def _flux_repo(remote_repo: str, local_name: str) -> str:
    return _first_existing(
        CKPT_ROOT / local_name,
        CKPT_ROOT / f"black-forest-labs--{local_name}",
        CKPT_ROOT / "hfd" / f"black-forest-labs--{local_name}",
    ) or remote_repo


def build_pano_gen_model(lora_path=None, device="cuda", low_vram=True):
    """Build a panorama generation model with optional Nunchaku low VRAM support."""
    if lora_path is None:
        lora_path = _worldgen_lora("worldgen_text2scene.safetensors")
    
    if low_vram:
        from nunchaku import NunchakuFluxTransformer2dModel
        from nunchaku.utils import get_precision

        # Get precision and initialize Nunchaku transformer
        precision = get_precision()
        print(f"Using Nunchaku with {precision} precision")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/svdq-{precision}-flux.1-dev",
            offload=True
        )
        # Initialize pipeline with Nunchaku transformer
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device=device
        )
        # Load and fix LoRA weights
        print(f"Loading LoRA weights from: {lora_path}")
        state_dict, _ = load_and_fix_lora(lora_path)
        transformer.update_lora_params(state_dict)
    else:
        # Standard pipeline initialization
        pipe = FluxPipeline.from_pretrained(
            _flux_repo("black-forest-labs/FLUX.1-dev", "FLUX.1-dev"),
            torch_dtype=torch.bfloat16,
        )
        # Load LoRA weights using standard diffusers method
        print(f"Loading LoRA weights from: {lora_path}")
        _load_standard_lora(pipe, lora_path)
    
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()
    return pipe

def build_pano_fill_model(lora_path=None, device="cuda", low_vram=True):
    """Build a panorama fill model with optional Nunchaku low VRAM support."""
    if lora_path is None:
        lora_path = _worldgen_lora("worldgen_img2scene.safetensors")
    
    if low_vram:
        from nunchaku import NunchakuFluxTransformer2dModel
        from nunchaku.utils import get_precision

        # Get precision and initialize Nunchaku transformer
        precision = get_precision()
        print(f"Using Nunchaku with {precision} precision")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/svdq-{precision}-flux.1-fill-dev",
            offload=True
        )
        # Initialize pipeline with Nunchaku transformer
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device=device
        )
        # Load and fix LoRA weights
        print(f"Loading LoRA weights from: {lora_path}")
        state_dict, _ = load_and_fix_lora(lora_path)
        transformer.update_lora_params(state_dict)
    else:
        # Standard pipeline initialization
        pipe = FluxFillPipeline.from_pretrained(
            _flux_repo("black-forest-labs/FLUX.1-Fill-dev", "FLUX.1-Fill-dev"),
            torch_dtype=torch.bfloat16,
        )
        # Load LoRA weights using standard diffusers method
        print(f"Loading LoRA weights from: {lora_path}")
        _load_standard_lora(pipe, lora_path)
    
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()
    return pipe

def gen_pano_image(
        model,
        prompt="", 
        output_path=None, 
        seed=42, 
        guidance_scale=7.0, 
        num_inference_steps=50, 
        height=800, 
        width=1600, 
        blend_extend=6,
        prefix="A high quality 360 panorama photo of",
        suffix="HDR, RAW, 360 consistent, omnidirectional",
    ):
    """Generates a panorama image using FLUX.1-dev and a LoRA."""
    prompt = f"{prefix}, {prompt}, {suffix}"
    generator = torch.Generator("cpu").manual_seed(seed)
    image = model(
        prompt,
        height=height,
        width=width,
        generator=generator,
        num_inference_steps=num_inference_steps,
        blend_extend=blend_extend,
        guidance_scale=guidance_scale
    ).images[0]
    
    if output_path is not None:
        image.save(output_path)
        print(f"Panorama image saved to {output_path}")
        
    return image

def gen_pano_fill_image(
        model,
        image,
        mask,
        prompt="a scene",
        output_path=None,
        seed=42,
        guidance_scale=30.0,
        num_inference_steps=50,
        height=800,
        width=1600,
        blend_extend=6,
        prefix="A high quality 360 panorama photo of",
        suffix="HDR, RAW, 360 consistent, omnidirectional",
    ):
    image = image.resize((width, height))
    mask = mask.resize((width, height))
    generator = torch.Generator("cpu").manual_seed(seed)
    prompt = f"{prefix} {prompt} {suffix}"
    image = model(
        prompt,
        height=height,
        width=width,
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        blend_extend=blend_extend
    ).images[0]

    if output_path is not None:
        image.save(output_path)
        print(f"Panorama image saved to {output_path}")
        
    return image

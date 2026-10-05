import os
import torch
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.light_22 import Wan2_2_VAE
VAE_DTYPE = torch.bfloat16

def _parse_lightvae_pruning_rate(value):
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("", "auto", "none"):
            return None
    return float(value)

def get_vae_config(args=None):
    if args.vae_type == 'mg_lightvae':
        vae_path = os.path.join(args.ckpt_dir, "MG-LightVAE.pth")
        args.lightvae_pruning_rate = 0.5
    elif args.vae_type == 'mg_lightvae_v2':
        vae_path = os.path.join(args.ckpt_dir, "MG-LightVAE_v2.pth")
        args.lightvae_pruning_rate = 0.75
    else:
        vae_path = os.path.join(args.ckpt_dir, "Wan2.2_VAE.pth")
        # The full Wan checkpoint must not inherit the lightweight VAE's default.
        args.lightvae_pruning_rate = 0.0
    return {
        "vae_path": vae_path,
        "vae_dtype": "bfloat16",
        "lightvae_pruning_rate":_parse_lightvae_pruning_rate(args.lightvae_pruning_rate),
        "lightvae_encoder_path": os.path.join(args.ckpt_dir, "Wan2.2_VAE.pth"),
    }

def load_vae(device_id=0, args=None, metadata=None,):

    if metadata:
        vae_path = metadata.get('vae_path')
        lightvae_pruning_rate = _parse_lightvae_pruning_rate(metadata.get('lightvae_pruning_rate', None))
        lightvae_encoder_path = str(metadata.get('lightvae_encoder_path'))
    elif args:
        config = get_vae_config(args)
        vae_path = config['vae_path']
        lightvae_pruning_rate = config["lightvae_pruning_rate"]
        lightvae_encoder_path = config["lightvae_encoder_path"]
    else:
        raise ValueError("Either metadata or args must be provided")

    if lightvae_pruning_rate is None:
        lightvae_pruning_rate = 0.0
    vae_type = "wan2.2" if float(lightvae_pruning_rate) <= 0.0 else "mg_lightvae"
    device = torch.device(f"cuda:{device_id}")
    # print(f"Loading VAE from {vae_path} with pruning rate {lightvae_pruning_rate}", flush=True)
    vae = Wan2_2_VAE(
        vae_pth=vae_path, 
        device=device, 
        dtype=VAE_DTYPE,
        vae_type=vae_type,
        lightvae_pruning_rate=lightvae_pruning_rate,
        lightvae_encoder_vae_pth=lightvae_encoder_path,
    )
    
    vae.model.to(device=device, dtype=VAE_DTYPE)
    if hasattr(vae, "encoder_model") and vae.encoder_model is not None:
        vae.encoder_model.to(device=device, dtype=VAE_DTYPE)
    vae.scale[0] = vae.scale[0].to(device=device, dtype=VAE_DTYPE)
    vae.scale[1] = vae.scale[1].to(device=device, dtype=VAE_DTYPE)
    
    return vae

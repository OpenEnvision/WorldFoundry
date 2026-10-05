from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import random

from worldfoundry.evaluation.utils import worldfoundry_data_path

DEFAULT_DYNAMICRAFTER_CONFIG_ROOT = worldfoundry_data_path(
    "models",
    "runtime", "configs",
    "dynamicrafter",
)
REQUIRED_IMPORTS = {
    "einops": "einops",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "open_clip": "open_clip",
    "torch": "torch",
    "torchvision": "torchvision",
}


@dataclass(frozen=True)
class DynamiCrafterRuntimePlan:
    """Describe DynamiCrafter runtime readiness without loading weights.

    Args:
        config_path: Resolved in-tree architecture config path.
        checkpoint_path: Resolved external checkpoint path.
        config_exists: Whether the architecture config exists.
        checkpoint_exists: Whether the checkpoint asset exists.
        missing_imports: Required Python modules missing from the environment.
    """

    config_path: Path
    checkpoint_path: Path
    config_exists: bool
    checkpoint_exists: bool
    missing_imports: tuple[str, ...]

    @property
    def status(self) -> str:
        """Return the runtime readiness state.

        Args:
            None.
        """
        if self.config_exists and self.checkpoint_exists and not self.missing_imports:
            return "ready"
        return "blocked"


def _resolve_config_path(config: str) -> Path:
    """Resolve a DynamiCrafter architecture config path.

    Args:
        config: Absolute path or data/models runtime config path.
    """
    config_path = Path(config).expanduser()
    if config_path.is_absolute():
        return config_path
    if config_path.parts and config_path.parts[:2] == ("runtime", "configs"):
        return worldfoundry_data_path("models", *config_path.parts)
    if len(config_path.parts) == 1:
        return DEFAULT_DYNAMICRAFTER_CONFIG_ROOT / config_path
    return config_path.resolve()


def _resolve_checkpoint_path(ckpt_path: str | Path) -> Path:
    checkpoint_path = Path(ckpt_path).expanduser()
    if checkpoint_path.is_dir():
        nested_checkpoint = checkpoint_path / "model.ckpt"
        if nested_checkpoint.is_file():
            return nested_checkpoint
    return checkpoint_path


def plan_runtime(config: str, ckpt_path: str) -> DynamiCrafterRuntimePlan:
    """Build a lightweight DynamiCrafter runtime readiness plan.

    Args:
        config: Absolute path or data/models runtime config path.
        ckpt_path: External checkpoint path.
    """
    config_path = _resolve_config_path(config)
    checkpoint_path = _resolve_checkpoint_path(ckpt_path)
    missing_imports = tuple(
        name
        for name, import_name in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(import_name) is None
    )
    return DynamiCrafterRuntimePlan(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        config_exists=config_path.is_file(),
        checkpoint_exists=checkpoint_path.is_file(),
        missing_imports=missing_imports,
    )


def _seed_everything(seed: int) -> int:
    """Seed DynamiCrafter without importing the full Lightning stack.

    DynamiCrafter previously imported :mod:`pytorch_lightning` solely for its
    ``seed_everything`` helper.  On network filesystems that transitively scans
    torchmetrics and transformers and can add several minutes to every cold
    model load.  Keep the helper's relevant ``workers=False`` semantics local.
    """
    import numpy as np
    import torch

    value = int(seed)
    if not 0 <= value <= 2**32 - 1:
        raise ValueError(f"{value} is not in bounds, numpy accepts from 0 to {2**32 - 1}")
    os.environ["PL_GLOBAL_SEED"] = str(value)
    os.environ["PL_SEED_WORKERS"] = "0"
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    return value


class DynamiCrafter:
    def __init__(
        self,
        model_name: str,
        generation_type: str,
        config: str,
        ckpt_path: str,
        height: int,
        width: int,
        perframe_ae: bool = False,
        video_length: int = 16,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 7.5,
        cfg_img: float | None = None,
        frame_stride: int = 3,
        text_input: bool = False,
        multiple_cond_cfg: bool = False,
        loop: bool = False,
        interp: bool = False,
        timestep_spacing: str = "uniform",
        guidance_rescale: float = 0.0,
        seed: int = 123,
        device: str = "cuda",
    ):
        """Initialize DynamiCrafter from an in-tree runtime.

        Args:
            model_name: Registry name used for reporting.
            generation_type: Supported generation mode, currently image-to-video.
            config: In-tree architecture config path.
            ckpt_path: External model checkpoint path.
            height: Output frame height.
            width: Output frame width.
            perframe_ae: Decode frames independently through the autoencoder.
            video_length: Number of generated frames.
            n_samples: Number of sampled variants.
            ddim_steps: Number of DDIM sampling steps.
            ddim_eta: DDIM eta parameter.
            unconditional_guidance_scale: Text guidance scale.
            cfg_img: Image guidance scale for multi-condition sampling.
            frame_stride: FPS/frame stride conditioning value.
            text_input: Whether prompt text is used.
            multiple_cond_cfg: Use the multi-condition sampler.
            loop: Preserve both first and last input latent frames.
            interp: Preserve endpoints for interpolation.
            timestep_spacing: DDIM timestep schedule name.
            guidance_rescale: Noise guidance rescale factor.
            seed: Sampling seed.
            device: Explicit torch device assigned by the Workspace GPU lease.
        """
        assert generation_type == "i2v"

        runtime_plan = plan_runtime(config, str(ckpt_path))

        if not runtime_plan.config_exists:
            raise FileNotFoundError(
                f"DynamiCrafter config not found: {runtime_plan.config_path}. "
                f"Expected an in-tree config under {DEFAULT_DYNAMICRAFTER_CONFIG_ROOT}."
            )
        if runtime_plan.missing_imports:
            raise ModuleNotFoundError(
                "DynamiCrafter runtime environment is incomplete. "
                f"Missing Python modules: {', '.join(runtime_plan.missing_imports)}."
            )
        if not runtime_plan.checkpoint_exists:
            raise FileNotFoundError(
                f"DynamiCrafter checkpoint not found: {runtime_plan.checkpoint_path}. "
                "Weights are external assets and must be provided separately."
            )

        from omegaconf import OmegaConf
        import torch

        from worldfoundry.core.model_loading.factory import (
            instantiate_from_config,
        )

        _seed_everything(seed)
        model_config = OmegaConf.load(runtime_plan.config_path).pop("model", OmegaConf.create())

        # set use_checkpoint as False as when using deepspeed, it encounters an error
        # "deepspeed backend not set"
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        self.device = torch.device(device)
        model = model.to(self.device)
        model.perframe_ae = perframe_ae

        self.model = load_model_checkpoint(model, str(runtime_plan.checkpoint_path))
        self.model.eval()

        ## sample shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Error: image size [h,w] should be multiples of 16!")

        self.model_name = model_name
        self.height = height
        self.width = width
        self.channels = self.model.model.diffusion_model.out_channels
        self.video_length = video_length
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.cfg_img = cfg_img
        self.frame_stride = frame_stride
        self.text_input = text_input
        self.multiple_cond_cfg = multiple_cond_cfg
        self.loop = loop
        self.interp = interp
        self.timestep_spacing = timestep_spacing
        self.guidance_rescale = guidance_rescale

    def generate_video(
        self,
        prompt: str,
        image_path: str,
    ):
        """Generate a video tensor from one input image.

        Args:
            prompt: Text prompt used for image-conditioned generation.
            image_path: Path to the input image.
        """
        import torch

        h, w = self.height // 8, self.width // 8
        n_frames = self.video_length
        noise_shape = [1, self.channels, self.video_length, h, w]

        with torch.no_grad(), torch.amp.autocast(
            self.device.type,
            enabled=self.device.type == "cuda",
        ):
            videos = load_data_images(
                image_path,
                video_size=(self.height, self.width),
                video_frames=n_frames,
            )
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to(self.device)
            else:
                videos = videos.unsqueeze(0).to(self.device)

            batch_samples = image_guided_synthesis(
                self.model,
                prompt,
                videos,
                noise_shape,
                self.n_samples,
                self.ddim_steps,
                self.ddim_eta,
                self.unconditional_guidance_scale,
                self.cfg_img,
                self.frame_stride,
                self.text_input,
                self.multiple_cond_cfg,
                self.loop,
                self.interp,
                self.timestep_spacing,
                self.guidance_rescale,
            )

            ### benchmark output
            batch_samples = batch_samples.detach().squeeze().cpu()
            batch_samples = torch.clamp(batch_samples.float(), -1.0, 1.0)
            batch_samples = (batch_samples + 1.0) / 2.0
            batch_samples = batch_samples.permute(1, 0, 2, 3)

        return batch_samples


def load_model_checkpoint(model, ckpt):
    """Load a DynamiCrafter checkpoint into an initialized model.

    Args:
        model: Instantiated DynamiCrafter architecture.
        ckpt: External checkpoint file path.
    """
    import torch

    state_dict = torch.load(
        ckpt,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        if state_keys != model_keys:
            state_dict = {
                key.replace("framestride_embed", "fps_embedding"): value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=True)
    else:
        # deepspeed
        new_pl_sd = dict()
        for key in state_dict["module"]:
            new_pl_sd[key[16:]] = state_dict["module"][key]
        model.load_state_dict(new_pl_sd)
    print(">>> model checkpoint loaded.")
    return model


def load_data_images(file_path, video_size=(256, 256), video_frames=16):
    """Load and repeat one image into an input video tensor.

    Args:
        file_path: Path to the source image.
        video_size: Target frame size as height and width.
        video_frames: Number of repeated frames.
    """
    import torchvision.transforms as transforms
    from einops import repeat
    from PIL import Image

    transform = transforms.Compose(
        [
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
    frame_tensor = repeat(
        image_tensor, "c t h w -> c (repeat t) h w", repeat=video_frames
    )

    return frame_tensor


def get_latent_z(model, videos):
    """Encode video frames into DynamiCrafter latent space.

    Args:
        model: Initialized DynamiCrafter model.
        videos: Video tensor with shape [batch, channels, frames, height, width].
    """
    from einops import rearrange

    b, c, t, h, w = videos.shape
    x = rearrange(videos, "b c t h w -> (b t) c h w")
    z = model.encode_first_stage(x)
    z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
    return z


def image_guided_synthesis(
    model,
    prompts,
    videos,
    noise_shape,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    unconditional_guidance_scale=1.0,
    cfg_img=None,
    fs=None,
    text_input=False,
    multiple_cond_cfg=False,
    loop=False,
    interp=False,
    timestep_spacing="uniform",
    guidance_rescale=0.0,
):
    """Run image-guided DDIM synthesis.

    Args:
        model: Initialized DynamiCrafter model.
        prompts: Prompt text or prompt list.
        videos: Conditioning video tensor.
        noise_shape: Latent sampling shape.
        n_samples: Number of variants to sample.
        ddim_steps: Number of DDIM steps.
        ddim_eta: DDIM eta parameter.
        unconditional_guidance_scale: Text guidance scale.
        cfg_img: Image guidance scale for multi-condition sampling.
        fs: FPS/frame stride conditioning value.
        text_input: Whether prompt text is used.
        multiple_cond_cfg: Use the multi-condition sampler.
        loop: Preserve first and last latent frames.
        interp: Preserve endpoints for interpolation.
        timestep_spacing: DDIM timestep schedule name.
        guidance_rescale: Noise guidance rescale factor.
    """
    import torch
    from einops import repeat

    from worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim import DDIMSampler
    from worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim_multiplecond import (
        DDIMSampler as DDIMSampler_multicond,
    )

    kwargs = {}
    ddim_sampler = (
        DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    )
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    img = videos[:, :, 0]  # bchw
    img_emb = model.embedder(img)  ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    if model.model.conditioning_key == "hybrid":
        z = get_latent_z(model, videos)  # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
            img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
        else:
            img_cat_cond = z[:, :, :1, :, :]
            img_cat_cond = repeat(
                img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2]
            )
        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):
        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                cfg_img=cfg_img,
                mask=cond_mask,
                x0=cond_z0,
                fs=fs,
                timestep_spacing=timestep_spacing,
                guidance_rescale=guidance_rescale,
                **kwargs,
            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

import json

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from .camera import get_camera_embedding
from .utils import load_video


def load_dataset(
    reference_image,
    render_path,
    nframe,
    max_area,
    pipe,
    use_camera_embedding,
    device,
    sp_degree=1,
    logger=None,
    load_human_info=False,
):
    image = Image.open(reference_image)
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    mod_value = mod_value * sp_degree
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    if logger is not None:
        logger.info(f"Resized image to {height}x{width}")
    image = image.resize((width, height))

    # load conditions
    render_frames = load_video(f"{render_path}/render.mp4")
    render_video = torch.stack([ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    render_video = F.interpolate(render_video, size=(height, width), mode="bicubic")[None]
    # replace the first frame with the high-quality reference image
    render_video[0, 0] = ToTensor()(image) * 2 - 1
    render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
    render_mask_frames = load_video(f"{render_path}/render_mask.mp4")
    render_mask = torch.stack([ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
    render_mask = F.interpolate(render_mask, size=(height, width), mode="nearest")[None]  # [0,1],[1,f,1,h,w]
    render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1

    # load camera
    cam_info = json.load(open(f"{render_path}/cam_info.json"))  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
    w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
    intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)
    intrinsic[0, :] = intrinsic[0, :] / cam_info["width"] * width
    intrinsic[1, :] = intrinsic[1, :] / cam_info["height"] * height
    intrinsic = intrinsic[None].repeat(nframe, 1, 1)
    if use_camera_embedding:
        camera_embedding = get_camera_embedding(intrinsic, w2cs, nframe, height, width, normalize=True)
    else:
        camera_embedding = None

    if not load_human_info:
        return image, render_video, render_mask, camera_embedding, height, width
    else:
        smpl_frames = load_video(f"{render_path}/smpl_render.mp4")
        smpl_video = torch.stack([ToTensor()(frame) for frame in smpl_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        smpl_video = F.interpolate(smpl_video, size=(height, width), mode="bicubic")[None]
        smpl_video = torch.clip(smpl_video, -1, 1)  # [-1~1], [1,f,c,h,w]
        hand_frames = load_video(f"{render_path}/hand_render.mp4")
        hand_video = torch.stack([ToTensor()(frame) for frame in hand_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        hand_video = F.interpolate(hand_video, size=(height, width), mode="bicubic")[None]
        hand_video = torch.clip(hand_video, -1, 1)  # [-1~1], [1,f,c,h,w]

        return image, render_video, render_mask, camera_embedding, smpl_video, hand_video, height, width

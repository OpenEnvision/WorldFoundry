import torch
import torch.nn.functional as F
from einops import rearrange


def convert_mask_video(mask_video):  # b 3 275 480 832
    b = mask_video.shape[0]  # 1
    h, w = mask_video.shape[-2:]  # 480, 832
    lat_h, lat_w = h // 8, w // 8  # 60, 104
    mask_video = mask_video[:, :1]  # b 1 275 480 832
    mask_video = rearrange(mask_video, "b c t h w -> (b t) c h w")  # (b 275) 1 480 832
    mask_video = F.interpolate(
        mask_video, size=(lat_h, lat_w), mode="bilinear", align_corners=False
    )  # (b 275) 1 60 104
    mask_video = rearrange(mask_video, "(b t) c h w -> b t c h w", b=b)  # b 275 1 60 104
    mask_video = torch.cat(
        [torch.repeat_interleave(mask_video[:, 0:1], repeats=4, dim=1), mask_video[:, 1:]], dim=1
    )  # b t+3 1 60 104
    if mask_video.shape[1] % 4 != 0:
        # remove the unused frames
        mask_video = mask_video[:, : mask_video.shape[1] - mask_video.shape[1] % 4]
        print(f"mask_video shape after removing unused frames: {mask_video.shape}")
    mask_video = mask_video.view(mask_video.shape[0], mask_video.shape[1] // 4, 4, lat_h, lat_w)  # b t+3 4 60 104
    return mask_video


def down_sample_video(video):
    bs = video.shape[0]
    video = rearrange(video, "b c t h w -> (b t) c h w")
    video = F.interpolate(video, scale_factor=(0.25, 0.25), mode="bilinear", align_corners=False)  # size=(h//4, w//4)
    video = rearrange(video, "(b t) c h w -> b c t h w", b=bs)
    return video

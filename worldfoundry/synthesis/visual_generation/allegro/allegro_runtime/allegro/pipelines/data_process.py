# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


from worldfoundry.core.media.processing.video_transforms import crop as crop
from worldfoundry.core.media.processing.video_transforms import is_tensor_video_clip as _is_tensor_video_clip
from worldfoundry.core.media.processing.video_transforms import resize as resize
from worldfoundry.core.media.processing.video_transforms import to_tensor as to_tensor


def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")

    # import ipdb;ipdb.set_trace()
    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    j = int(round((w - new_w) / 2.0))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    return crop(clip, i, j, new_h, new_w)


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
        self,
        size,
        top_crop=False,
        interpolation_mode="bilinear",
    ):
        if len(size) != 2:
            raise ValueError(f"size should be tuple (height, width), instead got {size}")
        self.size = size
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        # clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop = center_crop_th_tw(clip, self.size[0], self.size[1], top_crop=self.top_crop)
        # import ipdb;ipdb.set_trace()
        clip_center_crop_resize = resize(
            clip_center_crop, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"

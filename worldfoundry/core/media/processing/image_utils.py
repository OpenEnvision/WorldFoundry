"""Image visualization and tensor-to-display conversion helpers.

Responsibility
    Letterbox, horizontal view split/compose, PIL load, and a basic
    ``[0, 255] → normalize`` tensor preprocess for eval UIs.

Boundaries
    Shared by eval UIs. Model-specific preprocess should stay in the encoder
    pipeline. ``Cv2Display`` needs a GUI ``DISPLAY``; it is not headless.
    ``to_image`` assumes 3-channel RGB in ``auto`` mode.

Public surface
    :func:`to_image`, :func:`load_pil_image`, :func:`materialize_image_input`,
    :func:`compose_horizontal_views`, :func:`split_horizontal_views`,
    :func:`resize_and_letterbox`, :func:`imread` / :func:`imsave` / :func:`imshow`,
    :class:`Cv2Display`, :func:`basic_image_tensor_preprocess`.
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.utils.tensors.arrays import any_describe
from worldfoundry.core.utils.python.misc_utils import global_once
from worldfoundry.core.utils.tensors.torch import torch_normalize


# ──────────────────────────────────────────────────────────────────────────
# PIL / layout — RGB views for eval UIs; not encoder-specific preprocess
# ──────────────────────────────────────────────────────────────────────────


def to_image(img, channel_order="auto"):
    """
    Returns:
        numpy image of shape [H, W, C]
        in "auto" mode, we assume C == 3
    """
    assert channel_order in ["hwc", "chw", "auto"]
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    assert isinstance(img, np.ndarray)
    if img.ndim == 4:
        assert img.shape[0] == 1
        img = img[0]
    assert img.ndim == 3
    if channel_order == "auto":
        # use C==3 to detect order
        if img.shape[0] == 3:
            channel_order = "chw"
        else:
            assert img.shape[-1] == 3, "image should either have [3,H,W] or [H,W,3]"
            channel_order = "hwc"
    img = img.astype(np.uint8)
    if channel_order == "chw":
        return np.transpose(img, (1, 2, 0))
    else:
        return img


def load_pil_image(image_input, *, first_sequence_item: bool = True) -> Image.Image:
    """Normalize a path, PIL image, numpy array, torch tensor, or sequence to RGB PIL."""

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, os.PathLike)):
        return Image.open(image_input).convert("RGB")
    if (
        first_sequence_item
        and isinstance(image_input, Sequence)
        and not isinstance(image_input, (str, bytes, bytearray))
    ):
        if not image_input:
            raise ValueError("image input sequence is empty")
        return load_pil_image(image_input[0], first_sequence_item=first_sequence_item)
    if isinstance(image_input, np.ndarray):
        array = image_input
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                if array.size and array.min() >= -1.0 and array.max() <= 1.0:
                    array = (array + 1.0) * 127.5 if array.min() < 0.0 else array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array[..., :3]).convert("RGB")
    if torch.is_tensor(image_input):
        tensor = image_input.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 0)
        return load_pil_image(tensor.numpy(), first_sequence_item=first_sequence_item)
    raise TypeError(f"Unsupported image input type: {type(image_input)!r}")


def materialize_image_input(
    image_input,
    output_dir: str | os.PathLike[str],
    *,
    filename: str = "input.png",
) -> str:
    """Normalize an image-like input and save it at a stable local path."""

    output_path = Path(output_dir).expanduser().resolve() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_pil_image(image_input).save(output_path)
    return str(output_path)


def compose_horizontal_views(
    images,
    *,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Resize optional per-view images and concatenate them horizontally."""

    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
        images = [images]
    if not images:
        raise ValueError("at least one image view is required")
    views = [load_pil_image(image, first_sequence_item=False) for image in images]
    if target_size is not None:
        views = [image.resize(target_size, Image.Resampling.LANCZOS) for image in views]
    elif len({image.height for image in views}) > 1:
        target_height = min(image.height for image in views)
        views = [
            image.resize(
                (round(image.width * target_height / image.height), target_height),
                Image.Resampling.LANCZOS,
            )
            for image in views
        ]
    canvas = Image.new("RGB", (sum(image.width for image in views), views[0].height))
    offset = 0
    for image in views:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def split_horizontal_views(
    image,
    *,
    num_views: int,
    target_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Split a horizontally concatenated image into equally sized RGB views."""

    if num_views <= 0:
        raise ValueError("num_views must be positive")
    source = load_pil_image(image, first_sequence_item=False)
    if source.width % num_views:
        raise ValueError(f"image width {source.width} is not divisible by num_views={num_views}")
    view_width = source.width // num_views
    views = [
        source.crop((index * view_width, 0, (index + 1) * view_width, source.height)) for index in range(num_views)
    ]
    if target_size is not None:
        views = [view.resize(target_size, Image.Resampling.LANCZOS) for view in views]
    return views


# ──────────────────────────────────────────────────────────────────────────
# I/O and resize — uint8 HWC for display; center-crop vs letterbox are distinct
# ──────────────────────────────────────────────────────────────────────────


def imshow(img):
    """Show ``img`` via matplotlib after :func:`to_image` (blocking backend)."""
    import matplotlib.pyplot as plt

    plt.imshow(to_image(img))


def imsave(img, path):
    """Write ``img`` as uint8 HWC through imageio; ``~`` in ``path`` is expanded."""
    import imageio

    imageio.imsave(os.path.expanduser(path), to_image(img))


def resize_and_center_crop(image, target_width, target_height):
    """Scale HWC uint8 so the target fits, then center-crop (may discard edges)."""
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_and_letterbox(
    image,
    target_width: int,
    target_height: int,
    *,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Fit an RGB image inside a fixed canvas while preserving aspect ratio."""

    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive")
    source = load_pil_image(image, first_sequence_item=False)
    scale = min(target_width / source.width, target_height / source.height)
    resized_width = max(1, int(source.width * scale))
    resized_height = max(1, int(source.height * scale))
    resized = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), fill)
    canvas.paste(
        resized,
        ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
    )
    return canvas


def resize_and_center_crop_pytorch(image, target_width, target_height):
    """Batched NCHW bilinear resize + center crop; ``align_corners=False``."""
    B, C, H, W = image.shape

    if H == target_height and W == target_width:
        return image

    scale_factor = max(target_width / W, target_height / H)
    resized_width = int(round(W * scale_factor))
    resized_height = int(round(H * scale_factor))

    resized = torch.nn.functional.interpolate(
        image, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    cropped = resized[:, :, top : top + target_height, left : left + target_width]

    return cropped


def resize_without_crop(image, target_width, target_height):
    """Stretch HWC uint8 to the target size; aspect ratio is not preserved."""
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def imread(path, channel_order="chw", format="torch"):
    """Load an image file; default CHW torch tensor (not normalized)."""
    import imageio

    assert channel_order in ["hwc", "chw"]
    assert format in ["numpy", "torch"]
    img = imageio.imread(path)
    if channel_order == "chw":
        img = np.transpose(img, (2, 0, 1))  # hwc -> chw
    if format == "torch":
        return torch.from_numpy(img)
    else:
        return img


class Cv2Display:
    """OpenCV window pump for live tensors; no-ops when ``enabled=False``."""

    def __init__(
        self,
        window_name="display",
        image_size=None,
        channel_order="auto",
        bgr2rgb=True,
        step_sleep=0,
        enabled=True,
    ):
        """
        Use cv2.imshow() to pop a window, requires virtual desktop GUI

        Args:
            channel_order: auto, hwc, or chw
            image_size: None to use the original image size, otherwise resize
            step_sleep: sleep for a few seconds
        """
        self._window_name = window_name
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert image_size is None or len(image_size) == 2
        self._image_size = image_size
        assert channel_order in ["auto", "chw", "hwc"]
        self._channel_order = channel_order
        self._bgr2rgb = bgr2rgb
        self._step_sleep = step_sleep
        self._enabled = enabled

    def _resize(self, img):
        """Area-downsample or linear-upsample to ``image_size``; skip if unset."""
        import cv2

        if self._image_size is None:
            return img
        H, W = img.shape[:2]
        Ht, Wt = self._image_size  # target
        return cv2.resize(
            img,
            self._image_size,
            interpolation=cv2.INTER_AREA if Ht < H else cv2.INTER_LINEAR,
        )

    def _reorder(self, img):
        """CHW → HWC when ``auto`` sees a leading 1/3 channel dim."""
        if self._channel_order == "chw":
            return np.transpose(img, (1, 2, 0))
        elif self._channel_order == "hwc":
            return img
        else:
            if img.shape[0] in [1, 3]:  # chw
                return np.transpose(img, (1, 2, 0))
            else:
                return img

    def __call__(self, img):
        """Show one frame; invent ``DISPLAY=:0.0`` if unset (IsaacGym segfault guard)."""
        if not self._enabled:
            return
        import cv2
        import torch

        # prevent segfault in IsaacGym
        display_var = os.environ.get("DISPLAY", None)
        if not display_var:
            os.environ["DISPLAY"] = ":0.0"

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = self._resize(self._reorder(img))
        if self._bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time.sleep(self._step_sleep)
        cv2.imshow(self._window_name, img)
        cv2.waitKey(1)

        if display_var is not None:
            os.environ["DISPLAY"] = display_var

    def close(self):
        """Destroy the named OpenCV window; safe when the display was disabled."""
        if not self._enabled:
            return
        import cv2

        cv2.destroyWindow(self._window_name)


# ──────────────────────────────────────────────────────────────────────────
# Image tensor handling — catch un-normalized integer inputs before a NN
# ──────────────────────────────────────────────────────────────────────────


def sanity_check_image_tensor(img: torch.Tensor, on_error: Literal["raise", "warn", "ignore"] = "raise"):
    """
    Check if the input image tensor is all integers, which is wrong for any NN input.
    This is a common case if the user forgets to normalize the image first
    """
    assert on_error in [
        "raise",
        "warn",
        "ignore",
    ], 'on_error must be "raise", "warn", or "ignore"'
    if not img.dtype.is_floating_point:
        msg = f"Image tensor is not floating point format, but {img.dtype}!"
        if on_error == "raise":
            raise ValueError(msg)
        elif on_error == "warn":
            warnings.warn(msg)
        else:
            return False
    # check if all values in the image are close to an integer
    if (img - torch.round(img)).abs().max() < 1e-5:
        msg = "Input image is all close to integers, are you sure you have normalized it before passing it to a NN?"
        if on_error == "raise":
            raise ValueError(msg)
        elif on_error == "warn":
            warnings.warn(msg)
        else:
            return False
    return True


@torch.no_grad()
def basic_image_tensor_preprocess(
    img,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    shape: tuple[int, int] | None = None,
):
    """
    Check for resize, and divide by 255
    """
    import kornia

    assert torch.is_tensor(img)
    assert img.dim() >= 4, any_describe(img)
    original_shape = list(img.size())
    img = img.float()
    img = img.flatten(0, img.dim() - 4)
    assert img.dim() == 4

    input_size = img.size()[-2:]
    if global_once("worldfoundry.core.media.processing.image_utils.basic_image_preprocess:input_size"):
        assert img.max() > 2, "img should be between [0, 255] before normalize"

    if shape and input_size != shape:
        if global_once("worldfoundry.core.media.processing.image_utils.basic_image_preprocess:transform"):
            warnings.warn(
                f"{'Down' if shape < input_size else 'Up'}sampling image"
                f" from original resolution {input_size}x{input_size}"
                f" to {shape}x{shape}"
            )
        img = kornia.geometry.transform.resize(img, shape).clamp(0.0, 255.0)

    B, C, H, W = img.size()
    assert C % 3 == 0, "channel must divide 3"
    img = img.view(B * C // 3, 3, H, W)
    img = torch_normalize(img / 255.0, mean=mean, std=std)
    original_shape[-2:] = H, W
    return img.view(original_shape)

# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Oriented sRGB image loading and codec-matched conditioning transforms.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES).
"""

from __future__ import annotations

import logging
import math
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


def load_srgb_image(path: str | Path) -> np.ndarray:
    """Decode a local image as oriented, sRGB, three-channel uint8 pixels."""
    try:
        with Image.open(path) as image:
            orientation = image.getexif().get(274)
            rotation = {3: 180, 6: 270, 8: 90}.get(orientation)
            if rotation is not None:
                image = image.rotate(rotation, expand=True)
            icc_profile = image.info.get("icc_profile")
            if image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode == "LA":
                image = image.convert("L")
            if icc_profile:
                try:
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(BytesIO(icc_profile)),
                        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
                        outputMode="RGB",
                    )
                except (ImageCms.PyCMSError, OSError, ValueError) as error:
                    logger.warning("Failed to convert image to sRGB: %s", error)
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")
            return np.array(image, dtype=np.uint8)
    except UnidentifiedImageError as error:
        raise ValueError(f"Cannot decode image file {str(path)!r}.") from error


def recompress_image_h264(image: np.ndarray, crf: int) -> np.ndarray:
    """Round-trip one RGB frame through libx264/yuv420p; CRF 0 leaves it unchanged."""
    if crf == 0 or min(image.shape[:2]) < 2:
        return image
    import av

    height, width = (value // 2 * 2 for value in image.shape[:2])
    with BytesIO() as output:
        with av.open(output, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
            stream.height, stream.width = height, width
            frame = av.VideoFrame.from_ndarray(image[:height, :width], format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        encoded = output.getvalue()
    with BytesIO(encoded) as source, av.open(source) as container:
        return next(container.decode(container.streams.video[0])).to_ndarray(format="rgb24")


def resize_center_crop_bilinear(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize HWC or FHWC pixels to fill a crop, returning BCFHW with B=1."""
    import torch.nn.functional as functional

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("image must use HWC or FHWC layout")
    source_height, source_width = image.shape[1:3]
    scale = max(height / source_height, width / source_width)
    resized_height, resized_width = math.ceil(source_height * scale), math.ceil(source_width * scale)
    pixels = functional.interpolate(
        image.permute(0, 3, 1, 2),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    top, left = (resized_height - height) // 2, (resized_width - width) // 2
    return pixels[:, :, top : top + height, left : left + width].permute(1, 0, 2, 3).unsqueeze(0)

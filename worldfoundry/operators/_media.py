"""Module for the  Media operator implementation."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


def pil_to_png_bytes(image_input: Image.Image) -> bytes:
    """Encode a PIL image as RGB PNG bytes.

    Args:
        image_input: Source PIL image.
    """
    if not isinstance(image_input, Image.Image):
        raise TypeError(f"image_input must be PIL.Image, got {type(image_input)}")

    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    buffer = BytesIO()
    image_input.save(buffer, format="PNG")
    return buffer.getvalue()


def pil_to_png_base64(image_input: Image.Image) -> str:
    """Encode a PIL image as a PNG base64 string.

    Args:
        image_input: Source PIL image.
    """
    return base64.b64encode(pil_to_png_bytes(image_input)).decode("utf-8")


def pil_to_png_data_url(image_input: Image.Image) -> str:
    """Encode a PIL image as a PNG data URL.

    Args:
        image_input: Source PIL image.
    """
    return f"data:image/png;base64,{pil_to_png_base64(image_input)}"


def image_or_string_to_png_data_url(
    image_input: Image.Image | str,
    *,
    load_existing_path: bool = False,
) -> str:
    """Return strings unchanged or encode PIL/local-path images as PNG data URLs.

    Args:
        image_input: PIL image or string image reference.
        load_existing_path: Whether existing string paths should be opened and encoded.
    """
    if isinstance(image_input, str):
        path = Path(image_input)
        if load_existing_path and path.exists():
            return pil_to_png_data_url(Image.open(path).convert("RGB"))
        return image_input

    return pil_to_png_data_url(image_input)


def require_public_url(url: str, field_name: str, service_name: str) -> str:
    """Validate that a media reference is a public HTTP or HTTPS URL.

    Args:
        url: Candidate URL.
        field_name: Field name used in error messages.
        service_name: Service name used in error messages.
    """
    if not isinstance(url, str):
        raise TypeError(f"{field_name} must be a string, got {type(url)}")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{field_name} must be a public HTTP/HTTPS URL for {service_name}.")
    return url

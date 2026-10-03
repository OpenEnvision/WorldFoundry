"""Render the official OpenEnvision PNG without changing its artwork or colors."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None

BRAND_NAME = "OpenEnvision"
PRODUCT_NAME = "WorldFoundry"
PRODUCT_TAGLINE = "Model · Benchmark · Studio"
LOGO_BACKGROUND = "#ffffff"

_BRAILLE_DOTS = (
    (0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3),
    (1, 1, 4), (2, 1, 5), (3, 0, 6), (3, 1, 7),
)


def logo_asset_path() -> Path:
    """Locate the packaged official logo."""
    try:
        asset = resources.files("worldfoundry.cli.assets").joinpath("openenvision_logo.png")
        with resources.as_file(asset) as path:
            return Path(path)
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return Path(__file__).with_name("assets") / "openenvision_logo.png"


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def _prepare_logo_image(image: Image.Image) -> Image.Image:
    """Remove outer whitespace while retaining the complete swan and wordmark."""
    rgba = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
    rgb = rgba.convert("RGB")
    red, green, blue = rgb.split()
    darkest = ImageChops.darker(ImageChops.darker(red, green), blue)
    bounds = darkest.point(lambda value: 255 if value < 240 else 0).getbbox()
    if bounds is None:
        return rgb
    left, top, right, bottom = bounds
    padding = 6
    return rgb.crop((
        max(0, left - padding), max(0, top - padding),
        min(rgb.width, right + padding), min(rgb.height, bottom + padding),
    ))


@lru_cache(maxsize=1)
def brand_logo_image() -> Image.Image:
    """Load original logo pixels, including its original lettering."""
    if Image is None:
        raise RuntimeError("Pillow is required to render the OpenEnvision logo.")
    with Image.open(logo_asset_path()) as image:
        return _prepare_logo_image(image)


def brand_logo_aspect_ratio() -> float:
    image = brand_logo_image()
    return image.width / image.height


def _resize_for_cells(
    image: Image.Image, width_chars: int, *, pixels_per_column: int,
    pixels_per_row: int, cell_aspect_ratio: float,
) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is required to render the OpenEnvision logo.")
    if width_chars < 1 or cell_aspect_ratio <= 0:
        raise ValueError("Logo width and cell aspect ratio must be positive.")
    prepared = _prepare_logo_image(image)
    width = width_chars * pixels_per_column
    height = max(1, round(width_chars * pixels_per_row * cell_aspect_ratio / (prepared.width / prepared.height)))
    resized = prepared.resize((width, height), Image.Resampling.LANCZOS)
    padded_height = ((height + pixels_per_row - 1) // pixels_per_row) * pixels_per_row
    canvas = Image.new("RGB", (width, padded_height), "white")
    canvas.paste(resized, (0, 0))
    return canvas


def render_logo_braille(
    image: Image.Image, *, width_chars: int = 56, dark: bool = True, cell_aspect_ratio: float = 0.5,
) -> str:
    """Sample the original artwork at eight dots per cell on its white background."""
    resized = _resize_for_cells(
        image, width_chars, pixels_per_column=2, pixels_per_row=4, cell_aspect_ratio=cell_aspect_ratio,
    )
    pixels = resized.load()
    lines: list[str] = []
    for row in range(0, resized.height, 4):
        parts: list[str] = []
        for column in range(width_chars):
            bits = 0
            colors: list[tuple[int, int, int]] = []
            accent: list[tuple[int, int, int]] = []
            for dy, dx, bit in _BRAILLE_DOTS:
                color = pixels[column * 2 + dx, row + dy]
                red, green, blue = color
                if min(color) >= 225:
                    continue
                bits |= 1 << bit
                colors.append(color)
                if red > green + 20 and red > blue + 20:
                    accent.append(color)
            if not bits:
                parts.append("[on #ffffff] [/]")
                continue
            visible_colors = accent or colors
            average = tuple(sum(c[channel] for c in visible_colors) // len(visible_colors) for channel in range(3))
            parts.append(f"[{_rgb_to_hex(*average)} on #ffffff]{chr(0x2800 + bits)}[/]")
        lines.append("".join(parts))
    return "\n".join(lines)


def render_logo_halfblocks(
    image: Image.Image, *, width_chars: int = 40, dark: bool = True,
    background: str | None = None, cell_aspect_ratio: float = 0.5,
) -> str:
    """Render sampled source RGB pixels without palette substitution."""
    resized = _resize_for_cells(
        image, width_chars, pixels_per_column=1, pixels_per_row=2, cell_aspect_ratio=cell_aspect_ratio,
    )
    pixels = resized.load()
    return "\n".join(
        "".join(
            f"[{_rgb_to_hex(*pixels[column, row])} on {_rgb_to_hex(*pixels[column, row + 1])}]▀[/]"
            for column in range(width_chars)
        )
        for row in range(0, resized.height, 2)
    )


@lru_cache(maxsize=8)
def render_brand_logo(
    *, width_chars: int = 56, dark: bool = True, background: str | None = None,
    cell_aspect_ratio: float = 0.5,
) -> str:
    """Render the complete official image for terminals with text-only output."""
    return render_logo_braille(
        brand_logo_image(), width_chars=width_chars, dark=False, cell_aspect_ratio=cell_aspect_ratio,
    )


def render_fallback_header(*, width_chars: int = 56) -> str:
    """Return the brand header for the CLI's fallback mode."""
    try:
        logo = render_brand_logo(width_chars=width_chars)
        return f"\n{logo}\n\n        {PRODUCT_NAME} · {PRODUCT_TAGLINE}\n"
    except Exception:
        return "\n".join([BRAND_NAME, PRODUCT_NAME, PRODUCT_TAGLINE, ""])

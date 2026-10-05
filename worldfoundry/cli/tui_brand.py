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

# Use the long-established block elements supported by ordinary terminal fonts.
# Each mask covers an 8 x 8 sample of one character cell.
_BLOCK_MASKS = tuple(
    (char, tuple(y * 8 + x for y in range(8) for x in range(8) if predicate(x, y)))
    for char, predicate in (
        *((char, lambda x, y, n=n: y >= 8 - n) for n, char in enumerate("▁▂▃▄▅▆▇", 1)),
        *((char, lambda x, y, n=n: x < n) for n, char in enumerate("▏▎▍▌▋▊▉", 1)),
        *((char, lambda x, y, bits=bits: bits & (1 << ((y // 4) * 2 + x // 4)))
          for bits, char in enumerate(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█") if bits not in (0, 15)),
    )
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


def _fit_block(colors: list[tuple[int, int, int]]) -> str:
    """Choose a glyph and two source-derived colors with the least RGB error."""
    total = tuple(sum(color[channel] for color in colors) for channel in range(3))
    if all(color == colors[0] for color in colors):
        return f"[on {_rgb_to_hex(*colors[0])}] [/]"

    best_score = -1.0
    best_glyph = " "
    best_foreground = best_background = colors[0]
    for glyph, mask in _BLOCK_MASKS:
        count = len(mask)
        foreground = tuple(sum(colors[index][channel] for index in mask) for channel in range(3))
        background = tuple(total[channel] - foreground[channel] for channel in range(3))
        # The omitted squared-pixel term is constant for every glyph in this cell.
        score = sum(foreground[channel] ** 2 / count + background[channel] ** 2 / (64 - count)
                    for channel in range(3))
        if score > best_score:
            best_score = score
            best_glyph = glyph
            best_foreground = tuple(round(value / count) for value in foreground)
            best_background = tuple(round(value / (64 - count)) for value in background)
    return f"[{_rgb_to_hex(*best_foreground)} on {_rgb_to_hex(*best_background)}]{best_glyph}[/]"


def render_logo_blocks(
    image: Image.Image, *, width_chars: int = 56, cell_aspect_ratio: float = 0.5,
) -> str:
    """Approximate the original bitmap with continuous blocks and sampled colors.

    Fractional blocks retain fine edges; quadrants retain the wing separations
    and lettering. Both foreground and background come from the source image.
    """
    resized = _resize_for_cells(
        image, width_chars, pixels_per_column=8, pixels_per_row=8, cell_aspect_ratio=cell_aspect_ratio,
    )
    pixels = resized.load()
    return "\n".join(
        "".join(
            _fit_block([pixels[column + dx, row + dy] for dy in range(8) for dx in range(8)])
            for column in range(0, resized.width, 8)
        )
        for row in range(0, resized.height, 8)
    )


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
    return render_logo_blocks(
        brand_logo_image(), width_chars=width_chars, cell_aspect_ratio=cell_aspect_ratio,
    )


def render_fallback_header(*, width_chars: int = 56) -> str:
    """Return the brand header for the CLI's fallback mode."""
    try:
        logo = render_brand_logo(width_chars=width_chars)
        return f"\n{logo}\n\n        {PRODUCT_NAME} · {PRODUCT_TAGLINE}\n"
    except Exception:
        return "\n".join([BRAND_NAME, PRODUCT_NAME, PRODUCT_TAGLINE, ""])

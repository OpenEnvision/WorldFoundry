"""OpenEnvision brand rendering for the WorldFoundry TUI.

Renders the official swan mark with solid terminal half-blocks and a native
text wordmark. Neutral colors follow the terminal theme; the beak stays red.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageColor
except ImportError:
    Image = None
    np = None

# ── Brand constants ───────────────────────────────────────────────
BRAND_NAME = "OpenEnvision"
PRODUCT_NAME = "WorldFoundry"
PRODUCT_TAGLINE = "Model · Benchmark · Studio"
LOGO_BACKGROUND = "#ffffff"

# ── Braille dot encoding ──────────────────────────────────────────
# Braille dot positions as (row, col, bit_index) triples for 2×4 dot rendering.
_BRAILLE_DOTS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3),
    (1, 1, 4), (2, 1, 5), (3, 0, 6), (3, 1, 7),
)


# ── Asset discovery ────────────────────────────────────────────────

def logo_asset_path() -> Path:
    """Locate the OpenEnvision logo PNG asset from the package resources directory.

    Falls back to a sibling ``assets/`` directory if package resources are unavailable.
    """
    try:
        asset = resources.files("worldfoundry.cli.assets").joinpath("openenvision_logo.png")
        with resources.as_file(asset) as path:
            return Path(path)
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return Path(__file__).with_name("assets") / "openenvision_logo.png"


# ── Pixel helpers ──────────────────────────────────────────────────

def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    """Convert RGB channel values to a ``#rrggbb`` hex colour string."""
    return f"#{red:02x}{green:02x}{blue:02x}"


def _prepare_logo_image(image: Image.Image, *, mark_only: bool = False) -> Image.Image:
    """Crop the logo image to its non-white bounding box with padding.

    Composite transparency onto white before finding the foreground. The
    packaged swan and wordmark are separated by blank rows, so ``mark_only``
    retains the first foreground band for a readable native-text wordmark.
    """
    rgba = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
    pixels = np.array(rgba)
    mask = np.any(pixels[:, :, :3] < 240, axis=2)
    occupied_rows = np.flatnonzero(mask.any(axis=1))
    if mark_only and occupied_rows.size:
        first_row = int(occupied_rows[0])
        blank_rows = np.flatnonzero(~mask[first_row:].any(axis=1))
        if blank_rows.size:
            mask[first_row + int(blank_rows[0]):] = False
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return rgba

    padding = 6
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, rgba.width)
    bottom = min(int(ys.max()) + padding + 1, rgba.height)
    return rgba.crop((left, top, right, bottom))


# ── Braille rendering ──────────────────────────────────────────────

def render_logo_braille(image: Image.Image, *, width_chars: int = 56, dark: bool = True) -> str:
    """Render the logo using Braille characters for highest terminal resolution.

    Each Braille character covers a 2×4 pixel block, yielding much finer
    detail than half-block rendering. Dark-mode terminals benefit from
    automatic colour inversion — black/dark-grey shades are flipped to
    white/light-grey while the red accent is preserved.

    Args:
        image: Source logo image (typically RGBA PNG).
        width_chars: Target width in terminal character columns.
        dark: Invert neutral colors for dark backgrounds only.

    Returns:
        Rich-markup string with per-cell ``[color]`` annotations.

    Raises:
        RuntimeError: When Pillow is not installed.
    """
    if Image is None or np is None:
        raise RuntimeError("Pillow is required to render the OpenEnvision logo.")

    # ── Prepare and resize ──
    prepared = _prepare_logo_image(image)
    source_width, source_height = prepared.size
    
    # 1 Braille char = 2 pixels wide, 4 pixels high
    pixel_width = width_chars * 2
    pixel_height = int(source_height * (pixel_width / source_width))
    remainder = pixel_height % 4
    if remainder:
        pixel_height += 4 - remainder

    resized = prepared.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
    pixels = np.array(resized.convert("RGBA"))
    
    char_height = pixel_height // 4
        
    lines: list[str] = []

    # ── Iterate character cells ──
    for char_row in range(char_height):
        row_y = char_row * 4
        parts: list[str] = []
        for char_col in range(width_chars):
            col_x = char_col * 2
            # NOTE: Guard against rounding errors near edges
            if row_y >= pixels.shape[0] or col_x >= pixels.shape[1]:
                parts.append(" ")
                continue
                
            block = pixels[row_y : min(row_y + 4, pixels.shape[0]), 
                           col_x : min(col_x + 2, pixels.shape[1])]

            # ── Compute Braille bits and colour ──
            bits = 0
            colors: list[tuple[int, int, int]] = []
            for py, px, bit in _BRAILLE_DOTS:
                if py >= block.shape[0] or px >= block.shape[1]:
                    continue
                r, g, b, a = block[py, px]
                # Skip transparent or near-white pixels
                if a < 20 or (r > 240 and g > 240 and b > 240):
                    continue
                
                bits |= 1 << bit
                # NOTE: Dark-mode inversion — keep the red accent, invert everything else
                if int(r) > int(g) + 20 and int(r) > int(b) + 20:
                    colors.append((r, g, b))
                elif dark:
                    colors.append((255 - r, 255 - g, 255 - b))
                else:
                    colors.append((r, g, b))

            if bits == 0:
                parts.append(" ")
                continue

            # ── Average colour and emit Rich-markup cell ──
            avg_r = sum(int(c[0]) for c in colors) // len(colors)
            avg_g = sum(int(c[1]) for c in colors) // len(colors)
            avg_b = sum(int(c[2]) for c in colors) // len(colors)
            
            hex_color = _rgb_to_hex(avg_r, avg_g, avg_b)
            char = chr(0x2800 + bits)
            parts.append(f"[{hex_color}]{char}[/]")

        lines.append("".join(parts))
    return "\n".join(lines)


# ── Brand rendering ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _brand_mark_image() -> Image.Image:
    if Image is None or np is None:
        raise RuntimeError("Pillow and NumPy are required to render the OpenEnvision logo.")
    path = logo_asset_path()
    if not path.is_file():
        raise FileNotFoundError(f"OpenEnvision logo asset not found: {path}")
    with Image.open(path) as image:
        return _prepare_logo_image(image, mark_only=True)


def brand_logo_aspect_ratio() -> float:
    """Return the cropped swan's aspect ratio for terminal layout."""
    image = _brand_mark_image()
    return image.width / image.height


def render_logo_halfblocks(
    image: Image.Image,
    *,
    width_chars: int = 40,
    dark: bool = True,
    background: str | None = None,
) -> str:
    """Render two square image pixels per terminal cell with continuous fills."""
    if Image is None or np is None:
        raise RuntimeError("Pillow and NumPy are required to render the OpenEnvision logo.")
    if width_chars < 1:
        raise ValueError("Logo width must be positive.")
    prepared = _prepare_logo_image(image)
    pixel_height = max(1, round(prepared.height * width_chars / prepared.width))
    resized = prepared.resize((width_chars, pixel_height), Image.Resampling.LANCZOS)
    if pixel_height % 2:
        canvas = Image.new("RGBA", (width_chars, pixel_height + 1), "white")
        canvas.paste(resized, (0, 0))
        resized = canvas

    pixels = np.asarray(resized.convert("RGB"), dtype=np.float32)
    red = (pixels[:, :, 0] > pixels[:, :, 1] + 20) & (pixels[:, :, 0] > pixels[:, :, 2] + 20)
    coverage = 1 - pixels.mean(axis=2) / 255
    coverage[red] = 1 - np.minimum(pixels[:, :, 1], pixels[:, :, 2])[red] / 255
    coverage[coverage < 0.025] = 0

    panel_color = background or ("#1b1e24" if dark else "#eef1eb")
    panel = np.asarray(ImageColor.getrgb(panel_color), dtype=np.float32)
    panel_hex = _rgb_to_hex(*(int(channel) for channel in panel))
    ink = np.empty_like(pixels)
    ink[:] = (230, 232, 236) if dark else (16, 18, 20)
    ink[red] = (240, 52, 62) if dark else (217, 35, 45)
    colors = np.rint(panel + coverage[:, :, None] * (ink - panel)).astype(np.uint8)

    lines: list[str] = []
    for row in range(0, colors.shape[0], 2):
        parts: list[str] = []
        for column in range(width_chars):
            upper = _rgb_to_hex(*colors[row, column])
            lower = _rgb_to_hex(*colors[row + 1, column])
            if upper == lower == panel_hex:
                parts.append(" ")
            else:
                parts.append(f"[{upper} on {lower}]▀[/]")
        lines.append("".join(parts))
    return "\n".join(lines)


@lru_cache(maxsize=8)
def render_brand_logo(*, width_chars: int = 56, dark: bool = True, background: str | None = None) -> str:
    """Render the swan with half-blocks and the brand name as terminal text.

    Caches up to 8 width and theme variants. Reads the PNG asset via :func:`logo_asset_path`.

    Raises:
        FileNotFoundError: When the logo asset PNG is not found.
        RuntimeError: When Pillow is not installed.
    """
    mark = render_logo_halfblocks(_brand_mark_image(), width_chars=width_chars, dark=dark, background=background)
    if width_chars < len(BRAND_NAME):
        return mark
    text_color = "#e6e8ec" if dark else "#101214"
    return f"{mark}\n\n[bold {text_color}]{BRAND_NAME.center(width_chars)}[/]"


def render_fallback_header(*, width_chars: int = 56) -> str:
    """Return a plain-text header for the ``--fallback`` mode (no Textual dependency).

    Falls back to simple text lines if logo rendering fails.
    """
    try:
        logo = render_brand_logo(width_chars=width_chars)
        return f"\n{logo}\n\n        {PRODUCT_NAME} · {PRODUCT_TAGLINE}\n"
    except Exception:
        return "\n".join([BRAND_NAME, PRODUCT_NAME, PRODUCT_TAGLINE, ""])

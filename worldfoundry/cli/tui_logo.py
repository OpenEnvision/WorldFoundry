"""Display the official logo using terminal graphics when available."""

from __future__ import annotations

from math import ceil

from PIL import Image
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from .tui_brand import brand_logo_image, render_brand_logo

try:
    from textual_image._terminal import get_cell_size
    from textual_image.renderable import Image as AutoRenderable
    from textual_image.renderable.sixel import Image as SixelRenderable
    from textual_image.renderable.tgp import Image as TGPRenderable
    from textual_image.widget import Image as TerminalImage
except ImportError:
    TerminalImage = None


def logo_cell_aspect_ratio() -> float:
    """Use the terminal's measured cell dimensions to preserve image proportions."""
    if TerminalImage is not None:
        size = get_cell_size()
        if size.width > 0 and size.height > 0:
            return size.width / size.height
    return 0.5


class BrandLogo(Vertical):
    """Show the original bitmap, with a continuous block fallback for text-only terminals."""

    DEFAULT_CSS = """
    BrandLogo {
        height: auto;
        align: center middle;
    }
    BrandLogo > .logo-image {
        height: auto;
        width: auto;
        background: #ffffff;
    }
    """

    def __init__(self, *, width_chars: int, id: str | None = None) -> None:
        super().__init__(id=id)
        self._render_size: tuple[int, float] | None = None
        self.native_image = TerminalImage is not None and AutoRenderable in (SixelRenderable, TGPRenderable)
        if self.native_image:
            self._image = TerminalImage(classes="logo-image")
        else:
            self._image = Static("", markup=True, classes="logo-image")
        self.set_logo_width(width_chars)

    def compose(self) -> ComposeResult:
        yield self._image

    def on_unmount(self) -> None:
        if self.native_image:
            self._image.image = None

    def set_logo_width(self, width_chars: int) -> None:
        width_chars = max(1, width_chars)
        aspect = logo_cell_aspect_ratio()
        render_size = (width_chars, aspect)
        if render_size == self._render_size:
            return
        self._render_size = render_size
        self._image.styles.width = width_chars
        if self.native_image:
            cell = get_cell_size()
            source = brand_logo_image()
            pixel_width = width_chars * cell.width
            pixel_height = max(1, round(pixel_width * source.height / source.width))
            rows = ceil(pixel_height / cell.height)
            frame = Image.new("RGB", (pixel_width, rows * cell.height), "white")
            frame.paste(
                source.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS),
                (0, (frame.height - pixel_height) // 2),
            )
            self._image.image = frame
            self._image.styles.height = rows
        else:
            self._image.update(render_brand_logo(
                width_chars=width_chars, cell_aspect_ratio=aspect,
            ))

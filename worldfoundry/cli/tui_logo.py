"""Display the official logo using terminal graphics when available."""

from __future__ import annotations

from math import ceil

from PIL import Image
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from .tui_brand import brand_logo_aspect_ratio, brand_logo_image, render_brand_logo

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


class BrandLogo(Vertical, can_focus=True):
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

    BINDINGS = [("enter", "enlarge", "Enlarge logo")]

    def __init__(self, *, width_chars: int, id: str | None = None, zoomable: bool = True) -> None:
        super().__init__(id=id)
        self.can_focus = zoomable
        self.zoomable = zoomable
        if zoomable:
            self.tooltip = "Click or press Enter to enlarge"
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

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_enlarge()

    def action_enlarge(self) -> None:
        if self.zoomable:
            self.app.push_screen(LogoPreview())

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


class LogoPreview(ModalScreen[None]):
    """Convert the original image again at the largest size the screen can hold."""

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    DEFAULT_CSS = """
    LogoPreview {
        align: center middle;
        background: $background 90%;
    }
    LogoPreview > #logo-preview-panel {
        width: auto;
        height: auto;
        border: round $border;
        padding: 1 2;
        background: #ffffff;
    }
    LogoPreview #logo-preview-hint {
        height: 1;
        margin-top: 1;
        color: #666666;
        text-align: center;
    }
    """

    def _logo_width(self) -> int:
        rows = max(1, self.app.size.height - 8)
        columns = max(1, self.app.size.width - 8)
        return max(1, min(columns, int(rows * brand_logo_aspect_ratio() / logo_cell_aspect_ratio())))

    def compose(self) -> ComposeResult:
        width = self._logo_width()
        with Vertical(id="logo-preview-panel"):
            logo = BrandLogo(width_chars=width, id="logo-preview-image", zoomable=False)
            logo.styles.width = width
            yield logo
            yield Static("Esc to return", id="logo-preview-hint")

    def on_resize(self) -> None:
        self.call_after_refresh(self._resize_logo)

    def _resize_logo(self) -> None:
        width = self._logo_width()
        logo = self.query_one("#logo-preview-image", BrandLogo)
        logo.styles.width = width
        logo.set_logo_width(width)

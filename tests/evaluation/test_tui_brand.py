"""Contracts for the official bitmap and terminal graphics output."""

from __future__ import annotations

import asyncio
import base64
import io

import pytest
from rich.console import Console
from rich.text import Text

pytest.importorskip("PIL")
from PIL import Image, ImageOps

from worldfoundry.cli.tui_brand import (
    brand_logo_image,
    logo_asset_path,
    render_brand_logo,
    render_logo_halfblocks,
)


def test_brand_retains_official_swan_and_wordmark_pixels() -> None:
    with Image.open(logo_asset_path()) as source:
        expected = source.convert("RGB").crop((59, 83, 407, 355))
    actual = brand_logo_image()
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_text_fallback_keeps_official_colors_across_themes() -> None:
    dark = render_brand_logo(width_chars=33, dark=True, background="#000000")
    light = render_brand_logo(width_chars=33, dark=False, background="#ffffff")
    assert dark == light
    rendered = Text.from_markup(dark)
    assert all(len(line) == 33 for line in rendered.plain.splitlines())
    console = Console()
    styles = [console.get_style(span.style) for span in rendered.spans]
    colors = [style.color.get_truecolor() for style in styles if style.color]
    assert any(red > green + 20 and red > blue + 20 for red, green, blue in colors)

    solid = Text.from_markup(render_logo_halfblocks(Image.new("RGB", (8, 8), (23, 55, 88)), width_chars=4))
    style = console.get_style(solid.spans[0].style)
    assert style.color.get_truecolor() == (23, 55, 88)
    assert style.bgcolor.get_truecolor() == (23, 55, 88)


@pytest.mark.parametrize("backend", ["tgp", "sixel"])
def test_native_output_preserves_original_bitmap_when_resized(monkeypatch, backend: str) -> None:
    pytest.importorskip("textual_image")
    from textual.app import App, ComposeResult
    from textual_image.renderable import sixel, tgp
    from textual_image.widget import SixelImage, TGPImage
    from textual_image.widget.sixel import _ImageSixelImpl

    from worldfoundry.cli import tui_logo

    packets: list[dict] = []
    encoded: list[Image.Image] = []
    if backend == "tgp":
        monkeypatch.setattr(tui_logo, "AutoRenderable", tgp.Image)
        monkeypatch.setattr(tui_logo, "TerminalImage", TGPImage)
        monkeypatch.setattr(tgp, "_send_tgp_message", lambda **packet: packets.append(packet))
    else:
        monkeypatch.setattr(tui_logo, "AutoRenderable", sixel.Image)
        monkeypatch.setattr(tui_logo, "TerminalImage", SixelImage)
        original_encoder = _ImageSixelImpl._image_to_sixels

        def capture(self, image, *args, **kwargs):
            encoded.append(image.copy())
            return original_encoder(self, image, *args, **kwargs)

        monkeypatch.setattr(_ImageSixelImpl, "_image_to_sixels", capture)

    class LogoApp(App):
        CSS = "Screen { align: center middle; } BrandLogo { width: 40; }"

        def compose(self) -> ComposeResult:
            yield tui_logo.BrandLogo(width_chars=36, id="brand")

    async def exercise() -> None:
        app = LogoApp()
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            logo = app.query_one("#brand", tui_logo.BrandLogo)
            assert logo.native_image
            assert logo._image.content_size.width == 36
            logo.set_logo_width(24)
            await pilot.pause()
            assert logo._image.content_size.width == 24
            logo.display = False
            await pilot.pause()
            logo.display = True
            await pilot.pause()

    asyncio.run(exercise())
    if backend == "tgp":
        chunks: dict[int, list[str]] = {}
        for packet in packets:
            if packet.get("f") != 100:
                continue
            chunks.setdefault(packet["i"], []).append(packet["payload"])
            if packet["m"] == 0:
                encoded.append(Image.open(io.BytesIO(base64.b64decode("".join(chunks.pop(packet["i"]))))).convert("RGB"))
        assert any(packet.get("a") == "d" for packet in packets)
    assert {image.width for image in encoded} == {240, 360}
    with Image.open(logo_asset_path()) as source:
        artwork = source.convert("RGB").crop((59, 83, 407, 355))
    for image in encoded:
        expected = ImageOps.contain(artwork, image.size, method=Image.Resampling.LANCZOS)
        top = (image.height - expected.height) // 2
        assert image.crop((0, top, expected.width, top + expected.height)).tobytes() == expected.tobytes()

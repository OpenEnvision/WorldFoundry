from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote

import httpx
from PIL import Image

from .assets import STUDIO_ASSET_DIR
from .status import status_block as _status_block
from .urls import file_url as _file_url

STUDIO_LOGO_PATH = STUDIO_ASSET_DIR / "openenvision-logo.png"
DEMO_IMAGE_LIBRARY_ROOT = Path(__file__).resolve().parents[2] / "data" / "test_cases" / "studio_demo"
TRAY_DEMO_IMAGE_COUNT = 9


def _extract_uploaded_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "path", "file"):
            raw = value.get(key)
            if isinstance(raw, str):
                return raw
        return None
    raw_name = getattr(value, "name", None)
    return raw_name if isinstance(raw_name, str) else None


def _logo_nav_markup() -> str:
    """Navbar logo markup (data URI avoids /file routing gaps in nested embeds)."""

    if not STUDIO_LOGO_PATH.exists():
        return '<span class="wa-site-brand-fallback">WA</span>'
    raw = STUDIO_LOGO_PATH.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return (
        f'<img class="wa-site-brand-img" src="data:image/png;base64,{b64}" '
        f'width="28" height="28" decoding="sync" fetchpriority="high" alt="">'
    )


def _default_tray_demo_images(
    slot_count: int = TRAY_DEMO_IMAGE_COUNT,
    demo_image_paths: Sequence[Path] | None = None,
) -> tuple[Path, ...]:
    """Map example images onto the fixed tray slots.

    Args:
        slot_count: Number of tray thumbnail slots to populate.
        demo_image_paths: Optional explicit image paths for deterministic tests or overrides.
    """

    image_paths = tuple(DEMO_IMAGE_LIBRARY_FILES if demo_image_paths is None else demo_image_paths)
    if not image_paths:
        return ()
    return tuple(image_paths[index % len(image_paths)] for index in range(slot_count))


def _default_tray_head_html(demo_image_paths: Sequence[Path] | None = None) -> str:
    """Build a `<style>` block that points tray defaults at local example thumbnails.

    Args:
        demo_image_paths: Optional explicit image paths for deterministic tests or overrides.
    """

    tray_paths = _default_tray_demo_images(demo_image_paths=demo_image_paths)
    if not tray_paths:
        return ""
    css_lines = ["<style>", ":root {"]
    for index, path in enumerate(tray_paths, start=1):
        image_url = _file_url(str(path))
        if image_url:
            css_lines.append(f'  --wa-scene-thumb-{index}: url("{escape(image_url, quote=True)}");')
    css_lines.extend(["}", "</style>"])
    return "\n".join(css_lines)


def _world_tray_html(demo_image_paths: Sequence[Path] | None = None) -> str:
    """Build the bottom tray with deterministic default image sources.

    Args:
        demo_image_paths: Optional explicit image paths for deterministic tests or overrides.
    """

    tray_paths = _default_tray_demo_images(demo_image_paths=demo_image_paths)
    tray_sources = [_file_url(str(path)) for path in tray_paths]
    tray_sources.extend([""] * max(0, TRAY_DEMO_IMAGE_COUNT - len(tray_sources)))
    items = (
        ("Preview Video", "video", "Video", "wa-tray-thumb-video", tray_sources[0]),
        ("Preview Image", "image", "Image", "wa-tray-thumb-image", tray_sources[1]),
        ("3D World", "world", "3DGS", "wa-tray-thumb-3d", tray_sources[2]),
        ("Point Cloud (Viser)", "points", "Points", "wa-tray-thumb-points", ""),
        ("Gallery", "gallery-0", "Shot A", "wa-tray-thumb-gallery", tray_sources[3]),
        ("Gallery", "gallery-1", "Shot B", "wa-tray-thumb-gallery", tray_sources[4]),
        ("Gallery", "gallery-2", "Shot C", "wa-tray-thumb-gallery", tray_sources[5]),
        ("Gallery", "gallery-3", "Shot D", "wa-tray-thumb-gallery", tray_sources[6]),
        ("Embodied Sim", "embodied", "Sim", "wa-tray-thumb-embodied", tray_sources[7]),
        ("Artifacts", "artifacts", "Files", "wa-tray-thumb-artifacts", tray_sources[8]),
    )
    buttons: list[str] = ['<div class="wa-world-tray">']
    for index, (tab_target, thumb_source, label, thumb_class, input_source) in enumerate(items):
        active_class = " is-active" if index == 0 else ""
        source_attr = escape(input_source, quote=True)
        tab_target_attr = escape(tab_target, quote=True)
        thumb_source_attr = escape(thumb_source, quote=True)
        buttons.append(
            f'  <button class="wa-tray-item{active_class}" '
            f'data-tab-target="{tab_target_attr}" '
            f'data-thumb-source="{thumb_source_attr}" '
            f'data-input-source="{source_attr}" type="button" '
            f'onclick="return window.waHandleTrayClick(this);">'
        )
        buttons.append(f'    <span class="wa-tray-thumb {thumb_class}"></span>')
        buttons.append(f'    <span class="wa-tray-label">{escape(label)}</span>')
        buttons.append("  </button>")
    buttons.append("</div>")
    return "\n".join(buttons)


def _tray_gallery_items(demo_image_paths: Sequence[Path] | None = None) -> list[tuple[str, str]]:
    """Build the bottom input tray items from example images.

    Args:
        demo_image_paths: Optional explicit image paths for deterministic tests or overrides.
    """

    return [(str(path), "") for path in _default_tray_demo_images(demo_image_paths=demo_image_paths)]


def _load_tray_image_source(image_source: str) -> Image.Image | None:
    """Load a tray thumbnail source into a PIL image.

    Args:
        image_source: Local Gradio file URL, plain path, or remote image URL from the tray.
    """

    source = (image_source or "").strip()
    if not source:
        return None
    if source.startswith("/gradio_api/file="):
        local_path = Path(unquote(source[len("/gradio_api/file="):])).expanduser().resolve()
        if not local_path.exists():
            return None
        return Image.open(local_path).convert("RGB")
    if source.startswith("/file="):
        local_path = Path(unquote(source[len("/file="):])).expanduser().resolve()
        if not local_path.exists():
            return None
        return Image.open(local_path).convert("RGB")
    if source.startswith(("http://", "https://")):
        response = httpx.get(source, timeout=20.0, follow_redirects=True)
        if response.status_code >= 400:
            return None
        return Image.open(BytesIO(response.content)).convert("RGB")
    local_path = Path(unquote(source)).expanduser().resolve()
    if not local_path.exists():
        return None
    return Image.open(local_path).convert("RGB")


def _use_tray_image_as_input(image_source: str):
    """Promote a tray thumbnail into the main image input.

    Args:
        image_source: Resolved thumbnail source captured from the bottom tray.
    """

    image_value = _load_tray_image_source(image_source)
    if image_value is None:
        return (
            None,
            "",
            None,
            _status_block("select a tray image with a usable thumbnail first"),
        )
    return (
        image_value,
        "",
        None,
        _status_block("loaded tray image into Main Image"),
    )


def _demo_image_library_files(root: Path | None = None) -> tuple[Path, ...]:
    source_root = (root or DEMO_IMAGE_LIBRARY_ROOT).expanduser()
    if not source_root.exists():
        return ()
    return tuple(
        sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
    )


DEMO_IMAGE_LIBRARY_FILES = _demo_image_library_files()
TRAY_DEMO_IMAGE_FILES = _default_tray_demo_images(demo_image_paths=DEMO_IMAGE_LIBRARY_FILES)


def _demo_gallery_items() -> list[tuple[str, str]]:
    return [(str(path), path.parent.name) for path in DEMO_IMAGE_LIBRARY_FILES]


def _use_demo_image_as_input(index: Any, demo_image_paths: Sequence[Path] | None = None):
    image_paths = tuple(DEMO_IMAGE_LIBRARY_FILES if demo_image_paths is None else demo_image_paths)
    if not image_paths:
        return (
            None,
            "",
            None,
            _status_block("no example images are available"),
        )

    selected_index = index
    if isinstance(selected_index, (list, tuple)):
        if not selected_index:
            selected_index = None
        else:
            selected_index = selected_index[0]
    try:
        selected_index = int(selected_index)
    except (TypeError, ValueError):
        selected_index = None

    if selected_index is None or selected_index < 0 or selected_index >= len(image_paths):
        return (
            None,
            "",
            None,
            _status_block("select an example image first"),
        )

    path = image_paths[selected_index]
    image_value = _load_tray_image_source(str(path))
    if image_value is None:
        return (
            None,
            "",
            None,
            _status_block(f"could not load example image: {path.name}"),
        )

    return (
        image_value,
        "",
        None,
        _status_block(f"loaded example image into Main Image: {path.parent.name}/{path.name}"),
    )


def _on_demo_image_select(
    evt: Any,
    demo_image_paths: Sequence[Path] | None = None,
):
    return _use_demo_image_as_input(evt.index, demo_image_paths=demo_image_paths)


def _on_tray_image_select(evt: Any):
    return _on_demo_image_select(evt, demo_image_paths=TRAY_DEMO_IMAGE_FILES)

__all__ = [
    "DEMO_IMAGE_LIBRARY_FILES",
    "DEMO_IMAGE_LIBRARY_ROOT",
    "STUDIO_ASSET_DIR",
    "STUDIO_LOGO_PATH",
    "TRAY_DEMO_IMAGE_COUNT",
    "TRAY_DEMO_IMAGE_FILES",
    "_default_tray_demo_images",
    "_default_tray_head_html",
    "_demo_gallery_items",
    "_demo_image_library_files",
    "_extract_uploaded_path",
    "_file_url",
    "_load_tray_image_source",
    "_logo_nav_markup",
    "_on_demo_image_select",
    "_on_tray_image_select",
    "_tray_gallery_items",
    "_use_demo_image_as_input",
    "_use_tray_image_as_input",
    "_world_tray_html",
]

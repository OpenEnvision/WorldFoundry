#!/usr/bin/env python3
"""Capture deterministic terminal-style screenshots from the public CLI.

The commands run against the current checkout through ``python -m
worldfoundry`` while the screenshot displays the installed console-script
spelling used in the docs. A pseudo-terminal supplies the same width that is
shown in the rendered window.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import re
import struct
import subprocess
import sys
import termios
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - developer convenience path
    raise SystemExit("Pillow is required: python -m pip install pillow") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "fumadocs" / "public" / "images" / "cli"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@dataclass(frozen=True)
class Capture:
    filename: str
    display_lines: tuple[str, ...]
    arguments: tuple[str, ...]
    columns: int = 120
    rows: int = 48


CAPTURES = (
    Capture(
        filename="worldfoundry-eval-help.png",
        display_lines=("$ worldfoundry-eval --help",),
        arguments=("--help",),
        columns=120,
    ),
    Capture(
        filename="model-readiness.png",
        display_lines=(
            "$ worldfoundry-eval zoo model-show \\",
            "    --manifest-dir worldfoundry/data/models/catalog \\",
            "    --model-id self-forcing",
        ),
        arguments=(
            "zoo",
            "model-show",
            "--manifest-dir",
            "worldfoundry/data/models/catalog",
            "--model-id",
            "self-forcing",
        ),
        columns=120,
    ),
)


def _capture_pty(arguments: tuple[str, ...], *, columns: int, rows: int) -> str:
    """Run one CLI command in a fixed-size pseudo-terminal and return output."""

    master_fd, slave_fd = pty.openpty()
    window = struct.pack("HHHH", rows, columns, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, window)

    environment = os.environ.copy()
    environment.update(
        {
            "COLUMNS": str(columns),
            "LINES": str(rows),
            "NO_COLOR": "1",
            "PYTHONIOENCODING": "utf-8",
            "TERM": "xterm-256color",
        }
    )
    environment.pop("FORCE_COLOR", None)

    process = subprocess.Popen(
        [sys.executable, "-m", "worldfoundry", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(master_fd, 65536)
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        chunks.append(chunk)
    os.close(master_fd)

    return_code = process.wait()
    output = b"".join(chunks).decode("utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(
            f"worldfoundry command exited {return_code}: {' '.join(arguments)}\n{output}"
        )
    return output


def _clean_output(output: str) -> list[str]:
    """Normalize control sequences without changing visible CLI content."""

    plain = ANSI_ESCAPE.sub("", output).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in plain.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _wrap_terminal_line(line: str, columns: int) -> list[str]:
    """Apply terminal-style hard wrapping while preserving table spacing."""

    if not line:
        return [""]
    return [line[offset : offset + columns] for offset in range(0, len(line), columns)]


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.is_file():
        raise FileNotFoundError(f"terminal screenshot font not found: {path}")
    return ImageFont.truetype(str(path), size=size)


def _render(capture: Capture, output: str) -> Image.Image:
    """Render command and captured output inside a terminal window."""

    scale = 2
    font_size = 15 * scale
    regular = _font(FONT_REGULAR, font_size)
    bold = _font(FONT_BOLD, font_size)
    cell_width = int(round(regular.getlength("M")))
    line_height = 22 * scale
    side_padding = 24 * scale
    top_bar_height = 42 * scale
    content_padding_y = 18 * scale

    command_lines = list(capture.display_lines)
    output_lines: list[str] = []
    for line in _clean_output(output):
        output_lines.extend(_wrap_terminal_line(line, capture.columns))
    lines = [*command_lines, "", *output_lines]

    width = side_padding * 2 + capture.columns * cell_width
    content_height = content_padding_y * 2 + len(lines) * line_height
    height = top_bar_height + content_height

    image = Image.new("RGB", (width, height), "#070a0f")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (2 * scale, 2 * scale, width - 2 * scale, height - 2 * scale),
        radius=14 * scale,
        fill="#0d1117",
        outline="#30363d",
        width=scale,
    )
    draw.rounded_rectangle(
        (2 * scale, 2 * scale, width - 2 * scale, top_bar_height + 8 * scale),
        radius=14 * scale,
        fill="#161b22",
    )
    draw.rectangle(
        (2 * scale, top_bar_height - 8 * scale, width - 2 * scale, top_bar_height),
        fill="#161b22",
    )
    draw.line(
        (2 * scale, top_bar_height, width - 2 * scale, top_bar_height),
        fill="#30363d",
        width=scale,
    )

    dot_y = 21 * scale
    for dot_x, color in ((20, "#ff5f57"), (40, "#febc2e"), (60, "#28c840")):
        draw.ellipse(
            (
                (dot_x - 6) * scale,
                dot_y - 6 * scale,
                (dot_x + 6) * scale,
                dot_y + 6 * scale,
            ),
            fill=color,
        )

    title = "WorldFoundry CLI"
    title_width = draw.textlength(title, font=bold)
    draw.text(
        ((width - title_width) / 2, 11 * scale),
        title,
        font=bold,
        fill="#8b949e",
    )

    origin_x = side_padding
    origin_y = top_bar_height + content_padding_y
    for index, line in enumerate(lines):
        y = origin_y + index * line_height
        if index < len(command_lines):
            if line.startswith("$ "):
                draw.text((origin_x, y), "$", font=bold, fill="#7ee787")
                draw.text(
                    (origin_x + 2 * cell_width, y),
                    line[2:],
                    font=bold,
                    fill="#f0f6fc",
                )
            else:
                draw.text((origin_x, y), line, font=bold, fill="#f0f6fc")
            continue

        color = "#c9d1d9"
        active_font = regular
        stripped = line.strip()
        if stripped in {"positional arguments:", "options:", "Command areas:"}:
            color = "#d2a8ff"
            active_font = bold
        elif line.startswith("usage:"):
            color = "#79c0ff"
            active_font = bold
        elif re.match(r"^[a-z_]+:", line):
            key, separator, value = line.partition(":")
            draw.text((origin_x, y), key + separator, font=bold, fill="#79c0ff")
            draw.text(
                (origin_x + len(key + separator) * cell_width, y),
                value,
                font=regular,
                fill="#c9d1d9",
            )
            continue
        draw.text((origin_x, y), line, font=active_font, fill=color)

    return image


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for capture in CAPTURES:
        output = _capture_pty(capture.arguments, columns=capture.columns, rows=capture.rows)
        image = _render(capture, output)
        destination = OUTPUT_DIR / capture.filename
        image.save(destination, format="PNG", optimize=True)
        print(f"wrote {destination.relative_to(REPO_ROOT)} ({image.width}x{image.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared, dependency-free terminal presentation for every CLI command family.

JSON is owned by the handlers. This module renders human-readable responses,
with explicit plain-output fallbacks for pipes and terminals without Unicode.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO

RESET = "\033[0m"
ACCENT = "\033[38;5;100m"
BOLD = "\033[1m"
MUTED = "\033[2m"
SUCCESS = "\033[32m"
WARNING = "\033[33m"
ERROR = "\033[31m"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def color_enabled(stream: TextIO | None = None) -> bool:
    stream = sys.stdout if stream is None else stream
    if os.environ.get("NO_COLOR") is not None:
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force.lower() not in {"", "0", "false", "no"}
    return os.environ.get("TERM") != "dumb" and bool(stream.isatty())


def terminal_enabled(stream: TextIO | None = None) -> bool:
    stream = sys.stdout if stream is None else stream
    try:
        "╭─╮│╰╯".encode(getattr(stream, "encoding", None) or "utf-8")
    except (LookupError, UnicodeEncodeError):
        return False
    return color_enabled(stream) or (os.environ.get("TERM") != "dumb" and bool(stream.isatty()))


def paint(text: str, style: str, stream: TextIO | None = None) -> str:
    return f"{style}{text}{RESET}" if text and style and color_enabled(stream) else text


def terminal_width() -> int:
    return max(20, min(shutil.get_terminal_size((120, 24)).columns, 160))


@dataclass(frozen=True)
class Row:
    invocation: str
    description: str
    default: str | None = None


@dataclass(frozen=True)
class Panel:
    title: str
    rows: tuple[Row, ...]
    description: str = ""


def render_panels(panels: Sequence[Panel], terminal_width: int, *, min_column_width: int = 62) -> str:
    if not panels:
        return ""
    if terminal_width < 2 * min_column_width + 4 or len(panels) == 1:
        return "\n\n".join(_render_panel(panel, terminal_width) for panel in panels)

    gap = 2
    left_width = (terminal_width - gap) // 2
    right_width = terminal_width - gap - left_width
    columns: list[list[str]] = [[], []]
    heights = [0, 0]
    # Command catalogs can dwarf all other panels. Split at command boundaries
    # so both columns stay useful without hiding any entries.
    balanced_panels = []
    for panel in panels:
        if panel.title == "commands" and len(panel.rows) > 10:
            for start in range(0, len(panel.rows), 10):
                balanced_panels.append(
                    Panel(
                        "commands" if start == 0 else "commands (continued)",
                        panel.rows[start : start + 10],
                        panel.description if start == 0 else "",
                    )
                )
        else:
            balanced_panels.append(panel)
    for panel in balanced_panels:
        column_index = 0 if heights[0] <= heights[1] else 1
        panel_width = left_width if column_index == 0 else right_width
        panel_lines = _render_panel_lines(panel, panel_width)
        if columns[column_index]:
            columns[column_index].append(" " * panel_width)
            heights[column_index] += 1
        columns[column_index].extend(panel_lines)
        heights[column_index] += len(panel_lines)

    output: list[str] = []
    for row_index in range(max(heights)):
        left_line = columns[0][row_index] if row_index < heights[0] else " " * left_width
        right_line = columns[1][row_index] if row_index < heights[1] else ""
        output.append(left_line + " " * gap + right_line)
    return "\n".join(output)


def _render_panel(panel: Panel, width: int) -> str:
    return "\n".join(_render_panel_lines(panel, width))


def _render_panel_lines(panel: Panel, width: int) -> list[str]:
    inner_width = width - 4
    title = f" {panel.title} "
    # Long dotted group names move inside the box on small terminals.
    title_fits = len(title) <= width - 4
    top = f"╭─{title}{'─' * (width - len(title) - 3)}╮" if title_fits else f"╭{'─' * (width - 2)}╮"
    lines = [paint(top, ACCENT)]

    def append_line(text: str = "", style: str = "") -> None:
        lines.append(
            paint("│", ACCENT) + " " + paint(text, style) + " " * (inner_width - len(text)) + " " + paint("│", ACCENT)
        )

    if not title_fits:
        for title_line in textwrap.wrap(panel.title, width=inner_width, break_on_hyphens=False):
            append_line(title_line, ACCENT)

    for description_line in textwrap.wrap(
        panel.description,
        width=inner_width,
        break_long_words=True,
        break_on_hyphens=False,
    ):
        append_line(description_line)
    if panel.description or not title_fits:
        append_line("─" * inner_width, ACCENT)

    for index, row in enumerate(panel.rows):
        if index and panel.title.startswith("commands"):
            append_line()
        invocation_lines = textwrap.wrap(
            row.invocation,
            width=inner_width,
            subsequent_indent="  ",
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for invocation in invocation_lines:
            append_line(invocation, BOLD)
        for description, visible_length in _description_lines(row, inner_width - 2):
            lines.append(
                paint("│", ACCENT)
                + "   "
                + description
                + " " * (inner_width - 2 - visible_length)
                + " "
                + paint("│", ACCENT)
            )
    lines.append(paint(f"╰{'─' * (width - 2)}╯", ACCENT))
    return lines


def _description_lines(row: Row, width: int) -> list[tuple[str, int]]:
    description = " ".join(row.description.split())
    if row.default is not None:
        description = f"{description} (default: {row.default})".strip()
    default_start = description.find("(default:")
    offset = 0
    lines = []
    for line in textwrap.wrap(description, width=width, break_on_hyphens=False):
        offset = description.index(line, offset)
        split = len(line) if default_start < 0 else max(0, min(len(line), default_start - offset))
        lines.append((paint(line[:split], MUTED) + paint(line[split:], ACCENT), len(line)))
        offset += len(line)
    return lines


def _cell_width(text: str) -> int:
    return sum(
        0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        for char in _ANSI.sub("", text)
    )


def _wrap(text: object, width: int) -> list[str]:
    """Wrap paths and Unicode text by terminal cells, without ellipsizing data."""
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
        while paragraph:
            cells = 0
            end = 0
            for char in paragraph:
                cells += _cell_width(char)
                if cells > width:
                    break
                end += 1
            if end < len(paragraph):
                space = paragraph.rfind(" ", 0, end + 1)
                if space > 0:
                    end = space
            end = max(1, end)
            lines.append(paragraph[:end].rstrip())
            paragraph = paragraph[end:].lstrip()
    return lines


def _value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value)) or "-"
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={item}" for key, item in value.items()) or "-"
    return "-" if value is None or value == "" else str(value)


def _value_style(value: str) -> str:
    if value.lower() in {"true", "yes", "ready", "ok", "success", "completed", "passed"}:
        return SUCCESS
    if value.lower() in {"failed", "error", "invalid", "blocked"}:
        return ERROR
    if value.lower() in {"false", "no", "missing", "not ready", "planned", "unknown"}:
        return MUTED
    return ""


def _frame(title: str, lines: Sequence[str], width: int) -> str:
    inner = width - 4
    title_lines = _wrap(title, inner - 2)
    label = f" {title_lines[0]} "
    result = [paint(f"╭─{label}{'─' * (width - _cell_width(label) - 3)}╮", ACCENT)]
    for line in [*(paint(part, BOLD) for part in title_lines[1:]), *lines]:
        result.append(
            paint("│", ACCENT) + " " + line + " " * max(0, inner - _cell_width(line)) + " " + paint("│", ACCENT)
        )
    result.append(paint(f"╰{'─' * (width - 2)}╯", ACCENT))
    return "\n".join(result)


def _field_lines(fields: Mapping[str, object], width: int) -> list[str]:
    lines = []
    label_width = min(26, max((_cell_width(key.replace("_", " ")) for key in fields), default=0))
    for key, value in fields.items():
        label = key.replace("_", " ")
        text = _value(value)
        if width < 60:
            lines.extend(paint(line, MUTED) for line in _wrap(label, width))
            lines.extend("  " + paint(line, _value_style(text)) for line in _wrap(text, width - 2))
        else:
            labels = _wrap(label, label_width)
            values = _wrap(text, width - label_width - 2)
            for index in range(max(len(labels), len(values))):
                left = labels[index] if index < len(labels) else ""
                right = values[index] if index < len(values) else ""
                lines.append(
                    paint(left, MUTED) + " " * (label_width - _cell_width(left) + 2) + paint(right, _value_style(text))
                )
    return lines


def print_details(title: str, fields: Mapping[str, object], *, hint: str = "", plain: str | None = None) -> None:
    """Render a detail view or result; retain explicit plain output in pipes."""
    if not terminal_enabled():
        print(plain if plain is not None else "\n".join(f"{key}: {value}" for key, value in fields.items()))
        return
    width = terminal_width()
    print(_frame(title, _field_lines(fields, width - 4), width))
    print_hint(hint)


def print_hint(text: str) -> None:
    if text and terminal_enabled():
        print()
        for line in _wrap(text, terminal_width() - 2):
            print("  " + paint(line, MUTED))


def print_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[object]], *, hint: str = "") -> None:
    """Render a bounded table, switching to stacked records in small terminals."""
    values = [tuple(_value(value) for value in row) for row in rows]
    if any(len(row) != len(headers) for row in values):
        raise ValueError("table rows must match the column count")
    if not terminal_enabled():
        print("  ".join(headers))
        for row in values:
            print("  ".join(row))
        return
    width = terminal_width()
    inner = width - 4
    lines = []
    minimum = [min(24 if index == 0 else 12, max(8, _cell_width(header))) for index, header in enumerate(headers)]
    stacked = width < 80 or sum(minimum) + 2 * (len(headers) - 1) > inner
    if not values:
        lines = [paint("No matching entries.", MUTED)]
    elif stacked:
        for index, row in enumerate(values):
            if index:
                lines.append("")
            lines.extend(paint(line, BOLD) for line in _wrap(row[0], inner))
            lines.extend("  " + line for line in _field_lines(dict(zip(headers[1:], row[1:])), inner - 2))
    else:
        widths = [
            max(_cell_width(header), *(_cell_width(row[index]) for row in values))
            for index, header in enumerate(headers)
        ]
        available = inner - 2 * (len(headers) - 1)
        while sum(widths) > available:
            index = max(range(len(widths)), key=lambda i: widths[i] - minimum[i])
            widths[index] -= 1

        def cells(parts: Sequence[str], styles: Sequence[str]) -> str:
            return "  ".join(
                paint(part, style) + " " * (cell_width - _cell_width(part))
                for part, style, cell_width in zip(parts, styles, widths)
            )

        wrapped_headers = [_wrap(header, cell_width) for header, cell_width in zip(headers, widths)]
        for line in range(max(map(len, wrapped_headers))):
            lines.append(
                cells([part[line] if line < len(part) else "" for part in wrapped_headers], [MUTED] * len(headers))
            )
        lines.append(paint("─" * inner, ACCENT))
        for row in values:
            wrapped = [_wrap(value, cell_width) for value, cell_width in zip(row, widths)]
            styles = [BOLD, *(_value_style(value) for value in row[1:])]
            for line in range(max(map(len, wrapped))):
                lines.append(cells([part[line] if line < len(part) else "" for part in wrapped], styles))
    print(_frame(f"{title} · {len(values)}", lines, width))
    print_hint(hint)


def print_notice(
    message: str, *, level: str = "info", hint: str = "", stream: TextIO | None = None, plain: str | None = None
) -> None:
    stream = sys.stdout if stream is None else stream
    if not terminal_enabled(stream):
        print(plain if plain is not None else f"{level}: {message}", file=stream)
        return
    style = {"error": ERROR, "warning": WARNING, "success": SUCCESS}.get(level, ACCENT)
    print(paint(level.capitalize(), BOLD + style, stream), file=stream)
    for line in _wrap(message, terminal_width() - 2):
        print("  " + line, file=stream)
    if hint:
        print(file=stream)
        for line in _wrap(hint, terminal_width() - 2):
            print("  " + paint(line, MUTED, stream), file=stream)


def print_report(markdown: str) -> None:
    """Render the headings and tables emitted by our report builders."""
    if not terminal_enabled():
        print(markdown)
        return
    lines = markdown.splitlines()
    index = 0
    title = "Results"
    headers: list[str] = []

    def cells(line: str) -> list[str]:
        return [
            cell.strip().replace(r"\|", "|").replace("<br>", "\n")
            for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]

    while index < len(lines):
        line = lines[index]
        if line.startswith("|"):
            is_header = (
                index + 1 < len(lines)
                and lines[index + 1].startswith("|")
                and all(re.fullmatch(r":?-+:?", cell) for cell in cells(lines[index + 1]))
            )
            if is_header:
                headers = cells(line)
                index += 2
            if headers:
                rows = []
                while index < len(lines) and lines[index].startswith("|"):
                    row = cells(lines[index])
                    if len(row) != len(headers):
                        break
                    rows.append(row)
                    index += 1
                if rows or is_header:
                    print_table(title, headers, rows)
                    continue
        if line.startswith("#"):
            title = line.lstrip("# ")
            for part in _wrap(title, terminal_width()):
                print(paint(part, BOLD))
        else:
            for part in _wrap(line, terminal_width()):
                print(part)
        index += 1

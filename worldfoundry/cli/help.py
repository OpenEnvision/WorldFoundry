"""Terminal-aware help rendering for the public WorldFoundry CLIs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

_RESET = "\033[0m"
_ACCENT = "\033[38;5;142m"
_HEADING = "\033[1;38;5;187m"
_TEXT = "\033[38;5;252m"
_MUTED = "\033[38;5;245m"
_DEFAULT = "\033[38;5;149m"


def _color_enabled() -> bool:
    """Return whether interactive ANSI help should be emitted."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force.lower() not in {"", "0", "false", "no"}
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, style: str) -> str:
    return f"{style}{text}{_RESET}"


@dataclass(frozen=True)
class _HelpRow:
    invocation: str
    description: str


@dataclass(frozen=True)
class _HelpPanel:
    title: str
    rows: tuple[_HelpRow, ...]
    description: str = ""


class WorldFoundryArgumentParser(argparse.ArgumentParser):
    """Argument parser with compact rich help in interactive terminals.

    Parsing and non-interactive help remain standard ``argparse`` behavior.
    This keeps redirects, docs generation, shell completion, and tests stable.
    """

    def format_help(self) -> str:
        if not _color_enabled():
            return super().format_help()
        return self._format_terminal_help()

    def _format_terminal_help(self) -> str:
        terminal_width = max(64, min(shutil.get_terminal_size((120, 24)).columns, 180))
        usage = _wrap_usage(super().format_usage().strip(), terminal_width)
        if usage.startswith("usage:"):
            usage = _paint("usage:", _HEADING) + _paint(usage[len("usage:") :], _TEXT)

        blocks = [usage]
        if self.description:
            blocks.append(_paint(_wrap_usage(str(self.description).strip(), terminal_width), _TEXT))

        panels = self._help_panels()
        rendered = _render_panel_grid(panels, terminal_width)
        if rendered:
            blocks.append(rendered)

        if self.epilog:
            blocks.append(_render_epilog(str(self.epilog), terminal_width))
        return "\n\n".join(block for block in blocks if block).rstrip() + "\n"

    def _help_panels(self) -> tuple[_HelpPanel, ...]:
        panels: list[_HelpPanel] = []
        for group in self._action_groups:
            actions = tuple(
                action for action in group._group_actions if action.help is not argparse.SUPPRESS
            )
            if not actions:
                continue
            if group.title == "options" and len(actions) > 12:
                panels.extend(self._split_options(actions))
                continue
            title = (
                "commands"
                if any(isinstance(action, argparse._SubParsersAction) for action in actions)
                else group.title
            )
            panels.append(
                _HelpPanel(
                    str(title),
                    tuple(self._help_row(action) for action in actions),
                    str(group.description or ""),
                )
            )
        return tuple(panels)

    def _split_options(self, actions: Sequence[argparse.Action]) -> tuple[_HelpPanel, ...]:
        buckets: dict[str, list[argparse.Action]] = {}
        for action in actions:
            title = _option_section(action)
            buckets.setdefault(title, []).append(action)
        return tuple(
            _HelpPanel(title, tuple(self._help_row(action) for action in section_actions))
            for title, section_actions in buckets.items()
        )

    def _help_row(self, action: argparse.Action) -> _HelpRow:
        formatter = self._get_formatter()
        invocation = formatter._format_action_invocation(action)
        description = formatter._expand_help(action) if action.help else ""
        if _should_show_default(action, description):
            description = f"{description} (default: {_format_default(action.default)})".strip()
        return _HelpRow(invocation=invocation, description=description)


def _wrap_usage(usage: str, width: int) -> str:
    """Keep argparse choices and long dotted options inside the terminal width."""
    lines: list[str] = []
    for line in usage.splitlines():
        if len(line) <= width:
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=indent,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def _option_section(action: argparse.Action) -> str:
    dest = action.dest.replace("-", "_")
    if (
        dest in {"help", "json", "output", "verbose", "quiet"}
        or dest.startswith("output_")
    ):
        return "options"
    if dest.startswith("generation_cache") or "cache" in dest:
        return "cache options"
    if dest.startswith(("model", "ckpt")) or dest in {"device", "gpu", "dtype", "low_vram"}:
        return "model options"
    if dest.startswith(("benchmark", "suite", "task", "metric", "dataset")):
        return "benchmark and task options"
    if dest.startswith(
        ("plan", "resume", "engine", "mode", "timeout", "workdir", "env", "fail_", "skip_")
    ):
        return "execution options"
    if dest.startswith(
        ("input", "request", "result", "artifact", "data_", "prompt", "seed", "frame", "step")
    ):
        return "input and generation options"
    return "additional options"


def _should_show_default(action: argparse.Action, description: str) -> bool:
    if "%(default)" in str(action.help) or "(default:" in description:
        return False
    if action.default is None or action.default == argparse.SUPPRESS:
        return False
    if action.dest == "help" or action.required:
        return False
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return False
    return True


def _format_default(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _render_panel_grid(panels: Sequence[_HelpPanel], terminal_width: int) -> str:
    if not panels:
        return ""
    if terminal_width < 112 or len(panels) == 1:
        return "\n".join(_render_panel(panel, terminal_width) for panel in panels)

    gap = 2
    left_width = (terminal_width - gap) // 2
    right_width = terminal_width - gap - left_width
    columns: list[list[str]] = [[], []]
    heights = [0, 0]
    for panel in panels:
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


def _render_panel(panel: _HelpPanel, width: int) -> str:
    return "\n".join(_render_panel_lines(panel, width))


def _render_panel_lines(panel: _HelpPanel, width: int) -> list[str]:
    width = max(48, width)
    inner_width = width - 4
    title = f" {panel.title} "
    top_fill = max(0, inner_width - len(title))
    lines = [_paint(f"╭─{title}{'─' * top_fill}─╮", _ACCENT)]

    for description_line in textwrap.wrap(
        panel.description,
        width=inner_width,
        break_long_words=True,
        break_on_hyphens=False,
    ):
        lines.append(
            _paint("│ ", _ACCENT)
            + _paint(description_line.ljust(inner_width), _MUTED)
            + _paint(" │", _ACCENT)
        )

    invocation_width = min(
        max((len(row.invocation) for row in panel.rows), default=0),
        max(18, inner_width // 2),
    )
    description_width = max(16, inner_width - invocation_width - 2)
    for row in panel.rows:
        invocation_lines = textwrap.wrap(
            row.invocation,
            width=invocation_width,
            subsequent_indent="  ",
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        description_lines = textwrap.wrap(
            row.description,
            width=description_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        height = max(len(invocation_lines), len(description_lines))
        for index in range(height):
            invocation = invocation_lines[index] if index < len(invocation_lines) else ""
            description = description_lines[index] if index < len(description_lines) else ""
            body = (
                _paint(invocation.ljust(invocation_width), _TEXT)
                + "  "
                + _color_description(description.ljust(description_width))
            )
            lines.append(_paint("│ ", _ACCENT) + body + _paint(" │", _ACCENT))
    lines.append(_paint(f"╰{'─' * (width - 2)}╯", _ACCENT))
    return lines


def _color_description(text: str) -> str:
    marker = "(default:"
    if marker not in text:
        return _paint(text, _MUTED)
    prefix, default = text.split(marker, 1)
    return _paint(prefix, _MUTED) + _paint(marker + default, _DEFAULT)


def _render_epilog(epilog: str, width: int) -> str:
    rendered: list[str] = []
    for line in epilog.strip().splitlines():
        if not line:
            rendered.append("")
            continue
        if line.endswith(":"):
            rendered.append(_paint(line, _HEADING))
        else:
            rendered.extend(_paint(item, _MUTED) for item in textwrap.wrap(line, width=width))
    return "\n".join(rendered)

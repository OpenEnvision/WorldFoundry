"""Terminal-aware help rendering for the public WorldFoundry CLIs."""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence

from .presentation import (
    BOLD as _HEADING,
)
from .presentation import (
    MUTED as _MUTED,
)
from .presentation import (
    Panel as _HelpPanel,
)
from .presentation import (
    Row as _HelpRow,
)
from .presentation import (
    paint as _paint,
)
from .presentation import (
    print_notice,
    terminal_width,
)
from .presentation import (
    render_panels as _render_panel_grid,
)
from .presentation import (
    terminal_enabled as _terminal_help_enabled,
)


class WorldFoundryArgumentParser(argparse.ArgumentParser):
    """Argument parser with compact rich help in interactive terminals.

    Parsing and non-interactive help remain standard ``argparse`` behavior.
    This keeps redirects, docs generation, shell completion, and tests stable.
    """

    def format_help(self) -> str:
        if not _terminal_help_enabled():
            return super().format_help()
        return self._format_terminal_help()

    def error(self, message: str) -> None:
        if not _terminal_help_enabled(sys.stderr):
            super().error(message)
        print_notice(
            message, level="error", hint=f"Run {self.prog} --help to see commands and options.", stream=sys.stderr
        )
        self.exit(2)

    def _format_terminal_help(self) -> str:
        width = terminal_width()
        usage = _wrap_usage(self._terminal_usage(), width)
        if usage.startswith("usage:"):
            usage = _paint("usage:", _HEADING) + usage[len("usage:") :]

        breadcrumb = self.prog.split(maxsplit=1)
        title = "WorldFoundry" if breadcrumb[0].startswith("worldfoundry") else breadcrumb[0]
        if len(breadcrumb) > 1:
            title += " / " + breadcrumb[1]
        blocks = [_paint(_wrap_usage(title, width), _HEADING), usage]
        if self.description:
            blocks.append(_wrap_usage(str(self.description).strip(), width))

        panels = self._help_panels()
        rendered = _render_panel_grid(panels, width)
        if rendered:
            blocks.append(rendered)

        if self.epilog:
            blocks.append(_render_epilog(str(self.epilog), width))
        return "\n\n".join(block for block in blocks if block).rstrip() + "\n"

    def _terminal_usage(self) -> str:
        if self.usage is not None:
            return super().format_usage().strip()
        formatter = self._get_formatter()
        actions = [action for action in self._actions if action.help != argparse.SUPPRESS]
        required_groups = [group for group in self._mutually_exclusive_groups if group.required]
        required = [
            action
            for action in actions
            if action.option_strings
            and (action.required or any(action in group._group_actions for group in required_groups))
        ]
        parts = [f"usage: {self.prog}"]
        if required:
            parts.append(formatter._format_actions_usage(required, required_groups))
        if any(action.option_strings and action not in required for action in actions):
            parts.append("[OPTIONS]")
        for action in actions:
            if isinstance(action, argparse._SubParsersAction):
                parts.append("COMMAND ..." if action.required else "[COMMAND ...]")
            elif not action.option_strings:
                parts.append(formatter._format_actions_usage([action], []))
        return " ".join(parts)

    def _help_panels(self) -> tuple[_HelpPanel, ...]:
        panels: list[_HelpPanel] = []
        additional_panels: list[_HelpPanel] = []
        for group in self._action_groups:
            actions = tuple(action for action in group._group_actions if action.help != argparse.SUPPRESS)
            if not actions:
                continue
            if group.title == "options" and len(actions) > 12:
                # Keep explicitly named pipeline groups near the top of model help.
                for panel in self._split_options(actions):
                    (panels if panel.title == "options" else additional_panels).append(panel)
                continue
            ordinary_actions = [action for action in actions if not isinstance(action, argparse._SubParsersAction)]
            if ordinary_actions:
                panels.extend(
                    self._split_options(
                        ordinary_actions, title=str(group.title), description=str(group.description or "")
                    )
                )
            for action in actions:
                if not isinstance(action, argparse._SubParsersAction):
                    continue
                commands = tuple(action._get_subactions())
                rows = tuple(self._help_row(command) for command in commands if command.help != argparse.SUPPRESS)
                # argparse only creates help pseudo-actions for commands with help=.
                if not commands:
                    rows = tuple(
                        _HelpRow(name, str(parser.description or "")) for name, parser in action.choices.items()
                    )
                help_groups = getattr(action, "help_groups", {})
                grouped_names = set()
                for title, names in help_groups.items():
                    grouped_rows = tuple(
                        self._help_row(command)
                        for name in names
                        for command in commands
                        if command.dest == name and command.help != argparse.SUPPRESS
                    )
                    if grouped_rows:
                        panels.append(_HelpPanel(title, grouped_rows))
                    grouped_names.update(names)
                remaining = (
                    tuple(
                        self._help_row(command)
                        for command in commands
                        if command.dest not in grouped_names and command.help != argparse.SUPPRESS
                    )
                    if help_groups
                    else rows
                )
                if remaining:
                    panels.append(_HelpPanel("commands", remaining, str(group.description or "")))
        return tuple(panels + additional_panels)

    def _split_options(
        self, actions: Sequence[argparse.Action], *, title: str | None = None, description: str = ""
    ) -> tuple[_HelpPanel, ...]:
        buckets: dict[str, list[argparse.Action]] = {}
        for action in actions:
            dotted_prefix = next(
                (
                    option[2:].rsplit(".", 1)[0]
                    for option in action.option_strings
                    if option.startswith("--") and "." in option
                ),
                None,
            )
            section = f"{dotted_prefix} options" if dotted_prefix else title or _option_section(action)
            buckets.setdefault(section, []).append(action)
        return tuple(
            _HelpPanel(
                section, tuple(self._help_row(action) for action in section_actions), description if index == 0 else ""
            )
            for index, (section, section_actions) in enumerate(buckets.items())
        )

    def _help_row(self, action: argparse.Action) -> _HelpRow:
        formatter = self._get_formatter()
        original_metavar = formatter._get_default_metavar_for_optional

        def metavar(item: argparse.Action) -> str:
            kind = getattr(item.type, "__name__", "")
            if kind in {"Path", "int", "float", "str"}:
                return {"Path": "PATH", "int": "INT", "float": "FLOAT", "str": "STR"}[kind]
            if item.dest.endswith(("_path", "_dir", "_root", "_file")):
                return "PATH"
            return original_metavar(item)

        formatter._get_default_metavar_for_optional = metavar
        invocation = formatter._format_action_invocation(action)
        description = formatter._expand_help(action) if action.help else ""
        default = _format_default(action.default) if _should_show_default(action, description) else None
        return _HelpRow(invocation=invocation, description=description, default=default)


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
    if dest in {"help", "json", "output", "verbose", "quiet"} or dest.startswith("output_"):
        return "options"
    if dest.startswith("generation_cache") or "cache" in dest:
        return "cache options"
    if dest.startswith(("model", "ckpt")) or dest in {"device", "gpu", "dtype", "low_vram"}:
        return "model options"
    if dest.startswith(("benchmark", "suite", "task", "metric", "dataset")):
        return "benchmark and task options"
    if dest.startswith(("plan", "resume", "engine", "mode", "timeout", "workdir", "env", "fail_", "skip_")):
        return "execution options"
    if dest.startswith(("input", "request", "result", "artifact", "data_", "prompt", "seed", "frame", "step")):
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

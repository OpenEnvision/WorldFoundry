"""Terminal icons for the WorldFoundry TUI.

Icon sets:

- ``nerd`` — Nerd Font private-use glyphs (best with a patched Nerd Font)
- ``unicode`` — BMP symbols that render in DejaVu / Noto monospace (remote default)
- ``ascii`` — plain-text fallbacks

See ``docs/fumadocs/content/docs/guides/tui.mdx`` for setup instructions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

IconMode = Literal["nerd", "unicode", "ascii"]
"""Supported icon rendering modes for the TUI."""

# ── Recommended font stack ────────────────────────────────────────
# NOTE: Set this in your terminal emulator / Cursor settings for correct Nerd Font rendering.
NERD_FONT_FAMILY = (
    '"JetBrainsMono Nerd Font", '
    '"MesloLGS NF", '
    '"FiraCode Nerd Font", '
    '"Hack Nerd Font", '
    '"CaskaydiaCove Nerd Font", '
    '"Symbols Nerd Font", '
    "monospace"
)

# ── Nerd Font glyph table (private-use area) ──────────────────────
# NOTE: Requires a patched Nerd Font for correct rendering.
_NERD = {
    # key → glyph string; some values include a trailing label for select widgets
    "app": "\u25c8",
    "infer": "\U000f08b9",
    "eval": "\U000f0219",
    "studio": "\U000f05c0",
    "models": "\U000f06af Models",
    "benchmarks": "\U000f06ff Benchmarks",
    "run": "\uf04b Run",
    "stop": "\uf04d Stop",
    "copy": "\U000f018f Copy",
    "cmd": "\uf489 Cmd",
    "hide_cmd": "\uf489 Hide",
    "check": "\uf00c Check",
    "files": "\U000f024b Files",
    "gpu": "\U000f089f GPU",
    "sync": "\U000f0450 Sync",
    "general": "\uf013 General",
    "prompt_media": "\uf044 Prompt & Media",
    "paths_runtime": "\uf07c Paths & Runtime",
    "sampling": "\U000f03be Sampling",
    "camera": "\U000f0100 Camera",
    "model": "\U000f06af Model",
    "advanced": "\uf423 Advanced flags",
    "command_preview": "\uf489 Command Preview",
    "studio_title": "\U000f05c0 WorldFoundry Studio",
    "ready": "\uf00c Ready",
}

# ── Unicode glyph table (BMP symbols) ────────────────────────────
# NOTE: Compatible with DejaVu / Noto monospace — safe for remote terminals.
_UNICODE = {
    "app": "\u25c8",
    "infer": "\u26a1",
    "eval": "\u25ce",
    "studio": "\u25c9",
    "models": "\u25c8 Models",
    "benchmarks": "\u25a3 Benchmarks",
    "run": "\u25b6 Run",
    "stop": "\u25a0 Stop",
    "copy": "\u25a3 Copy",
    "cmd": "$ Cmd",
    "hide_cmd": "$ Hide",
    "check": "\u2713 Check",
    "files": "\u25a4 Files",
    "gpu": "\u2699 GPU",
    "sync": "\u21bb Sync",
    "general": "\u2699 General",
    "prompt_media": "\u270e Prompt & Media",
    "paths_runtime": "\u25a4 Paths & Runtime",
    "sampling": "\u25d0 Sampling",
    "camera": "\u25cf Camera",
    "model": "\u25c8 Model",
    "advanced": "\u2699 Advanced flags",
    "command_preview": "$ Command Preview",
    "studio_title": "\u25c9 WorldFoundry Studio",
    "ready": "\u2713 Ready",
}

# ── ASCII fallback glyph table ──────────────────────────────────
# NOTE: Plain-text labels — always renders correctly regardless of font.
_ASCII = {
    "app": "*",
    "infer": "[I]",
    "eval": "[E]",
    "studio": "[S]",
    "models": "> Models",
    "benchmarks": "> Benchmarks",
    "run": "> Run",
    "stop": "x Stop",
    "copy": "cp Copy",
    "cmd": "$ Cmd",
    "hide_cmd": "$ Hide",
    "check": "ok Check",
    "files": "dir Files",
    "gpu": "GPU",
    "sync": "~ Sync",
    "general": "* General",
    "prompt_media": "Prompt & Media",
    "paths_runtime": "Paths & Runtime",
    "sampling": "Sampling",
    "camera": "Camera",
    "model": "Model",
    "advanced": "Advanced flags",
    "command_preview": "$ Command Preview",
    "studio_title": "WorldFoundry Studio",
    "ready": "Ready",
}


# ── Icon-mode detection ────────────────────────────────────────────

def _remote_dev_host() -> bool:
    """Detect whether the terminal is running on a remote SSH / cloud-dev host."""
    return any(
        os.environ.get(key)
        for key in (
            "SSH_CONNECTION",
            "SSH_CLIENT",
            "DSW_INSTANCE_ID",
            "DSW_POD_HOSTNAME",
            "VSCODE_AGENT_FOLDER",
            "CURSOR_AGENT",
        )
    )


# ── Icon resolution ────────────────────────────────────────────────

def resolve_icon_mode() -> IconMode:
    """Use Unicode unless an icon mode or configured Nerd Font is requested."""
    raw = os.environ.get("WORLDFOUNDRY_TUI_ICONS", "").strip().lower()
    if raw in {"ascii", "plain", "text", "0", "false", "off"}:
        return "ascii"
    if raw in {"nerd", "nf", "nerd-font", "nerdfont"}:
        return "nerd"
    if raw in {"unicode", "uni", "compat", "default"}:
        return "unicode"
    return "nerd" if nerd_font_configured() else "unicode"


def nerd_font_configured() -> bool:
    """Return True when the user opts in via ``WORLDFOUNDRY_TUI_NERD_FONT=1``."""
    return os.environ.get("WORLDFOUNDRY_TUI_NERD_FONT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ── TuiIcons dataclass ──────────────────────────────────────────────

@dataclass(frozen=True)
class TuiIcons:
    """Glyph lookup for the WorldFoundry TUI, keyed by icon name.

    Attributes:
        mode: The active icon rendering mode (``"nerd"``, ``"unicode"``, or ``"ascii"``).
    """

    mode: IconMode

    def __post_init__(self) -> None:
        """Bind the appropriate glyph table based on :attr:`mode`."""
        table = {"nerd": _NERD, "unicode": _UNICODE, "ascii": _ASCII}[self.mode]
        object.__setattr__(self, "_glyphs", table)

    def __getitem__(self, key: str) -> str:
        """Return the glyph string for *key*."""
        return self._glyphs[key]

    @property
    def uses_nerd_font(self) -> bool:
        """Return whether the active mode uses Nerd Font private-use glyphs."""
        return self.mode == "nerd"

    def action_options(self) -> tuple[tuple[str, str], ...]:
        """Return label/value pairs for the TUI action mode selector."""
        return (
            (f"{self['infer']} Inference", "infer"),
            (f"{self['eval']} Evaluation", "eval"),
            (f"{self['studio']} Studio", "ui"),
        )

    def action_labels(self) -> dict[str, str]:
        """Return a mapping from action value to display label."""
        return {value: label for label, value in self.action_options()}

    def models_title(self, count: int) -> str:
        """Build a pane title showing the model icon and *count*."""
        return f"{self['models']} ({count})"

    def benchmarks_title(self, count: int) -> str:
        """Build a pane title showing the benchmark icon and *count*."""
        return f"{self['benchmarks']} ({count})"

    def status_infer(self, model: str, ckpt: str) -> str:
        """Build a status-line string for inference mode."""
        return f"{self['infer']} infer · {model} · {ckpt}"

    def status_eval(self, model: str, benchmark: str) -> str:
        """Build a status-line string for evaluation mode."""
        return f"{self['eval']} eval · {model} · {benchmark}"

    def status_studio(self) -> str:
        """Build a status-line string for Studio mode."""
        return f"{self['studio']} studio · launch local web UI"

    def startup_hint(self) -> str | None:
        """Return a Rich-markup hint about font setup, or ``None`` if no hint is needed."""
        if not self.uses_nerd_font or nerd_font_configured():
            return None
        return (
            "[dim]Icons use Nerd Font glyphs. Set your terminal font to a Nerd Font "
            "(e.g. JetBrainsMono Nerd Font) or export WORLDFOUNDRY_TUI_ICONS=unicode. "
            "See docs/guides/tui for details.[/]"
        )


icons = TuiIcons(resolve_icon_mode())

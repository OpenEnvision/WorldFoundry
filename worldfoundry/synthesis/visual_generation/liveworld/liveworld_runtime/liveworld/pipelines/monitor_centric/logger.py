"""Clean logging utilities for event-centric pipeline.

Provides structured, stage-based logging that makes pipeline progress clear at a glance.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager

# --------------------------------------------------------------------------
# Set env vars BEFORE any SAM3 / third-party imports evaluate them at module
# level.  Must happen at import time of this module, not inside a function.
# --------------------------------------------------------------------------
os.environ.setdefault("SAM3_TQDM_DISABLE", "1")
os.environ.setdefault("SAM3_DISABLE_TQDM", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# Suppress verbose third-party logs
def suppress_verbose_logs():
    """Suppress verbose logs from SAM3, Stream3R, and other noisy modules."""
    logging.getLogger("sam3").setLevel(logging.WARNING)
    logging.getLogger("worldfoundry.base_models.perception_core.segment.sam3.model.sam3_video_predictor").setLevel(
        logging.WARNING
    )
    logging.getLogger("stream3r").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"transformers\.utils\.hub",
    )


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


class PipelineLogger:
    """Structured logger for event-centric pipeline stages."""

    # ANSI color codes
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        self.use_color = use_color and sys.stdout.isatty()
        self._round_start: float | None = None
        self._current_round: int = 0

    def _c(self, color: str, text: str) -> str:
        if self.use_color:
            return f"{color}{text}{self.RESET}"
        return text

    def _round_tag(self) -> str:
        return f"R{self._current_round:02d}"

    def _clean_text(self, text: object) -> str:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        return " | ".join(lines) if lines else "-"

    def _format_value(self, value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, (list, tuple, set)):
            items = [self._clean_text(v) for v in value if self._clean_text(v) != "-"]
            if not items:
                return "[]"
            if len(items) > 4:
                return "[" + ", ".join(items[:4]) + f", +{len(items) - 4}]"
            return "[" + ", ".join(items) + "]"
        if isinstance(value, dict):
            keys = list(value.keys())
            if len(keys) > 4:
                keys = keys[:4]
                return "{" + ", ".join(str(k) for k in keys) + ", ...}"
            return "{" + ", ".join(str(k) for k in keys) + "}"
        return self._clean_text(value)

    def _emit(self, level: str, message: object, *, color: str = "", blank_before: bool = False):
        if blank_before:
            print()
        line = f"[{self._round_tag()}][{level}] {self._clean_text(message)}"
        if color:
            line = self._c(color, line)
        print(line)

    # --- Headers ---

    def header(self, title: str):
        self._emit("HDR", title, color=self.BOLD + self.BLUE, blank_before=True)

    def round_start(self, iter_idx: int, start_frame: int, end_frame: int):
        """Print round header and start timing."""
        self._current_round = int(iter_idx)
        self._round_start = time.time()
        self._emit(
            "ROUND",
            f"start frames={start_frame}-{end_frame}",
            color=self.BOLD + self.MAGENTA,
            blank_before=True,
        )

    def round_end(self, iter_idx: int, **kv):
        """Print round summary with elapsed time."""
        self._current_round = int(iter_idx)
        elapsed = time.time() - self._round_start if self._round_start else 0
        parts = [f"{k}={self._format_value(v)}" for k, v in kv.items()]
        summary = " | ".join(parts)
        msg = f"end ({_fmt_time(elapsed)})"
        if summary:
            msg = f"{msg} | {summary}"
        self._emit("ROUND", msg, color=self.MAGENTA)

    # --- Stages with timing ---

    @contextmanager
    def timed(self, label: str):
        """Context manager that prints label on enter and elapsed on exit."""
        clean_label = self._clean_text(label)
        t0 = time.time()
        self._emit("TASK", clean_label, color=self.CYAN)
        try:
            yield
        except Exception as exc:
            dt = time.time() - t0
            self._emit("ERR", f"{clean_label} failed after {_fmt_time(dt)}: {exc}", color=self.RED)
            raise
        dt = time.time() - t0
        self._emit("TIME", f"{clean_label} ({_fmt_time(dt)})", color=self.DIM)

    def step(self, label: str, detail: str = ""):
        """Print a step within a stage (no timing)."""
        msg = self._clean_text(label) if not detail else f"{self._clean_text(label)}: {self._clean_text(detail)}"
        self._emit("STEP", msg, color=self.CYAN)

    # --- Info lines ---

    def info(self, message: str):
        self._emit("INFO", message)

    def detail(self, message: str):
        """Dimmed detail line."""
        self._emit("DETAIL", message, color=self.DIM)

    def warning(self, message: str):
        self._emit("WARN", message, color=self.YELLOW)

    def error(self, message: str):
        self._emit("ERR", message, color=self.RED)

    def saved(self, filename: str, detail: str = ""):
        """Log a saved file."""
        msg = self._clean_text(filename) if not detail else f"{self._clean_text(filename)} | {self._clean_text(detail)}"
        self._emit("SAVE", msg, color=self.GREEN)

    # --- Convenience (kept for compatibility) ---

    def stage(self, stage_name: str, description: str = ""):
        msg = (
            self._clean_text(stage_name)
            if not description
            else f"{self._clean_text(stage_name)}: {self._clean_text(description)}"
        )
        self._emit("STAGE", msg, color=self.BOLD + self.CYAN, blank_before=True)

    def detection(self, entities: list[str]):
        if not entities:
            self._emit("DET", "entities=(none)")
            return
        seen = set()
        ordered = []
        for ent in entities:
            name = self._clean_text(ent)
            if name != "-" and name not in seen:
                seen.add(name)
                ordered.append(name)
        self._emit("DET", "entities=" + (", ".join(ordered) if ordered else "(none)"))

    def iteration_start(self, iter_idx: int, start_frame: int, end_frame: int):
        self.round_start(iter_idx, start_frame, end_frame)

    def final_summary(self, output_root: str, total_frames: int, total_points: int):
        self._emit("FINAL", "pipeline complete", color=self.BOLD + self.GREEN, blank_before=True)
        self._emit("FINAL", f"output={self._clean_text(output_root)}", color=self.GREEN)
        self._emit("FINAL", f"frames={total_frames} | points={total_points:,}", color=self.GREEN)


# Global logger instance
logger = PipelineLogger()

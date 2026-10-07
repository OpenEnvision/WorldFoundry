"""Denoising progress bar that stays readable in a redirected log."""

from __future__ import annotations

import re
import sys
from typing import TextIO

from tqdm.auto import tqdm
from tqdm.std import tqdm as std_tqdm

_STEP = re.compile(r"(\d+)/(\d+)")


class _LogTqdm(std_tqdm):
    """One short line per completed step.

    The stock bar redraws with a carriage return. A TTY paints that in place.
    A log file keeps every redraw, so a run becomes one enormous line. Closing
    the bar also reprints the last step with a new rate; that reprint is dropped.
    """

    @staticmethod
    def status_printer(file):
        fp_flush = getattr(file, "flush", lambda: None)
        last_step = [""]

        def print_status(text: str) -> None:
            line = text.rstrip()
            if not line:
                return
            match = _STEP.search(line)
            step = match.group(0) if match is not None else line
            if step == last_step[0]:
                return
            last_step[0] = step
            file.write(line + "\n")
            fp_flush()

        return print_status


def denoising_progress(total: int, *, file: TextIO | None = None):
    """Bar over ``total`` denoising steps.

    A terminal gets the usual in-place bar. A pipe or a log file gets one
    line of at most 80 characters per step.
    """
    stream = sys.stderr if file is None else file
    interactive = bool(getattr(stream, "isatty", lambda: False)())
    if interactive:
        return tqdm(total=total, desc="Denoising", unit="step", file=stream, dynamic_ncols=True)
    return _LogTqdm(
        total=total,
        desc="Denoising",
        unit="step",
        file=stream,
        ncols=80,
        ascii=True,
        dynamic_ncols=False,
        mininterval=0,
        miniters=1,
        leave=True,
    )

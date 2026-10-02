"""Logging, formatting, and debug-print helpers for training and evaluation runs.

Human-facing counters, timestamps, and pretty ``__repr__`` strings live
here so recipes and eval loops do not reimplement them. Also includes
stdout/stderr redirection context managers, a ``watch``-style decorator,
and logging filters that drop or rewrite noisy third-party messages.

This is not the distributed rank logger
(:mod:`worldfoundry.core.distributed.runtime.logging`).
"""

import ast
import io
import logging
import os
import pprint
import re
import string
import sys
import textwrap
import time
import traceback
from datetime import datetime
from typing import Callable, Union

import numpy as np
from typing_extensions import Literal

from worldfoundry.core.utils.python.functional_utils import meta_decorator
from worldfoundry.core.utils.python.misc_utils import match_patterns

# ──────────────────────────────────────────────────────────────────────────
# Human-facing numbers, timestamps, and pretty reprs (not rank loggers)
# ──────────────────────────────────────────────────────────────────────────


def to_readable_count_str(value: int, precision: int = 2) -> str:
    """Format a non-negative integer with ``K`` / ``M`` / ``B`` / ``T`` suffixes."""
    assert value >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(value)) + 1 if value > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    value = value * (10**shift)
    index = num_groups - 1
    rem = value - int(value)
    if precision > 0 and rem > 0.01:
        fmt = f"{{:.{precision}f}}"
        rem_str = fmt.format(rem).lstrip("0")
    else:
        rem_str = ""
    return f"{int(value):,d}{rem_str} {labels[index]}"


def to_scientific_str(value, precision: int = 1, capitalize: bool = False) -> str:
    """
    0.0015 -> "1.5e-3"
    """
    if value == 0:
        return "0"
    return f"{value:.{precision}e}".replace("e-0", "E-" if capitalize else "e-")


def print_str(*args, **kwargs):
    """
    Same as print() signature but returns a string
    """
    sstream = io.StringIO()
    kwargs.pop("file", None)
    print(*args, **kwargs, file=sstream)
    return sstream.getvalue()


def _eval_simple_format_node(node: ast.AST, env: dict) -> object:
    """Evaluate a name or integer arithmetic AST node against *env* only."""
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise NameError(node.id)
        return env[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_simple_format_node(node.operand, env)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _eval_simple_format_node(node.left, env)
        right = _eval_simple_format_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    raise ValueError(f"unsupported format expression: {ast.dump(node)}")


def fstring(fmt_str, **kwargs):
    """Format ``{name}`` / ``{name+1}`` placeholders without ``eval``.

    Call sites only use index-style templates (``{i}``, ``{i+1}``). Names
    must be bound in *kwargs*; only integer ``+`` / ``-`` / ``*`` is
    allowed inside braces.
    """

    def substitute(match):
        """Replace one ``{...}`` group using the restricted arithmetic evaluator."""

        tree = ast.parse(match.group(1), mode="eval")
        return str(_eval_simple_format_node(tree.body, kwargs))

    return re.sub(r"\{([^{}]+)\}", substitute, fmt_str)


def get_format_keys(fmt_str):
    """Return field names referenced by a :class:`string.Formatter` template."""
    keys = []
    for literal, field_name, fmt_spec, conversion in string.Formatter().parse(fmt_str):
        if field_name:
            keys.append(field_name)
    return keys


def get_timestamp(milli_precision: int = 3):
    """Return the local clock as ``yy-mm-dd HH:MM:SS`` with optional milliseconds."""
    fmt = "%y-%m-%d %H:%M:%S"
    if milli_precision > 0:
        fmt += ".%f"
    stamp = datetime.now().strftime(fmt)
    if milli_precision > 0:
        stamp = stamp[:-milli_precision]
    return stamp


def pretty_repr_str(obj, **kwargs):
    """
    Useful to produce __repr__()
    """
    if isinstance(obj, str):
        cls_name = obj
    else:
        cls_name = obj.__class__.__name__
    kw_strs = [k + "=" + pprint.pformat(v, indent=2, compact=True) for k, v in kwargs.items()]
    new_line = len(cls_name) + sum(len(kw) for kw in kw_strs) > 84
    if new_line:
        kw = ",\n".join(kw_strs)
        return f"{cls_name}(\n{textwrap.indent(kw, '  ')}\n)"
    else:
        kw = ", ".join(kw_strs)
        return f"{cls_name}({kw})"


def pprint_(*objs, **kwargs):
    """
    Use pprint to format the objects
    """
    print(
        *[pprint.pformat(obj, indent=2) if not isinstance(obj, str) else obj for obj in objs],
        **kwargs,
    )


def get_exception_info(to_str: bool = False):
    """
    Returns:
        {'type': ExceptionType, 'value': ExceptionObject, 'trace': <traceback str>}
    """
    typ_, value, trace = sys.exc_info()
    return {
        "type": typ_.__name__ if to_str else typ_,
        "value": str(value) if to_str else value,
        "trace": "".join(traceback.format_exception(typ_, value, trace)),
    }


class DebugPrinter:
    """
    Debug print, usage: dprint = DebugPrint(enabled=True)
    dprint(...)
    """

    def __init__(self, enabled, tensor_summary: Literal["shape", "shape+dtype", "none"] = "shape"):
        """
        Args:
            tensor_summary:
                - shape: only prints shape
                - shape+dtype: also prints dtype and device
                - none: print full tensor
        """
        self.enabled = enabled
        assert tensor_summary in ["shape", "shape+dtype", "none"]
        self.tensor_summary = tensor_summary

    def __call__(self, *args, **kwargs):
        """Pretty-print arguments when enabled, summarizing tensors by shape."""
        if not self.enabled:
            return
        args = [self._process_arg(a) for a in args]
        pprint_(*args, **kwargs)

    def _process_arg(self, arg):
        """Collapse tensors/arrays to shape (and dtype/device) so dumps stay readable."""

        import numpy as np
        import torch

        if torch.is_tensor(arg):
            if self.tensor_summary == "shape":
                return str(list(arg.size()))
            elif self.tensor_summary == "shape+dtype":
                return f"{arg.dtype}{list(arg.size())}|{arg.device}"
        elif isinstance(arg, np.ndarray):
            if self.tensor_summary == "shape":
                return str(list(arg.shape))
            elif self.tensor_summary == "shape+dtype":
                return f"{arg.dtype}{list(arg.shape)}"
        return arg


@meta_decorator
def watch(func, seconds: int = 5, max_times: int = 0, keep_returns: bool = False):
    """
    Decorator: executes a function repeated with the args and
        emulate `watch -n` capability

    See `gpustat` repo: https://github.com/wookayin/gpustat/pull/41/files

    Args:
        max_times: watch for `max_times` and then exit. If 0, never exits
        keep_returns: if True, will keep the return value from the function
            and return as a list at the end
    """
    from blessings import Terminal

    def _wrapped(*args, **kwargs):
        """Fullscreen refresh loop; ``KeyboardInterrupt`` exits without re-raising."""

        term = Terminal()
        N = 0
        returns = []
        with term.fullscreen():
            while True:
                try:
                    with term.location(0, 0):
                        ret = func(*args, **kwargs)
                        print(term.clear_eos, end="")
                        if keep_returns:
                            returns.append(ret)
                    N += 1
                    if max_times > 0 and N >= max_times:
                        break
                    time.sleep(seconds)
                except KeyboardInterrupt:
                    break
        return returns

    return _wrapped


# ──────────────────────────────────────────────────────────────────────────
# Stdout/stderr redirection — restore on exit even when the body raises
# ──────────────────────────────────────────────────────────────────────────


class PrintRedirection(object):
    """
    Context manager: temporarily redirects stdout and stderr
    """

    def __init__(self, stdout=None, stderr=None):
        """
        Args:
          stdout: if None, defaults to sys.stdout, unchanged
          stderr: if None, defaults to sys.stderr, unchanged
        """
        if stdout is None:
            stdout = sys.stdout
        if stderr is None:
            stderr = sys.stderr
        self._stdout, self._stderr = stdout, stderr

    def __enter__(self):
        """Swap stdout/stderr after flushing the previous streams."""

        self._old_out, self._old_err = sys.stdout, sys.stderr
        self._old_out.flush()
        self._old_err.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the original streams after flushing the replacements."""

        self.flush()
        # restore the normal stdout and stderr
        sys.stdout, sys.stderr = self._old_out, self._old_err

    def flush(self):
        "Manually flush the replaced stdout/stderr buffers."
        self._stdout.flush()
        self._stderr.flush()


class PrintToFile(PrintRedirection):
    """
    Print to file and save/close the handle at the end.
    """

    def __init__(self, out_file=None, err_file=None):
        """
        Args:
          out_file: file path
          err_file: file path. If the same as out_file, print both stdout
              and stderr to one file in order.
        """
        self.out_file = None
        self.err_file = None
        out_path = None
        if out_file:
            out_path = os.path.expanduser(out_file)
            self.out_file = open(out_path, "w")
        if err_file:
            err_path = os.path.expanduser(err_file)
            if out_path is not None and err_path == out_path:  # redirect both stdout/err to one file
                self.err_file = self.out_file
            else:
                self.err_file = open(err_path, "w")
        super().__init__(stdout=self.out_file, stderr=self.err_file)

    def __exit__(self, *args):
        """Restore streams, then close file handles (shared out/err is closed once)."""

        super().__exit__(*args)
        if self.out_file:
            self.out_file.close()
        if self.err_file and self.err_file is not self.out_file:
            self.err_file.close()


def PrintSuppress(no_out=True, no_err=False):
    """
    Args:
      no_out: stdout writes to sys.devnull
      no_err: stderr writes to sys.devnull
    """
    out_file = os.devnull if no_out else None
    err_file = os.devnull if no_err else None
    return PrintToFile(out_file=out_file, err_file=err_file)


class PrintString(PrintRedirection):
    """
    Redirect stdout and stderr to strings.
    """

    def __init__(self):
        """Capture both streams into in-memory string buffers."""
        self.out_stream = io.StringIO()
        self.err_stream = io.StringIO()
        super().__init__(stdout=self.out_stream, stderr=self.err_stream)

    def stdout(self):
        "Returns: stdout as one string."
        return self.out_stream.getvalue()

    def stderr(self):
        "Returns: stderr as one string."
        return self.err_stream.getvalue()

    def stdout_by_line(self):
        "Returns: a list of stdout line by line, ignore trailing blanks"
        return self.stdout().rstrip().split("\n")

    def stderr_by_line(self):
        "Returns: a list of stderr line by line, ignore trailing blanks"
        return self.stderr().rstrip().split("\n")


# ──────────────────────────────────────────────────────────────────────────
# Logging filters — drop or rewrite third-party noise without changing rank loggers
# ──────────────────────────────────────────────────────────────────────────


class ExcludeLoggingFilter(logging.Filter):
    """
    Usage: logging.getLogger('name').addFilter(
        ExcludeLoggingFilter(['info mess*age', 'Warning: *'])
    )
    Supports wildcard.
    https://relaxdiego.com/2014/07/logging-in-python.html
    """

    def __init__(self, patterns):
        """Drop records whose message matches any wildcard in ``patterns``."""
        super().__init__()
        self._patterns = patterns

    def filter(self, record):
        """Return False when ``record.msg`` matches an exclude pattern."""
        if match_patterns(record.msg, include=self._patterns):
            return False
        else:
            return True


class ReplaceStringLoggingFilter(logging.Filter):
    """Rewrite log records whose message matches any of ``patterns``."""

    def __init__(self, patterns, replacer: Callable):
        """Rewrite matching messages with ``replacer``; all records still pass."""
        super().__init__()
        self._patterns = patterns
        assert callable(replacer)
        self._replacer = replacer

    def filter(self, record):
        """Apply ``replacer`` to matching messages and always keep the record."""
        if match_patterns(record.msg, include=self._patterns):
            record.msg = self._replacer(record.msg)
        return True


def logging_exclude_pattern(
    logger_name,
    patterns: Union[str, list[str], Callable, list[Callable], None],
):
    """
    Args:
        patterns: see worldfoundry.core.utils.match_patterns
    """
    logging.getLogger(logger_name).addFilter(ExcludeLoggingFilter(patterns))


def logging_replace_string(
    logger_name,
    patterns: Union[str, list[str], Callable, list[Callable], None],
    replacer: Callable,
):
    """
    Args:
        patterns: see worldfoundry.core.utils.match_patterns
    """
    logging.getLogger(logger_name).addFilter(ReplaceStringLoggingFilter(patterns, replacer))

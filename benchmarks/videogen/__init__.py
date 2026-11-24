# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible single-GPU and multi-GPU benchmark suite for foundation video generation."""

from __future__ import annotations

__all__ = ["format_markdown_leaderboard"]


def __getattr__(name: str):
    if name == "format_markdown_leaderboard":
        from .report import format_markdown_leaderboard

        return format_markdown_leaderboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

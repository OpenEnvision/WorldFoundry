"""Workspace page asset, independent of API and model execution code."""

from pathlib import Path

WORKSPACE_HTML = Path(__file__).with_name("workspace.html").read_text(encoding="utf-8")

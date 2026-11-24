"""Public DB-CogACT API with lazy inference dependency loading."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "DBCogACTSynthesis":
        from .db_cogact_synthesis import DBCogACTSynthesis

        return DBCogACTSynthesis
    raise AttributeError(name)


__all__ = ["DBCogACTSynthesis"]

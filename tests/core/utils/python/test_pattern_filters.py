# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Callable and glob filtering follow the same include/exclude contracts."""

from typing import Literal

import pytest

from worldfoundry.core.utils.python.misc_utils import filter_patterns, match_patterns

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("matches", [False, True])
def test_callable_match_and_exclusion(matches: bool) -> None:
    def predicate(item: str) -> bool:
        assert item == "sample"
        return matches

    assert match_patterns("sample", include=predicate) is matches
    assert match_patterns("sample", exclude=predicate) is not matches


@pytest.mark.parametrize("ordering", ["original", "include"])
def test_filter_callable_preserves_input_order(ordering: Literal["original", "include"]) -> None:
    def predicate(item: str) -> bool:
        return item.startswith("keep")

    assert filter_patterns(["keep-b", "drop", "keep-a"], include=predicate, ordering=ordering) == [
        "keep-b",
        "keep-a",
    ]


def test_callable_list_continues_after_nonmatching_predicate() -> None:
    def starts_with_a(item: str) -> bool:
        return item.startswith("a")

    def starts_with_b(item: str) -> bool:
        return item.startswith("b")

    assert filter_patterns(["b1", "a1", "b2", "c1"], include=[starts_with_a, starts_with_b], ordering="include") == [
        "a1",
        "b1",
        "b2",
    ]


def test_callable_exclude_keeps_glob_precedence() -> None:
    def excluded(item: str) -> bool:
        return item.endswith("-skip")

    assert not match_patterns("keep-skip", include="keep*", exclude=excluded)
    assert match_patterns("keep-skip", include="keep*", exclude=excluded, precedence="include")
    assert match_patterns("keep-ok", include="keep*", exclude=excluded)
    assert filter_patterns(["b1", "a1", "b1"], include=["a*", "b*"], ordering="include") == ["a1", "b1", "b1"]

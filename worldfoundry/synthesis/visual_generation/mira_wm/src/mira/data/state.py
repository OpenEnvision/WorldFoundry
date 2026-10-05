"""Typed structure of the per-frame physics / game state attached to a clip.

A clip's ``physics`` is ``list[list[FrameState]]`` — one list per selected perspective, each a
per-frame state frame-aligned 1:1 with the clip's frames / actions along the T axis. The runtime
objects are plain JSON dicts decoded from a chunk's physics track; these :class:`typing.TypedDict`
definitions document and type that structure without changing it, so there is no per-frame parsing
cost and dict access is unchanged.

Coordinate system (standard Rocket League arena, Unreal units): x in ~[-4096, 4096] (side walls),
y in ~[-5120, 5120] (the goal-to-goal long axis), z in ~[0, 2044] (floor to ceiling).
"""

from __future__ import annotations

from typing import TypedDict


class Vec3(TypedDict):
    """A 3D vector (location, velocity, or angular velocity) in Unreal units."""

    x: float
    y: float
    z: float


class Quat(TypedDict):
    """An orientation quaternion."""

    x: float
    y: float
    z: float
    w: float


class GameInfo(TypedDict):
    """Match-wide state for one frame.

    ``time_remaining`` (regulation countdown, seconds) is populated **only on the local
    perspective** — the recording player's own POV — and is ``0.0`` on the other three. Read it
    from the ``is_local`` perspective (see ``perspective_has_clock``). It counts down from ~300 to
    0 and holds steady during replays/kickoffs. ``score_*`` and ``is_overtime`` are correct on
    every perspective.
    """

    time_remaining: float  # local-perspective-only; 0.0 on the others (see class docstring)
    score_blue: int
    score_orange: int
    is_overtime: bool


class BallState(TypedDict):
    location: Vec3
    velocity: Vec3
    rotation: Quat
    angular_velocity: Vec3


class _CarStateRequired(TypedDict):
    player_id: int
    team: int
    location: Vec3
    velocity: Vec3
    attacker_player_id: int  # the demolisher's player_id, -1 when none


class CarState(_CarStateRequired, total=False):
    """One car in one frame.

    ``player_id``, ``team``, ``location``, ``velocity``, and ``attacker_player_id`` are always
    present; the remaining fields are optional. Demolitions are not flagged by a boolean (the
    source export never sets one) — a car being demolished is indicated by ``attacker_player_id``
    (the demolisher's player_id; ``-1`` otherwise).
    """

    is_local: bool
    rotation: Quat
    angular_velocity: Vec3
    boost_amount: float
    is_on_ground: bool
    is_supersonic: bool


class FrameState(TypedDict):
    """One frame of game state: match clock/score, the ball, and all cars.

    All perspectives observe the same world state; it is kept per-perspective so it stays
    frame-aligned with that perspective's frames and actions.
    """

    game: GameInfo
    ball: BallState
    cars: list[CarState]

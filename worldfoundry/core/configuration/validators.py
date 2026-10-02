"""Descriptor base for reusable validated configuration fields.

Responsibility
    Type / range / choice checks on config objects without pulling pydantic
    into every runtime. Fail at assignment, not at first forward.

Boundaries
    Not a schema library and not Hydra. Descriptors store the validated value
    on ``_<name>``. ``hidden`` / ``tooltip`` exist for UI enumeration only;
    they do not affect validate. ``get_range_iterator`` is a sweep helper,
    not a proof that every yielded value is valid after later edits.

Public surface
    :class:`Validator`, :class:`Bool`, :class:`Int`, :class:`Float`,
    :class:`OneOf`, :class:`String`, :class:`JsonDict`, ``_UNSET``.
"""

import itertools
import json
from abc import ABC, abstractmethod

_UNSET = object()


# ──────────────────────────────────────────────────────────────────────────
# Descriptor protocol — validate on set; missing required fields fail on get
# ──────────────────────────────────────────────────────────────────────────


class Validator(ABC):
    """Data descriptor that validates on write and stores under ``_<name>``."""

    def __init__(self, default=_UNSET, hidden: bool = False):
        """Keep ``default`` until first set; ``hidden`` is UI-only."""
        self.default = default
        self.hidden = hidden

    def __set_name__(self, owner, name):
        """Bind the private storage name when the owning class is created."""
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        """Return the stored value, the default, or raise if still required.

        Class-level access returns the descriptor itself so UIs can inspect
        ``min`` / ``options``. Failure: :exc:`ValueError` when the field is
        required (``default`` is ``_UNSET``) and never assigned.
        """
        if obj is None:
            return self
        value = getattr(obj, self.private_name, self.default)
        if value is _UNSET:
            raise ValueError(f"required parameter {self.private_name[1:]!r} has not been set")
        return value

    def __set__(self, obj, value):
        """Validate then write; invalid values never reach the instance dict."""
        setattr(obj, self.private_name, self.validate(value))

    @abstractmethod
    def validate(self, value):
        """Return the coerced value or raise :exc:`TypeError` / :exc:`ValueError`."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────
# Concrete fields — coerce strings from env / CLI; reject the rest
# ──────────────────────────────────────────────────────────────────────────


class Bool(Validator):
    """Accept bool, ``0``/``1``, or common truthy / falsy strings."""

    def __init__(self, default=_UNSET, hidden: bool = False, tooltip=None):
        """Optional ``tooltip`` is UI copy; it is not used in validation."""
        super().__init__(default, hidden)
        self.tooltip = tooltip

    def validate(self, value):
        """Coerce ``true``/``false``/``1``/``0`` (any case) and nonzero ints.

        Failure: :exc:`ValueError` for other strings, :exc:`TypeError` for
        non-bool / non-int objects.
        """
        if isinstance(value, str):
            normalized = value.lower()
            if normalized not in {"true", "false", "1", "0"}:
                raise ValueError(f"Expected a boolean string, got {value!r}")
            return normalized in {"true", "1"}
        if isinstance(value, int):
            return value != 0
        if not isinstance(value, bool):
            raise TypeError(f"Expected bool, got {type(value).__name__}")
        return value

    def get_range_iterator(self):
        """Yield both boolean values for UI / sweep enumeration."""
        return [True, False]


class Int(Validator):
    """Bounded integer; strings are parsed with :func:`int`."""

    def __init__(self, default=_UNSET, min=None, max=None, step=1, hidden: bool = False, tooltip=None):
        """Inclusive ``min`` / ``max``; ``step`` is only for range iteration."""
        super().__init__(default, hidden)
        self.min = min
        self.max = max
        self.step = step
        self.tooltip = tooltip

    def validate(self, value):
        """Parse strings, then enforce optional inclusive bounds.

        Failure: :exc:`TypeError` if the value is not an int after parse,
        :exc:`ValueError` if it is outside ``[min, max]``.
        """
        if isinstance(value, str):
            value = int(value)
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        """Count from ``min`` (or default) by ``step`` up to ``max`` (or default+100)."""
        default = 0 if self.default is _UNSET else int(self.default)
        lower = default if self.min is None else self.min
        upper = default + 100 if self.max is None else self.max
        return itertools.takewhile(lambda item: item <= upper, itertools.count(lower, self.step))


class Float(Validator):
    """Validate a bounded floating-point configuration value."""

    def __init__(self, default=0.0, min=None, max=None, step=0.5, hidden: bool = False, tooltip=None):
        """Inclusive bounds; ``step`` is only for range iteration."""
        super().__init__(default, hidden)
        self.min = min
        self.max = max
        self.step = step
        self.tooltip = tooltip

    def validate(self, value):
        """Accept ``str`` / ``int`` / ``float``; reject other types after coerce."""
        if isinstance(value, (str, int)):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(f"Expected float, got {type(value).__name__}")
        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        """Count from ``min`` or ``default`` by ``step`` through ``max`` or ``default``."""
        lower = self.default if self.min is None else self.min
        upper = self.default if self.max is None else self.max
        return itertools.takewhile(lambda item: item <= upper, itertools.count(lower, self.step))


class OneOf(Validator):
    """Validate a value against an explicit set of choices."""

    def __init__(self, default, options, type_cast=None, hidden: bool = False, tooltip=None):
        """``type_cast`` runs before membership; failed casts become :exc:`ValueError`."""
        super().__init__(default, hidden)
        self.options = set(options)
        self.type_cast = type_cast
        self.tooltip = tooltip

    def validate(self, value):
        """Optionally cast, then require membership in ``options``."""
        if self.type_cast is not None:
            try:
                value = self.type_cast(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Expected {value!r} to be castable to {self.type_cast!r}") from exc
        if value not in self.options:
            raise ValueError(f"Expected {value!r} to be one of {self.options!r}")
        return value

    def get_range_iterator(self):
        """Iterate the allowed set (order is not stable)."""
        return iter(self.options)


class String(Validator):
    """Optional string with length bounds and an optional predicate."""

    def __init__(self, default=_UNSET, min=None, max=None, predicate=None, hidden: bool = False, tooltip=None):
        """``None`` is a valid stored value; length checks skip it."""
        super().__init__(default, hidden)
        self.min = min
        self.max = max
        self.predicate = predicate
        self.tooltip = tooltip

    def validate(self, value):
        """Accept ``None`` or ``str``; enforce length then ``predicate``.

        Failure: :exc:`TypeError` for non-strings, :exc:`ValueError` for
        length or predicate rejection.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"Expected str or None, got {type(value).__name__}")
        if self.min is not None and len(value) < self.min:
            raise ValueError(f"Expected {value!r} to have length at least {self.min}")
        if self.max is not None and len(value) > self.max:
            raise ValueError(f"Expected {value!r} to have length at most {self.max}")
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(f"Predicate {self.predicate!r} rejected {value!r}")
        return value

    def get_range_iterator(self):
        """Yield only the configured default (strings are not enumerable)."""
        return iter([self.default])


class JsonDict(Validator):
    """Accept a ``dict`` or a JSON object string; empty input becomes ``{}``."""

    def validate(self, value):
        """Parse JSON when needed; reject non-object JSON and decode errors."""
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        try:
            result = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Expected a JSON-encoded dictionary, got {value!r}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON dictionary, got {type(result).__name__}")
        return result


__all__ = ["Bool", "Float", "Int", "JsonDict", "OneOf", "String", "Validator", "_UNSET"]

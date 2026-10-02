"""Functional meta-programming and object-inspection utilities.

Responsibility
    Reflection (method / signature / keys), a lightweight ``state_dict``
    class decorator, and decorators that expand one-leaf functions over
    nested structures or alternate calling conventions.

Boundaries
    Not a registry for models — prefer
    ``worldfoundry.core.registry.TypedRegistry``. :class:`ClassRegistry` is
    legacy and silently overwrites duplicate names. Recursive wrappers need
    ``dm_tree`` when ``with_path=True``.

Public surface
    :func:`make_recursive_func`, :func:`meta_decorator`, :func:`state_dict_class`,
    :class:`ClassRegistry`, :class:`NoopObject`, :class:`NoopContext`,
    pack/enable varargs and kwargs helpers, signature inspectors.
"""

from __future__ import annotations

import collections.abc
import functools
import inspect
import pprint
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Literal


# ──────────────────────────────────────────────────────────────────────────
# Type tests — strings are scalars here, never walked as sequences
# ──────────────────────────────────────────────────────────────────────────


def is_sequence(value: Any) -> bool:
    """Return whether *value* is a non-string sequence."""

    return isinstance(value, Sequence) and not isinstance(value, str)


def is_mapping(value: Any) -> bool:
    """Return whether *value* implements the mapping protocol."""

    return isinstance(value, Mapping)


# ──────────────────────────────────────────────────────────────────────────
# State-dict mixin — named attrs only; missing keys fail at load
# ──────────────────────────────────────────────────────────────────────────


def state_dict_class(keys: list[str]):
    """Class decorator to inject clean `state_dict` and `load_state_dict` serialization logic.

    Adds the following attributes to the decorated class:
        - `state_dict()` -> Returns a dictionary mapping specified key names to active attribute values.
        - `load_state_dict(sdict)` -> Restores attributes from the provided state dictionary,
          throwing a ValueError if any key is missing.
        - `state_keys` -> Read-only list of configured state key names.
    """

    def _wrap_class(cls):
        """Attach ``state_dict`` / ``load_state_dict`` / ``state_keys`` to ``cls``."""
        assert inspect.isclass(cls)

        def state_dict(self):
            """Snapshot the configured attribute names; missing attrs raise :exc:`AttributeError`."""
            return {k: getattr(self, k) for k in keys}

        def load_state_dict(self, states: Dict[str, Any]):
            """Restore every configured key; extra keys in ``states`` are ignored.

            Failure: :exc:`ValueError` when any configured key is absent.
            """
            if not set(keys).issubset(set(states.keys())):
                raise ValueError(f"states does not have all the required keys: {keys}")
            for k in keys:
                setattr(self, k, states[k])

        @property
        def state_keys(self):
            """Read-only list of names this mixin will serialize."""
            return keys

        cls.state_dict = state_dict
        cls.load_state_dict = load_state_dict
        cls.state_keys = state_keys
        return cls

    return _wrap_class


def implements_method(object, method: str):
    """Returns True if the object exposes a callable method with the specified name."""
    return hasattr(object, method) and callable(getattr(object, method))


def assert_implements_method(object, method: str | list[str]):
    """Asserts that the object implements all specified callable method names."""
    if isinstance(method, str):
        method = [method]
    for m in method:
        assert implements_method(object, m), f"object {object.__class__} does not implement method {m}()"


# ──────────────────────────────────────────────────────────────────────────
# Decorator factories — @deco and @deco(...) must both be legal
# ──────────────────────────────────────────────────────────────────────────


def meta_decorator(decor):
    """Meta-decorator enabling a custom decorator to support both parenthesized and clean signatures.

    Supports:
        @decorator(*args, **kwargs)
        def my_func()

        -- OR --

        @decorator
        def my_func()
    """
    import functools

    def single_callable(args, kwargs):
        """True only for ``@decorator`` (one positional callable, no kwargs)."""
        return len(args) == 1 and len(kwargs) == 0 and callable(args[0])

    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        """Apply immediately, or return a closer that binds decorator args."""
        if single_callable(args, kwargs):
            return decor(args[0])
        else:
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_decorator
def make_recursive_func(fn, *, with_path=False):
    """
    Decorator that turns a function that works on a single array/tensor to working on
    arbitrary nested structures.
    """
    import functools

    try:
        import tree
    except ImportError:
        tree = None

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        """Map ``fn`` over a tree; fallback walk if ``dm_tree`` is missing."""
        if tree is not None and with_path:
            return tree.map_structure_with_path(lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct)
        if tree is not None:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)
        if with_path:
            raise ImportError("dm_tree is required when make_recursive_func(with_path=True)")
        if is_mapping(tensor_struct):
            return type(tensor_struct)((key, _wrapper(value, *args, **kwargs)) for key, value in tensor_struct.items())
        if is_sequence(tensor_struct):
            return type(tensor_struct)(_wrapper(value, *args, **kwargs) for value in tensor_struct)
        return fn(tensor_struct, *args, **kwargs)

    return _wrapper


@meta_decorator
def deprecated(func, msg="", action="warning", type=""):
    """
    Function/class decorator: designate deprecation.

    Args:
      msg: string message.
      action: string mode
      - 'warning': (default) prints `msg` to stderr
      - 'noop': do nothing, just for source code annotation purposes
      - 'raise': raise DeprecatedError(`msg`)
    """
    action = action.lower()
    type = type.lower()
    ALL_ACTIONS = ["warn", "warning", "noop", "raise"]
    if action not in ALL_ACTIONS:
        raise ValueError(f"Unknown action {action}. Choose from {ALL_ACTIONS}.")
    ALL_TYPES = {
        "": DeprecationWarning,
        "pending": PendingDeprecationWarning,
        "future": FutureWarning,
    }
    if type not in ALL_TYPES:
        raise ValueError(f"Unknown type {type}. Choose from {ALL_TYPES.keys()}.")
    if not msg:
        msg = "This is a deprecated feature."

    WarningExceptionCls = ALL_TYPES[type]

    # only does the deprecation when being called
    @functools.wraps(func)
    def _deprecated(*args, **kwargs):
        """Warn / raise / no-op on each call, then forward to ``func``."""
        if action in ["warning", "warn"]:
            warnings.warn(msg, WarningExceptionCls)
        elif action == "raise":
            raise WarningExceptionCls(msg)
        return func(*args, **kwargs)

    return _deprecated


@meta_decorator
def call_once(func, on_second_call: Literal["noop", "raise", "warn"] = "noop"):
    """
    Decorator to ensure that a function is only called once.

    Args:
      on_second_call (str): what happens when the function is called a second time.
    """
    assert on_second_call in [
        "noop",
        "raise",
        "warn",
    ], "mode must be one of 'noop', 'raise', 'warn'"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Run ``func`` once; later calls follow ``on_second_call``."""
        if wrapper._called:
            if on_second_call == "raise":
                raise RuntimeError(f"{func.__name__} has already been called. Can only call once.")
            elif on_second_call == "warn":
                warnings.warn(f"{func.__name__} has already been called. Should only call once.")
        else:
            wrapper._called = True
            return func(*args, **kwargs)

    wrapper._called = False
    return wrapper


class NoopObject:
    """
    Object that does nothing when called any method
    """

    def __init__(self, *args, **kwargs):
        """Keep construction args for debugging; they are never used."""
        self.init_args = args
        self.init_kwargs = kwargs

    def __getattr__(self, name):
        """Return a no-op callable so any method name is legal."""

        def _func(*args, **kwargs):
            """Swallow the call; used as a stand-in for optional loggers / hooks."""
            pass

        return _func


class NoopContext:
    """
    Placeholder context manager that does nothing.
    We could have written simply as:

    @contextmanager
    def noop_context(*args, **kwargs):
        yield

    but the returned context manager cannot be called twice, i.e.
    my_noop = NoopContext()
    with my_noop:
        do1()
    with my_noop: # trigger generator error
        do2()
    """

    def __init__(self, *args, **kwargs):
        """Accept any args so it can replace a real context manager at a call site."""
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        """Re-enterable; unlike a generator context this object can be reused."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Do not suppress exceptions; just leave the placeholder."""
        pass


class ClassRegistry:
    """
    LEGACY (vendored): duplicate names are silently overwritten; prefer
    ``worldfoundry.core.registry.TypedRegistry`` for new code.

    The former ``make_registry_metaclass`` factory had zero callers and was
    removed (XC-11). Use ``TypedRegistry`` or this ``__init_subclass__`` hook.

    Use in conjunction with `__init_subclass__` hook in your base class

    class BaseClass:
        registry = ClassRegistry()

        def __init_subclass__(cls, **kwargs):
            cls.registry.add(cls)
            super().__init_subclass__(**kwargs)

    print(registry)
    """

    def __init__(self, base_class_name: str = None):
        """Optional ``base_class_name`` is only used in missing-key error text."""
        self.registry = {}
        self._base_class_name = base_class_name

    def add(self, cls):
        """Register ``cls`` by ``__name__``; a later add silently overwrites."""
        self.registry[cls.__name__] = cls

    def get(self, name):
        """Look up a class; :exc:`KeyError` lists currently registered names."""
        if name not in self.registry:
            existing_cls = list(self.registry.keys())
            base_name = self._base_class_name + " " if self._base_class_name else ""
            raise KeyError(f"{base_name} subclass '{name}' not found in registry: {existing_cls}")
        return self.registry[name]

    def __str__(self):
        """Pretty-print the name → class map for debugging."""
        return pprint.pformat(self.registry)

    def __getitem__(self, name):
        """Same as :meth:`get` so the registry can be indexed like a dict."""
        return self.get(name)

    def instantiate(self, cls, **kwargs):
        """Construct the registered class named ``cls`` with ``kwargs``."""
        return self.get(cls)(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Inspect utils — signature / keys; bind failures stay boolean
# ──────────────────────────────────────────────────────────────────────────


def func_parameters(func):
    """Return ``inspect.signature(func).parameters`` (ordered mapping)."""
    return inspect.signature(func).parameters


def func_has_arg(func, arg_name):
    """True if ``arg_name`` is an explicit parameter (not covered by ``**kwargs``)."""
    return arg_name in func_parameters(func)


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args


def enable_list_arg(func):
    """
    Function decorator.
    If a function only accepts varargs (*args),
    make it support a single list arg as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Unpack a single list arg so ``f([a, b])`` matches ``f(a, b)``."""
        args = pack_varargs(args)
        return func(*args, **kwargs)

    return wrapper


def enable_varargs(func):
    """
    Function decorator.
    If a function only accepts a list arg, make it support varargs as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Pack ``*args`` into one list so a list-only function accepts varargs."""
        args = pack_varargs(args)
        return func(args, **kwargs)

    return wrapper


def pack_kwargs(args, kwargs):
    """
    Pack **kwargs or a single dict arg as dict

    def f(*args, **kwargs):
        kwdict = pack_kwargs(args, kwargs)
        # kwdict is now packed as a dict
    """
    if len(args) == 1 and is_mapping(args[0]):
        assert not kwargs, "cannot have both **kwargs and a dict arg"
        return args[0]  # single-dict
    else:
        assert not args, "cannot have positional args if **kwargs exist"
        return kwargs


def merge_kwargs(args, kwargs) -> Dict:
    """
    Merge all dicts in `args` and keywords in kwargs.

    E.g. merge_kwargs({"a.b": 1, "a.c": 2}, foo=6, bar=8)
       -> {"a.b": 1, "a.c": 2, "foo": 6, "bar": 8}
    """
    kw_all = {}
    for arg in args:
        assert is_mapping(arg), f"{arg} is not a dict."
        kw_all.update(arg)
    kw_all.update(kwargs)
    return kw_all


def enable_dict_arg(func):
    """
    Function decorator.
    If a function only accepts varargs (*args),
    make it support a single list arg as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Accept either ``f(dict)`` or ``f(**kwargs)`` and always call with kwargs."""
        kwargs = pack_kwargs(args, kwargs)
        return func(**kwargs)

    return wrapper


def enable_kwargs(func):
    """
    Function decorator.
    If a function only accepts a dict arg, make it support kwargs as well
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Pack kwargs into one dict so a dict-only function accepts ``**kwargs``."""
        kwargs = pack_kwargs(args, kwargs)
        return func(kwargs)

    return wrapper


def has_keys(D, keys: list):
    """True if every name in ``keys`` is present; ``D`` must be a mapping."""
    assert is_mapping(D)
    return all(key in D for key in keys)


def assert_has_keys(D, keys: list):
    """Like :func:`has_keys` but raise :exc:`KeyError` naming the first missing key."""
    assert is_mapping(D), "Input is not a dict"
    for key in keys:
        if key not in D:
            raise KeyError(f'Required key "{key}" is missing in dict {D}')
    return True


def method_decorator(decorator):
    """
    Decorator of decorator: transform a decorator that only works on normal
    functions to a decorator that works on class methods
    From Django form: https://goo.gl/XLjxKK
    """

    @functools.wraps(decorator)
    def wrapped_decorator(method):
        """Bind ``self`` before applying the original function decorator."""

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            """Re-bind ``method`` so the inner decorator sees a plain function."""

            def bound_func(*args2, **kwargs2):
                """Forward to the original method with the captured ``self``."""
                return method(self, *args2, **kwargs2)

            return decorator(bound_func)(*args, **kwargs)

        return wrapper

    return wrapped_decorator


def accepts_varargs(func):
    """
    If a function accepts *args
    """
    params = inspect.signature(func).parameters
    return any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())


def accepts_kwargs(func):
    """
    If a function accepts **kwargs
    """
    params = inspect.signature(func).parameters
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def is_signature_compatible(func, *args, **kwargs):
    """True if ``func(*args, **kwargs)`` would bind; :exc:`TypeError` stays False."""
    try:
        sig.bind(*args, **kwargs)
        return True
    except TypeError:
        return False


def make_list(x):
    """
    Turns a singleton object to a list. If already a list, no change.
    """
    if is_sequence(x):
        return x
    else:
        return [x]


def as_list(value, *, none_as_empty: bool = True) -> list:
    """Normalize optional scalar or sequence values into a plain list."""

    if value is None and none_as_empty:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, collections.abc.Sequence):
        return list(value)
    return [value]


def make_tuple(elem, repeats):
    """
    E.g. expand a singleton x into (x, x, x)
        useful for things like image_size or kernal, which can be a single int/float
        or a tuple of fixed size
    """
    if is_sequence(elem):
        assert len(elem) == repeats, f"length of input must be {repeats}: {elem}"
        return elem
    else:
        return (elem,) * repeats

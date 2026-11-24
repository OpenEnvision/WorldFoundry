from collections.abc import Mapping, Sequence
from typing import Any, Union

import jax

PRNGKey = jax.random.KeyArray
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]  # noqa: UP007 - Runtime type alias kept as typing.Union.
Config = Union[Any, Mapping[str, "Config"]]  # noqa: UP007 - Runtime type alias kept as typing.Union.
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
